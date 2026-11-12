"""Bounded installed-sorter comparison for Kimi E896/topk16/BM16.

No model load, source patch or serving change. Includes route metadata, both
auxiliary maps, and zeroing the full M x 3584 BF16 atomic output. The physical
512-row byte-mesh experiment includes copying the active IDs and weights.
"""

import argparse
import faulthandler
import hashlib
import importlib.util
import json
import os
import random
import signal
import statistics
import subprocess
import sys
import time
import traceback
from collections import Counter
from pathlib import Path

EXPERTS, TOPK, BLOCK, WIDTH = 896, 16, 16, 3584
HASHES = {
    "aiter/fused_moe.py": "dfe1f65b957656a58581651e3da9c4d0b94ca7ca65d7764551accc9bf6274541",
    "aiter/ops/moe_sorting_opus.py": "7086e43e975412b6ddbc4b262538532ec100752f13149cb283aecf20f20d6e01",
    "aiter/ops/moe_mxfp4_aux.py": "8f99bde381c7f15355490be39bcfcac228a69d8dfaa33b48990fdca029f12e84",
    "aiter_meta/csrc/include/moe_sorting_opus.h": "a2ef2d60565e3ae1a4c5302a6a222b3e96bc114e28b3336f1b13d4aa08333076",
    "aiter_meta/csrc/py_itfs_cu/moe_sorting_opus_kernels.cu": "983e6506483881b951742a8b93796d98598b225c2f5f3e66896ef483b57d71ee",
    "aiter_meta/csrc/kernels/mxfp4_moe/moe_aux/moe_sort_quant.cuh": "ba3653f85d7ddbe07957d54f7ff9fece0e2cf802a039661a44a52678bd6dffe1",
}
NAMES = ("native", "native_local", "single", "byte512_with_copy", "adaptive")


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--rows", default="259")
    p.add_argument(
        "--arms", default="native", help="Comma-separated arms; start with native only"
    )
    p.add_argument("--histograms", default="balanced")
    p.add_argument("--rounds", type=int, default=6)
    p.add_argument("--max-seconds", type=int, default=60)
    p.add_argument("--source-only", action="store_true")
    p.add_argument("--source-root", type=Path)
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    a = p.parse_args()
    try:
        a.rows = list(map(int, a.rows.split(",")))
    except ValueError:
        p.error("Rows must be comma-separated integers")
    if (
        not 1 <= len(a.rows) <= 12
        or len(set(a.rows)) != len(a.rows)
        or any(x not in range(259, 337, 7) for x in a.rows)
    ):
        p.error("Select up to twelve unique rows in 259,266,...,336")
    if not 6 <= a.rounds <= 60 or not 30 <= a.max_seconds <= 540:
        p.error("Use 6..60 rounds and 30..540 seconds")
    if a.source_root and not a.source_only:
        p.error("--source-root requires --source-only")
    a.arms = a.arms.split(",")
    a.histograms = a.histograms.split(",")
    if (
        not a.arms
        or a.arms[0] != "native"
        or len(set(a.arms)) != len(a.arms)
        or not set(a.arms) <= set(NAMES)
    ):
        p.error("Use unique known arms, starting with native")
    if (
        not a.histograms
        or len(set(a.histograms)) != len(a.histograms)
        or not set(a.histograms) <= {"balanced", "clustered64", "hot16"}
    ):
        p.error("Use unique histograms: balanced,clustered64,hot16")
    return a


def save(folder, result):
    path = folder / "result.json.tmp"
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    path.replace(folder / "result.json")


def inputs(rows, distribution, seed):
    rng = random.Random(seed)
    ids, weights = [], []
    pool = {"balanced": EXPERTS, "clustered64": 64, "hot16": 16}[distribution]
    for token in range(rows):
        selected = (
            [(token * TOPK + slot + seed) % EXPERTS for slot in range(TOPK)]
            if distribution == "balanced"
            else rng.sample(range(pool), TOPK)
        )
        ids.append(selected)
        # Exactly represented FP32 values isolate route/weight mapping correctness.
        weights.append([rng.randrange(1, 33) / 32 for _ in range(TOPK)])
    return ids, weights


def reference(ids, weights):
    routes = [[] for _ in range(EXPERTS)]
    for token, (experts, values) in enumerate(zip(ids, weights)):
        assert len(set(experts)) == TOPK
        for slot, (expert, weight) in enumerate(zip(experts, values)):
            routes[expert].append((token | (slot << 24), weight))
    return routes


def validate(snapshot, ids, weights):
    rows = len(ids)
    routes = reference(ids, weights)
    count = sum(((len(r) + BLOCK - 1) // BLOCK) * BLOCK for r in routes)
    assert snapshot["counts"] == [count, rows], snapshot["counts"]
    assert snapshot["zero_output"] and snapshot["guards_intact"]
    cursor, ordered = 0, True
    expected_reverse = [-1] * (rows * TOPK)
    signature = []
    for expert, wanted in enumerate(routes):
        if not wanted:
            continue
        end = cursor + ((len(wanted) + BLOCK - 1) // BLOCK) * BLOCK
        assert snapshot["experts"][cursor // BLOCK : end // BLOCK] == [expert] * (
            (end - cursor) // BLOCK
        )
        actual = []
        for index in range(cursor, end):
            packed = snapshot["ids"][index] & 0xFFFFFFFF
            token, slot = packed & 0xFFFFFF, packed >> 24
            weight = snapshot["weights"][index]
            assert snapshot["m_indices"][index] == token
            if token < rows:
                assert slot < TOPK and ids[token][slot] == expert
                assert weight == weights[token][slot]
                actual.append((packed, weight))
                route = token * TOPK + slot
                assert expected_reverse[route] == -1
                expected_reverse[route] = index
            else:
                # Opus packs (M, topk); adaptive uses plain M. Both are OOB
                # sentinels to consumers. This is not a valid-route reorder.
                assert token == rows and slot in (0, TOPK) and weight == 0
        assert Counter(actual) == Counter(wanted)
        ordered &= actual == wanted
        signature.extend(actual)
        cursor = end
    assert snapshot["reverse"][: rows * TOPK] == expected_reverse
    return {
        "status": "passed",
        "ordered_like_reference": ordered,
        "padded_rows": count,
        "active_experts": sum(bool(r) for r in routes),
        "max_routes_per_expert": max(map(len, routes)),
        "ordered_route_sha256": hashlib.sha256(
            json.dumps(signature).encode()
        ).hexdigest(),
    }


def source_checks(args):
    root = args.source_root
    if root is None:
        spec = importlib.util.find_spec("aiter")
        if spec is None or spec.origin is None:
            raise RuntimeError("Pinned AITER is not installed")
        root = Path(spec.origin).parent.parent
    for path, digest in HASHES.items():
        raw = (root / path).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == digest, path
        target = args.output / "sources" / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
    return {"status": "passed", "root": str(root), "sha256": HASHES}


def summary(values):
    return {
        "samples_us": values,
        "mean_us": statistics.mean(values),
        "median_us": statistics.median(values),
        "min_us": min(values),
        "max_us": max(values),
        "std_us": statistics.pstdev(values),
    }


def timing(torch, operations, rounds):
    samples = {name: [] for name in operations}
    names = list(operations)
    for round_id in range(rounds):
        # Rotate and reverse so five arms do not always occupy the same position.
        order = names[round_id % len(names) :] + names[: round_id % len(names)]
        if round_id % 2:
            order.reverse()
        for name in order:
            begin, end = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            begin.record()
            operations[name]()
            end.record()
            end.synchronize()
            samples[name].append(begin.elapsed_time(end) * 1000)
    out = {name: summary(v) for name, v in samples.items()}
    return {
        "arms": out,
        "mean_reduction_pct_vs_native": {
            name: 100 * (1 - v["mean_us"] / out["native"]["mean_us"])
            for name, v in out.items()
        },
    }


def run_gpu(args, result):
    started = time.monotonic()

    def mark(stage, **context):
        event = {
            "stage": stage,
            "elapsed_seconds": time.monotonic() - started,
            **context,
        }
        result.setdefault("progress", []).append(event)
        save(args.output, result)
        print(json.dumps(event), flush=True)

    mark("import_runtime")
    import aiter
    import torch

    mark("runtime_imported")
    if (
        not torch.cuda.is_available()
        or not torch.version.hip
        or "MI355X" not in torch.cuda.get_device_name(0).upper()
    ):
        raise RuntimeError("This experiment requires MI355X/ROCm")
    torch.cuda.set_device(0)
    torch.set_num_threads(1)
    result["runtime"] = {
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "device": torch.cuda.get_device_name(0),
    }
    mark("device_ready")

    class Arm:
        def __init__(self, name, ids, weights):
            self.name, self.ids, self.weights = name, ids, weights
            self.rows = ids.shape[0]
            physical = 512 if name == "byte512_with_copy" else self.rows
            self.physical = physical
            self.local = (
                torch.tensor([self.rows], device="cuda", dtype=torch.int32)
                if name in ("native_local", "byte512_with_copy")
                else None
            )
            self.policy = 1 if name == "single" else 0
            capacity = (
                (physical * TOPK + EXPERTS * BLOCK - TOPK + BLOCK - 1) // BLOCK
            ) * BLOCK
            self.parents, self.values = {}, {}
            for key, shape, dtype in [
                ("ids", (capacity,), torch.int32),
                ("weights", (capacity,), torch.float32),
                ("experts", (capacity // BLOCK,), torch.int32),
                ("counts", (2,), torch.int32),
                ("m_indices", (capacity,), torch.int32),
                ("reverse", (physical * TOPK,), torch.int32),
                ("output", (self.rows, WIDTH), torch.bfloat16),
            ]:
                parent = torch.full(
                    (shape[0] + 8, *shape[1:]), 17, device="cuda", dtype=dtype
                )
                self.parents[key], self.values[key] = parent, parent[: shape[0]]
            self.workspace_bytes = int(
                aiter.moe_sorting_opus_get_workspace_size(
                    physical, EXPERTS, TOPK, self.policy, 0
                )
            )
            self.workspace = (
                torch.empty(self.workspace_bytes, device="cuda", dtype=torch.uint8)
                if self.workspace_bytes
                else None
            )
            self.empty_bf16 = torch.empty(0, device="cuda", dtype=torch.bfloat16)
            self.empty_i32 = torch.empty(0, device="cuda", dtype=torch.int32)
            if physical == 512:
                # Deliberately valid, nonzero suffix routes: correctness must
                # depend on num_local_tokens, not invalid-ID postfiltering.
                self.padded_ids = torch.arange(
                    TOPK, device="cuda", dtype=torch.int32
                ).repeat(512, 1)
                self.padded_weights = torch.full(
                    (512, TOPK), 13 / 32, device="cuda", dtype=torch.float32
                )

        def __call__(self):
            ids, weights, v = self.ids, self.weights, self.values
            if self.name == "byte512_with_copy":
                self.padded_ids[: self.rows].copy_(ids)
                self.padded_weights[: self.rows].copy_(weights)
                ids, weights = self.padded_ids, self.padded_weights
            if self.name == "adaptive":
                aiter.mxfp4_moe_sort(
                    topk_ids=ids,
                    topk_weight=weights,
                    sorted_token_ids=v["ids"],
                    sorted_expert_ids=v["experts"],
                    cumsum_tensor=v["counts"],
                    reverse_sorted=v["reverse"],
                    sorted_weights=v["weights"],
                    m_indices=v["m_indices"],
                    bf16_zero_out=v["output"],
                    bf16_zero_workspace=self.empty_bf16,
                    sort3stage_ws=self.empty_i32,
                    M_logical=self.rows,
                    NE=EXPERTS,
                    TOPK=TOPK,
                    D_HIDDEN=WIDTH,
                    D_INTER=1,
                    MB=BLOCK,
                    prologue=0,
                    num_local_tokens=None,
                )
            else:
                aiter.moe_sorting_opus_fwd(
                    ids,
                    weights,
                    v["ids"],
                    v["weights"],
                    v["experts"],
                    v["counts"],
                    v["output"],
                    EXPERTS,
                    BLOCK,
                    None,
                    self.local,
                    self.workspace,
                    self.policy,
                    None,
                    v["m_indices"],
                    v["reverse"],
                )

        def poison(self):
            for parent in self.parents.values():
                parent.fill_(17)

        def snapshot(self):
            torch.cuda.synchronize()
            v = self.values
            return {
                **{
                    key: value.cpu().tolist()
                    for key, value in v.items()
                    if key != "output"
                },
                "zero_output": bool(torch.count_nonzero(v["output"]).item() == 0),
                "guards_intact": all(
                    bool((p[self.values[k].shape[0] :] == 17).all())
                    for k, p in self.parents.items()
                ),
            }

    for rows in args.rows:
        for distribution in args.histograms:
            mark("case_inputs", rows=rows, distribution=distribution)
            cpu_ids, cpu_weights = inputs(rows, distribution, seed=7)
            ids = torch.tensor(cpu_ids, device="cuda", dtype=torch.int32)
            weights = torch.tensor(cpu_weights, device="cuda", dtype=torch.float32)
            case = {
                "rows": rows,
                "distribution": distribution,
                "status": "running",
                "checks": {},
            }
            result["cases"].append(case)
            save(args.output, result)
            arms = {}
            for name in args.arms:
                mark("arm_prepare", rows=rows, distribution=distribution, arm=name)
                arms[name] = Arm(name, ids, weights)
                mark("arm_prepared", arm=name)
            graphs = {}
            for name, arm in arms.items():
                mark("eager_submit", arm=name)
                arm.poison()
                arm()
                mark("eager_sync", arm=name)
                original = arm.snapshot()
                report = validate(original, cpu_ids, cpu_weights)
                case["checks"][name] = {
                    "original": report,
                    "workspace_bytes": arm.workspace_bytes,
                    "physical_rows": arm.physical,
                }
                mark("eager_validated", arm=name)
                # Warm library/compiler paths before capturing. Outputs stay owned
                # by this arm and all graph inputs retain stable addresses.
                for _ in range(3):
                    arm()
                mark("warmup_sync", arm=name)
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                mark("graph_capture", arm=name)
                with torch.cuda.graph(graph):
                    arm()
                mark("graph_captured", arm=name)
                arm.poison()
                graph.replay()
                mark("graph_replay_sync", arm=name)
                replay = arm.snapshot()
                validate(replay, cpu_ids, cpu_weights)
                count = report["padded_rows"]
                case["checks"][name]["own_graph_exact"] = all(
                    replay[key][:length] == original[key][:length]
                    for key, length in [
                        ("ids", count),
                        ("weights", count),
                        ("experts", count // BLOCK),
                        ("counts", 2),
                        ("m_indices", count),
                        ("reverse", rows * TOPK),
                    ]
                )
                case["checks"][name]["graph_semantics_passed"] = True
                graphs[name] = graph
                mark("graph_validated", arm=name)
            # Changed expert assignments and weights must reach every graph;
            # changing the distribution also changes the padded expert histogram.
            changed_ids, changed_weights = inputs(
                rows, "hot16" if distribution != "hot16" else "balanced", seed=19
            )
            ids.copy_(torch.tensor(changed_ids, device="cuda", dtype=torch.int32))
            weights.copy_(
                torch.tensor(changed_weights, device="cuda", dtype=torch.float32)
            )
            for name, arm in arms.items():
                mark("changed_graph_replay", arm=name)
                arm.poison()
                graphs[name].replay()
                mark("changed_graph_sync", arm=name)
                replay = arm.snapshot()
                report = validate(replay, changed_ids, changed_weights)
                arm.poison()
                arm()
                mark("changed_eager_sync", arm=name)
                eager = arm.snapshot()
                validate(eager, changed_ids, changed_weights)
                count = report["padded_rows"]
                exact = all(
                    replay[key][:length] == eager[key][:length]
                    for key, length in [
                        ("ids", count),
                        ("weights", count),
                        ("experts", count // BLOCK),
                        ("counts", 2),
                        ("m_indices", count),
                        ("reverse", rows * TOPK),
                    ]
                )
                assert (
                    report["ordered_route_sha256"]
                    != case["checks"][name]["original"]["ordered_route_sha256"]
                )
                case["checks"][name]["changed"] = report
                case["checks"][name]["changed_graph_exact"] = exact
                mark("changed_input_validated", arm=name)
            ids.copy_(torch.tensor(cpu_ids, device="cuda", dtype=torch.int32))
            weights.copy_(torch.tensor(cpu_weights, device="cuda", dtype=torch.float32))
            for arm in arms.values():
                arm()
            mark("eager_timing", rows=rows, distribution=distribution)
            case["eager_timing"] = timing(torch, arms, args.rounds)
            mark("graph_timing", rows=rows, distribution=distribution)
            case["graph_timing"] = timing(
                torch, {name: g.replay for name, g in graphs.items()}, args.rounds
            )
            case["status"] = "passed"
            save(args.output, result)
            print(
                json.dumps(
                    {
                        "rows": rows,
                        "distribution": distribution,
                        "status": "passed",
                        "graph_reduction": case["graph_timing"][
                            "mean_reduction_pct_vs_native"
                        ],
                    }
                ),
                flush=True,
            )


def supervise(command, output, seconds):
    """Enforce the deadline outside the Python process that can block in HIP."""
    with (output / "probe.log").open("w") as stream:
        child = subprocess.Popen(
            command, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True
        )
        try:
            return child.wait(timeout=seconds)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass  # Report unreaped state; outer container cleanup owns recovery.
            path = output / "result.json"
            result = json.loads(path.read_text()) if path.exists() else {}
            result.update(
                status="timed_out",
                timeout_seconds=seconds,
                child_returncode=child.poll(),
                hard_kill_sent=True,
            )
            save(output, result)
            print(
                json.dumps(
                    {
                        "status": "timed_out",
                        "timeout_seconds": seconds,
                        "last_progress": result.get("progress", [])[-1:],
                    }
                ),
                flush=True,
            )
            return 124
        finally:
            if child.poll() is None:
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass


def main():
    args = arguments()
    if not args.worker:
        args.output.mkdir(parents=True, exist_ok=False)

        def cancelled(signum, _frame):
            raise SystemExit(128 + signum)

        signal.signal(signal.SIGTERM, cancelled)
        return supervise(
            [sys.executable, "-u", __file__, *sys.argv[1:], "--worker"],
            args.output,
            args.max_seconds,
        )
    faulthandler.enable()
    faulthandler.dump_traceback_later(20, repeat=True)
    result = {
        "status": "running",
        "gpu_executed": False,
        "cases": [],
        "scope": __doc__,
        "probe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "requested_arms": args.arms,
        "requested_histograms": args.histograms,
        "limits": [
            "Synthetic routing histograms; no checkpoint or full-MoE/E2E result",
            "Byte arm uses physical512 with num_local_tokens=M and charges active ID/weight copies; not a pure mesh-threshold change",
            "Timing includes all sorter kernels, output zeroing and both auxiliary maps; excludes caller allocation",
            "Within-expert order changes may change downstream atomic accumulation; report separately from routing correctness",
        ],
    }
    started = time.monotonic()

    try:
        result["sources"] = source_checks(args)
        if args.source_only:
            result["status"] = "source_verified_gpu_not_run"
        else:
            result["gpu_executed"] = True
            run_gpu(args, result)
            result["status"] = "completed"
    except Exception as error:  # noqa: BLE001 - persist failure evidence, return nonzero
        result.update(
            status="failed", error=str(error), traceback=traceback.format_exc()
        )
    finally:
        faulthandler.cancel_dump_traceback_later()
        result["elapsed_seconds"] = time.monotonic() - started
        save(args.output, result)
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k in ("status", "error", "elapsed_seconds")
            }
        ),
        flush=True,
    )
    return int(result["status"] == "failed")


if __name__ == "__main__":
    raise SystemExit(main())

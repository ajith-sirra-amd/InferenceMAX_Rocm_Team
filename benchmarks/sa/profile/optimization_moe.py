#!/usr/bin/env python3
"""Bounded, isolated native FP4 MoE tile comparison; never a model benchmark.

Uses the pinned image's existing implementations without editing AITER/vLLM.
The timed operation includes native sorting, inline quantization, both GEMMs,
route weighting and output initialization. Synthetic weights/routing are shared
by all variants. Correctness is relative to a repeated native baseline, not a
dense reference or model-quality check. Results are saved after every stage.
"""

import argparse
import hashlib
import importlib
import inspect
import json
import math
import os
from pathlib import Path
import random
import signal
import statistics
import sys
import time
import traceback


EXPECTED_MOE_SHA256 = "dfe1f65b957656a58581651e3da9c4d0b94ca7ca65d7764551accc9bf6274541"
IMAGE = (
    "vllm/vllm-openai-rocm:kimi-k31-d73dd529@sha256:"
    "5b54f223a150a9baad41a5d511e159e7cbf43e308c6cd44ad3f6a2463c245b3c"
)
E, H, I, TOPK = 896, 3584, 384, 16
ARMS = {
    "baseline_bn128": (128, 128),
    "g1_bn256": (256, 128),
    "g2_bn256": (128, 256),
    "both_bn256": (256, 256),
}


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path, help="New evidence directory")
    parser.add_argument("--rows", default="280,336", help="Comma-separated token counts")
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--seed", type=int, default=355)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--timeout-seconds", type=int, default=570)
    parser.add_argument("--no-graph", action="store_true")
    args = parser.parse_args()
    try:
        args.rows = [int(value) for value in args.rows.split(",")]
    except ValueError:
        parser.error("--rows must contain integers")
    if not (1 <= len(args.rows) <= 3 and all(1 <= m <= 2048 for m in args.rows)):
        parser.error("Use one to three cases with 1 <= M <= 2048")
    if len(set(args.rows)) != len(args.rows):
        parser.error("Duplicate row counts are not useful")
    if not (3 <= args.iterations <= 100 and 1 <= args.warmup <= 10):
        parser.error("Use 3..100 iterations and 1..10 warmup calls")
    if not (30 <= args.timeout_seconds <= 590):
        parser.error("Timeout must be 30..590 seconds; outer runner timeout is 600 seconds")
    return args


class Evidence:
    def __init__(self, args):
        self.folder = args.output
        self.folder.mkdir(parents=True, exist_ok=True)
        self.path = self.folder / "result.json"
        if self.path.exists():
            raise FileExistsError(f"Refusing to overwrite {self.path}; use a new directory")
        self.started = time.monotonic()
        self.data = {
            "schema_version": 1,
            "status": "starting",
            "scope": __doc__,
            "expected_image": IMAGE,
            "parameters": {**vars(args), "output": str(args.output)},
            "geometry": {"experts": E, "hidden": H, "intermediate": I, "topk": TOPK},
            "timing_scope": "Full native MoE pair, including sorting/quantization/output zeroing; excludes router logits/top-k selection and input setup",
            "limitations": [
                "Synthetic physical-layout FP4 weights and uniform random routing; no checkpoint or observed expert histogram",
                "One GPU; excludes serving, tensor-parallel collectives and per-request E2E",
                "Correctness compares to a repeated native FP4 baseline, not a high-precision reference",
                "Repeated buffers/cache state differ from a running model",
                "GPU event elapsed time includes stream gaps; graph replay measures the captured complete operation",
                "No HBM/counter or model-quality evidence is collected by this probe",
            ],
            "cases": [],
            "progress": [],
        }
        self.save("created")

    def save(self, message):
        elapsed = round(time.monotonic() - self.started, 3)
        self.data["elapsed_seconds"] = elapsed
        self.data["progress"].append({"elapsed_seconds": elapsed, "message": message})
        temporary = self.folder / "result.json.tmp"
        with temporary.open("w") as stream:
            json.dump(self.data, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, self.path)
        print(f"[{elapsed:8.3f}s] {message}", flush=True)


def finite_number(value):
    value = float(value)
    return value if math.isfinite(value) else None


def describe_samples(samples):
    return {
        "samples_us": samples,
        "count": len(samples),
        "mean_us": statistics.mean(samples),
        "median_us": statistics.median(samples),
        "min_us": min(samples),
        "max_us": max(samples),
        "population_std_us": statistics.pstdev(samples),
    }


def error_metrics(torch, actual, reference):
    a, r = actual.float(), reference.float()
    finite = torch.isfinite(a) & torch.isfinite(r)
    count = int(finite.sum().item())
    result = {"elements": a.numel(), "finite_elements": count, "all_finite": count == a.numel()}
    if count != a.numel():
        return result
    delta = a - r
    rms = r.square().mean().sqrt().item()
    error_rms = delta.square().mean().sqrt().item()
    result.update(
        max_abs_error=finite_number(delta.abs().max().item()),
        rms_error=finite_number(error_rms),
        reference_rms=finite_number(rms),
        reference_max_abs=finite_number(r.abs().max().item()),
        relative_l2_error=finite_number(error_rms / max(rms, 1e-30)),
        unequal_elements=int((a != r).sum().item()),
    )
    return result


def assess(metrics, repeats):
    """Use the baseline noise envelope; keep this a provisional smoke gate."""
    if not metrics["all_finite"] or not all(row["all_finite"] for row in repeats):
        return {"status": "failed_nonfinite"}
    # One BF16 relative rounding unit prevents division by an accidentally zero
    # baseline difference; the empirical atomic-repeat noise usually dominates.
    unit = 2.0 ** -7
    rms_limit = max(3 * max(row["rms_error"] for row in repeats), unit * metrics["reference_rms"])
    max_limit = max(3 * max(row["max_abs_error"] for row in repeats), unit * metrics["reference_max_abs"])
    passed = metrics["rms_error"] <= rms_limit and metrics["max_abs_error"] <= max_limit
    return {
        "status": "within_baseline_noise_envelope" if passed else "needs_numerical_review",
        "rms_error_limit": rms_limit,
        "max_abs_error_limit": max_limit,
        "rule": "Both RMS/max error <= max(3x observed baseline-repeat error, 1 BF16 rounding unit x reference RMS/max)",
        "model_quality_validated": False,
    }


def timed(torch, operation, iterations):
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    torch.cuda.synchronize()
    start_wall = time.perf_counter()
    out = None
    for start, end in zip(starts, ends):
        start.record()
        out = operation()
        end.record()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start_wall
    result = describe_samples([float(start.elapsed_time(end)) * 1000 for start, end in zip(starts, ends)])
    result["batch_wall_seconds"] = elapsed
    return result, out


def paired_graph_timing(torch, baseline_graph, candidate_graph, iterations):
    events = {key: [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
                    for _ in range(iterations)] for key in ("baseline", "candidate")}
    graphs = {"baseline": baseline_graph, "candidate": candidate_graph}
    torch.cuda.synchronize()
    started = time.perf_counter()
    for index in range(iterations):
        order = ("baseline", "candidate") if index % 2 == 0 else ("candidate", "baseline")
        for key in order:
            start, end = events[key][index]
            start.record()
            graphs[key].replay()
            end.record()
    torch.cuda.synchronize()
    samples = {key: [float(s.elapsed_time(e)) * 1000 for s, e in pairs]
               for key, pairs in events.items()}
    return {
        "order": "alternating baseline/candidate then candidate/baseline",
        "baseline": describe_samples(samples["baseline"]),
        "candidate": describe_samples(samples["candidate"]),
        "candidate_over_baseline_ratios": [c / b for b, c in zip(samples["baseline"], samples["candidate"])],
        "median_paired_latency_ratio": statistics.median(c / b for b, c in zip(samples["baseline"], samples["candidate"])),
        "batch_wall_seconds": time.perf_counter() - started,
    }


def source_record(module):
    path = Path(inspect.getsourcefile(module))
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def gpu_poisoned(exc):
    message = str(exc).lower()
    return any(text in message for text in ("illegal memory", "device-side assert", "illegal address", "memory access fault"))


def run(args, evidence):
    # These affect only this standalone process and explicitly select FP4/BF16.
    os.environ["AITER_FLYDSL_STAGE2_FP8"] = "0"
    evidence.save("importing pinned Torch/AITER")
    import torch

    fm = importlib.import_module("aiter.fused_moe")
    torch.set_num_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError("A ROCm GPU is required")
    torch.cuda.set_device(args.device)
    properties = torch.cuda.get_device_properties(args.device)
    arch = str(getattr(properties, "gcnArchName", ""))
    if "gfx950" not in arch:
        raise RuntimeError(f"Expected MI355X/gfx950, got {properties.name}: {arch}")
    source = source_record(fm)
    evidence.data["environment"] = {
        "torch": torch.__version__, "hip": torch.version.hip,
        "python": sys.version, "device": properties.name, "architecture": arch,
        "device_total_bytes": properties.total_memory,
        "fused_moe_source": source,
        "aiter_environment": {key: value for key, value in sorted(os.environ.items()) if key.startswith("AITER_")},
    }
    evidence.save("checked GPU and recorded source provenance")
    if source["sha256"] != EXPECTED_MOE_SHA256:
        raise RuntimeError("Installed aiter/fused_moe.py differs from the reviewed pinned image")
    if fm.dtypes.fp4x2 == torch.uint8:
        raise RuntimeError("Native packed FP4 Torch dtype is unavailable")
    required = ["_fused_moe_impl", "_make_mxfp4_metadata", "GateMode"]
    for name in required:
        if not hasattr(fm, name):
            raise RuntimeError(f"Pinned native API missing: {name}")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    metadata = {}
    for name, (bn1, bn2) in ARMS.items():
        g1 = f"flydsl_mxmoe_g1_a4w4_16x{bn1}x256_f16in_nt_situv2_xcd2"
        g2 = f"flydsl_moe2_layout_afp4_wfp4_bf16_t16x{bn2}x128_atomic_nt_sbm16"
        metadata[name] = fm._make_mxfp4_metadata(g1, g2, fm.GateMode.SEPARATED, 0, block_m=16)
    evidence.data["variants"] = {
        name: {
            "gemm1": item.stage1.keywords["kernelName1"],
            "gemm2": item.stage2.keywords["kernelName2"],
            "sort_block_m": item.block_m, "inline_input_quantization": True,
        }
        for name, item in metadata.items()
    }
    evidence.save("allocating shared synthetic packed weights")
    # IID bytes in physical preshuffled storage are a valid synthetic FP4
    # matrix under any permutation. Constant scales are permutation-invariant.
    # No claim is made that these values match the model checkpoint.
    w1 = torch.randint(0, 256, (E, 2 * I, H // 2), dtype=torch.uint8, device="cuda").view(fm.dtypes.fp4x2)
    w2 = torch.randint(0, 256, (E, H, I // 2), dtype=torch.uint8, device="cuda").view(fm.dtypes.fp4x2)
    for weight in (w1, w2):
        weight.is_shuffled = True
        weight.gate_up_interleave = False
    s1 = torch.full((E, 2 * I, H // 32), 121, dtype=torch.uint8, device="cuda")
    # GEMM2's v2 shuffled E8M0 layout pads the 12 scale columns to 16.
    s2 = torch.full((E, H, ((I // 32 + 7) // 8) * 8), 121, dtype=torch.uint8, device="cuda")
    evidence.data["weights"] = {
        "w1_packed_shape": list(w1.shape), "w2_packed_shape": list(w2.shape),
        "w1_scale_storage_shape": list(s1.shape), "w2_scale_storage_shape": list(s2.shape),
        "packed_dtype": str(w1.dtype), "scale_storage_dtype": str(s1.dtype),
        "scale_e8m0_byte": 121, "scale_value": 2 ** (121 - 127),
        "allocated_bytes": sum(t.numel() * t.element_size() for t in (w1, w2, s1, s2)),
        "is_shuffled": True, "gate_up_interleave": False,
    }
    torch.cuda.synchronize()
    evidence.save("shared weights ready")

    for m in args.rows:
        case = {"M": m, "status": "starting", "arms": {}}
        evidence.data["cases"].append(case)
        evidence.save(f"M{m}: preparing fixed hidden states and routing")
        try:
            x = torch.randn((m, H), dtype=torch.bfloat16, device="cuda")
            # Generate top-k without replacement on CPU, outside the timed op.
            generator = torch.Generator().manual_seed(args.seed + m)
            ids_cpu = torch.rand((m, E), generator=generator).topk(TOPK, dim=1).indices.to(torch.int32)
            ids = ids_cpu.to("cuda")
            route_cpu = torch.rand((m, TOPK), generator=generator)
            route_cpu /= route_cpu.sum(dim=1, keepdim=True)
            route = route_cpu.to("cuda")
            histogram = torch.bincount(ids_cpu.flatten().long(), minlength=E)
            case["routing"] = {
                "distribution": "uniform random top-k without replacement; fixed across arms",
                "expert_route_counts": histogram.tolist(),
                "padded_sorted_rows_bm16": int(((histogram + 15) // 16 * 16).sum().item()),
                "ids_sha256": hashlib.sha256(ids_cpu.numpy().tobytes()).hexdigest(),
                "weights_sha256": hashlib.sha256(route_cpu.numpy().tobytes()).hexdigest(),
            }

            def operation(name):
                return fm._fused_moe_impl(
                    x, w1, w2, route, ids,
                    activation=fm.ActivationType.Situv2.value,
                    quant_type=fm.QuantType.per_1x32.value,
                    doweight_stage1=False, w1_scale=s1, w2_scale=s2,
                    block_size_M=16, dtype=torch.bfloat16,
                    beta=4.0, linear_beta=25.0,
                    gate_mode=fm.GateMode.SEPARATED.value,
                    _q_dtype_a=fm.dtypes.fp4x2,
                    _metadata_transform=lambda _: metadata[name],
                    _enforce_metadata_block_size=True,
                )

            evidence.save(f"M{m}: compiling/warming native baseline")
            for _ in range(args.warmup):
                operation("baseline_bn128")
            torch.cuda.synchronize()
            reference = operation("baseline_bn128").clone()
            torch.cuda.synchronize()
            repeats = []
            for _ in range(3):
                repeated = operation("baseline_bn128")
                torch.cuda.synchronize()
                repeats.append(error_metrics(torch, repeated, reference))
            case["baseline_repeat_errors"] = repeats
            if not all(row["all_finite"] for row in repeats):
                raise RuntimeError("Native baseline produced nonfinite values")
            if repeats[0]["reference_rms"] == 0:
                raise RuntimeError("Native baseline is all zero; numerical comparison would be vacuous")
            evidence.save(f"M{m}: baseline reference and atomic noise floor saved")
            arm_names = list(ARMS)
            # Alternate the middle candidates across cases to expose order bias.
            random.Random(args.seed + m).shuffle(arm_names)
            case["measurement_order"] = arm_names
            for name in arm_names:
                result = {"status": "warming"}
                case["arms"][name] = result
                evidence.save(f"M{m} {name}: compiling/warming full pair")
                try:
                    for _ in range(args.warmup):
                        operation(name)
                    torch.cuda.synchronize()
                    actual = operation(name)
                    torch.cuda.synchronize()
                    result["error"] = error_metrics(torch, actual, reference)
                    result["numerical_check"] = assess(result["error"], repeats)
                    result["eager"], _ = timed(torch, lambda: operation(name), args.iterations)
                    evidence.save(f"M{m} {name}: eager timing/error saved")
                    if not args.no_graph:
                        result["graph"] = {"status": "capturing"}
                        evidence.save(f"M{m} {name}: capturing full-pair HIP graph")
                        try:
                            side = torch.cuda.Stream()
                            side.wait_stream(torch.cuda.current_stream())
                            with torch.cuda.stream(side):
                                for _ in range(args.warmup):
                                    operation(name)
                            torch.cuda.current_stream().wait_stream(side)
                            torch.cuda.synchronize()
                            graph = torch.cuda.CUDAGraph()
                            with torch.cuda.graph(graph, stream=side):
                                graph_output = operation(name)
                            graph.replay()
                            torch.cuda.synchronize()
                            graph_error = error_metrics(torch, graph_output, reference)
                            result["graph"] = {
                                "status": "checking_changed_input",
                                "error": graph_error,
                                "numerical_check": assess(graph_error, repeats),
                            }
                            evidence.save(f"M{m} {name}: graph first-input check saved; checking changed input")
                            original_x = x.clone()
                            try:
                                x.mul_(0.75).add_(0.0625)
                                changed_reference = operation("baseline_bn128").clone()
                                torch.cuda.synchronize()
                                changed_repeats = []
                                for _ in range(3):
                                    changed_actual = operation("baseline_bn128")
                                    torch.cuda.synchronize()
                                    changed_repeats.append(error_metrics(torch, changed_actual, changed_reference))
                                graph.replay()
                                torch.cuda.synchronize()
                                changed_error = error_metrics(torch, graph_output, changed_reference)
                                result["graph"]["changed_input"] = {
                                    "mutation": "hidden = hidden * 0.75 + 0.0625, in place at the original address",
                                    "reference_change": error_metrics(torch, changed_reference, reference),
                                    "baseline_repeat_errors": changed_repeats,
                                    "error": changed_error,
                                    "numerical_check": assess(changed_error, changed_repeats),
                                }
                            finally:
                                x.copy_(original_x)
                                torch.cuda.synchronize()
                            del original_x, changed_reference, changed_actual
                            evidence.save(f"M{m} {name}: changed-input replay saved; timing paired graphs")
                            baseline_graph = torch.cuda.CUDAGraph()
                            with torch.cuda.graph(baseline_graph, stream=side):
                                baseline_graph_output = operation("baseline_bn128")
                            baseline_graph.replay()
                            graph.replay()
                            torch.cuda.synchronize()
                            paired = paired_graph_timing(torch, baseline_graph, graph, args.iterations)
                            result["graph"]["paired_timing"] = paired
                            result["graph"]["timing"] = paired["candidate"]
                            result["graph"]["status"] = "completed"
                            del graph, graph_output, baseline_graph, baseline_graph_output
                        except Exception as exc:
                            result["graph"].update(status="failed", failure=str(exc), traceback=traceback.format_exc())
                            if gpu_poisoned(exc) or isinstance(exc, TimeoutError):
                                raise
                    result["numerical_and_graph_smoke_pass"] = (
                        result["numerical_check"]["status"] == "within_baseline_noise_envelope"
                        and result.get("graph", {}).get("status") == "completed"
                        and result["graph"]["numerical_check"]["status"] == "within_baseline_noise_envelope"
                        and result["graph"]["changed_input"]["numerical_check"]["status"] == "within_baseline_noise_envelope"
                        and result["graph"]["changed_input"]["reference_change"].get("max_abs_error", 0) > 0
                    )
                    result["status"] = "completed"
                    evidence.save(f"M{m} {name}: saved full result")
                except Exception as exc:
                    result["status"] = "failed"
                    result["failure"] = {"error": str(exc), "traceback": traceback.format_exc()}
                    evidence.save(f"M{m} {name}: failure retained: {exc}")
                    if gpu_poisoned(exc) or isinstance(exc, TimeoutError):
                        raise
            case["baseline_after"], _ = timed(torch, lambda: operation("baseline_bn128"), args.iterations)
            baseline = case["arms"].get("baseline_bn128", {})
            if "eager" in baseline:
                for result in case["arms"].values():
                    if "eager" in result:
                        result["eager_latency_ratio_vs_baseline"] = result["eager"]["median_us"] / baseline["eager"]["median_us"]
                    if result.get("graph", {}).get("status") == baseline.get("graph", {}).get("status") == "completed":
                        result["graph_latency_ratio_vs_baseline"] = result["graph"]["timing"]["median_us"] / baseline["graph"]["timing"]["median_us"]
            case["status"] = "completed" if all(r["status"] == "completed" for r in case["arms"].values()) else "partial"
            del reference, x, ids, route
        except Exception as exc:
            case["status"] = "failed"
            case["failure"] = {"error": str(exc), "traceback": traceback.format_exc()}
            evidence.save(f"M{m}: case failure retained: {exc}")
            if gpu_poisoned(exc) or isinstance(exc, TimeoutError):
                raise
        evidence.save(f"M{m}: case {case['status']}")

    evidence.data["max_memory_allocated_bytes"] = torch.cuda.max_memory_allocated()
    evidence.data["status"] = "completed" if all(case["status"] == "completed" for case in evidence.data["cases"]) else "partial"
    evidence.data["all_numerical_and_graph_smoke_checks_pass"] = all(
        case["status"] == "completed" and all(arm.get("numerical_and_graph_smoke_pass", False)
                                             for arm in case["arms"].values())
        for case in evidence.data["cases"]
    )
    evidence.data["exit_status_scope"] = "Process success means evidence collection completed; acceptance requires all_numerical_and_graph_smoke_checks_pass plus review of each candidate. No model correctness or E2E claim."
    evidence.save("bounded MoE comparison finished; inspect numerical and graph checks before accepting any candidate")


def main():
    args = arguments()
    evidence = Evidence(args)

    def deadline(_signal, _frame):
        raise TimeoutError(f"Probe reached {args.timeout_seconds}s budget")

    signal.signal(signal.SIGALRM, deadline)
    signal.alarm(args.timeout_seconds)
    try:
        run(args, evidence)
    except Exception as exc:
        evidence.data["status"] = "failed"
        evidence.data["failure"] = {"error": str(exc), "traceback": traceback.format_exc()}
        evidence.save(f"stopped; evidence preserved: {exc}")
        return 1
    finally:
        signal.alarm(0)
    return 0 if evidence.data["status"] == "completed" else 1


if __name__ == "__main__":
    sys.exit(main())

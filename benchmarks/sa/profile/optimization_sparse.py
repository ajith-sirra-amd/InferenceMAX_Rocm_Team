#!/usr/bin/env python3
"""Bounded GPU comparison of two sparse-DCP source candidates, without a model.

Extracts the installed Triton kernels into isolated in-memory modules. No package
files are edited. Rank values are emulated on one GPU; no collective is issued.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import linecache
import math
import signal
import statistics
import sys
import time
import traceback
import types
from pathlib import Path


SOURCE_SHA256 = "ede30a753116e4f8c0c26e2672b40419bc96a26f686ac91190bdb630214184bc"
SOURCE_RELATIVE = "vllm/v1/attention/backends/mla/rocm_aiter_mla_sparse.py"
K, WORLD, PAGE, PHYSICAL_STRIDE = 2048, 8, 64, 80


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def modified_source(source, kind):
    if kind == "pages":
        edits = [
            ("    Local,\n    Pages,\n    Staging,", "    Local,\n    Pages,\n    Requests,\n    Staging,"),
            ("    PS: tl.constexpr,\n    K: tl.constexpr,", "    PS: tl.constexpr,\n    PC: tl.constexpr,\n    RS: tl.constexpr,\n    K: tl.constexpr,"),
            ("    page = tl.load(Pages + row * PS + local // Page, keep, other=0)",
             "    request = tl.load(Requests + row * RS)\n    page = tl.load(Pages + request * PS + (local // Page) * PC, keep, other=0)"),
        ]
    else:
        edits = [
            ("    value = tl.load(source + row * stride + col, col < K, other=-1)\n    if ROWS == 0:\n        keep = value >= 0",
             "    if ROWS == 0:\n        value = tl.load(source + row * stride + col, col < K, other=-1)\n        keep = value >= 0"),
            ("        keep = col < count\n        slot = col\n        tl.store(indptr + row + 1, start + count)",
             "        keep = col < count\n        value = tl.load(source + row * stride + col, keep, other=-1)\n        slot = col\n        tl.store(indptr + row + 1, start + count)"),
        ]
    for old, new in edits:
        if source.count(old) != 1:
            raise RuntimeError(f"Source transform {kind} did not match exactly once")
        source = source.replace(old, new)
    return source


def kernel_module(source, name, triton, tl):
    names = {"_score_key", "_select_owned_slots", "_pack_valid_rows"}
    nodes = [node for node in ast.parse(source).body
             if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in nodes} != names:
        raise RuntimeError("Missing expected source kernels")
    code = "\n\n".join("@triton.jit\n" + ast.get_source_segment(source, node)
                         for node in nodes) + "\n"
    filename = f"<sparse_probe_{name}_{digest(code)[:16]}.py>"
    module_name = f"sparse_probe_{name}_{digest(code)[:16]}"
    module = types.ModuleType(module_name)
    module.__file__ = filename
    module.triton, module.tl = triton, tl
    sys.modules[module_name] = module
    linecache.cache[filename] = (len(code), None, code.splitlines(True), filename)
    exec(compile(code, filename, "exec"), module.__dict__)
    return module, digest(code)


class Results:
    def __init__(self, path, args):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.started = time.monotonic()
        self.data = {"status": "running", "kind": "isolated_sparse_gpu_comparison",
                     "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                     "correctness": [], "timings": [], "progress": [],
                     "limits": ["Synthetic local rank inputs; no distributed collective",
                                "No full model or request-E2E measurement",
                                "Page-table reuse is reported separately from materialization"]}

    def save(self, phase, **details):
        record = {"phase": phase, "elapsed_seconds": round(time.monotonic()-self.started, 3), **details}
        self.data["progress"].append(record)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(json.dumps(self.data, indent=2, allow_nan=False) + "\n")
        temporary.replace(self.path)
        print(json.dumps(record), flush=True)


def score_keys(np, bits, nan_high=False):
    signed = bits.view(np.int32)
    value = signed ^ ((signed >> 31) & 0x7fffffff)
    return np.where((bits & 0x7fffffff) > 0x7f800000,
                    2147483647 if nan_high else -2147483648, value)


def make_data(torch, np, rows, seed, special=False, strided=False):
    rng = np.random.default_rng(seed)
    requests = max(1, math.ceil(rows/7))
    req = np.arange(rows, dtype=np.int32) // 7
    page_width = 2048
    table = rng.integers(0, 50000, (requests, page_width), dtype=np.int32)
    local = ((np.arange(K)[None, :] * 61 + np.arange(rows)[:, None] * 17)
             % (page_width * PAGE)).astype(np.int32)
    local[:, ::37] = -1
    scores = rng.standard_normal((rows, WORLD*K)).astype(np.float32)
    scores.reshape(rows, WORLD, K)[:, :, ::37] = -np.inf
    bits = scores.view(np.uint32)
    if special:
        patterns = np.array([0, 0x80000000, 0x7f800000, 0xff800000,
                             0x7fc12345, 0xffc54321, 1], dtype=np.uint32)
        bits[:, 1:8] = patterns
        bits[::3, K+1:K+8] = patterns
    producer_keys = score_keys(np, bits, nan_high=True)
    selected = np.argpartition(producer_keys, -K, axis=1)[:, -K:]
    top = np.take_along_axis(bits, selected, axis=1).copy().view(np.float32)
    device = "cuda"
    if strided:
        table_store = torch.empty((requests, page_width*2+5), device=device, dtype=torch.int32)
        table_gpu = table_store[:, :page_width*2:2]
        req_store = torch.empty(rows*2, device=device, dtype=torch.int32)
        req_gpu = req_store[::2]
    else:
        table_gpu = torch.empty(table.shape, device=device, dtype=torch.int32)
        req_gpu = torch.empty(rows, device=device, dtype=torch.int32)
    table_gpu.copy_(torch.from_numpy(table))
    req_gpu.copy_(torch.from_numpy(req))
    return {"rows": rows, "table": table_gpu, "req": req_gpu,
            "scores": torch.from_numpy(scores).to(device), "top": torch.from_numpy(top).to(device),
            "local": torch.from_numpy(local).to(device),
            "cpu": {"table": table, "req": req, "scores": scores, "top": top, "local": local},
            "special": special, "strided": strided}


def update_data(torch, target, fresh):
    for key in ("table", "req", "scores", "top", "local"):
        target[key].copy_(fresh[key])
    target["cpu"] = fresh["cpu"]


def reference(np, data, rank):
    cpu = data["cpu"]
    top = score_keys(np, cpu["top"].view(np.uint32))
    threshold = top.min(1)
    room = K - (top > threshold[:, None]).sum(1)
    all_keys = score_keys(np, cpu["scores"].view(np.uint32))
    own = all_keys[:, rank*K:(rank+1)*K]
    live = cpu["local"] >= 0
    tied = live & (own == threshold[:, None])
    prior = (all_keys[:, :rank*K] == threshold[:, None]).sum(1)
    keep = ((live & (own > threshold[:, None])) |
            (tied & (np.cumsum(tied, axis=1)-tied < (room-prior)[:, None])))
    counts = keep.sum(1).astype(np.int32)
    staging = np.full(cpu["local"].shape, -777, dtype=np.int32)
    for row in range(data["rows"]):
        positions = cpu["local"][row, keep[row]]
        staging[row, :counts[row]] = (cpu["table"][cpu["req"][row], positions//PAGE]
                                     * PHYSICAL_STRIDE + positions % PAGE)
    return staging, counts


def select_call(torch, modules, data, rank, direct, cached=None, initialize=False):
    rows = data["rows"]
    staging = torch.empty_like(data["local"])
    counts = torch.empty(rows, device="cuda", dtype=torch.int32)
    if initialize:
        staging.fill_(-777)
    pages = data["table"] if direct else (cached if cached is not None else data["table"][data["req"]])
    args = [data["scores"], data["top"], data["local"], pages]
    if direct:
        args.append(data["req"])
    args += [staging, counts, rank, data["scores"].stride(0), data["top"].stride(0),
             data["local"].stride(0), pages.stride(0)]
    if direct:
        args += [pages.stride(1), data["req"].stride(0)]
    args += [K, PAGE, PHYSICAL_STRIDE]
    modules["pages" if direct else "base"]._select_owned_slots[(rows,)](*args, num_warps=8)
    return staging, counts


def pack_call(torch, modules, staging, counts, changed, initialize=False, no_counts=False):
    rows, k = staging.shape
    if no_counts:
        valid = (staging >= 0).sum(1, dtype=torch.int32)
        indptr = torch.zeros(rows+1, dtype=torch.int32, device="cuda")
        torch.cumsum(valid, dim=0, out=indptr[1:])
    else:
        indptr = torch.empty(rows+1, dtype=torch.int32, device="cuda")
    packed = torch.empty(rows*k, dtype=torch.int32, device="cuda")
    if initialize:
        packed.fill_(-777)
    modules["pack" if changed else "base"]._pack_valid_rows[(rows,)](
        staging, staging.stride(0), indptr, packed, counts, rows, k,
        1 << (k-1).bit_length(), 0 if no_counts else max(16, 1 << (rows-1).bit_length()))
    return packed, indptr


def capture(torch, fn):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            fn()
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        output = fn()
    torch.cuda.current_stream().wait_stream(stream)
    return graph, output


def assert_selection(torch, np, result, expected):
    staging, counts = result
    expected_staging, expected_counts = expected
    torch.cuda.synchronize()
    np.testing.assert_array_equal(counts.cpu().numpy(), expected_counts)
    actual = staging.cpu().numpy()
    for row, count in enumerate(expected_counts):
        np.testing.assert_array_equal(actual[row, :count], expected_staging[row, :count])


def assert_packed(torch, np, result, staging, counts):
    packed, indptr = result
    torch.cuda.synchronize()
    expected_indptr = np.r_[0, counts.cumsum()].astype(np.int32)
    expected = np.concatenate([staging[row, :count] for row, count in enumerate(counts)])
    np.testing.assert_array_equal(indptr.cpu().numpy(), expected_indptr)
    np.testing.assert_array_equal(packed[:expected.size].cpu().numpy(), expected)


def correctness(torch, np, modules, output):
    for rows, special, strided in ((280, False, False), (336, True, False), (14, True, True)):
        data = make_data(torch, np, rows, 1700+rows, special, strided)
        for rank in range(WORLD):
            expected = reference(np, data, rank)
            for direct in (False, True):
                result = select_call(torch, modules, data, rank, direct, initialize=True)
                assert_selection(torch, np, result, expected)
                for changed in (False, True):
                    packed = pack_call(torch, modules, *result, changed, initialize=True)
                    assert_packed(torch, np, packed, *expected)
        output.data["correctness"].append({"rows": rows, "ranks": list(range(WORLD)),
                                           "special_scores": special, "strided": strided,
                                           "eager_selection_and_csr": "passed"})
        output.save("eager_correctness", rows=rows, special=special, strided=strided)
        # Keep graph-bound storage fixed while modifying its contents.
        rank = 7
        cached = data["table"][data["req"]]
        graphs = []
        for direct, reuse in ((False, False), (False, True), (True, False)):
            fn = lambda direct=direct, reuse=reuse: select_call(
                torch, modules, data, rank, direct, cached if reuse else None)
            graph, result = capture(torch, fn)
            graphs.append((graph, result))
        fresh = make_data(torch, np, rows, 9100+rows, not special, strided)
        update_data(torch, data, fresh)
        # A metadata-reuse consumer assumes the shared cached table was refreshed.
        # Refresh is deliberately outside its timed consumer interval.
        cached.copy_(data["table"][data["req"]])
        expected = reference(np, data, rank)
        for graph, result in graphs:
            graph.replay()
            assert_selection(torch, np, result, expected)
        stage = torch.from_numpy(expected[0]).to("cuda")
        count = torch.from_numpy(expected[1]).to("cuda")
        pack_graphs = [capture(torch, lambda changed=changed: pack_call(
            torch, modules, stage, count, changed)) for changed in (False, True)]
        # Changed counts include zero/full rows; rewritten valid prefixes differ.
        counts2 = np.arange(rows, dtype=np.int32) * 37 % (K+1)
        counts2[0], counts2[-1] = 0, K
        stage2 = np.arange(rows*K, dtype=np.int32).reshape(rows, K)
        stage.copy_(torch.from_numpy(stage2))
        count.copy_(torch.from_numpy(counts2))
        for graph, result in pack_graphs:
            graph.replay()
            assert_packed(torch, np, result, stage2, counts2)
        output.data["correctness"][-1]["changed_data_graph_replay"] = "passed"
        output.save("graph_correctness", rows=rows)
    # The no-count branch must preserve interspersed invalid entries.
    stage_np = np.arange(7*K, dtype=np.int32).reshape(7, K)
    stage_np[:, ::3] = -1
    stage = torch.from_numpy(stage_np).to("cuda")
    count = (stage >= 0).sum(1, dtype=torch.int32)
    expected_counts = (stage_np >= 0).sum(1).astype(np.int32)
    expected_stage = np.full_like(stage_np, -1)
    for row in range(7):
        expected_stage[row, :expected_counts[row]] = stage_np[row, stage_np[row] >= 0]
    for changed in (False, True):
        result = pack_call(torch, modules, stage, count, changed, no_counts=True)
        assert_packed(torch, np, result, expected_stage, expected_counts)
    output.data["correctness"].append({"no_counts_branch": "passed"})
    output.save("correctness_complete")


def timed_pair(torch, baseline, candidate, rounds, repeats, warmup):
    for _ in range(warmup):
        baseline()
        candidate()
    torch.cuda.synchronize()
    samples = []
    for round_id in range(rounds):
        record = {"round": round_id, "order": ["baseline", "candidate"] if round_id%2 == 0 else ["candidate", "baseline"]}
        for name in record["order"]:
            fn = baseline if name == "baseline" else candidate
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(repeats):
                retained_output = fn()
            end.record()
            end.synchronize()
            record[name + "_us"] = start.elapsed_time(end)*1000/repeats
            del retained_output
        samples.append(record)
    def summary(name):
        values = [sample[name + "_us"] for sample in samples]
        return {"mean_us": statistics.mean(values), "median_us": statistics.median(values),
                "min_us": min(values), "max_us": max(values), "std_us": statistics.pstdev(values)}
    base, changed = summary("baseline"), summary("candidate")
    return {"baseline": base, "candidate": changed, "samples": samples,
            "mean_latency_reduction_pct": 100*(1-changed["mean_us"]/base["mean_us"]),
            "candidate_faster_rounds": sum(x["candidate_us"] < x["baseline_us"] for x in samples)}


def timings(torch, np, modules, output, args):
    for rows in (280, 336):
        data = make_data(torch, np, rows, 5000+rows)
        for rank in (0, 7):
            cached = data["table"][data["req"]]
            for reuse in (False, True):
                baseline = lambda: select_call(torch, modules, data, rank, False, cached if reuse else None)
                candidate = lambda: select_call(torch, modules, data, rank, True)
                for mode in ("eager", "graph"):
                    graph_refs = []
                    left, right = baseline, candidate
                    if mode == "graph":
                        for fn in (baseline, candidate):
                            graph_refs.append(capture(torch, fn))
                        left, right = graph_refs[0][0].replay, graph_refs[1][0].replay
                    result = timed_pair(torch, left, right, args.rounds, args.repeats, args.warmup)
                    result.update({"proposal": "direct_page_lookup", "rows": rows, "rank": rank, "mode": mode,
                                   "scope": "cached_table_consumer" if reuse else "table_materialization_plus_selection",
                                   "allocation_mode": "eager_wrapper" if mode == "eager" else "fixed_capture_pool",
                                   "cached_table_refresh_included": False,
                                   "page_table_shape": list(data["table"].shape), "page_table_stride": list(data["table"].stride())})
                    output.data["timings"].append(result)
                    output.save("timing_case", proposal=result["proposal"], rows=rows, rank=rank, mode=mode, scope=result["scope"])
        # Preserve a realistic approximately 1/8 selected population from the same selector.
        staging, counts = select_call(torch, modules, data, 7, True)
        torch.cuda.synchronize()
        cpu_counts = counts.cpu().tolist()
        baseline = lambda: pack_call(torch, modules, staging, counts, False)
        candidate = lambda: pack_call(torch, modules, staging, counts, True)
        for mode in ("eager", "graph"):
            graph_refs = []
            left, right = baseline, candidate
            if mode == "graph":
                graph_refs = [capture(torch, fn) for fn in (baseline, candidate)]
                left, right = graph_refs[0][0].replay, graph_refs[1][0].replay
            result = timed_pair(torch, left, right, args.rounds, args.repeats, args.warmup)
            result.update({"proposal": "masked_staging_read", "rows": rows, "rank": 7, "mode": mode,
                           "scope": "csr_allocation_and_pack",
                           "allocation_mode": "eager_wrapper" if mode == "eager" else "fixed_capture_pool",
                           "counts_min": min(cpu_counts), "counts_max": max(cpu_counts),
                           "counts_mean": statistics.mean(cpu_counts), "counts_total": sum(cpu_counts)})
            output.data["timings"].append(result)
            output.save("timing_case", proposal=result["proposal"], rows=rows, mode=mode)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="Incrementally written result.json path")
    parser.add_argument("--source", type=Path, help="Installed sparse backend source; auto-detected on sys.path")
    parser.add_argument("--rounds", type=int, default=12)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--max-seconds", type=int, default=600)
    args = parser.parse_args()
    if not (2 <= args.rounds <= 30 and 1 <= args.repeats <= 50 and 1 <= args.warmup <= 10
            and 30 <= args.max_seconds <= 900):
        parser.error("Require rounds2–30, repeats1–50, warmup1–10, max-seconds30–900")
    output = Results(args.output, args)
    output.save("starting")
    def expired(signum, frame):
        raise TimeoutError(f"Exceeded {args.max_seconds} seconds")
    signal.signal(signal.SIGALRM, expired)
    signal.alarm(args.max_seconds)
    try:
        import numpy as np
        import torch
        import triton
        import triton.language as tl
        if not torch.cuda.is_available() or not torch.version.hip:
            raise RuntimeError("A ROCm GPU is required")
        source_path = args.source
        if source_path is None:
            source_path = next((Path(p)/SOURCE_RELATIVE for p in sys.path if (Path(p)/SOURCE_RELATIVE).is_file()), None)
        if source_path is None:
            raise FileNotFoundError(SOURCE_RELATIVE)
        source = source_path.read_text()
        if digest(source) != SOURCE_SHA256:
            raise RuntimeError(f"Installed source hash mismatch: {digest(source)}; expected {SOURCE_SHA256}")
        modules, hashes = {}, {}
        for kind in ("base", "pages", "pack"):
            changed = source if kind == "base" else modified_source(source, kind)
            modules[kind], hashes[kind] = kernel_module(changed, kind, triton, tl)
        output.data["provenance"] = {"source": str(source_path), "source_sha256": digest(source),
                                      "extracted_kernel_sha256": hashes, "torch": torch.__version__,
                                      "hip": torch.version.hip, "triton": triton.__version__,
                                      "device": torch.cuda.get_device_name(), "K": K, "world_emulated": WORLD,
                                      "page_size": PAGE, "physical_block_stride_rows": PHYSICAL_STRIDE}
        output.save("source_verified")
        correctness(torch, np, modules, output)
        timings(torch, np, modules, output, args)
        output.data["status"] = "passed"
        output.data["peak_torch_allocated_bytes"] = torch.cuda.max_memory_allocated()
        output.save("complete", timing_cases=len(output.data["timings"]))
    except BaseException as error:
        output.data["status"] = "failed"
        output.data["error"] = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
        output.save("failed", error=str(error))
        raise
    finally:
        signal.alarm(0)


if __name__ == "__main__":
    main()

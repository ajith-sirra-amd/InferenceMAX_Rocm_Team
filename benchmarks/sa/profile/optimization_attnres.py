#!/usr/bin/env python3
"""Bounded, synthetic AttnRes product-hoist experiment; no model or source edits.

Default: H7168, rows280/336, blocks4/8. For the wider matrix use
--rows 252,280,336,1024 --blocks 1,4,8. Result/progress survives case failures.
Use --bf16-residual-only for a bounded row-coverage check at blocks4/8.
Timings are unprofiled graph event intervals, not request E2E measurements.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import statistics
import sys
import time
import traceback


BASELINE_SHA256 = "b90ad19c15e8d2732634909a5f0ae80ac7d440eaa340d3bc92d89c8576df4e9a"
CANDIDATE_SHA256 = "99d21f118d6a53b00e12596529955f68bd476e792124704b078b7fba5263fd9e"
PUBLISHED_PATCH_SHA256 = "a92d1506abacf53116295abc2d6ae0d3ac561294402f5121daf3a0f14675f78f"
SOURCE_RELATIVE = "models/kimi_k3/amd/ops/attn_res.py"
RMS_EPS = 1e-5
KERNEL_PATCH = r"""--- a/vllm/models/kimi_k3/amd/ops/attn_res.py
+++ b/vllm/models/kimi_k3/amd/ops/attn_res.py
@@ -35,6 +35,7 @@
     eps: tl.constexpr,
     output_norm_eps: tl.constexpr,
     HAS_DELTA: tl.constexpr,
+    PRECOMPUTED_INPUT_QK: tl.constexpr,
     WRITE_BLOCK: tl.constexpr,
     APPLY_OUTPUT_NORM: tl.constexpr,
     QUANT_MAX: tl.constexpr,
@@ -80,11 +81,16 @@
         # Reloading avoids keeping the full prefix vector live across the loop.
         if HAS_DELTA:
             tl.debug_barrier()
-        input_qk_weight = tl.load(
-            norm_weight_ptr + d_offsets, mask=d_mask, other=0.0
-        ).to(tl.float32) * tl.load(
-            qk_weight_ptr + d_offsets, mask=d_mask, other=0.0
-        ).to(tl.float32)
+        if PRECOMPUTED_INPUT_QK:
+            input_qk_weight = tl.load(
+                norm_weight_ptr + d_offsets, mask=d_mask, other=0.0
+            ).to(tl.float32)
+        else:
+            input_qk_weight = tl.load(
+                norm_weight_ptr + d_offsets, mask=d_mask, other=0.0
+            ).to(tl.float32) * tl.load(
+                qk_weight_ptr + d_offsets, mask=d_mask, other=0.0
+            ).to(tl.float32)
         max_logit = tl.full((), -float("inf"), tl.float32)
         denominator = tl.zeros((), tl.float32)
         mixed = tl.zeros((BLOCK_D,), tl.float32)
@@ -160,8 +166,14 @@
     output_norm_eps: float,
     *,
     quant_dtype: torch.dtype | None = None,
+    input_qk_weight: torch.Tensor | None = None,
 ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
     num_tokens, hidden_size = prefix.shape
+    if input_qk_weight is not None:
+        assert input_qk_weight.shape == (hidden_size,)
+        assert input_qk_weight.dtype == torch.float32
+        assert input_qk_weight.device == prefix.device
+        assert input_qk_weight.stride(0) == 1
     assert prefix.stride(-1) == 1
     assert delta is None or delta.stride(-1) == 1
     assert blocks.stride(-1) == 1
@@ -190,7 +202,7 @@
         prefix,
         delta,
         blocks,
-        norm_weight,
+        norm_weight if input_qk_weight is None else input_qk_weight,
         qk_weight,
         output_norm_weight,
         output,
@@ -206,6 +218,7 @@
         eps,
         output_norm_eps,
         HAS_DELTA=delta is not None,
+        PRECOMPUTED_INPUT_QK=input_qk_weight is not None,
         WRITE_BLOCK=block_write_idx >= 0,
         APPLY_OUTPUT_NORM=output_norm_weight is not None,
         QUANT_MAX=0.0 if quant_dtype is None else get_fp8_min_max()[1],
"""


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def apply_kernel_patch(source):
    """Apply exact context/offset-checked published hunks without a patch binary."""
    lines, result, consumed = source.splitlines(keepends=True), [], 0
    diff = KERNEL_PATCH.splitlines(keepends=True)
    i = 0
    while i < len(diff):
        if not diff[i].startswith("@@ "):
            i += 1
            continue
        match = re.match(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", diff[i])
        if match is None:
            raise ValueError("Invalid embedded hunk")
        start = int(match[1]) - 1
        old, new = [], []
        i += 1
        while i < len(diff) and not diff[i].startswith("@@ "):
            line = diff[i]
            if line[0] in " -":
                old.append(line[1:])
            if line[0] in " +":
                new.append(line[1:])
            if line[0] not in " +-":
                raise ValueError("Unexpected embedded patch line")
            i += 1
        if start < consumed or lines[start:start + len(old)] != old:
            raise ValueError("Frozen source does not match hunk context")
        if len(old) != int(match[2] or 1) or len(new) != int(match[4] or 1):
            raise ValueError("Embedded hunk line count differs")
        result.extend(lines[consumed:start])
        result.extend(new)
        consumed = start + len(old)
    result.extend(lines[consumed:])
    candidate = "".join(result)
    if digest(candidate.encode()) != CANDIDATE_SHA256:
        raise ValueError("Published candidate source hash differs")
    return candidate


def prepare_sources(output, source_file):
    if source_file is None:
        spec = importlib.util.find_spec("vllm")
        if spec is None or spec.origin is None:
            raise RuntimeError("Pinned vLLM package is not installed")
        source_file = Path(spec.origin).parent / SOURCE_RELATIVE
    baseline = source_file.read_bytes()
    if digest(baseline) != BASELINE_SHA256:
        raise ValueError(f"AttnRes source hash mismatch at {source_file}: {digest(baseline)}")
    candidate = apply_kernel_patch(baseline.decode())
    folder = output / "sources"
    folder.mkdir(parents=True, exist_ok=True)
    paths = [folder / "baseline_attn_res.py", folder / "candidate_attn_res.py"]
    paths[0].write_bytes(baseline)
    paths[1].write_text(candidate)
    return paths, dict(
        installed_source=str(source_file), baseline_sha256=BASELINE_SHA256,
        candidate_sha256=CANDIDATE_SHA256, published_full_patch_sha256=PUBLISHED_PATCH_SHA256,
        embedded_kernel_patch_sha256=digest(KERNEL_PATCH.encode()),
        patch_scope="Exact kernel-file portion of the published two-file patch; model loader is not benchmarked",
    )


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def case_plan(rows, blocks, bf16_residual_only=False):
    cases = []
    def add(row, count, mode, quant, delta=True, norm=True, write=-1):
        cases.append(dict(id=f"r{row}-b{count}-{mode}-{'fp8' if quant else 'bf16'}",
                          rows=row, blocks=count, mode=mode, fp8=quant,
                          delta=delta, output_norm=norm, write_index=write))
    for row in rows:
        for count in blocks:
            if bf16_residual_only:
                add(row, count, "residual", False)
                continue
            for quant in (False, True):
                add(row, count, "residual", quant)
                if count < 8:
                    add(row, count, "block-write", quant, write=count)
        if bf16_residual_only:
            continue
        # Extra semantics use one nontrivial block count to limit cold JIT cost.
        add(row, blocks[0], "fresh-prefix", False, delta=False)
        add(row, blocks[-1], "final-mixture", False, norm=False)
    if not bf16_residual_only:
        # The actual first writer has no historical source and no delta.
        add(rows[0], 0, "initial-writer", False, delta=False, write=0)
        add(rows[0], 0, "initial-writer", True, delta=False, write=0)
    return cases


def compare(torch, actual, expected):
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        raise AssertionError(f"Tensor contract differs: {actual.shape}/{actual.dtype} vs {expected.shape}/{expected.dtype}")
    a, b = actual.float(), expected.float()
    finite = bool(torch.isfinite(a).all().item() and torch.isfinite(b).all().item())
    difference = (a - b).abs()
    return dict(
        exact=bool(torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))),
        finite=finite, different_elements=int((a != b).sum().item()),
        max_abs=float(difference.max().item()) if finite else None,
        rms_abs=float(difference.square().mean().sqrt().item()) if finite else None,
    )


def outputs(value):
    return value if isinstance(value, tuple) else (value,)


def summary(values):
    ordered = sorted(values)
    return dict(n=len(values), mean_us=statistics.mean(values),
                median_us=statistics.median(values), min_us=min(values), max_us=max(values),
                std_us=statistics.pstdev(values), p90_us=ordered[math.ceil(0.9 * len(values)) - 1],
                samples_us=values)


def run_case(torch, baseline, candidate, case, rounds, seed):
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    gen = torch.Generator(device="cuda").manual_seed(seed)
    row, hidden = case["rows"], 7168
    def random(shape):
        return torch.randn(shape, device="cuda", dtype=torch.bfloat16, generator=gen)
    original = dict(prefix=random((row, hidden)), blocks=random((row, 9, hidden)))
    if case["delta"]:
        original["delta"] = random((row, hidden)) * 0.125
    norm = (random((hidden,)).float() * 0.05 + 1).to(torch.bfloat16)
    qk = (random((1, hidden)).float() / math.sqrt(hidden)).to(torch.bfloat16)
    out_norm = ((random((hidden,)).float() * 0.05 + 1).to(torch.bfloat16)
                if case["output_norm"] else None)
    quant = torch.float8_e4m3fn if case["fp8"] else None
    def product():
        return (norm.float() * qk.squeeze(0).float()).contiguous()
    cached_product = product()
    def new_state():
        return {key: value.clone() for key, value in original.items()}
    def reset(state, data):
        for key, value in data.items():
            state[key].copy_(value)
    def call(module, state, weight_product=None):
        kwargs = dict(quant_dtype=quant)
        if module is candidate:
            kwargs["input_qk_weight"] = weight_product
        return module.attn_res(
            state["prefix"], state.get("delta"), state["blocks"], norm, qk, out_norm,
            case["blocks"], case["write_index"], RMS_EPS, RMS_EPS, **kwargs)

    eager_state = new_state()
    names = ("baseline", "cached_product", "recompute_product_each_call")
    states = {name: new_state() for name in names}
    def operation(name):
        if name == "baseline":
            return call(baseline, states[name])
        if case["blocks"] == 0:
            # The actual model hook skips product preparation for its first call.
            return call(candidate, states[name], None)
        return call(candidate, states[name], cached_product if name == "cached_product" else product())

    # Warm both real wrappers before capture; compilation is never timed as a kernel.
    warm_results = {}
    for name in names:
        for _ in range(2):
            reset(states[name], original)
            warm_results[name] = operation(name)
    torch.cuda.synchronize()
    eager_comparison = {}
    for name in names[1:]:
        eager_comparison[name] = [compare(torch, a, b) for a, b in zip(
            outputs(warm_results[name]), outputs(warm_results["baseline"]))]

    graphs, graph_outputs = {}, {}
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    for name in names:
        reset(states[name], original)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            graph_outputs[name] = operation(name)
        graphs[name] = graph
    prep_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(prep_graph, stream=stream):
        prepared_output = product()
    torch.cuda.synchronize()

    # Compare graph output to a fresh baseline call for each input version.
    validations = []
    for variant in ("original", "changed", "zero"):
        if variant == "original":
            data = original
        elif variant == "changed":
            data = {key: (value.float() * -0.75 + 0.125).to(value.dtype)
                    for key, value in original.items()}
        else:
            data = {key: torch.zeros_like(value) for key, value in original.items()}
        reset(eager_state, data)
        expected = call(baseline, eager_state)
        expected_prefix = (data["prefix"].float() + data["delta"].float()).to(torch.bfloat16) \
            if case["delta"] else data["prefix"]
        expected_blocks = data["blocks"].clone()
        if case["write_index"] >= 0:
            expected_blocks[:, case["write_index"], :].copy_(expected_prefix)
        for name in names:
            reset(states[name], data)
            graphs[name].replay()
        torch.cuda.synchronize()
        checks = dict(variant=variant, arms={})
        for name in names:
            checks["arms"][name] = dict(
                outputs=[compare(torch, a, b) for a, b in zip(
                    outputs(graph_outputs[name]), outputs(expected))],
                prefix=compare(torch, states[name]["prefix"], expected_prefix),
                blocks=compare(torch, states[name]["blocks"], expected_blocks),
                delta=None if not case["delta"] else compare(torch, states[name]["delta"], data["delta"]),
            )
        checks["eager_baseline_mutations"] = dict(
            prefix=compare(torch, eager_state["prefix"], expected_prefix),
            blocks=compare(torch, eager_state["blocks"], expected_blocks))
        validations.append(checks)
    def all_exact(value):
        if isinstance(value, dict):
            if "exact" in value:
                return value["exact"] and value["finite"]
            return all(all_exact(v) for v in value.values())
        if isinstance(value, list):
            return all(all_exact(v) for v in value)
        return True
    passed = all_exact(eager_comparison) and all_exact(validations)
    record = dict(case=case, status="passed" if passed else "correctness_failed",
                  eager_comparison=eager_comparison, replay_validation=validations,
                  quant_max=None if quant is None else float(baseline.get_fp8_min_max()[1]),
                  hidden_size=hidden, eps=RMS_EPS, product_bytes=cached_product.numel() * 4,
                  compile_and_validation_seconds=time.monotonic() - started)
    if not passed:
        record["timing_skipped"] = "Strict finite/bitwise output or mutation comparison failed"
        return record
    if case["blocks"] == 0:
        record["timing_skipped"] = "Initial writer has no score product; actual model hook skips this consumer"
        return record

    for name in names:
        reset(states[name], original)
    torch.cuda.synchronize()
    # Separate reset copies are outside the recorded intervals; each replay sees
    # the same prefix, and historical blocks remain immutable for this one call.
    events = {name: [] for name in names}
    orders = []
    for turn in range(rounds):
        order = names[turn % len(names):] + names[:turn % len(names)]
        orders.append(list(order))
        for name in order:
            states[name]["prefix"].copy_(original["prefix"])
            start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            graphs[name].replay()
            stop.record()
            events[name].append((start, stop))
    prep_events = []
    for _ in range(rounds):
        start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        prep_graph.replay()
        stop.record()
        prep_events.append((start, stop))
    torch.cuda.synchronize()
    timings = {name: summary([a.elapsed_time(b) * 1000 for a, b in pairs])
               for name, pairs in events.items()}
    timings["product_preparation_only"] = summary([a.elapsed_time(b) * 1000 for a, b in prep_events])
    saving = timings["baseline"]["mean_us"] - timings["cached_product"]["mean_us"]
    record.update(timings=timings, timing_orders=orders,
                  paired_cached_latency_reduction_pct=[
                      (b - c) / b * 100 for b, c in zip(timings["baseline"]["samples_us"],
                                                      timings["cached_product"]["samples_us"])],
                  mean_cached_latency_reduction_pct=saving / timings["baseline"]["mean_us"] * 100,
                  approximate_gpu_prep_amortization_calls=None if saving <= 0 else
                  timings["product_preparation_only"]["mean_us"] / saving,
                  prep_amortization_limit="GPU event arithmetic only; excludes Python/model loading and requires immutable weights",
                  peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                  elapsed_seconds=time.monotonic() - started)
    # Keep captured allocation references alive through all validation/timing.
    assert prepared_output.shape == (hidden,)
    return record


def choices_csv(text, allowed):
    values = [int(x) for x in text.split(",")]
    if not values or len(values) != len(set(values)) or any(x not in allowed for x in values):
        raise argparse.ArgumentTypeError(f"Choose unique comma-separated values from {sorted(allowed)}")
    return values


def rows_csv(text):
    values = [int(x) for x in text.split(",")]
    if not values or len(values) != len(set(values)) or any(not 1 <= x <= 10240 for x in values):
        raise argparse.ArgumentTypeError("Choose unique comma-separated row counts from 1 through 10240")
    return values


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", type=rows_csv, default=[280, 336])
    parser.add_argument("--blocks", type=lambda s: choices_csv(s, {1, 4, 8}), default=[4, 8])
    parser.add_argument("--bf16-residual-only", action="store_true",
                        help="Only BF16 residual + output norm, no block writes, with blocks4/8")
    parser.add_argument("--rounds", type=int, default=18)
    parser.add_argument("--max-seconds", type=int, default=480, help="Soft bound checked between cases; cold compilation can overrun one case")
    parser.add_argument("--source-file", type=Path, help="Equivalent frozen source for CPU source validation")
    parser.add_argument("--source-only", action="store_true", help="Check exact patch/source hashes without importing Torch or using a GPU")
    args = parser.parse_args(argv)
    if not 3 <= args.rounds <= 60 or not 30 <= args.max_seconds <= 540:
        parser.error("rounds must be 3..60 and max-seconds 30..540")
    if args.bf16_residual_only and any(count not in (4, 8) for count in args.blocks):
        parser.error("--bf16-residual-only supports only --blocks 4,8")
    started = time.monotonic()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    sys.dont_write_bytecode = True
    # Preserve the transport's local tmpfs/compiler-cache configuration. In
    # particular, never redirect COMGR/Triton caches into NFS artifact storage.
    plan = case_plan(args.rows, args.blocks, args.bf16_residual_only)
    result = dict(probe="attnres_product_hoist", status="running", synthetic_controlled_tensors=True,
                  seed=20261112, case_plan=plan, cases=[], rounds=args.rounds,
                  bf16_residual_only=args.bf16_residual_only,
                  inherited_cache_paths={key: os.environ.get(key) for key in
                                         ("TMPDIR", "TRITON_CACHE_DIR", "XDG_CACHE_HOME")},
                  limits=["Isolated operator experiment; no captured model tensors or per-request E2E result",
                          "Strict output/scale/mutation equality; does not test real model post-load/offload lifecycle",
                          "Fixed-buffer warm graph replay, no profiler or cache flush; input reset excluded",
                          "Complete-operation arm rebuilds product inside every replay; cached arm assumes immutable weights",
                          "Norm-cache optimization not implemented; final agentic validation remains separate"])
    write_json(args.output / "result.json", result)
    def progress(event, **fields):
        message = dict(event=event, elapsed_seconds=time.monotonic() - started, **fields)
        with (args.output / "progress.jsonl").open("a") as f:
            f.write(json.dumps(message, allow_nan=False) + "\n")
        print(json.dumps(message, allow_nan=False), flush=True)
    try:
        paths, provenance = prepare_sources(args.output, args.source_file)
        result["provenance"] = provenance
        if args.source_only:
            result["status"] = "source_check_passed"
            result["gpu_executed"] = False
            write_json(args.output / "result.json", result)
            progress("source_check_passed", planned_cases=len(plan))
            return 0
        import torch
        if not torch.cuda.is_available() or not torch.version.hip:
            raise RuntimeError("A ROCm GPU is required")
        torch.cuda.set_device(0)
        prop = torch.cuda.get_device_properties(0)
        result["environment"] = dict(torch=torch.__version__, hip=torch.version.hip,
                                      device=prop.name, architecture=getattr(prop, "gcnArchName", None),
                                      total_memory=prop.total_memory, visible_devices=torch.cuda.device_count())
        baseline, candidate = [load_module(path, f"attnres_probe_{name}")
                               for path, name in zip(paths, ("baseline", "candidate"))]
        result["gpu_executed"] = True
        with torch.inference_mode():
            for index, case in enumerate(plan):
                if time.monotonic() - started >= args.max_seconds:
                    result["stopped_reason"] = "Soft time budget reached; remaining cases not attempted"
                    break
                progress("case_started", index=index, case=case)
                result["active_case"] = case["id"]
                write_json(args.output / "result.json", result)
                try:
                    record = run_case(torch, baseline, candidate, case, args.rounds, result["seed"] + index)
                except Exception as exc:
                    record = dict(case=case, status="error", error=str(exc), traceback=traceback.format_exc())
                    try:
                        torch.cuda.synchronize()
                    except Exception as gpu_exc:
                        result["stopped_reason"] = f"GPU context error: {gpu_exc}"
                result["cases"].append(record)
                result.pop("active_case", None)
                write_json(args.output / "result.json", result)
                progress("case_finished", index=index, case_id=case["id"], status=record["status"],
                         mean_cached_latency_reduction_pct=record.get("mean_cached_latency_reduction_pct"))
                if "stopped_reason" in result:
                    break
        result["status"] = "passed" if len(result["cases"]) == len(plan) and all(
            c["status"] == "passed" for c in result["cases"]) else "incomplete_or_failed"
    except Exception as exc:
        result.update(status="error", error=str(exc), traceback=traceback.format_exc())
    result["elapsed_seconds"] = time.monotonic() - started
    write_json(args.output / "result.json", result)
    progress("finished", status=result["status"], completed_cases=len(result["cases"]), planned_cases=len(plan))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

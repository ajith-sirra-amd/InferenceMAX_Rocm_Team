#!/usr/bin/env python3
"""Bounded producer/consumer AttnRes norm-cache experiment on synthetic tensors.

Eight real block writers precede repeated consumers at blocks4/8. Compare
baseline, product-only, norm-cache and combined arms; no model integration.
The sibling optimization_attnres.py provides the hash-verified product patch.
"""
from __future__ import annotations

import argparse
import ast
import gc
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

sys.dont_write_bytecode = True

import optimization_attnres as common

NORM_SHA256 = "c4bb07264a72ada42331c8535928ea98c3246d90db2fa4fe22644ddf5c14aac8"
ARMS = ("baseline", "product_only", "norm_cache", "combined")
ENVIRONMENT_KEYS = (
    'VLLM_ROCM_USE_AITER',
    'VLLM_ROCM_USE_AITER_MOE_SITUV2',
    'SAFETENSORS_FAST_GPU',
    'VLLM_USE_V2_MODEL_RUNNER',
    'VLLM_FASTSAFETENSORS_QUEUE_SIZE',
    'VLLM_USE_BREAKABLE_CUDAGRAPH',
    'VLLM_ROCM_AITER_MLA_ASM_PADDING',
    'VLLM_ROCM_AITER_MLA_DCP_VERIFY',
    'VLLM_ROCM_DCP_AITER_ALLGATHER',
    'VLLM_ROCM_USE_AITER_FLYDSL_SPARSE_MLA',
    'AITER_FLYDSL_SPARSE_MLA',
    'VLLM_ROCM_DSA_DCP_GLOBAL_KEY_PREFILL',
    'VLLM_ROCM_DSA_DCP_ROWSHARD_PREFILL',
    'VLLM_USE_DIRECT_DCP_A2A',
    'VLLM_USE_DIRECT_DCP_Q_GATHER',
    'VLLM_USE_DIRECT_DCP_KV_GATHER',
    'VLLM_ROCM_QUICK_REDUCE_QUANTIZATION',
    'AITER_FLYDSL_STAGE2_FP8',
    'AITER_BF16_FP8_MOE_BOUND',
    'HSA_NO_SCRATCH_RECLAIM',
    'THP_MEM_ALLOC_ENABLE',
    'VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS',
    'PYTHONNOUSERSITE',
    'VLLM_ENGINE_READY_TIMEOUT_S',
    'PYTHONHASHSEED',
    'TMPDIR',
    'TRITON_CACHE_DIR',
    'XDG_CACHE_HOME',
)


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError("Frozen source context is not unique: " + old[:80])
    return source.replace(old, new, 1)


def norm_source(product_source):
    """Keep two-dimensional source reductions; cache only immutable BF16 blocks."""
    source = replace_once(product_source, "    output_scale_ptr,\n", "    output_scale_ptr,\n    block_rstd_ptr,\n    stride_rstd_m: tl.constexpr,\n    stride_rstd_b: tl.constexpr,\n")
    source = replace_once(source, "    PRECOMPUTED_INPUT_QK: tl.constexpr,\n", "    PRECOMPUTED_INPUT_QK: tl.constexpr,\n    CACHE_BLOCK_RSTD: tl.constexpr,\n")
    source = replace_once(source, "    if num_blocks == 0:\n        mixed = updated_prefix\n", """    if num_blocks == 0:
        mixed = updated_prefix
        if CACHE_BLOCK_RSTD and WRITE_BLOCK:
            # Conservative initial writer: match the 2-D source expression,
            # rather than borrowing the separate 1-D output normalization.
            first_values = updated_prefix[None, :]
            first_rstd = tl.rsqrt(
                tl.sum(first_values * first_values, axis=1)
                * (1.0 / hidden_size) + eps
            )
            tl.store(
                block_rstd_ptr + row_idx * stride_rstd_m
                + block_write_idx * stride_rstd_b + tl.arange(0, 1) * 0,
                first_rstd,
            )
""")
    source = replace_once(source, """            reciprocal_std = tl.rsqrt(
                tl.sum(values * values, axis=1) * (1.0 / hidden_size) + eps
            )
""", """            if CACHE_BLOCK_RSTD:
                tl.static_assert(BLOCK_L == 1)
                if source_tile == num_blocks:
                    # Retain the live-prefix source reduction and its BF16
                    # rounding; writers publish this scalar for later calls.
                    reciprocal_std = tl.rsqrt(
                        tl.sum(values * values, axis=1) * (1.0 / hidden_size) + eps
                    )
                    if WRITE_BLOCK:
                        tl.store(
                            block_rstd_ptr + row_idx * stride_rstd_m
                            + block_write_idx * stride_rstd_b
                            + tl.arange(0, BLOCK_L) * 0,
                            reciprocal_std,
                        )
                else:
                    reciprocal_std = tl.load(
                        block_rstd_ptr + row_idx * stride_rstd_m
                        + source_offsets * stride_rstd_b,
                    )
            else:
                reciprocal_std = tl.rsqrt(
                    tl.sum(values * values, axis=1) * (1.0 / hidden_size) + eps
                )
""")
    source = replace_once(source, "    input_qk_weight: torch.Tensor | None = None,\n", "    input_qk_weight: torch.Tensor | None = None,\n    block_rstd_cache: torch.Tensor | None = None,\n")
    source = replace_once(source, "    num_tokens, hidden_size = prefix.shape\n", """    num_tokens, hidden_size = prefix.shape
    if block_rstd_cache is not None:
        assert num_tokens >= 256 and hidden_size == 7168
        assert prefix.dtype == blocks.dtype == torch.bfloat16
        assert delta is None or delta.dtype == torch.bfloat16
        assert block_rstd_cache.shape == blocks.shape[:2] == (num_tokens, 8)
        assert block_rstd_cache.dtype == torch.float32
        assert block_rstd_cache.device == prefix.device == blocks.device
        assert block_rstd_cache.is_contiguous()
        assert eps == 1e-5 and 0 <= num_blocks <= 8
        assert block_write_idx == -1 or block_write_idx == num_blocks < 8
""")
    source = replace_once(source, "        output,\n        scale,\n", "        output,\n        scale,\n        block_rstd_cache,\n        0 if block_rstd_cache is None else block_rstd_cache.stride(0),\n        0 if block_rstd_cache is None else block_rstd_cache.stride(1),\n")
    source = replace_once(source, "        PRECOMPUTED_INPUT_QK=input_qk_weight is not None,\n", "        PRECOMPUTED_INPUT_QK=input_qk_weight is not None,\n        CACHE_BLOCK_RSTD=block_rstd_cache is not None,\n")
    ast.parse(source)
    return source


def sequence_plan(consumers):
    """Reduced synthetic sequence, preserving writer-before-reader ownership."""
    plan = []
    for block in range(8):
        plan.append(dict(kind="writer", blocks=block, write=block, reset=True,
                         delta=block > 0, output_norm=True))
        count = block + 1
        if count in (4, 8):
            plan.append(dict(kind="fresh_prefix", blocks=count, write=-1,
                             reset=True, delta=False, output_norm=True))
            for _ in range(consumers):
                plan.append(dict(kind="consumer", blocks=count, write=-1,
                                 reset=False, delta=True, output_norm=True))
    plan.append(dict(kind="final_mixture", blocks=8, write=-1,
                     reset=False, delta=True, output_norm=False))
    for index, step in enumerate(plan):
        step["index"] = index
    return plan


def validate_plan(plan):
    written = set()
    historical_reads = 0
    for step in plan:
        assert set(range(step["blocks"])) <= written
        historical_reads += step["blocks"]
        if step["write"] >= 0:
            assert step["write"] not in written
            written.add(step["write"])
    assert written == set(range(8))
    return dict(calls=len(plan), writes=len(written), historical_block_uses=historical_reads,
                norm_cache_bytes_per_row=32,
                limits="Reduced producer/consumer sequence; not the complete 187-call model forward")


def prepare_sources(output, source_file):
    paths, provenance = common.prepare_sources(output, source_file)
    candidate = norm_source(paths[1].read_text())
    candidate_hash = common.digest(candidate.encode())
    if candidate_hash != NORM_SHA256:
        raise ValueError("Norm-cache candidate source hash mismatch: " + candidate_hash)
    path = output / "sources" / "norm_cached_attn_res.py"
    path.write_text(candidate)
    provenance.update(norm_candidate_sha256=candidate_hash,
                      probe_sha256=common.digest(Path(__file__).read_bytes()),
                      helper_sha256=common.digest(Path(common.__file__).read_bytes()))
    return paths + [path], provenance


def require_exact(checks):
    if isinstance(checks, dict):
        if "exact" in checks:
            return checks["exact"] and checks["finite"]
        return all(require_exact(v) for v in checks.values())
    if isinstance(checks, list):
        return all(require_exact(v) for v in checks)
    return True


def run_case(torch, modules, case, plan, rounds, seed, progress, check_budget):
    started = time.monotonic()
    rows, hidden = case["rows"], 7168
    torch.cuda.reset_peak_memory_stats()
    generator = torch.Generator(device="cuda").manual_seed(seed)
    def random(shape):
        return torch.randn(shape, device="cuda", dtype=torch.bfloat16, generator=generator)
    # Reused external signals stand in for intervening attention/MLP work.
    # They are immutable within one sequence; every arm sees identical inputs.
    signals = random((16, rows, hidden))
    originals = signals.clone()
    weights = random((len(plan), 3, hidden))
    norms = (weights[:, 0].float() * 0.05 + 1).to(torch.bfloat16)
    qks = (weights[:, 1].float() / math.sqrt(hidden)).to(torch.bfloat16)
    out_norms = (weights[:, 2].float() * 0.05 + 1).to(torch.bfloat16)
    products = (norms.float() * qks.float()).contiguous()
    states = {name: dict(prefix=torch.empty((rows, hidden), device="cuda", dtype=torch.bfloat16),
                        blocks=torch.full((rows, 8, hidden), float("nan"), device="cuda", dtype=torch.bfloat16),
                        cache=torch.full((rows, 8), float("nan"), device="cuda", dtype=torch.float32))
              for name in ARMS}
    captured_outputs = {}; graphs = {}
    quant_dtype = torch.float8_e4m3fn if case["dtype"] == "fp8" else None
    def operation(name):
        state = states[name]
        module = modules[0] if name == "baseline" else modules[1] if name == "product_only" else modules[2]
        values = []
        for step in plan:
            check_budget()
            index = step["index"]
            if step["reset"]:
                state["prefix"].copy_(signals[index % len(signals)])
            delta = signals[(index + 7) % len(signals)] if step["delta"] else None
            kwargs = dict(quant_dtype=quant_dtype if step["output_norm"] else None)
            if name != "baseline":
                kwargs["input_qk_weight"] = products[index] if name in ("product_only", "combined") and step["blocks"] else None
            if name in ("norm_cache", "combined"):
                kwargs["block_rstd_cache"] = state["cache"]
            values.append(module.attn_res(
                state["prefix"], delta, state["blocks"], norms[index], qks[index],
                out_norms[index] if step["output_norm"] else None,
                step["blocks"], step["write"], common.RMS_EPS,
                common.RMS_EPS if step["output_norm"] else 0.0, **kwargs))
        return values

    eager = {}
    for name in ARMS:
        check_budget()
        progress("arm_warmup", case=case, arm=name)
        eager[name] = operation(name)
        torch.cuda.synchronize()
    eager_checks = {name: [common.compare(torch, a, b)
                          for actual, expected in zip(eager[name], eager["baseline"])
                          for a, b in zip(common.outputs(actual), common.outputs(expected))]
                    for name in ARMS[1:]}
    eager_checks["mutations"] = {name: {
        "prefix": common.compare(torch, states[name]["prefix"], states["baseline"]["prefix"]),
        "blocks": common.compare(torch, states[name]["blocks"], states["baseline"]["blocks"])}
        for name in ARMS[1:]}
    # Fail before capture/timing when the conservative reduction is not exact.
    if not require_exact(eager_checks):
        return dict(case=case, status="correctness_failed", eager=eager_checks,
                    timing_skipped="Strict eager output/scale/mutation equality failed")
    del eager
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    for name in ARMS:
        check_budget()
        progress("arm_capture", case=case, arm=name)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            captured_outputs[name] = operation(name)
        graphs[name] = graph
        torch.cuda.synchronize()
    pointers = {name: {key: value.data_ptr() for key, value in state.items()}
                for name, state in states.items()}
    replay_checks = []
    all_passed = True
    for variant in ("original", "changed", "zero", "row_permuted", "original_again"):
        check_budget()
        if variant in ("original", "original_again"):
            signals.copy_(originals)
        elif variant == "changed":
            signals.copy_((originals.float() * -0.75 + 0.125).to(torch.bfloat16))
        elif variant == "zero":
            signals.zero_()
        else:
            signals.copy_(originals.roll(1, dims=1))
        if variant == "original":
            for state in states.values():
                state["cache"].fill_(float("nan"))
                state["blocks"].fill_(float("nan"))
        # Subsequent variants deliberately retain previous scratch contents;
        # writers must overwrite all required entries on each graph replay.
        expected = operation("baseline")
        expected_prefix = states["baseline"]["prefix"].clone()
        expected_blocks = states["baseline"]["blocks"].clone()
        expected_signals = signals.clone()
        checks = {}
        for name in ARMS:
            graphs[name].replay()
            torch.cuda.synchronize()
            checks[name] = dict(
                outputs=[common.compare(torch, a, b)
                         for actual, reference in zip(captured_outputs[name], expected)
                         for a, b in zip(common.outputs(actual), common.outputs(reference))],
                prefix=common.compare(torch, states[name]["prefix"], expected_prefix),
                blocks=common.compare(torch, states[name]["blocks"], expected_blocks),
                inputs=common.compare(torch, signals, expected_signals),
                stable_pointers=pointers[name] == {key: value.data_ptr() for key, value in states[name].items()})
            all_passed &= checks[name]["stable_pointers"] and require_exact(checks[name])
            if name in ("norm_cache", "combined"):
                reference = torch.rsqrt(states[name]["blocks"].double().square().mean(dim=-1) + common.RMS_EPS)
                cache = states[name]["cache"].double()
                relative = ((cache - reference) / reference).abs()
                finite = bool(torch.isfinite(cache).all().item())
                max_relative = float(relative.max().item()) if finite else None
                checks[name]["stored_bf16_norm_check"] = dict(
                    finite=finite, max_relative_error_vs_fp64=max_relative,
                    passed=finite and max_relative < 1e-5,
                    limit="Numerical sanity check against stored BF16 blocks; output comparisons remain bitwise exact")
                all_passed &= checks[name]["stored_bf16_norm_check"]["passed"]
        replay_checks.append(dict(variant=variant, arms=checks))
        progress("replay_checked", case=case, variant=variant, passed=bool(all_passed))
    record = dict(case=case, status="passed" if all_passed else "correctness_failed",
                  eager=eager_checks, replay_validation=replay_checks,
                  cache_bytes_per_arm=rows * 8 * 4,
                  product_bytes=products.numel() * products.element_size(),
                  compile_and_validation_seconds=time.monotonic() - started,
                  quant_max=None if quant_dtype is None else float(modules[0].get_fp8_min_max()[1]))
    if not all_passed:
        record["timing_skipped"] = "Strict graph outputs/mutations or scratch-lifetime check failed"
        return record
    signals.copy_(originals)
    torch.cuda.synchronize()
    events = {name: [] for name in ARMS}; orders = []
    for turn in range(rounds):
        check_budget()
        order = ARMS[turn % len(ARMS):] + ARMS[:turn % len(ARMS)]
        orders.append(list(order))
        for name in order:
            start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record(); graphs[name].replay(); stop.record()
            events[name].append((start, stop))
    torch.cuda.synchronize()
    timings = {name: common.summary([a.elapsed_time(b) * 1000 for a, b in pairs]) for name, pairs in events.items()}
    record.update(timings=timings, timing_orders=orders,
                  mean_sequence_reduction_pct={name: 100 * (1 - timings[name]["mean_us"] / timings["baseline"]["mean_us"]) for name in ARMS[1:]},
                  paired_sequence_reduction_pct={name: [100 * (1 - c / b) for b, c in zip(timings["baseline"]["samples_us"], timings[name]["samples_us"])] for name in ARMS[1:]},
                  peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                  elapsed_seconds=time.monotonic() - started)
    return record


def rows_csv(value):
    rows = [int(item) for item in value.split(",")]
    if not rows or len(set(rows)) != len(rows) or any(row not in (256, 280, 336, 1024) for row in rows):
        raise argparse.ArgumentTypeError("Choose unique rows from 256,280,336,1024; smaller layouts are excluded")
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source-file", type=Path)
    parser.add_argument("--source-only", action="store_true")
    parser.add_argument("--rows", type=rows_csv, default=[280, 336])
    parser.add_argument("--bf16-only", action="store_true", help="Skip FP8 qualification explicitly")
    parser.add_argument("--consumers", type=int, default=12, help="Repeated consumers at each of blocks4 and8")
    parser.add_argument("--rounds", type=int, default=12)
    parser.add_argument("--max-seconds", type=int, default=480, help="Soft bound; one in-flight cold JIT can overrun")
    args = parser.parse_args(argv)
    if not 2 <= args.consumers <= 24 or not 4 <= args.rounds <= 32 or not 30 <= args.max_seconds <= 540:
        parser.error("consumers:2..24, rounds:4..32, max-seconds:30..540")
    args.output = args.output.resolve()
    if args.output.exists():
        parser.error("Output exists; use a fresh datetime/run directory")
    args.output.mkdir(parents=True)
    sys.dont_write_bytecode = True
    started = time.monotonic()
    plan = sequence_plan(args.consumers)
    cases = [dict(rows=row, dtype=dtype) for row in args.rows for dtype in (("bf16",) if args.bf16_only else ("bf16", "fp8"))]
    result = dict(probe="attnres_historical_norm_cache", status="running", cases=[], case_plan=cases,
                  sequence=plan, sequence_summary=validate_plan(plan), rounds=args.rounds,
                  synthetic_inputs_and_weights=True, gpu_executed=False,
                  limits=["Reduced operator sequence, not model integration or per-request E2E",
                          "BF16 block bank, common epsilon, one-source/four-warp rows>=256 only",
                          "Per-arm scratch is retained for captured-graph lifetime; eight writers refresh every replay",
                          "No profiler capture requested; pinned runtime may still preload profiling libraries",
                          "Timing includes writers and prefix reset copies; immutable product preparation excluded",
                          "Full-model piecewise graphs, PP, specialization, concurrent forwards and quality unvalidated"])
    def save(): common.write_json(args.output / "result.json", result)
    def progress(event, **fields):
        value = dict(event=event, elapsed_seconds=time.monotonic() - started, **fields)
        with (args.output / "progress.jsonl").open("a") as handle: handle.write(json.dumps(value, allow_nan=False) + "\n")
        print(json.dumps(value, allow_nan=False), flush=True)
    def check_budget():
        if time.monotonic() - started >= args.max_seconds:
            raise TimeoutError("Soft time budget reached; retain partial evidence")
    save()
    try:
        paths, result["provenance"] = prepare_sources(args.output, args.source_file)
        if args.source_only:
            result["status"] = "source_check_passed"; save(); return 0
        import torch
        if not torch.cuda.is_available() or not torch.version.hip:
            raise RuntimeError("A ROCm GPU is required")
        torch.cuda.set_device(0)
        prop = torch.cuda.get_device_properties(0)
        result["environment"] = dict(torch=torch.__version__, hip=torch.version.hip,
            device=prop.name, architecture=getattr(prop, "gcnArchName", None), visible_devices=torch.cuda.device_count(),
            inherited_environment={key: os.environ.get(key) for key in ENVIRONMENT_KEYS})
        modules = [common.load_module(path, "attnres_norm_probe_" + name) for path, name in zip(paths, ("baseline", "product", "norm"))]
        result["gpu_executed"] = True; save()
        with torch.inference_mode():
            for index, case in enumerate(cases):
                check_budget(); progress("case_started", case=case)
                result["active_case"] = case; save()
                try:
                    record = run_case(torch, modules, case, plan, args.rounds, 20261112 + index, progress, check_budget)
                except Exception as exc:
                    record = dict(case=case, status="error", error=str(exc), traceback=traceback.format_exc())
                result["cases"].append(record); save()
                progress("case_finished", case=case, status=record["status"])
                gc.collect(); torch.cuda.empty_cache()
                if record["status"] != "passed": break
        result["status"] = "passed" if len(result["cases"]) == len(cases) and all(case["status"] == "passed" for case in result["cases"]) else "incomplete_or_failed"
    except Exception as exc:
        result.update(status="error", error=str(exc), traceback=traceback.format_exc())
    result["elapsed_seconds"] = time.monotonic() - started
    save(); progress("finished", status=result["status"], completed_cases=len(result["cases"]))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

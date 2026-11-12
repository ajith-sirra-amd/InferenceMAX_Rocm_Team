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

NORM_V1_SHA256 = "c4bb07264a72ada42331c8535928ea98c3246d90db2fa4fe22644ddf5c14aac8"
NORM_SHA256 = "40719ea01e4b6a3365086c364b8c9dea61dadf741497111283842aa33b86b8f1"
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


def norm_source_v1(product_source):
    """Historical failed candidate; retained byte-for-byte for source provenance."""
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


def norm_source(product_source):
    """Use the native initial masked reduction; later writer reductions stay intact.

    Diagnostic43193326018 binds the old initial2-D reduction to a different
    Triton layout ([1,1] instead of consumer[1,8]) and45 small rstd errors.
    The native initial1-D output reduction matches the consumer for all280 rows.
    This revision remains subject to unchanged exact GPU parity checks.
    """
    source = norm_source_v1(product_source)
    if common.digest(source.encode()) != NORM_V1_SHA256:
        raise ValueError("Historical norm-cache source hash mismatch")
    source = replace_once(source, """            # Conservative initial writer: match the 2-D source expression,
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
""", """            # Preserve the native initial-writer masked reduction layout.
            # A separate 2-D reduction changes its FP32 summation order.
            first_rstd = tl.rsqrt(
                tl.sum(tl.where(d_mask, updated_prefix * updated_prefix, 0.0), axis=0)
                * (1.0 / hidden_size) + eps
            )
            tl.store(
                block_rstd_ptr + row_idx * stride_rstd_m
                + block_write_idx * stride_rstd_b,
                first_rstd,
            )
""")
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
    historical = norm_source_v1(paths[1].read_text())
    if common.digest(historical.encode()) != NORM_V1_SHA256:
        raise ValueError("Historical norm-cache candidate source hash mismatch")
    (output / "sources" / "norm_cached_attn_res_v1.py").write_text(historical)
    candidate = norm_source(paths[1].read_text())
    candidate_hash = common.digest(candidate.encode())
    if candidate_hash != NORM_SHA256:
        raise ValueError("Norm-cache candidate source hash mismatch: " + candidate_hash)
    path = output / "sources" / "norm_cached_attn_res.py"
    path.write_text(candidate)
    provenance.update(norm_candidate_sha256=candidate_hash,
                      norm_revision="v2-masked-initial-reduction",
                      historical_norm_candidate_sha256=NORM_V1_SHA256,
                      diagnostic_source_run="43193326018",
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


def diagnostic_source(source):
    """Observe source sums/rstd/dot/logits; verify observer output parity on GPU."""
    source = replace_once(source, "    output_scale_ptr,\n", "    output_scale_ptr,\n    source_stats_ptr,\n")
    source = replace_once(source, "    HAS_DELTA: tl.constexpr,\n", "    HAS_DELTA: tl.constexpr,\n    SAVE_SOURCE_STATS: tl.constexpr,\n")
    source = replace_once(source, "            logits = tl.sum(values * input_qk_weight[None, :], axis=1) * reciprocal_std\n", """            logits = tl.sum(values * input_qk_weight[None, :], axis=1) * reciprocal_std
            if SAVE_SOURCE_STATS:
                stats = source_stats_ptr + row_idx * 40 + source_offsets * 4
                tl.store(stats + 0, tl.sum(values * values, axis=1), mask=source_mask)
                tl.store(stats + 1, reciprocal_std, mask=source_mask)
                tl.store(stats + 2, tl.sum(values * input_qk_weight[None, :], axis=1), mask=source_mask)
                tl.store(stats + 3, logits, mask=source_mask)
""")
    source = replace_once(source, "        output_norm_weight = tl.load(\n", """        if SAVE_SOURCE_STATS:
            tl.store(source_stats_ptr + row_idx * 40 + 36,
                     tl.sum(tl.where(d_mask, mixed * mixed, 0.0), axis=0))
            tl.store(source_stats_ptr + row_idx * 40 + 37, output_reciprocal_std)
        output_norm_weight = tl.load(
""")
    source = replace_once(source, "    quant_dtype: torch.dtype | None = None,\n", "    quant_dtype: torch.dtype | None = None,\n    source_stats: torch.Tensor | None = None,\n")
    source = replace_once(source, "    num_tokens, hidden_size = prefix.shape\n", """    num_tokens, hidden_size = prefix.shape
    if source_stats is not None:
        assert source_stats.shape == (num_tokens, 10, 4)
        assert source_stats.dtype == torch.float32 and source_stats.is_contiguous()
        assert source_stats.device == prefix.device
""")
    source = replace_once(source, "        output,\n        scale,\n", "        output,\n        scale,\n        source_stats,\n")
    source = replace_once(source, "        HAS_DELTA=delta is not None,\n", "        HAS_DELTA=delta is not None,\n        SAVE_SOURCE_STATS=source_stats is not None,\n")
    ast.parse(source)
    return source


def dump_kernel_artifacts(compiled, directory):
    """Save generated arithmetic and register/spill metadata, excluding binaries."""
    directory.mkdir(parents=True, exist_ok=True)
    result = dict(files={}, resources={})
    for name, value in getattr(compiled, "asm", {}).items():
        if isinstance(value, str) and name in ("ttir", "ttgir", "llir", "amdgcn", "ptx"):
            path = directory / ("kernel." + name)
            path.write_text(value)
            result["files"][name] = dict(path=str(path), sha256=common.digest(value.encode()), bytes=len(value.encode()))
    for name in ("n_regs", "n_spills"):
        value = getattr(compiled, name, None)
        if isinstance(value, (int, float, str, bool)) or value is None:
            result["resources"][name] = value
    metadata = getattr(compiled, "metadata", None)
    for name in ("num_warps", "num_stages", "shared", "warp_size", "enable_fp_fusion"):
        value = getattr(metadata, name, None)
        if isinstance(value, (int, float, str, bool)) or value is None:
            result["resources"][name] = value
    return result


def diagnose_first_consumer(torch, paths, output, plan, progress, check_budget,
                            *, seed=20261112, dtype="bf16"):
    """Observe the first divergent historical use without qualifying a candidate."""
    rows, hidden = 280, 7168
    if dtype not in ("bf16", "fp8"):
        raise ValueError("Diagnostic output dtype must be bf16 or fp8")
    generator = torch.Generator(device="cuda").manual_seed(seed)
    def random(shape):
        return torch.randn(shape, device="cuda", dtype=torch.bfloat16, generator=generator)
    signals = random((16, rows, hidden))
    # Keep the complete plan's RNG allocation even though only two steps run.
    weights = random((len(plan), 3, hidden))
    norms = (weights[:, 0].float() * 0.05 + 1).to(torch.bfloat16)
    qks = (weights[:, 1].float() / math.sqrt(hidden)).to(torch.bfloat16)
    out_norms = (weights[:, 2].float() * 0.05 + 1).to(torch.bfloat16)
    products = (norms.float() * qks.float()).contiguous()
    input_hashes = {name: common.digest(value.view(torch.uint8).cpu().numpy().tobytes())
                    for name, value in (("signals", signals), ("weights", weights))}
    quant_dtype = torch.float8_e4m3fn if dtype == "fp8" else None
    (output / "sources").mkdir(parents=True, exist_ok=True)
    diagnostic_paths = []
    for path, name in ((paths[0], "baseline_debug"), (paths[2], "norm_debug")):
        source = diagnostic_source(path.read_text())
        destination = output / "sources" / (name + ".py")
        destination.write_text(source)
        diagnostic_paths.append(destination)
    variants = (
        ("baseline", paths[0], False, False, False, False),
        ("product_only", paths[1], True, False, False, False),
        ("norm_cache", paths[2], False, True, False, False),
        ("norm_disabled", paths[2], False, False, False, False),
        ("baseline_debug", diagnostic_paths[0], False, False, True, False),
        ("norm_debug", diagnostic_paths[1], False, True, True, False),
        ("baseline_no_fma", paths[0], False, False, False, True),
        ("norm_no_fma", paths[2], False, True, False, True),
    )
    captures = {}; summaries = {}
    for name, path, product, cache, debug, no_fma in variants:
        check_budget(); progress("diagnostic_arm", arm=name, dtype=dtype, seed=seed)
        module = common.load_module(path, "attnres_norm_diagnostic_" + dtype + "_" + name)
        original_kernel = module._attn_res_kernel
        class Recorder:
            def __getitem__(self, grid):
                def launch(*args, **kwargs):
                    if no_fma:
                        kwargs["enable_fp_fusion"] = False
                    compiled = original_kernel[grid](*args, **kwargs)
                    self.compiled = compiled
                    return compiled
                return launch
        recorder = Recorder()
        module._attn_res_kernel = recorder
        state = dict(prefix=torch.empty((rows, hidden), device="cuda", dtype=torch.bfloat16),
                     blocks=torch.full((rows, 8, hidden), float("nan"), device="cuda", dtype=torch.bfloat16),
                     cache=torch.full((rows, 8), float("nan"), device="cuda", dtype=torch.float32))
        records = []; compiled_records = []
        try:
            for step in plan[:2]:
                check_budget(); index = step["index"]
                state["prefix"].copy_(signals[index % len(signals)])
                stats = torch.full((rows, 10, 4), float("nan"), device="cuda", dtype=torch.float32) if debug else None
                kwargs = dict(quant_dtype=quant_dtype)
                if path != paths[0] and name not in ("baseline_debug", "baseline_no_fma"):
                    kwargs["input_qk_weight"] = products[index] if product and step["blocks"] else None
                if path == paths[2] or name == "norm_debug":
                    kwargs["block_rstd_cache"] = state["cache"] if cache else None
                if debug:
                    kwargs["source_stats"] = stats
                value = module.attn_res(state["prefix"], signals[(index + 7) % len(signals)] if step["delta"] else None,
                    state["blocks"], norms[index], qks[index], out_norms[index], step["blocks"], step["write"],
                    common.RMS_EPS, common.RMS_EPS, **kwargs)
                torch.cuda.synchronize()
                records.append(dict(output=value, prefix=state["prefix"].clone(),
                    blocks=state["blocks"][:, :index + 1].clone(),
                    cache=state["cache"][:, :index + 1].clone() if cache else None,
                    stats=stats))
                compiled_records.append(dump_kernel_artifacts(recorder.compiled, output / "compiler" / name / ("step-" + str(index))))
            captures[name] = records
            summaries[name] = dict(status="collected", source_sha256=common.digest(path.read_bytes()),
                                   no_fma=no_fma, compiler=compiled_records)
        except Exception as exc:
            summaries[name] = dict(status="error", error=str(exc), traceback=traceback.format_exc())
        common.write_json(output / "diagnostic-progress.json", summaries)
    if "baseline" not in captures or "norm_cache" not in captures:
        raise RuntimeError("Could not collect the original baseline and failed norm candidate")
    def compare_output(actual, reference):
        actual, reference = common.outputs(actual), common.outputs(reference)
        if len(actual) != len(reference) or len(actual) != (2 if quant_dtype is not None else 1):
            raise AssertionError("Diagnostic output/scale contract changed")
        checks = [common.compare(torch, a, b) for a, b in zip(actual, reference)]
        return checks[0] if quant_dtype is None else dict(data=checks[0], scale=checks[1])

    def compare_record(record, reference):
        return dict(
            output=compare_output(record["output"], reference["output"]),
            prefix=common.compare(torch, record["prefix"], reference["prefix"]),
            blocks=common.compare(torch, record["blocks"], reference["blocks"]))

    for name, records in captures.items():
        summaries[name]["against_original"] = [compare_record(record, reference)
            for record, reference in zip(records, captures["baseline"])]
        if "baseline_no_fma" in captures and name == "norm_no_fma":
            summaries[name]["against_no_fma_baseline"] = [compare_output(record["output"], reference["output"])
                for record, reference in zip(records, captures["baseline_no_fma"])]
    observer_parity = {}
    for observer, original in (("baseline_debug", "baseline"), ("norm_debug", "norm_cache")):
        if observer not in captures:
            observer_parity[observer] = dict(available=False, passed=False)
            continue
        checks = [compare_record(a, b) for a, b in zip(captures[observer], captures[original])]
        if observer == "norm_debug":
            checks += [dict(cache=common.compare(torch, a["cache"], b["cache"]))
                       for a, b in zip(captures[observer], captures[original])]
        observer_parity[observer] = dict(available=True, passed=require_exact(checks), checks=checks)
    # Retain failed finite/parity checks even if a later diagnostic export fails.
    common.write_json(output / "diagnostic-comparisons.json", dict(arms=summaries, observer_parity=observer_parity))
    nonfinite_raw_values = 0
    def finite_json(value):
        nonlocal nonfinite_raw_values
        if isinstance(value, dict):
            return {key: finite_json(item) for key, item in value.items()}
        if isinstance(value, list):
            return [finite_json(item) for item in value]
        if isinstance(value, float) and not math.isfinite(value):
            nonfinite_raw_values += 1
            return None
        return value
    details = {}
    def compare_values(name, actual, reference):
        actual, reference = actual.contiguous(), reference.contiguous()
        report = common.compare(torch, actual, reference)
        unequal = (actual != reference).reshape(-1).nonzero().reshape(-1)[:32]
        a, b = actual.reshape(-1), reference.reshape(-1)
        report["first_differences"] = [dict(index=int(index), actual=float(a[index].item()), reference=float(b[index].item()),
            actual_bits=int(a.view(torch.int32)[index].item()) & 0xffffffff,
            reference_bits=int(b.view(torch.int32)[index].item()) & 0xffffffff) for index in unequal.tolist()]
        details[name] = finite_json(report)
    if "baseline_debug" in captures:
        reference = captures["baseline_debug"][1]["stats"]
        compare_values("block0_producer_cache_vs_original_consumer_rstd", captures["norm_cache"][0]["cache"][:, 0], reference[:, 0, 1])
        compare_values("block1_writer_cache_vs_original_live_prefix_rstd", captures["norm_cache"][1]["cache"][:, 1], reference[:, 1, 1])
        compare_values("block0_cache_vs_initial_output_rstd", captures["norm_cache"][0]["cache"][:, 0], captures["baseline_debug"][0]["stats"][:, 9, 1])
        compare_values("native_initial_output_vs_native_consumer_rstd", captures["baseline_debug"][0]["stats"][:, 9, 1], reference[:, 0, 1])
        if "norm_debug" in captures:
            for metric, index in (("sum_squares", 0), ("rstd", 1), ("score_dot", 2), ("logit", 3)):
                compare_values("observed_consumer_" + metric, captures["norm_debug"][1]["stats"][:, :2, index], reference[:, :2, index])
            compare_values("norm_observer_producer_cache_parity", captures["norm_debug"][0]["cache"], captures["norm_cache"][0]["cache"])
    # Compact raw FP32 arrays let offline review correlate error rows and ULPs.
    raw = {}
    for name, records in captures.items():
        raw[name] = []
        for index, record in enumerate(records):
            item = {}
            if record["cache"] is not None:
                item["written_cache"] = record["cache"].cpu().tolist()
            if record["stats"] is not None:
                item["source_stats"] = record["stats"][:, :index + 1].cpu().tolist() if index > 0 else []
                item["output_norm_sum_rstd"] = record["stats"][:, 9, :2].cpu().tolist()
            actual = common.outputs(record["output"])
            reference = common.outputs(captures["baseline"][index]["output"])
            different = (actual[0].float() != reference[0].float()).nonzero()
            item["output_difference_indices"] = different[:128].cpu().tolist()
            item["output_difference_values"] = [dict(
                row=row, column=column, actual=float(actual[0][row, column].item()),
                reference=float(reference[0][row, column].item()))
                for row, column in item["output_difference_indices"]]
            if quant_dtype is not None:
                item["output_scales"] = actual[1].cpu().tolist()
                item["scale_difference_indices"] = (actual[1] != reference[1]).nonzero()[:128].cpu().tolist()
            raw[name].append(item)
    common.write_json(output / "diagnostic-values.json", finite_json(raw))
    return dict(status="diagnostic_collected", candidate_qualified=False, rows=rows, dtype=dtype,
                seed=seed, rng_allocation_shapes=dict(signals=list(signals.shape), weights=list(weights.shape)),
                input_sha256=input_hashes, steps=plan[:2], arms=summaries, comparisons=details,
                observer_parity=observer_parity,
                observer_outputs_exact=all(item["passed"] for item in observer_parity.values()),
                nonfinite_raw_values_encoded_as_null=nonfinite_raw_values,
                limits=["Observed kernels add stores; compare observer outputs against originals before interpreting values",
                        "Exact observer outputs do not alone prove unchanged intermediate reductions; inspect saved compiler artifacts",
                        "No-FMA is a diagnostic control, not a proposed change to SA arithmetic",
                        "No timings, graph validation, tolerance relaxation or model-integration qualification"])


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
    parser.add_argument("--diagnose", action="store_true", help="Observe first two 280-row producer-consumer operations; no candidate qualification")
    parser.add_argument("--diagnose-seed", type=int, default=20261112, help="Diagnostic-only RNG seed; identical across dtype lanes")
    parser.add_argument("--diagnose-dtype", choices=("bf16", "fp8", "both"), default="bf16", help="Diagnostic-only output dtype; both runs BF16 and FP8 with identical inputs")
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
    diagnostic_dtypes = ("bf16", "fp8") if args.diagnose_dtype == "both" else (args.diagnose_dtype,)
    if args.diagnose:
        result["diagnostic_plan"] = [dict(rows=280, dtype=dtype, seed=args.diagnose_seed)
                                     for dtype in diagnostic_dtypes]
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
        if args.diagnose:
            with torch.inference_mode():
                if len(diagnostic_dtypes) == 1:
                    result["diagnostic"] = diagnose_first_consumer(
                        torch, paths, args.output, plan, progress, check_budget,
                        seed=args.diagnose_seed, dtype=diagnostic_dtypes[0])
                else:
                    result["diagnostics"] = []
                    for dtype in diagnostic_dtypes:
                        check_budget(); progress("diagnostic_lane_started", dtype=dtype, seed=args.diagnose_seed)
                        record = diagnose_first_consumer(
                            torch, paths, args.output / dtype, plan, progress, check_budget,
                            seed=args.diagnose_seed, dtype=dtype)
                        result["diagnostics"].append(record); save()
                        progress("diagnostic_lane_finished", dtype=dtype, observer_outputs_exact=record["observer_outputs_exact"])
                    result["diagnostic_inputs_identical"] = (
                        result["diagnostics"][0]["input_sha256"] == result["diagnostics"][1]["input_sha256"])
                    if not result["diagnostic_inputs_identical"]:
                        raise AssertionError("Diagnostic dtype lanes did not use identical RNG inputs")
            result["status"] = "diagnostic_collected"
            result["candidate_qualified"] = False
            result["elapsed_seconds"] = time.monotonic() - started
            save(); progress("finished", status=result["status"], candidate_qualified=False)
            return 0
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

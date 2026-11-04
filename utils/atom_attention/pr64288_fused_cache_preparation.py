"""Fuse Q preparation with a conservative cache-bound scan (experiment only)."""
import json
import sys
from pathlib import Path

import torch
import triton
import triton.language as tl
import triton.testing
from aiter.ops.shuffle import shuffle_weight
from pr64288_cache_bound_diagnostics import scales_from_cache
from pr64288_cache_bound_scan import _cache_bound_reduce, gpu_scales
from pr64288_cache_preparation_comparison import quantize_q
from vllm.v1.attention.ops.rocm_aiter_mla_fp8_prefill import ProjectionWeight, project
from vllm.v1.attention.ops.rocm_aiter_mla_fp8_quant import (
    _partial_amax,
    _quant_tensor,
    fused_qkv_per_tensor_quant,
)


@triton.jit
def _q_and_cache_partials(
    Q, QPartial, Cache, Indices, CachePartial, NQ, ROWS: tl.constexpr,
    SHAPE: tl.constexpr, STRIDE: tl.constexpr,
    QPARTS: tl.constexpr, QBLOCK: tl.constexpr,
    BLOCK_ROWS: tl.constexpr, BLOCK_COLS: tl.constexpr,
):
    program = tl.program_id(0)
    if program < QPARTS:
        _partial_amax(Q, QPartial, NQ, SHAPE, STRIDE, QPARTS, QBLOCK)
    else:
        tile = program - QPARTS
        rows = tile * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
        cols = tl.arange(0, BLOCK_COLS)
        indices = tl.load(Indices + rows, rows < ROWS, other=0).to(tl.int64)
        x = tl.load(
            Cache + indices[:, None] * 576 + cols[None, :],
            (rows[:, None] < ROWS) & (cols[None, :] < 576), other=0.0,
        ).to(tl.float32)
        squares = tl.sum(tl.where(cols[None, :] < 512, x * x, 0), 1)
        rope = tl.max(tl.where(cols[None, :] >= 512, tl.abs(x), 0), 1)
        tl.store(CachePartial + tile * 2, tl.max(squares, 0))
        tl.store(CachePartial + tile * 2 + 1, tl.max(rope, 0))


@triton.jit
def _q_quant_and_cache_reduce(
    Q, Q8, QPartial, QScale, CachePartial, WeightNorms, CacheScale, OutputScales,
    NQ, SHAPE: tl.constexpr, STRIDE: tl.constexpr,
    QPARTS: tl.constexpr, QBLOCK: tl.constexpr,
    CACHE_PARTS: tl.constexpr, CACHE_BLOCK: tl.constexpr,
):
    _quant_tensor(
        Q, Q8, QPartial, QScale, QScale, NQ, SHAPE, STRIDE,
        QPARTS, QBLOCK, False, False,
    )
    if tl.program_id(0) == 0:
        _cache_bound_reduce(
            CachePartial, WeightNorms, CacheScale, OutputScales,
            CACHE_PARTS, CACHE_BLOCK,
        )


def fused_prepare(q, cache, cache_scale, indices, weight_norms):
    n = q.numel()
    block = 8192 if n >= 4 * 1024 * 1024 else 4096
    qparts = max(1, min(128, triton.next_power_of_2(triton.cdiv(n, block))))
    cparts = max(1, triton.cdiv(indices.numel(), 16))
    qpartial = torch.empty(qparts, device=q.device, dtype=torch.float32)
    cpartial = torch.empty((cparts, 2), device=q.device, dtype=torch.float32)
    qscale = torch.empty(1, device=q.device, dtype=torch.float32)
    cscales = torch.empty(2, device=q.device, dtype=torch.float32)
    q8 = torch.empty(q.shape, device=q.device, dtype=torch.float8_e4m3fn)
    _q_and_cache_partials[(qparts + cparts,)](
        q, qpartial, cache, indices, cpartial, n, indices.numel(),
        tuple(q.shape[1:]), q.stride(), qparts, block, 16, 1024,
        num_warps=4,
    )
    _q_quant_and_cache_reduce[(max(1, triton.cdiv(n, block)),)](
        q, q8, qpartial, qscale, cpartial, weight_norms, cache_scale, cscales,
        n, tuple(q.shape[1:]), q.stride(), qparts, block,
        cparts, triton.next_power_of_2(cparts), num_warps=4,
    )
    return q8, qscale, cscales


def check_outputs(actual, q, k, v, expected_scales):
    reference = fused_qkv_per_tensor_quant(q, k, v)
    assert torch.equal(actual[0].view(torch.uint8), reference[0].view(torch.uint8))
    assert torch.equal(actual[1], reference[3])
    torch.testing.assert_close(actual[2], expected_scales, rtol=1e-6, atol=1e-8)


def run(report):
    torch.manual_seed(89)
    heads = 12
    weight = (torch.randn(heads * 256, 512, device="cuda") * 32).to(torch.float8_e4m3fn)
    wscales = torch.linspace(0.0002, 0.0008, heads * 256, device="cuda")
    params = ProjectionWeight(shuffle_weight(weight, (16, 16)), wscales, True)
    cache_scale = torch.tensor([0.137], device="cuda", dtype=torch.float32)
    for rows in (8192, 32768):
        cache = torch.randn(rows + 37, 576, device="cuda").to(torch.float8_e4m3fn)
        indices = torch.randperm(rows + 37, device="cuda")[:rows].to(torch.int32)
        expected, weight_norms = scales_from_cache(cache, cache_scale, indices, params, heads)
        for queries in (512, 2048, 8192):
            q = torch.randn(queries, heads, 192, device="cuda", dtype=torch.bfloat16)
            kv = torch.randn(queries, heads, 320, device="cuda", dtype=torch.bfloat16)
            k, v = kv[..., :192], kv[..., 192:]
            check_outputs(fused_prepare(q, cache, cache_scale, indices, weight_norms), q, k, v, expected)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured = fused_prepare(q, cache, cache_scale, indices, weight_norms)
            q.mul_(3)
            cache[0:1].copy_(torch.full((1, 576), 32, device="cuda").to(torch.float8_e4m3fn))
            indices[0] = 0
            cache_scale.mul_(0.9)
            graph.replay()
            expected, _ = scales_from_cache(cache, cache_scale, indices, params, heads)
            check_outputs(captured, q, k, v, expected)

            def current_prepare():
                quant = fused_qkv_per_tensor_quant(q, k, v)
                return quant[0], quant[3], torch.cat((quant[6], quant[7]))

            def separate_prepare():
                q8, qs = quantize_q(q)
                return q8, qs, gpu_scales(cache, cache_scale, indices, weight_norms)

            def combined_prepare():
                return fused_prepare(q, cache, cache_scale, indices, weight_norms)

            def current_path():
                q8, qs, scales = current_prepare()
                return q8, qs, project(cache, cache_scale, indices, params, heads, scales)

            def separate_path():
                q8, qs, scales = separate_prepare()
                return q8, qs, project(cache, cache_scale, indices, params, heads, scales)

            def combined_path():
                q8, qs, scales = combined_prepare()
                return q8, qs, project(cache, cache_scale, indices, params, heads, scales)

            functions = {
                "current_prepare_us": current_prepare,
                "separate_prepare_us": separate_prepare,
                "combined_prepare_us": combined_prepare,
                "current_prepare_project_us": current_path,
                "separate_prepare_project_us": separate_path,
                "combined_prepare_project_us": combined_path,
            }
            for function in functions.values():
                function()
            samples = {key: [] for key in functions}
            keys = list(functions)
            for repetition in range(3):
                for key in keys[repetition:] + keys[:repetition]:
                    samples[key].append(
                        triton.testing.do_bench_cudagraph(functions[key], rep=150) * 1000
                    )
            result = {
                "cached_rows": rows, "query_rows": queries,
                "q_bitwise_equal": True, "cache_scales_match": True,
                "changed_input_graph_replay": True, "samples_us": samples,
            }
            result.update({key: sorted(values)[1] for key, values in samples.items()})
            result["combined_over_current"] = (
                result["combined_prepare_project_us"] / result["current_prepare_project_us"]
            )
            report["cases"].append(result)
            print("FUSED_PREPARATION_COMPARISON", json.dumps(result), flush=True)
    report["status"] = "passed"


if __name__ == "__main__":
    report = {
        "status": "running", "cases": [],
        "native_source": "e2393d702a3736dffdfbfc7eb25abbabe484a1df",
        "scope": "Synthetic one-GPU preparation and cached projection only; no FMHA, model accuracy or end-to-end claim. One cached chunk, 12 heads. Experimental fused GPU preparation, no vLLM integration.",
    }
    try:
        run(report)
    except Exception as error:
        report.update(status="failed", error=str(error))
        raise
    finally:
        Path(sys.argv[1]).write_text(json.dumps(report, indent=2))

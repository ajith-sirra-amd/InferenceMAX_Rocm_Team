"""Prototype a GPU cache-range scan; no serving integration or model accuracy claim."""
import json
import sys
from pathlib import Path

import torch
import triton
import triton.language as tl
import triton.testing
from aiter.ops.shuffle import shuffle_weight
from pr64288_cache_bound_diagnostics import scales_from_cache
from vllm.v1.attention.ops.rocm_aiter_mla_fp8_prefill import ProjectionWeight, project


@triton.jit
def _cache_bound_partials(Cache, Indices, Partial, ROWS: tl.constexpr,
                          BLOCK_ROWS: tl.constexpr, BLOCK_COLS: tl.constexpr):
    block = tl.program_id(0)
    rows = block * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    cols = tl.arange(0, BLOCK_COLS)
    indices = tl.load(Indices + rows, rows < ROWS, other=0).to(tl.int64)
    x = tl.load(
        Cache + indices[:, None] * 576 + cols[None, :],
        (rows[:, None] < ROWS) & (cols[None, :] < 576), other=0.0,
    ).to(tl.float32)
    norm_sq = tl.sum(tl.where(cols[None, :] < 512, x * x, 0), 1)
    rope = tl.max(tl.where(cols[None, :] >= 512, tl.abs(x), 0), 1)
    tl.store(Partial + block * 2, tl.max(norm_sq, 0))
    tl.store(Partial + block * 2 + 1, tl.max(rope, 0))


@triton.jit
def _cache_bound_reduce(Partial, WeightNorms, CacheScale, Output,
                        PARTS: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    squares = tl.load(Partial + offsets * 2, offsets < PARTS, other=0)
    rope = tl.load(Partial + offsets * 2 + 1, offsets < PARTS, other=0)
    latent_norm = tl.sqrt(tl.max(squares, 0))
    rope_amax = tl.max(rope, 0)
    scale = tl.load(CacheScale)
    k_norm = tl.load(WeightNorms)
    v_norm = tl.load(WeightNorms + 1)
    margin = 1.0 + 1.0 / 1024.0
    ks = tl.maximum(latent_norm * k_norm, rope_amax) * scale * margin / 448.0
    vs = latent_norm * v_norm * scale * margin / 448.0
    tl.store(Output, tl.maximum(ks, 1e-6))
    tl.store(Output + 1, tl.maximum(vs, 1e-6))


def gpu_scales(cache, cache_scale, indices, weight_norms):
    assert cache.ndim == 2 and cache.shape[1] == 576 and cache.is_contiguous()
    assert cache.dtype == torch.float8_e4m3fn
    assert indices.dtype == torch.int32 and indices.is_contiguous()
    assert weight_norms.dtype == cache_scale.dtype == torch.float32
    assert weight_norms.numel() == 2 and cache_scale.numel() == 1
    rows = indices.numel()
    if not rows:
        return torch.full((2,), 1e-6, device=cache.device, dtype=torch.float32)
    parts = triton.cdiv(rows, 16)
    partial = torch.empty((parts, 2), device=cache.device, dtype=torch.float32)
    output = torch.empty(2, device=cache.device, dtype=torch.float32)
    _cache_bound_partials[(parts,)](
        cache, indices, partial, rows, 16, 1024, num_warps=4,
    )
    _cache_bound_reduce[(1,)](
        partial, weight_norms, cache_scale, output, parts,
        triton.next_power_of_2(parts), num_warps=4,
    )
    return output


def run(report):
    torch.manual_seed(79)
    heads = 12
    weight = (torch.randn(heads * 256, 512, device="cuda") * 32).to(torch.float8_e4m3fn)
    scales = torch.linspace(0.0002, 0.0008, heads * 256, device="cuda")
    params = ProjectionWeight(shuffle_weight(weight, (16, 16)), scales, True)
    cache_scale = torch.tensor([0.137], device="cuda", dtype=torch.float32)
    for rows in (0, 1, 386, 8192, 32768):
        raw = torch.randn(rows + 37, 576, device="cuda")
        cache = raw.to(torch.float8_e4m3fn)
        indices = torch.randperm(rows + 37, device="cuda")[:rows].to(torch.int32)
        if rows:
            expected, weight_norms = scales_from_cache(cache, cache_scale, indices, params, heads)
        else:
            weight_norms = torch.ones(2, device="cuda")
            expected = torch.full((2,), 1e-6, device="cuda")
        actual = gpu_scales(cache, cache_scale, indices, weight_norms)
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-8)
        record = {"rows": rows, "scales_match": True}
        if not rows:
            report["cases"].append(record)
            continue
        # Capture reads every invocation's live cache and scale, not host maxima.
        for _ in range(2):
            gpu_scales(cache, cache_scale, indices, weight_norms)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = gpu_scales(cache, cache_scale, indices, weight_norms)
        raw[indices[0].long(), :512] = 128
        raw[indices[0].long(), 512:] = 448
        cache.copy_(raw.to(torch.float8_e4m3fn))
        cache_scale.mul_(0.5)
        graph.replay()
        expected, _ = scales_from_cache(cache, cache_scale, indices, params, heads)
        torch.testing.assert_close(captured, expected, rtol=1e-6, atol=1e-8)
        record["changed_cache_and_scale_graph_replay"] = True
        if rows >= 386:
            def scan():
                return gpu_scales(cache, cache_scale, indices, weight_norms)

            def projection_only():
                return project(cache, cache_scale, indices, params, heads, expected)

            def scan_and_projection():
                live_scales = gpu_scales(cache, cache_scale, indices, weight_norms)
                return project(cache, cache_scale, indices, params, heads, live_scales)

            for fn in (scan, projection_only, scan_and_projection):
                fn()
            scan_us, projection_us, combined_us = [
                triton.testing.do_bench_cudagraph(fn, rep=200) * 1000
                for fn in (scan, projection_only, scan_and_projection)
            ]
            record.update(
                scan_us=scan_us, projection_us=projection_us,
                scan_and_projection_us=combined_us,
                combined_over_projection=combined_us/projection_us,
            )
        report["cases"].append(record)
        print("GPU_CACHE_BOUND", json.dumps(record), flush=True)
    report["status"] = "passed"


if __name__ == "__main__":
    output = Path(sys.argv[1])
    report = {
        "status": "running", "cases": [],
        "scope": "GPU scale scan and projection microbenchmark only; no serving or end-to-end uplift claim",
        "native_source": "e2393d702a3736dffdfbfc7eb25abbabe484a1df",
    }
    try:
        run(report)
    except Exception as error:
        report.update(status="failed", error=str(error))
        raise
    finally:
        output.write_text(json.dumps(report, indent=2))

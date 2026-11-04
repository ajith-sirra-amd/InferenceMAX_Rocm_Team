"""Compare existing preparation with Q-only quantization and cached scales.

This is a synthetic component benchmark; neither path is a serving validation.
"""
import json
import sys
from pathlib import Path

import torch
import triton
import triton.testing
from aiter.ops.shuffle import shuffle_weight
from pr64288_cache_bound_diagnostics import scales_from_cache
from pr64288_cache_bound_scan import gpu_scales
from vllm.v1.attention.ops.rocm_aiter_mla_fp8_prefill import ProjectionWeight, project
from vllm.v1.attention.ops.rocm_aiter_mla_fp8_quant import (
    _partial_amax,
    _quant_tensor,
    fused_qkv_per_tensor_quant,
)


def quantize_q(q):
    """Use the current PR's exact Q operations, without K/V launch programs."""
    n = q.numel()
    single = n <= 8192
    block = (max(256, triton.next_power_of_2(n)) if single
             else (8192 if n >= 4 * 1024 * 1024 else 4096))
    parts = min(128, triton.next_power_of_2(triton.cdiv(n, block))) if n else 1
    partial = torch.empty(parts, device=q.device, dtype=torch.float32)
    scale = torch.empty(1, device=q.device, dtype=torch.float32)
    output = torch.empty(q.shape, device=q.device, dtype=torch.float8_e4m3fn)
    if not single:
        _partial_amax[(parts,)](
            q, partial, n, tuple(q.shape[1:]), q.stride(), parts, block,
            num_warps=4,
        )
    _quant_tensor[(max(1, triton.cdiv(n, block)),)](
        q, output, partial, scale, scale, n, tuple(q.shape[1:]), q.stride(),
        parts, block, False, single, num_warps=4,
    )
    return output, scale


def check_q(q, k, v):
    full = fused_qkv_per_tensor_quant(q, k, v)
    q8, scale = quantize_q(q)
    assert torch.equal(q8.view(torch.uint8), full[0].view(torch.uint8))
    assert torch.equal(scale, full[3])


def run(report):
    torch.manual_seed(83)
    heads = 12
    weight = (torch.randn(heads * 256, 512, device="cuda") * 32).to(torch.float8_e4m3fn)
    scales = torch.linspace(0.0002, 0.0008, heads * 256, device="cuda")
    params = ProjectionWeight(shuffle_weight(weight, (16, 16)), scales, True)
    cache_scale = torch.tensor([0.137], device="cuda", dtype=torch.float32)

    # A selected cache row whose byte offset exceeds signed int32.
    # Only the selected row is read, so the rest need not be initialized.
    large_row = 2**31 // 576 + 1
    large_cache = torch.empty((large_row + 1, 576), device="cuda", dtype=torch.float8_e4m3fn)
    selected = torch.ones((1, 576), device="cuda", dtype=torch.float8_e4m3fn)
    large_cache[large_row:large_row + 1].copy_(selected)
    index = torch.tensor([large_row], device="cuda", dtype=torch.int32)
    norms = torch.tensor([0.3, 0.2], device="cuda", dtype=torch.float32)
    oracle = gpu_scales(selected, cache_scale, torch.zeros_like(index), norms)
    actual = gpu_scales(large_cache, cache_scale, index, norms)
    torch.testing.assert_close(actual, oracle, rtol=0, atol=0)
    report["int64_offset_check"] = {"row": large_row, "byte_offset": large_row * 576, "passed": True}
    del large_cache, selected, index, norms

    for rows in (8192, 32768):
        cache = torch.randn(rows + 37, 576, device="cuda").to(torch.float8_e4m3fn)
        indices = torch.randperm(rows + 37, device="cuda")[:rows].to(torch.int32)
        expected, weight_norms = scales_from_cache(cache, cache_scale, indices, params, heads)
        torch.testing.assert_close(
            gpu_scales(cache, cache_scale, indices, weight_norms), expected,
            rtol=1e-6, atol=1e-8,
        )
        for queries in (512, 2048, 8192):
            q = torch.randn(queries, heads, 192, device="cuda", dtype=torch.bfloat16)
            # Strided K/V views match an expanded projection's split layout.
            kv = torch.randn(queries, heads, 320, device="cuda", dtype=torch.bfloat16)
            k, v = kv[..., :192], kv[..., 192:]
            check_q(q, k, v)
            quantize_q(q)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured_q, captured_scale = quantize_q(q)
            q.mul_(3)
            graph.replay()
            ref = fused_qkv_per_tensor_quant(q, k, v)
            assert torch.equal(captured_q.view(torch.uint8), ref[0].view(torch.uint8))
            assert torch.equal(captured_scale, ref[3])

            def current_prepare():
                quant = fused_qkv_per_tensor_quant(q, k, v)
                return quant[0], quant[3], torch.cat((quant[6], quant[7]))

            def proposed_prepare():
                q8, qs = quantize_q(q)
                return q8, qs, gpu_scales(cache, cache_scale, indices, weight_norms)

            def current_path():
                q8, qs, output_scales = current_prepare()
                return q8, qs, project(cache, cache_scale, indices, params, heads, output_scales)

            def proposed_path():
                q8, qs, output_scales = proposed_prepare()
                return q8, qs, project(cache, cache_scale, indices, params, heads, output_scales)

            functions = {
                "current_prepare_us": current_prepare,
                "proposed_prepare_us": proposed_prepare,
                "current_prepare_project_us": current_path,
                "proposed_prepare_project_us": proposed_path,
            }
            for function in functions.values():
                function()
            # Rotate measurement order; report medians and ranges rather than one sample.
            samples = {key: [] for key in functions}
            keys = list(functions)
            for repetition in range(3):
                for key in keys[repetition:] + keys[:repetition]:
                    samples[key].append(
                        triton.testing.do_bench_cudagraph(functions[key], rep=150) * 1000
                    )
            result = {
                "cached_rows": rows, "query_rows": queries,
                "q_bitwise_equal": True, "q_changed_input_graph_replay": True,
                "samples_us": samples,
            }
            result.update({key: sorted(values)[1] for key, values in samples.items()})
            result["proposed_over_current"] = (
                result["proposed_prepare_project_us"] / result["current_prepare_project_us"]
            )
            report["cases"].append(result)
            print("PREPARATION_COMPARISON", json.dumps(result), flush=True)
    report["status"] = "passed"


if __name__ == "__main__":
    report = {
        "status": "running", "cases": [],
        "native_source": "e2393d702a3736dffdfbfc7eb25abbabe484a1df",
        "scope": "Synthetic one-GPU preparation and cached projection only; no FMHA, model accuracy or end-to-end claim. One cached chunk, 12 heads.",
    }
    try:
        run(report)
    except Exception as error:
        report.update(status="failed", error=str(error))
        raise
    finally:
        Path(sys.argv[1]).write_text(json.dumps(report, indent=2))

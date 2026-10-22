"""GPU microbenchmark: existing AITER fused FP8-weight reconstruction vs BF16.

Run inside the pinned benchmark container:
python3 patches/fp8_fused_reconstruction_probe.py --output report.json

This does not patch serving or enable FP8 FMHA.
"""
import argparse
import json
import pathlib

import torch
from aiter.ops.shuffle import shuffle_weight
from fp8_fused_kv_kernel import launch
from fp8_fused_kv_b import validate_outputs
from vllm.platforms import current_platform


@torch.no_grad()
def check_and_measure(device, rows):
    torch.cuda.set_device(device)
    torch.manual_seed(42)
    dtype = current_platform.fp8_dtype()
    cache = (torch.randn((rows + 7, 576), device=device) * 0.5).to(dtype)
    raw_weight = (torch.randn((3072, 512), device=device) * 0.5).to(dtype)
    scale = torch.linspace(0.015625, 0.0625, 3072, device=device)
    cache_scale = torch.tensor([0.5], device=device)
    indices = torch.randperm(rows + 7, device=device)[:rows].to(torch.int32)
    cu = torch.tensor([0, rows], device=device, dtype=torch.int32)
    k = torch.empty((rows, 12, 192), device=device, dtype=torch.bfloat16)
    v = torch.empty((rows, 12, 128), device=device, dtype=torch.bfloat16)
    kb, vb = torch.empty_like(k), torch.empty_like(v)
    bf16_weight = (raw_weight.float() * scale[:, None]).to(torch.bfloat16).contiguous()

    def fp8_call():
        launch(cache, cache_scale, indices, raw_weight, scale, k, v, False, 0)

    def bf16_call():
        launch(cache, cache_scale, indices, bf16_weight, None, kb, vb, False, 0)

    fp8_call()
    bf16_call()
    rank_scales = torch.tensor([0.5, 2.0], device=device)
    split = (cache.shape[0] + 1) // 2
    packed = shuffle_weight(raw_weight)
    for shuffled, candidate in [(False, raw_weight), (True, packed)]:
        launch(cache, rank_scales, indices, candidate, scale, k, v, shuffled, split)
        validate_outputs(cache, rank_scales, indices, candidate, scale, shuffled, split, k, v)
    fp8_call()
    # Independent explicit projection reference, including row permutation,
    # varying channel scales and the unprojected RoPE component.
    selected = cache.float().index_select(0, indices.long()) * cache_scale
    projected = torch.nn.functional.linear(
        selected[:, :512], raw_weight.float() * scale[:, None]
    ).reshape(rows, 12, 256)
    ref_k = torch.cat((projected[:, :, :128],
                       selected[:, None, 512:].expand(-1, 12, -1)), dim=-1)
    ref_v = projected[:, :, 128:]
    torch.testing.assert_close(k.float(), ref_k, rtol=0.03, atol=0.03125)
    torch.testing.assert_close(v.float(), ref_v, rtol=0.03, atol=0.03125)
    # The RoPE copy is independently scaled and should match BF16 rounding exactly.
    torch.testing.assert_close(k[:, :, 128:], ref_k[:, :, 128:].to(k.dtype),
                               rtol=0, atol=0)
    ref_b = torch.nn.functional.linear(selected[:, :512], bf16_weight.float()).reshape(rows,12,256)
    torch.testing.assert_close(kb[:, :, :128].float(), ref_b[:, :, :128],
                               rtol=0.03, atol=0.03125)
    torch.testing.assert_close(vb.float(), ref_b[:, :, 128:],
                               rtol=0.03, atol=0.03125)

    def timing(call):
        for _ in range(5):
            call()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(50):
            call()
        end.record()
        end.synchronize()
        return start.elapsed_time(end) / 50

    fp8_ms, bf16_ms = timing(fp8_call), timing(bf16_call)
    fused_ms = timing(lambda: launch(cache, rank_scales, indices, packed, scale, k, v, True, split))
    fp8_call()
    error = torch.cat(((k.float()-ref_k).flatten(), (v.float()-ref_v).flatten()))
    reference = torch.cat((ref_k.flatten(), ref_v.flatten()))
    result = {
        "device": str(device), "rows": rows,
        "fp8_weight_fused_ms": fp8_ms, "bf16_weight_fused_ms": bf16_ms,
        "integrated_preshuffled_rank_scaled_ms": fused_ms,
        "bf16_over_integrated_speed_ratio": bf16_ms / fused_ms,
        "bf16_over_fp8_speed_ratio": bf16_ms / fp8_ms,
        "max_abs_error": error.abs().max().item(),
        "relative_rms_error": (error.square().mean().sqrt() /
                               reference.square().mean().sqrt()).item(),
        "fp8_weight_layout": "unshuffled_and_AITER_preshuffled_validated",
        "unequal_rank_scales_validated": True,
        "output_dtype": str(k.dtype), "numerical_checks": "passed",
        "scope": "local_reconstruction_only_no_collective_no_fmha",
    }
    print("[FP8-FUSED-RECONSTRUCTION-PROBE]", json.dumps(result), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    results = []
    if torch.cuda.device_count() != 8:
        raise RuntimeError('Expected all8 GPUs for numerical preflight')
    for index in range(torch.cuda.device_count()):
        for rows in (33, 256, 2048):
            results.append(check_and_measure(torch.device(f"cuda:{index}"), rows))
    pathlib.Path(args.output).write_text(json.dumps(results, indent=2))

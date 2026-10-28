"""Measure cache widening, BF16 projection and final FP8 K/V errors separately."""
import json
import os

import torch

from inferencex_atom_attention.projection_fp8 import expand_context_fp8
from inferencex_atom_attention.quantize import fused_qkv_per_tensor_quant
from projection_experiment import fixture, native


def metric(actual, expected):
    a, e = actual.double(), expected.double()
    if not torch.isfinite(a).all().item() or not torch.isfinite(e).all().item():
        raise AssertionError("Nonfinite numerical result")
    diff = a - e
    rms = diff.square().mean().sqrt()
    ref_rms = e.square().mean().sqrt()
    return {
        "rms": rms.item(),
        "relative_rms": (rms / ref_rms.clamp_min(1e-12)).item(),
        "max_abs": diff.abs().max().item(),
        "reference_rms": ref_rms.item(),
    }


def run_case(cache_scale):
    torch.manual_seed(42)
    data = fixture(8192)
    cache, scale, indices, _, weight, heads = data
    scale.fill_(cache_scale)
    # The source is already FP8. This comparison does not claim to recover
    # precision lost when the cache was originally quantized.
    widened = cache.to(torch.bfloat16)
    torch.testing.assert_close(widened.float(), cache.float(), rtol=0, atol=0)
    selected = cache.float()[indices.long()].double()
    # Independent FP64 projection of the same stored FP8 cache and BF16 weights.
    projected64 = ((selected[:, :512] * scale.double()) @ weight.double().T).view(
        -1, heads, 256
    )
    k64 = torch.cat((
        projected64[:, :, :128],
        (selected[:, 512:] * scale.double()).unsqueeze(1).expand(-1, heads, -1),
    ), dim=-1)
    v64 = projected64[:, :, 128:]
    k, v = native(data)
    q = torch.randn((512, heads, 192), device="cuda", dtype=torch.bfloat16)
    existing = fused_qkv_per_tensor_quant(q, k, v)
    k8, v8, ks, vs = existing[1], existing[2], existing[4], existing[5]

    # Separately generated new-token tensors, with the same projection weights,
    # provide ATOM-style scales. No cached-output maxima are used for these.
    new = fixture(512)
    new[1].fill_(cache_scale)
    new = (*new[:4], weight, heads)
    knew, vnew = native(new)
    prepared = fused_qkv_per_tensor_quant(q, knew, vnew)
    supplied = torch.cat((prepared[6], prepared[7]))
    direct = expand_context_fp8(
        cache, scale, indices, weight, heads,
        mode="supplied_scales", output_scales=supplied,
    )
    same_scale_reference = tuple(
        (x.float() * s.reciprocal()).clamp(-448, 448).to(torch.float8_e4m3fn)
        for x, s in ((k, supplied[:1]), (v, supplied[1:]))
    )
    for actual, expected in zip(direct[:2], same_scale_reference):
        torch.testing.assert_close(actual.float(), expected.float(), rtol=0, atol=0)
    rows = {}
    for name, bf16, fp64, fp8, descale, supplied8, supplied_scale in [
        ("K", k, k64, k8, ks, direct[0], direct[2]),
        ("V", v, v64, v8, vs, direct[1], direct[3]),
    ]:
        dequant = fp8.double() * descale.double()
        supplied_dequant = supplied8.double() * supplied_scale.double()
        rows[name] = {
            "bf16_projection_vs_fp64": metric(bf16, fp64),
            "final_dynamic_fp8_vs_bf16_projection": metric(dequant, bf16),
            "whole_chain_dynamic_fp8_vs_fp64": metric(dequant, fp64),
            "atom_supplied_fp8_vs_bf16_projection": metric(supplied_dequant, bf16),
            "dynamic_clipping_fraction": (
                bf16.float().abs() > 448 * descale
            ).float().mean().item(),
            "supplied_clipping_fraction": (
                bf16.float().abs() > 448 * supplied_scale
            ).float().mean().item(),
            "dynamic_scale": descale.item(),
            "supplied_scale": supplied_scale.item(),
        }
    print("KV_CHAIN_NUMERICS", json.dumps({
        "cache_scale": scale.item(), "kv_tokens": 8192, "heads": heads,
        "raw_fp8_to_bf16_widening_max_abs": 0.0,
        "supplied_scale_projection_matches_separate_quant_exactly": True,
        "reference": "FP64 dequantization/projection of identical FP8 cache and BF16 weights",
        "native_order": "widen raw FP8 to BF16; BF16 dot with FP32 accumulation; apply cache scale; round to BF16",
        "K_includes": "128 projected columns plus 64 RoPE columns per head",
        "results": rows,
        "scope": "synthetic tensor errors; not model accuracy or loss versus an unquantized original cache",
    }, allow_nan=False), flush=True)


if __name__ == "__main__":
    if os.environ.get("ATOM_USE_FLYDSL_FP8_PREFILL_ATTN") != "1":
        raise RuntimeError("Expected experiment FP8 flag")
    print("KV_CHAIN_START", torch.cuda.get_device_properties(0), flush=True)
    run_case(0.125)
    run_case(0.137)
    print("KV_CHAIN_NUMERICS_PASS", flush=True)

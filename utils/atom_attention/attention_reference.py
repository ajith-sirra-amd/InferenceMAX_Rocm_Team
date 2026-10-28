"""Compare captured Kimi attention to FP64 on identical FP8 inputs.

AITER arithmetic criteria and the historical BF16-difference gate are both
reported. Passing one never changes the result of the other.
"""
import json
import os
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from inferencex_atom_attention.projection_fp8 import expand_context_fp8
from inferencex_atom_attention.quantize import fused_qkv_per_tensor_quant
from preflight import make_backend
from inferencex_atom_attention.fp8_up_projection import (
    project, projection_weight, reference_projection,
)
from fp8_weight_preflight import quantize_layer
from vllm.v1.attention.backends.mla.prefill.aiter_flash_attn import AiterFlashAttnPrefillBackend
from vllm._aiter_ops import rocm_aiter_ops
from vllm.v1.attention.ops.rocm_aiter_mla_prefill import expand_context


def reference(q, k, v, cuq, cuk, softmax_scale):
    """FP64 attention for every query/head, bounded to 64 query rows at a time."""
    out = torch.zeros((q.shape[0], q.shape[1], v.shape[2]), device=q.device, dtype=torch.float64)
    lse = torch.full((q.shape[1], q.shape[0]), -torch.inf, device=q.device, dtype=torch.float64)
    qlengths, klengths = cuq.tolist(), cuk.tolist()
    if len(qlengths) != len(klengths):
        raise ValueError("Q/KV request-count mismatch")
    for qa, qb, ka, kb in zip(qlengths[:-1], qlengths[1:], klengths[:-1], klengths[1:]):
        if ka == kb:
            continue
        kd = k[ka:kb].double().transpose(0, 1)
        vd = v[ka:kb].double().transpose(0, 1)
        for first in range(qa, qb, 64):
            last = min(first + 64, qb)
            qd = q[first:last].double().transpose(0, 1)
            scores = (qd @ kd.transpose(-1, -2)) * softmax_scale
            lse[:, first:last] = torch.logsumexp(scores, dim=-1)
            out[first:last] = (torch.softmax(scores, dim=-1) @ vd).transpose(0, 1)
    return out, lse


def metrics(actual, expected):
    out, lse = actual
    ref, ref_lse = expected
    if not torch.isfinite(out).all().item():
        raise RuntimeError("Nonfinite output")
    if not torch.equal(torch.isneginf(lse), torch.isneginf(ref_lse)):
        raise RuntimeError("Empty-row LSE mismatch")
    valid = torch.isfinite(ref_lse)
    if not torch.isfinite(lse[valid]).all().item():
        raise RuntimeError("Nonfinite LSE")
    diff = out.double() - ref
    relative = (diff.abs().max() / ref.abs().max().clamp_min(1e-12)).item()
    rms = (diff.square().mean().sqrt() / ref.square().mean().sqrt().clamp_min(1e-12)).item()
    cos = F.cosine_similarity(out.double().flatten(0, 1), ref.flatten(0, 1), dim=-1)
    both_zero = (out.flatten(0, 1).abs().sum(-1) == 0) & (ref.flatten(0, 1).abs().sum(-1) == 0)
    cos = torch.where(both_zero, 1.0, cos)
    error = (lse.double()[valid] - ref_lse[valid]).abs()
    lse_max = error.max().item() if error.numel() else 0.0
    lse_relative = lse_max / max(ref_lse[valid].abs().max().item(), 1.0) if error.numel() else 0.0
    return {"rms": rms, "normalized_max_output_error": relative,
            "min_cosine": cos.min().item(), "lse_max": lse_max,
            "normalized_max_lse_error": lse_relative}


def run(path):
    data = torch.load(path, map_location="cuda", weights_only=True)
    q, cache, weight = data["q"], data["cache"], data["weight"]
    heads = q.shape[1]
    rows = torch.arange(cache.shape[0], device=cache.device, dtype=torch.int32)
    chunk = SimpleNamespace(query_start_loc=data["cu_query"], cu_seq_lens=data["cu_kv"],
                            max_query_len=data["max_query_len"], max_seq_len=data["max_seq_len"])
    backend = make_backend(heads)
    backend.scale = data["softmax_scale"]
    k, v = expand_context(cache, data["cache_scale"], rows, data["cu_kv"], weight, heads, 128, 64, 128)
    dynamic = fused_qkv_per_tensor_quant(q, k, v)
    parity = {}
    for label, src, candidate, scale in zip(("q", "k", "v"), (q, k, v), dynamic[:3], dynamic[3:6]):
        hip = rocm_aiter_ops.per_tensor_quant(src.contiguous(), torch.float8_e4m3fn, scale)[0]
        mismatches = (hip.view(torch.uint8) != candidate.view(torch.uint8)).sum().item()
        parity[label] = {"mismatched_bytes": mismatches, "elements": src.numel(),
                         "max_fp8_value_difference": (hip.float() - candidate.float()).abs().max().item()}
    print("CAPTURED_QUANTIZER_HIP_PARITY", json.dumps(parity), flush=True)
    supplied = expand_context_fp8(cache, data["cache_scale"], rows, weight, heads,
                                  mode="supplied_scales", output_scales=data["supplied_scales"])
    historical_bf16 = AiterFlashAttnPrefillBackend.run_prefill_context_chunk(
        backend, chunk, q, k, v,
    )
    original_ref = reference(q, k, v, data["cu_query"], data["cu_kv"], backend.scale)
    print("CAPTURED_BF16_ARITHMETIC", json.dumps(metrics(historical_bf16, original_ref)), flush=True)
    layer = quantize_layer(weight)
    params = projection_weight(layer, heads)
    native_ref = reference_projection(cache, data["cache_scale"], rows, params, heads)
    # Hold captured scales fixed first to isolate changing projection operands.
    native_supplied = project(cache, data["cache_scale"], rows, params, heads, data["supplied_scales"])
    native_scales = torch.stack([x.abs().max().float() / 448 for x in native_ref])
    native_scales = torch.where(native_scales > 0, native_scales, 1e-6)
    # Round diagnostic bounds upward to cover the FP64 maximum.
    native_scales = torch.nextafter(native_scales, torch.full_like(native_scales, torch.inf)).contiguous()
    native_dynamic = project(cache, data["cache_scale"], rows, params, heads, native_scales)

    def projection_metrics(kv8):
        records = {}
        for name, value, expected, scale in zip(("k", "v"), kv8[:2], native_ref, kv8[2:]):
            actual = value.double() * scale.double()
            diff = actual - expected
            rms = (diff.square().mean().sqrt() / expected.square().mean().sqrt().clamp_min(1e-12)).item()
            peak = (diff.abs().max() / expected.abs().max().clamp_min(1e-12)).item()
            clip = (expected.abs() > 448 * scale.double()).double().mean().item()
            records[name] = {"rms_vs_fp64": rms, "max_error_over_max_reference": peak,
                             "clipping_fraction": clip,
                             "passes_projection_gate": rms < 0.04 and peak < 0.07 and clip == 0}
        return records

    projection_records = {name: projection_metrics(kv8) for name, kv8 in (
        ("native_fp8_supplied", native_supplied), ("native_fp8_dynamic", native_dynamic))}
    print("CAPTURED_NATIVE_FP8_PROJECTION", json.dumps({
        "native_linear_kernel": params.linear_kernel, "results": projection_records,
        "scope": "native quantization of captured BF16 weight; captured scales held fixed or recomputed from FP64",
    }, allow_nan=False), flush=True)
    # Isolate the added FP8 weight rounding before output quantization or FMHA.
    native_weight_ref = reference(q, *native_ref, data["cu_query"], data["cu_kv"], backend.scale)
    print("CAPTURED_WEIGHT_QUANTIZATION_EFFECT", json.dumps(metrics(native_weight_ref, original_ref)), flush=True)
    del native_weight_ref, native_ref
    # Isolate Q and KV rounding. These are references, not executable mixed-dtype FMHA paths.
    deq_dynamic = [x.double() * sc.double() for x, sc in zip(dynamic[:3], dynamic[3:])]
    for label, inputs in (("q_only", (deq_dynamic[0], k, v)),
                          ("kv_only", (q, deq_dynamic[1], deq_dynamic[2]))):
        isolated_ref = reference(*inputs, data["cu_query"], data["cu_kv"], backend.scale)
        print("CAPTURED_QUANTIZATION_EFFECT", json.dumps({
            "quantized": label, "vs_bf16": metrics(isolated_ref, original_ref),
        }), flush=True)
        del isolated_ref
    results = []
    for policy, kv8 in (
        ("supplied", supplied),
        ("native_fp8_supplied", native_supplied),
        ("native_fp8_dynamic", native_dynamic),
        ("dynamic", (dynamic[1], dynamic[2], dynamic[4], dynamic[5])),
    ):
        deq = [x.double() * scale.double() for x, scale in
               ((dynamic[0], dynamic[3]), (kv8[0], kv8[2]), (kv8[1], kv8[3]))]
        quantized_ref = reference(*deq, data["cu_query"], data["cu_kv"], backend.scale)
        for lazy, daz in ((True, True), (False, True), (True, False), (False, False)):
            actual = backend._fp8_attention(
                dynamic[0], kv8[0], kv8[1], softmax_scale=backend.scale,
                cu_seqlens_q=data["cu_query"], cu_seqlens_kv=data["cu_kv"],
                max_seqlen_q=data["max_query_len"], max_seqlen_kv=data["max_seq_len"],
                cross_seqlen=True, causal=False, return_lse=True,
                q_descale=dynamic[3], k_descale=kv8[2], v_descale=kv8[3],
                dualwave_swp_lazy_rescale=lazy, daz=daz,
            )
            same = metrics(actual, quantized_ref)
            bf16 = metrics(actual, original_ref)
            historical = metrics(actual, (historical_bf16[0].double(), historical_bf16[1].double()))
            record = {
                "scale_policy": policy, "lazy_rescale": lazy, "daz": daz,
                "same_fp8_inputs": same, "original_bf16_inputs": bf16, "historical_bf16_kernel": historical,
                "passes_aiter_style_arithmetic_gate": (
                    same["normalized_max_output_error"] < 0.08
                    and same["min_cosine"] > 0.98
                    and same["normalized_max_lse_error"] < 1e-4),
                "passes_original_bf16_gate": historical["rms"] < 0.10 and historical["lse_max"] < 0.20,
            }
            results.append(record)
            print("CAPTURED_FULL_REFERENCE", json.dumps(record, allow_nan=False), flush=True)
        del quantized_ref
    print("CAPTURED_FULL_REFERENCE_COMPLETE", json.dumps({
        "queries": q.shape[0], "heads": heads, "kv_tokens": cache.shape[0],
        "scope": "all rows in one captured chunk; not model accuracy or performance acceptance",
        "results": results,
    }), flush=True)


if __name__ == "__main__":
    run(os.environ["FAILURE_FIXTURE_PATH"])

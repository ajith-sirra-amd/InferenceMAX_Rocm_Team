"""Evaluate FP8 RoPE residual columns with the unchanged 256-dim AITER FMHA.

No serving integration. A second 64-column RoPE term carries Q's rounding
residual. K's primary RoPE term keeps the compact cache's FP8 representation
under power-of-two-aligned scaling; its duplicate is divided by 16.
"""
import json
import os
from types import SimpleNamespace

import torch

from attention_reference import metrics, reference
from fp8_weight_preflight import quantize_layer
from inferencex_atom_attention.fp8_up_projection import project, projection_weight, reference_projection
from inferencex_atom_attention.quantize import fused_qkv_per_tensor_quant
from preflight import make_backend
from vllm._aiter_ops import rocm_aiter_ops
from vllm.v1.attention.backends.mla.prefill.aiter_flash_attn import AiterFlashAttnPrefillBackend
from vllm.v1.attention.ops.rocm_aiter_mla_prefill import expand_context


def emit(marker, data):
    print(marker, json.dumps(data, allow_nan=False), flush=True)


def kernel_ms(invoke):
    for _ in range(3):
        invoke()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(20):
        invoke()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / 20


def run(path):
    data = torch.load(path, map_location="cuda", weights_only=True)
    q, cache, weight = data["q"], data["cache"], data["weight"]
    cuq, cuk, scale = data["cu_query"], data["cu_kv"], data["softmax_scale"]
    heads = q.shape[1]
    indices = torch.arange(cache.shape[0], device=q.device, dtype=torch.int32)
    k, v = expand_context(cache, data["cache_scale"], indices, cuk, weight, heads, 128, 64, 128)
    base_quant = fused_qkv_per_tensor_quant(q, k, v)
    backend = make_backend(heads)
    backend.scale = scale
    chunk = SimpleNamespace(query_start_loc=cuq, cu_seq_lens=cuk,
                            max_query_len=data["max_query_len"], max_seq_len=data["max_seq_len"])
    historical = AiterFlashAttnPrefillBackend.run_prefill_context_chunk(backend, chunk, q, k, v)
    historical = tuple(x.double() for x in historical)
    original = reference(q, k, v, cuq, cuk, scale)
    params = projection_weight(quantize_layer(weight), heads)
    native_k, native_v = reference_projection(cache, data["cache_scale"], indices, params, heads)

    qhigh, qs = base_quant[0], base_quant[3]
    residual_scaled = (q[:, :, 128:].float() / qs - qhigh[:, :, 128:].float()) * 16
    if residual_scaled.abs().max().item() > 448:
        raise AssertionError("Q residual does not fit FP8")
    qlow = residual_scaled.to(torch.float8_e4m3fn)
    qwide = torch.cat((qhigh, qlow), -1).contiguous()
    reconstructed_qrope = (qhigh[:, :, 128:].double() + qlow.double() / 16) * qs.double()
    qerr = reconstructed_qrope - q[:, :, 128:].double()
    emit("ROPE_RESIDUAL_Q_ERROR", {
        "relative_rms": (qerr.square().mean().sqrt() /
                         q[:, :, 128:].double().square().mean().sqrt()).item(),
        "max_absolute": qerr.abs().max().item(),
    })

    def call(q8, k8, v8, qscale, kscale, vscale, lazy):
        return backend._fp8_attention(
            q8, k8, v8, softmax_scale=scale,
            cu_seqlens_q=cuq, cu_seqlens_kv=cuk,
            max_seqlen_q=data["max_query_len"], max_seqlen_kv=data["max_seq_len"],
            cross_seqlen=True, causal=False, return_lse=True,
            q_descale=qscale, k_descale=kscale, v_descale=vscale,
            dualwave_swp_lazy_rescale=lazy,
        )

    baseline_ms = kernel_ms(lambda: call(*base_quant[:6], True))
    results = []
    for label, refk, refv in (("bf16_weight", k, v), ("native_fp8_weight", native_k, native_v)):
        exponent = torch.ceil(torch.log2(refk.abs().max().double() /
                                       (448 * data["cache_scale"].double()))).item()
        ks = (data["cache_scale"] * (2.0 ** exponent)).reshape(1).contiguous()
        vs = (refv.abs().max().float() / 448).reshape(1)
        vs = torch.nextafter(vs, torch.full_like(vs, torch.inf))
        if label == "native_fp8_weight":
            k8, v8, _, _ = project(cache, data["cache_scale"], indices, params, heads,
                                   torch.cat((ks, vs)).contiguous())
        else:
            k8 = rocm_aiter_ops.per_tensor_quant(k, torch.float8_e4m3fn, ks)[0]
            v8 = rocm_aiter_ops.per_tensor_quant(v, torch.float8_e4m3fn, vs)[0]
        # This FP32 expression rescales compact FP8 codes by an exact binary factor.
        # It does not pass the primary RoPE values through a BF16 buffer.
        raw = cache[:, 512:].float() * (2.0 ** (-exponent))
        rope_high = raw.to(torch.float8_e4m3fn)
        rope_low = (raw / 16).to(torch.float8_e4m3fn)
        k8[:, :, 128:].copy_(rope_high[:, None, :].expand(-1, heads, -1))
        kwide = torch.cat((k8, rope_low[:, None, :].expand(-1, heads, -1)), -1).contiguous()
        rope_error = rope_high.double() * ks.double() - cache[:, 512:].double() * data["cache_scale"].double()
        emit("ROPE_CACHE_ALIGNMENT", {
            "projection": label, "scale_exponent": exponent, "primary_rope_max_error": rope_error.abs().max().item(),
            "primary_rope_exact": bool(torch.equal(rope_error, torch.zeros_like(rope_error))),
            "key_clipping_fraction": (refk.double().abs() > 448 * ks.double()).double().mean().item(),
            "low_rope_underflow_fraction": ((raw != 0) & (rope_low.float() == 0)).float().mean().item(),
        })
        for mode, qa, ka in (("cache_aligned_only", qhigh, k8), ("rope_residual_256", qwide, kwide)):
            deq = (qa.double() * qs.double(), ka.double() * ks.double(), v8.double() * vs.double())
            same_ref = reference(*deq, cuq, cuk, scale)
            quant_error = metrics(same_ref, original)
            for lazy in (True, False):
                invoke = lambda: call(qa, ka, v8, qs, ks, vs, lazy)
                actual = invoke()
                same, bf16 = metrics(actual, same_ref), metrics(actual, historical)
                record = {
                    "projection": label, "case": mode, "lazy_rescale": lazy,
                    "same_fp8_inputs": same, "vs_historical_bf16": bf16, "quantization_only": quant_error,
                    "passes_arithmetic_gate": (
                        same["normalized_max_output_error"] < 0.08 and same["min_cosine"] > 0.98
                        and same["normalized_max_lse_error"] < 1e-4),
                    "passes_original_gate": bf16["rms"] < 0.10 and bf16["lse_max"] < 0.20,
                    "fmha_ms": kernel_ms(invoke), "baseline_fmha_ms": baseline_ms,
                    "timing_scope": "FMHA only, excludes residual construction and projection",
                }
                results.append(record)
                emit("CAPTURED_ROPE_RESIDUAL_CASE", record)
    emit("CAPTURED_ROPE_RESIDUAL_COMPLETE", {
        "scope": "diagnostic native projection plus unchanged AITER FMHA; no serving integration",
        "softmax_scale": scale, "results": results,
    })


if __name__ == "__main__":
    run(os.environ["FAILURE_FIXTURE_PATH"])

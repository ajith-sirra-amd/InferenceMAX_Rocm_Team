"""Diagnostic per-head NoPE balancing for the FP8 residual-column path."""
import json
import os
from types import SimpleNamespace

import torch

from attention_reference import metrics, reference
from fp8_weight_preflight import quantize_layer
from inferencex_atom_attention.fp8_up_projection import (
    ProjectionWeight, project, projection_weight, reference_projection,
)
from inferencex_atom_attention.quantize import fused_qkv_per_tensor_quant
from preflight import make_backend
from rope_residual_diagnostics import emit, kernel_ms
from vllm.v1.attention.backends.mla.prefill.aiter_flash_attn import AiterFlashAttnPrefillBackend
from vllm.v1.attention.ops.rocm_aiter_mla_prefill import expand_context


def run(path):
    data = torch.load(path, map_location="cuda", weights_only=True)
    q, cache, weight = data["q"], data["cache"], data["weight"]
    cuq, cuk, scale = data["cu_query"], data["cu_kv"], data["softmax_scale"]
    heads = q.shape[1]
    indices = torch.arange(cache.shape[0], device=q.device, dtype=torch.int32)
    k, v = expand_context(cache, data["cache_scale"], indices, cuk, weight, heads, 128, 64, 128)
    base = fused_qkv_per_tensor_quant(q, k, v)
    backend = make_backend(heads)
    backend.scale = scale
    chunk = SimpleNamespace(query_start_loc=cuq, cu_seq_lens=cuk,
                            max_query_len=data["max_query_len"], max_seq_len=data["max_seq_len"])
    historical = AiterFlashAttnPrefillBackend.run_prefill_context_chunk(backend, chunk, q, k, v)
    historical = tuple(x.double() for x in historical)
    original = reference(q, k, v, cuq, cuk, scale)
    params = projection_weight(quantize_layer(weight), heads)
    native_k, native_v = reference_projection(cache, data["cache_scale"], indices, params, heads)
    qhigh, qs = base[0], base[3]
    low_q = ((q[:, :, 128:].float() / qs - qhigh[:, :, 128:].float()) * 16).to(torch.float8_e4m3fn)
    results = []
    for label, kr, vr in (("bf16_weight", k, v), ("native_fp8_weight", native_k, native_v)):
        exponent = torch.ceil(torch.log2(kr.abs().max().double() /
                                       (448 * data["cache_scale"].double()))).item()
        ks = (data["cache_scale"] * (2.0 ** exponent)).reshape(1).contiguous()
        vs = torch.nextafter((vr.abs().max().float() / 448).reshape(1),
                             torch.full((1,), torch.inf, device=q.device))
        a = q[:, :, :128].double().abs().amax(dim=(0, 2)).clamp_min(1e-12)
        b = kr[:, :, :128].double().abs().amax(dim=(0, 2)).clamp_min(1e-12)
        # Q' = Q/c, K' = cK. Equalize their normalized per-head NoPE peaks.
        balance = torch.sqrt(a * ks.double() / (b * qs.double())).float()
        q_balanced = q[:, :, :128].float() / balance[None, :, None]
        k_balanced_ref = kr[:, :, :128].double() * balance.double()[None, :, None]
        q_clip = (q_balanced.abs() > 448 * qs).float().mean().item()
        k_clip = (k_balanced_ref.abs() > 448 * ks.double()).double().mean().item()
        if q_clip or k_clip:
            raise RuntimeError(f"Balancing would clip: Q={q_clip}, K={k_clip}")
        qw = torch.cat(((q_balanced / qs).to(torch.float8_e4m3fn),
                         qhigh[:, :, 128:], low_q), -1).contiguous()
        if label == "native_fp8_weight":
            modified_scale = params.scale.view(heads, 256).clone()
            modified_scale[:, :128] *= balance[:, None]
            balanced_params = ProjectionWeight(params.weight, modified_scale.flatten().contiguous(),
                                                params.preshuffled, params.linear_kernel)
            k8, v8, _, _ = project(cache, data["cache_scale"], indices, balanced_params, heads,
                                   torch.cat((ks, vs)))
        else:
            k8 = torch.cat(((k_balanced_ref.float() / ks).clamp(-448, 448).to(torch.float8_e4m3fn),
                             torch.empty((k.shape[0], heads, 64), device=q.device, dtype=torch.float8_e4m3fn)), -1)
            v8 = (v.float() / vs).clamp(-448, 448).to(torch.float8_e4m3fn)
        raw_rope = cache[:, 512:].float() * (2.0 ** (-exponent))
        k8[:, :, 128:].copy_(raw_rope.to(torch.float8_e4m3fn)[:, None, :].expand(-1, heads, -1))
        kw = torch.cat((k8, (raw_rope / 16).to(torch.float8_e4m3fn)[:, None, :].expand(-1, heads, -1)), -1).contiguous()
        same_ref = reference(qw.double() * qs.double(), kw.double() * ks.double(),
                             v8.double() * vs.double(), cuq, cuk, scale)
        # The residual columns now model RoPE accurately; quantify the remaining
        # error with NoPE held unquantized, without pretending this is a kernel.
        q_exact_nope = qw.double() * qs.double()
        k_exact_nope = kw.double() * ks.double()
        q_exact_nope[:, :, :128] = q[:, :, :128].double()
        k_exact_nope[:, :, :128] = kr[:, :, :128].double()
        rope_only_ref = reference(q_exact_nope, k_exact_nope, v8.double() * vs.double(), cuq, cuk, scale)
        emit("RESIDUAL_NOPE_BALANCE_SETUP", {
            "projection": label, "balance_per_head": balance.tolist(),
            "q_clipping_fraction": q_clip, "k_clipping_fraction": k_clip,
            "quantization_only": metrics(same_ref, original),
            "with_nope_unquantized": metrics(rope_only_ref, original),
            "scope": "scale calibration uses captured K maxima; diagnostic only",
        })
        for lazy in (True, False):
            def invoke():
                return backend._fp8_attention(
                    qw, kw, v8, softmax_scale=scale,
                    cu_seqlens_q=cuq, cu_seqlens_kv=cuk,
                    max_seqlen_q=data["max_query_len"], max_seqlen_kv=data["max_seq_len"],
                    cross_seqlen=True, causal=False, return_lse=True,
                    q_descale=qs, k_descale=ks, v_descale=vs, dualwave_swp_lazy_rescale=lazy,
                )
            actual = invoke()
            same, bf16 = metrics(actual, same_ref), metrics(actual, historical)
            record = {"projection": label, "lazy_rescale": lazy,
                      "same_fp8_inputs": same, "vs_historical_bf16": bf16,
                      "passes_arithmetic_gate": same["normalized_max_output_error"] < 0.08
                      and same["min_cosine"] > 0.98 and same["normalized_max_lse_error"] < 1e-4,
                      "passes_original_gate": bf16["rms"] < 0.10 and bf16["lse_max"] < 0.20,
                      "fmha_ms": kernel_ms(invoke)}
            results.append(record)
            emit("CAPTURED_BALANCED_RESIDUAL_CASE", record)
    emit("CAPTURED_BALANCED_RESIDUAL_COMPLETE", {
        "scope": "diagnostic with captured maxima; no production scale selection, model accuracy or performance claim",
        "results": results,
    })


if __name__ == "__main__":
    run(os.environ["FAILURE_FIXTURE_PATH"])

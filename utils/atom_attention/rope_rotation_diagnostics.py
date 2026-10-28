"""Diagnostic orthogonal mixing of RoPE columns before FP8 FMHA.

Uses reference transforms to assess numerical potential. These transforms are
not serving kernels and their time is not a performance result.
"""
import json
import os
from types import SimpleNamespace

import torch

from attention_reference import metrics, reference
from fp8_weight_preflight import quantize_layer
from inferencex_atom_attention.fp8_up_projection import (
    project, projection_weight, reference_projection,
)
from inferencex_atom_attention.quantize import fused_qkv_per_tensor_quant
from preflight import make_backend
from vllm.v1.attention.backends.mla.prefill.aiter_flash_attn import AiterFlashAttnPrefillBackend
from vllm.v1.attention.ops.rocm_aiter_mla_prefill import expand_context


def emit(marker, value):
    print(marker, json.dumps(value, allow_nan=False), flush=True)


def rope_matrix(device, signed):
    h = torch.ones((1, 1), dtype=torch.float64, device=device)
    while h.shape[0] < 64:
        h = torch.cat((torch.cat((h, h), 1), torch.cat((h, -h), 1)), 0)
    h /= 8
    if signed:
        generator = torch.Generator().manual_seed(20261028)
        signs = (torch.randint(0, 2, (64,), generator=generator) * 2 - 1).to(device)
        h = signs[:, None] * h
    return h


def rotate_rope(tensor, matrix):
    return torch.cat((tensor[:, :, :128].double(),
                      tensor[:, :, 128:].double() @ matrix), -1)


def run(path):
    data = torch.load(path, map_location="cuda", weights_only=True)
    q, cache, weight = data["q"], data["cache"], data["weight"]
    cuq, cuk, scale = data["cu_query"], data["cu_kv"], data["softmax_scale"]
    heads = q.shape[1]
    indices = torch.arange(cache.shape[0], device=q.device, dtype=torch.int32)
    k, v = expand_context(cache, data["cache_scale"], indices, cuk, weight, heads, 128, 64, 128)
    backend = make_backend(heads)
    backend.scale = scale
    chunk = SimpleNamespace(query_start_loc=cuq, cu_seq_lens=cuk,
                            max_query_len=data["max_query_len"], max_seq_len=data["max_seq_len"])
    historical = AiterFlashAttnPrefillBackend.run_prefill_context_chunk(backend, chunk, q, k, v)
    historical = tuple(x.double() for x in historical)
    original = reference(q, k, v, cuq, cuk, scale)
    layer = quantize_layer(weight)
    params = projection_weight(layer, heads)
    native_k, native_v = reference_projection(cache, data["cache_scale"], indices, params, heads)
    results = []
    for signed in (False, True):
        matrix = rope_matrix(q.device, signed)
        qr, kr = rotate_rope(q, matrix), rotate_rope(k, matrix)
        identity = metrics(reference(qr, kr, v, cuq, cuk, scale), original)
        emit("ROPE_ROTATION_IDENTITY", {"signed": signed, "errors": identity})
        if identity["rms"] > 1e-10 or identity["lse_max"] > 1e-10:
            raise AssertionError("RoPE rotation did not preserve original attention")
        # Existing ATOM quantizer sees BF16 transformed inputs in this prototype.
        # This adds explicit transform rounding; production fusion is not implied.
        bf16_quant = fused_qkv_per_tensor_quant(qr.to(torch.bfloat16), kr.to(torch.bfloat16), v)
        native_kr = rotate_rope(native_k, matrix)
        scales = torch.stack([t.abs().max().float() / 448 for t in (native_kr, native_v)])
        scales = torch.nextafter(scales, torch.full_like(scales, torch.inf)).contiguous()
        kv8 = project(cache, data["cache_scale"], indices, params, heads, scales)
        # Keep real AITER FP8 x FP8 projection for NoPE/V; overwrite only RoPE
        # with a diagnostic FP32 rotation of the original compact FP8 cache.
        raw_rope = cache[:, 512:].float() * data["cache_scale"]
        rotated_rope = raw_rope @ matrix.float()
        rope8 = (rotated_rope / scales[0]).clamp(-448, 448).to(torch.float8_e4m3fn)
        kv8[0][:, :, 128:].copy_(rope8[:, None, :].expand(-1, heads, -1))
        native_quant = (bf16_quant[0], kv8[0], kv8[1],
                        bf16_quant[3], kv8[2], kv8[3])
        for label, quant in (("bf16_weight", bf16_quant), ("native_fp8_weight", native_quant)):
            deq = [x.double() * s.double() for x, s in zip(quant[:3], quant[3:6])]
            same_ref = reference(*deq, cuq, cuk, scale)
            quant_error = metrics(same_ref, original)
            for lazy in (True, False):
                actual = backend._fp8_attention(
                    *quant[:3], softmax_scale=scale,
                    cu_seqlens_q=cuq, cu_seqlens_kv=cuk,
                    max_seqlen_q=data["max_query_len"], max_seqlen_kv=data["max_seq_len"],
                    cross_seqlen=True, causal=False, return_lse=True,
                    q_descale=quant[3], k_descale=quant[4], v_descale=quant[5],
                    dualwave_swp_lazy_rescale=lazy,
                )
                same, bf16 = metrics(actual, same_ref), metrics(actual, historical)
                record = {
                    "projection": label, "signed": signed, "lazy_rescale": lazy,
                    "same_fp8_inputs": same, "vs_historical_bf16": bf16,
                    "quantization_only": quant_error,
                    "passes_arithmetic_gate": (
                        same["normalized_max_output_error"] < 0.08 and same["min_cosine"] > 0.98
                        and same["normalized_max_lse_error"] < 1e-4),
                    "passes_original_gate": bf16["rms"] < 0.10 and bf16["lse_max"] < 0.20,
                }
                results.append(record)
                emit("CAPTURED_ROPE_ROTATION_CASE", record)
    emit("CAPTURED_ROPE_ROTATION_COMPLETE", {
        "scope": "one captured chunk; unfused reference transform; not serving or performance validation",
        "native_projection_kernel": params.linear_kernel,
        "signed_seed": 20261028, "results": results,
    })


if __name__ == "__main__":
    run(os.environ["FAILURE_FIXTURE_PATH"])

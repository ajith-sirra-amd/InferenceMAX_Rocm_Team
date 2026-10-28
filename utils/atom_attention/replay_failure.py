"""Replay a saved Kimi projection/attention failure without loading model weights."""
import json
import os
from types import SimpleNamespace

import torch

from inferencex_atom_attention.numerical_debug import diagnose_lse
from inferencex_atom_attention.projection_fp8 import expand_context_fp8
from inferencex_atom_attention.quantize import fused_qkv_per_tensor_quant
from preflight import make_backend
from vllm._aiter_ops import rocm_aiter_ops
from vllm.v1.attention.backends.mla.prefill.aiter_flash_attn import AiterFlashAttnPrefillBackend
from vllm.v1.attention.ops.rocm_aiter_mla_prefill import expand_context


def run(path):
    data = torch.load(path, map_location="cuda", weights_only=True)
    q, cache, weight = data["q"], data["cache"], data["weight"]
    heads = q.shape[1]
    indices = torch.arange(cache.shape[0], device="cuda", dtype=torch.int32)
    chunk = SimpleNamespace(
        query_start_loc=data["cu_query"], cu_seq_lens=data["cu_kv"],
        max_query_len=data["max_query_len"], max_seq_len=data["max_seq_len"],
    )
    backend = make_backend(heads)
    backend.scale = data["softmax_scale"]
    k, v = expand_context(cache, data["cache_scale"], indices, data["cu_kv"],
                          weight, heads, 128, 64, 128)
    quant = fused_qkv_per_tensor_quant(q, k, v)
    supplied = expand_context_fp8(
        cache, data["cache_scale"], indices, weight, heads,
        mode="supplied_scales", output_scales=data["supplied_scales"],
    )
    for actual, bf16, scale in zip(supplied[:2], (k, v), supplied[2:]):
        reference = rocm_aiter_ops.per_tensor_quant(bf16, torch.float8_e4m3fn, scale)[0]
        torch.testing.assert_close(actual.view(torch.uint8), reference.view(torch.uint8),
                                   rtol=0, atol=0)
    bf16, bf16_lse = AiterFlashAttnPrefillBackend.run_prefill_context_chunk(
        backend, chunk, q, k, v,
    )
    dynamic, dynamic_lse = backend._call_projected_fp8(
        chunk, quant[0], quant[3], (quant[1], quant[2], quant[4], quant[5]),
    )
    original = backend._call_projected_fp8(chunk, quant[0], quant[3], supplied)
    diagnose_lse(chunk, backend.scale, q, k, v, quant, supplied,
                 original[0], original[1], bf16, bf16_lse, dynamic_lse)

    cases = [("supplied", original), ("dynamic", (dynamic, dynamic_lse))]
    # Existing AITER accuracy control: rebase online softmax on every tile.
    eager = backend._fp8_attention(
        quant[0], supplied[0], supplied[1], softmax_scale=backend.scale,
        cu_seqlens_q=chunk.query_start_loc, cu_seqlens_kv=chunk.cu_seq_lens,
        max_seqlen_q=chunk.max_query_len, max_seqlen_kv=chunk.max_seq_len,
        cross_seqlen=True, causal=False, return_lse=True,
        q_descale=quant[3], k_descale=supplied[2], v_descale=supplied[3],
        dualwave_swp_lazy_rescale=False,
    )
    cases.append(("supplied_eager_softmax_rescale", eager))
    # A rounding diagnostic, not a proposed change to ATOM's scale policy.
    pot_scales = [torch.exp2(torch.ceil(torch.log2(s))) for s in quant[3:6]]
    pot = [rocm_aiter_ops.per_tensor_quant(x, torch.float8_e4m3fn, s)[0]
           for x, s in zip((q, k, v), pot_scales)]
    cases.append(("power_of_two_scale_diagnostic", backend._call_projected_fp8(
        chunk, pot[0], pot_scales[0], (pot[1], pot[2], pot_scales[1], pot_scales[2]),
    )))
    records = []
    mask = torch.isfinite(bf16_lse)
    for name, (actual, lse) in cases:
        error = (lse[mask] - bf16_lse[mask]).abs()
        rms = ((actual.float() - bf16.float()).square().mean().sqrt()
               / bf16.float().square().mean().sqrt().clamp_min(1e-6)).item()
        lse_max = error.max().item()
        record = {"case": name, "rms_vs_bf16": rms, "lse_max_vs_bf16": lse_max,
                  "lse_p99_vs_bf16": torch.quantile(error, 0.99).item(),
                  "passes_original_gate": rms < 0.10 and lse_max < 0.20}
        records.append(record)
        print("CAPTURED_KIMI_CASE", json.dumps(record, allow_nan=False), flush=True)
    print("CAPTURED_KIMI_REPLAY_COMPLETE", json.dumps({
        "q_shape": list(q.shape), "cache_shape": list(cache.shape),
        "cases": records, "scope": "diagnostic; completion is not numerical acceptance",
    }), flush=True)


if __name__ == "__main__":
    run(os.environ["FAILURE_FIXTURE_PATH"])

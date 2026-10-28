"""Locate captured Q/K rounding error and test exact key-centering math.

The centering cases are diagnostic prototypes only. Production projection,
routing, and acceptance thresholds are not changed by this script.
"""
import json
import os
from types import SimpleNamespace

import torch

from attention_reference import metrics, reference
from inferencex_atom_attention.quantize import fused_qkv_per_tensor_quant
from preflight import make_backend
from vllm.v1.attention.ops.rocm_aiter_mla_prefill import expand_context
from vllm.v1.attention.backends.mla.prefill.aiter_flash_attn import AiterFlashAttnPrefillBackend


def emit(marker, value):
    print(marker, json.dumps(value, allow_nan=False), flush=True)


def centered_keys(q, k, cuq, cuk, scale):
    centered = torch.empty_like(k, dtype=torch.float64)
    correction = torch.zeros((q.shape[1], q.shape[0]), device=q.device, dtype=torch.float64)
    mean_norms = []
    for qa, qb, ka, kb in zip(cuq.tolist()[:-1], cuq.tolist()[1:],
                              cuk.tolist()[:-1], cuk.tolist()[1:]):
        if ka == kb:
            continue
        mean = k[ka:kb].double().mean(0)
        centered[ka:kb] = k[ka:kb].double() - mean
        correction[:, qa:qb] = ((q[qa:qb].double() * mean).sum(-1) * scale).t()
        mean_norms.append({"tokens": kb-ka, "mean_norm_per_head": mean.norm(dim=-1).tolist(),
                           "key_rms": k[ka:kb].double().square().mean().sqrt().item(),
                           "centered_key_rms": centered[ka:kb].square().mean().sqrt().item()})
    return centered, correction, mean_norms


def run(path):
    data = torch.load(path, map_location="cuda", weights_only=True)
    q, cache, weight = data["q"], data["cache"], data["weight"]
    cuq, cuk = data["cu_query"], data["cu_kv"]
    heads, scale = q.shape[1], data["softmax_scale"]
    indices = torch.arange(cache.shape[0], device=q.device, dtype=torch.int32)
    k, v = expand_context(cache, data["cache_scale"], indices, cuk, weight, heads, 128, 64, 128)
    quant = fused_qkv_per_tensor_quant(q, k, v)
    qd, kd, vd = [x.double() * s.double() for x, s in zip(quant[:3], quant[3:])]
    original = reference(q, k, v, cuq, cuk, scale)
    backend = make_backend(heads)
    backend.scale = scale
    chunk = SimpleNamespace(query_start_loc=cuq, cu_seq_lens=cuk,
                            max_query_len=data["max_query_len"], max_seq_len=data["max_seq_len"])
    historical = AiterFlashAttnPrefillBackend.run_prefill_context_chunk(backend, chunk, q, k, v)
    historical = tuple(x.double() for x in historical)
    for source_name, source, quantized in (("q", q, qd), ("k", k, kd)):
        for part, lo, hi in (("nope", 0, 128), ("rope", 128, 192)):
            mixed = source.double().clone()
            mixed[:, :, lo:hi] = quantized[:, :, lo:hi]
            inputs = (mixed, k, v) if source_name == "q" else (q, mixed, v)
            effect = reference(*inputs, cuq, cuk, scale)
            emit("CAPTURED_COMPONENT_ROUNDING", {
                "source": source_name, "part": part, "vs_bf16": metrics(effect, original),
            })
            del effect, mixed

    dynamic_ref = reference(qd, kd, vd, cuq, cuk, scale)
    lse_error = (dynamic_ref[1] - original[1]).abs()
    lse_error = torch.where(torch.isfinite(lse_error), lse_error, -1)
    worst = torch.topk(lse_error.flatten(), min(8, lse_error.numel())).indices.tolist()
    qbounds, kbounds = cuq.tolist(), cuk.tolist()
    for offset in worst:
        head, row = divmod(offset, q.shape[0])
        request = next(i for i in range(len(qbounds)-1) if qbounds[i] <= row < qbounds[i+1])
        ka, kb = kbounds[request:request+2]
        qr, kr = q[row, head].double(), k[ka:kb, head].double()
        qerror = qd[row, head] - qr
        kerror = kd[ka:kb, head] - kr
        logits = (kr @ qr) * scale
        changed = (kd[ka:kb, head] @ qd[row, head]) * scale
        peak = logits.argmax().item()
        pieces = {}
        for name, lo, hi in (("nope", 0, 128), ("rope", 128, 192)):
            pieces[name] = {
                "q_rounding_logit_at_peak": (kr[peak, lo:hi] @ qerror[lo:hi] * scale).item(),
                "k_rounding_logit_at_peak": (kerror[peak, lo:hi] @ qr[lo:hi] * scale).item(),
                "cross_term_at_peak": (kerror[peak, lo:hi] @ qerror[lo:hi] * scale).item(),
            }
        emit("CAPTURED_WORST_ROUNDING_ROW", {
            "query": row, "head": head, "request": request, "kv_tokens": kb-ka,
            "lse_error": lse_error[head, row].item(), "bf16_lse": original[1][head, row].item(),
            "peak_logit": logits[peak].item(), "quantized_peak_logit": changed[peak].item(),
            "top_key_probability": torch.softmax(logits, 0)[peak].item(), "components": pieces,
        })
    del dynamic_ref
    centered, correction, mean_norms = centered_keys(q, k, cuq, cuk, scale)
    invariant = reference(q, centered, v, cuq, cuk, scale)
    invariant = (invariant[0], invariant[1] + correction)
    invariant_metrics = metrics(invariant, original)
    emit("KEY_CENTERING_UNQUANTIZED_IDENTITY", {"errors": invariant_metrics,
                                             "requests": mean_norms})
    if invariant_metrics["rms"] > 1e-10 or invariant_metrics["lse_max"] > 1e-10:
        raise AssertionError("Key-centering identity failed")
    del invariant
    centered_quant = fused_qkv_per_tensor_quant(q, centered.to(torch.bfloat16), v)
    records = []
    for label, current, shift in (("dynamic_baseline", quant, torch.zeros_like(correction)),
                                  ("centered_keys_prototype", centered_quant, correction)):
        deq = [x.double() * s.double() for x, s in zip(current[:3], current[3:])]
        exact = reference(*deq, cuq, cuk, scale)
        exact = (exact[0], exact[1] + shift)
        for lazy in (True, False):
            output, lse = backend._fp8_attention(
                *current[:3], softmax_scale=scale, cu_seqlens_q=cuq, cu_seqlens_kv=cuk,
                max_seqlen_q=data["max_query_len"], max_seqlen_kv=data["max_seq_len"],
                cross_seqlen=True, causal=False, return_lse=True,
                q_descale=current[3], k_descale=current[4], v_descale=current[5],
                dualwave_swp_lazy_rescale=lazy,
            )
            actual = (output, lse.double() + shift)
            same, bf16 = metrics(actual, exact), metrics(actual, historical)
            record = {"case": label, "lazy_rescale": lazy,
                      "same_inputs": same, "vs_historical_bf16": bf16,
                      "passes_original_gate": bf16["rms"] < 0.10 and bf16["lse_max"] < 0.20}
            records.append(record)
            emit("CAPTURED_KEY_CENTERING_CASE", record)
    emit("CAPTURED_KEY_CENTERING_COMPLETE", {
        "scope": "diagnostic only; BF16 centering and FP64 LSE correction are not production kernels",
        "results": records,
    })


if __name__ == "__main__":
    run(os.environ["FAILURE_FIXTURE_PATH"])

# SPDX-License-Identifier: Apache-2.0
"""Failure-only diagnostics on the actual Kimi tensors; no tolerance changes."""
import json
import os

import torch


def diagnose_lse(chunk, scale, q, k, v, quant, kv8, actual, actual_lse,
                 bf16, bf16_lse, dynamic_lse):
    """Compare selected worst rows with independent FP64 softmax attention."""
    finite = torch.isfinite(actual_lse) & torch.isfinite(bf16_lse)
    error = torch.where(finite, (actual_lse - bf16_lse).abs(), -1.0)
    count = min(4, error.numel())
    positions = torch.topk(error.reshape(-1), count).indices.tolist()
    cuq = chunk.query_start_loc.cpu().tolist()
    cuk = chunk.cu_seq_lens.cpu().tolist()
    nqueries = q.shape[0]
    if actual_lse.shape != (q.shape[1], nqueries):
        raise ValueError(f"Unexpected LSE layout {actual_lse.shape}")
    rows = []
    for position in positions:
        head, query = divmod(position, nqueries)
        request = next(i for i in range(len(cuq) - 1) if cuq[i] <= query < cuq[i + 1])
        start, end = cuk[request], cuk[request + 1]
        q64 = q[query, head].double()
        k64 = k[start:end, head].double()
        v64 = v[start:end, head].double()
        qd = quant[0][query, head].double() * quant[3].double()
        kd = kv8[0][start:end, head].double() * kv8[2].double()
        vd = kv8[1][start:end, head].double() * kv8[3].double()
        k_dynamic = quant[1][start:end, head].double() * quant[4].double()
        score_sets = {
            "bf16_inputs": (k64 @ q64) * scale,
            "quantized_q_only": (k64 @ qd) * scale,
            "supplied_quantized_k_only": (kd @ q64) * scale,
            "supplied_quantized_qk": (kd @ qd) * scale,
            "dynamic_quantized_qk": (k_dynamic @ qd) * scale,
        }
        lses = {name: torch.logsumexp(scores, 0).item()
                for name, scores in score_sets.items()}
        ref_output = torch.softmax(score_sets["bf16_inputs"], 0) @ v64
        quant_output = torch.softmax(score_sets["supplied_quantized_qk"], 0) @ vd
        observed = actual[query, head].double()
        relative = lambda a, b: ((a - b).square().mean().sqrt()
                                / b.square().mean().sqrt().clamp_min(1e-12)).item()
        rows.append({
            "head": head, "query": query, "request": request, "kv_tokens": end - start,
            "reference_fp64_lse": lses,
            "actual_lse": actual_lse[head, query].item(),
            "bf16_fmha_lse": bf16_lse[head, query].item(),
            "dynamic_fp8_lse": dynamic_lse[head, query].item(),
            "fmha_lse_error_on_quantized_inputs":
                actual_lse[head, query].item() - lses["supplied_quantized_qk"],
            "bf16_lse_error_on_bf16_inputs":
                bf16_lse[head, query].item() - lses["bf16_inputs"],
            "row_output_rms_vs_fp64": relative(observed, ref_output),
            "row_fmha_rms_on_quantized_inputs": relative(observed, quant_output),
            "original_logit_min": score_sets["bf16_inputs"].min().item(),
            "original_logit_max": score_sets["bf16_inputs"].max().item(),
        })
    dynamic_error = (dynamic_lse[finite] - bf16_lse[finite]).abs()
    print("[ATOM-FP8-LSE-DIAGNOSTIC] " + json.dumps({
        "pid": os.getpid(),
        "q_shape": list(q.shape), "k_shape": list(k.shape), "v_shape": list(v.shape),
        "q_descale": quant[3].item(),
        "supplied_k_descale": kv8[2].item(), "dynamic_k_descale": quant[4].item(),
        "old_dynamic_lse_max_vs_bf16": dynamic_error.max().item(),
        "old_dynamic_lse_p99_vs_bf16": torch.quantile(dynamic_error, 0.99).item(),
        "rows": rows,
    }, allow_nan=False), flush=True)


_capture_done = False


def save_failure_input(q, gathered, cache_scale, row_indices, weight, chunk, scales, scale):
    """Retain one rank-zero synthetic-prefix fixture for isolated GPU experiments."""
    global _capture_done
    if _capture_done or os.environ.get("ATTENTION_PATH_PROBE_ONLY") != "true":
        return
    if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
        return
    from pathlib import Path

    destination = Path(os.environ["RESULT_DIR"]) / "atom_fp8_failure.pt"
    # Save only referenced compressed rows, not the full gather workspace.
    selected = gathered.view(torch.uint8)[row_indices.long()].contiguous().view(gathered.dtype)
    data = {
        "q": q.detach().contiguous().cpu(),
        "cache": selected.detach().cpu(),
        "cache_scale": cache_scale.detach().cpu(),
        "weight": weight.detach().cpu(),
        "supplied_scales": scales.detach().cpu(),
        "cu_query": chunk.query_start_loc.detach().cpu(),
        "cu_kv": chunk.cu_seq_lens.detach().cpu(),
        "max_query_len": chunk.max_query_len,
        "max_seq_len": chunk.max_seq_len,
        "softmax_scale": scale,
        "source": "Kimi controlled synthetic prefix; first rank-zero failing cached chunk",
    }
    torch.save(data, destination)
    _capture_done = True
    print(f"[ATOM-FP8-FAILURE-SAVED] path={destination} bytes={destination.stat().st_size}",
          flush=True)

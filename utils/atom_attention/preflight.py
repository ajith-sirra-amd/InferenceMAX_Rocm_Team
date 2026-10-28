"""Execute the vLLM adapter on small controlled inputs before loading Kimi."""
import json
import math
import os
from types import SimpleNamespace

import torch

from inferencex_atom_attention.backend import AtomFlyDSLPrefillBackend
from inferencex_atom_attention.register import register

register()


def make_backend(heads):
    return AtomFlyDSLPrefillBackend(
        num_heads=heads, scale=192 ** -0.5, kv_lora_rank=512,
        qk_nope_head_dim=128, qk_rope_head_dim=64, v_head_dim=128,
        vllm_config=None,
    )


def reference(q, k, v, qlens, klens, causal):
    # Independent float64 softmax attention, including empty KV rows.
    outputs, lses = [], []
    qi = ki = 0
    for nq, nk in zip(qlens, klens):
        qb = q[qi:qi+nq].double().transpose(0, 1)
        kb = k[ki:ki+nk].double().transpose(0, 1)
        vb = v[ki:ki+nk].double().transpose(0, 1)
        if nk == 0:
            outputs.append(torch.zeros((nq, q.shape[1], v.shape[2]), device=q.device))
            lses.append(torch.full((q.shape[1], nq), -math.inf, device=q.device))
        else:
            logits = (qb @ kb.transpose(-1, -2)) * (192 ** -0.5)
            if causal:
                allowed = torch.arange(nk, device=q.device)[None, :] <= (
                    torch.arange(nq, device=q.device)[:, None] + nk - nq
                )
                logits = logits.masked_fill(~allowed, -math.inf)
            outputs.append((torch.softmax(logits, -1) @ vb).transpose(0, 1).float())
            lses.append(torch.logsumexp(logits, -1).float())
        qi += nq
        ki += nk
    return torch.cat(outputs), torch.cat(lses, dim=1)


def check_output(actual, expected, lse, expected_lse, label):
    assert actual.dtype == torch.bfloat16
    assert lse.dtype == torch.float32
    assert tuple(actual.shape) == tuple(expected.shape)
    assert torch.isfinite(actual).all().item(), label
    assert torch.equal(torch.isneginf(lse), torch.isneginf(expected_lse)), label
    relative_rms = ((actual.float() - expected).square().mean().sqrt()
                    / expected.square().mean().sqrt().clamp_min(1e-6)).item()
    mask = torch.isfinite(expected_lse)
    lse_max_error = (lse[mask] - expected_lse[mask]).abs().max().item() if mask.any() else 0.0
    assert relative_rms < 0.10, (label, relative_rms)
    assert lse_max_error < 0.20, (label, lse_max_error)
    print("NUMERICAL_PASS", json.dumps({
        "case": label, "relative_rms_vs_unquantized_reference": relative_rms,
        "max_lse_error": lse_max_error,
    }), flush=True)


def controlled_cases():
    for heads, qlens, klens, causal in [
        (16, [17, 9], [17, 9], True),
        (24, [17, 9], [33, 0], False),
        (16, [1, 3], [0, 41], False),
    ]:
        q = torch.randn((sum(qlens), heads, 192), device="cuda", dtype=torch.bfloat16)
        k = torch.randn((sum(klens), heads, 192), device="cuda", dtype=torch.bfloat16)
        v = torch.randn((sum(klens), heads, 128), device="cuda", dtype=torch.bfloat16)
        cuq = torch.tensor([0, qlens[0], sum(qlens)], device="cuda", dtype=torch.int32)
        cuk = torch.tensor([0, klens[0], sum(klens)], device="cuda", dtype=torch.int32)
        backend = make_backend(heads)
        meta = SimpleNamespace(
            query_start_loc=cuq, max_query_len=max(qlens),
            cu_seq_lens=cuk, max_seq_len=max(klens),
        )
        backend.prepare_metadata(meta)
        def invoke():
            if causal:
                return backend.run_prefill_new_tokens(q, k, v, True)
            return backend.run_prefill_context_chunk(meta, q, k, v)
        actual, lse = invoke()
        torch.cuda.synchronize()
        expected, expected_lse = reference(q, k, v, qlens, klens, causal)
        label = f"heads={heads},q={qlens},kv={klens},causal={causal}"
        check_output(actual, expected, lse, expected_lse, label)
        if not causal:
            zero = sum(qlens[:klens.index(0)])
            count = qlens[klens.index(0)]
            assert torch.count_nonzero(actual[zero:zero+count]).item() == 0
        for _ in range(3):
            invoke()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured, captured_lse = invoke()
        q.zero_()
        graph.replay()
        torch.cuda.synchronize()
        expected, expected_lse = reference(q, k, v, qlens, klens, causal)
        check_output(captured, expected, captured_lse, expected_lse, "graph_replay:"+label)


def elapsed_ms(call):
    for _ in range(3):
        call()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(10):
        call()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / 10


def timing_cases():
    from aiter import flash_attn_varlen_func
    for nq, nk, causal in [(1024, 1024, True), (4096, 4096, True),
                           (256, 4096, False), (1024, 8192, False)]:
        heads = 16
        q = torch.randn((nq, heads, 192), device="cuda", dtype=torch.bfloat16)
        k = torch.randn((nk, heads, 192), device="cuda", dtype=torch.bfloat16)
        v = torch.randn((nk, heads, 128), device="cuda", dtype=torch.bfloat16)
        cuq = torch.tensor([0, nq], device="cuda", dtype=torch.int32)
        cuk = torch.tensor([0, nk], device="cuda", dtype=torch.int32)
        backend = make_backend(heads)
        meta = SimpleNamespace(query_start_loc=cuq, max_query_len=nq,
                               cu_seq_lens=cuk, max_seq_len=nk)
        backend.prepare_metadata(meta)
        def candidate():
            if causal:
                return backend.run_prefill_new_tokens(q, k, v, True)
            return backend.run_prefill_context_chunk(meta, q, k, v)
        def baseline():
            return flash_attn_varlen_func(
                q=q, k=k, v=v, cu_seqlens_q=cuq, cu_seqlens_k=cuk,
                max_seqlen_q=nq, max_seqlen_k=nk, softmax_scale=192**-0.5,
                causal=causal, return_lse=True,
            )
        bf16 = elapsed_ms(baseline)
        fp8 = elapsed_ms(candidate)
        print("TIMING", json.dumps({
            "q_tokens": nq, "kv_tokens": nk, "heads": heads, "causal": causal,
            "bf16_fmha_ms": bf16, "fp8_including_qkv_quant_ms": fp8,
            "bf16_over_fp8": bf16 / fp8,
            "scope": "attention plus adapter preparation; shared-host diagnostic",
        }), flush=True)


if __name__ == "__main__":
    torch.manual_seed(42)
    print("PREFLIGHT_GPU", torch.cuda.get_device_properties(0), flush=True)
    controlled_cases()
    timing_cases()
    print("ATOM_ATTENTION_PREFLIGHT_PASS", flush=True)

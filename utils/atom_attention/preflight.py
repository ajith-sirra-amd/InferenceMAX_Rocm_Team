"""Execute the vLLM adapter on small controlled inputs before loading Kimi."""
import json
import math
import os
import time
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
        (12, [17, 9], [17, 9], True),
        (24, [17, 9], [33, 0], False),
        (12, [1, 3], [0, 41], False),
        (12, [512, 9], [8192, 0], False),
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
        heads = 12
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
            "bf16_fmha_ms": bf16, "candidate_complete_call_ms": fp8,
            "bf16_over_candidate": bf16 / fp8,
            "candidate_path": "BF16" if causal or nq < 512 or nk < 8192 else "FlyDSL FP8 cached context",
            "scope": "attention plus adapter preparation; shared-host diagnostic",
        }), flush=True)


def query_reuse_case():
    heads, nq = 12, 521
    q = torch.randn((nq, heads, 384), device="cuda", dtype=torch.bfloat16)[:, :, ::2]
    k1 = torch.randn((8192, heads, 384), device="cuda", dtype=torch.bfloat16)[:, :, ::2]
    v1 = torch.randn((8192, heads, 256), device="cuda", dtype=torch.bfloat16)[:, :, ::2]
    k2 = torch.randn((8199, heads, 384), device="cuda", dtype=torch.bfloat16)[:, :, ::2]
    v2 = torch.randn((8199, heads, 256), device="cuda", dtype=torch.bfloat16)[:, :, ::2]
    cuq = torch.tensor([0, 512, 521], device="cuda", dtype=torch.int32)
    meta1 = SimpleNamespace(query_start_loc=cuq, max_query_len=512,
        cu_seq_lens=torch.tensor([0, 8192, 8192], device="cuda", dtype=torch.int32), max_seq_len=8192)
    meta2 = SimpleNamespace(query_start_loc=cuq, max_query_len=512,
        cu_seq_lens=torch.tensor([0, 8192, 8199], device="cuda", dtype=torch.int32), max_seq_len=8192)
    backend = make_backend(heads)
    def invoke():
        backend.begin_forward()
        try:
            first = backend.run_prefill_context_chunk(meta1, q, k1, v1)
            second = backend.run_prefill_context_chunk(meta2, q, k2, v2)
        finally:
            stats = backend.end_forward()
        return first, second, stats
    first, second, stats = invoke()
    torch.cuda.synchronize()
    assert stats == {"q_quantizations": 1, "q_reuses": 1}, stats
    for actual, k, v, lengths, label in [
        (first, k1, v1, [8192, 0], "reuse:first_strided_chunk"),
        (second, k2, v2, [8192, 7], "reuse:second_strided_chunk"),
    ]:
        expected, expected_lse = reference(q, k, v, [512, 9], lengths, False)
        check_output(actual[0], expected, actual[1], expected_lse, label)
    for _ in range(3):
        invoke()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        first, second, stats = invoke()
    q.zero_()
    graph.replay()
    torch.cuda.synchronize()
    for actual, k, v, lengths, label in [
        (first, k1, v1, [8192, 0], "reuse:changed_input_graph:first"),
        (second, k2, v2, [8192, 7], "reuse:changed_input_graph:second"),
    ]:
        expected, expected_lse = reference(q, k, v, [512, 9], lengths, False)
        check_output(actual[0], expected, actual[1], expected_lse, label)
    first, second, stats = invoke()
    expected, expected_lse = reference(q, k2, v2, [512, 9], [8192, 7], False)
    check_output(second[0], expected, second[1], expected_lse, "reuse:fresh_forward_zero_q")
    assert stats == {"q_quantizations": 1, "q_reuses": 1}, stats
    print("QUERY_REUSE_PASS", json.dumps(stats), flush=True)


def preparation_breakdown():
    from aiter import flash_attn_varlen_func
    from inferencex_atom_attention.quantize import fused_qkv_per_tensor_quant
    for nq, nk in [(256, 4096), (1024, 8192)]:
        heads = 12
        q = torch.randn((nq, heads, 192), device="cuda", dtype=torch.bfloat16)
        k = torch.randn((nk, heads, 192), device="cuda", dtype=torch.bfloat16)
        v = torch.randn((nk, heads, 128), device="cuda", dtype=torch.bfloat16)
        backend = make_backend(heads)
        cuq = torch.tensor([0, nq], device="cuda", dtype=torch.int32)
        cuk = torch.tensor([0, nk], device="cuda", dtype=torch.int32)
        meta = SimpleNamespace(query_start_loc=cuq, max_query_len=nq, cu_seq_lens=cuk, max_seq_len=nk)
        def baseline():
            return flash_attn_varlen_func(q=q, k=k, v=v, cu_seqlens_q=cuq,
                cu_seqlens_k=cuk, max_seqlen_q=nq, max_seqlen_k=nk,
                softmax_scale=192**-0.5, causal=False, return_lse=True)
        def pair():
            backend.begin_forward()
            try:
                backend.run_prefill_context_chunk(meta, q, k, v)
                return backend.run_prefill_context_chunk(meta, q, k, v)
            finally:
                backend.end_forward()
        q8, k8, v8, qs, ks, vs, _, _ = fused_qkv_per_tensor_quant(q, k, v)
        def fmha_only():
            return backend._fp8_attention(q8, k8, v8, cu_seqlens_q=cuq,
                cu_seqlens_kv=cuk, max_seqlen_q=nq, max_seqlen_kv=nk,
                softmax_scale=192**-0.5, causal=False, cross_seqlen=True,
                return_lse=True, q_descale=qs, k_descale=ks, v_descale=vs)
        def baseline_pair():
            baseline()
            return baseline()
        bf16_pair = elapsed_ms(baseline_pair)
        candidate_pair = elapsed_ms(pair)
        quant_ms = elapsed_ms(lambda: fused_qkv_per_tensor_quant(q, k, v))
        kv_only_ms = elapsed_ms(lambda: fused_qkv_per_tensor_quant(q, k, v, quantized_q=(q8, qs)))
        fmha_ms = elapsed_ms(fmha_only)
        torch.cuda.synchronize()
        started = time.perf_counter()
        for _ in range(10): pair()
        torch.cuda.synchronize()
        wall_ms = (time.perf_counter()-started)*100
        print("OVERHEAD_BREAKDOWN", json.dumps({
            "q_tokens": nq, "kv_tokens": nk, "heads": heads,
            "qkv_quant_ms": quant_ms, "kv_quant_with_reused_q_ms": kv_only_ms,
            "fp8_fmha_without_quant_ms": fmha_ms,
            "two_chunk_bf16_ms": bf16_pair, "two_chunk_candidate_ms": candidate_pair,
            "two_chunk_candidate_host_wall_ms": wall_ms,
            "pair_speed_ratio": bf16_pair/candidate_pair,
            "q_reuse_logical_bytes_saved_per_extra_chunk": 5*q.numel(),
            "fp8_q_retained_bytes": q.numel(),
            "expanded_bf16_kv_bytes_per_chunk": 2*(k.numel()+v.numel()),
            "fp8_kv_bytes_per_chunk": k.numel()+v.numel(),
            "traffic_note": "Logical tensor traffic estimate; no measured HBM counters.",
            "scope": "single GPU shared-host kernel diagnostic; excludes KV expansion and collectives",
        }), flush=True)


if __name__ == "__main__":
    torch.manual_seed(42)
    print("PREFLIGHT_GPU", torch.cuda.get_device_properties(0), flush=True)
    controlled_cases()
    query_reuse_case()
    timing_cases()
    preparation_breakdown()
    print("ATOM_ATTENTION_PREFLIGHT_PASS", flush=True)

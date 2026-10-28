"""Compare projection epilogues and FP8 FMHA on the pinned MI355X image.

This is a kernel experiment, not a model-accuracy or serving benchmark.
The serving plugin does not import or select these projection candidates.
"""
import json
import math
import os
import statistics
import time

import torch

from inferencex_atom_attention.projection_fp8 import expand_context_fp8
from inferencex_atom_attention.quantize import fused_qkv_per_tensor_quant
from preflight import make_backend, reference
from vllm.v1.attention.ops.rocm_aiter_mla_prefill import expand_context
from aiter import flash_attn_varlen_func

MODES = ("fused_amax", "recompute", "supplied_scales")


def emit(kind, **fields):
    print(kind, json.dumps(fields, allow_nan=False), flush=True)


def error(actual, expected):
    actual, expected = actual.float(), expected.float()
    if not torch.isfinite(actual).all() or not torch.isfinite(expected).all():
        raise AssertionError("nonfinite output")
    if actual.numel() == 0:
        return {"relative_rms": 0.0, "max_abs": 0.0}
    diff = actual - expected
    return {
        "relative_rms": (diff.square().mean().sqrt()
                         / expected.square().mean().sqrt().clamp_min(1e-8)).item(),
        "max_abs": diff.abs().max().item(),
    }


def fixture(tokens, heads=12, index_dtype=torch.int32):
    cache = (torch.randn((max(tokens, 1) + 19, 576), device="cuda") * 8).to(
        torch.float8_e4m3fn
    )
    weight = (torch.randn((heads * 256, 512), device="cuda") / math.sqrt(512)).to(
        torch.bfloat16
    )
    # Non-monotonic repeated indices exercise actual gather/reorganization.
    indices = torch.randint(0, max(tokens, 1), (tokens,), device="cuda",
                            dtype=index_dtype)
    scale = torch.tensor([0.125], device="cuda", dtype=torch.float32)
    cu = torch.tensor([0, tokens], device="cuda", dtype=torch.int32)
    return cache, scale, indices, cu, weight, heads


def native(data):
    cache, scale, indices, cu, weight, heads = data
    return expand_context(cache, scale, indices, cu, weight, heads, 128, 64, 128)


def candidate(data, mode, scales):
    cache, scale, indices, _, weight, heads = data
    return expand_context_fp8(
        cache, scale, indices, weight, heads, mode=mode,
        output_scales=scales if mode == "supplied_scales" else None,
    )


def baseline_quant(q, k, v, q8=None):
    result = fused_qkv_per_tensor_quant(q, k, v, quantized_q=q8)
    return result[1], result[2], result[4], result[5]


def exact_dynamic(actual, expected, label):
    for i, (a, e) in enumerate(zip(actual, expected)):
        # Bit views also check FP8 sign/encoding, not just dequantized values.
        equal = torch.equal(a.view(torch.uint8), e.view(torch.uint8))
        if not equal:
            raise AssertionError((label, i, error(a.float(), e.float())))
    emit("PROJECTION_EXACT_PASS", case=label)


def supplied_reference(k, v, scales):
    # Independent Torch conversion, rather than invoking candidate helpers.
    return tuple(
        (x.float() * torch.reciprocal(s)).clamp(-448, 448).to(torch.float8_e4m3fn)
        for x, s in zip((k, v), (scales[:1], scales[1:]))
    )


def correctness_cases():
    for tokens, heads, index_dtype in [
        (0, 12, torch.int32), (1, 12, torch.int32),
        (33, 12, torch.int64), (257, 24, torch.int32),
    ]:
        data = fixture(tokens, heads, index_dtype)
        q = torch.randn((17, heads, 192), device="cuda", dtype=torch.bfloat16)
        k, v = native(data)
        expected = baseline_quant(q, k, v)
        supplied = torch.tensor([0.03125, 0.0625], device="cuda")
        for mode in MODES:
            actual = candidate(data, mode, supplied)
            if mode != "supplied_scales":
                exact_dynamic(actual, expected, f"{mode}:tokens={tokens}:heads={heads}")
            else:
                k_ref, v_ref = supplied_reference(k, v, supplied)
                torch.testing.assert_close(actual[0].float(), k_ref.float(), rtol=0, atol=0)
                torch.testing.assert_close(actual[1].float(), v_ref.float(), rtol=0, atol=0)
                torch.testing.assert_close(actual[2], supplied[:1], rtol=0, atol=0)
                torch.testing.assert_close(actual[3], supplied[1:], rtol=0, atol=0)
                emit("PROJECTION_SUPPLIED_PASS", tokens=tokens, heads=heads)

    # Hand-worked projection: latent=2, weight=1/512, cache scale=1/2:
    # each NoPE K and V is 1; RoPE cache=4 becomes 2.
    data = fixture(33)
    cache, scale, indices, cu, weight, heads = data
    values = torch.full(cache.shape, 2.0, device="cuda")
    values[:, 512:] = 4.0
    cache.copy_(values.to(cache.dtype))
    scale.fill_(0.5)
    weight.fill_(1.0 / 512)
    supplied = torch.tensor([1.0 / 16, 1.0 / 32], device="cuda")
    actual = candidate(data, "supplied_scales", supplied)
    assert torch.all(actual[0][:, :, :128].float() == 16).item()
    assert torch.all(actual[0][:, :, 128:].float() == 32).item()
    assert torch.all(actual[1].float() == 32).item()
    emit("PROJECTION_HAND_REFERENCE_PASS")

    # All-zero values must overwrite every partial and produce positive scales.
    cache.zero_()
    for mode in MODES[:2]:
        actual = candidate(data, mode, supplied)
        assert not torch.count_nonzero(actual[0].float()).item()
        assert not torch.count_nonzero(actual[1].float()).item()
        torch.testing.assert_close(actual[2], torch.full_like(actual[2], 1e-6))
        torch.testing.assert_close(actual[3], torch.full_like(actual[3], 1e-6))
    emit("PROJECTION_ZERO_PASS")


def graph_case():
    data = fixture(257)
    supplied = torch.tensor([0.03125, 0.0625], device="cuda")
    q = torch.randn((17, 12, 192), device="cuda", dtype=torch.bfloat16)
    for mode in MODES:
        for _ in range(3):
            candidate(data, mode, supplied)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = candidate(data, mode, supplied)
        # Reuse storage with new data/scales; graph must not retain old values.
        data[0].copy_((torch.randn(data[0].shape, device="cuda") * 4).to(data[0].dtype))
        supplied.mul_(1.25)
        graph.replay()
        torch.cuda.synchronize()
        k, v = native(data)
        if mode == "supplied_scales":
            expected = supplied_reference(k, v, supplied)
            for a, e in zip(actual[:2], expected):
                torch.testing.assert_close(a.float(), e.float(), rtol=0, atol=0)
        else:
            exact_dynamic(actual, baseline_quant(q, k, v), "graph:" + mode)
        emit("PROJECTION_GRAPH_PASS", mode=mode)


def attention_case(nq=512, nk=8192, stress=False):
    data = fixture(nk)
    q = torch.randn((nq, 12, 384), device="cuda", dtype=torch.bfloat16)[:, :, ::2]
    # Separate new-token K/V supply ATOM's scales, not a slice of the cached data.
    new_data = fixture(nq)
    new_data = (*new_data[:4], data[4], data[5])
    knew, vnew = native(new_data)
    prepared = fused_qkv_per_tensor_quant(q, knew, vnew)
    q8, qs = prepared[0], prepared[3]
    supplied = torch.cat((prepared[6], prepared[7]))
    if stress:
        # Deliberately violate the assumption that new-token range represents
        # cached range. This is a clipping diagnostic, not a passing accuracy case.
        data[0].copy_((data[0].float() * 4).clamp(-448, 448).to(data[0].dtype))
    k, v = native(data)
    dynamic = baseline_quant(q, k, v, (q8, qs))
    cuq = torch.tensor([0, nq], device="cuda", dtype=torch.int32)
    cuk = torch.tensor([0, nk], device="cuda", dtype=torch.int32)
    backend = make_backend(12)
    if not backend._fp8_enabled:
        raise AssertionError("ATOM_USE_FLYDSL_FP8_PREFILL_ATTN=1 is required")
    from types import SimpleNamespace
    chunk = SimpleNamespace(query_start_loc=cuq, max_query_len=nq,
                            cu_seq_lens=cuk, max_seq_len=nk)

    def fmha(kv):
        k8, v8, ks, vs = kv
        return backend._fp8_attention(
            q8, k8, v8, cu_seqlens_q=cuq, cu_seqlens_kv=cuk,
            max_seqlen_q=nq, max_seqlen_kv=nk, softmax_scale=192**-0.5,
            causal=False, cross_seqlen=True, return_lse=True,
            q_descale=qs, k_descale=ks, v_descale=vs,
        )

    bf16, bf16_lse = flash_attn_varlen_func(
        q=q, k=k, v=v, cu_seqlens_q=cuq, cu_seqlens_k=cuk,
        max_seqlen_q=nq, max_seqlen_k=nk, softmax_scale=192**-0.5,
        causal=False, return_lse=True,
    )
    old_out, old_lse = fmha(dynamic)
    if not stress:
        # Run the real plugin entry point and prove its quantization path executed.
        backend.begin_forward()
        try:
            routed, routed_lse = backend.run_prefill_context_chunk(chunk, q, k, v)
        finally:
            stats = backend.end_forward()
        assert stats["q_quantizations"] == 1, stats
        torch.testing.assert_close(routed, old_out, rtol=0, atol=0)
        torch.testing.assert_close(routed_lse, old_lse, rtol=0, atol=0)
        emit("FP8_ROUTE_PASS", q_tokens=nq, kv_tokens=nk, **stats)
        ref, ref_lse = reference(q, k, v, [nq], [nk], False)
        bf16_error = error(bf16, ref)
        dynamic_error = error(old_out, ref)
        dynamic_lse = error(old_lse, ref_lse)
        assert dynamic_error["relative_rms"] < 0.10, dynamic_error
        assert dynamic_lse["max_abs"] < 0.20, dynamic_lse
        # Independent attention on the *dequantized FP8 inputs* separates FMHA
        # arithmetic error from the Q/K/V conversion error.
        qdq = q8.float() * qs
        kdq = dynamic[0].float() * dynamic[2]
        vdq = dynamic[1].float() * dynamic[3]
        ref8, ref8_lse = reference(qdq, kdq, vdq, [nq], [nk], False)
        fmha_error = error(old_out, ref8)
        fmha_lse_error = error(old_lse, ref8_lse)
        assert fmha_error["relative_rms"] < 0.10, fmha_error
        assert fmha_lse_error["max_abs"] < 0.20, fmha_lse_error
        emit("FP8_FMHA_ACCURACY", q_tokens=nq, kv_tokens=nk,
             bf16_vs_fp64=bf16_error, fp8_vs_fp64=dynamic_error,
             fp8_vs_bf16=error(old_out, bf16), fp8_fmha_vs_dequantized_fp64=fmha_error,
             fp8_lse_vs_fp64=dynamic_lse, fp8_lse_vs_dequantized_fp64=fmha_lse_error,
             rms_limit=0.10, lse_abs_limit=0.20,
             scope="synthetic kernel numerical check; not model accuracy")
    for mode in (("supplied_scales",) if stress else MODES):
        kv = candidate(data, mode, supplied)
        out, lse = fmha(kv)
        if mode != "supplied_scales":
            exact_dynamic(kv, dynamic, f"{mode}:long_context")
            torch.testing.assert_close(out, old_out, rtol=0, atol=0)
            torch.testing.assert_close(lse, old_lse, rtol=0, atol=0)
        else:
            for a, e in zip(kv[:2], supplied_reference(k, v, supplied)):
                torch.testing.assert_close(a.float(), e.float(), rtol=0, atol=0)
        clipping = {
            name: (x.float().abs() > 448 * s).float().mean().item()
            for name, x, s in zip(("k", "v"), (k, v), (kv[2], kv[3]))
        }
        vs_bf16 = error(out, bf16)
        emit("PROJECTION_ATTENTION", mode=mode, stress=stress,
             q_tokens=nq, kv_tokens=nk, vs_dynamic=error(out, old_out),
             vs_bf16=vs_bf16, lse_vs_dynamic=error(lse, old_lse),
             clipping_fraction=clipping, scales=[kv[2].item(), kv[3].item()],
             numerical_gate_pass=vs_bf16["relative_rms"] < 0.10,
             diagnostic_only=stress)
        if not stress:
            assert vs_bf16["relative_rms"] < 0.10, (mode, vs_bf16)


def graph_timing(call, iterations=20):
    for _ in range(3):
        call()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(iterations):
            call()
    graph.replay()
    torch.cuda.synchronize()
    values = []
    for _ in range(5):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        values.append(start.elapsed_time(end) / iterations)
    return {"median_ms": statistics.median(values), "samples_ms": values}


def timing_cases():
    for tokens in (8192, 16384, 65536):
        data = fixture(tokens)
        q = torch.randn((512, 12, 192), device="cuda", dtype=torch.bfloat16)
        new_data = fixture(512)
        new_data = (*new_data[:4], data[4], data[5])
        knew, vnew = native(new_data)
        prepared = fused_qkv_per_tensor_quant(q, knew, vnew)
        qpair = (prepared[0], prepared[3])
        scales = torch.cat((prepared[6], prepared[7]))
        def baseline():
            k, v = native(data)
            return baseline_quant(q, k, v, qpair)
        results = {"native_projection_plus_dynamic_quant": graph_timing(baseline)}
        for mode in MODES:
            results[mode] = graph_timing(lambda: candidate(data, mode, scales))
        # Include the entire actual ATOM Q/new-KV quantizer as a conservative
        # per-forward preparation charge. It also quantizes new K/V, which our
        # BF16-causal policy would not consume.
        preparation = graph_timing(lambda: fused_qkv_per_tensor_quant(q, knew, vnew))
        base_ms = results["native_projection_plus_dynamic_quant"]["median_ms"]
        for result in results.values():
            result["speed_ratio"] = base_ms / result["median_ms"]
        emit("PROJECTION_TIMING", kv_tokens=tokens, heads=12, results=results,
             atom_scale_preparation=preparation,
             supplied_scales_total_one_chunk_ms=results["supplied_scales"]["median_ms"]
                 + preparation["median_ms"],
             supplied_scales_total_four_chunks_ms=4 * results["supplied_scales"]["median_ms"]
                 + preparation["median_ms"],
             expanded_bf16_bytes=2 * tokens * 12 * (192 + 128),
             scope="single GPU graph kernel timing; excludes DCP, model and serving")


if __name__ == "__main__":
    torch.manual_seed(42)
    if os.environ.get("ATOM_USE_FLYDSL_FP8_PREFILL_ATTN") != "1":
        raise RuntimeError("Set ATOM_USE_FLYDSL_FP8_PREFILL_ATTN=1")
    emit("PROJECTION_EXPERIMENT_START",
         gpu=str(torch.cuda.get_device_properties(0)),
         torch=torch.__version__, hip=torch.version.hip,
         atom_reference="773eb36254e43060bfd5fabcab7d3840bf591358",
         aiter_projection_reference="bbef5a6f2d3733b293c80d2cee6a8d464e88bc01",
         bf16_projection=True, serving_enabled=False)
    started = time.monotonic()
    correctness_cases()
    graph_case()
    attention_case()
    attention_case(stress=True)
    timing_cases()
    emit("PROJECTION_EXPERIMENT_PASS", elapsed_seconds=time.monotonic() - started)

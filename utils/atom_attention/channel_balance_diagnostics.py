"""Compare per-head and per-channel NoPE balancing on a captured Kimi chunk.

Diagnostic only: calibration reads captured K maxima. This does not implement
serving calibration, validate model accuracy, or change the native PR.
"""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from aiter import flash_attn_varlen_func
from aiter.ops.flydsl import flydsl_flash_attn_fp8_func
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.kernels.linear import init_fp8_linear_kernel
from vllm.model_executor.kernels.linear.scaled_mm.aiter import (
    AiterPerTokenFp8ScaledMMLinearKernel,
    AiterPreshuffledPerTokenFp8ScaledMMLinearKernel,
)
from vllm.model_executor.layers.quantization.online.fp8 import Fp8PtpcOnlineLinearMethod
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kFp8DynamicTokenSym, kFp8StaticChannelSym,
)
from vllm.v1.attention.backends.mla.prefill.aiter_flash_attn import AiterFlashAttnPrefillBackend
from vllm.v1.attention.ops.rocm_aiter_mla_fp8_prefill import (
    ProjectionWeight, project, projection_weight,
)
from vllm.v1.attention.ops.rocm_aiter_mla_fp8_quant import fused_qkv_per_tensor_quant
from vllm.v1.attention.ops.rocm_aiter_mla_prefill import expand_context

def quantize_layer(weight, packing_kernel=None):
    # Supply an already-loaded layer; execute native online quantization and packing.
    # No Kimi model is constructed for this isolated test.
    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(weight.clone(), requires_grad=False)
    layer.tp_size = 1
    layer.input_size_per_partition = weight.shape[1]
    layer.bias = None
    method = Fp8PtpcOnlineLinearMethod.__new__(Fp8PtpcOnlineLinearMethod)
    method.requantization_source = None
    method.requantization_source_parameters = {}
    method.input_dtype = method.out_dtype = torch.bfloat16
    if packing_kernel is None:
        with set_current_vllm_config(VllmConfig()):
            method.fp8_linear = init_fp8_linear_kernel(
                activation_quant_key=kFp8DynamicTokenSym,
                weight_quant_key=kFp8StaticChannelSym, weight_shape=weight.shape,
                input_dtype=torch.bfloat16, out_dtype=torch.bfloat16,
                module_name="FP8ProjectionPreflight",
            )
    else:
        # Exercise each native packing implementation, including layouts that the
        # current tuning table does not select. No GEMM is invoked on these objects.
        kernel = packing_kernel.__new__(packing_kernel)
        kernel.layer_param_names = ("weight", "weight_scale", "input_scale", "input_scale_ub")
        kernel.config = SimpleNamespace(activation_quant_key=kFp8DynamicTokenSym)
        method.fp8_linear = kernel
    layer.quant_method = method
    method.process_weights_after_loading(layer)
    return layer


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



def emit(name, value):
    print(name, json.dumps(value, allow_nan=False), flush=True)


def run(path, report):
    torch.cuda.set_device(0)
    actual_blas = str(torch.backends.cuda.preferred_blas_library())
    if actual_blas != "_BlasBackend.Cublas":
        raise RuntimeError(f"Unexpected BLAS backend: {actual_blas}")
    data = torch.load(path, map_location="cuda", weights_only=True)
    q, cache, weight = data["q"], data["cache"], data["weight"]
    cuq, cuk, scale = data["cu_query"], data["cu_kv"], data["softmax_scale"]
    heads = q.shape[1]
    if q.shape[-1] != 192 or cache.shape[-1] != 576:
        raise ValueError("Expected captured 128-NoPE + 64-RoPE MLA dimensions")
    rows = torch.arange(cache.shape[0], device=q.device, dtype=torch.int32)
    k, v = expand_context(
        cache, data["cache_scale"], rows, cuk, weight, heads, 128, 64, 128,
    )
    backend = SimpleNamespace(scale=scale, flash_attn_varlen_func=flash_attn_varlen_func)
    chunk = SimpleNamespace(
        query_start_loc=cuq, cu_seq_lens=cuk,
        max_query_len=data["max_query_len"], max_seq_len=data["max_seq_len"],
    )
    historical = AiterFlashAttnPrefillBackend.run_prefill_context_chunk(
        backend, chunk, q, k, v,
    )
    historical = tuple(t.double() for t in historical)
    original = reference(q, k, v, cuq, cuk, scale)
    report.update(
        fixture_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        queries=q.shape[0], heads=heads, kv_rows=cache.shape[0],
        blas_backend=actual_blas,
        bf16_arithmetic=metrics(historical, original),
        scope="All captured rows; GPU0; native FP8 weights/projector; existing 256-dim FMHA. Diagnostic calibration uses captured K maxima.",
    )
    plain = projection_weight(
        quantize_layer(weight, AiterPerTokenFp8ScaledMMLinearKernel), heads,
    )
    packed = projection_weight(
        quantize_layer(weight, AiterPreshuffledPerTokenFp8ScaledMMLinearKernel), heads,
    )
    torch.testing.assert_close(plain.scale, packed.scale, rtol=0, atol=0)
    raw = cache.double() * data["cache_scale"].double()
    decoded_weight = plain.weight.double() * plain.scale.double()[:, None]
    expanded = (raw[:, :512] @ decoded_weight.t()).view(-1, heads, 256)
    native_k = torch.cat(
        (expanded[:, :, :128], raw[:, None, 512:].expand(-1, heads, -1)), -1,
    )
    native_v = expanded[:, :, 128:]
    del expanded, decoded_weight, raw
    base = fused_qkv_per_tensor_quant(q, k, v)
    qhigh, qs = base[0], base[3]
    low_scaled = (q[:, :, 128:].float() / qs - qhigh[:, :, 128:].float()) * 16
    if low_scaled.abs().max().item() > 448:
        raise AssertionError("RoPE residual does not fit FP8")
    qlow = low_scaled.to(torch.float8_e4m3fn)
    exponent = int(torch.ceil(torch.log2(
        native_k.abs().max() / (448 * data["cache_scale"].double())
    )).item())
    ks = (data["cache_scale"] * (2.0 ** exponent)).reshape(1).contiguous()
    vs = (native_v.abs().max().float() / 448).reshape(1)
    vs = torch.nextafter(vs, torch.full_like(vs, torch.inf))
    raw_rope = cache[:, 512:].float() * (2.0 ** (-exponent))
    rope_high = raw_rope.to(torch.float8_e4m3fn)
    rope_low = (raw_rope / 16).to(torch.float8_e4m3fn)
    rope_error = (
        rope_high.double() * ks.double()
        - cache[:, 512:].double() * data["cache_scale"].double()
    )
    if torch.count_nonzero(rope_error).item():
        raise AssertionError("Primary cached RoPE was not preserved exactly")
    candidates = []
    for mode in ("per_head", "per_channel"):
        reduction = (0, 2) if mode == "per_head" else 0
        a = q[:, :, :128].double().abs().amax(dim=reduction).clamp_min(1e-12)
        b = native_k[:, :, :128].abs().amax(dim=reduction).clamp_min(1e-12)
        balance = torch.sqrt(a * ks.double() / (b * qs.double())).float()
        if mode == "per_head":
            balance = balance[:, None].expand(heads, 128)
        q_nope = q[:, :, :128].float() / balance[None, :, :]
        k_nope_ref = native_k[:, :, :128] * balance.double()[None, :, :]
        q_clip = (q_nope.abs() > 448 * qs).float().mean().item()
        k_clip = (k_nope_ref.abs() > 448 * ks.double()).double().mean().item()
        if q_clip or k_clip:
            raise AssertionError(f"{mode} clips Q={q_clip}, K={k_clip}")
        qw = torch.cat(
            ((q_nope / qs).to(torch.float8_e4m3fn), qhigh[:, :, 128:], qlow),
            -1,
        ).contiguous()
        for layout, params in (("plain", plain), ("preshuffled", packed)):
            modified = params.scale.view(heads, 256).clone()
            modified[:, :128] *= balance
            adjusted = ProjectionWeight(
                params.weight, modified.flatten().contiguous(), params.preshuffled,
            )
            k8, v8, _, _ = project(
                cache, data["cache_scale"], rows, adjusted, heads,
                torch.cat((ks, vs)).contiguous(),
            )
            if k8.dtype != torch.float8_e4m3fn or v8.dtype != torch.float8_e4m3fn:
                raise AssertionError("Native projection did not return FP8")
            k8[:, :, 128:].copy_(rope_high[:, None, :].expand(-1, heads, -1))
            kw = torch.cat(
                (k8, rope_low[:, None, :].expand(-1, heads, -1)), -1,
            ).contiguous()
            expected_k = torch.cat(
                (k_nope_ref, native_k[:, :, 128:]), -1,
            )
            projection_checks = {}
            for label, actual, expected, descale in (
                ("k", k8, expected_k, ks), ("v", v8, native_v, vs),
            ):
                diff = actual.double() * descale.double() - expected
                rms = (
                    diff.square().mean().sqrt()
                    / expected.square().mean().sqrt().clamp_min(1e-12)
                ).item()
                peak = (diff.abs().max() / expected.abs().max().clamp_min(1e-12)).item()
                clipping = (expected.abs() > 448 * descale.double()).double().mean().item()
                projection_checks[label] = dict(rms=rms, peak=peak, clipping=clipping)
                if not (rms < 0.04 and peak < 0.07 and clipping == 0):
                    raise AssertionError(f"Projection arithmetic failed: {projection_checks}")
            same_ref = reference(
                qw.double() * qs.double(), kw.double() * ks.double(),
                v8.double() * vs.double(), cuq, cuk, scale,
            )
            for lazy in (True, False):
                actual = flydsl_flash_attn_fp8_func(
                    qw, kw, v8, softmax_scale=scale,
                    cu_seqlens_q=cuq, cu_seqlens_kv=cuk,
                    max_seqlen_q=data["max_query_len"], max_seqlen_kv=data["max_seq_len"],
                    cross_seqlen=True, causal=False, return_lse=True,
                    q_descale=qs, k_descale=ks, v_descale=vs,
                    dualwave_swp_lazy_rescale=lazy,
                )
                same = metrics(actual, same_ref)
                bf16 = metrics(actual, historical)
                arithmetic_pass = (
                    same["normalized_max_output_error"] < 0.08
                    and same["min_cosine"] > 0.98
                    and same["normalized_max_lse_error"] < 1e-4
                )
                original_pass = bf16["rms"] < 0.10 and bf16["lse_max"] < 0.20
                record = dict(
                    balance_mode=mode, layout=layout, lazy_rescale=lazy,
                    role="candidate" if mode == "per_channel" else "prior_control",
                    balance_min=balance.min().item(), balance_max=balance.max().item(),
                    projection=projection_checks, same_fp8_inputs=same,
                    vs_historical_bf16=bf16,
                    passes_arithmetic_gate=arithmetic_pass,
                    passes_original_gate=original_pass,
                )
                report["cases"].append(record)
                emit("CHANNEL_BALANCE_CASE", record)
                if mode == "per_channel":
                    candidates.append(arithmetic_pass and original_pass)
            del same_ref, kw, k8, v8, expected_k
    if not all(case["passes_arithmetic_gate"] for case in report["cases"]):
        raise AssertionError("FMHA arithmetic gate failed; all finite cases recorded")
    if len(candidates) != 4 or not all(candidates):
        raise AssertionError("Per-channel candidate did not pass all four unchanged numerical gates")
    report["status"] = "diagnostic_candidate_passed"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = dict(
        status="started", cases=[],
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        thresholds=dict(output_rms=0.10, absolute_lse=0.20),
        serving_integrated=False,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        run(args.fixture, report)
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()

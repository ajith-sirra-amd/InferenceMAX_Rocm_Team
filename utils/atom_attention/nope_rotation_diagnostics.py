"""Test fixed NoPE rotation with native FP8 projection and residual RoPE.

Diagnostic only: captured-context calibration and unfused transforms are not
serving code or a performance claim. Original numerical gates remain fixed.
"""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from aiter import flash_attn_varlen_func
from aiter.ops.flydsl import flydsl_flash_attn_fp8_func
from channel_balance_diagnostics import emit, metrics, quantize_layer, reference
from vllm.model_executor.kernels.linear.scaled_mm.aiter import (
    AiterPerTokenFp8ScaledMMLinearKernel,
    AiterPreshuffledPerTokenFp8ScaledMMLinearKernel,
)
from vllm.v1.attention.backends.mla.prefill.aiter_flash_attn import (
    AiterFlashAttnPrefillBackend,
)
from vllm.v1.attention.ops.rocm_aiter_mla_fp8_prefill import (
    ProjectionWeight, project, projection_weight,
)
from vllm.v1.attention.ops.rocm_aiter_mla_prefill import expand_context
from vllm.v1.attention.ops.rocm_aiter_mla_fp8_quant import fused_qkv_per_tensor_quant


def hadamard(device):
    h = torch.ones((1, 1), device=device, dtype=torch.float64)
    while h.shape[0] < 128:
        h = torch.cat((torch.cat((h, h), 1), torch.cat((h, -h), 1)), 0)
    return h / (128 ** 0.5)


def rotate(x, h):
    return torch.cat((x[..., :128].double() @ h, x[..., 128:].double()), -1)


def note_stage(report, mode, label, actual, expected, historical):
    record = dict(mode=mode, stage=label, incremental=metrics(actual, expected),
                  vs_historical_bf16=metrics(actual, historical))
    report["stages"].append(record)
    emit("NOPE_STAGE", record)


def run(path, report):
    torch.cuda.set_device(0)
    blas = str(torch.backends.cuda.preferred_blas_library())
    if blas != "_BlasBackend.Cublas":
        raise RuntimeError(f"Unexpected BLAS backend: {blas}")
    data = torch.load(path, map_location="cuda", weights_only=True)
    q, cache, weight = data["q"], data["cache"], data["weight"]
    cuq, cuk, scale = data["cu_query"], data["cu_kv"], data["softmax_scale"]
    heads = q.shape[1]
    if q.shape[-1] != 192 or cache.shape[-1] != 576:
        raise ValueError("Expected captured Kimi MLA dimensions")
    rows = torch.arange(cache.shape[0], device=q.device, dtype=torch.int32)
    k, v = expand_context(cache, data["cache_scale"], rows, cuk, weight,
                          heads, 128, 64, 128)
    backend = SimpleNamespace(scale=scale, flash_attn_varlen_func=flash_attn_varlen_func)
    chunk = SimpleNamespace(query_start_loc=cuq, cu_seq_lens=cuk,
                            max_query_len=data["max_query_len"],
                            max_seq_len=data["max_seq_len"])
    historical = AiterFlashAttnPrefillBackend.run_prefill_context_chunk(
        backend, chunk, q, k, v)
    historical = tuple(t.double() for t in historical)
    original = reference(q, k, v, cuq, cuk, scale)
    report.update(
        fixture_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        queries=q.shape[0], heads=heads, kv_rows=cache.shape[0], blas_backend=blas,
        bf16_arithmetic=metrics(historical, original),
        native_sha="492080779e9feea19308092b0aba0840e87c6115",
        scope="All captured rows on GPU0; both native weight layouts; captured "
              "K maxima; no serving integration or performance claim")
    raw = cache.double() * data["cache_scale"].double()
    h = hadamard(q.device)
    torch.testing.assert_close(h @ h.t(), torch.eye(128, device=q.device,
                               dtype=torch.float64), rtol=0, atol=1e-12)
    base = fused_qkv_per_tensor_quant(q, k, v)
    candidates = []
    for mode in ("unrotated_control", "fixed_hadamard_candidate"):
        rotated = mode == "fixed_hadamard_candidate"
        qr = rotate(q, h) if rotated else q.double()
        kr = rotate(k, h) if rotated else k.double()
        if rotated:
            invariant = reference(qr, kr, v, cuq, cuk, scale)
            identity = metrics(invariant, original)
            emit("NOPE_ROTATION_IDENTITY", identity)
            report["rotation_identity"] = identity
            if identity["rms"] >= 1e-10 or identity["lse_max"] >= 1e-10:
                raise AssertionError("Unquantized rotation changed attention")
            del invariant
        wr64 = weight.double().view(heads, 256, 512).clone()
        if rotated:
            wr64[:, :128] = h.t() @ wr64[:, :128]
        wr = wr64.reshape_as(weight).to(torch.bfloat16).contiguous()
        expanded = (raw[:, :512] @ wr64.reshape_as(weight).t()).view(-1, heads, 256)
        exact_k = torch.cat((expanded[:, :, :128],
                            raw[:, None, 512:].expand(-1, heads, -1)), -1)
        exact_v = expanded[:, :, 128:]
        weight_exact_ref = reference(qr, exact_k, exact_v, cuq, cuk, scale)
        note_stage(report, mode, "exact_projection_before_weight_quantization",
                   weight_exact_ref, original, historical)
        del expanded, exact_k, exact_v, wr64
        params_by_layout = [
            (label, projection_weight(quantize_layer(wr, cls), heads))
            for label, cls in (
                ("plain", AiterPerTokenFp8ScaledMMLinearKernel),
                ("preshuffled", AiterPreshuffledPerTokenFp8ScaledMMLinearKernel),
            )
        ]
        plain = params_by_layout[0][1]
        decoded_weight = plain.weight.double() * plain.scale.double()[:, None]
        expanded = (raw[:, :512] @ decoded_weight.t()).view(-1, heads, 256)
        native_k = torch.cat((expanded[:, :, :128],
                             raw[:, None, 512:].expand(-1, heads, -1)), -1)
        native_v = expanded[:, :, 128:]
        native_ref = reference(qr, native_k, native_v, cuq, cuk, scale)
        note_stage(report, mode, "weight_quantization", native_ref,
                   weight_exact_ref, historical)
        del weight_exact_ref, expanded, decoded_weight
        # RoPE is unchanged by NoPE rotation; retain the native Q quantizer's
        # exact scale and RoPE codes to match the previous control.
        qhigh, qs = base[0], base[3]
        residual = (qr[:, :, 128:].float() / qs - qhigh[:, :, 128:].float()) * 16
        if residual.abs().max().item() > 448:
            raise AssertionError("RoPE residual overflow")
        qlow = residual.to(torch.float8_e4m3fn)
        exponent = int(torch.ceil(torch.log2(
            native_k.abs().max() / (448 * data["cache_scale"].double()))).item())
        ks = (data["cache_scale"] * (2.0 ** exponent)).reshape(1).contiguous()
        vs = (native_v.abs().max().float() / 448).reshape(1)
        vs = torch.nextafter(vs, torch.full_like(vs, torch.inf))
        a = qr[:, :, :128].abs().amax(dim=(0, 2)).clamp_min(1e-12)
        b = native_k[:, :, :128].abs().amax(dim=(0, 2)).clamp_min(1e-12)
        balance = torch.sqrt(a * ks.double() / (b * qs.double())).float()
        q_nope = qr[:, :, :128].float() / balance[None, :, None]
        k_nope = native_k[:, :, :128] * balance.double()[None, :, None]
        if (q_nope.abs() > 448 * qs).any().item() or (
                k_nope.abs() > 448 * ks.double()).any().item():
            raise AssertionError("NoPE balancing clips")
        qw = torch.cat(((q_nope / qs).to(torch.float8_e4m3fn),
                        qhigh[:, :, 128:], qlow), -1).contiguous()
        raw_rope = cache[:, 512:].float() * (2.0 ** (-exponent))
        rope_high = raw_rope.to(torch.float8_e4m3fn)
        rope_low = (raw_rope / 16).to(torch.float8_e4m3fn)
        rope_error = rope_high.double() * ks.double() - raw[:, 512:]
        if torch.count_nonzero(rope_error).item():
            raise AssertionError("Primary cached RoPE is not preserved exactly")
        for layout, params in params_by_layout:
            modified = params.scale.view(heads, 256).clone()
            modified[:, :128] *= balance[:, None]
            adjusted = ProjectionWeight(params.weight, modified.flatten().contiguous(),
                                        params.preshuffled)
            k8, v8, _, _ = project(cache, data["cache_scale"], rows, adjusted,
                                    heads, torch.cat((ks, vs)).contiguous())
            if k8.dtype != torch.float8_e4m3fn or v8.dtype != torch.float8_e4m3fn:
                raise AssertionError("Native projector did not output FP8")
            k8[:, :, 128:].copy_(rope_high[:, None, :].expand(-1, heads, -1))
            kw = torch.cat((k8, rope_low[:, None, :].expand(-1, heads, -1)), -1).contiguous()
            projection_checks = {}
            expected_k = torch.cat((k_nope, native_k[:, :, 128:]), -1)
            for label, actual, expected, descale in (
                    ("k", k8, expected_k, ks), ("v", v8, native_v, vs)):
                error = actual.double() * descale.double() - expected
                rms = (error.square().mean().sqrt()
                       / expected.square().mean().sqrt().clamp_min(1e-12)).item()
                peak = (error.abs().max() / expected.abs().max().clamp_min(1e-12)).item()
                clipping = (expected.abs() > 448 * descale.double()).double().mean().item()
                projection_checks[label] = dict(rms=rms, peak=peak, clipping=clipping)
                if not (rms < 0.04 and peak < 0.07 and clipping == 0):
                    raise AssertionError(f"Projection arithmetic failed: {projection_checks}")
            same_ref = reference(qw.double() * qs.double(), kw.double() * ks.double(),
                                 v8.double() * vs.double(), cuq, cuk, scale)
            if layout == "plain":
                q_effective = torch.cat((
                    qw[:, :, :128].double() * qs.double() * balance.double()[None, :, None],
                    (qhigh[:, :, 128:].double() + qlow.double() / 16) * qs.double()), -1)
                k_effective = torch.cat((
                    k8[:, :, :128].double() * ks.double() / balance.double()[None, :, None],
                    native_k[:, :, 128:]), -1)
                q_effect = reference(q_effective, native_k, native_v, cuq, cuk, scale)
                note_stage(report, mode, "q_rounding_only", q_effect, native_ref, historical)
                k_effect = reference(qr, k_effective, native_v, cuq, cuk, scale)
                note_stage(report, mode, "k_rounding_only", k_effect, native_ref, historical)
                note_stage(report, mode, "combined_input_rounding", same_ref, native_ref, historical)
                del q_effective, k_effective, q_effect, k_effect
            for lazy in (True, False):
                actual = flydsl_flash_attn_fp8_func(
                    qw, kw, v8, softmax_scale=scale,
                    cu_seqlens_q=cuq, cu_seqlens_kv=cuk,
                    max_seqlen_q=data["max_query_len"], max_seqlen_kv=data["max_seq_len"],
                    cross_seqlen=True, causal=False, return_lse=True,
                    q_descale=qs, k_descale=ks, v_descale=vs,
                    dualwave_swp_lazy_rescale=lazy)
                same, bf16 = metrics(actual, same_ref), metrics(actual, historical)
                arithmetic = (same["normalized_max_output_error"] < 0.08
                              and same["min_cosine"] > 0.98
                              and same["normalized_max_lse_error"] < 1e-4)
                original_pass = bf16["rms"] < 0.10 and bf16["lse_max"] < 0.20
                record = dict(mode=mode, layout=layout, lazy_rescale=lazy,
                              projection=projection_checks, same_fp8_inputs=same,
                              vs_historical_bf16=bf16, passes_arithmetic_gate=arithmetic,
                              passes_original_gate=original_pass)
                report["cases"].append(record)
                emit("NOPE_ROTATION_CASE", record)
                if rotated:
                    candidates.append(arithmetic and original_pass)
            del same_ref, kw, k8, v8, expected_k
        del native_ref, native_k, native_v, params_by_layout
    if not all(c["passes_arithmetic_gate"] for c in report["cases"]):
        raise AssertionError("FMHA arithmetic gate failed")
    if len(candidates) != 4 or not all(candidates):
        raise AssertionError("Fixed rotation candidate failed unchanged numerical gates")
    report["status"] = "diagnostic_candidate_passed"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = dict(status="started", cases=[], stages=[], serving_integrated=False,
                  thresholds=dict(output_rms=0.10, absolute_lse=0.20),
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
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

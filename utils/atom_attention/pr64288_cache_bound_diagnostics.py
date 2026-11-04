"""Measure cache-derived scale bounds without changing the serving implementation."""
import json
import math
from pathlib import Path
import sys

import torch
from aiter.ops.shuffle import shuffle_weight
from vllm.v1.attention.ops.rocm_aiter_mla_fp8_prefill import ProjectionWeight, project


def scales_from_cache(cache, cache_scale, indices, params, heads):
    weight = params.weight.float()
    if params.preshuffled:
        row_squares = weight.view(heads * 16, 16, 2, 16, 16).square().sum((1, 2, 4)).reshape(-1)
    else:
        row_squares = weight.square().sum(-1)
    norms = row_squares.sqrt() * params.scale
    norms = norms.view(heads, 256)
    weight_norms = torch.stack((norms[:, :128].amax(), norms[:, 128:].amax()))
    selected = cache.float()[indices.long()] * cache_scale
    latent_norm = selected[:, :512].square().sum(-1).sqrt().amax()
    bounds = latent_norm * weight_norms
    bounds[0] = torch.maximum(bounds[0], selected[:, 512:].abs().amax())
    return (bounds * (1 + 2**-10) / 448).clamp_min(1e-6), weight_norms


def main(output):
    torch.manual_seed(71)
    torch.set_num_threads(8)
    heads, rows = 12, 512
    weight = (torch.randn(heads * 256, 512, device="cuda") * 32).to(torch.float8_e4m3fn)
    weight_scale = torch.linspace(0.0002, 0.0008, heads * 256, device="cuda")
    cache_scale = torch.tensor([0.125], device="cuda")
    raw_cache = torch.randn(rows, 576, device="cuda")
    indices = torch.randperm(rows, device="cuda")[:386].to(torch.int32)
    reference_weight = weight.cpu().double() * weight_scale.cpu().double()[:, None]
    results, failures = [], []
    for multiplier in (1, 8, 32):
        raw = raw_cache.clone()
        raw[indices[:8].long()] *= multiplier
        cache = raw.to(torch.float8_e4m3fn)
        latent = cache.cpu().double()[indices.cpu().long()] * cache_scale.cpu().double()
        expanded = (latent[:, :512] @ reference_weight.t()).view(-1, heads, 256)
        expected = (
            torch.cat((expanded[..., :128], latent[:, None, 512:].expand(-1, heads, -1)), -1),
            expanded[..., 128:],
        )
        for shuffled in (False, True):
            params = ProjectionWeight(
                shuffle_weight(weight, (16, 16)) if shuffled else weight,
                weight_scale, shuffled,
            )
            scales, norms = scales_from_cache(cache, cache_scale, indices, params, heads)
            expected_norms = reference_weight.norm(dim=-1).view(heads, 256)
            expected_norms = torch.stack((expected_norms[:, :128].max(), expected_norms[:, 128:].max()))
            torch.testing.assert_close(norms.cpu().double(), expected_norms, rtol=1e-6, atol=1e-8)
            result = project(cache, cache_scale, indices, params, heads, scales)
            record = {"multiplier": multiplier, "preshuffled": shuffled, "scales": scales.tolist(), "kv": []}
            for value, scale, reference in zip(result[:2], result[2:], expected):
                scale = scale.cpu().double()
                error = value.cpu().double() * scale - reference
                metrics = {
                    "clipped": int((reference.abs() > 448 * scale).sum()),
                    "relative_rms": (error.square().mean()/reference.square().mean()).sqrt().item(),
                    "normalized_peak": (error.abs().max()/reference.abs().max()).item(),
                    "range_over_actual_max": (448*scale/reference.abs().max()).item(),
                }
                metrics["passed"] = metrics["clipped"] == 0 and metrics["relative_rms"] < 0.04 and metrics["normalized_peak"] < 0.07
                record["kv"].append(metrics)
                if not metrics["passed"]:
                    failures.append(metrics)
            results.append(record)
            print("CACHE_BOUND_PROJECTION", json.dumps(record), flush=True)
    # Predict the observed two-times-outlier FMHA error from AITER's documented
    # cast of unnormalized softmax probabilities to FP8 before the P*V matmul.
    prob = math.exp((0.125 - 0.5) / math.sqrt(192))
    denom = 124 * prob + 4
    exact_output = (124 * prob * 0.125 + 4 * 0.25) / denom
    fp8_probability_output = (124 * 0.125 + 4 * 0.25) / denom
    predicted_bf16 = torch.tensor(fp8_probability_output).to(torch.bfloat16).item()
    report = {
        "scope": "new scale-selection experiment; Torch intermediates; no serving or performance claim",
        "native_source": "e2393d702a3736dffdfbfc7eb25abbabe484a1df",
        "cases": results, "failures": len(failures),
        "existing_aiter_probability_rounding": {
            "probability_before_fp8": prob, "probability_after_fp8": 1.0,
            "high_precision_output": exact_output, "predicted_bf16_output": predicted_bf16,
            "predicted_absolute_error": abs(predicted_bf16-exact_output),
            "observed_absolute_error_run41821548931": 0.0038025281567669322,
        },
    }
    output.write_text(json.dumps(report, indent=2))
    if failures:
        raise AssertionError("Cache-derived bounds failed existing projection error gates")


if __name__ == "__main__":
    main(Path(sys.argv[1]))

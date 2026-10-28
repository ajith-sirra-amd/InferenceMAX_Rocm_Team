# SPDX-License-Identifier: Apache-2.0
"""Adapt native vLLM online FP8 weights to ATOM's unchanged FlyDSL projection."""
from dataclasses import dataclass

import torch

from vllm.model_executor.kernels.linear.scaled_mm.aiter import (
    AiterHipbMMPerTokenFp8ScaledMMLinearKernel,
    AiterPerTokenFp8ScaledMMLinearKernel,
    AiterPreshuffledPerTokenFp8ScaledMMLinearKernel,
)
from vllm.model_executor.kernels.linear.scaled_mm.pytorch import (
    RowWiseTorchFP8ScaledMMLinearKernel,
)
from vllm.model_executor.layers.quantization.online.fp8 import Fp8PtpcOnlineLinearMethod


@dataclass(frozen=True)
class ProjectionWeight:
    weight: torch.Tensor
    scale: torch.Tensor
    preshuffled: bool
    linear_kernel: str


def projection_weight(layer, heads):
    """Return views only. Never re-quantize, repack, or alter the native layer."""
    method = layer.quant_method
    if type(method) is not Fp8PtpcOnlineLinearMethod:
        raise ValueError("FP8 projection requires native fp8_per_channel online quantization")
    kernel = type(method.fp8_linear)
    weight, scale = layer.weight, layer.weight_scale
    if kernel is AiterHipbMMPerTokenFp8ScaledMMLinearKernel:
        # Native hipb_mm stores a transposed VIEW of shuffled [N,K].
        # Transpose back before any contiguous operation.
        weight = weight.t()
        preshuffled = True
    elif kernel is AiterPreshuffledPerTokenFp8ScaledMMLinearKernel:
        preshuffled = True
    elif kernel is AiterPerTokenFp8ScaledMMLinearKernel:
        preshuffled = False
    elif kernel is RowWiseTorchFP8ScaledMMLinearKernel:
        weight = weight.t()
        preshuffled = False
    else:
        raise ValueError(f"Unaudited FP8 weight layout: {kernel.__module__}.{kernel.__name__}")
    if (weight.dtype != torch.float8_e4m3fn
            or weight.shape != (heads * 256, 512) or not weight.is_contiguous()):
        raise ValueError("Expected contiguous OCP E4M3 [heads*256,512] projection storage")
    if (scale.dtype != torch.float32 or scale.numel() != heads * 256
            or not scale.is_contiguous() or scale.device != weight.device):
        raise ValueError("Expected contiguous FP32 per-output-channel descales")
    if not weight.is_cuda or getattr(layer, "bias", None) is not None:
        raise ValueError("FP8 projection requires GPU weights without bias")
    return ProjectionWeight(weight, scale.view(-1), preshuffled, kernel.__name__)


def project(gathered, cache_scale, indices, params, heads, output_scales=None):
    """FP8 cache x FP8 weight, FP32 accumulation, direct FP8 or BF16 output."""
    from aiter.ops.flydsl.gather_kv_b_proj import gather_kv_b_proj_flydsl

    if (gathered.ndim != 2 or gathered.shape[1] != 576
            or gathered.dtype != torch.float8_e4m3fn or not gathered.is_contiguous()):
        raise ValueError("Expected contiguous OCP E4M3 cache [rows,576]")
    if (indices.ndim != 1 or indices.dtype != torch.int32 or not indices.is_contiguous()):
        raise ValueError("Expected contiguous int32 cache row indices")
    if (cache_scale.dtype != torch.float32 or cache_scale.numel() != 1
            or not cache_scale.is_contiguous()):
        raise ValueError("Expected scalar FP32 cache descale")
    tensors = [cache_scale, indices, params.weight, params.scale]
    if output_scales is not None:
        if (output_scales.dtype != torch.float32 or output_scales.shape != (2,)
                or not output_scales.is_contiguous()):
            raise ValueError("Expected contiguous FP32 [K,V] output descales")
        tensors.append(output_scales)
    if not gathered.is_cuda or any(t.device != gathered.device for t in tensors):
        raise ValueError("All projection tensors must share a GPU")
    dtype = torch.bfloat16 if output_scales is None else torch.float8_e4m3fn
    with torch.cuda.device(gathered.device):
        k = torch.empty((indices.numel(), heads, 192), device=gathered.device, dtype=dtype)
        v = torch.empty((indices.numel(), heads, 128), device=gathered.device, dtype=dtype)
        kwargs = {} if output_scales is None else {
            "k_out_scale": output_scales[:1], "v_out_scale": output_scales[1:],
        }
        gather_kv_b_proj_flydsl(
            gathered.view(-1, 1, 576), cache_scale, None, indices, None,
            params.weight, params.scale, k, v,
            weight_preshuffle=params.preshuffled, **kwargs,
        )
    if output_scales is None:
        return k, v
    return k, v, output_scales[:1], output_scales[1:]


def reference_projection(gathered, cache_scale, indices, params, heads):
    """Diagnostic only: independent FP64 matmul on dequantized FP8 operands."""
    weight = params.weight.view(torch.uint8)
    if params.preshuffled:
        n, width = weight.shape
        # Inverse of AITER's (16,16) shuffle for one-byte FP8 elements.
        weight = (weight.view(n // 16, width // 32, 2, 16, 16)
                  .permute(0, 3, 1, 2, 4).contiguous().view(n, width))
    weight = weight.view(torch.float8_e4m3fn).double() * params.scale.double()[:, None]
    cache = gathered.double()[indices.long()] * cache_scale.double()
    expanded = (cache[:, :512] @ weight.t()).view(-1, heads, 256)
    k = torch.cat((expanded[:, :, :128],
                   cache[:, None, 512:].expand(-1, heads, -1)), dim=-1)
    return k, expanded[:, :, 128:]


def validate_projection(kv8, reference):
    """Check FP8 rounding against FP64, separately from attention/model effects."""
    import json
    import os

    metrics = {}
    for name, value, expected, scale in zip(("k", "v"), kv8[:2], reference, kv8[2:]):
        actual = value.double() * scale.double()
        if not torch.isfinite(actual).all().item() or not torch.isfinite(expected).all().item():
            raise RuntimeError(f"Nonfinite FP8 projection {name}")
        error = (actual - expected).abs()
        rms = (error.square().mean().sqrt()
               / expected.square().mean().sqrt().clamp_min(1e-12)).item()
        max_relative = (error.max() / expected.abs().max().clamp_min(1e-12)).item()
        clip = (expected.abs() > 448 * scale.double()).double().mean().item()
        metrics[name] = {"rms_vs_fp64": rms, "max_error_over_max_reference": max_relative,
                         "clipping_fraction": clip}
        if rms >= 0.04 or max_relative >= 0.07 or clip != 0:
            raise RuntimeError(f"FP8 projection gate failed: {name} {metrics[name]}")
    print("[ATOM-FP8-PROJECTION-VALIDATED] " + json.dumps({"pid": os.getpid(), **metrics}), flush=True)
    return metrics

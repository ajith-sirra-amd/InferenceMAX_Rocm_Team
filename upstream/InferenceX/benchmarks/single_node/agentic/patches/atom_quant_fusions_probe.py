# SPDX-License-Identifier: Apache-2.0
"""Fail before graph capture unless the main-model producer consumers are FP8."""
import torch


def verify_model(model):
    from vllm.model_executor.layers.fusion.quant_activation import get_input_quant_key
    from vllm.model_executor.layers.quantization.utils.quant_utils import kFp8DynamicTokenSym
    from vllm.models.kimi_k3.amd.linear import KimiMLP

    counts = {"strided": 0, "gated": 0, "shared": 0, "dense": 0}
    savings = 0
    device = None
    for name, layer in model.named_modules():
        kind = None
        if name.endswith(".self_attn.f_b_proj"):
            kind = "strided"
        elif name.endswith(".self_attn.o_proj"):
            parent = model.get_submodule(name.rsplit(".", 1)[0])
            if getattr(parent, "g_proj", None) is not None:
                kind = "gated"
        elif isinstance(layer, KimiMLP):
            if ".block_sparse_moe.shared_experts" in name:
                kind = "shared"
            elif name.endswith(".layers.0.mlp"):
                kind = "dense"
            if kind is not None:
                for suffix in ("gate_up_proj", "down_proj"):
                    linear = getattr(layer, suffix)
                    weight = linear.weight
                    scale = getattr(linear, "weight_scale", None)
                    if weight.dtype not in (torch.float8_e4m3fn, torch.float8_e4m3fnuz) or scale is None:
                        raise RuntimeError(f"MLP weight is not online per-channel FP8: {name}.{suffix}")
                    if get_input_quant_key(linear) != kFp8DynamicTokenSym:
                        raise RuntimeError(f"MLP consumer cannot accept fused FP8 input: {name}.{suffix}; {type(linear.quant_method).__name__}")
                    savings += weight.numel() - scale.numel() * scale.element_size()
                    kernel = getattr(linear.quant_method, "fp8_linear", None)
                    device = weight.device
                    print("[ATOM-MLP-WEIGHT-FP8]", "device=", str(device),
                          "name=", name + "." + suffix, "weight_dtype=", str(weight.dtype),
                          "weight_shape=", tuple(weight.shape), "scale_shape=", tuple(scale.shape),
                          "kernel=", type(kernel).__name__, "producer_quant_capability=True", flush=True)
                layer = layer.down_proj
        if kind is None:
            continue
        if get_input_quant_key(layer) != kFp8DynamicTokenSym:
            raise RuntimeError(f"Producer consumer capability missing: {name}; {type(layer.quant_method).__name__}")
        device = layer.weight.device
        counts[kind] += 1
    expected = {"strided": 69, "gated": 24, "shared": 92, "dense": 1}
    if counts != expected:
        raise RuntimeError(f"Unexpected main-model fusion inventory: {counts}, expected {expected}")
    print("[ATOM-QUANT-CONSUMERS-VERIFIED]", "device=", str(device),
          "counts=", counts, "new_mlp_net_weight_savings_bytes=", savings,
          "fp8_per_token_input=True", "bf16_gemm_output=True", flush=True)

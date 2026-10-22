"""Targeted pre-pack Quark FP8 kv_b_proj conversion for the native DCP path."""
import pathlib
import shutil

import torch


@torch.no_grad()
def convert_weight(weight, scale):
    if weight.dtype != torch.float8_e4m3fn or weight.ndim != 2:
        raise ValueError("Expected unpacked E4M3FN matrix")
    if scale.numel() != weight.shape[0]:
        raise ValueError("Expected one scale per output channel")
    scale = scale.reshape(-1, 1).float()
    if not torch.isfinite(scale).all() or not (scale > 0).all():
        raise ValueError("Invalid checkpoint scales")
    result = (weight.float() * scale).to(torch.bfloat16).contiguous()
    if not torch.isfinite(result).all():
        raise ValueError("Nonfinite converted weight")
    return result


def targeted_method(original, prefix):
    from vllm.model_executor.layers.linear import (
        UnquantizedLinearMethod, WEIGHT_LOADER_V2_SUPPORTED,
    )

    class BF16KVMethod(type(original)):
        def create_weights(self, layer, *args, **kwargs):
            super().create_weights(layer, *args, **kwargs)
            layer.requires_unquantized_input = True
            if hasattr(layer, "_input_quant_key"):
                delattr(layer, "_input_quant_key")

        @torch.no_grad()
        def process_weights_after_loading(self, layer):
            if type(layer.scheme).__name__ != "QuarkW8A8Fp8":
                raise ValueError("Unexpected kv_b_proj Quark scheme")
            if layer.scheme.weight_qscheme != "per_channel":
                raise ValueError("Expected per-channel kv_b_proj")
            if layer.weight.shape != (3072, 512):
                raise ValueError(f"Unexpected TP8 kv_b_proj shape: {layer.weight.shape}")
            old_bytes = layer.weight.numel() + layer.weight_scale.numel() * 4
            weight = convert_weight(layer.weight, layer.weight_scale)
            layer.weight = torch.nn.Parameter(weight, requires_grad=False)
            for name in ("weight_scale", "input_scale", "_input_quant_key"):
                if hasattr(layer, name):
                    delattr(layer, name)
            layer.scheme = None
            layer.quant_method = UnquantizedLinearMethod()
            layer.requires_unquantized_input = True
            # Keep the native fused kernel's required contiguous [out,in] layout.
            # No postpack/padding: UnquantizedLinearMethod.apply consumes this layout.
            print("[KV-B-BF16-CONVERTED]", "prefix=", prefix,
                  "device=", str(weight.device), "shape=", tuple(weight.shape),
                  "dtype=", str(weight.dtype), "contiguous=", weight.is_contiguous(),
                  "weight_scale_present=False", "load_time_only=True",
                  "weight_bytes_delta=", weight.numel() * 2 - old_bytes,
                  flush=True)

    if "BF16KVMethod" not in WEIGHT_LOADER_V2_SUPPORTED:
        WEIGHT_LOADER_V2_SUPPORTED.append("BF16KVMethod")
    return BF16KVMethod(original.quantization_config)


@torch.no_grad()
def preflight():
    from vllm.platforms import current_platform
    from vllm.v1.attention.ops.rocm_aiter_mla_prefill import expand_context

    for index in range(torch.cuda.device_count()):
        torch.cuda.set_device(index)
        device = f"cuda:{index}"
        # Hand-computed channel-scale case, including negative and zero values.
        raw = torch.tensor([[1., -2.], [0., 4.]], device=device).to(torch.float8_e4m3fn)
        scales = torch.tensor([0.5, 2.], device=device)
        expected = torch.tensor([[0.5, -1.], [0., 8.]], device=device,
                                dtype=torch.bfloat16)
        torch.testing.assert_close(convert_weight(raw, scales), expected, rtol=0, atol=0)
        try:
            convert_weight(raw, torch.tensor([-1., 1.], device=device))
        except ValueError:
            pass
        else:
            raise AssertionError("Negative scale accepted")
        # Real native fused reconstruction: cache rows reordered, scale=0.5,
        # selected projection columns. These powers of two have exact BF16 results.
        cache = torch.zeros((4, 576), device=device)
        cache[:, :512] = torch.arange(1, 5, device=device).reshape(4, 1)
        cache[:, 512:] = 8.
        cache = cache.to(current_platform.fp8_dtype())
        weight = torch.zeros((3072, 512), device=device, dtype=torch.bfloat16)
        weight[:, 0] = 2.
        rows = torch.tensor([3, 1, 0, 2], device=device, dtype=torch.int32)
        cu = torch.tensor([0, 4], device=device, dtype=torch.int32)
        k, v = expand_context(cache, torch.tensor([0.5], device=device),
                              rows, cu, weight, 12, 128, 64, 128)
        values = torch.tensor([4., 2., 1., 3.], device=device,
                              dtype=torch.bfloat16).reshape(4, 1, 1)
        torch.testing.assert_close(k[:, :, :128], values.expand(4, 12, 128), rtol=0, atol=0)
        torch.testing.assert_close(v, values.expand(4, 12, 128), rtol=0, atol=0)
        torch.testing.assert_close(k[:, :, 128:], torch.full_like(k[:, :, 128:], 4.),
                                   rtol=0, atol=0)
        torch.cuda.synchronize()
        print("[KV-B-NATIVE-PREFLIGHT]", "device=", device,
              "conversion_exact=True fused_reconstruction_exact=True", flush=True)


def patch(source):
    anchor = "            return QuarkLinearMethod(self)"
    if source.count(anchor) != 1:
        raise ValueError("Unexpected Quark method dispatch layout")
    replacement = """            method = QuarkLinearMethod(self)
            if prefix.endswith(".kv_b_proj"):
                from vllm.model_executor.layers.quantization.quark._bf16_kv_b import targeted_method
                return targeted_method(method, prefix)
            return method"""
    result = source.replace(anchor, replacement)
    compile(result, "<bf16-kv-quark>", "exec")
    return result


if __name__ == "__main__":
    import vllm
    site = pathlib.Path(vllm.__file__).parent
    path = site / "model_executor/layers/quantization/quark/quark.py"
    result = patch(path.read_text())
    shutil.copyfile(__file__, path.parent / "_bf16_kv_b.py")
    path.write_text(result)
    preflight()
    print("[KV-B-BF16-INSTALL] targeted kv_b_proj only; other Quark schemes retained")

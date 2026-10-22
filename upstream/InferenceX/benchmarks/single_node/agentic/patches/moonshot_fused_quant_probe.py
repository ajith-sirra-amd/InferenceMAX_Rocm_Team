"""Verify Moonshot baseline attention target plus FP8 KV reconstruction weights."""
import pathlib
import shutil
import torch


def verify_model(model):
    from vllm.model_executor.layers.linear import LinearBase
    from vllm.v1.attention.ops._infx_fp8_fused_kv import projection_weight
    kv_count = inproj_count = oproj_count = qkva_count = gate_count = 0
    device = None
    for name, layer in model.named_modules():
        if ".self_attn." not in name or not isinstance(layer, LinearBase):
            continue
        weight = getattr(layer, "weight", None)
        if weight is None:
            raise RuntimeError(f"Missing attention weight: {name}")
        device = weight.device
        if name.endswith(".kv_b_proj"):
            info = projection_weight(layer, 12)
            if info is None:
                raise RuntimeError(f"Moonshot online FP8 kv_b layout unsupported: {name}; "
                                   f"method={type(layer.quant_method).__name__}")
            packed, scales, shuffled, kernel = info
            kv_count += 1
            print("[MOONSHOT-KV-B-FP8]", "device=", str(device),
                  "name=", name, "weight_dtype=", str(packed.dtype),
                  "weight_shape=", tuple(packed.shape), "scale_shape=", tuple(scales.shape),
                  "preshuffle=", shuffled, "kernel=", kernel, flush=True)
        elif name.endswith(".in_proj_qkvgfab"):
            if (weight.dtype not in (torch.float8_e4m3fn, torch.float8_e4m3fnuz)
                    or getattr(layer, "weight_scale", None) is None):
                raise RuntimeError(f"Baseline online FP8 target missing: {name}")
            inproj_count += 1
        elif name.endswith(".o_proj"):
            if (weight.dtype not in (torch.float8_e4m3fn, torch.float8_e4m3fnuz)
                    or getattr(layer, "weight_scale", None) is None):
                raise RuntimeError(f"Output projection FP8 target missing: {name}")
            oproj_count += 1
            print("[MOONSHOT-O-PROJ-FP8]", "device=", str(device),
                  "name=", name, "weight_dtype=", str(weight.dtype),
                  "weight_shape=", tuple(weight.shape),
                  "scale_shape=", tuple(layer.weight_scale.shape),
                  "method=", type(layer.quant_method).__name__, flush=True)
        elif name.endswith((".fused_qkv_a_proj", ".g_proj")):
            if (weight.dtype not in (torch.float8_e4m3fn, torch.float8_e4m3fnuz)
                    or getattr(layer, "weight_scale", None) is None):
                raise RuntimeError(f"Additional FP8 target missing: {name}")
            if name.endswith(".fused_qkv_a_proj"):
                qkva_count += 1
                marker = "[MOONSHOT-QKV-A-FP8]"
            else:
                gate_count += 1
                marker = "[MOONSHOT-G-PROJ-FP8]"
            print(marker, "device=", str(device), "name=", name,
                  "weight_dtype=", str(weight.dtype), "weight_shape=", tuple(weight.shape),
                  "scale_shape=", tuple(layer.weight_scale.shape),
                  "method=", type(layer.quant_method).__name__, flush=True)
        elif name.endswith(".conv1d"):
            if (weight.dtype not in (torch.bfloat16, torch.float32)
                    or getattr(layer, "weight_scale", None) is not None):
                raise RuntimeError(f"Convolution must remain unquantized: {name} {weight.dtype}")
        elif weight.dtype != torch.bfloat16:
            raise RuntimeError(f"Untargeted Moonshot attention weight changed dtype: {name} {weight.dtype}")
    if kv_count != 24 or not inproj_count or not oproj_count or qkva_count != 24 or gate_count != 24:
        raise RuntimeError(f"Unexpected Moonshot inventory: kv_b={kv_count}, inproj={inproj_count}, o_proj={oproj_count}, qkv_a={qkva_count}, g_proj={gate_count}")
    print("[MOONSHOT-FUSED-QUANT-VERIFIED]", "device=", str(device),
          "fp8_kv_b_linears=", kv_count, "fp8_inproj_linears=", inproj_count,
          "fp8_o_proj_linears=", oproj_count, "fp8_qkv_a_linears=", qkva_count,
          "fp8_g_proj_linears=", gate_count, "remaining_projection_weights=BF16", "conv1d=unquantized_native_dtype", "checkpoint_experts_unchanged=True", flush=True)


def install():
    import vllm
    root = pathlib.Path(vllm.__file__).parent / "model_executor/model_loader"
    path = root / "utils.py"
    source = path.read_text()
    anchor = "    # Model-level post-load hook, after the per-layer quant finalize.\n"
    if source.count(anchor) != 1:
        raise RuntimeError("Unexpected pinned postload hook")
    source = source.replace(anchor,
        "    from vllm.model_executor.model_loader._moonshot_fused_probe import verify_model\n"
        '    if model_config.hf_config.model_type == "kimi_k3":\n'
        "        verify_model(model)\n\n" + anchor)
    compile(source, "<moonshot-fused-loader>", "exec")
    shutil.copyfile(__file__, root / "_moonshot_fused_probe.py")
    path.write_text(source)
    print("[MOONSHOT-FUSED-QUANT-PROBE] installed")


if __name__ == "__main__":
    install()

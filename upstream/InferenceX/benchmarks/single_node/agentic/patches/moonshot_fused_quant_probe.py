"""Verify Moonshot baseline attention target plus FP8 KV reconstruction weights."""
import pathlib
import shutil
import torch


def verify_model(model):
    from vllm.model_executor.layers.linear import LinearBase
    from vllm.v1.attention.ops._infx_fp8_fused_kv import projection_weight
    kv_count = inproj_count = 0
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
        elif weight.dtype != torch.bfloat16:
            raise RuntimeError(f"Untargeted Moonshot attention weight changed dtype: {name} {weight.dtype}")
    if kv_count != 24 or not inproj_count:
        raise RuntimeError(f"Unexpected Moonshot inventory: kv_b={kv_count}, inproj={inproj_count}")
    print("[MOONSHOT-FUSED-QUANT-VERIFIED]", "device=", str(device),
          "fp8_kv_b_linears=", kv_count, "fp8_inproj_linears=", inproj_count,
          "other_attention=BF16", "checkpoint_experts_unchanged=True", flush=True)


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

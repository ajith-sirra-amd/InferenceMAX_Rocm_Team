"""Verify selected online attention quantization after real model loading."""
import pathlib
import runpy
import shutil
import torch


def verify_model(model):
    from vllm.model_executor.layers.linear import LinearBase
    quantized = excluded = 0
    device = None
    for name, layer in model.named_modules():
        if ".self_attn." not in name or not isinstance(layer, LinearBase):
            continue
        weight = getattr(layer, "weight", None)
        if name.endswith(".kv_b_proj"):
            if (weight is None or weight.dtype != torch.bfloat16
                    or not weight.is_contiguous()
                    or getattr(layer, "weight_scale", None) is not None):
                raise ValueError(f"Native BF16 kv_b_proj requirement failed: {name}")
            excluded += 1
            device = weight.device
            print("[MOONSHOT-KV-B-BF16]", "device=", str(weight.device),
                  "name=", name, "shape=", tuple(weight.shape),
                  "dtype=", str(weight.dtype), "contiguous=True scale_present=False",
                  flush=True)
        elif "conv1d" in name:
            if weight is not None and weight.dtype in (torch.float8_e4m3fn, torch.float8_e4m3fnuz):
                raise ValueError(f"Excluded convolution quantized: {name}")
        else:
            if (weight is None or weight.dtype not in (torch.float8_e4m3fn, torch.float8_e4m3fnuz)
                    or getattr(layer, "weight_scale", None) is None):
                raise ValueError(f"Attention FP8 target did not engage: {name}")
            quantized += 1
            print("[MOONSHOT-ATTN-FP8]", "device=", str(weight.device),
                  "name=", name, "shape=", tuple(weight.shape),
                  "dtype=", str(weight.dtype), "method=", type(layer.quant_method).__name__,
                  flush=True)
    if not quantized or excluded != 24:
        raise ValueError(f"Unexpected attention inventory: quantized={quantized}, kv_b={excluded}")
    print("[MOONSHOT-QUANT-VERIFIED]", "device=", str(device),
          "fp8_attention_linears=", quantized, "bf16_kv_b_linears=", excluded,
          "checkpoint_expert_quantization_retained=True", flush=True)


def patch(source):
    anchor = "    # Model-level post-load hook, after the per-layer quant finalize.\n"
    if source.count(anchor) != 1:
        raise ValueError("Unexpected postload hook layout")
    result = source.replace(anchor,
        "    from vllm.model_executor.model_loader._moonshot_quant_probe import verify_model\n"
        '    if model_config.hf_config.model_type == "kimi_k3":\n'
        "        verify_model(model)\n\n" + anchor)
    compile(result, "<moonshot-quant-probe>", "exec")
    return result


if __name__ == "__main__":
    import vllm
    root = pathlib.Path(vllm.__file__).parent / "model_executor/model_loader"
    path = root / "utils.py"
    result = patch(path.read_text())
    shutil.copyfile(__file__, root / "_moonshot_quant_probe.py")
    path.write_text(result)
    # Reuse numerical preflight only; do not install the AMD conversion.
    runpy.run_path(str(pathlib.Path(__file__).with_name("bf16_kv_b.py")))["preflight"]()
    print("[MOONSHOT-QUANT-PROBE] installed; online quantization controlled by CLI")

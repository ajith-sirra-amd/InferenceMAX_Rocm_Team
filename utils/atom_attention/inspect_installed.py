"""Read installed attention paths without loading a model or modifying packages."""
import importlib
import importlib.metadata
import inspect
import json
import os
import subprocess

print("ATOM_ATTENTION_DIAGNOSTIC", flush=True)
for name in ("vllm", "torch", "amd-aiter", "flydsl", "atom", "triton"):
    try:
        print("PACKAGE", name, importlib.metadata.version(name), flush=True)
    except importlib.metadata.PackageNotFoundError:
        print("PACKAGE", name, "not installed", flush=True)
for name in (
    "vllm.v1.attention.backends.mla.prefill.registry",
    "vllm.v1.attention.backends.mla.prefill.aiter_flash_attn",
    "aiter.ops.flydsl",
):
    try:
        module = importlib.import_module(name)
        print("MODULE", name, module.__file__, flush=True)
        if name.endswith("registry"):
            print("PREFILL_BACKENDS", [(x.name, x.value) for x in module.MLAPrefillBackendEnum])
        if name == "aiter.ops.flydsl":
            for symbol in ("flydsl_flash_attn_fp8_func", "flydsl_flash_attn_fp8_supported"):
                fn = getattr(module, symbol)
                print("API", symbol, str(inspect.signature(fn)), flush=True)
                print("SOURCE", inspect.getsource(fn)[:6500], flush=True)
    except Exception as error:
        print("UNAVAILABLE", name, type(error).__name__, str(error), flush=True)
import torch
print("GPU_COUNT", torch.cuda.device_count(), flush=True)
if torch.cuda.is_available():
    print("GPU", torch.cuda.get_device_properties(0), flush=True)
    try:
        from aiter.ops.flydsl import flydsl_flash_attn_fp8_supported
        for heads in (16, 24):
            print("FP8_SUPPORTED", heads, 192, 128,
                  flydsl_flash_attn_fp8_supported(torch.device("cuda:0"), heads, heads, 192, 128), flush=True)
    except Exception as error:
        print("SUPPORT_PROBE", type(error).__name__, str(error), flush=True)

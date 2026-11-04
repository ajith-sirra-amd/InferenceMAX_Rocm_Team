"""Observe real serving calls without changing dispatch, scales, or tensor values."""
import functools
import hashlib
import json
import os
from pathlib import Path

_REGISTERED = False
MARKER = "PR64288_SERVING "


def register():
    global _REGISTERED
    if _REGISTERED:
        return
    import torch
    import torch.distributed as dist
    import vllm
    from vllm.v1.attention.backends.mla.rocm_aiter_mla import AiterMLAImpl

    source = Path(vllm.__file__).resolve().parent.parent
    manifest = json.loads(Path(os.environ["REVIEW_MANIFEST"]).read_text())
    if str(source) != manifest["source_root"]:
        raise RuntimeError("Wrong serving source root")
    for path, digest in manifest["source_hashes"].items():
        if hashlib.sha256((source / path).read_bytes()).hexdigest() != digest:
            raise RuntimeError("Serving source differs: " + path)
    enabled = os.environ["VLLM_ROCM_USE_AITER_FLYDSL_FP8_PREFILL_ATTN"] == "1"
    phase_file = Path(os.environ["REVIEW_PHASE_FILE"])
    emitted = set()

    def record(kind, **details):
        if not phase_file.is_file() or torch.cuda.is_current_stream_capturing():
            return
        phase = phase_file.read_text().strip()
        key = (phase, kind, details.get("selected"))
        if key in emitted:
            return
        print(MARKER + json.dumps({
            "phase": phase, "kind": kind, "rank": dist.get_rank(),
            "pid": os.getpid(), "native_sha": manifest["native_sha"],
            "flag_enabled": enabled, "forced_routing": False,
            "graph_capture": False, **details,
        }, sort_keys=True), flush=True)
        emitted.add(key)

    forward = AiterMLAImpl.forward_mha

    @functools.wraps(forward)
    def observed_forward(self, q, *args, **kwargs):
        result = forward(self, q, *args, **kwargs)
        if q.numel():
            record("prefill", dtype=str(q.dtype), query_rows=q.shape[0],
                   native_enabled=getattr(self, "_use_flydsl_fp8_prefill", False))
        return result

    AiterMLAImpl.forward_mha = observed_forward
    # The upstream-base arm does not contain the native integration module.
    native_path = source / "vllm/v1/attention/ops/rocm_aiter_mla_fp8_prefill.py"
    if native_path.is_file():
        from vllm.v1.attention.ops import rocm_aiter_mla_fp8_prefill as native
        original_project = native.project
        original_route = native.use_fp8_context

        @functools.wraps(original_route)
        def observed_route(query, context):
            selected = original_route(query, context)
            record("routing", selected=selected, max_query_len=query,
                   max_context_len=context)
            return selected

        @functools.wraps(original_project)
        def observed_project(*args, **kwargs):
            result = original_project(*args, **kwargs)
            if not enabled:
                raise RuntimeError("Flag-OFF arm executed native projection")
            if result[0].numel():
                kind = "fp8_projection" if result[0].dtype == torch.float8_e4m3fn else "bf16_projection"
                record(kind, dtype=str(result[0].dtype), rows=result[0].shape[0],
                       cache_dtype=str(args[0].dtype), weight_dtype=str(args[3].weight.dtype))
            return result

        native.project = observed_project
        native.use_fp8_context = observed_route
    from aiter.ops import flydsl
    attention = flydsl.flydsl_flash_attn_fp8_func

    @functools.wraps(attention)
    def observed_attention(q, k, v, *args, **kwargs):
        result = attention(q, k, v, *args, **kwargs)
        if not enabled:
            raise RuntimeError("Flag-OFF arm executed FlyDSL FP8 attention")
        if q.numel():
            record("fp8_fmha", dtype=str(q.dtype), k_dtype=str(k.dtype),
                   v_dtype=str(v.dtype), query_rows=q.shape[0],
                   context_rows=k.shape[0], causal=bool(kwargs.get("causal")))
        return result

    flydsl.flydsl_flash_attn_fp8_func = observed_attention
    _REGISTERED = True

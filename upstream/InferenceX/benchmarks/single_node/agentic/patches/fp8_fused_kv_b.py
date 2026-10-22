"""Runtime integration for FP8-weight fused Kimi KV reconstruction.

Installed only into the ephemeral benchmark container. FMHA remains BF16.
"""
import pathlib
import torch
from vllm.platforms import current_platform


def projection_weight(layer, heads):
    scheme = getattr(layer, "scheme", None)
    kernel = getattr(scheme, "fp8_linear", None)
    name = type(kernel).__name__
    weight = getattr(layer, "weight", None)
    scale = getattr(layer, "weight_scale", None)
    dtype = current_platform.fp8_dtype()
    if (type(scheme).__name__ != "QuarkW8A8Fp8"
            or getattr(scheme, "weight_qscheme", None) != "per_channel"
            or weight is None or scale is None
            or weight.dtype != dtype or scale.numel() != heads * 256
            or getattr(layer, "bias", None) is not None):
        return None
    if name == "AiterPreshuffledPerTokenFp8ScaledMMLinearKernel":
        packed, shuffled = weight, True
    elif name == "AiterHipbMMPerTokenFp8ScaledMMLinearKernel":
        packed, shuffled = weight.t(), True
    elif name in {"ChannelWiseTorchFP8ScaledMMLinearKernel",
                  "RowWiseTorchFP8ScaledMMLinearKernel"}:
        packed, shuffled = weight.t(), False
    else:
        return None
    if packed.shape != (heads * 256, 512) or not packed.is_contiguous():
        return None
    if not scale.is_contiguous():
        return None
    return packed, scale.reshape(-1), shuffled, name


@torch.library.custom_op("inferencex::fp8_fused_kv_expand", mutates_args=())
def fused_expand(cache: torch.Tensor, scales: torch.Tensor,
                 indices: torch.Tensor, weight: torch.Tensor,
                 weight_scale: torch.Tensor, preshuffle: bool,
                 rows_per_rank: int, heads: int) -> tuple[torch.Tensor, torch.Tensor]:
    from vllm.v1.attention.ops._infx_fp8_kv_kernel import launch
    k = torch.empty((indices.numel(), heads, 192), device=cache.device,
                    dtype=torch.bfloat16)
    v = torch.empty((indices.numel(), heads, 128), device=cache.device,
                    dtype=torch.bfloat16)
    if indices.numel():
        launch(cache, scales, indices, weight, weight_scale, k, v,
               preshuffle, rows_per_rank)
    return k, v


@fused_expand.register_fake
def _fake(cache, scales, indices, weight, weight_scale, preshuffle,
          rows_per_rank, heads):
    return (torch.empty((indices.numel(), heads, 192), device=cache.device,
                        dtype=torch.bfloat16),
            torch.empty((indices.numel(), heads, 128), device=cache.device,
                        dtype=torch.bfloat16))


def rank_scales(backend, scale, manager):
    # Model KV scales are static after loading. Gather once per attention layer.
    scales = getattr(backend, "_infx_rank_scales", None)
    if scales is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("FP8 rank-scale initialization must precede graph capture")
        scales = torch.empty(manager.group.world_size, device=scale.device,
                             dtype=torch.float32)
        torch.distributed.all_gather_into_tensor(
            scales, scale.reshape(1).float(), group=manager.group.device_group)
        if not torch.isfinite(scales).all().item() or not (scales > 0).all().item():
            raise RuntimeError("Invalid source-rank KV scales")
        backend._infx_rank_scales = scales
        print("[FP8-FUSED-KV-SCALES]", "device=", str(scale.device),
              "values=", scales.tolist(), "source_rank_scale=True", flush=True)
    return scales


@torch.no_grad()
def validate_outputs(cache, scales, indices, packed, weight_scale,
                     shuffled, rows_per_rank, k, v):
    """Independent FP32 projection on decoded checkpoint values; no requantization."""
    count = min(33, indices.numel())
    selected_indices = indices[:count].long()
    selected = cache.view(torch.uint8).index_select(0, selected_indices).view(cache.dtype).float()
    selected = selected * scales[selected_indices // rows_per_rank, None]
    if shuffled:
        # Inverse of the documented AITER (16,16) byte packing.
        n, width = packed.shape
        raw = packed.view(torch.uint8).reshape(n // 16, width // 32, 2, 16, 16)
        raw = raw.permute(0, 3, 1, 2, 4).contiguous().reshape(n, width).view(packed.dtype)
    else:
        raw = packed
    real_weight = raw.float() * weight_scale.float().reshape(-1, 1)
    projection = torch.nn.functional.linear(selected[:, :512], real_weight)
    projection = projection.reshape(count, k.shape[1], 256)
    ref_k = torch.cat((projection[:, :, :128],
                       selected[:, None, 512:].expand(-1, k.shape[1], -1)), dim=-1)
    ref_v = projection[:, :, 128:]
    ref = torch.cat((ref_k.flatten(), ref_v.flatten()))
    actual = torch.cat((k[:count].float().flatten(), v[:count].float().flatten()))
    error = actual - ref
    if not torch.isfinite(actual).all().item():
        raise RuntimeError("Fused reconstruction produced nonfinite output")
    rms = (error.square().mean().sqrt() / ref.square().mean().sqrt().clamp_min(1e-12)).item()
    normalized_max = (error.abs().max() / ref.abs().max().clamp_min(1e-12)).item()
    if rms > 0.005 or normalized_max > 0.02:
        raise RuntimeError(f"FP8 reconstruction reference mismatch: rms={rms}, max={normalized_max}")
    torch.testing.assert_close(k[:count, :, 128:],
                               ref_k[:, :, 128:].bfloat16(), rtol=0, atol=0)
    return rms, normalized_max


def expand(backend, gathered, scale, indices, cu, layer, heads,
           qk_nope, qk_rope, v_head, chunk, manager):
    info = projection_weight(layer, heads)
    if info is None:
        raise RuntimeError("Unsupported FP8 KV weight scheme/layout; no silent BF16 fallback")
    weight, weight_scale, shuffled, kernel_name = info
    if (qk_nope, qk_rope, v_head) != (128, 64, 128):
        raise RuntimeError("Unsupported Kimi head dimensions")
    scales = rank_scales(backend, scale, manager)
    k, v = fused_expand(gathered, scales, indices, weight, weight_scale,
                        shuffled, chunk.num_local_context_tokens, heads)
    if indices.numel() and not getattr(backend, "_infx_fused_verified", False):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Numerical verification must precede graph capture")
        rms, maximum = validate_outputs(gathered, scales, indices, weight,
                                        weight_scale, shuffled,
                                        chunk.num_local_context_tokens, k, v)
        backend._infx_fused_verified = True
        print("[FP8-FUSED-KV-VERIFIED]", "device=", str(k.device),
              "weight_dtype=", str(weight.dtype), "weight_shape=", tuple(weight.shape),
              "weight_scale_shape=", tuple(weight_scale.shape),
              "linear_kernel=", kernel_name, "preshuffle=", shuffled,
              "k_dtype=", str(k.dtype), "v_dtype=", str(v.dtype),
              "k_shape=", tuple(k.shape), "v_shape=", tuple(v.shape),
              "relative_rms=", rms, "normalized_max=", maximum,
              "reference=FP32_decoded_checkpoint", "separate_latent_decode=False",
              flush=True)
    if indices.numel() >= 128 and not getattr(backend, "_infx_large_context_logged", False):
        backend._infx_large_context_logged = True
        print("[FP8-FUSED-KV-CONTEXT-EXECUTED]", "device=", str(k.device),
              "rows=", indices.numel(), "weight_dtype=", str(weight.dtype),
              "gathered_dtype=", str(gathered.dtype),
              "k_dtype=", str(k.dtype), "v_dtype=", str(v.dtype), flush=True)
    return k, v


def patch_backend(source):
    anchor = "    def _context_parallel_compute_prefill_context("
    start = source.index(anchor)
    stop = source.find("\n    def ", start + len(anchor))
    stop = len(source) if stop < 0 else stop
    method = source[start:stop]
    pairs = [
        ('        weight = getattr(self.kv_b_proj, "weight", None)',
         '        from vllm.v1.attention.ops import _infx_fp8_fused_kv as fp8_fusion\n'
         '        weight = getattr(self.kv_b_proj, "weight", None)\n'
         '        fp8_info = fp8_fusion.projection_weight(self.kv_b_proj, self.num_heads)'),
        ('            and weight.dtype == torch.bfloat16',
         '            and (weight.dtype == torch.bfloat16 or fp8_info is not None)'),
        ('            and weight.shape == (self.num_heads * 256, 512)',
         '            and (weight.shape == (self.num_heads * 256, 512) or fp8_info is not None)'),
        ('            and weight.is_contiguous()',
         '            and (weight.is_contiguous() or fp8_info is not None)'),
        ('            and getattr(self.kv_b_proj, "weight_scale", None) is None',
         '            and (getattr(self.kv_b_proj, "weight_scale", None) is None or fp8_info is not None)'),
        ('            k, v = expand_context(',
         '            if fp8_info is not None:\n'
         '                k, v = fp8_fusion.expand(\n'
         '                    self, gathered, k_scale, row_indices, chunk.cu_seq_lens,\n'
         '                    self.kv_b_proj, self.num_heads, self.qk_nope_head_dim,\n'
         '                    self.qk_rope_head_dim, self.v_head_dim, chunk, chunked.dcp_manager)\n'
         '            else:\n'
         '                k, v = expand_context('),
    ]
    # Indent the original expand_context invocation under the BF16 branch.
    i = method.index('            k, v = expand_context(')
    j = method.index('\n            attn_output,', i)
    block = method[i:j]
    method = method[:i] + '\n'.join('    ' + line for line in block.splitlines()) + method[j:]
    pairs[-1] = (pairs[-1][0].replace('            k,', '                k,'),
                 pairs[-1][1])
    for old, new in pairs:
        if method.count(old) != 1:
            raise ValueError("Unexpected pinned vLLM source: " + old)
        method = method.replace(old, new)
    # For this experiment unsupported quantized weights must fail visibly.
    method = method.replace('        if not eligible:\n',
        '        if weight is not None and weight.dtype == current_platform.fp8_dtype() and not eligible:\n'
        '            raise RuntimeError("FP8 fused KV reconstruction ineligible; inspect layout/metadata")\n'
        '        if not eligible:\n')
    # current_platform is needed before eligibility; delete the later local import.
    method = method.replace('        from vllm.platforms import current_platform\n', '')
    method = method.replace('        attn_metadata = cast(', 
                            '        from vllm.platforms import current_platform\n'
                            '        attn_metadata = cast(', 1)
    result = source[:start] + method + source[stop:]
    compile(result, "<fp8-fused-kv-backend>", "exec")
    return result


def install():
    import hashlib
    import vllm
    import aiter
    root = pathlib.Path(vllm.__file__).parent
    here = pathlib.Path(__file__).parent
    (root / "v1/attention/ops/_infx_fp8_fused_kv.py").write_text(here.joinpath("fp8_fused_kv_b.py").read_text())
    (root / "v1/attention/ops/_infx_fp8_kv_kernel.py").write_text(here.joinpath("fp8_fused_kv_kernel.py").read_text())
    path = root / "v1/attention/backends/mla/rocm_aiter_mla.py"
    path.write_text(patch_backend(path.read_text()))
    print("[FP8-FUSED-KV-INSTALLED]", "vllm=", vllm.__version__,
          "aiter_version=", getattr(aiter, "__version__", "unreported"),
          "aiter_path=", aiter.__file__,
          "kernel_sha256=", hashlib.sha256(here.joinpath("fp8_fused_kv_kernel.py").read_bytes()).hexdigest(),
          "kernel_base=a29547a2f8a986c90b9173fcf13769f6b3cf0502",
          "source_rank_scales=True", "fmha=BF16", flush=True)


if __name__ == "__main__":
    install()

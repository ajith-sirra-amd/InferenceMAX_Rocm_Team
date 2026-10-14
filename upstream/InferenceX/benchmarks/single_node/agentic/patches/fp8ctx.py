# [FP8-CTX smoke test] 2026-10-13
# Hook the AITER MLA prefill backend's run_prefill_{context_chunk,new_tokens} to
# run attention on the aiter FP8 ASM kernel (mla_prefill_ps_asm_fwd + mla_reduce_v1)
# instead of bf16 flash_attn_varlen_func. Purely a FUNCTIONALITY probe:
#   - q/k/v arrive bf16 (gate NOT bypassed); we quantize to fp8 in-hook.
#   - context chunks use is_causal=False; new-tokens use is_causal=True.
#   - ANY problem -> return None -> caller falls back to the original bf16 path.
# Every decision is logged with the [FP8-CTX] tag so the server log tells us
# exactly whether the fp8 kernel was invoked and whether it ran.
#
# The aiter call shapes/args are replicated verbatim from
# rocm_aiter_mla.py::_mla_fp8_prefill_attn / _build_fp8_prefill_ps_metadata /
# _init_fp8_prefill_ps_buffers (fe4df3df).

import torch

from vllm.logger import init_logger

logger = init_logger("fp8ctx")

_TILE_Q = 256
_selftest_done = False
_kernel_ok = None  # set by selftest_noncausal: True=kernel works, False=broken -> disable real fp8 path
_count = {"ctx_ok": 0, "ctx_fail": 0, "new_ok": 0, "new_fail": 0}


def _fp8_dtype():
    from vllm.platforms import current_platform
    return current_platform.fp8_dtype()


def _num_head_k(real_nhead):
    from vllm.v1.attention.backends.mla.rocm_aiter_mla import AiterMLAHelper
    return AiterMLAHelper.get_fp8_prefill_num_heads(real_nhead)


def _pad_heads(real_nhead, nhead, *tensors):
    from vllm.v1.attention.backends.mla.rocm_aiter_mla import AiterMLAHelper
    if nhead == real_nhead:
        return tensors
    return tuple(AiterMLAHelper.get_mla_padded_q(real_nhead, t, nhead) for t in tensors)


def _ensure_ps_buffers(backend, nhead, max_qlen, device):
    """Lazily allocate the PS-metadata buffers once per backend instance.

    Mirrors _init_fp8_prefill_ps_buffers: work_metadata stays on device; the five
    map tensors are filled by get_ps_metadata_v1 on the host then copied to device.
    """
    cache = getattr(backend, "_fp8ctx_ps", None)
    if cache is not None and cache["nhead"] == nhead:
        return cache
    from aiter import get_ps_metadata_info_v1
    info = get_ps_metadata_info_v1(
        batch_size=512, num_head_k=nhead, max_qlen=max_qlen, qlen_granularity=_TILE_Q
    )

    def _mk(spec, dev):
        size, dt = spec
        if isinstance(size, (tuple, list)):
            return torch.empty(*size, dtype=dt, device=dev)
        return torch.empty(size, dtype=dt, device=dev)

    work_metadata = _mk(info[0], device)          # device, not host-written
    dev_outs = [_mk(s, device) for s in info[1:]]  # work_indptr, work_info, reduce_{indptr,final,partial}
    host_outs = [_mk(s, "cpu") for s in info[1:]]
    cache = {
        "nhead": nhead,
        "work_metadata": work_metadata,
        "dev": dev_outs,
        "host": host_outs,
    }
    backend._fp8ctx_ps = cache
    logger.info("[FP8-CTX] PS buffers allocated (nhead=%d max_qlen=%d)", nhead, max_qlen)
    return cache


def _run_fp8_asm(backend, q, k, v, qo_indptr, kv_indptr, seq_lens, max_q_len, is_causal):
    """Core: q/k/v already decompressed (nhead heads, gqa=1). Returns (out[bf16], lse[f32])."""
    from aiter import get_ps_metadata_v1, mla_prefill_ps_asm_fwd, mla_reduce_v1

    device = q.device
    fp8 = _fp8_dtype()
    real_nhead = q.shape[1]
    nhead = _num_head_k(real_nhead)
    q, k, v = _pad_heads(real_nhead, nhead, q, k, v)
    if q.dtype != fp8:
        q = q.to(fp8)
    if k.dtype != fp8:
        k = k.to(fp8)
    if v.dtype != fp8:
        v = v.to(fp8)

    v_head_dim = v.shape[-1]
    total_q = q.shape[0]
    total_kv = k.shape[0]

    # Always size for the worst-case Q length (max_num_batched_tokens) so buffers
    # allocated by the tiny self-test are also valid for real 24576-token chunks.
    ps = _ensure_ps_buffers(backend, nhead, 24576, device)
    wm = ps["work_metadata"]
    dev_o = ps["dev"]      # [work_indptr, work_info, reduce_indptr, reduce_final_map, reduce_partial_map]
    host_o = ps["host"]

    qo_cpu = qo_indptr.detach().to("cpu", torch.int32)
    kv_cpu = kv_indptr.detach().to("cpu", torch.int32)
    seq_cpu = seq_lens.detach().to("cpu", torch.int32)

    get_ps_metadata_v1(
        qo_cpu, kv_cpu, seq_cpu, 1, nhead,
        wm, host_o[0], host_o[1], host_o[2], host_o[3], host_o[4],
        qhead_granularity=1, qlen_granularity=_TILE_Q,
        kvlen_granularity=128, block_size=1, is_causal=is_causal,
    )
    for d, h in zip(dev_o, host_o):
        d.copy_(h, non_blocking=True)
    num_partial = int(host_o[2][-1].item())  # reduce_indptr[-1]

    kv_indices = torch.arange(total_kv, device=device, dtype=torch.int32)
    logits = torch.empty(num_partial * _TILE_Q, nhead, v_head_dim, dtype=torch.float32, device=device)
    attn_lse = torch.empty(num_partial * _TILE_Q, nhead, dtype=torch.float32, device=device)
    final_lse = torch.empty(total_q, nhead, dtype=torch.float32, device=device)
    out3d = torch.empty(total_q, nhead, v_head_dim, dtype=torch.bfloat16, device=device)
    one = torch.ones((), dtype=torch.float32, device=device)

    mla_prefill_ps_asm_fwd(
        q, k, v,
        qo_indptr.to(torch.int32), kv_indptr.to(torch.int32), kv_indices,
        dev_o[0], dev_o[1], int(max_q_len), float(backend.scale), bool(is_causal),
        logits, attn_lse, out3d, one, one, one,
    )
    mla_reduce_v1(
        logits, attn_lse, dev_o[2], dev_o[3], dev_o[4], _TILE_Q, 0, out3d, final_lse,
    )

    if nhead != real_nhead:
        out3d = out3d[:, :real_nhead, :].contiguous()
        final_lse = final_lse[:, :real_nhead].contiguous()
    return out3d, final_lse


def fp8_context_chunk(backend, chunk, q, k, v):
    """Hook for run_prefill_context_chunk (non-causal). Returns (out, lse) or None."""
    global _selftest_done
    if _kernel_ok is False:
        return None  # self-test showed the non-causal kernel is broken -> stay on bf16
    try:
        qo = chunk.query_start_loc
        kv = chunk.cu_seq_lens
        seq = (kv[1:] - kv[:-1])
        if not _selftest_done:
            _selftest_done = True
            logger.info("[FP8-CTX] FIRST context-chunk invocation: q=%s k=%s v=%s max_q=%s",
                        tuple(q.shape), tuple(k.shape), tuple(v.shape), int(chunk.max_query_len))
        out, lse = _run_fp8_asm(backend, q, k, v, qo, kv, seq, int(chunk.max_query_len), False)
        _count["ctx_ok"] += 1
        if _count["ctx_ok"] <= 3 or _count["ctx_ok"] % 200 == 0:
            logger.info("[FP8-CTX] context-chunk fp8 OK (n=%d) out=%s lse=%s",
                        _count["ctx_ok"], tuple(out.shape), tuple(lse.shape))
        return out, lse
    except Exception as e:
        _count["ctx_fail"] += 1
        if _count["ctx_fail"] <= 3:
            import traceback
            logger.info("[FP8-CTX] context-chunk fp8 FAILED (n=%d): %s\n%s",
                        _count["ctx_fail"], e, traceback.format_exc())
        return None


def fp8_new_tokens(backend, q, k, v):
    """Hook for run_prefill_new_tokens (causal). Returns (out, lse) or None."""
    try:
        md = backend._prefill_metadata
        qo = md.query_start_loc
        seq = (qo[1:] - qo[:-1])
        out, lse = _run_fp8_asm(backend, q, k, v, qo, qo, seq, int(md.max_query_len), True)
        _count["new_ok"] += 1
        if _count["new_ok"] <= 3:
            logger.info("[FP8-CTX] new-tokens fp8 OK (n=%d) out=%s", _count["new_ok"], tuple(out.shape))
        return out, lse
    except Exception as e:
        _count["new_fail"] += 1
        if _count["new_fail"] <= 3:
            import traceback
            logger.info("[FP8-CTX] new-tokens fp8 FAILED (n=%d): %s\n%s",
                        _count["new_fail"], e, traceback.format_exc())
        return None


_selftest_nc_done = False


def selftest_noncausal(backend):
    """Run ONCE per worker: a tiny synthetic NON-causal fp8 ASM prefill to prove the
    kernel exists/runs on gfx950, independent of whether real context chunks occur.
    Triggered from the run_prefill_new_tokens hook (fires on the first prefill)."""
    global _selftest_nc_done, _kernel_ok
    if _selftest_nc_done:
        return
    _selftest_nc_done = True
    try:
        import torch
        dev = torch.device("cuda", torch.cuda.current_device())
        fp8 = _fp8_dtype()
        nhead, qk, vd, qlen, kvlen = 16, 192, 128, 8, 64
        q = torch.randn(qlen, nhead, qk, device=dev).to(fp8)
        k = torch.randn(kvlen, nhead, qk, device=dev).to(fp8)
        v = torch.randn(kvlen, nhead, vd, device=dev).to(fp8)
        qo = torch.tensor([0, qlen], dtype=torch.int32, device=dev)
        kv = torch.tensor([0, kvlen], dtype=torch.int32, device=dev)
        seq = torch.tensor([kvlen], dtype=torch.int32, device=dev)
        out, lse = _run_fp8_asm(backend, q, k, v, qo, kv, seq, qlen, False)
        torch.cuda.synchronize()
        finite = bool(torch.isfinite(out.float()).all().item())
        _kernel_ok = True
        logger.info("[FP8-CTX] STARTUP SELF-TEST: non-causal fp8 ASM kernel OK "
                    "(out=%s lse=%s finite=%s sample=%s)",
                    tuple(out.shape), tuple(lse.shape), finite,
                    out.flatten()[:3].float().tolist())
    except Exception as e:
        _kernel_ok = False
        import traceback
        logger.info("[FP8-CTX] STARTUP SELF-TEST: non-causal fp8 ASM kernel FAILED: %s\n%s",
                    e, traceback.format_exc())


_flash_log_count = 0


def wrap_flash(fn):
    """Trace wrapper for aiter.flash_attn_varlen_func: logs the first calls with the
    causal flag, q/k/v shapes+dtypes, and the caller stack -> reveals K3's actual
    prefill attention path and the right hook point."""
    def _wrapped(*args, **kwargs):
        global _flash_log_count
        if _flash_log_count < 12:
            _flash_log_count += 1
            try:
                import traceback
                causal = kwargs.get("causal", None)
                q = kwargs.get("q", args[0] if len(args) > 0 else None)
                k = kwargs.get("k", args[1] if len(args) > 1 else None)
                v = kwargs.get("v", args[2] if len(args) > 2 else None)

                def _sh(t):
                    try:
                        return (tuple(t.shape), str(t.dtype))
                    except Exception:
                        return None

                caller = "".join(traceback.format_stack(limit=7)[:-1])[-700:]
                logger.info("[FP8-CTX] flash_attn_varlen_func CALL #%d causal=%s q=%s k=%s v=%s\n  CALLER:\n%s",
                            _flash_log_count, causal, _sh(q), _sh(k), _sh(v), caller)
            except Exception:
                pass
        return fn(*args, **kwargs)

    return _wrapped

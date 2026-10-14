# [FP8-CTX] 2026-10-14
# Hook the AITER MLA prefill backend's run_prefill_context_chunk (cached/chunked
# context, is_causal=False) to run attention on the aiter FP8 ASM kernel
# (mla_prefill_ps_asm_fwd + mla_reduce_v1) instead of bf16 flash_attn_varlen_func.
# New-token (no-context) prefill needs no patch: vLLM's own AiterMLAImpl.forward_mha
# already dispatches that case straight to the native _mla_fp8_prefill_attn kernel
# whenever kv_cache_dtype=fp8 (confirmed via the "FP8 MLA prefill PS buffers
# allocated" log line in rocm_aiter_mla.py), so run_prefill_new_tokens's hook here
# only self-tests and never overrides it.
#   - q/k/v arrive bf16 (gate NOT bypassed); we quantize to fp8 in-hook.
#   - ANY problem -> return None -> caller falls back to the original bf16 path.
# Every decision is logged with the [FP8-CTX] tag.
#
# PS-metadata and scratch-buffer sizing mirrors rocm_aiter_mla.py's own
# _init_fp8_prefill_ps_buffers exactly (same vllm_config fields, same
# tile/compute-unit formula) so the reservation is a correct upper bound for any
# batch this engine can schedule, not a hardcoded guess tied to one recipe's
# --max-num-batched-tokens value. Buffers are allocated once per backend
# instance and sliced per call instead of re-allocated every chunk.
#
# The aiter call shapes/args are replicated verbatim from
# rocm_aiter_mla.py::_mla_fp8_prefill_attn / _build_fp8_prefill_ps_metadata /
# _init_fp8_prefill_ps_buffers (fe4df3df).

import os
import torch

from vllm.logger import init_logger

# Must be a "vllm.*" child logger: vLLM's dictConfig only attaches a handler to
# the top-level "vllm" logger (propagate=False). A bare "fp8ctx" logger is
# unrelated to that hierarchy, so its records silently vanish into the root
# logger (no handler there) -- this is why no [FP8-CTX] lines ever appeared in
# server.log even while the hooks were actually running.
logger = init_logger("vllm.fp8ctx")

# "asm" (default): mla_prefill_ps_asm_fwd + mla_reduce_v1, 2 kernel launches +
# CPU-side PS-metadata planning per call.
# "fused": single aiter.flash_attn_varlen_func call (the SAME function the
# bf16 baseline already uses) with fp8-quantized q/k/v + q/k/v_descale --
# aiter's own dtype dispatch (aiter/ops/mha.py) routes this through the
# identical single CK kernel call used for bf16, zero host-side planning.
_FP8CTX_MODE = os.environ.get("FP8CTX_MODE", "asm")

_TILE_Q = 256
_selftest_done = False
_kernel_ok = None  # set by selftest_noncausal: True=kernel works, False=broken -> disable real fp8 path (asm mode only)
_count = {"ctx_ok": 0, "ctx_fail": 0, "new_ok": 0, "new_fail": 0, "fused_ok": 0, "fused_fail": 0}


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


def _build_ps_buffers(nhead, v_head_dim, max_num_reqs, max_prefill_qlen,
                       max_num_batched_tokens, device):
    """Shared buffer-construction logic, parameterized by explicit sizing
    inputs instead of reading vllm_config directly -- lets both the real
    per-backend worst-case cache (_ensure_ps_buffers, sourced from
    backend.vllm_config) and the tiny one-off startup self-test (sourced from
    its own synthetic shapes) share one code path without either needing a
    vllm_config instance that may not exist/be reachable at that call site."""
    from aiter import get_ps_metadata_info_v1
    from vllm.platforms import current_platform
    from vllm.utils.math_utils import cdiv

    info = get_ps_metadata_info_v1(
        batch_size=max_num_reqs, num_head_k=nhead, max_qlen=max_prefill_qlen,
        qlen_granularity=_TILE_Q,
    )

    def _mk(spec, dev):
        size, dt = spec
        if isinstance(size, (tuple, list)):
            return torch.empty(*size, dtype=dt, device=dev)
        return torch.empty(size, dtype=dt, device=dev)

    work_metadata = _mk(info[0], device)          # device, not host-written
    dev_outs = [_mk(s, device) for s in info[1:]]  # work_indptr, work_info, reduce_{indptr,final,partial}
    host_outs = [_mk(s, "cpu") for s in info[1:]]

    # Same worst-case partial-tile count rocm_aiter_mla.py reserves for its own
    # logits/attn_lse scratch (qo tiles + one extra per request for splits at
    # request boundaries + one extra per compute unit for in-flight work).
    qo_tile_cnt = cdiv(max_num_batched_tokens, _TILE_Q) + max_num_reqs - 1
    max_num_partial_tiles = qo_tile_cnt + current_platform.num_compute_units()

    logits = torch.empty(
        max_num_partial_tiles * _TILE_Q, nhead, v_head_dim,
        dtype=torch.float32, device=device,
    )
    attn_lse = torch.empty(
        max_num_partial_tiles * _TILE_Q, nhead, dtype=torch.float32, device=device
    )
    final_lse = torch.empty(
        max_num_batched_tokens, nhead, dtype=torch.float32, device=device
    )
    out3d = torch.empty(
        max_num_batched_tokens, nhead, v_head_dim,
        dtype=torch.bfloat16, device=device,
    )
    one = torch.ones((), dtype=torch.float32, device=device)
    # kv_indices is always just 0..total_kv-1 (no gather/permutation here) --
    # precompute the worst-case arange once and slice it, instead of launching a
    # fresh arange kernel on every chunk call.
    kv_indices_full = torch.arange(max_num_batched_tokens, device=device, dtype=torch.int32)

    return {
        "nhead": nhead,
        "work_metadata": work_metadata,
        "dev": dev_outs,
        "host": host_outs,
        "logits": logits,
        "attn_lse": attn_lse,
        "final_lse": final_lse,
        "out3d": out3d,
        "one": one,
        "kv_indices": kv_indices_full,
        "max_num_partial_tiles": max_num_partial_tiles,
    }


def _ensure_ps_buffers(backend, nhead, device):
    """Lazily allocate persistent worst-case PS-metadata + scratch buffers once
    per backend instance, sized exactly like rocm_aiter_mla.py's own
    _init_fp8_prefill_ps_buffers (same vllm_config fields, same tile/CU
    formula) -- a correct upper bound for any batch this engine can schedule,
    not a hardcoded guess tied to one recipe's --max-num-batched-tokens value.

    Scratch tensors (logits/attn_lse/final_lse/out3d) are sliced per call
    instead of re-allocated. Safe to reuse across calls without an extra copy:
    prefill isn't CUDA-graph-captured and this recipe runs with
    --no-async-scheduling, so every kernel that reads or writes these buffers
    is enqueued on one stream in program order -- a later chunk's write is
    always ordered after the previous chunk's consumer (the merge/copy in
    accumulate_mla_context_chunk) by stream-ordering semantics alone.

    Only ever called with an AiterFlashAttnPrefillBackend instance (the real
    context-chunk call path), which always stores self.vllm_config -- the
    startup self-test builds its own tiny one-off buffers instead
    (_build_ps_buffers directly, see selftest_noncausal) because its caller,
    AiterMLAImpl, never stores vllm_config as an attribute, and the global
    get_current_vllm_config() accessor isn't reachable either: that context
    manager is only active during model construction, not during the live
    serving forward pass where this self-test actually fires (confirmed by
    both approaches crashing in back-to-back smoke tests).
    """
    cache = getattr(backend, "_fp8ctx_ps", None)
    if cache is not None and cache["nhead"] == nhead:
        return cache
    vllm_config = backend.vllm_config
    max_num_reqs = vllm_config.scheduler_config.max_num_seqs
    max_num_batched_tokens = vllm_config.scheduler_config.max_num_batched_tokens
    max_prefill_qlen = min(
        vllm_config.model_config.max_model_len, max_num_batched_tokens
    )
    v_head_dim = backend.v_head_dim

    cache = _build_ps_buffers(
        nhead, v_head_dim, max_num_reqs, max_prefill_qlen,
        max_num_batched_tokens, device,
    )
    backend._fp8ctx_ps = cache
    logger.info(
        "[FP8-CTX] PS buffers allocated (nhead=%d max_qlen=%d max_batch=%d "
        "max_partial_tiles=%d)",
        nhead, max_prefill_qlen, max_num_reqs, cache["max_num_partial_tiles"],
    )
    return cache


def _run_fp8_asm(backend, q, k, v, qo_indptr, kv_indptr, seq_lens, max_q_len, is_causal,
                  ps_override=None):
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

    # Persistent worst-case buffers, sized once from backend.vllm_config (see
    # _ensure_ps_buffers) and sliced per call -- no per-chunk allocator
    # traffic. The startup self-test passes its own tiny one-off buffers via
    # ps_override instead, since it has no backend.vllm_config to size from.
    ps = ps_override if ps_override is not None else _ensure_ps_buffers(backend, nhead, device)
    wm = ps["work_metadata"]
    dev_o = ps["dev"]      # [work_indptr, work_info, reduce_indptr, reduce_final_map, reduce_partial_map]
    host_o = ps["host"]
    one = ps["one"]

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
    # No `.item()` read-back here on purpose: neither mla_prefill_ps_asm_fwd nor
    # mla_reduce_v1 takes a num_partial scalar -- they derive actual work extent
    # from the GPU-resident dev_o tensors (work_info/reduce_indptr/...) internally.
    # The only consumer of num_partial was slicing logits/attn_lse down from the
    # worst-case allocation; since those buffers are already sized at the static
    # worst case (_ensure_ps_buffers), passing them unsliced is correct (the
    # kernel simply never touches rows beyond what its own metadata indicates)
    # and avoids a synchronous GPU->CPU stall on every single chunk call -- this
    # was firing ~85x/sec over a 1hr run and was the dominant cause of a measured
    # +35% TPOT / -18% per-GPU throughput regression vs the bf16 baseline.
    kv_full = ps["kv_indices"]
    if total_kv <= kv_full.shape[0]:
        kv_indices = kv_full[:total_kv]
    else:
        # Defensive fallback only -- should not happen since total_kv is bounded
        # by the same max_num_batched_tokens the buffer was sized from.
        kv_indices = torch.arange(total_kv, device=device, dtype=torch.int32)
    logits = ps["logits"]
    attn_lse = ps["attn_lse"]
    final_lse = ps["final_lse"][:total_q]
    out3d = ps["out3d"][:total_q]

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
        final_lse = final_lse[:, :real_nhead]
    # mla_reduce_v1 writes lse as [total_q, nhead] (matching out3d's token-major
    # layout), but every caller (accumulate_mla_context_chunk, merge_attn_states)
    # indexes lse as [nhead, total_q] -- the same convention aiter.flash_attn_varlen_func
    # returns natively. Transpose here or chunk-merge crashes with a shape
    # mismatch between the token count and head count on dim 1.
    final_lse = final_lse.transpose(0, 1).contiguous()
    return out3d, final_lse


def _run_fp8_fused(backend, q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, is_causal):
    """Single-kernel fp8 path: per-tensor-quantize q/k/v and call the SAME
    aiter.flash_attn_varlen_func the bf16 baseline already uses
    (backend.flash_attn_varlen_func, stored by AiterFlashAttnPrefillBackend.
    __init__), passing q_descale/k_descale/v_descale. aiter's own dtype
    dispatch (aiter/ops/mha.py: `has_all_descales` branch) routes fp8-dtype
    q/k/v + all three descales through the identical single CK kernel call
    used for bf16 -- same cu_seqlens/max_seqlen/causal args as the baseline,
    zero host-side PS-metadata planning, zero second reduce kernel. No head
    padding needed: unlike the ASM kernel, flash_attn_varlen_func has no
    head-count-multiple constraint (the bf16 baseline already calls it with
    raw, unpadded head counts). Output dtype is bf16 regardless of q/k/v
    dtype (confirmed in aiter's own fp8 test suite)."""
    from aiter import per_tensor_quant

    fp8 = _fp8_dtype()
    q_quant, q_descale = per_tensor_quant(q, quant_dtype=fp8)
    k_quant, k_descale = per_tensor_quant(k, quant_dtype=fp8)
    v_quant, v_descale = per_tensor_quant(v, quant_dtype=fp8)
    out, lse = backend.flash_attn_varlen_func(
        q=q_quant, k=k_quant, v=v_quant,
        cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q, max_seqlen_k=max_seqlen_k,
        softmax_scale=backend.scale, causal=is_causal, return_lse=True,
        q_descale=q_descale, k_descale=k_descale, v_descale=v_descale,
    )
    return out, lse


def fp8_context_chunk(backend, chunk, q, k, v):
    """Hook for run_prefill_context_chunk (non-causal). Returns (out, lse) or None."""
    global _selftest_done
    if _FP8CTX_MODE == "fused":
        try:
            qo = chunk.query_start_loc
            kv = chunk.cu_seq_lens
            if not _selftest_done:
                _selftest_done = True
                logger.info("[FP8-CTX] FIRST context-chunk invocation (fused): q=%s k=%s v=%s max_q=%s",
                            tuple(q.shape), tuple(k.shape), tuple(v.shape), int(chunk.max_query_len))
            out, lse = _run_fp8_fused(backend, q, k, v, qo, kv,
                                       int(chunk.max_query_len), int(chunk.max_seq_len), False)
            _count["fused_ok"] += 1
            if _count["fused_ok"] <= 3 or _count["fused_ok"] % 200 == 0:
                logger.info("[FP8-CTX] context-chunk fp8 fused OK (n=%d) out=%s lse=%s",
                            _count["fused_ok"], tuple(out.shape), tuple(lse.shape))
            return out, lse
        except Exception as e:
            _count["fused_fail"] += 1
            if _count["fused_fail"] <= 3:
                import traceback
                logger.info("[FP8-CTX] context-chunk fp8 fused FAILED (n=%d): %s\n%s",
                            _count["fused_fail"], e, traceback.format_exc())
            return None
    if _kernel_ok is False:
        return None  # self-test showed the non-causal ASM kernel is broken -> stay on bf16
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
        real_nhead, qk, vd, qlen, kvlen = 16, 192, 128, 8, 64
        nhead = _num_head_k(real_nhead)
        q = torch.randn(qlen, real_nhead, qk, device=dev).to(fp8)
        k = torch.randn(kvlen, real_nhead, qk, device=dev).to(fp8)
        v = torch.randn(kvlen, real_nhead, vd, device=dev).to(fp8)
        qo = torch.tensor([0, qlen], dtype=torch.int32, device=dev)
        kv = torch.tensor([0, kvlen], dtype=torch.int32, device=dev)
        seq = torch.tensor([kvlen], dtype=torch.int32, device=dev)
        # Tiny one-off buffers sized directly from these synthetic shapes --
        # not _ensure_ps_buffers, which reads backend.vllm_config (unavailable
        # here: see _ensure_ps_buffers' docstring for why).
        ps = _build_ps_buffers(
            nhead, vd, max_num_reqs=1, max_prefill_qlen=qlen,
            max_num_batched_tokens=max(qlen, kvlen), device=dev,
        )
        out, lse = _run_fp8_asm(backend, q, k, v, qo, kv, seq, qlen, False, ps_override=ps)
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

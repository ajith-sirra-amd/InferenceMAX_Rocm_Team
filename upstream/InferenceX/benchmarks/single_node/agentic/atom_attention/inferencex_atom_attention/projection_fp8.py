# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Experimental projection epilogues; not selected by the serving backend.

Projection body adapted from ROCm/aiter gather_kv_b_proj.py at
bbef5a6f2d3733b293c80d2cee6a8d464e88bc01. BF16 arithmetic and
launch partition are retained; only output/statistics handling changes.
"""

import torch
import triton
import triton.language as tl
from aiter.ops.triton._triton_kernels.gather_kv_b_proj import _load_unshuffle_segment
from aiter.ops.triton.utils.device_info import get_num_sms


@triton.jit
def _descale(amax):
    raw = amax * (1.0 / 448.0)
    return tl.where(raw > 0, raw, 1e-6)


@triton.jit
def _reciprocal(x):
    return tl.inline_asm_elementwise(
        "v_rcp_f32 $0, $1;", constraints="=v,v", args=[x],
        dtype=tl.float32, is_pure=True, pack=1,
    )


@triton.jit
def _reduce_projection_amax(Partial, Scales, PARTS: tl.constexpr, PAD: tl.constexpr):
    kind = tl.program_id(0)
    offsets = tl.arange(0, PAD)
    amax = tl.max(tl.load(Partial + kind * PARTS + offsets, offsets < PARTS, 0), 0)
    tl.store(Scales + kind, _descale(amax))


@triton.jit
def _quantize_projection(K, V, K8, V8, Partial, Scales, NK, NV,
                         PARTS: tl.constexpr, PAD: tl.constexpr, BLOCK: tl.constexpr):
    kind = tl.program_id(1)
    src = K if kind == 0 else V
    dst = K8 if kind == 0 else V8
    n = NK if kind == 0 else NV
    parts = tl.arange(0, PAD)
    amax = tl.max(tl.load(Partial + kind * PARTS + parts, parts < PARTS, 0), 0)
    scale = _descale(amax)
    inv = _reciprocal(scale)
    for start in range(tl.program_id(0) * BLOCK, n, tl.num_programs(0) * BLOCK):
        offsets = start + tl.arange(0, BLOCK)
        value = tl.load(src + offsets, offsets < n, 0).to(tl.float32)
        value = tl.minimum(tl.maximum(value * inv, -448.0), 448.0)
        tl.store(dst + offsets, value, offsets < n)
    if tl.program_id(0) == 0:
        tl.store(Scales + kind, scale)


def expand_context_fp8(gathered, scale, row_indices, weight, num_heads, *,
                       mode="fused_amax", output_scales=None):
    """Experimental Kimi BF16 projection with exact per-tensor dynamic FP8 output.

    fused_amax: one projection writes BF16 plus partial maxima; one kernel
    reduces those maxima and quantizes. Saves the separate BF16 amax scan.
    recompute: one projection writes only partial maxima; reduce scales, then
    repeat projection and quantize in registers. No expanded BF16 allocation.
    supplied_scales: one projection quantizes in registers using caller-provided
    positive finite descales. Following ATOM, the experiment derives these from
    new-token K/V with 2x headroom. This changes cached quantization numerics and
    can clip outliers. The caller owns scale validity; no CPU synchronization.
    BF16 rounding is retained in registers to isolate materialization cost.
    No mode changes the projection operand dtype.
    Row indices must be valid offsets into gathered, as in native expand_context.
    """
    if mode not in ("fused_amax", "recompute", "supplied_scales"):
        raise ValueError("mode must be fused_amax, recompute or supplied_scales")
    if (gathered.ndim != 2 or gathered.shape[1] != 576
            or not gathered.is_contiguous()
            or gathered.dtype not in (torch.float8_e4m3fn, torch.float8_e4m3fnuz)):
        raise ValueError("gathered must be contiguous FP8 [rows, 576]")
    if (num_heads <= 0 or weight.shape != (num_heads * 256, 512)
            or weight.dtype != torch.bfloat16 or not weight.is_contiguous()):
        raise ValueError("weight must be contiguous BF16 [heads * 256, 512]")
    if (row_indices.ndim != 1 or not row_indices.is_contiguous()
            or row_indices.dtype not in (torch.int32, torch.int64)):
        raise ValueError("row_indices must be a contiguous integer vector")
    if scale.numel() != 1 or scale.dtype != torch.float32:
        raise ValueError("cache scale must be a single FP32 value")
    if not gathered.is_cuda or any(
        x.device != gathered.device for x in (scale, row_indices, weight)
    ):
        raise ValueError("all inputs must be on the same GPU")
    if mode == "supplied_scales":
        if (output_scales is None or output_scales.shape != (2,)
                or output_scales.dtype != torch.float32
                or output_scales.device != gathered.device
                or not output_scales.is_contiguous()):
            raise ValueError("supplied_scales requires contiguous FP32 [K, V] scales")
    elif output_scales is not None:
        raise ValueError("output_scales is only accepted by supplied_scales mode")
    tokens = row_indices.numel()
    with torch.cuda.device(gathered.device):
        k8 = torch.empty((tokens, num_heads, 192), dtype=torch.float8_e4m3fn,
                         device=gathered.device)
        v8 = torch.empty((tokens, num_heads, 128), dtype=torch.float8_e4m3fn,
                         device=gathered.device)
        scales = (output_scales if output_scales is not None else
                  torch.empty(2, dtype=torch.float32, device=gathered.device))
        if tokens == 0:
            if output_scales is None:
                scales.fill_(1e-6)
            return k8, v8, scales[:1], scales[1:]
        chunks = triton.cdiv(tokens, 32)
        chunk_workers = min(chunks, max(1, triton.cdiv(get_num_sms() * 6, num_heads)))
        workers = num_heads * chunk_workers
        partial = torch.empty((2, workers), device=gathered.device, dtype=torch.float32)
        if mode == "fused_amax":
            k = torch.empty(k8.shape, device=gathered.device, dtype=torch.bfloat16)
            v = torch.empty(v8.shape, device=gathered.device, dtype=torch.bfloat16)
        else:
            # Pointers are not read or written in the stats-only specialization.
            k, v = k8, v8
        kwargs = dict(
            TpNumHeads=num_heads, QkNopeHeadDim=128, VHeadDim=128,
            KV_CDim=512, KV_PeDim=64, ChunkK=32, PaddedK=128, PaddedV=128,
            NO_SCALE=True, GRID_STRIDE=chunk_workers < chunks, num_stages=3,
        )
        if mode == "supplied_scales":
            _projection_with_amax[(workers,)](
                tokens, gathered, scale, row_indices, weight, weight, k8, v8,
                partial, scales, COLLECT_AMAX=False, STORE_OUTPUT=True,
                FP8_OUTPUT=True, **kwargs,
            )
            return k8, v8, scales[:1], scales[1:]
        _projection_with_amax[(workers,)](
            tokens, gathered, scale, row_indices, weight, weight, k, v,
            partial, scales, COLLECT_AMAX=True, STORE_OUTPUT=mode == "fused_amax",
            FP8_OUTPUT=False, **kwargs,
        )
        if mode == "fused_amax":
            block = 8192 if k8.numel() >= 4 * 1024 * 1024 else 4096
            _quantize_projection[(min(1024, triton.cdiv(k8.numel(), block)), 2)](
                k, v, k8, v8, partial, scales, k8.numel(), v8.numel(),
                workers, triton.next_power_of_2(workers), block, num_warps=4,
            )
        else:
            _reduce_projection_amax[(2,)](
                partial, scales, workers, triton.next_power_of_2(workers), num_warps=4,
            )
            _projection_with_amax[(workers,)](
                tokens, gathered, scale, row_indices, weight, weight, k8, v8,
                partial, scales, COLLECT_AMAX=False, STORE_OUTPUT=True,
                FP8_OUTPUT=True, **kwargs,
            )
        return k8, v8, scales[:1], scales[1:]


@triton.jit
def _projection_with_amax(
    total_kv,
    k_buffer,
    k_scale,
    kv_indices,
    kv_proj_weight,
    kv_proj_scale,
    k_prefix,
    v_prefix,
    partial_amax,
    output_scales,
    TpNumHeads: tl.constexpr,
    QkNopeHeadDim: tl.constexpr,
    VHeadDim: tl.constexpr,
    KV_CDim: tl.constexpr,
    KV_PeDim: tl.constexpr,
    ChunkK: tl.constexpr,
    PaddedK: tl.constexpr,
    PaddedV: tl.constexpr,
    WEIGHT_PRESHUFFLE: tl.constexpr = False,
    PER_ROW_SCALE: tl.constexpr = False,
    NO_SCALE: tl.constexpr = False,
    GRID_STRIDE: tl.constexpr = False,
    COLLECT_AMAX: tl.constexpr = True,
    STORE_OUTPUT: tl.constexpr = True,
    FP8_OUTPUT: tl.constexpr = False,
):
    ScaleKGranularity: tl.constexpr = 128
    ScaleNGranularity: tl.constexpr = 128
    assert KV_CDim == 4 * ScaleKGranularity

    flat_pid = tl.program_id(0)
    pid_head = flat_pid % TpNumHeads
    chunk_id = flat_pid // TpNumHeads
    chunk_stride = tl.num_programs(0) // TpNumHeads
    num_chunks = (total_kv + ChunkK - 1) // ChunkK
    if chunk_id >= num_chunks:
        return
    if GRID_STRIDE:
        num_iters = (num_chunks - chunk_id + chunk_stride - 1) // chunk_stride
    else:
        num_iters: tl.constexpr = 1

    stride_k_buffer = tl.full([], KV_CDim + KV_PeDim, dtype=tl.int64)
    stride_k_prefix = tl.full(
        [], TpNumHeads * (QkNopeHeadDim + KV_PeDim), dtype=tl.int64
    )
    stride_v_prefix = tl.full([], TpNumHeads * VHeadDim, dtype=tl.int64)

    k_type = k_buffer.dtype.element_ty
    w_type = kv_proj_weight.dtype.element_ty
    # tl.dot needs both operands in one dtype
    dot_type = (
        w_type if w_type.primitive_bitwidth > k_type.primitive_bitwidth else k_type
    )
    k_scalar_scale = 1.0 if k_type == tl.bfloat16 else tl.load(k_scale)
    offs_n_k = tl.arange(0, PaddedK)
    offs_n_v = tl.arange(0, PaddedV)
    mask_k = offs_n_k < QkNopeHeadDim
    mask_v = offs_n_v < VHeadDim
    offs_k = tl.arange(0, ScaleKGranularity)
    k_head_base = kv_proj_weight + pid_head * (QkNopeHeadDim + VHeadDim) * KV_CDim
    v_head_base = k_head_base + QkNopeHeadDim * KV_CDim

    if NO_SCALE:
        pass
    elif PER_ROW_SCALE:
        row0 = pid_head * (QkNopeHeadDim + VHeadDim)
        k_nope_scale_vec = tl.load(
            kv_proj_scale + row0 + offs_n_k, mask=mask_k, other=1.0
        ).to(tl.float32)
        v_nope_scale_vec = tl.load(
            kv_proj_scale + row0 + QkNopeHeadDim + offs_n_v,
            mask=mask_v,
            other=1.0,
        ).to(tl.float32)
    else:
        num_scale_cols: tl.constexpr = KV_CDim // ScaleKGranularity
        k_abs_rows = pid_head * (QkNopeHeadDim + VHeadDim) + offs_n_k
        v_abs_rows = pid_head * (QkNopeHeadDim + VHeadDim) + QkNopeHeadDim + offs_n_v
        k_scale_n_idx = k_abs_rows // ScaleNGranularity
        v_scale_n_idx = v_abs_rows // ScaleNGranularity

    if WEIGHT_PRESHUFFLE:
        k_nope_weight_0 = _load_unshuffle_segment(
            k_head_base, 0, QkNopeHeadDim, PaddedK, KV_CDim, ScaleKGranularity
        ).to(dot_type)
        k_nope_weight_1 = _load_unshuffle_segment(
            k_head_base, 1, QkNopeHeadDim, PaddedK, KV_CDim, ScaleKGranularity
        ).to(dot_type)
        k_nope_weight_2 = _load_unshuffle_segment(
            k_head_base, 2, QkNopeHeadDim, PaddedK, KV_CDim, ScaleKGranularity
        ).to(dot_type)
        k_nope_weight_3 = _load_unshuffle_segment(
            k_head_base, 3, QkNopeHeadDim, PaddedK, KV_CDim, ScaleKGranularity
        ).to(dot_type)
        v_nope_weight_0 = _load_unshuffle_segment(
            v_head_base, 0, VHeadDim, PaddedV, KV_CDim, ScaleKGranularity
        ).to(dot_type)
        v_nope_weight_1 = _load_unshuffle_segment(
            v_head_base, 1, VHeadDim, PaddedV, KV_CDim, ScaleKGranularity
        ).to(dot_type)
        v_nope_weight_2 = _load_unshuffle_segment(
            v_head_base, 2, VHeadDim, PaddedV, KV_CDim, ScaleKGranularity
        ).to(dot_type)
        v_nope_weight_3 = _load_unshuffle_segment(
            v_head_base, 3, VHeadDim, PaddedV, KV_CDim, ScaleKGranularity
        ).to(dot_type)
    else:
        k_weight_offset = k_head_base + offs_n_k[:, None] * KV_CDim + offs_k[None, :]
        v_weight_offset = v_head_base + offs_n_v[:, None] * KV_CDim + offs_k[None, :]
        k_nope_weight_0 = tl.load(k_weight_offset, mask=mask_k[:, None], other=0.0).to(
            dot_type
        )
        k_nope_weight_1 = tl.load(
            k_weight_offset + ScaleKGranularity,
            mask=mask_k[:, None],
            other=0.0,
        ).to(dot_type)
        k_nope_weight_2 = tl.load(
            k_weight_offset + 2 * ScaleKGranularity,
            mask=mask_k[:, None],
            other=0.0,
        ).to(dot_type)
        k_nope_weight_3 = tl.load(
            k_weight_offset + 3 * ScaleKGranularity,
            mask=mask_k[:, None],
            other=0.0,
        ).to(dot_type)
        v_nope_weight_0 = tl.load(v_weight_offset, mask=mask_v[:, None], other=0.0).to(
            dot_type
        )
        v_nope_weight_1 = tl.load(
            v_weight_offset + ScaleKGranularity,
            mask=mask_v[:, None],
            other=0.0,
        ).to(dot_type)
        v_nope_weight_2 = tl.load(
            v_weight_offset + 2 * ScaleKGranularity,
            mask=mask_v[:, None],
            other=0.0,
        ).to(dot_type)
        v_nope_weight_3 = tl.load(
            v_weight_offset + 3 * ScaleKGranularity,
            mask=mask_v[:, None],
            other=0.0,
        ).to(dot_type)

    if (not NO_SCALE) and (not PER_ROW_SCALE):
        k_nope_scale_0 = tl.load(
            kv_proj_scale + k_scale_n_idx * num_scale_cols,
            mask=mask_k,
            other=0.0,
        ).to(tl.float32)
        k_nope_scale_1 = tl.load(
            kv_proj_scale + k_scale_n_idx * num_scale_cols + 1,
            mask=mask_k,
            other=0.0,
        ).to(tl.float32)
        k_nope_scale_2 = tl.load(
            kv_proj_scale + k_scale_n_idx * num_scale_cols + 2,
            mask=mask_k,
            other=0.0,
        ).to(tl.float32)
        k_nope_scale_3 = tl.load(
            kv_proj_scale + k_scale_n_idx * num_scale_cols + 3,
            mask=mask_k,
            other=0.0,
        ).to(tl.float32)
        v_nope_scale_0 = tl.load(
            kv_proj_scale + v_scale_n_idx * num_scale_cols,
            mask=mask_v,
            other=0.0,
        ).to(tl.float32)
        v_nope_scale_1 = tl.load(
            kv_proj_scale + v_scale_n_idx * num_scale_cols + 1,
            mask=mask_v,
            other=0.0,
        ).to(tl.float32)
        v_nope_scale_2 = tl.load(
            kv_proj_scale + v_scale_n_idx * num_scale_cols + 2,
            mask=mask_v,
            other=0.0,
        ).to(tl.float32)
        v_nope_scale_3 = tl.load(
            kv_proj_scale + v_scale_n_idx * num_scale_cols + 3,
            mask=mask_v,
            other=0.0,
        ).to(tl.float32)

    max_k = tl.full((), 0.0, tl.float32)
    max_v = tl.full((), 0.0, tl.float32)
    if FP8_OUTPUT:
        inv_k = _reciprocal(tl.load(output_scales))
        inv_v = _reciprocal(tl.load(output_scales + 1))

    for _ in range(num_iters):
        token_offsets = chunk_id * ChunkK + tl.arange(0, ChunkK)
        token_mask = token_offsets < total_kv
        kv_block_idx = tl.load(kv_indices + token_offsets, mask=token_mask, other=0)
        data_offset = (
            kv_block_idx[:, None] * stride_k_buffer
            + tl.arange(0, ScaleKGranularity)[None, :]
        )
        kv_c_data_0 = tl.load(
            k_buffer + data_offset, mask=token_mask[:, None], other=0.0
        ).to(dot_type)
        kv_c_data_1 = tl.load(
            k_buffer + data_offset + ScaleKGranularity,
            mask=token_mask[:, None],
            other=0.0,
        ).to(dot_type)
        kv_c_data_2 = tl.load(
            k_buffer + data_offset + 2 * ScaleKGranularity,
            mask=token_mask[:, None],
            other=0.0,
        ).to(dot_type)
        kv_c_data_3 = tl.load(
            k_buffer + data_offset + 3 * ScaleKGranularity,
            mask=token_mask[:, None],
            other=0.0,
        ).to(dot_type)
        kv_pe_data = tl.load(
            k_buffer
            + kv_block_idx[:, None] * stride_k_buffer
            + KV_CDim
            + tl.arange(0, KV_PeDim)[None, :],
            mask=token_mask[:, None],
            other=0.0,
        )
        accum_k = tl.zeros((ChunkK, PaddedK), dtype=tl.float32)
        accum_v = tl.zeros((ChunkK, PaddedV), dtype=tl.float32)

        if NO_SCALE:
            accum_k = tl.dot(kv_c_data_0, k_nope_weight_0.T, acc=accum_k)
            accum_v = tl.dot(kv_c_data_0, v_nope_weight_0.T, acc=accum_v)
            accum_k = tl.dot(kv_c_data_1, k_nope_weight_1.T, acc=accum_k)
            accum_v = tl.dot(kv_c_data_1, v_nope_weight_1.T, acc=accum_v)
            accum_k = tl.dot(kv_c_data_2, k_nope_weight_2.T, acc=accum_k)
            accum_v = tl.dot(kv_c_data_2, v_nope_weight_2.T, acc=accum_v)
            accum_k = tl.dot(kv_c_data_3, k_nope_weight_3.T, acc=accum_k)
            accum_v = tl.dot(kv_c_data_3, v_nope_weight_3.T, acc=accum_v)
        elif PER_ROW_SCALE:
            accum_k += (
                tl.dot(kv_c_data_0, k_nope_weight_0.T) * k_nope_scale_vec[None, :]
            )
            accum_v += (
                tl.dot(kv_c_data_0, v_nope_weight_0.T) * v_nope_scale_vec[None, :]
            )
            accum_k += (
                tl.dot(kv_c_data_1, k_nope_weight_1.T) * k_nope_scale_vec[None, :]
            )
            accum_v += (
                tl.dot(kv_c_data_1, v_nope_weight_1.T) * v_nope_scale_vec[None, :]
            )
            accum_k += (
                tl.dot(kv_c_data_2, k_nope_weight_2.T) * k_nope_scale_vec[None, :]
            )
            accum_v += (
                tl.dot(kv_c_data_2, v_nope_weight_2.T) * v_nope_scale_vec[None, :]
            )
            accum_k += (
                tl.dot(kv_c_data_3, k_nope_weight_3.T) * k_nope_scale_vec[None, :]
            )
            accum_v += (
                tl.dot(kv_c_data_3, v_nope_weight_3.T) * v_nope_scale_vec[None, :]
            )
        else:
            accum_k += tl.dot(kv_c_data_0, k_nope_weight_0.T) * k_nope_scale_0[None, :]
            accum_v += tl.dot(kv_c_data_0, v_nope_weight_0.T) * v_nope_scale_0[None, :]
            accum_k += tl.dot(kv_c_data_1, k_nope_weight_1.T) * k_nope_scale_1[None, :]
            accum_v += tl.dot(kv_c_data_1, v_nope_weight_1.T) * v_nope_scale_1[None, :]
            accum_k += tl.dot(kv_c_data_2, k_nope_weight_2.T) * k_nope_scale_2[None, :]
            accum_v += tl.dot(kv_c_data_2, v_nope_weight_2.T) * v_nope_scale_2[None, :]
            accum_k += tl.dot(kv_c_data_3, k_nope_weight_3.T) * k_nope_scale_3[None, :]
            accum_v += tl.dot(kv_c_data_3, v_nope_weight_3.T) * v_nope_scale_3[None, :]

        accum_k *= k_scalar_scale
        accum_v *= k_scalar_scale
        kv_pe_data *= k_scalar_scale
        # Match the native BF16 store rounding before computing max or FP8.
        accum_k = accum_k.to(tl.bfloat16).to(tl.float32)
        accum_v = accum_v.to(tl.bfloat16).to(tl.float32)
        kv_pe_data = kv_pe_data.to(tl.bfloat16).to(tl.float32)
        if COLLECT_AMAX:
            k_abs = tl.where(token_mask[:, None] & mask_k[None, :], tl.abs(accum_k), 0.0)
            v_abs = tl.where(token_mask[:, None] & mask_v[None, :], tl.abs(accum_v), 0.0)
            rope_abs = tl.where(token_mask[:, None], tl.abs(kv_pe_data), 0.0)
            tile_k = tl.maximum(tl.max(tl.max(k_abs, 1), 0), tl.max(tl.max(rope_abs, 1), 0))
            max_k = tl.maximum(max_k, tile_k)
            max_v = tl.maximum(max_v, tl.max(tl.max(v_abs, 1), 0))
        if FP8_OUTPUT:
            accum_k = tl.minimum(tl.maximum(accum_k * inv_k, -448.0), 448.0)
            accum_v = tl.minimum(tl.maximum(accum_v * inv_v, -448.0), 448.0)
            kv_pe_data = tl.minimum(tl.maximum(kv_pe_data * inv_k, -448.0), 448.0)
        if STORE_OUTPUT:
            tl.store(
                k_prefix
                + token_offsets[:, None] * stride_k_prefix
                + pid_head * (QkNopeHeadDim + KV_PeDim)
                + QkNopeHeadDim
                + tl.arange(0, KV_PeDim)[None, :],
                kv_pe_data,
                mask=token_mask[:, None],
            )
            tl.store(
                k_prefix
                + token_offsets[:, None] * stride_k_prefix
                + pid_head * (QkNopeHeadDim + KV_PeDim)
                + offs_n_k[None, :],
                accum_k,
                mask=token_mask[:, None] & mask_k[None, :],
            )
            tl.store(
                v_prefix
                + token_offsets[:, None] * stride_v_prefix
                + pid_head * VHeadDim
                + offs_n_v[None, :],
                accum_v,
                mask=token_mask[:, None] & mask_v[None, :],
            )
        chunk_id += chunk_stride

    if COLLECT_AMAX:
        tl.store(partial_amax + flat_pid, max_k)
        tl.store(partial_amax + tl.num_programs(0) + flat_pid, max_v)

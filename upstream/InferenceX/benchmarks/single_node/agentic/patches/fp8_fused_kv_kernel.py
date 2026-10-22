# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
# Derived from ROCm/aiter a29547a2f8a986c90b9173fcf13769f6b3cf0502.
# Modification: scale selected cache rows by their source DCP rank.
import triton
import triton.language as tl

@triton.jit
def _load_unshuffle_segment(
    base_ptr,
    seg_idx,
    HeadDim: tl.constexpr,
    PaddedHeadDim: tl.constexpr,
    KV_CDim: tl.constexpr,
    ScaleKGranularity: tl.constexpr,
):
    """Load one [PaddedHeadDim, ScaleKGranularity] weight segment from a
    preshuffled weight matrix via coalesced row-major loads, then unshuffle
    in registers.  PaddedHeadDim is HeadDim rounded up to the next power of 2.
    Out-of-range rows are zero-filled so dot-products stay correct.

    Inverse of ``aiter.ops.shuffle.shuffle_weight``
    """
    NumNBlk: tl.constexpr = HeadDim // 16
    PaddedNumNBlk: tl.constexpr = PaddedHeadDim // 16
    SegKBlocks: tl.constexpr = ScaleKGranularity // 32
    NumKBlkTotal: tl.constexpr = KV_CDim // 32
    PaddedTotalRows: tl.constexpr = PaddedNumNBlk * SegKBlocks
    ElemsPerGroup: tl.constexpr = 128 // base_ptr.type.element_ty.primitive_bitwidth
    GroupsPerKBlk: tl.constexpr = 32 // ElemsPerGroup

    offs_nb = tl.arange(0, PaddedNumNBlk)
    offs_kb = tl.arange(0, SegKBlocks)
    row_indices = (
        offs_nb[:, None] * NumKBlkTotal + seg_idx * SegKBlocks + offs_kb[None, :]
    )
    row_indices_flat = tl.reshape(row_indices, (PaddedTotalRows,))
    mask_flat = tl.reshape(
        (offs_nb[:, None] < NumNBlk).broadcast_to(PaddedNumNBlk, SegKBlocks),
        (PaddedTotalRows,),
    )

    offs_col = tl.arange(0, KV_CDim)
    raw = tl.load(
        base_ptr + row_indices_flat[:, None] * KV_CDim + offs_col[None, :],
        mask=mask_flat[:, None],
        other=0.0,
    )

    w = tl.reshape(
        tl.permute(
            tl.reshape(
                raw, (PaddedNumNBlk, SegKBlocks, GroupsPerKBlk, 16, ElemsPerGroup)
            ),
            (0, 3, 1, 2, 4),
        ),
        (PaddedHeadDim, ScaleKGranularity),
    )
    return w


@triton.jit
def _triton_gather_kv_b_proj_flat(
    total_kv,
    k_buffer,
    k_scale,
    kv_indices,
    kv_proj_weight,
    kv_proj_scale,
    k_prefix,
    v_prefix,
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
    ROWS_PER_RANK: tl.constexpr = 0,
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

        if ROWS_PER_RANK > 0:
            row_scale = tl.load(k_scale + kv_block_idx // ROWS_PER_RANK,
                                mask=token_mask, other=1.0).to(tl.float32)
            accum_k *= row_scale[:, None]
            accum_v *= row_scale[:, None]
            kv_pe_data = kv_pe_data.to(tl.float32) * row_scale[:, None]
        else:
            accum_k *= k_scalar_scale
            accum_v *= k_scalar_scale
            kv_pe_data = kv_pe_data.to(tl.float32) * k_scalar_scale
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


def launch(cache, scales, indices, weight, weight_scale, k, v, shuffled, rows_per_rank):
    import triton
    from aiter.ops.triton.utils.device_info import get_num_sms
    total = indices.numel()
    if not total:
        return
    chunks = triton.cdiv(total, 32)
    workers = min(chunks, max(1, (get_num_sms() * 6 + k.shape[1] - 1) // k.shape[1]))
    _triton_gather_kv_b_proj_flat[(k.shape[1] * workers,)](
        total, cache, scales, indices, weight, weight if weight_scale is None else weight_scale, k, v,
        TpNumHeads=k.shape[1], QkNopeHeadDim=128, VHeadDim=128,
        KV_CDim=512, KV_PeDim=64, ChunkK=32, PaddedK=128, PaddedV=128,
        WEIGHT_PRESHUFFLE=shuffled, PER_ROW_SCALE=weight_scale is not None, NO_SCALE=weight_scale is None,
        GRID_STRIDE=workers < chunks, ROWS_PER_RANK=rows_per_rank,
        num_stages=3)

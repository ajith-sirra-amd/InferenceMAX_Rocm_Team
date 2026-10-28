# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in direct FP8 K/V output using native compact-cache DCP orchestration.

The gather and accumulation loop follows pinned vLLM 1b2a66639.
Projection operands remain BF16. Output scales follow ATOM's new-token policy.
"""
import os

import torch

from vllm.model_executor.layers.attention.mla_attention import (
    accumulate_mla_context_chunk,
    init_mla_context_partial,
)
from vllm.platforms import current_platform
from vllm.v1.attention.backends.mla.rocm_aiter_mla import (
    _supports_fused_dcp_prefill_config,
)
from vllm.v1.attention.ops.rocm_aiter_mla_prefill import (
    expand_context,
    gather_compressed_context,
)

from .projection_fp8 import expand_context_fp8


def eligible(impl, q, cache, metadata):
    prefill = metadata.prefill
    chunked = prefill.chunked_context
    weight = getattr(impl.kv_b_proj, "weight", None)
    return (
        chunked is not None
        and any(c.max_query_len >= 512 and c.max_seq_len >= 8192 for c in chunked.chunks)
        and _supports_fused_dcp_prefill_config(
            impl.kv_cache_dtype, prefill.q_data_type, impl.kv_lora_rank,
            impl.qk_nope_head_dim, impl.qk_rope_head_dim, impl.v_head_dim,
        )
        and q.dtype == torch.bfloat16
        and weight is not None and weight.dtype == torch.bfloat16
        and weight.shape == (impl.num_heads * 256, 512)
        and weight.is_contiguous()
        and getattr(impl.kv_b_proj, "bias", None) is None
        and getattr(impl.kv_b_proj, "weight_scale", None) is None
        and cache.is_contiguous()
        and metadata.dcp_context_row_indices is not None
        and chunked.dcp_manager is not None
    )


def compute_context(impl, q, cache, metadata, k_scale):
    prefill = metadata.prefill
    backend = prefill.prefill_backend
    chunked = prefill.chunked_context
    row_groups = metadata.dcp_context_row_indices
    if len(chunked.chunks) != len(row_groups):
        raise ValueError("DCP chunk/index metadata length mismatch")
    output = output_lse = None
    for chunk, row_indices in zip(chunked.chunks, row_groups):
        gathered = gather_compressed_context(
            cache, chunked.workspace, prefill.block_table[chunk.request_slice],
            chunk, chunked.dcp_manager.kv_gather, current_platform.fp8_dtype(),
        )
        query = q[chunk.token_slice]
        if chunk.max_query_len >= 512 and chunk.max_seq_len >= 8192:
            kv8 = expand_context_fp8(
                gathered, k_scale, row_indices, impl.kv_b_proj.weight, impl.num_heads,
                mode="supplied_scales", output_scales=backend._gather_scales,
            )
            attn_output, attn_lse = backend.run_prefill_projected_fp8(chunk, query, kv8)
            if backend._validate_direct_kv:
                # Graph capture cannot execute host assertions. The numerical
                # smoke explicitly runs eager and checks every optimized call.
                if torch.cuda.is_current_stream_capturing():
                    raise RuntimeError("Direct-KV validation requires eager execution")
                k, v = expand_context(
                    gathered, k_scale, row_indices, chunk.cu_seq_lens,
                    impl.kv_b_proj.weight, impl.num_heads, impl.qk_nope_head_dim,
                    impl.qk_rope_head_dim, impl.v_head_dim,
                )
                backend.validate_projected_fp8(
                    chunk, query, k, v, kv8, attn_output, attn_lse,
                )
            if not getattr(impl, "_direct_kv_reported", False) and not torch.cuda.is_current_stream_capturing():
                print(
                    f"[ATOM-FP8-KV-OUTPUT] pid={os.getpid()} kv_b=BF16 output=FP8 "
                    "scale_source=new_tokens headroom=2 bf16_intermediate=none",
                    flush=True,
                )
                impl._direct_kv_reported = True
        else:
            k, v = expand_context(
                gathered, k_scale, row_indices, chunk.cu_seq_lens,
                impl.kv_b_proj.weight, impl.num_heads, impl.qk_nope_head_dim,
                impl.qk_rope_head_dim, impl.v_head_dim,
            )
            attn_output, attn_lse = backend.run_prefill_context_chunk(chunk, query, k, v)
        if output is None:
            if len(chunked.chunks) == 1 and not chunked.empty_token_slices:
                return attn_output, attn_lse
            output, output_lse = init_mla_context_partial(
                chunked, attn_output, attn_lse, num_tokens=q.shape[0],
            )
        accumulate_mla_context_chunk(chunk, attn_output, attn_lse, output, output_lse)
    return output, output_lse

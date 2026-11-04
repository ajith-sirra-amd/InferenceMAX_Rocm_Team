"""Focused review diagnostics; outlier observations do not certify model accuracy."""
import argparse
import json
import math
import os
from pathlib import Path

os.environ["VLLM_ROCM_USE_AITER"] = "1"
os.environ["VLLM_ROCM_USE_AITER_FLYDSL_FP8_PREFILL_ATTN"] = "1"

import torch


def asm_prefill(output):
    from tests.v1.attention.utils import (
        BatchSpec, create_common_attn_metadata, create_vllm_config,
    )
    from vllm.config.vllm import set_current_vllm_config
    from vllm.distributed import (
        cleanup_dist_env_and_memory, ensure_model_parallel_initialized,
        init_distributed_environment,
    )
    from vllm.model_executor.layers.attention.mla_attention import MLAAttention
    from vllm.model_executor.layers.linear import ColumnParallelLinear
    from vllm.utils.torch_utils import set_default_torch_dtype
    from vllm.v1.attention.backends.mla.prefill.aiter_flash_attn import (
        AiterFlashAttnPrefillBackend,
    )
    from vllm.v1.attention.backends.mla.rocm_aiter_mla import (
        AiterMLABackend, AiterMLAMetadataBuilder,
    )
    from vllm.v1.kv_cache_interface import MLAAttentionSpec
    from vllm.v1.worker.workspace import init_workspace_manager, reset_workspace_manager

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.manual_seed(19)
    heads, tokens, scale = 12, 128, 192**-0.5
    config = create_vllm_config(
        model_name="deepseek-ai/DeepSeek-R1", dtype="bfloat16",
        max_model_len=512, max_num_batched_tokens=512,
        max_num_seqs=1, num_gpu_blocks=32, block_size=16,
        hf_config_override={"num_attention_heads": heads},
    )
    config.cache_config.cache_dtype = "fp8"
    rendezvous = output / "dist-init"
    init_distributed_environment(
        world_size=1, rank=0, local_rank=0,
        distributed_init_method=rendezvous.as_uri(), backend="nccl",
    )
    ensure_model_parallel_initialized(1, 1)
    init_workspace_manager(device)
    try:
        with set_current_vllm_config(config), set_default_torch_dtype(torch.bfloat16):
            projection = ColumnParallelLinear(
                512, heads * 256, bias=False, params_dtype=torch.bfloat16,
                prefix="model.layers.0.self_attn.kv_b_proj", disable_tp=True,
            ).to(device)
            with torch.no_grad():
                projection.weight.normal_(mean=0, std=0.02)
            layer = MLAAttention(
                num_heads=heads, scale=scale, qk_nope_head_dim=128,
                qk_rope_head_dim=64, v_head_dim=128, q_lora_rank=None,
                kv_lora_rank=512, kv_b_proj=projection,
                cache_config=config.cache_config,
                prefix="model.layers.0.self_attn.attn",
                attn_backend=AiterMLABackend,
                prefill_backend_cls=AiterFlashAttnPrefillBackend,
            )
            spec = MLAAttentionSpec(
                block_size=16, num_kv_heads=1, head_size=576,
                dtype=torch.float8_e4m3fn, cache_dtype_str="fp8",
            )
            builder = AiterMLAMetadataBuilder(
                spec, [layer.layer_name], config, device,
            )
            common = create_common_attn_metadata(
                BatchSpec(seq_lens=[tokens], query_lens=[tokens]),
                16, device, arange_block_indices=True,
            )
            metadata = builder.build(common_prefix_len=0, common_attn_metadata=common)
            assert metadata.prefill.chunked_context is None
            assert metadata.fp8_prefill_qo_indptr is not None, "ASM_METADATA_MISSING"
            q = torch.randn(tokens, heads, 192, device=device)
            latent = torch.randn(tokens, 512, device=device)
            rope = torch.randn(tokens, 1, 64, device=device)
            cache = torch.empty(32, 16, 576, device=device, dtype=torch.float8_e4m3fn)
            result = torch.empty(tokens, heads * 128, device=device)
            cache_scale = torch.ones(1, device=device, dtype=torch.float32)
            with torch.inference_mode():
                layer.impl.forward_mha(q, latent, rope, cache, metadata, cache_scale, result)
                projected = projection(latent)[0].view(tokens, heads, 256)
                k = torch.cat((projected[..., :128], rope.expand(-1, heads, -1)), dim=-1)
                v = projected[..., 128:]
                qf, kf, vf = [
                    x.to(torch.float8_e4m3fn).float().transpose(0, 1) for x in (q, k, v)
                ]
                reference = torch.nn.functional.scaled_dot_product_attention(
                    qf, kf, vf, is_causal=True, scale=scale,
                ).transpose(0, 1)
                torch.testing.assert_close(
                    result.view_as(reference).float(), reference, atol=0.1, rtol=0.05,
                )
            report = {
                "status": "passed", "heads": heads, "tokens": tokens,
                "asm_metadata_present": True,
                "max_absolute_error": (result.view_as(reference).float()-reference).abs().max().item(),
                "scope": "real MLA layer, builder and forward_mha; single-rank routing regression",
            }
            (output / "asm.json").write_text(json.dumps(report, indent=2))
            print("ASM_REVIEW_RESULT", json.dumps(report), flush=True)
    finally:
        reset_workspace_manager()
        cleanup_dist_env_and_memory()


def outliers(output):
    from aiter.ops.shuffle import shuffle_weight
    from vllm.model_executor.layers.attention.mla_attention import MLACommonPrefillMetadata
    from vllm.v1.attention.ops.rocm_aiter_mla_fp8_prefill import (
        FP8PrefillContext, ProjectionWeight, project,
    )

    device = torch.device("cuda")
    heads, rows, queries = 12, 128, 16
    weight = torch.zeros(heads * 256, 512, device=device)
    weight[:, 0] = 1
    weight = weight.to(torch.float8_e4m3fn)
    weight_scale = torch.full((heads * 256,), 0.5, device=device)
    cache_scale = torch.tensor([0.25], device=device)
    indices = torch.arange(rows - 1, -1, -1, dtype=torch.int32, device=device)
    q = torch.zeros(queries, heads, 192, dtype=torch.bfloat16, device=device)
    q[..., 0] = 1
    q[..., 128] = 0.5
    new_k = torch.full_like(q, 0.125)
    new_v = torch.full((queries, heads, 128), 0.125, dtype=torch.bfloat16, device=device)
    chunk = MLACommonPrefillMetadata.ContextChunk(
        index=0, request_slice=slice(0, 1), token_slice=slice(0, queries),
        continuation_token_end=0, is_continuation=False, num_context_tokens=rows,
        query_start_loc=torch.tensor([0, queries], dtype=torch.int32, device=device),
        max_query_len=queries,
        cu_seq_lens=torch.tensor([0, rows], dtype=torch.int32, device=device),
        starts=torch.zeros(1, dtype=torch.int32, device=device),
        max_seq_len=rows,
        seq_lens=torch.tensor([rows], dtype=torch.int32, device=device),
        token_to_seq=torch.zeros(rows, dtype=torch.int32, device=device),
        all_rows_active=True,
    )
    observations = []
    for shuffled in (False, True):
        params = ProjectionWeight(
            shuffle_weight(weight, (16, 16)) if shuffled else weight,
            weight_scale, shuffled,
        )
        for multiplier in (1, 2, 8, 32):
            cache_float = torch.zeros(rows, 576, device=device)
            cache_float[:, 0] = 1
            cache_float[-4:, 0] = multiplier
            cache_float[-4:, 512] = multiplier
            cache = cache_float.to(torch.float8_e4m3fn)
            selected = cache_float[indices.long()]
            projected = selected[:, 0] * 0.125
            expected_v = projected[:, None, None].expand(-1, heads, 128)
            expected_k = torch.cat((
                expected_v,
                (selected[:, 512:] * 0.25)[:, None, :].expand(-1, heads, -1),
            ), dim=-1)
            # Independent oracle: each output has exactly one nonzero multiply.
            bf16_k, bf16_v = project(cache, cache_scale, indices, params, heads)
            for actual, expected in ((bf16_k, expected_k), (bf16_v, expected_v)):
                torch.testing.assert_close(actual.float(), expected, rtol=0, atol=0)
            state = FP8PrefillContext()
            state.prepare(q, new_k, new_v)
            exact_scales = torch.stack((
                bf16_k.float().abs().amax(), bf16_v.float().abs().amax(),
            )).clamp_min(448e-6) / 448
            qf = q.double()
            logits = torch.einsum("qhd,khd->hqk", qf, expected_k.double()) / math.sqrt(192)
            reference = torch.einsum(
                "hqk,khd->qhd", logits.softmax(-1), expected_v.double(),
            )
            reference_lse = logits.logsumexp(-1)
            for method, scales in (
                ("atom_new_token_2x", state.gather_scales),
                ("cached_amax_diagnostic_two_projection_passes", exact_scales),
            ):
                kv8 = project(cache, cache_scale, indices, params, heads, scales)
                actual, lse = state.run(chunk, q, kv8, 192**-0.5)
                clipped = [
                    int((expected.abs() > 448 * descale).sum().item())
                    for expected, descale in zip((expected_k, expected_v), scales)
                ]
                row = {
                    "preshuffled": shuffled, "cached_multiplier": multiplier,
                    "method": method, "scales": scales.tolist(),
                    "out_of_range_kv_elements": clipped,
                    "max_output_absolute_error": (actual.double()-reference).abs().max().item(),
                    "max_lse_absolute_error": (lse.double()-reference_lse).abs().max().item(),
                }
                if method.startswith("cached_amax"):
                    assert clipped == [0, 0], row
                    torch.testing.assert_close(actual.double(), reference, atol=2e-3, rtol=2e-3)
                    torch.testing.assert_close(lse.double(), reference_lse, atol=1e-3, rtol=0)
                observations.append(row)
                print("OUTLIER_REVIEW_RESULT", json.dumps(row), flush=True)
    report = {
        "scope": "constructed outliers; cached-amax uses two projections and is not a performance candidate",
        "observations": observations,
    }
    (output / "outliers.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("asm", "outliers"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    (asm_prefill if args.mode == "asm" else outliers)(args.output)

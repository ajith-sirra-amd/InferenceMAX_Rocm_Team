#!/usr/bin/env bash

# ========= Required Env Vars =========
# HF_TOKEN
# HF_HUB_CACHE
# MODEL
# PORT
# TP
# CONC
# MAX_MODEL_LEN

# Reference
# https://amd.atlassian.net/wiki/spaces/RPLBAS/pages/1149960861/WIP+rocm+7.0+rocm7.0_ubuntu_22.04_vllm_0.10.1_instinct_20250927_rc1

max_model_len=$MAX_MODEL_LEN            # Must be >= the input + output length
max_seq_len_to_capture=$MAX_MODEL_LEN   # Beneficial to set this to max_model_len
max_num_seqs=$CONC
max_num_batched_tokens=131072  # Smaller values may result in better TTFT but worse TPOT / Throughput
# Note: this flag may not be compatible with MI325X
#export VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=INT4

unset FLATMM_HIP_CLANG_PATH
export VLLM_USE_V1=1 \
    VLLM_DISABLE_COMPILE_CACHE=1 \
    AMDGCN_USE_BUFFER_OPS=1 \
    VLLM_ROCM_USE_AITER=1 \
    VLLM_TRITON_FP4_GEMM_USE_ASM=0 \
    VLLM_ROCM_USE_AITER_FP4_ASM_GEMM=0 \
    VLLM_ROCM_USE_AITER_MHA=1 \
    VLLM_ROCM_USE_AITER_MLA=0 \
    VLLM_ROCM_USE_CK_MXFP4_MOE=1 \
    VLLM_ROCM_USE_AITER_TRITON_MLA=0 \
    VLLM_ROCM_USE_AITER_TRITON_FUSED_SHARED_EXPERTS=1 \
    VLLM_ROCM_USE_AITER_TRITON_FUSED_RMSNORM_FP4_QUANT=1 \
    VLLM_ROCM_USE_AITER_TRITON_FUSED_ROPE_ZEROS_KV_CACHE=1 \
    VLLM_ROCM_USE_AITER_TRITON_MXFP4_BMM=1 \
    VLLM_ROCM_USE_AITER_TRITON_FUSED_MUL_ADD=1 \
    VLLM_ROCM_USE_AITER_TRITON_FP8_BMM=0

vllm serve ${MODEL} \
    --host=0.0.0.0 \
    --port $PORT \
    --swap-space 64 \
    --tensor-parallel-size $TP \
    --max-num-seqs ${max_num_seqs} \
    --no-enable-prefix-caching \
    --max-num-batched-tokens ${max_num_batched_tokens} \
    --max-model-len ${max_model_len} \
    --block-size 1 \
    --compilation-config '{"pass_config":{"enable_attn_fusion":true,"enable_noop":true,"enable_fusion":true},"cudagraph_mode":"FULL","custom_ops":["+rms_norm","+silu_and_mul","+quant_fp8"],"splitting_ops":[]}' \
    --gpu-memory-utilization 0.95 \
    --max-seq-len-to-capture ${max_seq_len_to_capture} \
    --async-scheduling \
    --kv-cache-dtype fp8

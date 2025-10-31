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
# Note: Using `--kv-cache-dtype fp8` with DeepSeek may cause accuracy issues

# Only for FP4
export VLLM_ROCM_USE_CK_MXFP4_MOE=1

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
    --gpu-memory-utilization 0.95 \
    --max-seq-len-to-capture ${max_seq_len_to_capture} \
    --async-scheduling \
    --kv-cache-dtype auto

#!/usr/bin/env bash

# Source benchmark utilities early
source "$(dirname "$0")/benchmark_lib.sh"

check_env_vars \
    MODEL \
    PORT \
    TP \
    CONC \
    ISL \
    OSL \
    MAX_MODEL_LEN \
    RANDOM_RANGE_RATIO \
    RESULT_FILENAME \
    NUM_PROMPTS

export VLLM_ROCM_USE_AITER=1
export VLLM_ROCM_USE_AITER_MHA=0
export VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION=1
export VLLM_ROCM_USE_AITER_FUSED_MOE_A16W4=1
export VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=INT4

SERVER_LOG=$(mktemp /tmp/server-XXXXXX.log)

set -x
vllm serve $MODEL --port $PORT \
--tensor-parallel-size=$TP \
--gpu-memory-utilization 0.95 \
--max-model-len $MAX_MODEL_LEN \
--block-size=64 \
--no-enable-prefix-caching \
--disable-log-requests \
--async-scheduling > $SERVER_LOG 2>&1 &

SERVER_PID=$!

# Wait for server to be ready
wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

run_benchmark_serving \
    --model "$MODEL" \
    --port "$PORT" \
    --backend vllm \
    --input-len "$ISL" \
    --output-len "$OSL" \
    --random-range-ratio "$RANDOM_RANGE_RATIO" \
    --num-prompts "$NUM_PROMPTS" \
    --max-concurrency "$CONC" \
    --result-filename "$RESULT_FILENAME" \
    --result-dir /workspace/

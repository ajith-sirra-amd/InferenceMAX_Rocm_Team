#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# LongBench-v2 evaluation
# Expects: endpoint [max_context_length] [num_threads] [max_tokens] [num_examples] [categories]

set -e

ENDPOINT=$1
MAX_CONTEXT_LENGTH=${2:-128000}
NUM_THREADS=${3:-16}
MAX_TOKENS=${4:-16384}
NUM_EXAMPLES=${5:-}
CATEGORIES=${6:-}

MODEL_NAME="nvidia/DeepSeek-R1-0528-NVFP4-v2"

echo "LongBench-v2 Config: endpoint=${ENDPOINT}; max_context_length=${MAX_CONTEXT_LENGTH}; num_threads=${NUM_THREADS}; max_tokens=${MAX_TOKENS}; num_examples=${NUM_EXAMPLES:-all}; categories=${CATEGORIES:-all}"

# Create results directory
result_dir="/logs/accuracy"
mkdir -p "$result_dir"

# Set OPENAI_API_KEY if not set
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"

echo "Running LongBench-v2 evaluation..."

# Build command
cmd="python3 -m sglang.test.run_eval \
    --base-url ${ENDPOINT} \
    --model ${MODEL_NAME} \
    --eval-name longbench_v2 \
    --max-tokens ${MAX_TOKENS} \
    --max-context-length ${MAX_CONTEXT_LENGTH} \
    --num-threads ${NUM_THREADS}"

# Add optional arguments
if [ -n "$NUM_EXAMPLES" ]; then
    cmd="$cmd --num-examples ${NUM_EXAMPLES}"
fi

if [ -n "$CATEGORIES" ]; then
    cmd="$cmd --categories ${CATEGORIES}"
fi

echo "Executing: $cmd"
eval "$cmd"

# Copy result files
result_file=$(ls -t /tmp/longbench_v2_*.json 2>/dev/null | head -n1)
if [ -f "$result_file" ]; then
    cp "$result_file" "$result_dir/"
    echo "Results saved to: $result_dir/$(basename "$result_file")"
else
    echo "Warning: Could not find result file in /tmp"
fi

html_file=$(ls -t /tmp/longbench_v2_*.html 2>/dev/null | head -n1)
if [ -f "$html_file" ]; then
    cp "$html_file" "$result_dir/"
    echo "HTML report saved to: $result_dir/$(basename "$html_file")"
fi

echo "LongBench-v2 evaluation complete"


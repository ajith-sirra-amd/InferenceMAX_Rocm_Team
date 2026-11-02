#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Router benchmark using Dynamo's benchmarks/router scripts
# Expects: endpoint [isl] [osl] [requests] [concurrency] [prefix_ratios]

set -e

ENDPOINT=$1
ISL=${2:-14000}
OSL=${3:-200}
REQUESTS=${4:-200}
CONCURRENCY=${5:-20}
PREFIX_RATIOS=${6:-"0.1 0.3 0.5 0.7 0.9"}

# Parse endpoint
HOST=$(echo "$ENDPOINT" | sed 's|http://||' | cut -d: -f1)
PORT=$(echo "$ENDPOINT" | sed 's|http://||' | cut -d: -f2 | cut -d/ -f1)

echo "Router Benchmark Config: endpoint=${ENDPOINT}; isl=${ISL}; osl=${OSL}; requests=${REQUESTS}; concurrency=${CONCURRENCY}; prefix_ratios=${PREFIX_RATIOS}"

# Clone dynamo if not present
DYNAMO_DIR="/tmp/dynamo"
if [ ! -d "$DYNAMO_DIR" ]; then
    echo "Cloning dynamo repository..."
    git clone --depth 1 https://github.com/ai-dynamo/dynamo.git "$DYNAMO_DIR"
fi

# Install dependencies if needed
pip install aiperf matplotlib 2>/dev/null || true

# Run router benchmark
cd "$DYNAMO_DIR/benchmarks/router"

result_dir="/logs/router-bench"
mkdir -p "$result_dir"

echo "Running prefix ratio benchmark..."
echo "Results will be saved to: $result_dir"

# shellcheck disable=SC2086
python prefix_ratio_benchmark.py \
    --prefix-ratios $PREFIX_RATIOS \
    --isl "$ISL" \
    --osl "$OSL" \
    --requests "$REQUESTS" \
    --concurrency "$CONCURRENCY" \
    --output-dir "$result_dir"

echo "Router benchmark complete. Results in $result_dir"


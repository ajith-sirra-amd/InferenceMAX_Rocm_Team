#!/usr/bin/env bash
set -euo pipefail
test "${PROFILE:?}" = 0
source /inferencex/benchmarks/benchmark_lib.sh --validation-only
run_benchmark_serving \
    --model moonshotai/Kimi-K3.1-Pro --tokenizer /model --tokenizer-mode kimi_k3 \
    --port "${PORT:?}" --backend vllm --input-len 8192 --output-len 1024 \
    --random-range-ratio 1 --num-prompts 192 --max-concurrency 48 \
    --result-filename baseline --result-dir /logs --bench-serving-dir /inferencex \
    --use-chat-template --trust-remote-code

#!/usr/bin/env bash

set -x

source "$(dirname "$0")/../benchmark_lib.sh"

export PYTHONDONTWRITEBYTECODE=1

check_env_vars \
    MODEL \
    TP \
    CONC \
    ISL \
    OSL \
    RANDOM_RANGE_RATIO \
    RESULT_FILENAME

if [[ -n "$SLURM_JOB_ID" ]]; then
  echo "JOB $SLURM_JOB_ID running on $SLURMD_NODENAME"
fi

hf download "$MODEL"

export SGLANG_USE_AITER=1

SERVER_LOG=/workspace/server.log
PORT=${PORT:-8888}
MEM_FRAC_STATIC=${MEM_FRAC_STATIC:-0.9}

MEM_FRAC_STATIC=0.9

echo "SCHEDULER_RECV_INTERVAL: $SCHEDULER_RECV_INTERVAL, CONC: $CONC, ISL: $ISL, OSL: $OSL"

# Start GPU monitoring (power, temperature, clocks every second)
start_gpu_monitor

PYTHONNOUSERSITE=1 python3 -m sglang.launch_server --model-path=$MODEL --host=0.0.0.0 --port=$PORT \
--trust-remote-code \
--tensor-parallel-size=$TP --ep-size $EP_SIZE \
--cuda-graph-max-bs $CUDA_GRAPH_MAX_BATCH_SIZE \
--mem-fraction-static $MEM_FRAC_STATIC \
--context-length $CONTEXT_LENGTH --disable-radix-cache \
--attention-backend aiter $EXTRA_ARGS \
--tokenizer-worker-num 4 > $SERVER_LOG 2>&1 &

SERVER_PID=$!

# Wait for server to be ready
wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID" --sleep-interval 60

run_benchmark_serving \
    --model "$MODEL" \
    --port "$PORT" \
    --backend vllm \
    --input-len "$ISL" \
    --output-len "$OSL" \
    --random-range-ratio "$RANDOM_RANGE_RATIO" \
    --num-prompts "$((CONC * 10))" \
    --max-concurrency "$CONC" \
    --metric-percentiles 90,95,99 \
    --result-filename "$RESULT_FILENAME" \
    --result-dir /workspace/

# After throughput, run evaluation only if RUN_EVAL is true
if [ "${RUN_EVAL}" = "true" ]; then
    run_eval --framework lm-eval --port "$PORT"
    append_lm_eval_summary
fi

# Stop GPU monitoring
stop_gpu_monitor
set +x
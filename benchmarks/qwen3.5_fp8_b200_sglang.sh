#!/usr/bin/env bash

source "$(dirname "$0")/benchmark_lib.sh"

check_env_vars \
    MODEL \
    TP \
    CONC \
    ISL \
    OSL \
    RANDOM_RANGE_RATIO \
    RESULT_FILENAME \
    EP_SIZE

if [[ -n "$SLURM_JOB_ID" ]]; then
  echo "JOB $SLURM_JOB_ID running on $SLURMD_NODENAME"
fi

nvidia-smi
hostname
ifconfig

hf download "$MODEL"

export NCCL_NVLS_ENABLE=1
export SGL_ENABLE_JIT_DEEPGEMM=false
export SGLANG_ENABLE_FLASHINFER_GEMM=true
export PYTHONUNBUFFERED=1

SERVER_LOG=/workspace/server.log
PORT=${PORT:-8888}

# Default: recv every ~10 requests; if CONC ≥ 16, relax to ~30 requests between scheduler recv polls.
if [[ $CONC -ge 16 ]]; then
  SCHEDULER_RECV_INTERVAL=30
else
  SCHEDULER_RECV_INTERVAL=10
fi

MEM_FRAC_STATIC=0.82
CHUNKED_PREFILL_SIZE=32768
MAX_PREFILL_TOKENS=32768
CUDA_GRAPH_MAX_BATCH_SIZE=$CONC
MAX_RUNNING_REQUESTS=128
CONTEXT_LENGTH=$((ISL + OSL + 20))

echo "SCHEDULER_RECV_INTERVAL: $SCHEDULER_RECV_INTERVAL, CONC: $CONC, ISL: $ISL, OSL: $OSL"

set -x
PYTHONNOUSERSITE=1 python3 -m sglang.launch_server --model-path=$MODEL --host=0.0.0.0 --port=$PORT \
--served-model-name "Qwen/Qwen3.5-397B-A17B-FP8" --trust-remote-code \
--tensor-parallel-size=$TP --data-parallel-size=1 --ep-size $EP_SIZE \
--enable-flashinfer-allreduce-fusion --attention-backend trtllm_mha \
--tokenizer-worker-num 6 > $SERVER_LOG 2>&1 &

SERVER_PID=$!

# Wait for server to be ready
wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

pip install -q datasets pandas

run_benchmark_serving \
    --model "$MODEL" \
    --port "$PORT" \
    --backend vllm \
    --input-len "$ISL" \
    --output-len "$OSL" \
    --random-range-ratio "$RANDOM_RANGE_RATIO" \
    --num-prompts "$((CONC * 10))" \
    --max-concurrency "$CONC" \
    --result-filename "$RESULT_FILENAME" \
    --result-dir /workspace/

# After throughput, run evaluation only if RUN_EVAL is true
if [ "${RUN_EVAL}" = "true" ]; then
    run_eval --framework lm-eval --port "$PORT" --concurrent-requests $CONC
    append_lm_eval_summary
fi
set +x

# #!/usr/bin/env bash

# source "$(dirname "$0")/benchmark_lib.sh"

# check_env_vars \
#     MODEL \
#     TP \
#     CONC \
#     ISL \
#     OSL \
#     RANDOM_RANGE_RATIO \
#     RESULT_FILENAME

# if [[ -n "$SLURM_JOB_ID" ]]; then
#   echo "JOB $SLURM_JOB_ID running on $SLURMD_NODENAME"
# fi

# nvidia-smi

# hf download "$MODEL"

# SERVER_LOG=/workspace/server.log
# PORT=${PORT:-8888}

# MEM_FRAC_STATIC=0.8

# ps aux

# set -x
# PYTHONNOUSERSITE=1 python3 -m sglang.launch_server \
#     --model-path=$MODEL \
#     --host=0.0.0.0 \
#     --port=$PORT \
#     --tensor-parallel-size=$TP \
#     --mem-fraction-static $MEM_FRAC_STATIC \
#     > $SERVER_LOG 2>&1 &

# SERVER_PID=$!

# # Wait for server to be ready
# wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

# pip install -q datasets pandas
# export PYTHONDONTWRITEBYTECODE=1
# run_benchmark_serving \
#     --model "$MODEL" \
#     --port "$PORT" \
#     --backend vllm \
#     --input-len "$ISL" \
#     --output-len "$OSL" \
#     --random-range-ratio "$RANDOM_RANGE_RATIO" \
#     --num-prompts "$((CONC * 10))" \
#     --max-concurrency "$CONC" \
#     --result-filename "$RESULT_FILENAME" \
#     --result-dir /workspace/

# # After throughput, run evaluation only if RUN_EVAL is true
# if [ "${RUN_EVAL}" = "true" ]; then
#     run_eval --framework lm-eval --port "$PORT" --concurrent-requests $CONC
#     append_lm_eval_summary
# fi
# set +x

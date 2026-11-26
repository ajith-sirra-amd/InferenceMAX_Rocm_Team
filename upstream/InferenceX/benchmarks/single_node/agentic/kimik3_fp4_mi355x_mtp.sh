#!/usr/bin/env bash
set -eo pipefail
source "$(dirname "$0")/../../benchmark_lib.sh"
PROFILE_ENABLED=0
if [[ -n "${PROFILE+x}" ]]; then
    case "$PROFILE" in
        0|1) PROFILE_ENABLED="$PROFILE" ;;
        *) echo 'PROFILE must be 0 or 1' >&2; exit 2 ;;
    esac
fi
for option in "$@"; do
    case "$option" in
        --profile) PROFILE_ENABLED=1 ;;
        --no-profile) PROFILE_ENABLED=0 ;;
        --help|-h)
            echo "Usage: BENCHMARK_MODE=serving|vllm|agentic PROFILE=0|1 $0 [--profile|--no-profile]"
            echo 'Without BENCHMARK_MODE: TEST=1 selects serving; TEST=0 (default) selects agentic.'
            echo 'Defaults: CONC=48 MAX_NUM_SEQS=48 TEST_OSL=1024 TEST_NUM_PROMPTS=192.'
            echo 'Serving: TEST_ISL=8192 TEST_PREFIX_LEN=0. vLLM prefix repetition: TEST_ISL=3584 TEST_PREFIX_LEN=4608.'
            echo 'Synthetic profiling: delay=10s capture=10s. Agentic: DURATION=3600s delay=1800s capture=130s.'
            echo 'Override using PROFILE_DELAY_SECONDS, PROFILE_SECONDS, PROFILE_MIN_SECONDS,'
            echo 'PROFILE_PHASE_TIMEOUT_SECONDS, PROFILE_POLL_SECONDS and PROFILE_EXPORT_TIMEOUT_SECONDS.'
            exit 0 ;;
        *) echo "Unknown option: $option" >&2; exit 2 ;;
    esac
done
TEST=${TEST-0}
case "$TEST" in 0|1) ;; *) echo 'TEST must be 0 or 1' >&2; exit 2 ;; esac
if [[ -z "${BENCHMARK_MODE+x}" ]]; then
    if [[ "$TEST" == 1 ]]; then BENCHMARK_MODE=serving; else BENCHMARK_MODE=agentic; fi
fi
case "$BENCHMARK_MODE" in serving|vllm|agentic) ;; *) echo 'BENCHMARK_MODE must be serving, vllm or agentic' >&2; exit 2 ;; esac
# User-requested defaults; explicit caller values take precedence.
export CONC=${CONC-48}
export MAX_NUM_SEQS=${MAX_NUM_SEQS-48}
export DURATION=${DURATION-3600}
TEST_OSL=${TEST_OSL-1024}
TEST_NUM_PROMPTS=${TEST_NUM_PROMPTS-192}
TEST_RANGE_RATIO=${TEST_RANGE_RATIO-1}
if [[ "$BENCHMARK_MODE" == vllm ]]; then
    TEST_ISL=${TEST_ISL-3584}
    TEST_PREFIX_LEN=${TEST_PREFIX_LEN-4608}
    VLLM_NUM_PREFIXES=${VLLM_NUM_PREFIXES-1}
    VLLM_BENCH_PYTHON=${VLLM_BENCH_PYTHON-/opt/runtime-venv/bin/python}
else
    TEST_ISL=${TEST_ISL-8192}
    TEST_PREFIX_LEN=${TEST_PREFIX_LEN-0}
fi
if [[ "$BENCHMARK_MODE" == agentic ]]; then
    PROFILE_DELAY_SECONDS=${PROFILE_DELAY_SECONDS-1800}
    PROFILE_SECONDS=${PROFILE_SECONDS-130}
    PROFILE_EXPORT_TIMEOUT_SECONDS=${PROFILE_EXPORT_TIMEOUT_SECONDS-600}
else
    PROFILE_DELAY_SECONDS=${PROFILE_DELAY_SECONDS-10}
    PROFILE_SECONDS=${PROFILE_SECONDS-10}
    PROFILE_EXPORT_TIMEOUT_SECONDS=${PROFILE_EXPORT_TIMEOUT_SECONDS-120}
fi
PROFILE_PHASE_TIMEOUT_SECONDS=${PROFILE_PHASE_TIMEOUT_SECONDS-7200}
PROFILE_POLL_SECONDS=${PROFILE_POLL_SECONDS-0.5}
if [[ -z "${PROFILE_MIN_SECONDS+x}" ]]; then
    PROFILE_MIN_SECONDS=$(python3 -c 'import sys; print(float(sys.argv[1]) * 0.8)' "$PROFILE_SECONDS")
fi
check_env_vars MODEL MODEL_PATH DRAFT_MODEL_PATH TP DCP_SIZE CONC MAX_NUM_SEQS EP_SIZE
for setting in CONC MAX_NUM_SEQS; do
    if [[ ! "${!setting}" =~ ^[1-9][0-9]*$ ]]; then
        echo "$setting must be a positive integer" >&2; exit 2
    fi
done
check_env_vars KV_OFFLOADING KV_OFFLOAD_BACKEND TOTAL_CPU_DRAM_GB RESULT_DIR PORT EVAL_ONLY
if [[ "$EVAL_ONLY" != false ]]; then
    echo 'This script pins SA throughput settings with synthetic acceptance; use a separate natural-verification configuration for eval.' >&2
    exit 1
fi
if [[ "$TP" != 8 || "$DCP_SIZE" != 8 || "$EP_SIZE" != 1 ]]; then
    echo 'This script uses SA topology: TP=8 DCP_SIZE=8 EP_SIZE=1' >&2
    exit 1
fi
if [[ "$KV_OFFLOADING" != dram || "$KV_OFFLOAD_BACKEND" != vllm-simple || "$TOTAL_CPU_DRAM_GB" != 1799 ]]; then
    echo 'The SA C48 point uses vllm-simple DRAM offload with TOTAL_CPU_DRAM_GB=1799' >&2
    exit 1
fi
if [[ "$PROFILE_ENABLED" == 1 ]]; then
    check_env_vars PROFILE_DELAY_SECONDS PROFILE_SECONDS PROFILE_MIN_SECONDS
    check_env_vars PROFILE_PHASE_TIMEOUT_SECONDS PROFILE_POLL_SECONDS PROFILE_EXPORT_TIMEOUT_SECONDS
fi
if [[ "$BENCHMARK_MODE" != agentic ]]; then
    check_env_vars TEST_ISL TEST_OSL TEST_RANGE_RATIO TEST_NUM_PROMPTS TEST_PREFIX_LEN RESULT_FILENAME
    if [[ ! "$TEST_PREFIX_LEN" =~ ^[0-9]+$ ]]; then
        echo 'TEST_PREFIX_LEN must be a nonnegative integer (0 disables the shared prefix)' >&2
        exit 2
    fi
    for setting in TEST_ISL TEST_OSL TEST_NUM_PROMPTS; do
        if [[ ! "${!setting}" =~ ^[1-9][0-9]*$ ]]; then
            echo "$setting must be a positive integer" >&2; exit 2
        fi
    done
    PROFILE_WORKLOAD_ARGS=(--workload fixed --result-filename "$RESULT_FILENAME")
    if [[ "$BENCHMARK_MODE" == vllm ]]; then
        check_env_vars VLLM_NUM_PREFIXES VLLM_BENCH_PYTHON
        if [[ ! "$VLLM_NUM_PREFIXES" =~ ^[1-9][0-9]*$ ]]; then
            echo 'VLLM_NUM_PREFIXES must be a positive integer' >&2; exit 2
        fi
        PROFILE_WORKLOAD_ARGS=(--workload vllm --result-filename "$RESULT_FILENAME")
    fi
else
    check_env_vars DURATION INFMAX_CONTAINER_WORKSPACE RESULT_FILENAME
    check_env_vars MODEL_PREFIX FRAMEWORK
    check_env_vars ENABLE_AGENTX_POWER IS_MULTINODE REQUIRE_POWER
    PROFILE_WORKLOAD_ARGS=(--workload agentic --duration "$DURATION")
    if [[ "${AIPERF_EXPERIMENTAL_FAST-}" == 1 ]]; then
        PROFILE_WORKLOAD_ARGS=(--workload agentic --duration 1200)
    fi
fi
for model_directory in "$MODEL_PATH" "$DRAFT_MODEL_PATH"; do
    if [[ ! -d "$model_directory" ]]; then
        echo "Missing model directory: $model_directory" >&2
        exit 1
    fi
done
if [[ "$PROFILE_ENABLED" == 1 ]]; then
    PROFILE_HELPER="$(dirname "$0")/../../profiling/run.py"
    PROFILER_CONFIG=$(python3 "$PROFILE_HELPER" prepare --logs "$RESULT_DIR" "${PROFILE_WORKLOAD_ARGS[@]}" \
        --delay "$PROFILE_DELAY_SECONDS" --seconds "$PROFILE_SECONDS" --minimum "$PROFILE_MIN_SECONDS" \
        --phase-timeout "$PROFILE_PHASE_TIMEOUT_SECONDS" --poll "$PROFILE_POLL_SECONDS" \
        --export-timeout "$PROFILE_EXPORT_TIMEOUT_SECONDS")
fi
if [[ "$BENCHMARK_MODE" == agentic ]]; then
    mkdir -p "$RESULT_DIR/agentic"
    resolve_trace_source
    build_replay_cmd "$RESULT_DIR/agentic"
    export REPLAY_CMD AIPERF_RUNTIME_DIR
fi
wait_for_amd_gpu_clean
rocm-smi || true
export VLLM_ROCM_USE_AITER=1
export VLLM_ROCM_USE_AITER_MOE_SITUV2=1
export SAFETENSORS_FAST_GPU=1
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_FASTSAFETENSORS_QUEUE_SIZE=-1
export VLLM_USE_BREAKABLE_CUDAGRAPH=1
export VLLM_ROCM_AITER_MLA_ASM_PADDING=asm
export VLLM_ROCM_AITER_MLA_DCP_VERIFY=asm
export VLLM_ROCM_DCP_AITER_ALLGATHER=1
export VLLM_ROCM_DCP_VUP_RS=1
export VLLM_ROCM_DSA_MULTISTREAM=1
export VLLM_ROCM_MEM_HEADROOM_GIB=8
export VLLM_K3_SP_PREFILL=1
export VLLM_ROCM_USE_AITER_FLYDSL_SPARSE_MLA=1
export AITER_FLYDSL_SPARSE_MLA=1
export VLLM_ROCM_DSA_DCP_GLOBAL_KEY_PREFILL=1
export VLLM_ROCM_DSA_DCP_ROWSHARD_PREFILL=1
export VLLM_USE_DIRECT_DCP_A2A=0
export VLLM_USE_DIRECT_DCP_Q_GATHER=0
export VLLM_USE_DIRECT_DCP_KV_GATHER=0
export VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=NONE
export AITER_FLYDSL_STAGE2_FP8=0
export AITER_BF16_FP8_MOE_BOUND=0
export HSA_NO_SCRATCH_RECLAIM=1
export THP_MEM_ALLOC_ENABLE=1
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1200
export PYTHONNOUSERSITE=1
export VLLM_ENGINE_READY_TIMEOUT_S=7200
export PYTHONHASHSEED=42

CPU_BYTES_PER_RANK=$(( TOTAL_CPU_DRAM_GB * 1000 * 1000 * 1000 / TP ))
VLLM_CMD=(
    /opt/runtime-venv/bin/vllm serve "$MODEL_PATH" --host 0.0.0.0 --port "$PORT"
    --served-model-name "$MODEL"
    --tensor-parallel-size "$TP"
    --decode-context-parallel-size "$DCP_SIZE"
    --dcp-comm-backend a2a
    --cp-kv-cache-interleave-size 1
    --trust-remote-code
    --moe-backend auto
    --seed 7
    --language-model-only
    --load-format fastsafetensors
    --max-model-len 1048576
    --enable-prefix-caching
    --prefix-match-unit 1152
    --kv-cache-dtype fp8
    --mamba-ssm-cache-dtype float32
    --attention-config '{"indexer_kv_dtype":"mxfp4"}'
    --additional-config '{"kda_prefill_backend":"fused"}'
    --max-num-batched-tokens 10240
    --gpu-memory-utilization 0.91
    --stream-interval 10
    --disable-uvicorn-access-log
    --kv-transfer-config "{\"kv_connector\":\"SimpleCPUOffloadConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"cpu_bytes_to_use_per_rank\":$CPU_BYTES_PER_RANK,\"lazy_offload\":false}}"
    --max-num-seqs "$MAX_NUM_SEQS"
    --compilation-config '{"mode":3,"cudagraph_mode":"FULL_AND_PIECEWISE","max_cudagraph_capture_size":10240,"custom_ops":["+fused_rms_norm_gated"],"cudagraph_capture_sizes":[5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100,105,110,115,120,125,130,135,140,145,150,155,160,165,170,175,180,185,190,195,200,205,210,215,220,225,230,235,240,256,288,320,352,384,416,448,480,512,640,768,896,1024,1280,1536,1792,2048,2560,3072,3584,4096,4608,5120,6144,6400,7168,8192,8704,10240]}'
    --speculative-config "{\"method\": \"dspark\", \"model\": \"$DRAFT_MODEL_PATH\", \"revision\": \"e4e6e6261e9f8b6d5f249f9627eb6b109d762d5f\", \"num_speculative_tokens\": 4, \"attention_backend\": \"TRITON_ATTN\", \"kv_cache_dtype\": \"fp8\", \"draft_sample_method\": \"probabilistic\", \"rejection_sample_method\": \"synthetic\", \"synthetic_acceptance_length\": 3.46}"
    --device-ids 0,1,2,3,4,5,6,7
)

mkdir -p "$RESULT_DIR"
SERVER_LOG="$RESULT_DIR/server.log"
SERVER_PID=""
PROFILE_PID=""
cleanup_services() {
    local exit_code=$?
    trap - EXIT INT TERM
    set +e
    if [[ -n "$PROFILE_PID" ]] && kill -0 "$PROFILE_PID" 2>/dev/null; then
        kill -TERM "$PROFILE_PID"
        wait "$PROFILE_PID"
    fi
    stop_background_process_tree "$SERVER_PID" "vLLM server" 60
    exit "$exit_code"
}
trap cleanup_services EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
if [[ "$PROFILE_ENABLED" == 1 ]]; then
    write_command "$RESULT_DIR/vllm_core_command.txt" "${VLLM_CMD[@]}"
else
    write_command "$RESULT_DIR/vllm_command.txt" "${VLLM_CMD[@]}"
fi
if [[ -n "${SA_CONFIG_AUDIT-}" ]]; then
    check_env_vars SA_REFERENCE_CONFIG
    python3 "$SA_CONFIG_AUDIT" "$SA_REFERENCE_CONFIG" "$RESULT_DIR" "$MODEL_PATH" "$PORT" "${VLLM_CMD[@]}"
fi
if [[ "$PROFILE_ENABLED" == 1 ]]; then
    VLLM_CMD+=(--profiler-config "$PROFILER_CONFIG")
    write_command "$RESULT_DIR/vllm_command.txt" "${VLLM_CMD[@]}"
fi
"${VLLM_CMD[@]}" > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

echo "[workload] mode=$BENCHMARK_MODE concurrency=$CONC max_num_seqs=$MAX_NUM_SEQS profile=$PROFILE_ENABLED"
if [[ "$BENCHMARK_MODE" == serving ]]; then
    echo "[fixed-length] random-input=$TEST_ISL prefix=$TEST_PREFIX_LEN output=$TEST_OSL ratio=$TEST_RANGE_RATIO C=$CONC requests=$TEST_NUM_PROMPTS"
    WORKLOAD_CMD=(run_benchmark_serving
        --model "$MODEL" --tokenizer "$MODEL_PATH" --tokenizer-mode kimi_k3
        --port "$PORT" --backend vllm
        --input-len "$TEST_ISL" --output-len "$TEST_OSL"
        --random-range-ratio "$TEST_RANGE_RATIO"
        --random-prefix-len "$TEST_PREFIX_LEN"
        --num-prompts "$TEST_NUM_PROMPTS" --max-concurrency "$CONC"
        --result-filename "$RESULT_FILENAME" --result-dir "$RESULT_DIR"
        --bench-serving-dir "$INFERENCEX_REPO_ROOT"
        --use-chat-template --trust-remote-code
    )
elif [[ "$BENCHMARK_MODE" == vllm ]]; then
    VLLM_BENCH_CMD=("$VLLM_BENCH_PYTHON" "$(dirname "$0")/../../profiling/vllm_client.py"
        --model "$MODEL" --tokenizer "$MODEL_PATH" --tokenizer-mode kimi_k3
        --backend vllm --base-url "http://127.0.0.1:$PORT" --endpoint /v1/completions
        --dataset-name prefix_repetition
        --prefix-repetition-prefix-len "$TEST_PREFIX_LEN"
        --prefix-repetition-suffix-len "$TEST_ISL"
        --prefix-repetition-num-prefixes "$VLLM_NUM_PREFIXES"
        --prefix-repetition-output-len "$TEST_OSL"
        --num-prompts "$TEST_NUM_PROMPTS" --max-concurrency "$CONC"
        --num-warmups "$((2 * CONC))" --request-rate inf --ignore-eos
        --temperature 0 --seed 42 --trust-remote-code
        --save-result --save-detailed --result-dir "$RESULT_DIR"
        --result-filename "$RESULT_FILENAME.json"
    )
    write_command "$RESULT_DIR/$RESULT_FILENAME-command.txt" "${VLLM_BENCH_CMD[@]}"
    WORKLOAD_CMD=(run_server_client "${VLLM_BENCH_CMD[@]}")
else
    date +%z > "$RESULT_DIR/agentic/agentic_power_timezone_offset.txt"
    WORKLOAD_CMD=(run_agentic_replay_and_write_outputs "$RESULT_DIR/agentic")
fi
if [[ "$PROFILE_ENABLED" == 1 ]]; then
    export INFERENCEX_SERVER_STATE INFERENCEX_SERVER_PID
    python3 "$PROFILE_HELPER" run --logs "$RESULT_DIR" --port "$PORT" -- \
        bash -c 'set -eo pipefail; source "$1"; shift; PROFILE=0 "$@"' -- \
        "$INFERENCEX_REPO_ROOT/benchmarks/benchmark_lib.sh" "${WORKLOAD_CMD[@]}" &
    PROFILE_PID=$!
    wait "$PROFILE_PID"
    PROFILE_PID=""
else
    PROFILE=0 "${WORKLOAD_CMD[@]}"
fi

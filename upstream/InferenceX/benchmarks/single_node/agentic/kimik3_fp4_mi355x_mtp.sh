#!/usr/bin/env bash
set -eo pipefail
source "$(dirname "$0")/../../benchmark_lib.sh"
check_env_vars MODEL MODEL_PATH DRAFT_MODEL_PATH TP DCP_SIZE CONC EP_SIZE
check_env_vars KV_OFFLOADING KV_OFFLOAD_BACKEND TOTAL_CPU_DRAM_GB RESULT_DIR PORT EVAL_ONLY
if [[ "$EVAL_ONLY" != false ]]; then
    echo 'This script pins SA throughput settings with synthetic acceptance; use a separate natural-verification configuration for eval.' >&2
    exit 1
fi
if [[ "$TP" != 8 || "$DCP_SIZE" != 8 || "$CONC" != 48 || "$EP_SIZE" != 1 ]]; then
    echo 'This script pins the SA C48 point: TP=8 DCP_SIZE=8 CONC=48 EP_SIZE=1' >&2
    exit 1
fi
if [[ "$KV_OFFLOADING" != dram || "$KV_OFFLOAD_BACKEND" != vllm-native || "$TOTAL_CPU_DRAM_GB" != 1799 ]]; then
    echo 'The SA C48 point uses vllm-native DRAM offload with TOTAL_CPU_DRAM_GB=1799' >&2
    exit 1
fi
if [[ "${TEST-}" == 1 ]]; then
    check_env_vars TEST_ISL TEST_OSL TEST_RANGE_RATIO TEST_NUM_PROMPTS RESULT_FILENAME
else
    check_env_vars DURATION
fi
for model_directory in "$MODEL_PATH" "$DRAFT_MODEL_PATH"; do
    if [[ ! -d "$model_directory" ]]; then
        echo "Missing model directory: $model_directory" >&2
        exit 1
    fi
done
wait_for_amd_gpu_clean
rocm-smi || true
if [[ "${TEST-}" != 1 ]]; then
    resolve_trace_source
    install_agentic_deps
fi

export VLLM_ROCM_USE_AITER=1
export VLLM_ROCM_USE_AITER_MOE_SITUV2=1
export SAFETENSORS_FAST_GPU=1
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_FASTSAFETENSORS_QUEUE_SIZE=-1
export VLLM_USE_BREAKABLE_CUDAGRAPH=1
export VLLM_ROCM_AITER_MLA_ASM_PADDING=asm
export VLLM_ROCM_AITER_MLA_DCP_VERIFY=asm
export VLLM_ROCM_DCP_AITER_ALLGATHER=1
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
    --prefix-match-unit 64
    --kv-cache-dtype fp8
    --mamba-ssm-cache-dtype float32
    --attention-config '{"indexer_kv_dtype":"mxfp4"}'
    --additional-config '{"kda_prefill_backend":"fused"}'
    --max-num-batched-tokens 10240
    --gpu-memory-utilization 0.91
    --stream-interval 10
    --disable-uvicorn-access-log
    --kv-transfer-config "{\"kv_connector\":\"SimpleCPUOffloadConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"cpu_bytes_to_use_per_rank\":$CPU_BYTES_PER_RANK,\"lazy_offload\":false}}"
    --max-num-seqs "$CONC"
    --prefill-schedule-interval 8
    --long-prefill-token-threshold 10240
    --long-prefill-token-threshold-adaptive
    --compilation-config '{"mode":3,"cudagraph_mode":"FULL_AND_PIECEWISE","max_cudagraph_capture_size":10240,"custom_ops":["+fused_rms_norm_gated"],"cudagraph_capture_sizes":[7,14,21,28,35,42,49,56,63,70,77,84,91,98,105,112,119,126,133,140,147,154,161,168,175,182,189,196,203,210,217,224,231,238,245,252,259,266,273,280,287,294,301,308,315,322,329,336,352,384,416,448,480,512,640,768,896,1024,1280,1536,1792,2048,2560,3072,3584,4096,4608,5120,6144,6400,7168,8192,8704,10240]}'
    --speculative-config "{\"method\": \"dspark\", \"model\": \"$DRAFT_MODEL_PATH\", \"num_speculative_tokens\": 6, \"attention_backend\": \"ROCM_AITER_MLA\", \"kv_cache_dtype\": \"fp8\", \"draft_sample_method\": \"probabilistic\", \"rejection_sample_method\": \"synthetic\", \"synthetic_acceptance_length\": 3.57}"
    --device-ids 0,1,2,3,4,5,6,7
)

mkdir -p "$RESULT_DIR"
SERVER_LOG="$RESULT_DIR/server.log"
SERVER_PID=""
cleanup_services() {
    local exit_code=$?
    trap - EXIT INT TERM
    set +e
    stop_background_process_tree "$SERVER_PID" "vLLM server" 60
    exit "$exit_code"
}
trap cleanup_services EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
write_command "$RESULT_DIR/vllm_command.txt" "${VLLM_CMD[@]}"
if [[ -n "${SA_CONFIG_AUDIT-}" ]]; then
    check_env_vars SA_REFERENCE_CONFIG
    python3 "$SA_CONFIG_AUDIT" "$SA_REFERENCE_CONFIG" "$RESULT_DIR" "$MODEL_PATH" "$PORT" "${VLLM_CMD[@]}"
fi
"${VLLM_CMD[@]}" > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

if [[ "${TEST-}" == 1 ]]; then
    echo "[fixed-length] input=$TEST_ISL output=$TEST_OSL ratio=$TEST_RANGE_RATIO C=$CONC requests=$TEST_NUM_PROMPTS"
    run_benchmark_serving \
        --model "$MODEL" --tokenizer "$MODEL_PATH" --tokenizer-mode kimi_k3 \
        --port "$PORT" --backend vllm \
        --input-len "$TEST_ISL" --output-len "$TEST_OSL" \
        --random-range-ratio "$TEST_RANGE_RATIO" \
        --num-prompts "$TEST_NUM_PROMPTS" --max-concurrency "$CONC" \
        --result-filename "$RESULT_FILENAME" --result-dir "$RESULT_DIR" \
        --bench-serving-dir "$INFERENCEX_REPO_ROOT" \
        --use-chat-template --trust-remote-code
else
    build_replay_cmd "$RESULT_DIR"
    run_agentic_replay_and_write_outputs "$RESULT_DIR"
fi

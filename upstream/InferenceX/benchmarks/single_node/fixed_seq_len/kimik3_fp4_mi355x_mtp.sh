#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/../../benchmark_lib.sh"
wait_for_amd_gpu_clean

check_env_vars MODEL TP CONC ISL OSL MAX_MODEL_LEN RANDOM_RANGE_RATIO RESULT_FILENAME

DP_SIZE=1
export DP_SIZE
TOTAL_RANKS=$(( TP * DP_SIZE ))

if [ -n "${ROCR_VISIBLE_DEVICES:-}" ]; then
    export HIP_VISIBLE_DEVICES="$ROCR_VISIBLE_DEVICES"
fi

if [[ -n "${MODEL_PATH:-}" ]]; then
    if [[ ! -d "$MODEL_PATH" || -z "$(ls -A "$MODEL_PATH" 2>/dev/null)" ]]; then
        hf download "$MODEL" --local-dir "$MODEL_PATH"
    fi
else
    hf download "$MODEL"
    export MODEL_PATH="$MODEL"
fi

rocm-smi || true

export VLLM_ROCM_AITER_MLA_ASM_PADDING=asm
export VLLM_ROCM_USE_AITER=1
export VLLM_ROCM_USE_AITER_MLA=1
export VLLM_ROCM_USE_AITER_MOE=1
export VLLM_ROCM_USE_AITER_MOE_SITUV2_A8W4=1
export VLLM_ROCM_QUICK_REDUCE_QUANTIZATION="${VLLM_ROCM_QUICK_REDUCE_QUANTIZATION:-INT4}"
export VLLM_ROCM_QUICK_REDUCE_CAST_BF16_TO_FP16="${VLLM_ROCM_QUICK_REDUCE_CAST_BF16_TO_FP16:-0}"
export VLLM_ROCM_QUICK_REDUCE_QUANTIZATION_MIN_SIZE_KB="${VLLM_ROCM_QUICK_REDUCE_QUANTIZATION_MIN_SIZE_KB:-256}"

export AITER_SITUV2_A8W4=1
export AITER_FLYDSL_STAGE2_FP8="${AITER_FLYDSL_STAGE2_FP8:-1}"
export AITER_BF16_FP8_MOE_BOUND=0
export AITER_DISABLE_FMHA_OPUS=1
export SAFETENSORS_FAST_GPU=1
export GPU_ARCHS=gfx950
export HSA_NO_SCRATCH_RECLAIM=1
export VLLM_USE_BREAKABLE_CUDAGRAPH=0
export VLLM_K3_KDA_SAFE_STAGES=1
export VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1
export VLLM_ENGINE_READY_TIMEOUT_S=7200
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3600
export PYTHONNOUSERSITE=1
export PYTHONHASHSEED=42

SERVER_LOG="${RESULT_DIR:-/workspace}/server.log"
mkdir -p "$(dirname "$SERVER_LOG")"
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

DRAFT_KV_DTYPE="${DRAFT_KV_DTYPE:-fp8}"
SPEC_NUM_TOKENS="${SPEC_NUM_TOKENS:-3}"
case "$SPEC_NUM_TOKENS" in
    1) SYNTHETIC_ACCEPT_LEN=1.85 ;;
    2) SYNTHETIC_ACCEPT_LEN=2.51 ;;
    3) SYNTHETIC_ACCEPT_LEN=3.00 ;;
    4) SYNTHETIC_ACCEPT_LEN=3.36 ;;
    5) SYNTHETIC_ACCEPT_LEN=3.62 ;;
    6) SYNTHETIC_ACCEPT_LEN=3.75 ;;
    7) SYNTHETIC_ACCEPT_LEN=3.84 ;;
    8) SYNTHETIC_ACCEPT_LEN=4.00 ;;
    *) echo "[spec] no golden AL for k=$SPEC_NUM_TOKENS" >&2; exit 1 ;;
esac
SPEC_BASE="\"model\":\"Inferact/Kimi-K3-DSpark\",\"num_speculative_tokens\":$SPEC_NUM_TOKENS,\"method\":\"dspark\",\"attention_backend\":\"TRITON_MLA\",\"kv_cache_dtype\":\"$DRAFT_KV_DTYPE\",\"draft_sample_method\":\"probabilistic\""
if [ "${EVAL_ONLY:-false}" = "true" ] || [ "${RUN_EVAL:-false}" = "true" ]; then
    SPEC_ARGS=(--speculative-config "{$SPEC_BASE}")
    echo "MTP: k=$SPEC_NUM_TOKENS LIVE block rejection (accuracy gate) draft_kv=$DRAFT_KV_DTYPE"
else
    SPEC_ARGS=(--speculative-config "{$SPEC_BASE,\"rejection_sample_method\": \"synthetic\", \"synthetic_acceptance_length\": $SYNTHETIC_ACCEPT_LEN}")
    echo "MTP: k=$SPEC_NUM_TOKENS synthetic_accept=$SYNTHETIC_ACCEPT_LEN draft_kv=$DRAFT_KV_DTYPE"
fi
SPEC_ROWS=$(( SPEC_NUM_TOKENS + 1 ))

DCP_SIZE="${DCP_SIZE:-1}"
export DCP_SIZE
MAX_NUM_SEQS="${MAX_NUM_SEQS:-$(( CONC * 2 > 2 ? CONC * 2 : 2 ))}"
MAX_BATCHED_TOKENS="${MAX_BATCHED_TOKENS:-8192}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
CUDAGRAPH_MODE="${CUDAGRAPH_MODE:-FULL_DECODE_ONLY}"

LADDER=$(( MAX_NUM_SEQS * SPEC_ROWS ))
CUDAGRAPH_CAPTURE_SIZES=$(seq -s, 1 "$LADDER")
COMPILATION_CONFIG_ARGS=(--compilation-config "{\"mode\":3,\"cudagraph_mode\":\"$CUDAGRAPH_MODE\",\"max_cudagraph_capture_size\":$LADDER,\"custom_ops\":[\"+fused_rms_norm_gated\"],\"cudagraph_capture_sizes\":[$CUDAGRAPH_CAPTURE_SIZES]}")

CP_ARGS=(--attention-backend ROCM_AITER_MLA)
if [ "$DCP_SIZE" -gt 1 ]; then
    CP_ARGS+=(--decode-context-parallel-size "$DCP_SIZE" --dcp-comm-backend a2a --cp-kv-cache-interleave-size 1)
fi

EP_ARGS=()
if [ "${EP_SIZE:-1}" -gt 1 ]; then EP_ARGS=(--enable-expert-parallel); fi

echo "[cfg] conc=$CONC isl=$ISL osl=$OSL dcp=$DCP_SIZE gmu=$GPU_MEM_UTIL mns=$MAX_NUM_SEQS ladder=1..$LADDER spec_rows=$SPEC_ROWS chunk=$MAX_BATCHED_TOKENS cudagraph=$CUDAGRAPH_MODE qr=$VLLM_ROCM_QUICK_REDUCE_QUANTIZATION"

VLLM_CMD=(
    vllm serve "$MODEL_PATH" --served-model-name "$MODEL"
    --host 0.0.0.0
    --port "$PORT"
    --trust-remote-code
    --tensor-parallel-size "$TP"
    --max-model-len "$MAX_MODEL_LEN"
    --gpu-memory-utilization "$GPU_MEM_UTIL"
    --kv-cache-dtype fp8
    --block-size 64
    --max-num-seqs "$MAX_NUM_SEQS"
    --max-num-batched-tokens "$MAX_BATCHED_TOKENS"
    --additional-config '{"kda_prefill_backend":"triton"}'
    "${CP_ARGS[@]}"
    "${EP_ARGS[@]}"
    "${SPEC_ARGS[@]}"
    "${COMPILATION_CONFIG_ARGS[@]}"
)
printf '%q ' "${VLLM_CMD[@]}" | tee "${RESULT_DIR:-/workspace}/vllm_command.txt"
printf '\n' | tee -a "${RESULT_DIR:-/workspace}/vllm_command.txt"

"${VLLM_CMD[@]}" > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo "Server PID: $SERVER_PID"

wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

if [ "${EVAL_ONLY:-false}" = "true" ]; then
    run_eval --framework lm-eval --port "$PORT"
    append_lm_eval_summary
else
    run_benchmark_serving \
        --model "$MODEL" \
        --port "$PORT" \
        --backend vllm \
        --input-len "$ISL" \
        --output-len "$OSL" \
        --random-range-ratio "$RANDOM_RANGE_RATIO" \
        --num-prompts "$(( CONC * 10 ))" \
        --max-concurrency "$CONC" \
        --result-filename "$RESULT_FILENAME" \
        --result-dir /workspace/ \
        --trust-remote-code

    if [ "${RUN_EVAL:-false}" = "true" ]; then
        run_eval --framework lm-eval --port "$PORT"
        append_lm_eval_summary
    fi
fi

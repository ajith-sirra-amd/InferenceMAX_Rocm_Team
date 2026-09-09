#!/usr/bin/env bash
set -euo pipefail
set -x
source "$(dirname "$0")/../../benchmark_lib.sh"
wait_for_amd_gpu_clean

export EVAL_ONLY="${EVAL_ONLY:-false}"
export AIPERF_EXPERIMENTAL_FAST=0
export AIPERF_WARMUP_REQUESTS_PER_LANE=1
check_env_vars MODEL TP CONC KV_OFFLOADING TOTAL_CPU_DRAM_GB RESULT_DIR DURATION EP_SIZE

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
resolve_trace_source
install_agentic_deps

export VLLM_ROCM_AITER_MLA_ASM_PADDING=asm
export VLLM_ROCM_USE_AITER=1
export VLLM_ROCM_USE_AITER_MLA=1
export VLLM_ROCM_USE_AITER_MOE=1
export VLLM_ROCM_USE_AITER_MOE_SITUV2_A8W4=1
# Quick-reduce: quantizes the allreduce payload to cut interconnect traffic.
# Values: FP | INT8 | INT6 | INT4 | INT3 | NONE.  Lower = less traffic, more
# numeric loss. NONE disables it and the two companions below are inert.
#
# CHANGES NUMERICS -- run the GSM8K-200 gate BEFORE trusting any perf number.
# Anchor is 0.995; treat anything below ~0.98 as a fail and back off a level.
#
# Precedent on this same hardware: minimaxm3_fp4_mi355x_mtp.sh:131-133 runs
# INT4 / CAST_BF16_TO_FP16=0 / MIN_SIZE_KB=256. Start there rather than guess.
#   CAST_BF16_TO_FP16=0  MI3xx lacks a bf16 asm path, so vLLM casts bf16->fp16
#                        by default. 0 keeps bf16 and avoids a second rounding.
#   MIN_SIZE_KB=256      skip quick-reduce for small allreduces, where the
#                        quantize/dequantize costs more than the transfer saves.
export VLLM_ROCM_QUICK_REDUCE_QUANTIZATION="${VLLM_ROCM_QUICK_REDUCE_QUANTIZATION:-NONE}"
export VLLM_ROCM_QUICK_REDUCE_CAST_BF16_TO_FP16="${VLLM_ROCM_QUICK_REDUCE_CAST_BF16_TO_FP16:-0}"
export VLLM_ROCM_QUICK_REDUCE_QUANTIZATION_MIN_SIZE_KB="${VLLM_ROCM_QUICK_REDUCE_QUANTIZATION_MIN_SIZE_KB:-256}"
export AITER_SITUV2_A8W4=1
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
export AIPERF_HTTP_TCP_USER_TIMEOUT=900000
export PYTHONNOUSERSITE=1
export PYTHONHASHSEED=42

# Per-PR runtime patching. Flip a flag to 1 to apply that PR. The three touch
# disjoint files, so any combination is valid and order does not matter. Only
# meaningful on an image WITHOUT the PRs already baked in -- against a patched
# image the hunks are present, patch --forward exits non-zero, and the PR is
# reported failed when nothing is wrong.
#   54736  SimpleCPU fine-grained hybrid prefix hits (carries 54735). The only
#          one measured as load-bearing: bare could not finish warmup (T289).
#   52968  DRAFT PR. Never isolated; effect unknown.
#   54889  Fuse empty-shard LSE mask into A2A pack kernel. +0.74%, inside noise.
# export is required: apply_prs.sh is a subprocess and will not see plain vars.
export APPLY_PR_54736="${APPLY_PR_54736:-0}"
export APPLY_PR_52968="${APPLY_PR_52968:-0}"
export APPLY_PR_54889="${APPLY_PR_54889:-0}"
"$(cd "$(dirname "$0")" && pwd)/k3_patches/apply_prs.sh" || true

SERVER_LOG="$RESULT_DIR/server.log"
mkdir -p "$RESULT_DIR"
SERVER_PID=""

cleanup_agentic_services() {
    local exit_code=$?
    trap - EXIT INT TERM
    set +e
    stop_background_process_tree "$SERVER_PID" "vLLM server" 60
    exit "$exit_code"
}
trap cleanup_agentic_services EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# C1 is latency-bound and C72/C76 are throughput-bound; they need different
# geometry. DCP>1 and MTP are mutually exclusive - the MTP draft uses TRITON_MLA,
# which rejects non-causal MLA under DCP - so C1 runs DCP 1 + MTP, and the high
# concurrencies run DCP 8 without spec-decode.
SPEC_ARGS=()
SPEC_ROWS=1
case "$CONC" in
    1|2|4)
        DCP_SIZE="${DCP_SIZE:-1}"
        SPEC_NUM_TOKENS="${SPEC_NUM_TOKENS:-8}"
        # Golden Kimi-K3 acceptance curve. Without rejection_sample_method
        # synthetic + this length, the run uses LIVE draft acceptance and the
        # number is not comparable to other MTP submissions.
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
        DRAFT_KV_DTYPE="${DRAFT_KV_DTYPE:-fp8}"
        SPEC_ARGS=(--speculative-config "{\"model\":\"Inferact/Kimi-K3-DSpark\",\"num_speculative_tokens\":$SPEC_NUM_TOKENS,\"method\":\"dspark\",\"attention_backend\":\"TRITON_MLA\",\"kv_cache_dtype\":\"$DRAFT_KV_DTYPE\",\"draft_sample_method\":\"probabilistic\",\"rejection_sample_method\": \"synthetic\", \"synthetic_acceptance_length\": $SYNTHETIC_ACCEPT_LEN}")
        echo "MTP: k=$SPEC_NUM_TOKENS synthetic_accept=$SYNTHETIC_ACCEPT_LEN draft_kv=$DRAFT_KV_DTYPE"
        SPEC_ROWS=$(( SPEC_NUM_TOKENS + 1 ))
        MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"
        MAX_BATCHED_TOKENS="${MAX_BATCHED_TOKENS:-8192}"
        ;;
    *)
        DCP_SIZE="${DCP_SIZE:-8}"
        MAX_BATCHED_TOKENS="${MAX_BATCHED_TOKENS:-16384}"
        # mns must stay above the peak running count or batches fall off the
        # cudagraph ladder into eager execution, measured at -39.5%.
        # Peak observed at C72 was 88, i.e. CONC+16.
        if [ "$CONC" -le 72 ]; then MAX_NUM_SEQS="${MAX_NUM_SEQS:-96}"
        else MAX_NUM_SEQS="${MAX_NUM_SEQS:-112}"; fi
        ;;
esac
export DCP_SIZE

GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
CUDAGRAPH_MODE="${CUDAGRAPH_MODE:-FULL_DECODE_ONLY}"

LADDER=$(( MAX_NUM_SEQS * SPEC_ROWS ))
CUDAGRAPH_CAPTURE_SIZES=$(seq -s, 1 "$LADDER")
COMPILATION_CONFIG_ARGS=(--compilation-config "{\"mode\":3,\"cudagraph_mode\":\"$CUDAGRAPH_MODE\",\"max_cudagraph_capture_size\":$LADDER,\"custom_ops\":[\"+fused_rms_norm_gated\"],\"cudagraph_capture_sizes\":[$CUDAGRAPH_CAPTURE_SIZES]}")

CP_ARGS=(--attention-backend ROCM_AITER_MLA)
if [ "$DCP_SIZE" -gt 1 ]; then
    CP_ARGS+=(--decode-context-parallel-size "$DCP_SIZE" --dcp-comm-backend a2a --cp-kv-cache-interleave-size 1)
fi

OFFLOAD_ARGS=()
if agentic_kv_offload_enabled; then
    CPU_BYTES_PER_RANK=$(( TOTAL_CPU_DRAM_GB * 1000 * 1000 * 1000 / TOTAL_RANKS ))
    OFFLOAD_ARGS=(--kv-transfer-config "{\"kv_connector\":\"SimpleCPUOffloadConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"cpu_bytes_to_use_per_rank\":$CPU_BYTES_PER_RANK,\"lazy_offload\":false}}")
fi

EP_ARGS=()
if [ "${EP_SIZE:-1}" -gt 1 ]; then EP_ARGS=(--enable-expert-parallel); fi

echo "[cfg] conc=$CONC dcp=$DCP_SIZE gmu=$GPU_MEM_UTIL mns=$MAX_NUM_SEQS ladder=1..$LADDER spec_rows=$SPEC_ROWS chunk=$MAX_BATCHED_TOKENS cudagraph=$CUDAGRAPH_MODE offload=${KV_OFFLOADING:-none}"

VLLM_CMD=(
    vllm serve "$MODEL_PATH" --served-model-name "$MODEL"
    --host 0.0.0.0
    --port "$PORT"
    --trust-remote-code
    --moe-backend auto
    --tensor-parallel-size "$TP"
    --load-format fastsafetensors
    --gpu-memory-utilization "$GPU_MEM_UTIL"
    --language-model-only
    --max-num-seqs "$MAX_NUM_SEQS"
    --max-num-batched-tokens "$MAX_BATCHED_TOKENS"
    --max-model-len 1048576
    --kv-cache-dtype fp8
    --enable-auto-tool-choice
    --tool-call-parser kimi_k3
    --reasoning-parser kimi_k3
    --enable-prefix-caching
    --enable-prompt-tokens-details
    --no-async-scheduling
    --attention-config '{"mla_prefill_backend":"ROCM_AITER_FA"}'
    "${OFFLOAD_ARGS[@]}"
    "${CP_ARGS[@]}"
    "${EP_ARGS[@]}"
    "${SPEC_ARGS[@]}"
    "${COMPILATION_CONFIG_ARGS[@]}"
)

printf '%q ' "${VLLM_CMD[@]}" | tee "$RESULT_DIR/vllm_command.txt"
printf '\n' | tee -a "$RESULT_DIR/vllm_command.txt"

"${VLLM_CMD[@]}" > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo "Server PID: $SERVER_PID"

wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

if [ "${EVAL_ONLY:-false}" = "true" ]; then
    run_eval --port "$PORT"
else
    build_replay_cmd "$RESULT_DIR"
    run_agentic_replay_and_write_outputs "$RESULT_DIR"
fi

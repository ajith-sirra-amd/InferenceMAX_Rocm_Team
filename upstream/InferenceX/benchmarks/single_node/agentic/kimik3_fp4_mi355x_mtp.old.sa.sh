#!/usr/bin/env bash
# Kimi-K3 FP4 / MI355X on a BARE upstream nightly -- no overlay, no PR stack,
# nothing patched at runtime.
#
# Target base: vllm/vllm-openai-rocm:nightly-7c5dc571cbd1064ecc8a9b1045637ff647aa22cb
# That nightly already contains four of the carried PRs (#51705 DSpark/DCP
# attention+verification, #53598 DCP prefix-cache hits, #52707 negative external
# block allocation, #52033 ROCm dual-stream decode), none of which are in the
# older 46638857 base the overlay was cut against.
#
# Measured on this stack:
#   C1  -> TPOT 8.52 ms mean / 8.90 ms p99   (T213) -- best C1 in the ledger,
#          better than the fully patched image (9.06 / 9.31).
#
# Intended sweep: CONC = 1, 8, 16, 32, 52.
#   C1        -> MTP on (only if the yaml says so), DCP OFF
#   C8..C52   -> DCP 8, no spec
set -euo pipefail
set -x
source "$(dirname "$0")/../../benchmark_lib.sh"
wait_for_amd_gpu_clean

export EVAL_ONLY="${EVAL_ONLY:-false}"
export EVAL_LIMIT="${EVAL_LIMIT:-200}"
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

# Nothing is patched here. Make sure the legacy in-container patcher stays off,
# and shout if someone points this script at a pre-baked image by mistake --
# the numbers would not be comparable.
export SKIP_KIMI_PATCHES=1
if [ -f /etc/k3-image-manifest ]; then
    echo "[warn] this image is PRE-PATCHED (/etc/k3-image-manifest present)." >&2
    echo "[warn] this script is for a BARE nightly; results are not comparable." >&2
    sed 's/^/[k3-image] /' /etc/k3-image-manifest >&2
else
    echo "[bare] no image manifest -- unpatched nightly, as intended"
fi

export VLLM_ROCM_AITER_MLA_ASM_PADDING=asm
export VLLM_ROCM_USE_AITER=1
export VLLM_ROCM_USE_AITER_MLA=1
export VLLM_ROCM_USE_AITER_MOE=1
export VLLM_ROCM_USE_AITER_MOE_SITUV2_A8W4=1
export VLLM_ROCM_QUICK_REDUCE_QUANTIZATION="${VLLM_ROCM_QUICK_REDUCE_QUANTIZATION:-NONE}"
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

# ---- spec: MTP only at CONC<=4, and only if the yaml asked for it ------------
# SPEC_DECODING comes from the matrix row (spec-decoding: mtp). Above C4 the
# drafter is off and SPEC_ROWS is 1.
SPEC_ARGS=()
SPEC_ROWS=1
if [ "${SPEC_DECODING:-}" = "mtp" ] && [ "$CONC" -le 4 ]; then
    SPEC_NUM_TOKENS="${SPEC_NUM_TOKENS:-8}"
    SPEC_ROWS=$(( SPEC_NUM_TOKENS + 1 ))
    SPEC_ARGS=(--speculative-config "{\"model\":\"Inferact/Kimi-K3-DSpark\",\"num_speculative_tokens\":$SPEC_NUM_TOKENS,\"method\":\"dspark\",\"attention_backend\":\"TRITON_MLA\",\"kv_cache_dtype\":\"fp8\",\"draft_sample_method\":\"probabilistic\",\"rejection_sample_method\":\"synthetic\",\"synthetic_acceptance_length\":4.0}")
fi

# ---- mns: track concurrency, never leave dead capture ------------------------
# THE key finding (T208). At C1 with MTP the batch is ALWAYS 1 seq x 9 spec
# rows. Capturing 1..72 (the old mns=8 default) left 63 dead graph sizes and
# every step landing on a mismatched bucket paid for it:
#
#   mns 8, ladder 1..72 -> 9.69 ms mean, 11.70 p99, p50..p99.9 spread 2.64 ms
#   mns 1, ladder 1..9  -> 9.06 ms mean,  9.31 p99, spread 0.22 ms   (-92%)
#
# Same logic applies at C8/C16/C32: a fixed mns=80 would capture 80 sizes for a
# batch that never exceeds ~conc. So mns tracks conc with 25% headroom for
# agentic lanes, capped at 80.
MAX_NUM_SEQS="${MAX_NUM_SEQS:-$(( CONC + CONC / 4 ))}"
if [ "$MAX_NUM_SEQS" -lt 1 ];  then MAX_NUM_SEQS=1;  fi
if [ "$MAX_NUM_SEQS" -gt 80 ]; then MAX_NUM_SEQS=80; fi
#   C1 -> 1   C8 -> 10   C16 -> 20   C32 -> 40   C52 -> 65

# ---- ladder: ALWAYS covers mns x SPEC_ROWS, never more, never less ----------
# Less than the max batch is the signature that precedes
# HSA_STATUS_ERROR_OUT_OF_RESOURCES. More is the dead-capture jitter above.
LADDER=$(( MAX_NUM_SEQS * SPEC_ROWS ))
CUDAGRAPH_CAPTURE_SIZES=$(seq -s, 1 "$LADDER")
COMPILATION_CONFIG_ARGS=(--compilation-config "{\"mode\":3,\"cudagraph_mode\":\"FULL_AND_PIECEWISE\",\"max_cudagraph_capture_size\":$LADDER,\"custom_ops\":[\"+fused_rms_norm_gated\"],\"cudagraph_capture_sizes\":[$CUDAGRAPH_CAPTURE_SIZES]}")

# ---- DCP: off at C<=4 (MTP path), 8 above -----------------------------------
# NO VLLM_USE_DIRECT_DCP_* / VLLM_DCP_Q_REPLICATE overrides. Those were
# aigmkt-image workarounds and they hang _ALLGATHER on a modern base (T184).
if [ "$CONC" -le 4 ]; then DCP_SIZE=1; else DCP_SIZE=8; fi
export DCP_SIZE
CP_ARGS=(--attention-backend ROCM_AITER_MLA)
if [ "$DCP_SIZE" -gt 1 ]; then
    CP_ARGS+=(--decode-context-parallel-size "$DCP_SIZE" --dcp-comm-backend a2a --cp-kv-cache-interleave-size 1)
fi

# ---- chunk ------------------------------------------------------------------
# C1: 8192. 16384 costs p99 (12.31 vs 9.18 on the older stack); 4096 buys
# nothing on TPOT and costs 27% TTFT (T210). 8192 is the measured floor.
# C>4: 16384. Measured flat vs 8192 at C72 (T199, 0.07%), and larger keeps
# prefill cheaper.
if [ "$CONC" -le 4 ]; then MAX_BATCHED_TOKENS=8192; else MAX_BATCHED_TOKENS=16384; fi

# 0.92 was NEVER measured before and is catastrophic at C1 (T211: mean
# 9.06 -> 21.61 ms, p90 9.26 -> 43.40). 0.9 everywhere.
GPU_MEM_UTIL=0.9

OFFLOAD_ARGS=()
if agentic_kv_offload_enabled; then
    CPU_BYTES_PER_RANK=$(( TOTAL_CPU_DRAM_GB * 1000 * 1000 * 1000 / TOTAL_RANKS ))
    OFFLOAD_ARGS=(--kv-transfer-config "{\"kv_connector\":\"SimpleCPUOffloadConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"cpu_bytes_to_use_per_rank\":$CPU_BYTES_PER_RANK,\"lazy_offload\":false}}")
fi

EP_ARGS=()
if [ "${EP_SIZE:-1}" -gt 1 ]; then EP_ARGS=(--enable-expert-parallel); fi

echo "[cfg] conc=$CONC dcp=$DCP_SIZE gmu=$GPU_MEM_UTIL mns=$MAX_NUM_SEQS ladder=1..$LADDER (mns x $SPEC_ROWS rows) chunk=$MAX_BATCHED_TOKENS spec=${#SPEC_ARGS[@]} offload=${KV_OFFLOADING:-none}"

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

# GSM8K is only meaningful where spec is OFF. At CONC<=4 the drafter uses
# rejection_sample_method=synthetic with synthetic_acceptance_length=4.0, which
# accepts a fixed number of draft tokens regardless of the target model, so the
# emitted text is not the model's distribution and any accuracy number from it
# is noise. Run accuracy at C52.
if [ "${EVAL_ONLY:-false}" = "true" ]; then
    if [ "$CONC" -le 4 ]; then
        echo "[eval] WARNING: GSM8K at CONC<=4 is structurally invalid (synthetic acceptance). Use C52." >&2
    fi
    run_eval --port "$PORT"
else
    build_replay_cmd "$RESULT_DIR"
    run_agentic_replay_and_write_outputs "$RESULT_DIR"
fi

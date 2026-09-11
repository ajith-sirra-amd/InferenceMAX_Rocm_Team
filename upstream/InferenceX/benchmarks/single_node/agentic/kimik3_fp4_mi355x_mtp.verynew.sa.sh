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
# DEFAULT IS INT4 (owner decision 2026-09-09). This is ON unless a run sets
# VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=NONE explicitly. NOT YET GSM8K-GATED on
# Kimi-K3 -- the INT4 precedent is MiniMax-M3, a different model. Gate before
# trusting any accuracy-sensitive result from this script.
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
# ===== ACTIVE: quick-reduce ON, INT4 =========================================
# MEASURED 2026-09-11: INT4 at gmu 0.90 HANGS THE ENGINE. It allocates extra
# allreduce buffers the memory profiler does not account for, the KFD driver
# then thrashes evicting/restoring GPU pages (kworker kfd_restore_wq ~90% CPU,
# GPUs 0%, warmup flat, NO RCCL watchdog line) and never recovers.
# Run 34503406329. Same failure class as VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0
# and gmu>0.90. INT4 only survives at gmu<=0.88, where the gmu cost (-11% KV)
# exceeds anything INT4 returns. DEFAULT IS NONE.
export VLLM_ROCM_QUICK_REDUCE_QUANTIZATION="${VLLM_ROCM_QUICK_REDUCE_QUANTIZATION:-NONE}"
export VLLM_ROCM_QUICK_REDUCE_CAST_BF16_TO_FP16="${VLLM_ROCM_QUICK_REDUCE_CAST_BF16_TO_FP16:-1}"
export VLLM_ROCM_QUICK_REDUCE_QUANTIZATION_MIN_SIZE_KB="${VLLM_ROCM_QUICK_REDUCE_QUANTIZATION_MIN_SIZE_KB:-256}"

# ===== ALTERNATIVE: quick-reduce OFF (upstream stock) ========================
# Comment the THREE lines above, uncomment the THREE below.
#export VLLM_ROCM_QUICK_REDUCE_QUANTIZATION="${VLLM_ROCM_QUICK_REDUCE_QUANTIZATION:-NONE}"
#export VLLM_ROCM_QUICK_REDUCE_CAST_BF16_TO_FP16="${VLLM_ROCM_QUICK_REDUCE_CAST_BF16_TO_FP16:-1}"
#export VLLM_ROCM_QUICK_REDUCE_QUANTIZATION_MIN_SIZE_KB="${VLLM_ROCM_QUICK_REDUCE_QUANTIZATION_MIN_SIZE_KB:-256}"
#
# Back off one level at a time if INT4 fails the GSM8K gate. Swap INT4 for:
#   INT6   less loss, less saving
#   INT8   conservative
#   FP     least loss, smallest saving -- skip unless INT8 also fails
#   INT3   NOT recommended: nothing in this repo uses it
#
# WARNING -- do NOT leave both blocks uncommented. These use ${VAR:-default},
# so the FIRST assignment wins and the second is silently ignored: the file
# would read as OFF while the run is actually INT4. Exactly one block active.
export AITER_SITUV2_A8W4=1
# MEASURED +0.6% (11,990 -> 12,064 @ C72, run 34511864403). Our only confirmed
# win on the current stack. Supported in aiter/fused_moe.py:2092; raises loudly
# if model_dim is not divisible, so a bad build fails fast rather than silently.
# Do NOT swap A8W4 for A4W4: A4W4 FAILED the GSM8K gate (0.975 vs 0.995 anchor).
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
export AIPERF_HTTP_TCP_USER_TIMEOUT=900000
export PYTHONNOUSERSITE=1
export PYTHONHASHSEED=42

# No runtime PR patching in this script. It targets a nightly cut AFTER the
# relevant merges, so the PRs arrive in the image itself:
#   54736  SimpleCPU fine-grained hybrid prefix hits  -- merged 2026-09-11 01:36
#   54889  Fuse empty-shard LSE mask into A2A pack    -- merged 2026-09-10 15:43
#   54038  KDA prefill fused kernels                  -- merged 2026-09-10 07:37
# 54038 could not be patched in even if we wanted to: it ships a HIP .cu plus
# torch bindings, so the Python half alone would call torch.ops._C.fused_kda_chunk,
# which does not exist without a rebuild. Only a nightly delivers it.
#
# PICK THE NIGHTLY BY SOURCE COMMIT, NOT TAG DATE. rocm/vllm-dev tags are named
# by build date and can lag the source badly (nightly_cdna4_..._0910_b230 was
# built 09-10 but its vLLM is from 09-01). The vllm/vllm-openai-rocm:nightly-<sha>
# line puts the commit in the tag and runs ~1 h behind main. Verify inside the
# image before trusting a run:
#     python3 -c "import torch; print(hasattr(torch.ops._C,'fused_kda_chunk'))"
# True means 54038 is really there.
#
# 52968 (attn res + sigmoid_mul + conv fusions) is still an OPEN DRAFT and is
# deliberately not included.

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
        # mnbt: 16384 is the proven operating point (12,064 @ C72, n>=2 basis).
        # Measured KV pool vs chunk: 8192 -> 30,089,572 | 16384 -> 28,733,261
        # 24576 -> 27,160,397 | 32768 -> 27,319,963 but DIES in warmup
        # deterministically on the agentic replay (T275/T276, same trace both
        # times) -- do not use 32768. 24576 produced the single highest number
        # ever seen (12,161, n=1, older stack) and is the one upside worth a
        # try; it costs -5.5% KV, affordable at C72 where usage is only ~66%.
        MAX_BATCHED_TOKENS="${MAX_BATCHED_TOKENS:-16384}"
        # mns must stay above the peak running count or batches fall off the
        # cudagraph ladder into eager execution, measured at -39.5%.
        # Peak observed at C72 was 88, i.e. CONC+16.
        # C72 -> 96 gives 8 slots over the observed peak running count (88).
        # Above 72 the peak scales too, so 112. mns is MEMORY-NEUTRAL (96->140
        # moved the KV pool by only -0.28%: cudagraphs share one pool, so a
        # longer ladder costs almost nothing) -- the only risk of raising it is
        # capture time, and the only risk of lowering it is falling off the
        # ladder into eager decode, measured at -39.5%.
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

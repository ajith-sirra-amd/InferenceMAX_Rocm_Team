#!/usr/bin/env bash
set -euo pipefail
set -x
source "$(dirname "$0")/../../benchmark_lib.sh"
wait_for_amd_gpu_clean

# T154: GSM8K accuracy gate BEFORE any perf number, because the #51705 rebase
# hand-merged 3 conflicts in kimi_k3/nvidia/mla.py including the MLA forward
# body, and rejection_sample_method "synthetic" IMPOSES the accept length, so a
# draft numerics regression cannot show up as a lower AL. Perf runs are blind to
# it; only accuracy is not.
# Forced, not defaulted: the runner exports EVAL_ONLY=false, so ${EVAL_ONLY:-...}
# never fires -- same trap as ISL/OSL. Set FORCE_EVAL=0 to go back to benchmarks.
if [ "${FORCE_EVAL:-1}" = "1" ]; then EVAL_ONLY=true; else EVAL_ONLY="${EVAL_ONLY:-false}"; fi
export EVAL_ONLY
EVAL_LIMIT="${EVAL_LIMIT:-200}"
export EVAL_LIMIT
echo "[eval] EVAL_ONLY=$EVAL_ONLY limit=$EVAL_LIMIT (gsm8k via utils/evals/gsm8k.yaml)"
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
amd-smi || true
resolve_trace_source
install_agentic_deps

# ---------------------------------------------------------------------------
# T147: runtime patch, so we can run a STOCK nightly without building an image.
#
# The DSpark draft can only run on TRITON_MLA (it is the only ROCm MLA backend
# declaring supports_non_causal_multi_token_decode), and cudagraph capability is
# the MINIMUM across attention groups. Upstream TritonMLAMetadataBuilder is
# UNIFORM_SINGLE_TOKEN_DECODE -- still is, as of 6f7df92a8e -- which demotes the
# whole engine FULL_AND_PIECEWISE -> PIECEWISE and leaves the drafter eager.
# Measured cost of losing it: 14.05 -> 77.65 tok/s, ITL 71.16 -> 12.88 ms.
#
# aigmkt/kimi-k3-vllm:latest ships this inside the vendored #51705 diff. A stock
# nightly does not, so apply it here. It is the ONLY thing a C1 run needs from
# that 3,830-line diff: the rest of #51705 is DCP, and T144 vs T145 measured
# DCP=8 at C1 as +36.5% TPOT, so C1 does not want DCP at all.
#
# Fails the run rather than serving a silently 5.5x-slower config.
patch_triton_mla_cudagraph_runtime() {
    local f
    f=$(python3 -c "import vllm.v1.attention.backends.mla.triton_mla as m; print(m.__file__)" 2>/dev/null) || {
        echo "[cg-patch] cannot locate triton_mla.py" >&2; return 1; }
    if grep -q "AttentionCGSupport.UNIFORM_BATCH" "$f"; then
        echo "[cg-patch] already UNIFORM_BATCH ($f)"; return 0
    fi
    python3 - "$f" <<'PY' || return 1
import re, sys
p = sys.argv[1]
s = open(p).read()
new, n = re.subn(
    r"_cudagraph_support:\s*ClassVar\[AttentionCGSupport\]\s*=\s*\(\s*AttentionCGSupport\.UNIFORM_SINGLE_TOKEN_DECODE\s*,?\s*\)",
    "_cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH",
    s, count=1)
if n != 1:
    sys.exit("[cg-patch] pattern not found -- refusing to run")
open(p, "w").write(new)
print(f"[cg-patch] UNIFORM_SINGLE_TOKEN_DECODE -> UNIFORM_BATCH in {p}")
PY
    grep -q "AttentionCGSupport.UNIFORM_BATCH" "$f" || { echo "[cg-patch] verify failed" >&2; return 1; }
}
# T152: apply #51705 (DCP for Kimi-K3 DSpark) at runtime, rebased onto this
# nightly. The vendored diff was cut against f94666b60d and does NOT apply to
# 6f7df92a8e -- 4 files conflict. Rebased with git apply --3way; the three
# conflicting files we do not use (models/kimi_k3/nvidia/mla.py, which collides
# with #54015's merged QKV-gate refactor, and both mooncake store files, whose
# transfer_groups API was refactored upstream) were taken at their NIGHTLY
# version, so only the ROCm-side DCP changes are carried. speculator.py's one
# conflict was two different added imports; both are kept and both are used.
# Result: 23 files, every changed .py parses, patch -p1 applies with 0 rejects.
# This supersedes the one-line cg patch below -- the diff already carries the
# TritonMLA UNIFORM_BATCH bump -- but that stays as a guard and no-ops.
apply_pr51705_nightly() {
    local d="$(dirname "$0")/pr51705_nightly.diff"
    local root
    root=$(python3 -c "import vllm,os;print(os.path.dirname(os.path.dirname(vllm.__file__)))" 2>/dev/null) || return 1
    [ -f "$d" ] || { echo "[pr51705] diff not found at $d" >&2; return 1; }
    if grep -q "prepare_dcp_local_seq_lens" "$root/vllm/v1/worker/gpu/spec_decode/speculator.py" 2>/dev/null; then
        echo "[pr51705] already applied"; return 0
    fi
    patch -p1 -d "$root" --forward --batch < "$d" || { echo "[pr51705] FAILED" >&2; return 1; }
    echo "[pr51705] applied (rebased onto 6f7df92a8e)"
}
if [ "${RUNTIME_PR51705:-1}" = "1" ]; then
    apply_pr51705_nightly || { echo "FATAL: pr51705 runtime patch failed" >&2; exit 1; }
fi
if [ "${RUNTIME_CG_PATCH:-1}" = "1" ]; then
    patch_triton_mla_cudagraph_runtime || { echo "FATAL: TritonMLA cudagraph patch failed" >&2; exit 1; }
fi
# ---------------------------------------------------------------------------

if [ -n "${DCP_SIZE:-}" ]; then
    DCP_SOURCE=matrix
else
    # T141: DCP=8 at CONC <= 4 too. The old "DCP off below 5" rule was never
    # earned -- T106 -> T121 flipped DCP off AND the DRAM offload off in one
    # step, and the summary attributes the whole -3.5% to the offload. The one
    # DCP-on/offload-off C1 point on record is SA's, at TPOT 0.02156, which is
    # BETTER than our DCP-off 0.02165. And every C1+MTP run we have (T122,
    # T123, T138-T140) was DCP off, so DCP=8 + MTP k=8 at CONC 1 has never
    # been measured here -- while SA runs exactly that.
    # T145: back to DCP=1 at CONC <= 4. T143 (DCP=8) measured TPOT 11.93 ms at
    # the 122k context; T145 is its DCP-off control at the same context, same
    # draft, same ladder.
    if [ "$CONC" -le 4 ]; then DCP_SIZE=1; else DCP_SIZE=8; fi
    DCP_SOURCE=conc-fallback
fi
export DCP_SIZE
echo "[dcp] size=$DCP_SIZE source=$DCP_SOURCE conc=$CONC"

export VLLM_ROCM_AITER_MLA_ASM_PADDING=asm
export VLLM_ROCM_USE_AITER=1
export SAFETENSORS_FAST_GPU=1
export VLLM_ROCM_USE_AITER_MOE_SITUV2_A8W4=1
export AITER_BF16_FP8_MOE_BOUND=0
export VLLM_USE_BREAKABLE_CUDAGRAPH=0
export GPU_ARCHS=gfx950
export VLLM_ROCM_USE_AITER_MOE=1
export VLLM_ROCM_QUICK_REDUCE_QUANTIZATION="${VLLM_ROCM_QUICK_REDUCE_QUANTIZATION:-NONE}"
export AITER_SITUV2_A8W4=1
export HSA_NO_SCRATCH_RECLAIM=1
export VLLM_K3_KDA_SAFE_STAGES=1
export VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1

export VLLM_ENGINE_READY_TIMEOUT_S=7200
export AIPERF_HTTP_TCP_USER_TIMEOUT=900000
export PYTHONNOUSERSITE=1
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1200

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

export PYTHONHASHSEED=42

OFFLOAD_ARGS=()
if agentic_kv_offload_enabled; then
    case "${KV_OFFLOAD_BACKEND:-}" in
      vllm-simple)
        require_agentic_kv_offload_backend "$KV_OFFLOAD_BACKEND"
        CPU_BYTES_PER_RANK=$(( TOTAL_CPU_DRAM_GB * 1000 * 1000 * 1000 / TOTAL_RANKS ))
        export PYTHONHASHSEED=42
        SIMPLE_LAZY_OFFLOAD="${SIMPLE_LAZY_OFFLOAD:-false}"
        OFFLOAD_ARGS=(
            --kv-transfer-config
            "{\"kv_connector\":\"SimpleCPUOffloadConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"cpu_bytes_to_use_per_rank\":$CPU_BYTES_PER_RANK,\"lazy_offload\":$SIMPLE_LAZY_OFFLOAD}}"
        )
        echo "SimpleCPUOffloadConnector: ${CPU_BYTES_PER_RANK} B/rank x ${TOTAL_RANKS} ranks, lazy_offload=$SIMPLE_LAZY_OFFLOAD"
        ;;
      *)
        echo "KV offload requested (KV_OFFLOADING=$KV_OFFLOADING) but backend '${KV_OFFLOAD_BACKEND:-unset}' is not handled here" >&2
        ;;
    esac
fi

KV_CACHE_DTYPE=fp8
EP_ARGS=()
if [ "${EP_SIZE:-1}" -gt 1 ]; then
    EP_ARGS=(--enable-expert-parallel)
    echo "EP: expert parallelism ON (EP_SIZE=$EP_SIZE)"
fi

CP_ARGS=(--attention-backend ROCM_AITER_MLA)
if [ "$DCP_SIZE" -gt 1 ]; then
    CP_ARGS+=(
        --decode-context-parallel-size "$DCP_SIZE"
        --dcp-comm-backend a2a
        --cp-kv-cache-interleave-size 1
    )
    export VLLM_USE_DIRECT_DCP_A2A=0
    export VLLM_USE_DIRECT_DCP_Q_GATHER=0
    export VLLM_USE_DIRECT_DCP_KV_GATHER=0
    export VLLM_ALLOW_DCP_FULL_CUDAGRAPH=1
    export VLLM_DCP_Q_REPLICATE=1
    echo "[dcp] ENABLED size=$DCP_SIZE backend=a2a interleave=1"
elif [ "${DCP_COMM_ARGS_AT_1:-0}" = "1" ]; then
    # T146: DCP=1 but keep the two comm flags. With decode_context_parallel_size
    # 1 there is no CP group for them to act on, so the expectation is that they
    # are inert and T146 reproduces T145 exactly. That is the point -- it is a
    # cheap check that the DCP effect measured in T143 comes from the CP group
    # itself and not from a side effect of these flags. Any difference between
    # T145 and T146 means one of them is doing something undocumented.
    # Note the engine default is dcp_comm_backend=ag_rs, so a2a IS a change to
    # the recorded config even at size 1.
    CP_ARGS+=(--dcp-comm-backend a2a --cp-kv-cache-interleave-size 1)
    echo "[dcp] size=1, comm args RETAINED (a2a, interleave=1), no DCP env"
else
    echo "[dcp] DISABLED -- no DCP args, no DCP env"
fi
export VLLM_ROCM_USE_AITER_MLA=1
export AITER_DISABLE_FMHA_OPUS=1
export VLLM_ROCM_USE_AITER_MLA=1
export AITER_DISABLE_FMHA_OPUS=1

SPEC_ENABLE="${SPEC_DECODING:-}"
case "${RESULT_FILENAME:-}" in *_spec-mtp_*) SPEC_ENABLE=mtp;; esac
# Speculative depth per concurrency. k=0 disables MTP and is the single gate --
# it replaces the old CONC<=4 test, which could not express "on, but shallower".
# Override by exporting SPEC_NUM_TOKENS.
case "$CONC" in
    # T155: k=0 at C1 TEMPORARILY, to separate two causes of the 0.14 GSM8K.
    # C52 (k=0, DCP=8) scored 0.99 on this same stack; C1 (k=8, DCP=1) scored
    # 0.14. The only candidates are (A) rejection_sample_method "synthetic",
    # which imposes acceptance without real verification and would corrupt
    # output by construction, or (B) my #51705 rebase breaking the draft path.
    # k=0 with everything else identical separates them: ~0.99 means A, ~0.14
    # means B. Restore to 8 once answered.
    1|2|4)   SPEC_NUM_TOKENS="${SPEC_NUM_TOKENS:-0}" ;;   # T139: 6 -> 8, AL 3.75 -> 4.00
    8|12|16) SPEC_NUM_TOKENS="${SPEC_NUM_TOKENS:-3}" ;;
    *)       SPEC_NUM_TOKENS="${SPEC_NUM_TOKENS:-0}" ;;
esac
if [ "$SPEC_NUM_TOKENS" -eq 0 ]; then SPEC_ENABLE=""; fi
SPEC_ARGS=()
if [ "$SPEC_ENABLE" = "mtp" ]; then
    # AL is a function of k alone. Transcribed from golden_al_distribution/
    # kimik3_dspark_probabilistic_sample_method_block_rejection_sample_method.yaml
    # (kimi-k3 / thinking_on). Never set k and AL independently -- they desync.
    case "$SPEC_NUM_TOKENS" in
        1) SYNTHETIC_ACCEPT_LEN=1.85 ;;
        2) SYNTHETIC_ACCEPT_LEN=2.51 ;;
        3) SYNTHETIC_ACCEPT_LEN=3.00 ;;
        4) SYNTHETIC_ACCEPT_LEN=3.36 ;;
        5) SYNTHETIC_ACCEPT_LEN=3.62 ;;
        6) SYNTHETIC_ACCEPT_LEN=3.75 ;;
        7) SYNTHETIC_ACCEPT_LEN=3.84 ;;
        8) SYNTHETIC_ACCEPT_LEN=4.00 ;;
        *) echo "[spec] no golden AL wired for num_speculative_tokens=$SPEC_NUM_TOKENS; take it from golden_al_distribution/kimik3_dspark_probabilistic_sample_method_block_rejection_sample_method.yaml and add the case" >&2; exit 1 ;;
    esac
    # T144: draft KV fp8, was "auto" (bf16). The target model already runs
    # --kv-cache-dtype fp8; the draft was left on the default, so every draft
    # forward re-read its KV at 2 bytes/element while the target read 1. At the
    # 122k context this screen runs that is the draft's single largest memory
    # cost, and the draft runs k times per accepted-token group.
    #
    # attention_backend stays TRITON_MLA and is NOT a free choice: it is the
    # only ROCm MLA backend declaring supports_non_causal_multi_token_decode
    # (flashinfer_mla and tokenspeed_mla are the other two, both NVIDIA), which
    # DSpark's non-causal draft requires. ROCM_AITER_MLA inherits the base
    # default False and mla_attention.py raises on it.
    #
    # WARNING when reading any result from this arm: rejection_sample_method is
    # "synthetic", so the accept length is IMPOSED by synthetic_acceptance_length
    # and is not measured. A numerics regression in the draft therefore cannot
    # show up as a lower AL. Judge this change on TPOT only, and validate draft
    # quality separately with real rejection sampling + GSM8K before shipping.
    DRAFT_KV_DTYPE="${DRAFT_KV_DTYPE:-fp8}"
    SPEC_ARGS=(
        --speculative-config
        "{\"model\":\"Inferact/Kimi-K3-DSpark\",\"num_speculative_tokens\":$SPEC_NUM_TOKENS,\"method\":\"dspark\",\"attention_backend\":\"TRITON_MLA\",\"kv_cache_dtype\":\"$DRAFT_KV_DTYPE\",\"draft_sample_method\":\"probabilistic\",\"rejection_sample_method\": \"synthetic\", \"synthetic_acceptance_length\": $SYNTHETIC_ACCEPT_LEN}"
    )
    echo "MTP: speculative decoding ON (k=$SPEC_NUM_TOKENS, synthetic accept=$SYNTHETIC_ACCEPT_LEN, draft kv=$DRAFT_KV_DTYPE)"
fi

CHUNKED_PREFILL_ARGS=(--max-num-batched-tokens "${MAX_BATCHED_TOKENS:-8192}")
if [ "${ASYNC_SCHED:-0}" = "1" ]; then
    ASYNC_SCHED_ARGS=(--async-scheduling)
else
    ASYNC_SCHED_ARGS=(--no-async-scheduling)
fi
MLA_PREFILL_ARGS=(--attention-config "{\"mla_prefill_backend\":\"ROCM_AITER_FA\"}")

# T148: load-format is a live lever again. It was a UNIFIED_LOAD_FORMAT
# variable until 3412f0fa ("reduce the launcher to the code that actually
# runs") collapsed it to a hardcoded fastsafetensors -- the same class of
# mistake as deleting the KV-offload block as dead code. T123 is the ONLY run
# on record with auto, and it is our best C1 ever (TPOT 6.70 mean / 7.71 p50,
# 1,288.20 tok/s/GPU, KV pool 2,826,382 vs T133's 2,687,226 on the same node).
# Once T146 showed the DCP comm flags inert at size 1, load-format is the only
# thing left that separated T123 from T133 -- but that is inference from a
# single pair, not an A/B. This makes it one, against T147 on the same nightly.
LOAD_FORMAT="${LOAD_FORMAT:-auto}"
echo "[load] load_format=$LOAD_FORMAT conc=$CONC"

if [ -z "${MAX_NUM_SEQS:-}" ]; then
    MAX_NUM_SEQS=$(( CONC + CONC / 4 ))
    # The floor of 8 exists for the concurrent arms, where mns must leave room
    # above CONC for requests in flight. At CONC <= 4 it does the opposite: at
    # CONC 1 / k 8 the decode batch is exactly 9 rows, yet the floor captures
    # 8 x 9 = 72 -- 63 graphs that can never be selected. T140 drops the floor
    # here to measure what that costs. Ladder stays DENSE, only shorter, so the
    # out-of-bounds padding fault from the sparse-ladder trials cannot recur.
    # T141 restores the floor of 8 at CONC <= 4. T140's right-sizing is a real
    # win on capture cost (65s -> 26s, 1.46 -> 0.83 GiB/GPU) and that stands,
    # but its -0.6% latency read came from 10 requests on a harness whose
    # prompt lengths turned out to scatter 1121-7979, so it is not evidence.
    # T141 is a DCP A/B against T133, and T133 ran mns 8 / ladder 1..72 --
    # match it exactly so DCP is the ONLY variable.
    if [ "$DCP_SIZE" -gt 1 ]; then MAX_NUM_SEQS=80; fi
    if [ "$MAX_NUM_SEQS" -lt 8 ]; then MAX_NUM_SEQS=8; fi
    if [ "$MAX_NUM_SEQS" -gt 80 ]; then MAX_NUM_SEQS=80; fi
fi

SPEC_ROWS=1
if [ "${#SPEC_ARGS[@]}" -gt 0 ]; then SPEC_ROWS=$(( SPEC_NUM_TOKENS + 1 )); fi
# T149: cap the ladder. At C1 the decode batch is always CONC*(k+1) = 9 rows,
# so mns 8 x 9 rows = 72 captures 63 graphs that can never be selected. Dense,
# just shorter -- batches above the cap take PIECEWISE rather than being padded
# out of bounds, so the sparse-ladder fault cannot recur. Sole variable vs T148.
if [ "$CONC" -le 4 ]; then
    LADDER_MAX=16
elif [ "$CONC" -le 16 ]; then
    LADDER_MAX=32
else
    LADDER_MAX=80
fi
MAX_CUDAGRAPH_CAPTURE_SIZE=$(( MAX_NUM_SEQS * SPEC_ROWS ))
if [ "$MAX_CUDAGRAPH_CAPTURE_SIZE" -gt "$LADDER_MAX" ]; then MAX_CUDAGRAPH_CAPTURE_SIZE=$LADDER_MAX; fi
CUDAGRAPH_CAPTURE_SIZES=$(seq -s, 1 "$MAX_CUDAGRAPH_CAPTURE_SIZE")
echo "graphs: dense ladder 1..$MAX_CUDAGRAPH_CAPTURE_SIZE (mns=$MAX_NUM_SEQS x $SPEC_ROWS rows), DCP=$DCP_SIZE"
CUDAGRAPH_MODE=FULL_AND_PIECEWISE
COMPILATION_CONFIG_ARGS=(--compilation-config "{\"mode\":3,\"cudagraph_mode\":\"$CUDAGRAPH_MODE\",\"max_cudagraph_capture_size\":$MAX_CUDAGRAPH_CAPTURE_SIZE,\"custom_ops\":[\"+fused_rms_norm_gated\"],\"cudagraph_capture_sizes\":[$CUDAGRAPH_CAPTURE_SIZES]}")

GPU_MEM_UTIL=0.9

VLLM_CMD=(
    vllm serve "$MODEL_PATH" --served-model-name "$MODEL"
    --host 0.0.0.0
    --port "$PORT"
    --trust-remote-code
    --moe-backend auto
    --tensor-parallel-size "$TP"
    --load-format "$LOAD_FORMAT"
    --gpu-memory-utilization "$GPU_MEM_UTIL"
    --language-model-only
    --max-num-seqs "$MAX_NUM_SEQS"
    --enable-auto-tool-choice
    --tool-call-parser kimi_k3
    --reasoning-parser kimi_k3
    --max-model-len 1048576
    --enable-prefix-caching
    --enable-prompt-tokens-details
    --kv-cache-dtype "$KV_CACHE_DTYPE"
    "${CHUNKED_PREFILL_ARGS[@]}"
    "${OFFLOAD_ARGS[@]}"
    "${CP_ARGS[@]}"
    "${EP_ARGS[@]}"
    "${SPEC_ARGS[@]}"
    "${ASYNC_SCHED_ARGS[@]}"
    "${MLA_PREFILL_ARGS[@]}"
    "${COMPILATION_CONFIG_ARGS[@]}"
)

for _a in CP_ARGS SPEC_ARGS CHUNKED_PREFILL_ARGS ASYNC_SCHED_ARGS MLA_PREFILL_ARGS OFFLOAD_ARGS COMPILATION_CONFIG_ARGS; do
    grep -q "\${$_a\[@\]}" "$0" || echo "[orphan-check] WARNING: $_a is built but never passed to VLLM_CMD" >&2
done

printf '%q ' "${VLLM_CMD[@]}" | tee "$RESULT_DIR/vllm_command.txt"
printf '\n' | tee -a "$RESULT_DIR/vllm_command.txt"

"${VLLM_CMD[@]}" > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo "Server PID: $SERVER_PID"

wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

# Fixed-length serving, not the agentic replay. The agentic path is already
# well sampled (T133 C1: 184 requests over its DURATION cap); the gap was here.
#
# NUM_PROMPTS, not CONC*10. T138-T140 ran CONC*10 = TEN requests at C1, and
# TPOT percentiles are PER-REQUEST, so p90 over 10 samples is essentially the
# maximum -- not a number to draw a conclusion from. (ITL percentiles were fine
# even then: those are per-interval, ~10k samples.) 1000 requests at C1 costs
# ~2.7 h at the 9.56 s mean e2e latency T140 measured, well inside the job's
# 500-minute timeout.
#
# Still to be aware of: this harness is not as fixed-length as its name says.
# With --random-range-ratio 0 the T140 prompts still came out 1121-7979 input
# tokens and 277-1828 output tokens, because the random dataset round-trips
# token ids through decode/encode and Kimi's BPE re-merges faster than the 10
# convergence retries can correct. More requests fixes the SAMPLING problem,
# not the length-scatter one -- but with 1000 samples the scatter averages out
# instead of dominating.
#
# Do NOT write ${ISL:-8000}. The runner exports ISL, OSL and RANDOM_RANGE_RATIO
# into the container (see the -e list on its docker run) and, for the agentic
# scenario, sets them to 0 / 0 / 0.8 because the replay supplies the prompts.
# They are SET, just to zero, so :- never fires. T137 (33157520672) reached the
# client with --random-input-len 0 and died in sample_random_requests with
#   ValueError: low >= high
# Treat 0 as "not supplied", and override with ITL_ISL / ITL_OSL, which the
# runner does not touch.
ISL="${ITL_ISL:-${ISL:-0}}"
OSL="${ITL_OSL:-${OSL:-0}}"
# 122k, matching the agentic replay's 122,657-token mean input (T133 C1), not
# 8k. Context length is the one thing that decides whether DCP is worth its
# per-layer a2a + KV gather + LSE merge: at 8k the KV read is ~20 us against a
# ~29 ms step, so DCP is pure overhead and the screen would say "DCP loses"
# regardless of what it does on the real workload. At 122k the attention work
# DCP parallelises is 28x larger. Screen at the length you intend to serve.
[ "$ISL" -gt 0 ] 2>/dev/null || ISL=122000
# OSL 500 for the C1 screening sweep, not 2000. At 1000 requests OSL 2000 costs
# ~2.7 h per trial, which is too slow to rank configs; OSL 500 costs ~55 min and
# still averages TPOT over ~125 decode steps per request at AL 4.00. All arms in
# the sweep share it, so the ranking is internally consistent. The winner gets
# re-run at OSL 2000 before any headline number is quoted.
[ "$OSL" -gt 0 ] 2>/dev/null || OSL=500
# Fixed length: hard 0, not the runner's 0.8 jitter. ITL variance is the metric
# here, so the prompts must not vary in length.
RANDOM_RANGE_RATIO=0
# Screening budget: ~15 min of benchmark, not a fixed 1000. At ISL 122k a C1
# request costs roughly 6-7 s (long prefill + ~250 decode tokens), so 100
# requests lands near the budget. 100 per-request TPOT samples is enough to
# RANK configs; it is not enough to publish one, so the winner gets a long run
# before any headline number is quoted.
# The old CONC*10 rule gave TEN requests at C1, which is what made T138-T140's
# TPOT percentiles unusable -- those percentiles are per-request.
NUM_PROMPTS="${NUM_PROMPTS:-0}"
[ "$NUM_PROMPTS" -gt 0 ] 2>/dev/null || NUM_PROMPTS=$(( CONC * 10 ))
if [ "$NUM_PROMPTS" -lt 100 ]; then NUM_PROMPTS=100; fi
if [ $(( ISL + OSL )) -gt 1048576 ]; then
    echo "Error: ISL+OSL = $(( ISL + OSL )) exceeds max-model-len 1048576." >&2
    exit 1
fi
echo "[client] fixed-length: ISL=$ISL OSL=$OSL range-ratio=$RANDOM_RANGE_RATIO conc=$CONC prompts=$NUM_PROMPTS"

if [ "${EVAL_ONLY:-false}" = "true" ]; then
    run_eval --port "$PORT"
else
    # build_replay_cmd "$RESULT_DIR"
    # run_agentic_replay_and_write_outputs "$RESULT_DIR"
    build_replay_cmd "$RESULT_DIR"
    run_agentic_replay_and_write_outputs "$RESULT_DIR"
fi

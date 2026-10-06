#!/usr/bin/env bash
set -eo pipefail
set -x
source "$(dirname "$0")/../../benchmark_lib.sh"
wait_for_amd_gpu_clean

export EVAL_ONLY="${EVAL_ONLY:-false}"
check_env_vars MODEL TP CONC KV_OFFLOADING TOTAL_CPU_DRAM_GB RESULT_DIR DURATION EP_SIZE
check_env_vars DCP_SIZE EVAL_ONLY

# =============================================================================
# Patch activation flags -- one place to toggle every staged, unmerged
# vllm-project/vllm PR for this recipe. 1 = apply, 0 = skip (vLLM runs
# unpatched for that PR). See each PR's own comment block below for what it
# does, requirements, and known conflicts with other staged PRs.
# =============================================================================
export APPLY_PR59591="${APPLY_PR59591:-0}"  # Kimi-K3: shard latent-MoE up-proj by TP rank -- CONFLICTS with #59693, leave 0 while that's 1
export APPLY_PR59069="${APPLY_PR59069:-1}"  # Kimi-K3: fuse AttnRes output + per-token FP8 quant
export APPLY_PR59070="${APPLY_PR59070:-1}"  # ROCm MLA: keep DCP prefill context FP8 through AllGather
export APPLY_PR59693="${APPLY_PR59693:-1}"  # Kimi-K3: token-sharded residual stream for long prefills -- requires APPLY_PR59591=0
export APPLY_PR59965="${APPLY_PR59965:-1}"  # ROCm DCP: default MLA DCP verify to round-robin asm
export APPLY_PR59966="${APPLY_PR59966:-1}"  # ROCm DCP: gather MLA decode query without byte-wise strided copies
export APPLY_PR54627="${APPLY_PR54627:-1}"  # prefill_schedule_interval outside DP -- +2.6% tput/-7.5-17% TPOT but +313-352% TTFT (real trade-off, see block below)
export APPLY_PR54625="${APPLY_PR54625:-1}"  # cache-aware admission ordering -- measured together with #54627 above
export APPLY_PR58743="${APPLY_PR58743:-0}"  # Kimi-K3: support BF16 KDA recurrent state -- OFF: crashes decode, see block below
# #58861/#58723 NOT staged: both conflict (text-level) with #59069/#59693 in
# attn_res.py/linear.py -- needs rebuild + live-verify, left for follow-up.

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
export VLLM_ROCM_USE_AITER_MOE_SITUV2=a8w4
export VLLM_ROCM_QUICK_REDUCE_QUANTIZATION="${VLLM_ROCM_QUICK_REDUCE_QUANTIZATION:-INT4}"
export AITER_SITUV2_A8W4=1
export AITER_FLYDSL_STAGE2_FP8="${AITER_FLYDSL_STAGE2_FP8:-1}"
export AITER_BF16_FP8_MOE_BOUND=0
export AITER_DISABLE_FMHA_OPUS=1
export SAFETENSORS_FAST_GPU=1
export GPU_ARCHS=gfx950
export HSA_NO_SCRATCH_RECLAIM=1
export VLLM_USE_BREAKABLE_CUDAGRAPH=0
export VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1
export VLLM_ENGINE_READY_TIMEOUT_S=7200
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3600
export AIPERF_HTTP_TCP_USER_TIMEOUT=900000
export PYTHONNOUSERSITE=1
export PYTHONHASHSEED=42

export VLLM_USE_DIRECT_DCP_A2A=0
# 1 (not the reference recipe's 0): this is the gate #59966 (staged below)
# optimizes -- the patch alone is inert at 0.
export VLLM_USE_DIRECT_DCP_Q_GATHER=1
export VLLM_USE_DIRECT_DCP_KV_GATHER=0

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

SPEC_ARGS=()
SPEC_ROWS=1
KDA_ARGS=()
case "$CONC" in
    1|2|4|8|10|12|14|16)
        DCP_SIZE=1
        OFFLOAD_POLICY=harness
        case "$CONC" in
            1)  SPEC_NUM_TOKENS="${SPEC_NUM_TOKENS:-${SPEC_K:-6}}" ;;
            4)  SPEC_NUM_TOKENS="${SPEC_NUM_TOKENS:-${SPEC_K:-5}}" ;;
            10)  SPEC_NUM_TOKENS="${SPEC_NUM_TOKENS:-${SPEC_K:-5}}" ;;
            *)  SPEC_NUM_TOKENS="${SPEC_NUM_TOKENS:-${SPEC_K:-3}}" ;;
        esac
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
        SPEC_BASE="\"model\":\"Inferact/Kimi-K3-DSpark\",\"num_speculative_tokens\":$SPEC_NUM_TOKENS,\"method\":\"dspark\",\"attention_backend\":\"ROCM_AITER_MLA\",\"kv_cache_dtype\":\"$DRAFT_KV_DTYPE\",\"draft_sample_method\":\"probabilistic\""
        if [ "${EVAL_ONLY:-false}" = "true" ]; then
            SPEC_ARGS=(--speculative-config "{$SPEC_BASE,\"rejection_sample_method\": \"block\"}")
            echo "MTP: k=$SPEC_NUM_TOKENS LIVE block rejection (accuracy gate) draft_kv=$DRAFT_KV_DTYPE"
        else
            SPEC_ARGS=(--speculative-config "{$SPEC_BASE,\"rejection_sample_method\": \"synthetic\", \"synthetic_acceptance_length\": $SYNTHETIC_ACCEPT_LEN}")
            echo "MTP: k=$SPEC_NUM_TOKENS synthetic_accept=$SYNTHETIC_ACCEPT_LEN draft_kv=$DRAFT_KV_DTYPE"
        fi
        SPEC_ROWS=$(( SPEC_NUM_TOKENS + 1 ))
        case "$CONC" in
            1)  SPEC_SEATS=2  ;;
            2)  SPEC_SEATS=4  ;;
            4)  SPEC_SEATS=8  ;;
            8)  SPEC_SEATS=10 ;;
            10) SPEC_SEATS=12 ;;
            12) SPEC_SEATS=24 ;;
            14) SPEC_SEATS=16 ;;
            16) SPEC_SEATS=18 ;;
            *)  SPEC_SEATS=$(( CONC + 2 )) ;;
        esac
        MAX_NUM_SEQS="${MAX_NUM_SEQS:-$SPEC_SEATS}"
        if [ "$CONC" -eq 1 ]; then MAX_BATCHED_TOKENS="${MAX_BATCHED_TOKENS:-16384}"
        else MAX_BATCHED_TOKENS="${MAX_BATCHED_TOKENS:-8192}"; fi
        ;;
    *)
        DCP_SIZE="${DCP_SIZE:-8}"
        OFFLOAD_POLICY=harness
        if [ "$CONC" -gt 64 ]; then MAX_BATCHED_TOKENS="${MAX_BATCHED_TOKENS:-24576}"
        else MAX_BATCHED_TOKENS="${MAX_BATCHED_TOKENS:-8192}"; fi
        if [ "$CONC" -lt 72 ]; then MAX_NUM_SEQS="${MAX_NUM_SEQS:-$(( CONC * 14 / 10 ))}"
        elif [ "$CONC" -eq 72 ]; then MAX_NUM_SEQS="${MAX_NUM_SEQS:-96}"
        else MAX_NUM_SEQS="${MAX_NUM_SEQS:-112}"; fi
        ;;
esac
export DCP_SIZE

# -----------------------------------------------------------------------------
# #59591 -- shard Kimi-K3 latent-MoE up-proj by rank (~3.9GB/rank freed).
# OFF by default: conflicts with #59693's SP path (confirmed live crash).
# -----------------------------------------------------------------------------
apply_pr59591() {
    [ "${APPLY_PR59591:-0}" = "1" ] || { echo "[pr59591] disabled"; return 0; }
    local diff_file
    diff_file="$(cd "$(dirname "$0")" && pwd)/patches/pr59591-tp-shard-moe-upproj.diff"
    [ -f "$diff_file" ] || { echo "[pr59591] missing $diff_file" >&2; return 1; }
    local site
    site="$(python3 -c 'import vllm,os;print(os.path.dirname(os.path.dirname(vllm.__file__)))')"
    if python3 -c 'import inspect,vllm.models.kimi_k3.amd.linear as m
import sys; sys.exit(0 if "row_sharded" in inspect.getsource(m) else 1)' 2>/dev/null; then
        echo "[pr59591] already present, nothing to do"; return 0
    fi
    ( cd "$site" && patch -p1 --forward --silent < "$diff_file" ) || return 1
    python3 -c 'import py_compile
for f in ["vllm/models/kimi_k3/amd/latent_moe_runner.py","vllm/models/kimi_k3/amd/linear.py"]:
    py_compile.compile("'"$site"'/"+f,doraise=True)' || return 1
    echo "[pr59591] applied"
}
apply_pr59591 || { echo "[pr59591] patch failed, refusing to run" >&2; exit 1; }

# -----------------------------------------------------------------------------
# #59069 -- fuse AttnRes output + per-token FP8 quant. 3-5.2% faster, graceful
# fallback if quant scheme doesn't match. No flag.
# -----------------------------------------------------------------------------
apply_pr59069() {
    [ "${APPLY_PR59069:-1}" = "1" ] || { echo "[pr59069] disabled"; return 0; }
    local diff_file
    diff_file="$(cd "$(dirname "$0")" && pwd)/patches/pr59069-attnres-fp8-fusion.diff"
    [ -f "$diff_file" ] || { echo "[pr59069] missing $diff_file" >&2; return 1; }
    local site
    site="$(python3 -c 'import vllm,os;print(os.path.dirname(os.path.dirname(vllm.__file__)))')"
    if python3 -c 'import inspect,vllm.models.kimi_k3.amd.linear as m
import sys; sys.exit(0 if "get_input_quant_key" in inspect.getsource(m) else 1)' 2>/dev/null; then
        echo "[pr59069] already present, nothing to do"; return 0
    fi
    ( cd "$site" && patch -p1 --forward --silent < "$diff_file" ) || return 1
    python3 -c 'import py_compile
for f in ["vllm/models/kimi_k3/amd/kda.py","vllm/models/kimi_k3/amd/linear.py","vllm/models/kimi_k3/amd/mla.py","vllm/models/kimi_k3/amd/ops/attn_res.py"]:
    py_compile.compile("'"$site"'/"+f,doraise=True)' || return 1
    echo "[pr59069] applied"
}
apply_pr59069 || { echo "[pr59069] patch failed, refusing to run" >&2; exit 1; }

# -----------------------------------------------------------------------------
# #59070 -- keep DCP prefill context FP8 through AllGather (was BF16 upcast).
# 1.8-2.1x faster AllGather. No flag.
# -----------------------------------------------------------------------------
apply_pr59070() {
    [ "${APPLY_PR59070:-1}" = "1" ] || { echo "[pr59070] disabled"; return 0; }
    local diff_file
    diff_file="$(cd "$(dirname "$0")" && pwd)/patches/pr59070-dcp-fp8-allgather.diff"
    [ -f "$diff_file" ] || { echo "[pr59070] missing $diff_file" >&2; return 1; }
    local site
    site="$(python3 -c 'import vllm,os;print(os.path.dirname(os.path.dirname(vllm.__file__)))')"
    if python3 -c 'import inspect,vllm._aiter_ops as m
import sys; sys.exit(0 if "gather_kv_b_proj" in inspect.getsource(m) else 1)' 2>/dev/null; then
        echo "[pr59070] already present, nothing to do"; return 0
    fi
    ( cd "$site" && patch -p1 --forward --silent < "$diff_file" ) || return 1
    python3 -c 'import py_compile
for f in ["vllm/_aiter_ops.py","vllm/v1/attention/backends/mla/rocm_aiter_mla.py","vllm/v1/attention/ops/rocm_aiter_mla_prefill.py"]:
    py_compile.compile("'"$site"'/"+f,doraise=True)' || return 1
    echo "[pr59070] applied"
}
apply_pr59070 || { echo "[pr59070] patch failed, refusing to run" >&2; exit 1; }

# -----------------------------------------------------------------------------
# #59693 -- token-sharded residual stream for long prefills. Geomean +6.1%,
# gate VLLM_KIMI_K3_AMD_PREFILL_SP_MIN_TOKENS=1024 (set below). Requires
# APPLY_PR59591=0 (see that block).
# -----------------------------------------------------------------------------
apply_pr59693() {
    [ "${APPLY_PR59693:-1}" = "1" ] || { echo "[pr59693] disabled"; return 0; }
    local diff_file
    diff_file="$(cd "$(dirname "$0")" && pwd)/patches/pr59693-prefill-sp.diff"
    [ -f "$diff_file" ] || { echo "[pr59693] missing $diff_file" >&2; return 1; }
    local site
    site="$(python3 -c 'import vllm,os;print(os.path.dirname(os.path.dirname(vllm.__file__)))')"
    if python3 -c 'import vllm.envs as e; import sys
sys.exit(0 if hasattr(e, "VLLM_KIMI_K3_AMD_PREFILL_SP_MIN_TOKENS") else 1)' 2>/dev/null; then
        echo "[pr59693] already present, nothing to do"; return 0
    fi
    ( cd "$site" && patch -p1 --forward --silent < "$diff_file" ) || return 1
    python3 -c 'import py_compile
for f in ["vllm/envs.py","vllm/models/kimi_k3/amd/latent_moe_runner.py","vllm/models/kimi_k3/amd/linear.py","vllm/models/kimi_k3/amd/sp.py"]:
    py_compile.compile("'"$site"'/"+f,doraise=True)' || return 1
    echo "[pr59693] applied"
}
apply_pr59693 || { echo "[pr59693] patch failed, refusing to run" >&2; exit 1; }
export VLLM_KIMI_K3_AMD_PREFILL_SP_MIN_TOKENS="${VLLM_KIMI_K3_AMD_PREFILL_SP_MIN_TOKENS:-1024}"

# -----------------------------------------------------------------------------
# #59965 -- DCP MLA verify default segmented->auto. +50.8% tput but only for
# spec-decode batches we don't run at C70/C72 -- safety-net default, not an
# expected win here.
# -----------------------------------------------------------------------------
apply_pr59965() {
    [ "${APPLY_PR59965:-1}" = "1" ] || { echo "[pr59965] disabled"; return 0; }
    local diff_file
    diff_file="$(cd "$(dirname "$0")" && pwd)/patches/pr59965-dcp-verify-auto-default.diff"
    [ -f "$diff_file" ] || { echo "[pr59965] missing $diff_file" >&2; return 1; }
    local site
    site="$(python3 -c 'import vllm,os;print(os.path.dirname(os.path.dirname(vllm.__file__)))')"
    if python3 -c 'import vllm.envs as e; import sys
sys.exit(0 if e.VLLM_ROCM_AITER_MLA_DCP_VERIFY == "auto" else 1)' 2>/dev/null; then
        echo "[pr59965] already present, nothing to do"; return 0
    fi
    ( cd "$site" && patch -p1 --forward --silent < "$diff_file" ) || return 1
    python3 -c 'import py_compile
for f in ["vllm/envs.py","vllm/v1/attention/backends/mla/rocm_aiter_mla.py"]:
    py_compile.compile("'"$site"'/"+f,doraise=True)' || return 1
    echo "[pr59965] applied"
}
apply_pr59965 || { echo "[pr59965] patch failed, refusing to run" >&2; exit 1; }

# -----------------------------------------------------------------------------
# #59966 -- direct MLA decode query gather, no byte-wise strided copies.
# -2.7% step time, bit-identical. Needs VLLM_USE_DIRECT_DCP_Q_GATHER=1
# (flipped above, was 0).
# -----------------------------------------------------------------------------
apply_pr59966() {
    [ "${APPLY_PR59966:-1}" = "1" ] || { echo "[pr59966] disabled"; return 0; }
    local diff_file
    diff_file="$(cd "$(dirname "$0")" && pwd)/patches/pr59966-dcp-query-gather.diff"
    [ -f "$diff_file" ] || { echo "[pr59966] missing $diff_file" >&2; return 1; }
    local site
    site="$(python3 -c 'import vllm,os;print(os.path.dirname(os.path.dirname(vllm.__file__)))')"
    if python3 -c 'import inspect,vllm.v1.attention.ops.dcp as m
import sys; sys.exit(0 if "copy_rows_" in inspect.getsource(m) else 1)' 2>/dev/null; then
        echo "[pr59966] already present, nothing to do"; return 0
    fi
    ( cd "$site" && patch -p1 --forward --silent < "$diff_file" ) || return 1
    python3 -c 'import py_compile
for f in ["vllm/v1/attention/backends/mla/rocm_aiter_mla.py","vllm/v1/attention/ops/dcp.py"]:
    py_compile.compile("'"$site"'/"+f,doraise=True)' || return 1
    echo "[pr59966] applied"
}
apply_pr59966 || { echo "[pr59966] patch failed, refusing to run" >&2; exit 1; }

# -----------------------------------------------------------------------------
# #58743 -- BF16 KDA recurrent state. OFF by default: only patches the
# prefill-path fused kernel's FP32 handling, not decode's -- live run crashed
# with "fused_kda_decode ... state must be a GPU float32 tensor" (run
# 37421994913). PR doesn't touch decode at all; not safe to enable as-is.
# -----------------------------------------------------------------------------
apply_pr58743() {
    [ "${APPLY_PR58743:-1}" = "1" ] || { echo "[pr58743] disabled"; return 0; }
    local diff_file
    diff_file="$(cd "$(dirname "$0")" && pwd)/patches/pr58743-kda-bf16-recurrent-state.diff"
    [ -f "$diff_file" ] || { echo "[pr58743] missing $diff_file" >&2; return 1; }
    local site
    site="$(python3 -c 'import vllm,os;print(os.path.dirname(os.path.dirname(vllm.__file__)))')"
    if python3 -c 'import inspect,vllm.models.kimi_k3.amd.kda as m
import sys; sys.exit(0 if "mamba_ssm_cache_dtype" in inspect.getsource(m) else 1)' 2>/dev/null; then
        echo "[pr58743] already present, nothing to do"; return 0
    fi
    ( cd "$site" && patch -p1 --forward --silent < "$diff_file" ) || return 1
    python3 -c 'import py_compile
for f in ["vllm/models/kimi_k3/amd/kda.py","vllm/models/kimi_k3/amd/linear.py","vllm/models/kimi_k3/amd/ops/kda_prefill.py"]:
    py_compile.compile("'"$site"'/"+f,doraise=True)' || return 1
    echo "[pr58743] applied"
}
apply_pr58743 || { echo "[pr58743] patch failed, refusing to run" >&2; exit 1; }
MAMBA_SSM_CACHE_DTYPE="${MAMBA_SSM_CACHE_DTYPE:-auto}"

# -----------------------------------------------------------------------------
# #54627 -- prefill_schedule_interval under DCP (was a DP-only no-op).
# interval=33: +2.6% tput, TPOT -7.5/-17.3%, but TTFT +313/+352%. Real
# trade-off, not noise -- only worth it if TTFT isn't the priority metric.
# -----------------------------------------------------------------------------
apply_pr54627() {
    [ "${APPLY_PR54627:-1}" = "1" ] || { echo "[pr54627] disabled"; return 0; }
    local diff_file
    diff_file="$(cd "$(dirname "$0")" && pwd)/patches/pr54627-prefill-interval-nondp.diff"
    [ -f "$diff_file" ] || { echo "[pr54627] missing $diff_file" >&2; return 1; }
    local site
    site="$(python3 -c 'import vllm,os;print(os.path.dirname(os.path.dirname(vllm.__file__)))')"
    if python3 -c 'import inspect,vllm.v1.core.sched.scheduler as s
import sys; sys.exit(0 if "last_prefill_step" in inspect.getsource(s) else 1)' 2>/dev/null; then
        echo "[pr54627] already present, nothing to do"; return 0
    fi
    ( cd "$site" && patch -p1 --forward --silent < "$diff_file" ) || return 1
    python3 -c 'import py_compile;py_compile.compile("'"$site"'/vllm/v1/core/sched/scheduler.py",doraise=True);py_compile.compile("'"$site"'/vllm/config/scheduler.py",doraise=True)' || return 1
    echo "[pr54627] applied"
}
apply_pr54627 || { echo "[pr54627] patch failed, refusing to run" >&2; exit 1; }
PREFILL_SCHEDULE_INTERVAL="${PREFILL_SCHEDULE_INTERVAL:-33}"

# -----------------------------------------------------------------------------
# #54625 -- cache-aware admission ordering. window=64/threshold=0.5. Measured
# together with #54627 above, not isolated.
# -----------------------------------------------------------------------------
apply_pr54625() {
    [ "${APPLY_PR54625:-1}" = "1" ] || { echo "[pr54625] disabled"; return 0; }
    local diff_file
    diff_file="$(cd "$(dirname "$0")" && pwd)/patches/pr54625-cache-aware-admission.diff"
    [ -f "$diff_file" ] || { echo "[pr54625] missing $diff_file" >&2; return 1; }
    local site
    site="$(python3 -c 'import vllm,os;print(os.path.dirname(os.path.dirname(vllm.__file__)))')"
    if python3 -c 'import vllm.config.scheduler as s; import sys; sys.exit(0 if hasattr(s.SchedulerConfig, "cache_aware_admission_window") else 1)' 2>/dev/null; then
        echo "[pr54625] already present, nothing to do"; return 0
    fi
    ( cd "$site" && patch -p1 --forward --silent < "$diff_file" ) || return 1
    python3 -c 'import py_compile
for f in ["vllm/config/scheduler.py","vllm/engine/arg_utils.py","vllm/v1/core/kv_cache_manager.py","vllm/v1/core/sched/scheduler.py"]:
    py_compile.compile("'"$site"'/"+f,doraise=True)' || return 1
    echo "[pr54625] applied"
}
apply_pr54625 || { echo "[pr54625] patch failed, refusing to run" >&2; exit 1; }
# The two flags below do not exist on vLLM's CLI parser unless #54625 is
# applied -- unlike --prefill-schedule-interval, which is already a real flag
# pre-patch. Keep them out of VLLM_CMD entirely when the patch is off, or
# "unrecognized arguments" kills every non-#54625 dispatch.
CACHE_AWARE_ARGS=()
if [ "${APPLY_PR54625:-1}" = "1" ]; then
    CACHE_AWARE_ARGS=(
        --cache-aware-admission-window "${CACHE_AWARE_ADMISSION_WINDOW:-64}"
        --cache-aware-admission-threshold "${CACHE_AWARE_ADMISSION_THRESHOLD:-0.5}"
    )
fi

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
OFFLOAD_LABEL="$OFFLOAD_POLICY"
if [ "$OFFLOAD_POLICY" = "none" ]; then
    :
elif agentic_kv_offload_enabled; then
    OFFLOAD_LABEL="${KV_OFFLOADING}"
    CPU_BYTES_PER_RANK=$(( TOTAL_CPU_DRAM_GB * 1000 * 1000 * 1000 / TOTAL_RANKS ))
    OFFLOAD_ARGS=(--kv-transfer-config "{\"kv_connector\":\"SimpleCPUOffloadConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"cpu_bytes_to_use_per_rank\":$CPU_BYTES_PER_RANK,\"lazy_offload\":false}}")
else
    OFFLOAD_LABEL=none
fi

EP_ARGS=()
if [ "${EP_SIZE:-1}" -gt 1 ]; then EP_ARGS=(--enable-expert-parallel); fi

echo "[cfg] conc=$CONC dcp=$DCP_SIZE gmu=$GPU_MEM_UTIL mns=$MAX_NUM_SEQS ladder=1..$LADDER spec_rows=$SPEC_ROWS chunk=$MAX_BATCHED_TOKENS cudagraph=$CUDAGRAPH_MODE offload=$OFFLOAD_LABEL"

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
    --mamba-ssm-cache-dtype "$MAMBA_SSM_CACHE_DTYPE"
    --enable-auto-tool-choice
    --tool-call-parser kimi_k3
    --reasoning-parser kimi_k3
    --enable-prefix-caching
    --enable-prompt-tokens-details
    --no-async-scheduling
    --attention-config '{"mla_prefill_backend":"ROCM_AITER_FA"}'
    --prefill-schedule-interval "$PREFILL_SCHEDULE_INTERVAL"
    "${CACHE_AWARE_ARGS[@]}"
    "${OFFLOAD_ARGS[@]}"
    "${CP_ARGS[@]}"
    "${EP_ARGS[@]}"
    "${SPEC_ARGS[@]}"
    "${KDA_ARGS[@]}"
    "${COMPILATION_CONFIG_ARGS[@]}"
)

# -----------------------------------------------------------------------------
# Opt-in rocprofv3 wrap. Off by default -- set ROCPROF_ENABLE=1 to capture a
# kernel-trace + stats window (hotspots, call counts, avg duration) mid-run.
# rocprofv3 only wraps a process at launch (no attach-to-running-PID), so the
# window is expressed as a collection-period relative to server start:
# ROCPROF_START_DELAY seconds to clear model load/warmup/cudagraph capture,
# then ROCPROF_DURATION seconds of actual collection.
# -----------------------------------------------------------------------------
# ROCPROF_ENABLE="${ROCPROF_ENABLE:-0}"
# LAUNCH_CMD=("${VLLM_CMD[@]}")
# if [ "$ROCPROF_ENABLE" = "1" ]; then
#     ROCPROF_START_DELAY="${ROCPROF_START_DELAY:-300}"
#     ROCPROF_DURATION="${ROCPROF_DURATION:-60}"
#     ROCPROF_DIR="$RESULT_DIR/rocprof"
#     mkdir -p "$ROCPROF_DIR"
#     LAUNCH_CMD=(
#         rocprofv3 --kernel-trace --stats
#         -d "$ROCPROF_DIR" -f csv
#         --summary-output-file "$RESULT_DIR/rocprof_summary.txt"
#         -P "${ROCPROF_START_DELAY}:${ROCPROF_DURATION}:1"
#         --collection-period-unit sec
#         --
#         "${VLLM_CMD[@]}"
#     )
#     echo "[rocprof] enabled: start_delay=${ROCPROF_START_DELAY}s duration=${ROCPROF_DURATION}s output=$ROCPROF_DIR"
# fi

printf '%q ' "${VLLM_CMD[@]}" | tee "$RESULT_DIR/vllm_command.txt"
printf '\n' | tee -a "$RESULT_DIR/vllm_command.txt"

"${VLLM_CMD[@]}" > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo "Server PID: $SERVER_PID"

python3 - <<'CCDPY' > "$RESULT_DIR/ccdmap.txt" 2>/dev/null || true
import subprocess, re, os, glob
def expand(s):
    v=[]
    for part in s.split(','):
        if '-' in part:
            a,b=part.split('-'); v+=list(range(int(a),int(b)+1))
        else: v.append(int(part))
    return v
def l3_domains():
    seen,out=set(),[]
    for c in sorted(int(re.search(r'cpu(\d+)$',x).group(1)) for x in glob.glob('/sys/devices/system/cpu/cpu[0-9]*')):
        f=f'/sys/devices/system/cpu/cpu{c}/cache/index3/shared_cpu_list'
        if not os.path.exists(f): continue
        d=open(f).read().strip()
        if d not in seen: seen.add(d); out.append(d)
    return out
def node_of(cpus):
    for n in glob.glob('/sys/devices/system/node/node[0-9]*'):
        nid=int(re.search(r'node(\d+)$',n).group(1))
        if cpus[0] in expand(open(f'{n}/cpulist').read().strip()): return nid
    return -1
topo=""
try: topo=subprocess.run(["rocm-smi","--showtoponuma"],capture_output=True,text=True).stdout
except Exception: pass
gpu_node={int(m.group(1)):int(m.group(2)) for m in re.finditer(r"GPU\[(\d+)\].*?Numa Node:\s*(\d+)",topo)}
if not gpu_node: raise SystemExit
by={}
for d in l3_domains(): by.setdefault(node_of(expand(d)),[]).append(d)
for n in by: by[n].sort(key=lambda d: expand(d)[0])
for n in sorted(by):
    for i,g in enumerate(sorted(k for k,v in gpu_node.items() if v==n)):
        if i < len(by[n]): print(f"{g} {by[n][i]}")
CCDPY

PIN_CCD="${PIN_CCD:-1}"
pin_workers_to_ccd() {
    [ "$PIN_CCD" = "1" ] || return 0
    [ -s "$RESULT_DIR/ccdmap.txt" ] || return 0
    local pinned=0
    while read -r _g _cpus; do
        for _p in $(pgrep -f "VLLM::Worker_TP${_g}([^0-9]|$)" 2>/dev/null); do
            for _t in /proc/$_p/task/*; do
                taskset -pc "$_cpus" "${_t##*/}" >/dev/null 2>&1 && pinned=$((pinned+1)) || true
            done
        done
    done < "$RESULT_DIR/ccdmap.txt"
    echo "[pin-ccd] pinned $pinned threads"
}

wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

pin_workers_to_ccd || true

if [ "${EVAL_ONLY:-false}" = "true" ]; then
    run_eval --port "$PORT"
else
    build_replay_cmd "$RESULT_DIR"
    run_agentic_replay_and_write_outputs "$RESULT_DIR"
fi

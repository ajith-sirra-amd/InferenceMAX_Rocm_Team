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
export APPLY_PR59591="${APPLY_PR59591:-0}"  # Kimi-K3: shard latent-MoE up-proj by TP rank -- measured net loss (-1.3% tput), replaced by #59693 below
export APPLY_PR59069="${APPLY_PR59069:-1}"  # Kimi-K3: fuse AttnRes output + per-token FP8 quant -- part of the known-working bundle (+1.2% measured together with #59070/#59693/#59966/#54625)
export APPLY_PR59070="${APPLY_PR59070:-1}"  # ROCm MLA: keep DCP prefill context FP8 through AllGather -- part of the known-working bundle
export APPLY_PR59693="${APPLY_PR59693:-1}"  # Kimi-K3: token-sharded residual stream for long prefills -- part of the known-working bundle
export APPLY_PR59965="${APPLY_PR59965:-0}"  # ROCm DCP: default MLA DCP verify to round-robin asm -- MERGED + already native in current pinned image; no-op for our workload (spec-decode batches only), excluded
export APPLY_PR59966="${APPLY_PR59966:-1}"  # ROCm DCP: gather MLA decode query without byte-wise strided copies -- part of the known-working bundle
export APPLY_PR54627="${APPLY_PR54627:-0}"  # prefill_schedule_interval outside DP -- +2.6% tput/-7.5-17% TPOT but +313-352% TTFT -- not worth it, TTFT cost too large for the TPOT gain
export APPLY_PR54625="${APPLY_PR54625:-1}"  # cache-aware admission ordering -- part of the known-working bundle
export APPLY_PR58743="${APPLY_PR58743:-0}"  # Kimi-K3: support BF16 KDA recurrent state -- OFF: crashes decode, see block below
export APPLY_PR54494="${APPLY_PR54494:-0}"  # ROCm DCP: MLA query replication, skip per-layer query all-gather -- OFF: hit a real NCCL _ALLGATHER_BASE collective deadlock (600s watchdog timeout, engine crash) ~40min into the C96 isolated run, likely DCPGroupColumnParallelLinear's per-forward collective under asymmetric DCP-rank batches; unsafe until root-caused
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
        else MAX_BATCHED_TOKENS="${MAX_BATCHED_TOKENS:-24576}"; fi
        ;;
    64|72)
        DCP_SIZE="${DCP_SIZE:-8}"
        OFFLOAD_POLICY=harness
        MAX_BATCHED_TOKENS="${MAX_BATCHED_TOKENS:-24576}"
        if [ "$CONC" -eq 64 ]; then
            MAX_NUM_SEQS="${MAX_NUM_SEQS:-80}"
        else
            MAX_NUM_SEQS="${MAX_NUM_SEQS:-96}"
        fi
        SPEC_NUM_TOKENS="${SPEC_NUM_TOKENS:-${SPEC_K:-3}}"
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
            echo "MTP: k=$SPEC_NUM_TOKENS LIVE block rejection (accuracy gate) under DCP=$DCP_SIZE, draft_kv=$DRAFT_KV_DTYPE"
        else
            SPEC_ARGS=(--speculative-config "{$SPEC_BASE,\"rejection_sample_method\": \"synthetic\", \"synthetic_acceptance_length\": $SYNTHETIC_ACCEPT_LEN}")
            echo "MTP: k=$SPEC_NUM_TOKENS synthetic_accept=$SYNTHETIC_ACCEPT_LEN under DCP=$DCP_SIZE, draft_kv=$DRAFT_KV_DTYPE"
        fi
        # Ladder must be mns x spec_rows (HANDOFF.md 2026-09-23) -- capping
        # it sends batches above the cap to eager, costs ~40% TPOT.
        SPEC_ROWS=$(( SPEC_NUM_TOKENS + 1 ))
        ;;
    *)
        DCP_SIZE="${DCP_SIZE:-8}"
        OFFLOAD_POLICY=harness
        if [ "$CONC" -gt 64 ]; then MAX_BATCHED_TOKENS="${MAX_BATCHED_TOKENS:-24576}"
        else MAX_BATCHED_TOKENS="${MAX_BATCHED_TOKENS:-8192}"; fi
        if [ "$CONC" -lt 72 ]; then MAX_NUM_SEQS="${MAX_NUM_SEQS:-$(( CONC * 14 / 10 ))}"
        elif [ "$CONC" -eq 80 ]; then MAX_NUM_SEQS="${MAX_NUM_SEQS:-112}"
        else MAX_NUM_SEQS="${MAX_NUM_SEQS:-144}"; fi
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
# #59693 -- token-sharded residual stream for long prefills. Previously
# disabled after a live hang (run 37474503331, isolated -- only this patch
# active): Rank 0 and Rank 4 stuck at wildly different NCCL SeqNums.
# 2026-10-08: PR updated upstream -- replaced two addmm_ calls with a plain
# GEMM + add. hipBLASLt's C-accumulating bf16 GEMM faults at specific row
# counts (2914-2925 rows at 7168x3584; 23393-23405 rows at 896x3584), both
# reachable by our long-prefill agentic workload at TP8. Plausible root cause
# of the earlier hang. Patch re-staged (diff updated, dry-run + py_compile
# verified) and re-enabled below for a fresh isolated test -- not yet
# confirmed fixed by a live run.
# -----------------------------------------------------------------------------
apply_pr59693() {
    [ "${APPLY_PR59693:-0}" = "1" ] || { echo "[pr59693] disabled"; return 0; }
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
if [ "${APPLY_PR59693:-0}" = "1" ]; then
    export VLLM_KIMI_K3_AMD_PREFILL_SP_MIN_TOKENS="${VLLM_KIMI_K3_AMD_PREFILL_SP_MIN_TOKENS:-1024}"
fi

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
# Hardset, not "${VAR:-default}": the outer launcher (runners/launch_mi355x-amd.sh)
# always exports this non-empty before the container starts, so a ":-" fallback
# here never fires -- it's shadowed by whatever the outer launcher set first.
PREFILL_SCHEDULE_INTERVAL=33

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
    # Hardset, not "${VAR:-default}": see PREFILL_SCHEDULE_INTERVAL comment above --
    # the outer launcher exports these non-empty before the container starts.
    CACHE_AWARE_ARGS=(
        --cache-aware-admission-window 64
        --cache-aware-admission-threshold 0.0
    )
fi

# -----------------------------------------------------------------------------
# #54494 -- DCP MLA query replication. Each rank materializes the DCP group's
# full query head set locally instead of all-gathering it every layer --
# trades a small redundant local projection for one fewer per-layer collective.
# Unmeasured upstream ("4K/4K serving perf still being rerun, not reported
# yet"); GSM8K passed clean (0.9629 vs 0.9598 off). Hand-adapted: the PR's
# fp4-branch hunk targets code since refactored to a replace_parameter-based
# pattern (CUDA-graph-safe in-place reload); the PR's own already-written
# fp4-branch logic was moved to the correct post-refactor insertion point, not
# invented. Needs VLLM_DCP_Q_REPLICATE=1 -- off by default upstream.
# -----------------------------------------------------------------------------
apply_pr54494() {
    [ "${APPLY_PR54494:-0}" = "1" ] || { echo "[pr54494] disabled"; return 0; }
    local diff_file
    diff_file="$(cd "$(dirname "$0")" && pwd)/patches/pr54494-dcp-query-replication.diff"
    [ -f "$diff_file" ] || { echo "[pr54494] missing $diff_file" >&2; return 1; }
    local site
    site="$(python3 -c 'import vllm,os;print(os.path.dirname(os.path.dirname(vllm.__file__)))')"
    if python3 -c 'import inspect,vllm.v1.attention.ops.dcp as m
import sys; sys.exit(0 if "resolve_dcp_q_replicate" in inspect.getsource(m) else 1)' 2>/dev/null; then
        echo "[pr54494] already present, nothing to do"; return 0
    fi
    ( cd "$site" && patch -p1 --forward --silent < "$diff_file" ) || return 1
    python3 -c 'import py_compile
for f in ["vllm/model_executor/layers/attention/mla_attention.py","vllm/models/kimi_k3/amd/linear.py","vllm/v1/attention/ops/dcp.py"]:
    py_compile.compile("'"$site"'/"+f,doraise=True)' || return 1
    echo "[pr54494] applied"
}
apply_pr54494 || { echo "[pr54494] patch failed, refusing to run" >&2; exit 1; }
if [ "${APPLY_PR54494:-0}" = "1" ]; then
    export VLLM_DCP_Q_REPLICATE=1
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

# KERNEL_INSPECT=1 -- dump the BF16 prefill attention kernel's real signature
# from inside the production container (needs a live GPU, aiter queries
# rocminfo at import time), then exit before the server starts. Answers one
# question: does flash_attn_varlen_func expose a softmax/accumulator-dtype
# knob. See Kimi-K3-Where-The-Time-Goes.md FMHA plan.
if [ "${KERNEL_INSPECT:-0}" = "1" ]; then
    echo "[kernel-inspect] VRAM canary + backend_supports_prefill_query_quantization investigation"
    {
        echo "=== rocm-smi VRAM canary (checking for stranded memory from the cancelled run) ==="
        rocm-smi --showmeminfo vram 2>&1
        echo
    } > "$RESULT_DIR/kernel_inspect.txt" 2>&1
    python3 - >> "$RESULT_DIR/kernel_inspect.txt" 2>&1 <<'PYEOF' || true
import inspect

def find_all(src, needle):
    out = []
    i = src.find(needle)
    while i != -1:
        out.append(i)
        i = src.find(needle, i + 1)
    return out

print("=" * 80)
print("mla_attention.py -- search for backend_supports_prefill_query_quantization")
print("(the call site found earlier lives here, not necessarily its definition)")
print("=" * 80)
try:
    import vllm.model_executor.layers.attention.mla_attention as mla
    src = inspect.getsource(mla)
    for needle in ("backend_supports_prefill_query_quantization", "supports_prefill_query_quantization"):
        hits = find_all(src, needle)
        print(f"'{needle}': {len(hits)} hits")
        for idx in hits:
            print(src[max(0, idx - 300):idx + 400])
            print("---")
except Exception as e:
    print("mla_attention scan failed:", e)

print("\n" + "=" * 80)
print("base.py -- MLAPrefillBackend class: ALL methods/ClassVars (not just 'quant' name match)")
print("=" * 80)
try:
    import vllm.v1.attention.backends.mla.prefill.base as base_mod
    src2 = inspect.getsource(base_mod)
    print(src2)
except Exception as e:
    print("base.py dump failed:", e)

print("\n" + "=" * 80)
print("AiterFlashAttnPrefillBackend -- full dir(), not just name-filtered")
print("=" * 80)
try:
    import vllm.v1.attention.backends.mla.prefill.aiter_flash_attn as aff
    cls = aff.AiterFlashAttnPrefillBackend
    for name in sorted(dir(cls)):
        if name.startswith("__"):
            continue
        try:
            val = getattr(cls, name)
        except Exception as e:
            val = f"<error: {e}>"
        print(f"  {name} = {val}")
except Exception as e:
    print("aiter_flash_attn scan failed:", e)
PYEOF
    cat "$RESULT_DIR/kernel_inspect.txt"
    echo "[kernel-inspect] done, exiting before server start"
    exit 0
fi

# Opt-in: FP8 prefill query quantization. Native vLLM feature, no patch
# needed -- the backend we use (ROCM_AITER_FA) has no special-cased FP8 path
# of its own, it just forwards whatever dtype q/k/v arrive in to
# aiter.flash_attn_varlen_func, which already has a dedicated gfx950 FP8
# kernel (gated purely on tensor dtype, confirmed SUPPORTED=True for our
# exact MLA shapes: 12 heads/rank, qk=192, v=128). vLLM's own code recommends
# this explicitly for ISL>=4K workloads -- ours runs ~100K. Unverified
# whether our backend is on the allowlist; this run is the test.
ENABLE_FP8_PREFILL_QUERY_QUANT="${ENABLE_FP8_PREFILL_QUERY_QUANT:-0}"
if [ "$ENABLE_FP8_PREFILL_QUERY_QUANT" = "1" ]; then
    ATTENTION_CONFIG_JSON='{"mla_prefill_backend":"ROCM_AITER_FA","use_prefill_query_quantization":true}'
else
    ATTENTION_CONFIG_JSON='{"mla_prefill_backend":"ROCM_AITER_FA"}'
fi

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
    # --mamba-ssm-cache-dtype "$MAMBA_SSM_CACHE_DTYPE"
    --enable-auto-tool-choice
    --tool-call-parser kimi_k3
    --reasoning-parser kimi_k3
    --enable-prefix-caching
    --enable-prompt-tokens-details
    --no-async-scheduling
    --attention-config "$ATTENTION_CONFIG_JSON"
    # --prefill-schedule-interval "$PREFILL_SCHEDULE_INTERVAL"  # #54627 disabled -- TTFT cost too large for the TPOT gain
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

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
export APPLY_PR54627="${APPLY_PR54627:-0}"  # prefill_schedule_interval outside DP -- dormant, apply call commented out below
export APPLY_PR54625="${APPLY_PR54625:-0}"  # cache-aware admission ordering -- dormant, apply call commented out below

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
# vllm-project/vllm#59591 -- Kimi-K3: store only the current rank's shard of
# the latent-MoE up-proj weight, instead of a full ReplicatedLinear copy on
# every rank. At TP=8 that is ~3.9GB/rank of unused weight across 92 MoE
# layers, HBM that would otherwise go to KV cache. Open, unmerged. No new CLI
# flag -- pure internal behavior change, always active once patched.
# PR's own measurements (agentic, prefix caching): +0.9% to +31% request
# throughput, up to +28.91% total token throughput at concurrency 70.
# Test file hunk dropped -- not shipped in the installed package.
# Dry-run + real apply + py_compile + import verified clean against
# nightly-rocm100-18f8f960 on 2026-10-01 (0 rejects, 0 fuzz).
#
# OFF by default (APPLY_PR59591=0): conflicts with #59693's token-sharded SP
# path -- #59693's _sp_tail does `out.addmm_(latent, up_proj.weight.t())`
# assuming the stock full-width up_proj; this patch shards that same weight,
# producing a confirmed live-run crash (input [*,7168] vs output [*,896]).
# Pick one: this patch sharded-always, or #59693 sharded-above-a-token-
# threshold. Currently favoring #59693 (broader win, same shape of benefit).
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
# vllm-project/vllm#59069 -- Kimi-K3: fuse AttnRes output with per-token FP8
# input quantization, eliminating a separate quantization kernel launch when
# the consuming layer's quant scheme is kFp8DynamicTokenSym. Gracefully falls
# back to the original path for non-matching consumers -- safe to apply even
# if it never engages for this model's actual quant scheme. Open, unmerged.
# No new CLI flag. PR's own measurements (N2112/K7168 MLA op): 3-5.2% faster
# across 16-16384 token batches; 931 per-token quant kernel calls eliminated.
# Test/benchmark file hunks dropped -- not shipped in the installed package.
# Stacks cleanly on top of #59591 (both touch linear.py). Dry-run + real
# apply + py_compile + import verified clean against nightly-rocm100-18f8f960
# on 2026-10-05 (0 rejects, 0 fuzz).
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
# vllm-project/vllm#59070 -- ROCm MLA: keep DCP prefill context in FP8 through
# the AllGather instead of upcasting to BF16 first, with a fused kernel
# combining reorganization, dequantization, projection, and packing. Directly
# targets our DCP=8 MLA configuration. Open, unmerged. No new CLI flag.
# PR's own measurements (8x MI355X): AllGather 2.1-1.8x faster across
# 1,446-98,214 token contexts; cold 32K median TTFT 1701ms -> 1674ms.
# Test/CI-config file hunks dropped -- not shipped in the installed package.
# Dry-run + real apply + py_compile + import verified clean against
# nightly-rocm100-18f8f960 on 2026-10-05 (0 rejects, some offset, 0 fuzz).
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
# vllm-project/vllm#59693 -- Kimi-K3: token-sharded residual stream for long
# prefills on ROCm. Every TP rank repeats identical per-token work during
# prefill; this shards the residual stream over tokens instead, same comm
# volume, less redundant compute. Exclusively ROCm + Kimi-K3, off by default,
# does not touch CUDA-graph decode. Open, unmerged.
# Gate: VLLM_KIMI_K3_AMD_PREFILL_SP_MIN_TOKENS (env var, default 0 = off).
# PR's own recommended value is 1024; our agentic ISLs (70k-360k+ typical)
# are squarely in its target range. PR's own measurements (8x MI355X, TP8):
# 10k tok +5.7% throughput/-14.1% TTFT, 50k tok +11.1%, 200k tok +9.3%,
# geomean +6.1%. Accuracy validated on GSM8K + passkey retrieval to 190k.
# Test file hunk dropped -- not shipped in the installed package. Stacks
# cleanly on top of #59591/#59069 (all three touch linear.py; two hunks land
# with fuzz 1-2, both verified landing in the right place -- see commit).
# Dry-run + real apply + py_compile + import verified clean against
# nightly-rocm100-18f8f960 on 2026-10-05.
#
# MIN_TOKENS is set below (1024, the PR's own recommendation) -- this is now
# ACTIVE, not inert. Its should_shard() path (sp.py) conflicts with #59591's
# sharded up_proj (confirmed via a live run: RuntimeError in
# latent_moe_runner._sp_tail's out.addmm_, input [*, 7168] full width vs
# output [*, 896] = 7168/8 the #59591 shard), so #59591 is OFF by default
# (see its own block above) while this is on. _sp_tail's own docstring says
# "up-project this rank's tokens with the full weight" -- it was authored
# and tested against the stock, unsharded up_proj, which is exactly what's
# present with #59591 off. Do not set APPLY_PR59591=1 while this is 1.
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
# vllm-project/vllm#59965 -- ROCm DCP: default VLLM_ROCM_AITER_MLA_DCP_VERIFY
# to "auto" (was "segmented"), which auto-routes to the round-robin asm decode
# where it applies and falls back otherwise. Open, unmerged. No new CLI flag
# -- changes an existing env var's default, falls back gracefully when
# conditions (DSpark spec-decode, interleave=1) aren't met, so safe even in
# our no-MTP C70/C72 config. PR's own measurements (8x MI355X, TP8/DCP8, with
# spec-decoding): +50.8% throughput, -40.8% median TPOT, -46.6% e2e latency.
# Gains are tied to spec-decode batches we don't currently run at C70/C72, so
# treat as a safety-net default change here, not a guaranteed win for us.
# Test file hunk dropped -- not shipped in the installed package. Stacks
# cleanly with #59070 (both touch rocm_aiter_mla.py). Dry-run + real apply +
# py_compile + import verified clean against nightly-rocm100-18f8f960 on
# 2026-10-06 (0 rejects, some offset, 0 fuzz).
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
# vllm-project/vllm#59966 -- ROCm DCP: gather the MLA decode query without
# byte-wise strided copies -- writes token-major results directly into the
# backend's padded storage in one pass instead of all_gather + movedim/
# reshape + repeat + contiguous. Directly targets our TP8/DCP8 Kimi-K3 config.
# Open, unmerged. Gate: VLLM_USE_DIRECT_DCP_Q_GATHER (our script already
# exports this as 0 -- see env exports above -- so flip it to 1 below to
# actually benefit; the patch alone is otherwise inert). PR's own
# measurements (MI355X, Kimi-K3, 24 MLA layers): byte-wise copies 3.07ms ->
# 1.67ms/decode step (-46%), overall step time 72.40ms -> 70.45ms (-2.7%),
# bit-identical outputs. Test file hunks dropped -- not shipped in the
# installed package. Stacks cleanly with #59070/#59965 (all touch
# rocm_aiter_mla.py). Dry-run + real apply + py_compile + import verified
# clean against nightly-rocm100-18f8f960 on 2026-10-06 (0 rejects, some
# offset, 0 fuzz).
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

# # -----------------------------------------------------------------------------
# # vllm-project/vllm#54627 -- prefill_schedule_interval outside data parallelism
# # -----------------------------------------------------------------------------
# # Open, unmerged. `prefill_schedule_interval` (SchedulerConfig, default 1 = off)
# # already exists and is CLI-exposed (--prefill-schedule-interval), but today it
# # is a no-op outside data-parallel deployments -- our config is DCP=8, DP=1, so
# # the flag alone does nothing. This PR makes the scheduler-side interval logic
# # work under DCP too. Targets Kimi-K3-Where-The-Time-Goes.md's finding that 57%
# # of C72 steps carry prefill and balloon 46ms -> 440-754ms per step.
# # 2/4 hunks (test files) dropped -- not shipped in the installed package.
# # Dry-run + real apply + py_compile verified clean against
# # nightly-rocm100-e9757321 on 2026-09-25 (0 rejects, stacks cleanly under #54625).
# apply_pr54627() {
#     [ "${APPLY_PR54627:-1}" = "1" ] || { echo "[pr54627] disabled"; return 0; }
#     local diff_file
#     diff_file="$(cd "$(dirname "$0")" && pwd)/patches/pr54627-prefill-interval-nondp.diff"
#     [ -f "$diff_file" ] || { echo "[pr54627] missing $diff_file" >&2; return 1; }
#     local site
#     site="$(python3 -c 'import vllm,os;print(os.path.dirname(os.path.dirname(vllm.__file__)))')"
#     if python3 -c 'import inspect,vllm.v1.core.sched.scheduler as s
# import sys; sys.exit(0 if "last_prefill_step" in inspect.getsource(s) else 1)' 2>/dev/null; then
#         echo "[pr54627] already present, nothing to do"; return 0
#     fi
#     ( cd "$site" && patch -p1 --forward --silent < "$diff_file" ) || return 1
#     python3 -c 'import py_compile;py_compile.compile("'"$site"'/vllm/v1/core/sched/scheduler.py",doraise=True);py_compile.compile("'"$site"'/vllm/config/scheduler.py",doraise=True)' || return 1
#     echo "[pr54627] applied"
# }
# apply_pr54627 || { echo "[pr54627] patch failed, refusing to run" >&2; exit 1; }
# PREFILL_SCHEDULE_INTERVAL="${PREFILL_SCHEDULE_INTERVAL:-1}"

# apply_pr54625() {
#     [ "${APPLY_PR54625:-1}" = "1" ] || { echo "[pr54625] disabled"; return 0; }
#     local diff_file
#     diff_file="$(cd "$(dirname "$0")" && pwd)/patches/pr54625-cache-aware-admission.diff"
#     [ -f "$diff_file" ] || { echo "[pr54625] missing $diff_file" >&2; return 1; }
#     local site
#     site="$(python3 -c 'import vllm,os;print(os.path.dirname(os.path.dirname(vllm.__file__)))')"
#     if python3 -c 'import vllm.config.scheduler as s; import sys; sys.exit(0 if hasattr(s.SchedulerConfig, "cache_aware_admission_window") else 1)' 2>/dev/null; then
#         echo "[pr54625] already present, nothing to do"; return 0
#     fi
#     ( cd "$site" && patch -p1 --forward --silent < "$diff_file" ) || return 1
#     python3 -c 'import py_compile
# for f in ["vllm/config/scheduler.py","vllm/engine/arg_utils.py","vllm/v1/core/kv_cache_manager.py","vllm/v1/core/sched/scheduler.py"]:
#     py_compile.compile("'"$site"'/"+f,doraise=True)' || return 1
#     echo "[pr54625] applied"
# }
# apply_pr54625 || { echo "[pr54625] patch failed, refusing to run" >&2; exit 1; }
# # The two flags below do not exist on vLLM's CLI parser unless #54625 is
# # applied -- unlike --prefill-schedule-interval, which is already a real flag
# # pre-patch. Keep them out of VLLM_CMD entirely when the patch is off, or
# # "unrecognized arguments" kills every non-#54625 dispatch.
# CACHE_AWARE_ARGS=()
# if [ "${APPLY_PR54625:-1}" = "1" ]; then
#     CACHE_AWARE_ARGS=(
#         --cache-aware-admission-window "${CACHE_AWARE_ADMISSION_WINDOW:-0}"
#         --cache-aware-admission-threshold "${CACHE_AWARE_ADMISSION_THRESHOLD:-0.5}"
#     )
# fi

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
    --enable-auto-tool-choice
    --tool-call-parser kimi_k3
    --reasoning-parser kimi_k3
    --enable-prefix-caching
    --enable-prompt-tokens-details
    --no-async-scheduling
    --attention-config '{"mla_prefill_backend":"ROCM_AITER_FA"}'
    # --prefill-schedule-interval "$PREFILL_SCHEDULE_INTERVAL"
    # "${CACHE_AWARE_ARGS[@]}"
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

# python3 - <<'CCDPY' > "$RESULT_DIR/ccdmap.txt" 2>/dev/null || true
# import subprocess, re, os, glob
# def expand(s):
#     v=[]
#     for part in s.split(','):
#         if '-' in part:
#             a,b=part.split('-'); v+=list(range(int(a),int(b)+1))
#         else: v.append(int(part))
#     return v
# def l3_domains():
#     seen,out=set(),[]
#     for c in sorted(int(re.search(r'cpu(\d+)$',x).group(1)) for x in glob.glob('/sys/devices/system/cpu/cpu[0-9]*')):
#         f=f'/sys/devices/system/cpu/cpu{c}/cache/index3/shared_cpu_list'
#         if not os.path.exists(f): continue
#         d=open(f).read().strip()
#         if d not in seen: seen.add(d); out.append(d)
#     return out
# def node_of(cpus):
#     for n in glob.glob('/sys/devices/system/node/node[0-9]*'):
#         nid=int(re.search(r'node(\d+)$',n).group(1))
#         if cpus[0] in expand(open(f'{n}/cpulist').read().strip()): return nid
#     return -1
# topo=""
# try: topo=subprocess.run(["rocm-smi","--showtoponuma"],capture_output=True,text=True).stdout
# except Exception: pass
# gpu_node={int(m.group(1)):int(m.group(2)) for m in re.finditer(r"GPU\[(\d+)\].*?Numa Node:\s*(\d+)",topo)}
# if not gpu_node: raise SystemExit
# by={}
# for d in l3_domains(): by.setdefault(node_of(expand(d)),[]).append(d)
# for n in by: by[n].sort(key=lambda d: expand(d)[0])
# for n in sorted(by):
#     for i,g in enumerate(sorted(k for k,v in gpu_node.items() if v==n)):
#         if i < len(by[n]): print(f"{g} {by[n][i]}")
# CCDPY

# PIN_CCD="${PIN_CCD:-1}"
# pin_workers_to_ccd() {
#     [ "$PIN_CCD" = "1" ] || return 0
#     [ -s "$RESULT_DIR/ccdmap.txt" ] || return 0
#     local pinned=0
#     while read -r _g _cpus; do
#         for _p in $(pgrep -f "VLLM::Worker_TP${_g}([^0-9]|$)" 2>/dev/null); do
#             for _t in /proc/$_p/task/*; do
#                 taskset -pc "$_cpus" "${_t##*/}" >/dev/null 2>&1 && pinned=$((pinned+1)) || true
#             done
#         done
#     done < "$RESULT_DIR/ccdmap.txt"
#     echo "[pin-ccd] pinned $pinned threads"
# }

wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

# pin_workers_to_ccd || true

if [ "${EVAL_ONLY:-false}" = "true" ]; then
    run_eval --port "$PORT"
else
    build_replay_cmd "$RESULT_DIR"
    run_agentic_replay_and_write_outputs "$RESULT_DIR"
fi

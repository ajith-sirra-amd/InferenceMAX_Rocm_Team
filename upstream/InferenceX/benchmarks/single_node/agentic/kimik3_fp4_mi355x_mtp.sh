#!/usr/bin/env bash
set -eo pipefail
set -x
source "$(dirname "$0")/../../benchmark_lib.sh"
wait_for_amd_gpu_clean

export EVAL_ONLY="${EVAL_ONLY:-false}"
check_env_vars MODEL TP CONC KV_OFFLOADING TOTAL_CPU_DRAM_GB RESULT_DIR DURATION EP_SIZE
check_env_vars DCP_SIZE EVAL_ONLY

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
# QUICK_REDUCE=NONE under test on run 38359850695 (INT4 preempts the fast AR,
# HANDOFF: FP variant -8.39%). INT4 here (baseline) to isolate the ag_rs variable.
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

# A2A direct symm-mem DCP collective -- EXPERIMENT 2026-10-13 (vs baseline 14,445).
# Probe confirmed symm_mem_available=True on this image and Q_GATHER direct is
# already working (-2.7% step time, bit-identical). Testing whether A2A-direct
# cuts the ~27% generic-PYNCCL DCP collective. KV_GATHER left 0 deliberately --
# it is the hang-path collective; test it separately only if A2A-direct wins.
export VLLM_USE_DIRECT_DCP_A2A=0  # A2A-direct under test on run 38359405528; baseline here to isolate the QUICK_REDUCE variable
export VLLM_USE_DIRECT_DCP_Q_GATHER=1
export VLLM_USE_DIRECT_DCP_KV_GATHER=0

SERVER_LOG="$RESULT_DIR/server.log"
mkdir -p "$RESULT_DIR"
SERVER_PID=""
cleanup_agentic_services() {
    local exit_code=$?
    trap - EXIT INT TERM
    set +e
    # Prior profiles stopped at "will wait for 2 children to exit". Request
    # shutdown of descendants as well as the root, then allow finalization.
    # The summary is diagnostic; the kernel CSVs are the measurement evidence.
    if [ "${ROCPROF_ENABLE:-0}" = "1" ] && [ -n "$SERVER_PID" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "[rocprof] sending SIGTERM to PID=$SERVER_PID, polling for $RESULT_DIR/rocprof_summary.txt..."
        # rocprof waits for its instrumented descendants. Signal the captured
        # tree from leaves to root before waiting for output generation.
        local profiler_children profiler_child
        profiler_children=$(_background_process_descendants "$SERVER_PID")
        while read -r profiler_child; do
            [[ -n "$profiler_child" ]] && kill -TERM "$profiler_child" 2>/dev/null || true
        done < <(printf '%s\n' "$profiler_children" | tac)
        kill -TERM "$SERVER_PID" 2>/dev/null || true
        local rocprof_deadline=$((SECONDS + 1200))
        while [ ! -s "$RESULT_DIR/rocprof_summary.txt" ] && [ $SECONDS -lt $rocprof_deadline ] && kill -0 "$SERVER_PID" 2>/dev/null; do
            sleep 5
        done
        if [ -s "$RESULT_DIR/rocprof_summary.txt" ]; then
            echo "[rocprof] rocprof_summary.txt detected after $SECONDS s -- flush complete."
        else
            echo "[rocprof] WARNING: rocprof_summary.txt never appeared within the 1200s cap; forcing teardown."
        fi
    fi
    stop_background_process_tree "$SERVER_PID" "vLLM server" 60
    exit "$exit_code"
}
trap cleanup_agentic_services EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

SPEC_ARGS=()
SPEC_ROWS=1
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
    64)
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
    echo "[kernel-inspect] FP8 chunked-context feasibility + DCP facts probe (new image)"
    {
        echo "=== rocm-smi VRAM canary ==="
        rocm-smi --showmeminfo vram 2>&1
        echo
    } > "$RESULT_DIR/kernel_inspect.txt" 2>&1
    python3 - >> "$RESULT_DIR/kernel_inspect.txt" 2>&1 <<'PYEOF' || true
import os, re, json, glob, inspect

R = {}
def sect(name, fn):
    try:
        R[name] = fn()
    except Exception as e:
        R[name] = f"ERR: {type(e).__name__}: {e}"

def _plat():
    from vllm.platforms import current_platform
    d = {"is_rocm": current_platform.is_rocm(), "is_cuda": current_platform.is_cuda()}
    try: d["device_capability"] = str(current_platform.get_device_capability())
    except Exception as e: d["device_capability"] = f"ERR {e}"
    try:
        from vllm.platforms.rocm import on_gfx950, on_mi3xx
        d["on_gfx950"] = on_gfx950(); d["on_mi3xx"] = on_mi3xx()
    except Exception as e: d["gfx"] = f"ERR {e}"
    return d
sect("platform", _plat)

def _aiter_meta():
    import aiter
    return {"version": getattr(aiter, "__version__", "?"), "file": aiter.__file__,
            "mla_symbols": sorted(s for s in dir(aiter)
                                  if any(k in s.lower() for k in ("mla", "prefill", "reduce")))}
sect("aiter_meta", _aiter_meta)

def _fp8_support():
    from vllm.v1.attention.backends.mla.rocm_aiter_mla import (
        _fp8_mla_prefill_supported, _aiter_mla_non_causal_asm_kernels)
    d = {"_fp8_mla_prefill_supported": _fp8_mla_prefill_supported(),
         "_aiter_mla_non_causal_asm_kernels(decode)": _aiter_mla_non_causal_asm_kernels()}
    try:
        from aiter import mla_prefill_ps_asm_fwd, mla_reduce_v1
        d["import_prefill_pair"] = "OK"
        d["prefill.__module__"] = getattr(mla_prefill_ps_asm_fwd, "__module__", "?")
        try: d["prefill.sig"] = str(inspect.signature(mla_prefill_ps_asm_fwd))
        except Exception as e: d["prefill.sig"] = f"no-sig ({e})"
        d["prefill.doc"] = (getattr(mla_prefill_ps_asm_fwd, "__doc__", None) or "")[:1500]
    except Exception as e:
        d["import_prefill_pair"] = f"ERR {e}"
    return d
sect("fp8_prefill_support", _fp8_support)

# DECISIVE: does aiter ship a NON-CAUSAL mla_prefill_ps_asm_fwd kernel for gfx950?
def _noncausal_evidence():
    import aiter
    adir = os.path.dirname(aiter.__file__)
    out = {"aiter_dir": adir}
    arts = []
    for pat in ("**/*prefill*", "**/*mla*asm*", "**/*.co", "**/*.hsaco", "**/*mla_prefill*"):
        arts += glob.glob(os.path.join(adir, pat), recursive=True)
    arts = sorted(set(os.path.relpath(a, adir) for a in arts))
    out["kernel_artifacts"] = arts[:150]
    out["artifacts_mentioning_causal"] = [a for a in arts if "causal" in a.lower()][:80]
    hits = []
    for root, _, files in os.walk(adir):
        for fn in files:
            if not fn.endswith(".py"):
                continue
            p = os.path.join(root, fn)
            try: txt = open(p, errors="ignore").read()
            except Exception: continue
            if "mla_prefill_ps_asm_fwd" in txt or (
                "prefill" in txt.lower() and "causal" in txt.lower() and "mla" in txt.lower()):
                rel = os.path.relpath(p, adir)
                for m in re.finditer(r".{0,70}(non.?causal|is_causal|causal).{0,70}", txt, re.I):
                    hits.append(f"{rel}: ...{m.group(0).strip()}...")
    out["source_causal_lines"] = hits[:100]
    return out
sect("noncausal_prefill_evidence", _noncausal_evidence)

def _heads():
    from vllm.v1.attention.backends.mla.rocm_aiter_mla import AiterMLAHelper
    d = {}
    for h in (8, 12, 16, 24, 32, 64):
        try:
            d[str(h)] = {"is_valid": AiterMLAHelper.is_valid_num_heads(h),
                         "fp8_prefill_heads": AiterMLAHelper.get_fp8_prefill_num_heads(h)}
        except Exception as e:
            d[str(h)] = f"ERR {e}"
    return d
sect("head_count_gate", _heads)

def _impl():
    from vllm.v1.attention.backends.mla import rocm_aiter_mla as RM
    cand = [n for n in dir(RM) if "Impl" in n or "Aiter" in n]
    d = {"module_classes": cand}
    for cn in cand:
        cls = getattr(RM, cn)
        if isinstance(cls, type):
            d[cn] = {"has_forward_mha_override": "forward_mha" in cls.__dict__,
                     "has__mla_fp8_prefill_attn": hasattr(cls, "_mla_fp8_prefill_attn")}
    return d
sect("backend_impl", _impl)

def _k3():
    import vllm
    vdir = os.path.dirname(vllm.__file__)
    files = []
    for root, _, fs in os.walk(vdir):
        for fn in fs:
            if fn.endswith(".py") and "kimi" in fn.lower():
                files.append(os.path.relpath(os.path.join(root, fn), vdir))
    grep = {}
    for rel in files:
        try: txt = open(os.path.join(vdir, rel), errors="ignore").read()
        except Exception: continue
        for sym in ("forward_mha", "_forward_prefill_fused", "forward_mqa",
                    "_mla_fp8_prefill", "mla_prefill_ps_asm_fwd", "chunked_context"):
            c = txt.count(sym)
            if c: grep.setdefault(rel, {})[sym] = c
    return {"kimi_files": files, "prefill_symbol_counts": grep}
sect("k3_model_routing", _k3)

def _dcp():
    d = {}
    try:
        import torch.distributed._symmetric_memory as _sm  # noqa
        d["torch_symm_mem_import"] = "OK"
    except Exception as e: d["torch_symm_mem_import"] = f"ERR {e}"
    try:
        from vllm.v1.attention.ops.cp_common import symm_mem_available
        d["cp_common.symm_mem_available"] = bool(symm_mem_available)
    except Exception as e: d["cp_common.symm_mem_available"] = f"ERR {e}"
    try:
        import vllm.envs as E
        d["VLLM_USE_DIRECT_DCP_*"] = {k: getattr(E, k, "ABSENT") for k in
            ("VLLM_USE_DIRECT_DCP_A2A", "VLLM_USE_DIRECT_DCP_Q_GATHER", "VLLM_USE_DIRECT_DCP_KV_GATHER")}
    except Exception as e: d["envs"] = f"ERR {e}"
    try:
        import typing
        from vllm.config.parallel import DCPCommBackend
        d["DCPCommBackend_options"] = list(typing.get_args(DCPCommBackend))
    except Exception as e: d["DCPCommBackend"] = f"ERR {e}"
    return d
sect("dcp_facts", _dcp)

print("KERNEL_INSPECT_JSON_BEGIN")
print(json.dumps(R, indent=2, default=str))
print("KERNEL_INSPECT_JSON_END")
PYEOF
    cat "$RESULT_DIR/kernel_inspect.txt"
    echo "[kernel-inspect] done, exiting before server start"
    exit 0
fi

# Opt-in: FP8 prefill query quantization (use_prefill_query_quantization).
# CONFIRMED NO-OP on this hardware -- do not re-enable expecting an effect.
# backend_supports_prefill_query_quantization() (vLLM mla_attention.py) gates
# on current_platform.is_device_capability_family(100) FIRST, which is
# NVIDIA GB200-only; it also requires an allowlisted backend name
# (FLASHINFER/TRTLLM_RAGGED/TOKENSPEED_MLA) that doesn't include ROCM_AITER_FA
# either. On AMD/ROCm this returns False unconditionally, so query dtype
# always stays model dtype (bf16) regardless of this flag -- any perf delta
# observed with it set is run-to-run noise, not real quantization.
#
# The real single-kernel FP8 path for this backend is a different mechanism,
# unrelated to this flag: aiter.flash_attn_varlen_func (which
# AiterFlashAttnPrefillBackend already calls for bf16) accepts q_descale/
# k_descale/v_descale kwargs and, when given fp8-dtype q/k/v + all three
# descales, routes through the same single CK kernel call with no extra host
# planning (aiter/ops/mha.py). That's the fp8ctx.py patch's next experiment
# track, not this flag.
ENABLE_FP8_PREFILL_QUERY_QUANT="${ENABLE_FP8_PREFILL_QUERY_QUANT:-0}"
if [ "$ENABLE_FP8_PREFILL_QUERY_QUANT" = "1" ]; then
    ATTENTION_CONFIG_JSON='{"mla_prefill_backend":"ROCM_AITER_FA","use_prefill_query_quantization":true}'
else
    ATTENTION_CONFIG_JSON='{"mla_prefill_backend":"ROCM_AITER_FA"}'
fi

# [FP8-CTX] Install the fp8 context-chunk prefill hook into the installed aiter
# MLA prefill backend (run_prefill_context_chunk). Heavily instrumented
# ([FP8-CTX] tag); any error -> returns None -> bf16 fallback.
# APPLY_FP8CTX=0 is the pure baseline (stock vLLM/aiter chunked-context BF16
# prefill, patch not installed at all); FP8CTX_MODE selects which kernel path
# the patch itself takes when APPLY_FP8CTX=1 (asm vs fused).
export APPLY_FP8CTX=1  # Optimized ASM smoke: compare real inputs before steady state.
export FP8CTX_MODE=asm
# Forced (not ${VAR:-default}): upstream/InferenceX/benchmarks/runtime_settings.sh
# does an unconditional `export AIPERF_WARMUP_REQUESTS_PER_LANE='10'` on the
# HOST runner before the container (and this script) ever starts, so a :-
# fallback here can never fire -- confirmed in run 38589134239's actual
# AIPerf command (`--warmup-requests-per-lane 10`, not the 1 a prior :-1
# override here was supposed to produce). Forcing it unconditionally here,
# after that host-side export has already landed, wins the same way
# ROCPROF_START_DELAY/DURATION's fix below does. Duration is controlled
# separately via the workflow's duration-override input (not bundled into
# AIPERF_EXPERIMENTAL_FAST's hardcoded 1200s) so it can be sized for conc=96's
# larger cudagraph capture ladder.
export AIPERF_WARMUP_REQUESTS_PER_LANE=1  # TEMP: profiling run, revert (unset) after dispatch
if [ "${APPLY_FP8CTX:-0}" = "1" ]; then
    _fp8_pdir="$(cd "$(dirname "$0")" && pwd)/patches"
    _fp8_site="$(python3 -c 'import vllm,os;print(os.path.dirname(vllm.__file__))')"
    cp "$_fp8_pdir/fp8ctx.py" "$_fp8_site/v1/attention/backends/mla/prefill/_fp8ctx.py" \
        && echo "[FP8-CTX-PATCH] helper copied"
    cp "$_fp8_pdir/fp8ctx_prep.py" "$_fp8_site/v1/attention/backends/mla/prefill/_fp8ctx_prep.py" || exit 1
    python3 - "$_fp8_site" <<'PYEOF'
import sys, py_compile
site = sys.argv[1]
f = site + "/v1/attention/backends/mla/prefill/aiter_flash_attn.py"
src = open(f).read()
if "_fp8ctx" in src:
    print("[FP8-CTX-PATCH] already present"); raise SystemExit(0)
ctx_anchor = ("        out, lse = self.flash_attn_varlen_func(\n"
              "            q=q,\n            k=k,\n            v=v,\n"
              "            cu_seqlens_q=chunk.query_start_loc,\n")
ctx_hook = ("        _fp8_r = None\n"
            "        try:\n"
            "            from vllm.v1.attention.backends.mla.prefill import _fp8ctx as _fp8_fx\n"
            "            _fp8_r = _fp8_fx.fp8_context_chunk(self, chunk, q, k, v)\n"
            "        except Exception:\n"
            "            _fp8_r = None\n"
            "        if _fp8_r is not None:\n"
            "            return _fp8_r\n")
new_anchor = ("        result = self.flash_attn_varlen_func(\n"
              "            q=q,\n            k=k,\n            v=v,\n"
              "            cu_seqlens_q=self._prefill_metadata.query_start_loc,\n")
new_hook = ("        try:\n"
            "            from vllm.v1.attention.backends.mla.prefill import _fp8ctx as _fp8_st\n"
            "            _fp8_st.selftest_noncausal(self)\n"
            "        except Exception:\n"
            "            pass\n")
missing = [n for n, a in (("ctx", ctx_anchor), ("new", new_anchor)) if a not in src]
if missing:
    print("[FP8-CTX-PATCH] ANCHOR NOT FOUND:", missing, "-- aborting"); raise SystemExit(3)
src = src.replace(ctx_anchor, ctx_hook + ctx_anchor, 1)
src = src.replace(new_anchor, new_hook + new_anchor, 1)
open(f, "w").write(src)
py_compile.compile(f, doraise=True)
# 3rd hook: native no-context fp8 path (_mla_fp8_prefill_attn) -- fires the self-test
# on cold-cache prefills, which bypass the prefill backend entirely.
f2 = site + "/v1/attention/backends/mla/rocm_aiter_mla.py"
src2 = open(f2).read()
if "_fp8_st3" not in src2:
    nat_anchor = ("        from vllm.platforms import current_platform\n"
                  "        from vllm.v1.worker.workspace import current_workspace_manager\n"
                  "\n"
                  "        fp8_dtype = current_platform.fp8_dtype()\n")
    nat_hook = ("        try:\n"
                "            from vllm.v1.attention.backends.mla.prefill import _fp8ctx as _fp8_st3\n"
                "            _fp8_st3.selftest_noncausal(self)\n"
                "        except Exception:\n"
                "            pass\n")
    # tracer: wrap flash_attn_varlen_func to reveal K3's actual prefill path
    flash_anchor = ("        from aiter import flash_attn_varlen_func\n"
                    "\n"
                    "        self.flash_attn_varlen_func = flash_attn_varlen_func\n")
    flash_add = ("        try:\n"
                 "            from vllm.v1.attention.backends.mla.prefill import _fp8ctx as _fp8_tr\n"
                 "            self.flash_attn_varlen_func = _fp8_tr.wrap_flash(self.flash_attn_varlen_func)\n"
                 "        except Exception:\n"
                 "            pass\n")
    miss2 = [n for n, a in (("native", nat_anchor), ("flash", flash_anchor)) if a not in src2]
    if miss2:
        print("[FP8-CTX-PATCH] ROCM_AITER_MLA ANCHOR NOT FOUND:", miss2, "-- aborting"); raise SystemExit(4)
    src2 = src2.replace(nat_anchor, nat_hook + nat_anchor, 1)
    src2 = src2.replace(flash_anchor, flash_anchor + flash_add, 1)
    open(f2, "w").write(src2)
    py_compile.compile(f2, doraise=True)
py_compile.compile(site + "/v1/attention/backends/mla/prefill/_fp8ctx.py", doraise=True)
print("[FP8-CTX-PATCH] hooks injected (ctx + newtok + native self-test) + compiled OK")
PYEOF
    _fp8_rc=$?
    [ "$_fp8_rc" = "0" ] || { echo "[FP8-CTX-PATCH] FAILED rc=$_fp8_rc -- refusing to run"; exit 1; }
fi

# Record the actual context-attention inputs without changing dispatch.
if [ "$APPLY_FP8CTX" = "0" ]; then
    python3 - <<'PRECISIONPY'
import pathlib
import py_compile
import vllm

path = pathlib.Path(vllm.__file__).parent / "v1/attention/backends/mla/prefill/aiter_flash_attn.py"
src = path.read_text()
anchor = ("        out, lse = self.flash_attn_varlen_func(\n"
          "            q=q,\n            k=k,\n            v=v,\n"
          "            cu_seqlens_q=chunk.query_start_loc,\n")
if anchor not in src:
    raise RuntimeError("Baseline precision probe: context anchor missing")
probe = (
    "        if not getattr(self, '_baseline_precision_logged', False):\n"
    "            self._baseline_precision_logged = True\n"
    "            import os\n"
    "            print('[BASELINE-PRECISION]', 'pid=', os.getpid(), "
    "'q=', tuple(q.shape), str(q.dtype), 'k=', tuple(k.shape), str(k.dtype), "
    "'v=', tuple(v.shape), str(v.dtype), 'causal=False', 'scale=', self.scale, flush=True)\n"
)
path.write_text(src.replace(anchor, probe + anchor, 1))
py_compile.compile(str(path), doraise=True)
print("[BASELINE-PRECISION] context input probe installed")
PRECISIONPY
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
    --enable-auto-tool-choice
    --tool-call-parser kimi_k3
    --reasoning-parser kimi_k3
    --enable-prefix-caching
    --enable-prompt-tokens-details
    --no-async-scheduling
    --attention-config "$ATTENTION_CONFIG_JSON"
    "${OFFLOAD_ARGS[@]}"
    "${CP_ARGS[@]}"
    "${EP_ARGS[@]}"
    "${SPEC_ARGS[@]}"
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
ROCPROF_ENABLE=0  # 600-second smoke: exclude profiler overhead from throughput.
LAUNCH_CMD=("${VLLM_CMD[@]}")
if [ "$ROCPROF_ENABLE" = "1" ]; then
    # Run A began measured replay ~2423s after server launch. Collect at
    # 3300s for 60s, within the 1500s replay; verify actual timing in logs.
    # Explicit recipe values override the launcher's older profiling values.
    ROCPROF_START_DELAY=3300
    ROCPROF_DURATION=60
    ROCPROF_DIR="$RESULT_DIR/rocprof"
    mkdir -p "$ROCPROF_DIR"
    LAUNCH_CMD=(
        rocprofv3 --kernel-trace --stats
        -d "$ROCPROF_DIR" -f csv
        --summary-output-file "$RESULT_DIR/rocprof_summary.txt"
        -P "${ROCPROF_START_DELAY}:${ROCPROF_DURATION}:1"
        --collection-period-unit sec
        --
        "${VLLM_CMD[@]}"
    )
    echo "[rocprof] enabled: start_delay=${ROCPROF_START_DELAY}s duration=${ROCPROF_DURATION}s output=$ROCPROF_DIR"
fi

printf '%q ' "${VLLM_CMD[@]}" | tee "$RESULT_DIR/vllm_command.txt"
printf '\n' | tee -a "$RESULT_DIR/vllm_command.txt"

"${LAUNCH_CMD[@]}" > "$SERVER_LOG" 2>&1 &  # LAUNCH_CMD = VLLM_CMD, or rocprofv3-wrapped when ROCPROF_ENABLE=1
SERVER_PID=$!
echo "Server PID: $SERVER_PID"

# rocprof_summary.txt and $ROCPROF_DIR/**/*.csv are picked up directly by the
# "Upload agentic raw results" artifact step (benchmark-tmpl.yml), so no
# watcher/copy step is needed here -- they just need to exist under
# $RESULT_DIR when the job finishes, which rocprofv3 guarantees once its
# collection window closes and the wrapped server process is torn down by
# cleanup_agentic_services on exit.

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

# SMOKE_TEST=1 -- fast reproduction for the #54494 NCCL deadlock
# investigation. Skips the full ~1hr agentic replay (whose warmup alone
# takes 30-50min before the hang has historically manifested) in favor of
# firing a bounded, sustained batch of direct curl requests against the
# live server, with VLLM_DEBUG_QREP=1 set so qrep_decode gets logged per
# rank per decode step. See PR54494-ROOT-CAUSE-AND-FIX.md.
if [ "${SMOKE_TEST:-0}" = "1" ]; then
    SMOKE_DURATION="${SMOKE_DURATION:-480}"
    SMOKE_CONCURRENCY="${SMOKE_CONCURRENCY:-32}"
    echo "[smoke-test] duration=${SMOKE_DURATION}s concurrency=${SMOKE_CONCURRENCY}"
    PROMPT_JSON=$(python3 -c '
import json
filler = ("The quick brown fox jumps over the lazy dog. " * 400)
print(json.dumps(filler))
')
    smoke_fire_one() {
        curl -s -o /dev/null -w "" --max-time 120 \
            -X POST "http://127.0.0.1:${PORT}/v1/chat/completions" \
            -H "Content-Type: application/json" \
            -d "{\"model\":\"${MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":${PROMPT_JSON}}],\"max_tokens\":$(( (RANDOM % 350) + 50 )),\"stream\":false}" \
            >> "$RESULT_DIR/smoke_test_curl.log" 2>&1
    }
    SMOKE_END=$(( $(date +%s) + SMOKE_DURATION ))
    SMOKE_PIDS=()
    for _ in $(seq 1 "$SMOKE_CONCURRENCY"); do
        smoke_fire_one &
        SMOKE_PIDS+=("$!")
        sleep 0.3
    done
    while [ "$(date +%s)" -lt "$SMOKE_END" ]; do
        for i in "${!SMOKE_PIDS[@]}"; do
            if ! kill -0 "${SMOKE_PIDS[$i]}" 2>/dev/null; then
                smoke_fire_one &
                SMOKE_PIDS[$i]=$!
            fi
        done
        sleep 5
    done
    echo "[smoke-test] duration elapsed, collecting results"
    for pid in "${SMOKE_PIDS[@]}"; do kill "$pid" 2>/dev/null || true; done
    wait 2>/dev/null || true
    echo "[smoke-test] qrep-debug lines from server log:"
    grep -c "qrep-debug" "$SERVER_LOG" 2>/dev/null || echo "0"
    grep "qrep-debug" "$SERVER_LOG" 2>/dev/null | tail -40
    echo "[smoke-test] any collective/engine errors:"
    grep -iE "ALLGATHER|Watchdog|DistBackendError|EngineDeadError|Traceback" "$SERVER_LOG" 2>/dev/null | tail -40
    echo "[smoke-test] done, exiting before full harness"
    exit 0
fi

if [ "${EVAL_ONLY:-false}" = "true" ]; then
    run_eval --port "$PORT"
else
    build_replay_cmd "$RESULT_DIR"
    run_agentic_replay_and_write_outputs "$RESULT_DIR"
fi

# A successful benchmark must not mask a failed experimental attention path.
if ! grep -q '\[FP8-VALIDATION\].*optimized_matches_current=True' "$SERVER_LOG"; then
    echo "[FP8-SMOKE] no successful live numerical comparison; refusing promotion"
    exit 1
fi
if grep -q 'context-chunk fp8 FAILED' "$SERVER_LOG"; then
    echo "[FP8-SMOKE] ASM fallback observed; refusing promotion"
    exit 1
fi
echo "[FP8-SMOKE] live comparison and ASM dispatch succeeded; full run requires timing review"

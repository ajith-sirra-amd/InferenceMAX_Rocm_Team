#!/usr/bin/env bash
set -euo pipefail
set -x
source "$(dirname "$0")/../../benchmark_lib.sh"
wait_for_amd_gpu_clean

export EVAL_ONLY="${EVAL_ONLY:-false}"
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
export VLLM_ROCM_QUICK_REDUCE_QUANTIZATION="${VLLM_ROCM_QUICK_REDUCE_QUANTIZATION:-NONE}"

export AITER_SITUV2_A8W4=1
export AITER_FLYDSL_STAGE2_FP8="${AITER_FLYDSL_STAGE2_FP8:-1}"
export AITER_BF16_FP8_MOE_BOUND=0
export AITER_DISABLE_FMHA_OPUS=1
export SAFETENSORS_FAST_GPU=1
export GPU_ARCHS=gfx950
export HSA_NO_SCRATCH_RECLAIM=1
export VLLM_USE_BREAKABLE_CUDAGRAPH="${VLLM_USE_BREAKABLE_CUDAGRAPH:-0}"
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

DCP_OVERRIDE="${DCP_OVERRIDE:-8}"
ALLOW_MTP_WITH_DCP="${ALLOW_MTP_WITH_DCP:-1}"

SPEC_ARGS=()
SPEC_ROWS=1
# KDA is a prefill kernel and lands on TTFT. The C32 no-MTP baseline ran triton
# (the old hardcoded value); leaving this at auto->fused makes KDA an
# uncontrolled variable in the MTP comparison. Pin triton to match the
# baseline -- measured neutral on the MTP arms at C4 and C12, so it costs
# nothing. Set KDA_PREFILL_BACKEND=fused for the DCP no-MTP arm, where fused
# was worth 3.1% at C48 and 4.4% at C72.
if [ -n "${KDA_PREFILL_BACKEND:-}" ]; then
    KDA_ARGS=(--additional-config "{\"kda_prefill_backend\":\"$KDA_PREFILL_BACKEND\"}")
else
    KDA_ARGS=(--additional-config '{"kda_prefill_backend":"triton"}')
fi
case "$CONC" in
    1|2|4|8|10|12|14|16)
        DCP_SIZE="${DCP_SIZE:-1}"
        OFFLOAD_POLICY=harness
        # Draft depth per concurrency. c1=6 is SA-matched and measured best at SA
        # (1,412 tok/s/GPU, ITL p90 8.13 = 123.0 tok/s/user). c4=5 and c12=4 come
        # from the C4 fixed-length sweep, where TPOT fell monotonically
        # 12.46 -> 11.26 -> 10.85 -> 10.38 ms across k=2..5 with throughput rising
        # 2,794 -> 3,321. Everything else stays on SA's k=3 for the band.
        case "$CONC" in
            1)  SPEC_NUM_TOKENS="${SPEC_NUM_TOKENS:-${SPEC_K:-6}}" ;;
            4)  SPEC_NUM_TOKENS="${SPEC_NUM_TOKENS:-${SPEC_K:-5}}" ;;
            10) SPEC_NUM_TOKENS="${SPEC_NUM_TOKENS:-${SPEC_K:-5}}" ;;
            12) SPEC_NUM_TOKENS="${SPEC_NUM_TOKENS:-${SPEC_K:-4}}" ;;
            14) SPEC_NUM_TOKENS="${SPEC_NUM_TOKENS:-${SPEC_K:-3}}" ;;
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
        SPEC_BASE="\"model\":\"Inferact/Kimi-K3-DSpark\",\"num_speculative_tokens\":$SPEC_NUM_TOKENS,\"method\":\"dspark\",\"attention_backend\":\"${DRAFT_ATTN_BACKEND:-ROCM_AITER_MLA}\",\"kv_cache_dtype\":\"$DRAFT_KV_DTYPE\",\"draft_sample_method\":\"probabilistic\""
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
        # MTP on the DCP arm, gated because it needs vllm#57085. k=3 matches the
        # band default and NV's d0, which runs mtp at dcp 8 on one aggregated
        # 8-GPU worker -- the config this arm has never been able to reach.
        if [ "${HIGH_CONC_MTP:-1}" = "1" ]; then
            SPEC_NUM_TOKENS="${SPEC_NUM_TOKENS:-${SPEC_K:-3}}"
            case "$SPEC_NUM_TOKENS" in
                1) SYNTHETIC_ACCEPT_LEN=1.85 ;;  2) SYNTHETIC_ACCEPT_LEN=2.51 ;;
                3) SYNTHETIC_ACCEPT_LEN=3.00 ;;  4) SYNTHETIC_ACCEPT_LEN=3.36 ;;
                5) SYNTHETIC_ACCEPT_LEN=3.62 ;;  6) SYNTHETIC_ACCEPT_LEN=3.75 ;;
                *) echo "[spec] no golden AL for k=$SPEC_NUM_TOKENS" >&2; exit 1 ;;
            esac
            DRAFT_KV_DTYPE="${DRAFT_KV_DTYPE:-fp8}"
            SPEC_BASE="\"model\":\"Inferact/Kimi-K3-DSpark\",\"num_speculative_tokens\":$SPEC_NUM_TOKENS,\"method\":\"dspark\",\"attention_backend\":\"${DRAFT_ATTN_BACKEND:-ROCM_AITER_MLA}\",\"kv_cache_dtype\":\"$DRAFT_KV_DTYPE\",\"draft_sample_method\":\"probabilistic\""
            SPEC_ARGS=(--speculative-config "{$SPEC_BASE,\"rejection_sample_method\": \"synthetic\", \"synthetic_acceptance_length\": $SYNTHETIC_ACCEPT_LEN}")
            SPEC_ROWS=$(( SPEC_NUM_TOKENS + 1 ))
            echo "MTP: k=$SPEC_NUM_TOKENS synthetic_accept=$SYNTHETIC_ACCEPT_LEN draft_kv=$DRAFT_KV_DTYPE (dcp arm)"
        fi
        if [ "$CONC" -gt 64 ]; then MAX_BATCHED_TOKENS="${MAX_BATCHED_TOKENS:-24576}"
        else MAX_BATCHED_TOKENS="${MAX_BATCHED_TOKENS:-8192}"; fi
        # Seats bound the max decode batch (mns * spec_rows) and so the size of
        # every captured graph. Trimming seats keeps full ladder coverage;
        # capping the ladder instead leaves the scheduler free to build batches
        # that have no graph and fall back to eager.
        if [ "$CONC" -eq 64 ] && [ "${#SPEC_ARGS[@]}" -gt 0 ]; then MAX_NUM_SEQS="${MAX_NUM_SEQS:-72}"
        elif [ "$CONC" -lt 72 ]; then MAX_NUM_SEQS="${MAX_NUM_SEQS:-$(( CONC * 14 / 10 ))}"
        elif [ "$CONC" -eq 72 ]; then MAX_NUM_SEQS="${MAX_NUM_SEQS:-96}"
        else MAX_NUM_SEQS="${MAX_NUM_SEQS:-112}"; fi
        ;;
esac
# e2e-tests.yml forwards dcp-size for some job types but not the agentic one, so
# a yaml dcp-size never reaches this script and DCP_SIZE silently falls back to
# the per-branch default (1 below conc 16, 8 above). Pin it here instead.
DCP_SIZE="${DCP_OVERRIDE:-$DCP_SIZE}"
export DCP_SIZE

# MTP draft verify under DCP is gated on aiter's segmented MLA decode; when the
# route is unavailable the run dies mid-serve rather than at startup. Drop
# speculation whenever DCP is on so the ladder collapses to one row per seat.
if [ "$DCP_SIZE" -gt 1 ] && [ "${#SPEC_ARGS[@]}" -gt 0 ] && [ "${ALLOW_MTP_WITH_DCP:-0}" != "1" ]; then
    SPEC_ARGS=()
    SPEC_ROWS=1
    echo "MTP: off (dcp=$DCP_SIZE)"
fi

GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
LAZY_OFFLOAD="${LAZY_OFFLOAD:-false}"
# FULL_DECODE_ONLY on every arm. Measured at C4 k=4 n=400 (runs 34936346363 vs
# 34940495620): piecewise cost 22.8 GiB of graph memory and 41.9% of the KV pool
# (3,295,310 -> 1,916,156 tokens) for a 0.6% TPOT change -- i.e. nothing. This is
# T284's -35.5% finding reproduced on the DCP-1 arm at mnbt 8192, so the penalty is
# chunk-size-driven, not arm-specific.
CUDAGRAPH_MODE="${CUDAGRAPH_MODE:-FULL_DECODE_ONLY}"

LADDER=$(( MAX_NUM_SEQS * SPEC_ROWS ))
# Graph memory is allocated on top of the gpu-memory-utilization budget, not
# inside it: C72 with ladder 384 measured ~29 GiB over the 0.90 budget and sat at
# 100% VRAM. Two of the three MTP+DCP attempts then hung inside capture_model.
# Cap the MTP+DCP arm at the same 96 the no-MTP arm and SA both capture; batches
# above the cap fall back to eager rather than failing.
if [ "$DCP_SIZE" -gt 1 ] && [ "${#SPEC_ARGS[@]}" -gt 0 ] && [ -n "${LADDER_CAP:-}" ] && [ "$LADDER" -gt "$LADDER_CAP" ]; then
    echo "[ladder] capping $LADDER -> $LADDER_CAP (mns=$MAX_NUM_SEQS spec_rows=$SPEC_ROWS)"
    LADDER="$LADDER_CAP"
fi
CUDAGRAPH_CAPTURE_SIZES=$(seq -s, 1 "$LADDER")
COMPILATION_CONFIG_ARGS=(--compilation-config "{\"mode\":3,\"cudagraph_mode\":\"$CUDAGRAPH_MODE\",\"max_cudagraph_capture_size\":$LADDER,\"custom_ops\":[\"+fused_rms_norm_gated\"],\"cudagraph_capture_sizes\":[$CUDAGRAPH_CAPTURE_SIZES]}")

CP_ARGS=(--attention-backend ROCM_AITER_MLA)
if [ "$DCP_SIZE" -gt 1 ]; then
    # a2a was picked for the no-MTP DCP arm and never validated against a
    # multi-token non-causal draft block. vLLM defaults to ag_rs
    # (set_dcp_defaults), and _ALLGATHER_BASE is exactly what deadlocks under
    # MTP+DCP, so the MTP arm takes the default while the shipping arm keeps a2a.
    if [ "${#SPEC_ARGS[@]}" -gt 0 ]; then
        DCP_COMM_BACKEND="${DCP_COMM_BACKEND:-a2a}"
    else
        DCP_COMM_BACKEND="${DCP_COMM_BACKEND:-a2a}"
    fi
    CP_ARGS+=(--decode-context-parallel-size "$DCP_SIZE" --dcp-comm-backend "$DCP_COMM_BACKEND" --cp-kv-cache-interleave-size 1)
fi

OFFLOAD_ARGS=()
OFFLOAD_LABEL="$OFFLOAD_POLICY"
if [ "$OFFLOAD_POLICY" = "none" ]; then
    :
elif agentic_kv_offload_enabled; then
    OFFLOAD_LABEL="${KV_OFFLOADING}"
    CPU_BYTES_PER_RANK=$(( TOTAL_CPU_DRAM_GB * 1000 * 1000 * 1000 / TOTAL_RANKS ))
    OFFLOAD_ARGS=(--kv-transfer-config "{\"kv_connector\":\"SimpleCPUOffloadConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"cpu_bytes_to_use_per_rank\":$CPU_BYTES_PER_RANK,\"lazy_offload\":$LAZY_OFFLOAD}}")
else
    OFFLOAD_LABEL=none
fi

EP_ARGS=()
if [ "${EP_SIZE:-1}" -gt 1 ]; then EP_ARGS=(--enable-expert-parallel); fi

echo "[cfg] conc=$CONC dcp=$DCP_SIZE gmu=$GPU_MEM_UTIL mns=$MAX_NUM_SEQS ladder=1..$LADDER spec_rows=$SPEC_ROWS chunk=$MAX_BATCHED_TOKENS cudagraph=$CUDAGRAPH_MODE offload=$OFFLOAD_LABEL"

# -----------------------------------------------------------------------------
# vllm-project/vllm#57085 -- ROCM_AITER_MLA non-causal DSpark draft under DCP
# -----------------------------------------------------------------------------
# Unmerged as of nightly 3df4ae15 (2026-09-21); rocm_aiter_mla.py is functionally
# identical to af1c0149 there, only a docstring reformat apart. Without it the
# config validator rejects the drafter at startup with
#   "non-causal MLA attention with DCP not supported"
# because supports_non_causal_multi_token_dcp is set by flashinfer_mla and
# tokenspeed_mla only, both of which gate on CUDA capability.major == 10.
# Applied by exact string match, not line offsets, so the upstream blank-line
# shift at 712 does not matter. Idempotent; hard-fails rather than running an
# unpatched engine and attributing the result to the patch.
apply_pr57085() {
    [ "${APPLY_PR57085:-1}" = "1" ] || { echo "[pr57085] disabled"; return 0; }
    export PR57085_FULL="${PR57085_FULL:-1}"
    python3 - <<'PYPATCH'
import os, sys, vllm
p = os.path.join(os.path.dirname(vllm.__file__),
                 "v1", "attention", "backends", "mla", "rocm_aiter_mla.py")
s = open(p).read()
if "supports_non_causal_multi_token_dcp" in s:
    print("[pr57085] already present, nothing to do"); sys.exit(0)
FULL = os.environ.get("PR57085_FULL", "0") == "1"
# Hunk 1 advertises the capability. Hunks 2-4 additionally REROUTE the non-causal
# block off aiter's segmented DCP verify onto a per-row LSE cross-rank merge --
# that is the path whose _ALLGATHER_BASE hung the 601 s watchdog at TP8/DCP8/k=3.
# Flag-only keeps the block on segmented DCP verify (the fused aiter route).
subs = [
 ("    supports_non_causal_multi_token_decode: ClassVar[bool] = True\n",
  "    supports_non_causal_multi_token_decode: ClassVar[bool] = True\n"
  "    supports_non_causal_multi_token_dcp: ClassVar[bool] = True\n"),
]
if FULL:
    subs += [
 ("            self._supports_segmented_dcp_verify and max_qo_len > 1\n",
  "            self._supports_segmented_dcp_verify and max_qo_len > 1 and causal\n"),
 ("        if self.dcp_world_size > 1 and int(decode.max_qo_len) > 1:\n",
  "        if (\n            attn_metadata.causal\n            and self.dcp_world_size > 1\n"
  "            and int(decode.max_qo_len) > 1\n        ):\n"),
 ("                decode.max_qo_len,\n                sm_scale=self.scale,\n                return_lse=True,\n",
  "                decode.max_qo_len,\n                sm_scale=self.scale,\n                return_lse=True,\n"
  "                causal=attn_metadata.causal,\n"),
    ]
for old, new in subs:
    if s.count(old) != 1:
        print(f"[pr57085] FAILED: anchor count {s.count(old)} != 1 for {old[:60]!r}")
        sys.exit(1)
    s = s.replace(old, new)
open(p, "w").write(s)
import py_compile; py_compile.compile(p, doraise=True)
print(f"[pr57085] applied {len(subs)} hunk(s) (mode={'full' if FULL else 'flag-only'}) and byte-compiled OK")
PYPATCH
}
apply_pr57085 || { echo "[pr57085] patch failed, refusing to run" >&2; exit 1; }

# -----------------------------------------------------------------------------
# vllm-project/vllm#54546 -- Triton MLA non-causal multi-token DCP
# -----------------------------------------------------------------------------
# The aiter route (#57085) boots but deadlocks: a _ALLGATHER_BASE hung the full
# 601 s watchdog at TP8/DCP8/k=3 and killed the engine mid-warmup. Triton takes a
# different route -- per the PR, "forward_mqa flattens the block to one decode row
# per query token and every row sees the same committed prefix, so a rank-local
# seq_len is the whole story" -- so there is no per-row cross-rank LSE merge to
# hang on. The author scopes the capability to ROCm as the validated platform.
#
# Only the flag hunk is applied. Upstream refactored triton_mla.py after the PR
# was written: the _init_reorder_batch_threshold(1, supports_spec_as_decode=True)
# call the PR edits no longer exists, replaced by
# supports_draft_decode_metadata_update = self.dcp_world_size == 1.
apply_pr54546() {
    [ "${APPLY_PR54546:-0}" = "1" ] || { echo "[pr54546] disabled"; return 0; }
    python3 - <<'PYPATCH'
import os, sys, vllm
p = os.path.join(os.path.dirname(vllm.__file__),
                 "v1", "attention", "backends", "mla", "triton_mla.py")
s = open(p).read()
if "supports_non_causal_multi_token_dcp" in s:
    print("[pr54546] already present, nothing to do"); sys.exit(0)
old = "    supports_non_causal_multi_token_decode: ClassVar[bool] = True\n"
new = (old +
       "    supports_non_causal_multi_token_dcp: ClassVar[bool] = "
       "current_platform.is_rocm()\n")
if s.count(old) != 1:
    print(f"[pr54546] FAILED: anchor count {s.count(old)} != 1"); sys.exit(1)
if "current_platform" not in s:
    print("[pr54546] FAILED: current_platform not imported"); sys.exit(1)
open(p, "w").write(s.replace(old, new))
import py_compile; py_compile.compile(p, doraise=True)
print("[pr54546] applied and byte-compiled OK")
PYPATCH
}
apply_pr54546 || { echo "[pr54546] patch failed, refusing to run" >&2; exit 1; }

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
    "${KDA_ARGS[@]}"
    "${COMPILATION_CONFIG_ARGS[@]}"
)

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
elif [ "${FIXED_LEN_HARNESS:-0}" = "1" ]; then
    # Fixed-length client instead of the trace replay. The agentic-coding
    # scenario emits ISL=OSL=0, and ${VAR:-default} does not substitute for
    # "0" -- only for unset/empty -- so guard on >0.
    ISL="${ISL:-8192}"; [ "$ISL" -gt 0 ] 2>/dev/null || ISL=8192
    OSL="${OSL:-1024}"; [ "$OSL" -gt 0 ] 2>/dev/null || OSL=1024
    RANDOM_RANGE_RATIO="${RANDOM_RANGE_RATIO:-0.8}"
    case "$RANDOM_RANGE_RATIO" in ""|0|0.0) RANDOM_RANGE_RATIO=0.8 ;; esac
    run_benchmark_serving \
        --model "$MODEL" \
        --port "$PORT" \
        --backend vllm \
        --input-len "$ISL" \
        --output-len "$OSL" \
        --random-range-ratio "$RANDOM_RANGE_RATIO" \
        --num-prompts "${NUM_PROMPTS:-200}" \
        --max-concurrency "$CONC" \
        --result-filename "${RESULT_FILENAME:-kimik3_fixedlen_conc${CONC}}" \
        --result-dir /outputs/ \
        --trust-remote-code \
        --use-chat-template
else
    build_replay_cmd "$RESULT_DIR"
    run_agentic_replay_and_write_outputs "$RESULT_DIR"
fi

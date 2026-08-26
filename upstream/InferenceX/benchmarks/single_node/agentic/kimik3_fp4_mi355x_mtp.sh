#!/usr/bin/env bash
set -euo pipefail
set -x
source "$(dirname "$0")/../../benchmark_lib.sh"
wait_for_amd_gpu_clean

export EVAL_FRAMEWORK="lm-eval"
EVAL_ONLY="${EVAL_ONLY:-false}"
EVAL_LIMIT="${EVAL_LIMIT:-200}"
export EVAL_ONLY EVAL_LIMIT
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

# DCP_SIZE is concurrency-dependent. DCP exists to enlarge the KV pool, which is
# what buys throughput at CONC 40-64; at CONC 1-4 the pool is irrelevant (T106
# ran at 95.5% prefix hit on a fraction of it) and DCP is pure added latency --
# an a2a plus a KV gather and a merge on every one of the 24 MLA layers, on top
# of the ~118 global barriers a decode step already issues. At batch 1 there is
# no work in flight to hide any of it, so TPOT is barrier-latency-bound.
# LOW_CONC_INTERACTIVE lets the matrix drive this without a second script.
LOW_CONC_INTERACTIVE="${LOW_CONC_INTERACTIVE:-auto}"
if [ "$LOW_CONC_INTERACTIVE" = "auto" ]; then
    if [ "${CONC:-52}" -le 4 ]; then LOW_CONC_INTERACTIVE=1; else LOW_CONC_INTERACTIVE=0; fi
fi
if [ "$LOW_CONC_INTERACTIVE" = "1" ]; then
    DCP_SIZE="${DCP_SIZE_OVERRIDE:-1}"
else
    DCP_SIZE="${DCP_SIZE_OVERRIDE:-8}"
fi
export DCP_SIZE

# Patches are pre-applied in kimi-k3-vllm:latest (kv-blockpool, pr51705
# e72380a5, pr51705-rejects), so the in-container patch step is not run.
# Re-enable the line below when running against a stock upstream image.
# bash "$(dirname "$0")/apply_kimi_k3_patches.sh" || true


export VLLM_ROCM_AITER_MLA_ASM_PADDING=asm
export VLLM_ROCM_USE_AITER=1
export SAFETENSORS_FAST_GPU=1
export VLLM_ROCM_USE_AITER_MOE_SITUV2_A8W4=1
export AITER_BF16_FP8_MOE_BOUND=0
export VLLM_USE_BREAKABLE_CUDAGRAPH=0
export GPU_ARCHS=gfx950
export VLLM_ROCM_USE_AITER_MOE=1
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

# KV offload. TOTAL_CPU_DRAM_GB is the aggregate host-DRAM budget the matrix
# generator derives from dram-utilization and the runner's available CPU DRAM;
# per the agentic README it must be consumed as given, never replaced with a
# model-specific constant. Worth 3.3x on the non-DCP path (T92 vs T64), so the
# absence of this block is not a neutral simplification.
OFFLOAD_ARGS=()
if agentic_kv_offload_enabled; then
    case "${KV_OFFLOAD_BACKEND:-}" in
      vllm-simple)
        require_agentic_kv_offload_backend "$KV_OFFLOAD_BACKEND"
        CPU_BYTES_PER_RANK=$(( TOTAL_CPU_DRAM_GB * 1000 * 1000 * 1000 / TOTAL_RANKS ))
        # Identical prefixes must hash to identical block keys across ranks.
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
# Expert parallelism. EP_SIZE is validated at the top of this script; without
# this block it was validated and then ignored, so `ep: 8` in the matrix did
# nothing. That is how EP stayed silently unreachable for 55 trials before T57.
EP_ARGS=()
if [ "${EP_SIZE:-1}" -gt 1 ]; then
    EP_ARGS=(--enable-expert-parallel)
    echo "EP: expert parallelism ON (EP_SIZE=$EP_SIZE)"
fi

CP_ARGS=(
    --decode-context-parallel-size "$DCP_SIZE"
    --dcp-comm-backend a2a
    --attention-backend ROCM_AITER_MLA
    --cp-kv-cache-interleave-size 1
)
export VLLM_USE_DIRECT_DCP_A2A=0
export VLLM_USE_DIRECT_DCP_Q_GATHER=0
export VLLM_USE_DIRECT_DCP_KV_GATHER=0
export VLLM_ALLOW_DCP_FULL_CUDAGRAPH=1
export VLLM_DCP_Q_REPLICATE=1
export VLLM_ROCM_USE_AITER_MLA=1
export AITER_DISABLE_FMHA_OPUS=1
export VLLM_ROCM_USE_AITER_MLA=1
export AITER_DISABLE_FMHA_OPUS=1

# Speculative decoding (MTP), wired from the matrix's spec-decoding field.
# SPEC_ARGS was previously built empty and unconditionally, so `spec-decoding:
# mtp` in the matrix was validated and then ignored -- the same orphan class as
# EP_ARGS (55 trials) and the KV-offload block (cost 3.3x). The orphan-check at
# the bottom catches unused arrays, not arrays that are used but always empty.
SPEC_NUM_TOKENS="${SPEC_NUM_TOKENS:-2}"
SYNTHETIC_ACCEPT_LEN="${SYNTHETIC_ACCEPT_LEN:-2.51}"
# The harness does NOT export SPEC_DECODING -- T121 proved it: the container had
# CONC, TP, EP_SIZE, KV_OFFLOADING and no SPEC_* at all, so gating on
# $SPEC_DECODING left speculative_config=None while `spec-decoding: mtp` sat in
# the matrix. The field does reach us, but only inside RESULT_FILENAME
# (kimik3_tp8_conc1_kvnone_spec-mtp_...), which is what we key off here. Keep the
# SPEC_DECODING check too, in case a future harness starts exporting it.
SPEC_ENABLE="${SPEC_DECODING:-}"
case "${RESULT_FILENAME:-}" in *spec-mtp*) SPEC_ENABLE=mtp;; esac
SPEC_ARGS=()
if [ "$SPEC_ENABLE" = "mtp" ]; then
    SPEC_ARGS=(
        --speculative-config
        "{\"model\":\"Inferact/Kimi-K3-DSpark\",\"num_speculative_tokens\":$SPEC_NUM_TOKENS,\"method\":\"dspark\",\"attention_backend\":\"TRITON_MLA\",\"kv_cache_dtype\":\"auto\",\"draft_sample_method\":\"probabilistic\",\"rejection_sample_method\": \"synthetic\", \"synthetic_acceptance_length\": $SYNTHETIC_ACCEPT_LEN}"
    )
    echo "MTP: speculative decoding ON (k=$SPEC_NUM_TOKENS, synthetic accept=$SYNTHETIC_ACCEPT_LEN)"
fi

CHUNKED_PREFILL_ARGS=(--max-num-batched-tokens 8192)
ASYNC_SCHED_ARGS=(--no-async-scheduling)
MLA_PREFILL_ARGS=(--attention-config "{\"mla_prefill_backend\":\"ROCM_AITER_FA\"}")

# max_num_seqs needs branching headroom (~1.25x CONC): too tight queues the
# replay's branched sub-requests and raises TTFT (T106 at mns 4), too loose lets
# residents run away and abort (T98, mns 144 -> 91 residents -> TTFT 20.2 s).
if [ "$LOW_CONC_INTERACTIVE" = "1" ]; then
    MAX_NUM_SEQS="${MAX_NUM_SEQS_OVERRIDE:-8}"
else
    MAX_NUM_SEQS="${MAX_NUM_SEQS_OVERRIDE:-80}"
fi

# The capture ladder must be DENSE and must reach the largest batch the model
# can actually see. A sparse ladder pads decode batches up to the next captured
# size and the padded rows read out of bounds -- that was the DCP GPU page fault
# that cost ~20 trials. With MTP every scheduled sequence expands to (k+1) rows,
# so the ceiling is MAX_NUM_SEQS*(k+1); T99 used 240 for mns 80 at k=2.
SPEC_ROWS=1
if [ "${#SPEC_ARGS[@]}" -gt 0 ]; then SPEC_ROWS=$(( SPEC_NUM_TOKENS + 1 )); fi
MAX_CUDAGRAPH_CAPTURE_SIZE=$(( MAX_NUM_SEQS * SPEC_ROWS ))
CUDAGRAPH_CAPTURE_SIZES=$(seq -s, 1 "$MAX_CUDAGRAPH_CAPTURE_SIZE")
echo "graphs: dense ladder 1..$MAX_CUDAGRAPH_CAPTURE_SIZE (mns=$MAX_NUM_SEQS x $SPEC_ROWS rows), DCP=$DCP_SIZE"
CUDAGRAPH_MODE=FULL_AND_PIECEWISE
COMPILATION_CONFIG_ARGS=(--compilation-config "{\"mode\":3,\"cudagraph_mode\":\"$CUDAGRAPH_MODE\",\"max_cudagraph_capture_size\":$MAX_CUDAGRAPH_CAPTURE_SIZE,\"custom_ops\":[\"+fused_rms_norm_gated\"],\"cudagraph_capture_sizes\":[$CUDAGRAPH_CAPTURE_SIZES]}")

# ROCPROF=1 wraps the server in rocprofv3 kernel tracing. This replaces the
# torch profiler, which added a per-step annotate_profile hook and hung a worker
# in shm_broadcast at both concurrency 52 (T114) and 8 (T115). rocprofv3 traces
# at the HIP layer and never touches vLLM's step loop.
# Default OFF: T116 collected the trace we needed (2.3 GB, 6.36M dispatches) and
# tracing costs ~17% input throughput, so any run whose *number* we intend to
# quote must have this off. Set ROCPROF=1 explicitly to profile again.
ROCPROF="${ROCPROF:-0}"
ROCPROF_PREFIX=()
if [ "$ROCPROF" = "1" ]; then
    RP_DIR="/mnt/hf_hub_cache/kimi-profiles/rocprof_$(date -u +%Y%m%d-%H%M%S)_dcp${DCP_SIZE}_conc${CONC}"
    mkdir -p "$RP_DIR"
    ROCPROF_PREFIX=(rocprofv3 --kernel-trace --stats -f csv -d "$RP_DIR" -o k --)
    echo "[rocprof] tracing -> $RP_DIR"
fi

GPU_MEM_UTIL=0.9

# fastsafetensors stages a whole shard batch in device memory before scattering:
# 96 shards / 12 batches = 8 shards x ~15.7 GB = ~117 GiB on top of the weights
# already resident. T109 survives that by a hair (weights alone are 192.56 GiB of
# the 268 GiB card, leaving ~75 GiB), but T117 and T118 both died at shard 11/12
# with ~270 MB free -- and T118 proved it is not the MoE kernel, since it OOM'd
# identically after the backend was moved back to AITER_MXFP4_BF16. 'auto' stages
# through host memory instead, trading startup time for load-time headroom. It
# does not affect steady-state throughput, so T103 stays comparable.
LOAD_FORMAT="${LOAD_FORMAT:-auto}"

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

"${ROCPROF_PREFIX[@]}" "${VLLM_CMD[@]}" > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo "Server PID: $SERVER_PID"

wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

python3 - <<'PINPY' > /tmp/pinmap.txt 2>/dev/null || true
import subprocess, re
def cpulist(n):
    return open(f"/sys/devices/system/node/node{n}/cpulist").read().strip()
def expand(s):
    out=[]
    for part in s.split(","):
        if "-" in part:
            a,b=part.split("-"); out += list(range(int(a),int(b)+1))
        else: out.append(int(part))
    return out
def compress(v):
    v=sorted(v); runs=[]; a=b=v[0]
    for x in v[1:]:
        if x==b+1: b=x
        else: runs.append((a,b)); a=b=x
    runs.append((a,b))
    return ",".join(f"{x}-{y}" if x!=y else str(x) for x,y in runs)
try:
    topo = subprocess.run(["rocm-smi","--showtoponuma"],capture_output=True,text=True).stdout
except Exception:
    topo = ""
gpu_node={}
for m in re.finditer(r"GPU\[(\d+)\].*?Numa Node:\s*(\d+)", topo):
    gpu_node[int(m.group(1))]=int(m.group(2))
if not gpu_node: raise SystemExit
bynode={}
for g,n in sorted(gpu_node.items()): bynode.setdefault(n,[]).append(g)
for n,gpus in bynode.items():
    cpus=expand(cpulist(n)); k=len(gpus)
    half=len(cpus)//2
    lo,hi=cpus[:half],cpus[half:]
    per_lo,per_hi=len(lo)//k,len(hi)//k
    for i,g in enumerate(gpus):
        sl = lo[i*per_lo:(i+1)*per_lo] + hi[i*per_hi:(i+1)*per_hi]
        print(f"{g} {compress(sl)}")
PINPY

if [ -s /tmp/pinmap.txt ]; then
    while read -r _g _cpus; do
        for _p in $(pgrep -f "VLLM::Worker_TP${_g}_" 2>/dev/null); do
            taskset -pc "$_cpus" "$_p" >/dev/null 2>&1 && echo "[pin-ranks] TP$_g pid=$_p -> cpus=$_cpus"
        done
    done < /tmp/pinmap.txt
fi

# EVAL_ONLY=true runs the GSM8K accuracy gate instead of the throughput replay.
# EVAL_LIMIT bounds the sample count; unset/full runs the whole dataset (the
# earlier attempt timed out needing ~15 h at 780 tok/s -- see ledger row 6).
if [ "${EVAL_ONLY:-false}" = "true" ]; then
    echo "[eval] GSM8K accuracy gate, EVAL_LIMIT=${EVAL_LIMIT:-full}"
    run_eval --port "$PORT"
else
    build_replay_cmd "$RESULT_DIR"
    run_agentic_replay_and_write_outputs "$RESULT_DIR"
fi

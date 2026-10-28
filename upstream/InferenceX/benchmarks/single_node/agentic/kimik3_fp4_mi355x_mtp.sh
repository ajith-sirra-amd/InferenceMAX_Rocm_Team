#!/usr/bin/env bash
set -eo pipefail
set -x
source "$(dirname "$0")/../../benchmark_lib.sh"
wait_for_amd_gpu_clean

check_env_vars MODEL TP CONC KV_OFFLOADING TOTAL_CPU_DRAM_GB RESULT_DIR DURATION EP_SIZE
check_env_vars DCP_SIZE EVAL_ONLY ATTENTION_PATH_PROBE_ONLY
case "$ATTENTION_PATH_PROBE_ONLY" in true|false) ;; *) echo "ATTENTION_PATH_PROBE_ONLY must be true or false" >&2; exit 1 ;; esac
if [ "$MODEL" != "moonshotai/Kimi-K3" ]; then
    echo "[NATIVE-QUANT-CONFIG] This experiment requires moonshotai/Kimi-K3" >&2
    exit 1
fi

if [ "$DURATION" = "600" ]; then
    export AIPERF_WARMUP_REQUESTS_PER_LANE=1
elif [ "$DURATION" = "3600" ]; then
    export AIPERF_WARMUP_REQUESTS_PER_LANE=10
else
    echo "[ATOM-FP8-CONFIG] Expected duration 600 (smoke) or 3600 (full)" >&2
    exit 1
fi

NATIVE_MODE=0
if [ -n "${NATIVE_VLLM_SHA:-}" ]; then
    check_env_vars INFERENCEX_NATIVE_REQUIRE_CAUSAL RESULT_FILENAME
    case "$INFERENCEX_NATIVE_REQUIRE_CAUSAL" in
        0|1) ;;
        *) echo "INFERENCEX_NATIVE_REQUIRE_CAUSAL must be 0 or 1" >&2; exit 1 ;;
    esac
    if [ "$ATTENTION_PATH_PROBE_ONLY" = "true" ]; then
        echo "Native validation uses kernel tests and full evaluation/replay" >&2
        exit 1
    fi
    NATIVE_MODE=1
    unset PYTHONPATH
    for task_atom_env in $(compgen -A variable ATOM_); do
        unset "$task_atom_env"
    done
    unset task_atom_env
    mkdir -p "$RESULT_DIR"
    native_work_dir="/outputs/.native-fp8-validation/$RESULT_FILENAME"
    python3 /outputs/utils/atom_attention/native_vllm_gpu_validate.py \
        --native-sha "$NATIVE_VLLM_SHA" --prepare-only \
        --work-dir "$native_work_dir" 2>&1 | tee "$RESULT_DIR/native_setup.log"
    source "$native_work_dir/evidence/native-env.sh"
    export INFERENCEX_NATIVE_PHASE_FILE="$RESULT_DIR/native_fp8_phase"
    export INFERENCEX_NATIVE_ACCURACY_STRESS=0
    if [ "$EVAL_ONLY" = "true" ]; then
        export INFERENCEX_NATIVE_ACCURACY_STRESS=1
        export VLLM_GPU_SYNC_CHECK=error
    fi
    if [ -e "$INFERENCEX_NATIVE_PHASE_FILE" ]; then
        echo "Stale native execution-phase marker exists" >&2
        exit 1
    fi
    echo "[NATIVE-FP8-CONFIG] sha=$NATIVE_VLLM_SHA stress=$INFERENCEX_NATIVE_ACCURACY_STRESS require_causal=$INFERENCEX_NATIVE_REQUIRE_CAUSAL"
else
    check_env_vars ATOM_USE_FLYDSL_FP8_PREFILL_ATTN ATOM_USE_FUSED_KV_FP8_OUTPUT
    check_env_vars ATOM_VALIDATE_FUSED_KV_FP8_OUTPUT ATOM_USE_FLYDSL_FP8_CAUSAL_ATTN
    # Load only this repository's vLLM extension; do not patch installed source.
    ATOM_ATTENTION_PLUGIN_DIR="$(cd "$(dirname "$0")/atom_attention" && pwd)"
    export PYTHONPATH="$ATOM_ATTENTION_PLUGIN_DIR${PYTHONPATH:+:$PYTHONPATH}"
    case "$ATOM_USE_FLYDSL_FP8_PREFILL_ATTN" in
        0|1) ;;
        *) echo "ATOM_USE_FLYDSL_FP8_PREFILL_ATTN must be 0 or 1" >&2; exit 1 ;;
    esac
    for flag_name in ATOM_USE_FUSED_KV_FP8_OUTPUT ATOM_VALIDATE_FUSED_KV_FP8_OUTPUT ATOM_USE_FLYDSL_FP8_CAUSAL_ATTN; do
        case "${!flag_name}" in
            0|1) ;;
            *) echo "$flag_name must be 0 or 1" >&2; exit 1 ;;
        esac
    done
    if [ "$ATOM_VALIDATE_FUSED_KV_FP8_OUTPUT" = "1" ]; then
        if [ "$ATTENTION_PATH_PROBE_ONLY" != "true" ] || [ "$ATOM_USE_FUSED_KV_FP8_OUTPUT" != "1" ] || [ "$ATOM_USE_FLYDSL_FP8_PREFILL_ATTN" != "1" ]; then
            echo "Direct KV numerical validation requires the FP8 path probe" >&2
            exit 1
        fi
    fi
    export ATOM_USE_FLYDSL_FP8_PREFILL_ATTN ATOM_USE_FUSED_KV_FP8_OUTPUT ATOM_VALIDATE_FUSED_KV_FP8_OUTPUT ATOM_USE_FLYDSL_FP8_CAUSAL_ATTN
    echo "[ATOM-FP8-CONFIG] framework=vllm fp8_prefill=$ATOM_USE_FLYDSL_FP8_PREFILL_ATTN model=$MODEL plugin=$ATOM_ATTENTION_PLUGIN_DIR"

fi

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
# Native single-node RCCL bootstrap; no runtime source patches or probes.
export NCCL_SOCKET_IFNAME="=lo"
export NCCL_SOCKET_FAMILY=AF_INET
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=VERSION
if [ "$ATTENTION_PATH_PROBE_ONLY" = "false" ]; then
    resolve_trace_source
    install_agentic_deps
fi
mkdir -p "$RESULT_DIR"
if [ "$ATTENTION_PATH_PROBE_ONLY" = "true" ]; then
    python3 /outputs/utils/atom_attention/preflight.py 2>&1 | tee "$RESULT_DIR/atom_attention_preflight.log"
    if [ "$ATOM_USE_FUSED_KV_FP8_OUTPUT" = "1" ]; then
        python3 /outputs/utils/atom_attention/fp8_weight_preflight.py 2>&1 | tee "$RESULT_DIR/atom_fp8_weight_preflight.log"
        python3 /outputs/utils/atom_attention/direct_kv_preflight.py 2>&1 | tee "$RESULT_DIR/atom_direct_kv_preflight.log"
    fi
fi

export AIPERF_EXPERIMENTAL_FAST=0

export VLLM_ROCM_USE_AITER=1
export VLLM_ROCM_USE_AITER_MOE_SITUV2=1
export AITER_BF16_FP8_MOE_BOUND=0
export AITER_FLYDSL_STAGE2_FP8=1
export VLLM_ROCM_QUICK_REDUCE_QUANTIZATION="${VLLM_ROCM_QUICK_REDUCE_QUANTIZATION:-INT4}"
export VLLM_ROCM_QUICK_REDUCE_MIN_SIZE_BYTES_MB=16
export SAFETENSORS_FAST_GPU=1
export HSA_NO_SCRATCH_RECLAIM=1
export VLLM_USE_BREAKABLE_CUDAGRAPH=0
export VLLM_USE_DIRECT_DCP_A2A=0
export VLLM_USE_DIRECT_DCP_Q_GATHER=0
export VLLM_USE_DIRECT_DCP_KV_GATHER=0
export VLLM_ENGINE_READY_TIMEOUT_S=7200
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1200
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
    48|56)
        DCP_SIZE="${DCP_SIZE:-8}"
        OFFLOAD_POLICY=harness
        MAX_BATCHED_TOKENS="${MAX_BATCHED_TOKENS:-8192}"
        MAX_NUM_SEQS="${MAX_NUM_SEQS:-$(( CONC * 2 ))}"
        SPEC_NUM_TOKENS="${SPEC_NUM_TOKENS:-${SPEC_K:-4}}"
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
        SPEC_ROWS=$(( SPEC_NUM_TOKENS + 1 ))
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

if [ "$NATIVE_MODE" = "1" ]; then
    if [ "$EVAL_ONLY" = "true" ]; then
        MAX_BATCHED_TOKENS=512
    fi
else
    # Accuracy changes scheduler chunk size and cached-path eligibility, not GSM8K.
    # This exercises the same FP8 kernels on short fixed-task prompts.
    export ATOM_FP8_ACCURACY_STRESS=0
    if [ "$EVAL_ONLY" = "true" ]; then
        if [ "$ATOM_USE_FLYDSL_FP8_PREFILL_ATTN" != "1" ] || [ "$ATOM_USE_FUSED_KV_FP8_OUTPUT" != "1" ]; then
            echo "This accuracy experiment requires native FP8 projection and FMHA" >&2
            exit 1
        fi
        export ATOM_FP8_ACCURACY_STRESS=1
        MAX_BATCHED_TOKENS=512
        export ATOM_FP8_ACCURACY_ACTIVE_FILE="$RESULT_DIR/atom_fp8_accuracy_active"
        if [ -e "$ATOM_FP8_ACCURACY_ACTIVE_FILE" ]; then
            echo "Stale evaluation-phase marker exists" >&2
            exit 1
        fi
    fi

fi

GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
CUDAGRAPH_MODE="${CUDAGRAPH_MODE:-FULL_AND_PIECEWISE}"

LADDER=$(( MAX_NUM_SEQS * SPEC_ROWS ))
CUDAGRAPH_CAPTURE_SIZES=$(seq -s, 2 2 "$LADDER")
COMPILATION_CONFIG_ARGS=(--compilation-config "{\"mode\":3,\"cudagraph_mode\":\"$CUDAGRAPH_MODE\",\"max_cudagraph_capture_size\":$LADDER,\"custom_ops\":[\"+fused_rms_norm_gated\"],\"cudagraph_capture_sizes\":[$CUDAGRAPH_CAPTURE_SIZES]}")

if [ "$NATIVE_MODE" = "1" ]; then
    CP_ARGS=(--attention-backend ROCM_AITER_MLA)
else
    CP_ARGS=(--attention-backend CUSTOM)
fi
if [ "$DCP_SIZE" -gt 1 ]; then
    CP_ARGS+=(--decode-context-parallel-size "$DCP_SIZE" --dcp-comm-backend a2a)
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

if [ "$NATIVE_MODE" = "1" ]; then
    ATTENTION_CONFIG_JSON='{"mla_prefill_backend":"ROCM_AITER_FA"}'
else
    ATTENTION_CONFIG_JSON='{"mla_prefill_backend":"CUSTOM"}'
fi

SERVER_MAX_MODEL_LEN=1048576
if [ "$EVAL_ONLY" = "true" ]; then
    SERVER_MAX_MODEL_LEN=32768
fi

VLLM_CMD=(
    vllm serve "$MODEL_PATH" --served-model-name "$MODEL"
    --host 0.0.0.0
    --port "$PORT"
    --trust-remote-code
    --moe-backend auto
    --tensor-parallel-size "$TP"
    --gpu-memory-utilization "$GPU_MEM_UTIL"
    --language-model-only
    --quantization-config '{"targets":{"*in_proj_qkvgfab":"fp8_per_channel","*.self_attn.o_proj":"fp8_per_channel","*.self_attn.fused_qkv_a_proj":"fp8_per_channel","*.self_attn.g_proj":"fp8_per_channel","*.self_attn.f_b_proj":"fp8_per_channel","*.self_attn.g_a_proj":"fp8_per_channel","*.self_attn.g_b_proj":"fp8_per_channel","*.self_attn.q_b_proj":"fp8_per_channel","*.self_attn.kv_b_proj":"fp8_per_channel","*.self_attn.q_proj":"fp8_per_channel","*.self_attn.kv_a_proj_with_mqa":"fp8_per_channel","*.block_sparse_moe.shared_experts.gate_up_proj":"fp8_per_channel","*.block_sparse_moe.shared_experts.down_proj":"fp8_per_channel","language_model.model.layers.0.mlp.gate_up_proj":"fp8_per_channel","language_model.model.layers.0.mlp.down_proj":"fp8_per_channel"}}'
    --max-num-seqs "$MAX_NUM_SEQS"
    --max-num-batched-tokens "$MAX_BATCHED_TOKENS"
    --max-model-len "$SERVER_MAX_MODEL_LEN"
    --kv-cache-dtype fp8
    --enable-auto-tool-choice
    --tool-call-parser kimi_k3
    --reasoning-parser kimi_k3
    --enable-prefix-caching
    --stream-interval 10
    --attention-config "$ATTENTION_CONFIG_JSON"
    "${OFFLOAD_ARGS[@]}"
    "${CP_ARGS[@]}"
    "${EP_ARGS[@]}"
    "${SPEC_ARGS[@]}"
    "${COMPILATION_CONFIG_ARGS[@]}"
)

if [ "$NATIVE_MODE" = "1" ]; then
    if [ "$EVAL_ONLY" = "true" ]; then
        VLLM_CMD+=(--enforce-eager)
    fi
elif [ "$ATOM_VALIDATE_FUSED_KV_FP8_OUTPUT" = "1" ] || [ "$ATOM_FP8_ACCURACY_STRESS" = "1" ]; then
    VLLM_CMD+=(--enforce-eager)
fi
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


if [ "$ATTENTION_PATH_PROBE_ONLY" = "true" ]; then
    if [ "$EVAL_ONLY" = "true" ]; then
        echo "Attention path probe cannot be combined with eval-only" >&2
        exit 1
    fi
    python3 "$ATOM_ATTENTION_PLUGIN_DIR/path_probe.py" \
        --model "$MODEL" --tokenizer "$MODEL_PATH" --port "$PORT" \
        --result-dir "$RESULT_DIR" --server-log "$SERVER_LOG"
elif [ "$EVAL_ONLY" = "true" ]; then
    check_env_vars RUN_EVAL IS_AGENTIC
    export EVAL_FRAMEWORK=lm-eval
    export EVAL_TASKS_DIR="$INFERENCEX_REPO_ROOT/infx/evals/gsm8k.yaml"
    export EVAL_MAX_MODEL_LEN="$SERVER_MAX_MODEL_LEN"
    export EVAL_CONCURRENT_REQUESTS="$CONC"
    export EVAL_RESULT_DIR="$RESULT_DIR/gsm8k"
    export MODEL_NAME="$MODEL"
    export OPENAI_API_KEY=EMPTY
    unset EVAL_SUITE EVAL_LIMIT
    mkdir -p "$EVAL_RESULT_DIR"
    echo "[GSM8K-EVAL] eval_only=$EVAL_ONLY task=$EVAL_TASKS_DIR concurrency=$EVAL_CONCURRENT_REQUESTS context=$EVAL_MAX_MODEL_LEN temperature=0 full_test_split=True real_verification=True"
    if [ "$NATIVE_MODE" = "1" ]; then
        printf '%s\n' eval > "$INFERENCEX_NATIVE_PHASE_FILE"
    else
        printf '%s\n' "GSM8K evaluation phase" > "$ATOM_FP8_ACCURACY_ACTIVE_FILE"
    fi
    (
        cd "$AGENTIC_OUTPUT_DIR"
        run_eval --port "$PORT"
    ) 2>&1 | tee "$RESULT_DIR/gsm8k_eval.log"
    causal_proof_args=()
    if [ "$NATIVE_MODE" = "1" ]; then
        if [ "$INFERENCEX_NATIVE_REQUIRE_CAUSAL" = "1" ]; then
            causal_proof_args=(--require-causal)
        fi
        "$VIRTUAL_ENV/bin/python" -m native_fp8_validation "${causal_proof_args[@]}" \
            --server-log "$SERVER_LOG" --workers "$TP" --native-sha "$NATIVE_VLLM_SHA" \
            --phase eval --output "$RESULT_DIR/gsm8k_native_fp8_execution_proof.json"
    else
        if [ "$ATOM_USE_FLYDSL_FP8_CAUSAL_ATTN" = "1" ]; then
            causal_proof_args=(--require-causal)
        fi
        python3 /outputs/utils/atom_attention/verify_accuracy_path.py "${causal_proof_args[@]}" \
            --server-log "$SERVER_LOG" --workers "$TP" \
            --output "$RESULT_DIR/gsm8k_fp8_execution_proof.json"
    fi
else
    build_replay_cmd "$RESULT_DIR"
    if [ "$NATIVE_MODE" = "1" ]; then
        printf '%s\n' replay > "$INFERENCEX_NATIVE_PHASE_FILE"
    fi
    run_agentic_replay_and_write_outputs "$RESULT_DIR"
    if [ "$NATIVE_MODE" = "1" ]; then
        causal_proof_args=()
        if [ "$INFERENCEX_NATIVE_REQUIRE_CAUSAL" = "1" ]; then
            causal_proof_args=(--require-causal)
        fi
        "$VIRTUAL_ENV/bin/python" -m native_fp8_validation "${causal_proof_args[@]}" \
            --server-log "$SERVER_LOG" --workers "$TP" --native-sha "$NATIVE_VLLM_SHA" \
            --phase replay --output "$RESULT_DIR/native_fp8_execution_proof.json"
    fi
fi

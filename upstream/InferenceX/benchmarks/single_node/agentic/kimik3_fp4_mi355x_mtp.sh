#!/usr/bin/env bash
set -euo pipefail
set -x

# Agentic trace replay benchmark for Kimi-K3 MXFP4 on MI355X / MI350X (gfx950)
# using vLLM.
#
# The server command is the AMD reference `vllm serve` for this model, i.e. the
# upstream vLLM recipe's amd block (vllm-project/recipes,
# https://recipes.vllm.ai/moonshotai/Kimi-K3) as run in practice:
#
#   --trust-remote-code --moe-backend auto --tensor-parallel-size 8
#   --load-format auto --gpu-memory-utilization 0.95 --mm-encoder-tp-mode data
#   --max-num-seqs 128 --max-num-batched-tokens 4096 --enable-auto-tool-choice
#   --tool-call-parser kimi_k3 --reasoning-parser kimi_k3
#
# with env VLLM_ROCM_USE_AITER=1 SAFETENSORS_FAST_GPU=1 AITER_SITUV2_A8W4=1
# AITER_BF16_FP8_MOE_BOUND=0 VLLM_USE_BREAKABLE_CUDAGRAPH=0.
#
# K3 is a 2.8T-parameter natively-multimodal MoE (896 routed experts, 16/token
# plus shared) on Kimi Delta Attention, gated MLA and Attention Residuals, with
# a 1M-token native context.
#
# TP=8 ONLY. The MXFP4 checkpoint is 1.561 TB decimal (1.420 TiB, 96
# safetensors), ~195 GB/GPU across 8 GPUs of the 288 GB part; TP=4 would need
# ~390 GB/GPU and cannot load. Upstream strategy_min_gpus agrees (single_node_tp
# and multi_node_tep both 8, DEP 16+), which is why there is no DP-attention arm.
#
# Required env vars:
#   MODEL, TP, CONC, KV_OFFLOADING, TOTAL_CPU_DRAM_GB, RESULT_DIR, DURATION,
#   EP_SIZE
#
# Perf-search knobs. Each defaults to the reference command's value, so an
# otherwise-unset run reproduces the reference exactly:
#   GPU_MEM_UTIL             0.95   (reference)
#   MAX_NUM_BATCHED_TOKENS   8192   (default)
#   AITER_A8W4               1      (reference; 0 = aiter a16w4 MoE path)
#   LANGUAGE_MODEL_ONLY      true   
#   KV_CACHE_DTYPE           fp8    (default for every arm; =auto for a bf16 A/B)
#   KV_BLOCK_SIZE            unset  (unset -> vLLM sizes the page; 128 under fp8)
#   MAX_MODEL_LEN            1M     
#   SPEC_DECODE              true   (this is the _mtp DSpark recipe; =false for a no-spec A/B)
#   SPEC_NUM_TOKENS          2      (DSpark draft length; validated by the _mtp config)

source "$(dirname "$0")/../../benchmark_lib.sh"

wait_for_amd_gpu_clean

# ACCURACY RUN. Set to "false" to go back to the throughput/agentic-replay arm.
# Note EVAL_ONLY=true also flips the speculative config from
# rejection_sample_method "synthetic" to "block" (see the SPEC_ARGS branch
# below), i.e. draft tokens are actually verified against the target, so this
# is the only arm whose generated text is valid. It therefore doubles as the
# correctness check for the triton_mla cudagraph patch.
EVAL_ONLY="false"
export EVAL_FRAMEWORK="lm-eval"

check_env_vars MODEL TP CONC KV_OFFLOADING TOTAL_CPU_DRAM_GB RESULT_DIR DURATION EP_SIZE

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "JOB $SLURM_JOB_ID running on ${SLURMD_NODENAME:-unknown}"
fi

if [ "$TP" -ne 8 ]; then
    echo "Error: Kimi-K3 MXFP4 is a 1.56 TB checkpoint and only fits at TP=8 on" >&2
    echo "       288 GB gfx950 parts (~195 GB/GPU). Got TP=$TP." >&2
    exit 1
fi

# ROCR/HIP visibility for vLLM 0.14+
if [ -n "${ROCR_VISIBLE_DEVICES:-}" ]; then
    export HIP_VISIBLE_DEVICES="$ROCR_VISIBLE_DEVICES"
fi

# `hf download` creates the target dir if missing and is itself idempotent. The
# 1.56 TB checkpoint is normally pre-staged, so these calls are a no-op there.
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

# ---- Resolve traces and install deps ----------------------------------------
resolve_trace_source
install_agentic_deps

# ---- In-container patches ----------------------------------------------------
# Three fixes, all confined to this container's site-packages, all idempotent
# and all self-disabling once the image ships them:
#   [1] aiter pybind11 internals mismatch  -> unblocks ROCM_AITER_FA prefill
#   [2] TritonMLA cudagraph support        -> FULL cudagraphs for DSpark (5.52x TPOT)
#   [3] KV block-pool negative-count clamp -> stops the mid-run engine crash
# Set SKIP_KIMI_PATCHES=1 to run stock.
# bash "$(dirname "$0")/apply_kimi_k3_patches.sh" || true

# ---- Reference env block ----------------------------------------------------
# Keep ALL of these. Commenting them out does not avoid the AITER FMHA crash:
# that crash is gated on VLLM_ROCM_USE_AITER alone (AiterFlashAttnPrefillBackend
# .is_available() consults only rocm_aiter_ops.is_enabled()), so disabling the
# others just loses the MoE kernels while keeping the failure.
# export VLLM_ROCM_AITER_MLA_ASM_PADDING=asm
# export VLLM_ROCM_USE_AITER=1
# export SAFETENSORS_FAST_GPU=1
# export VLLM_ROCM_USE_AITER_MOE_SITUV2_A8W4=1
# export AITER_BF16_FP8_MOE_BOUND=0
# # REQUIRED on ROCm per the upstream recipe: the build auto-enables this to 1.
# export VLLM_USE_BREAKABLE_CUDAGRAPH=0

# Workaround for MEC FW <177 RCCL memory reclaim issue (shared with the other
# gfx950 recipes in this tree).
mec_version=$(rocm-smi --showfw 2>/dev/null | grep MEC | head -n 1 | awk '{print $NF}')
if [[ "$mec_version" == "" || ${mec_version:-0} -lt 177 ]]; then
    export HSA_NO_SCRATCH_RECLAIM=1
fi

# 2.8T of weights off a shared/NFS mount takes far longer than the default.
export VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-7200}"

# Long agentic turns against a 1M context: keep the client from timing out
# mid-request while the server is prefill-bound.
export AIPERF_HTTP_TCP_USER_TIMEOUT=900000

# ---- Server config ----------------------------------------------------------
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

# ---- KV offload -------------------------------------------------------------
# TOTAL_CPU_DRAM_GB is the aggregate host-DRAM budget the matrix generator
# derives from dram-utilization and the runner's available-cpu-dram-mib, capped
# at the 3,095,781 MiB (3 TB decimal) agentic limit. Per
# benchmarks/single_node/agentic/README.md it must be consumed as given and
# never replaced with a model-specific constant.
OFFLOAD_ARGS=()

if agentic_kv_offload_enabled; then
case "${KV_OFFLOAD_BACKEND:-}" in
  vllm-simple)
    require_agentic_kv_offload_backend "$KV_OFFLOAD_BACKEND"
    CPU_BYTES_PER_RANK=$(( TOTAL_CPU_DRAM_GB * 1000 * 1000 * 1000 / TP ))
    # Identical prefixes must hash to identical block keys across ranks.
    export PYTHONHASHSEED=42
    SIMPLE_LAZY_OFFLOAD="${SIMPLE_LAZY_OFFLOAD:-false}"
    OFFLOAD_ARGS=(
        --kv-transfer-config
        "{\"kv_connector\":\"SimpleCPUOffloadConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"cpu_bytes_to_use_per_rank\":$CPU_BYTES_PER_RANK,\"lazy_offload\":$SIMPLE_LAZY_OFFLOAD}}"
    )
    echo "SimpleCPUOffloadConnector: ${CPU_BYTES_PER_RANK} B/rank x ${TP} ranks, lazy_offload=$SIMPLE_LAZY_OFFLOAD"
    ;;
esac
fi

# ---- LLM server  ------------------------------------------------------------


# ---- Parallelism ------------------------------------------------------------
EP_ARGS=()
if [ "$EP_SIZE" -gt 1 ]; then
    EP_ARGS=(--enable-expert-parallel)
fi

# ---- Speculative ------------------------------------------------------------
SPEC_NUM_TOKENS="${SPEC_NUM_TOKENS:-2}"
SYNTHETIC_ACCEPT_LEN=2.51

if [ "${EVAL_ONLY:-false}" = "true" ]; then
    SPEC_ARGS=(
        --speculative-config
        "{\"model\":\"Inferact/Kimi-K3-DSpark\",\"num_speculative_tokens\":$SPEC_NUM_TOKENS,\"method\":\"dspark\",\"attention_backend\":\"TRITON_MLA\",\"kv_cache_dtype\":\"auto\",\"draft_sample_method\":\"probabilistic\",\"rejection_sample_method\": \"block\"}"
    )
else
    SPEC_ARGS=(
        --speculative-config
        "{\"model\":\"Inferact/Kimi-K3-DSpark\",\"num_speculative_tokens\":$SPEC_NUM_TOKENS,\"method\":\"dspark\",\"attention_backend\":\"TRITON_MLA\",\"kv_cache_dtype\":\"auto\",\"draft_sample_method\":\"probabilistic\",\"rejection_sample_method\": \"block\", \"synthetic_acceptance_length\": $SYNTHETIC_ACCEPT_LEN}"
    )
fi

# ---- Chunked prefill sizing --------------------------------------------------
# Chunked prefill is ON by default and max_num_batched_tokens defaults to 16384
# (confirmed in the c12 server log; neither was previously set by this recipe).
# With ~99k-token mean inputs that is ~6 prefill chunks per request.
# 8192 doubles that to ~12 smaller chunks: finer interleaving with decode, at the
# cost of more per-chunk boundaries. Note vLLM PR #51862 (in the new image) exists
# specifically to remove a per-chunk KDA prefill stall, so more chunks may cut
# against that gain -- this run measures which effect dominates.
# B300 does not set this flag at all, so we are moving off their config here.
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
CHUNKED_PREFILL_ARGS=(--max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS")

# ---- Async scheduling / KV block-pool stability ------------------------------
# DSpark is the ONLY spec method exempted from vLLM's async-scheduling disable
# list (config/vllm.py:1181), so async_scheduling resolves True here. That gives
# max_concurrent_batches = pp_size + 1 = 2 (vllm.py:563-569), and with
# kv_role=kv_both (is_kv_consumer=True) the scheduler sets defer_block_free=True
# (sched/scheduler.py:155-157). Its own comment: "a step may still be writing a
# freed request's KV blocks. A consumer KV Connector can reallocate and fill
# those blocks via a load that isn't ordered against that write."
#
# That limbo state matches our crash signature exactly -- the engine dies with
#   block_pool.py:667  assert block.ref_cnt == 0
# i.e. a block sitting on the FREE list that is still referenced. Crash time
# scales inversely with concurrency: c10 survived 3612 s, c12 died at 487 s,
# c16 at 354 s. Note vLLM already disables async scheduling for ROCm DeepEP DBO
# because "that combination can corrupt" state.
#
# Setting max_concurrent_batches back to 1 makes defer_block_free unreachable.
# Cost: async scheduling exists to fill GPU-utilisation gaps, so expect to give
# some throughput back. Set ASYNC_SCHEDULING=1 to restore the default.
ASYNC_SCHED_ARGS=()
if [ "${ASYNC_SCHEDULING:-0}" != "1" ]; then
    ASYNC_SCHED_ARGS=(--no-async-scheduling)
fi

# ---- MLA prefill backend -----------------------------------------------------
# On ROCm the prefill priority is [ROCM_AITER_FA, FLASH_ATTN]. ROCM_AITER_FA
# JIT-builds module_fmha_fwd_bf16_opus at runtime; that module registers its own
# aiter_tensor_t, distinct from the one in the prebuilt module_aiter_core, so the
# first call dies with:
#   TypeError: fmha_fwd_bf16_opus_fwd(): incompatible function arguments
# during compile_or_warm_up_model -> _dummy_run, before the server binds.
# Pinning FLASH_ATTN keeps every AITER MoE kernel (and its throughput) while
# skipping only the broken FMHA prefill path.
# UPDATE: the AITER packaging issue is now fixed at source by
# apply_kimi_k3_patches.sh (run above), so ROCM_AITER_FA is usable again and
# is the default. Measured on 8x MI355X / Kimi-K3 MXFP4 TP8, cold prefill:
#   ~24k ctx  FLASH_ATTN 12,953 -> AITER 13,524 tok/s  (+4.4%)
#   ~93k ctx  FLASH_ATTN 11,174 -> AITER 13,423 tok/s  (+20.1%)
# This workload averages ~99k input tokens, so the ~93k figure is the relevant
# one. Set MLA_PREFILL_BACKEND=FLASH_ATTN to fall back if AITER regresses.
MLA_PREFILL_BACKEND="${MLA_PREFILL_BACKEND:-ROCM_AITER_FA}"
MLA_PREFILL_ARGS=()
if [ -n "$MLA_PREFILL_BACKEND" ]; then
    MLA_PREFILL_ARGS=(
        --attention-config
        "{\"mla_prefill_backend\":\"$MLA_PREFILL_BACKEND\"}"
    )
fi

# ---- HIP graph ------------------------------------------------------------
# max_num_seqs scales with concurrency, matching what the B300 reference actually
# does (read out of its vllm_command.txt): conc 1/2/4/8/16 -> max_num_seqs
# 2/4/8/16/32, i.e. 2 x CONC. We had this hardcoded at 20 for every conc, which
# was tight at c16 and over-provisioned at c1 -- neither like-for-like.
#
# Headroom matters because in-flight sequences EXCEED nominal concurrency: the
# agentic harness branches, and c12 was measured at peak 14 running vs nominal
# 12 (1.17x). 2x covers that comfortably.
MAX_NUM_SEQS="${MAX_NUM_SEQS:-$(( CONC * 2 ))}"

# INVARIANT: with DSpark a decode step is num_seqs * (1 + num_speculative_tokens)
# = num_seqs * 3 tokens, so the capture ceiling must be >= 3 * max_num_seqs or
# large decode batches fall out of the captured graphs and attention runs eager
# every step (the get_mla_metadata_v1 host bubble, ~75 ms ITL).
# NOTE: 3 * CONC would be WRONG here -- at c12 that is 36, but the real peak was
# 14 seqs = 42 tokens, which would have escaped capture. Derive from
# max_num_seqs, not from CONC.
# The list is dense, so every 3*C is an exact captured size (no rounding up) and
# there is no need to hand-add sizes the way a sparse ladder would.
# Cost measured on this recipe: 1.07 GiB and ~54 s to capture, so headroom is cheap.
MAX_CUDAGRAPH_CAPTURE_SIZE="${MAX_CUDAGRAPH_CAPTURE_SIZE:-$(( MAX_NUM_SEQS * 3 ))}"
CUDAGRAPH_CAPTURE_SIZES="$(seq -s, 1 "$MAX_CUDAGRAPH_CAPTURE_SIZE")"
echo "cudagraph sizing: CONC=$CONC max_num_seqs=$MAX_NUM_SEQS capture<=$MAX_CUDAGRAPH_CAPTURE_SIZE (3x max_num_seqs)"
COMPILATION_CONFIG_ARGS=(--compilation-config "{\"mode\":3,\"cudagraph_mode\":\"FULL_AND_PIECEWISE\",\"max_cudagraph_capture_size\":$MAX_CUDAGRAPH_CAPTURE_SIZE,\"custom_ops\":[\"+fused_rms_norm_gated\"],\"cudagraph_capture_sizes\":[$CUDAGRAPH_CAPTURE_SIZES]}")

GPU_MEM_UTIL="0.9"

echo "Starting vllm server..."
export PYTHONNOUSERSITE=1
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS="${VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS:-1200}"

{ set +x; } 2>/dev/null
# ---- Unified-image mode (aigmkt/k3-unified-v2-from-cb810) --------------------
# Self-detecting: the unified image ships /opt/aiter-local with a merged tuned
# GEMM table. On the stock nightly that path does not exist and this whole block
# is skipped, so the working recipe is untouched.
#
# Everything below mirrors validate_live_graph_capture.sh from the image's own
# build context. Do NOT mix these with our in-container patches -- the image
# already carries the PR2585 bundle, and our ROCM_AITER_FA prefill pin is exactly
# what made the first smoke test die with
#   ValueError: Selected MLA prefill backend ROCM_AITER_FA is not valid ...
#               Reason: ['required dependencies not available']
# (the image ships a different Triton, so triton_kernels.matmul_ogs is absent).
UNIFIED_GEMM_CSV=/opt/aiter-local/aiter/configs/merged_bf16_tuned_gemm.csv
if [ -f "$UNIFIED_GEMM_CSV" ]; then
    echo "=== unified image detected: applying its documented runtime config ==="
    export GPU_ARCHS=gfx950
    export VLLM_ROCM_USE_AITER=1
    export VLLM_ROCM_USE_AITER_MOE=1
    export AITER_SITUV2_A8W4=1
    export VLLM_ROCM_USE_AITER_MOE_SITUV2_A8W4=1
    export AITER_BF16_FP8_MOE_BOUND=0
    export VLLM_USE_BREAKABLE_CUDAGRAPH=0
    export SAFETENSORS_FAST_GPU=1
    export VLLM_ROCM_AITER_MLA_ASM_PADDING=asm
    export AITER_CONFIG_GEMM_BF16="$UNIFIED_GEMM_CSV"
    export HSA_NO_SCRATCH_RECLAIM=1
    export VLLM_K3_KDA_SAFE_STAGES=1
    export VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1

    # apply_dspark_fp8asm.sh step 2/7 forces the DSpark draft causal so that
    # ROCM_AITER_MLA is a legal draft backend (otherwise vLLM rejects it with
    # "non-causal attention not supported"). Their script hardcodes
    # /dev/shm/hf-cache, which does not exist here, so it would silently no-op
    # and leave the draft non-causal. Search the real cache instead.
    python3 - <<'PYFORCE'
import glob, json, os, shutil
roots = [os.environ.get("HF_HUB_CACHE",""), os.environ.get("HF_HOME",""),
         "/mnt/hf_hub_cache", "/home/models", "/dev/shm/hf-cache", "/root/.cache/huggingface/hub"]
pats = []
for r in roots:
    if r:
        pats += [f"{r}/models--Inferact--Kimi-K3-DSpark/snapshots/*/config.json",
                 f"{r}/**/models--Inferact--Kimi-K3-DSpark/snapshots/*/config.json"]
hits = []
for p in pats:
    hits += glob.glob(p, recursive=True)
if not hits:
    print("  !! DSpark draft config NOT found -- draft stays non-causal; "
          "ROCM_AITER_MLA will be rejected for the draft")
for f in sorted(set(hits)):
    c = json.load(open(f))
    d = c.setdefault("dflash_config", {})
    if d.get("causal") is True:
        print("  draft already causal:", f)
    else:
        if not os.path.exists(f + ".orig.bak"):
            shutil.copy2(f, f + ".orig.bak")
        d["causal"] = True
        json.dump(c, open(f, "w"), indent=2)
        print("  forced causal:", f)
PYFORCE

    # Their serve flags. ROCM_AITER_MLA everywhere; no separate prefill pin.
    MLA_PREFILL_ARGS=()
    ASYNC_SCHED_ARGS=(--async-scheduling)
    CHUNKED_PREFILL_ARGS=()
    UNIFIED_ARGS=(
        --attention-backend ROCM_AITER_MLA
        --distributed-executor-backend mp
        --enable-prompt-tokens-details
        --no-disable-hybrid-kv-cache-manager
        --disable-uvicorn-access-log
    )
    UNIFIED_MOE_BACKEND=aiter
    UNIFIED_LOAD_FORMAT=auto
    SPEC_ARGS=(
        --speculative-config
        "{\"method\":\"dspark\",\"model\":\"Inferact/Kimi-K3-DSpark\",\"num_speculative_tokens\":2,\"attention_backend\":\"ROCM_AITER_MLA\",\"draft_sample_method\":\"probabilistic\",\"rejection_sample_method\":\"synthetic\",\"synthetic_acceptance_length\":2.51}"
    )
    # Their capture ladder (sparse, to 384) rather than our dense 1..N.
    CUDAGRAPH_CAPTURE_SIZES="1,2,4,8,12,16,24,32,36,40,48,56,64,72,80,88,96,104,112,120,128,136,144,152,160,168,176,184,192,200,208,216,224,232,240,248,256,272,288,304,320,336,352,368,384"
    COMPILATION_CONFIG_ARGS=(--compilation-config "{\"mode\":3,\"cudagraph_mode\":\"FULL_AND_PIECEWISE\",\"custom_ops\":[\"+fused_rms_norm_gated\"],\"cudagraph_capture_sizes\":[$CUDAGRAPH_CAPTURE_SIZES]}")
else
    UNIFIED_ARGS=()
    UNIFIED_MOE_BACKEND=auto
    UNIFIED_LOAD_FORMAT=fastsafetensors
fi

VLLM_CMD=(
    vllm serve "$MODEL_PATH" --served-model-name "$MODEL"
    --host 0.0.0.0
    --port "$PORT"
    --trust-remote-code
    --moe-backend "$UNIFIED_MOE_BACKEND"
    --tensor-parallel-size "$TP"
    --load-format "$UNIFIED_LOAD_FORMAT"
    --gpu-memory-utilization "$GPU_MEM_UTIL"
    --language-model-only
    --max-num-seqs "$MAX_NUM_SEQS"
    --enable-auto-tool-choice
    --tool-call-parser kimi_k3
    --reasoning-parser kimi_k3
    --max-model-len 1048576
    --enable-prefix-caching
    --kv-cache-dtype "fp8"
    "${CHUNKED_PREFILL_ARGS[@]}"
    "${UNIFIED_ARGS[@]}"
    "${ASYNC_SCHED_ARGS[@]}"
    "${MLA_PREFILL_ARGS[@]}"
    "${COMPILATION_CONFIG_ARGS[@]}"
    "${SPEC_ARGS[@]}"
    "${OFFLOAD_ARGS[@]}"
)
printf '%q ' "${VLLM_CMD[@]}" | tee "$RESULT_DIR/vllm_command.txt"
printf '\n' | tee -a "$RESULT_DIR/vllm_command.txt"
"${VLLM_CMD[@]}" > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo "Server PID: $SERVER_PID"

wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

run_benchmark_serving \
    --model "$MODEL" \
    --port "$PORT" \
    --backend vllm \
    --input-len 1024 \
    --output-len 100 \
    --random-range-ratio 1.0 \
    --num-prompts "$((CONC * 10))" \
    --max-concurrency "$CONC" \
    --result-filename "${RESULT_DIR}/RESULT.txt" \
    --result-dir "$RESULT_DIR" \
    --trust-remote-code

# if [ "${EVAL_ONLY}" = "true" ]; then
#     run_eval --port "$PORT"
# else
#     build_replay_cmd "$RESULT_DIR"
#     run_agentic_replay_and_write_outputs "$RESULT_DIR"
# fi


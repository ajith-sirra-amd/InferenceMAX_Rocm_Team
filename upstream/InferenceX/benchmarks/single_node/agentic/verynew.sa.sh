#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-moonshotai/Kimi-K3}"
MODEL_PATH="${MODEL_PATH:-/mnt/hf_hub_cache/Kimi-K3}"
PORT="${PORT:-8888}"
CONC="${CONC:-72}"

TP=8
DCP_SIZE=8
MAX_NUM_SEQS=96
MAX_BATCHED_TOKENS=16384
GPU_MEM_UTIL=0.9
KV_CACHE_DTYPE=fp8
LOAD_FORMAT=fastsafetensors
MAX_MODEL_LEN=1048576
CPU_BYTES_PER_RANK=243625000000

CUDAGRAPH_MODE=FULL_DECODE_ONLY
MAX_CUDAGRAPH_CAPTURE_SIZE=$MAX_NUM_SEQS
CUDAGRAPH_CAPTURE_SIZES=$(seq -s, 1 "$MAX_CUDAGRAPH_CAPTURE_SIZE")

export VLLM_ROCM_USE_AITER=1
export VLLM_ROCM_USE_AITER_MLA=1
export VLLM_ROCM_USE_AITER_MOE=1
export VLLM_ROCM_USE_AITER_MOE_SITUV2_A8W4=1
export VLLM_ROCM_AITER_MLA_ASM_PADDING=asm
export VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=NONE
export VLLM_USE_BREAKABLE_CUDAGRAPH=0
export VLLM_DCP_Q_REPLICATE=0
export VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1
export VLLM_K3_KDA_SAFE_STAGES=1
export VLLM_ENGINE_READY_TIMEOUT_S=7200
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3600
export AITER_SITUV2_A8W4=1
export AITER_DISABLE_FMHA_OPUS=1
export AITER_QUICK_REDUCE_QUANTIZATION=NONE
export AITER_BF16_FP8_MOE_BOUND=0
export HSA_NO_SCRATCH_RECLAIM=1
export SAFETENSORS_FAST_GPU=1
export GPU_ARCHS=gfx950

vllm serve "$MODEL_PATH" \
    --served-model-name "$MODEL" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --trust-remote-code \
    --language-model-only \
    --moe-backend auto \
    --tensor-parallel-size "$TP" \
    --decode-context-parallel-size "$DCP_SIZE" \
    --dcp-comm-backend a2a \
    --cp-kv-cache-interleave-size 1 \
    --attention-backend ROCM_AITER_MLA \
    --attention-config '{"mla_prefill_backend":"ROCM_AITER_FA"}' \
    --load-format "$LOAD_FORMAT" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --max-num-batched-tokens "$MAX_BATCHED_TOKENS" \
    --max-model-len "$MAX_MODEL_LEN" \
    --kv-cache-dtype "$KV_CACHE_DTYPE" \
    --enable-prefix-caching \
    --enable-prompt-tokens-details \
    --enable-auto-tool-choice \
    --tool-call-parser kimi_k3 \
    --reasoning-parser kimi_k3 \
    --no-async-scheduling \
    --kv-transfer-config "{\"kv_connector\":\"SimpleCPUOffloadConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"cpu_bytes_to_use_per_rank\":$CPU_BYTES_PER_RANK,\"lazy_offload\":false}}" \
    --compilation-config "{\"mode\":3,\"cudagraph_mode\":\"$CUDAGRAPH_MODE\",\"max_cudagraph_capture_size\":$MAX_CUDAGRAPH_CAPTURE_SIZE,\"custom_ops\":[\"+fused_rms_norm_gated\"],\"cudagraph_capture_sizes\":[$CUDAGRAPH_CAPTURE_SIZES]}"

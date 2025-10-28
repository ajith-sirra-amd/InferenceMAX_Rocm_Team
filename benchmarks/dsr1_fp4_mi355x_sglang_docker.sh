#!/usr/bin/env bash

# ========= Required Env Vars =========
# HF_TOKEN
# HF_HUB_CACHE
# MODEL
# PORT
# TP
# CONC
# MAX_MODEL_LEN

# Reference
# https://rocm.docs.amd.com/en/docs-7.0-docker/benchmark-docker/inference-sglang-deepseek-r1-fp8.html

export SGLANG_USE_AITER=1
export RCCL_MSCCL_ENABLE=0
export SGLANG_INT4_WEIGHT=0
export SGLANG_MOE_PADDING=1
export SGLANG_USE_ROCM700A=1
export SGLANG_SET_CPU_AFFINITY=1
export SGLANG_ROCM_FUSED_DECODE_MLA=1
export SGLANG_ROCM_DISABLE_LINEARQUANT=1

pip install -U xgrammar

python3 -m sglang.launch_server \
    --model-path $MODEL \
    --host=0.0.0.0 \
    --port $PORT \
    --tensor-parallel-size $TP \
    --trust-remote-code \
    --chunked-prefill-size 196608 \
    --mem-fraction-static 0.8 --disable-radix-cache \
    --num-continuous-decode-steps 4 \
    --max-prefill-tokens 196608 \
    --cuda-graph-max-bs 128
    

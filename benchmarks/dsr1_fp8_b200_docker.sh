#!/usr/bin/env bash

# ========= Required Env Vars =========
# HF_TOKEN
# HF_HUB_CACHE
# MODEL
# PORT
# TP
# CONC
# MAX_MODEL_LEN

export SGL_ENABLE_JIT_DEEPGEMM=0
export SGLANG_ENABLE_FLASHINFER_GEMM=1

set -x
python3 -m sglang.launch_server --model-path=$MODEL --host=0.0.0.0 --port=$PORT --trust-remote-code \
--tensor-parallel-size=$TP --data-parallel-size=1 \
--kv-cache-dtype=fp8_e4m3 --mem-fraction-static=0.82 \
--max-prefill-tokens=32768 --chunked-prefill-size=32768 --cuda-graph-max-bs=128 --max-running-requests=128 \
--disable-radix-cache --enable-flashinfer-trtllm-moe --attention-backend=trtllm_mla --stream-interval=1

#!/usr/bin/env bash

# ========= Required Env Vars =========
# HF_TOKEN
# HF_HUB_CACHE
# MODEL
# PORT
# TP
# CONC
# MAX_MODEL_LEN

set -x
python3 -m atom.entrypoints.openai_server \
    --model openai/gpt-oss-120b \
    --server-port $PORT \
    -tp $TP \
    --kv_cache_dtype fp8

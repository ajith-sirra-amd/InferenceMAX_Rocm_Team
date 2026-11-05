#!/usr/bin/env bash
set -euo pipefail
bash /repo/utils/atom_attention/pr64288_green_validation.sh
export PYTHONPATH=/work/vllm
export VLLM_ROCM_USE_AITER=1
export TRITON_CACHE_DIR=/tmp/pr64288-runtime-counts-triton
cd /work/vllm
/work/vllm/.venv/bin/python /repo/utils/atom_attention/pr64288_runtime_counts_diagnostic.py 2>&1 | tee /work/evidence/runtime-counts.log
git -c safe.directory=/work/vllm diff --exit-code -- vllm tests

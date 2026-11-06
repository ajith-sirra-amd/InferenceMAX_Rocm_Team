#!/usr/bin/env bash
set -euo pipefail
export UV_CACHE_DIR=/work/cache/uv XDG_CACHE_HOME=/work/cache
export HF_HOME=/work/cache/huggingface TRITON_CACHE_DIR=/work/cache/triton
export PYTHONPYCACHEPREFIX=/work/cache/pycache PYTHONUNBUFFERED=1
export VLLM_PLUGINS=""
mkdir -p /work/evidence
uv venv --system-site-packages /work/bootstrap
/work/bootstrap/bin/python /repo/utils/atom_attention/native_vllm_gpu_validate.py \
  --native-sha "$NATIVE_SHA" --work-dir /work --prepare-only \
  2>&1 | tee /work/evidence/setup.log
export PYTHONPATH=/work/vllm
export PATH="/work/vllm/.venv/bin:$PATH"
export VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_AITER_LINEAR=1
export VLLM_ROCM_USE_AITER_FLYDSL_FP8_PREFILL_ATTN=1
cd /work/vllm
.venv/bin/python - <<'PY'
import json, os
from pathlib import Path
record = json.loads(Path("/work/evidence/native-validation.json").read_text())
assert record["status"] == "prepared"
assert record["native_sha"] == os.environ["NATIVE_SHA"]
assert record["precompiled_base"] == os.environ["BASE_SHA"]
print("SETUP_PROVENANCE_VERIFIED", record["native_sha"], record["precompiled_base"], flush=True)
PY
.venv/bin/python /repo/utils/atom_attention/pr64288_mixed_context_diagnostic.py \
  --output /work/evidence/mixed 2>&1 | tee /work/evidence/mixed.log
git diff --exit-code -- vllm tests

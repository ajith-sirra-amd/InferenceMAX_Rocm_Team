#!/usr/bin/env bash
set -euo pipefail
case "${REVIEW_SCOPE:?}" in kernels|niah|serving) ;; *) exit 2 ;; esac
export REVIEW_EVIDENCE=/work/evidence
mkdir -p "$REVIEW_EVIDENCE"
export UV_CACHE_DIR=/work/cache/uv XDG_CACHE_HOME=/work/cache
export HF_HOME=/work/cache/huggingface TRITON_CACHE_DIR=/work/cache/triton
export PYTHONPYCACHEPREFIX=/work/cache/pycache PYTHONUNBUFFERED=1
export VLLM_PLUGINS=""
# This builds a fresh environment AND executes the candidate's GPU kernel tests.
python3 /repo/utils/atom_attention/native_vllm_gpu_validate.py \
  --native-sha "$NATIVE_SHA" --work-dir /work 2>&1 | tee "$REVIEW_EVIDENCE/setup-kernels.log"
export PYTHONPATH=/work/vllm
export PATH="/work/vllm/.venv/bin:$PATH"
export VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_AITER_LINEAR=1
export VLLM_ROCM_USE_AITER_FLYDSL_FP8_PREFILL_ATTN=1
cd /work/vllm
.venv/bin/python - <<'PY'
import json, os, pathlib, xml.etree.ElementTree as ET
root = pathlib.Path(os.environ["REVIEW_EVIDENCE"])
report = json.loads((root / "native-validation.json").read_text())
assert report["status"] == "passed"
assert report["native_sha"] == os.environ["NATIVE_SHA"]
assert report["precompiled_base"] == os.environ["BASE_SHA"]
cases = ET.parse(root / "native-kernel-tests.xml").getroot().findall(".//testcase")
assert len(cases) == 12, len(cases)
assert not any(case.find(tag) is not None for case in cases
               for tag in ("skipped", "failure", "error"))
print("Verified 12 executed kernel tests and pinned binary/source provenance")
PY
if [ "$REVIEW_SCOPE" = kernels ]; then
  .venv/bin/python /repo/utils/atom_attention/pr64288_native_cache_numerical.py \
    "$REVIEW_EVIDENCE/native-cache-numerical.json" 2>&1 | tee "$REVIEW_EVIDENCE/numerical.log"
  .venv/bin/python /repo/utils/atom_attention/pr64288_review_diagnostics.py \
    --mode asm --output "$REVIEW_EVIDENCE/asm" 2>&1 | tee "$REVIEW_EVIDENCE/asm.log"
else
  observer=/work/review-observer
  mkdir -p "$observer"
  cp /repo/utils/atom_attention/review_serving_observer.py "$observer/review_serving_observer.py"
  cp /repo/utils/atom_attention/review_serving_observer.toml "$observer/pyproject.toml"
  uv pip install --python .venv/bin/python --no-deps "$observer"
  if [ "$REVIEW_SCOPE" = niah ]; then
    .venv/bin/python /repo/utils/atom_attention/pr64288_niah_validation.py \
      2>&1 | tee "$REVIEW_EVIDENCE/niah-driver.log"
  else
    .venv/bin/python /repo/utils/atom_attention/pr64288_bench_client_preflight.py
    .venv/bin/python /repo/utils/atom_attention/pr64288_matched_serve.py \
      2>&1 | tee "$REVIEW_EVIDENCE/serving-driver.log"
  fi
fi
git -c safe.directory=/work/vllm diff --exit-code -- vllm tests
.venv/bin/python /repo/utils/atom_attention/pr64288_verify_evidence.py "$REVIEW_EVIDENCE" "$REVIEW_SCOPE"

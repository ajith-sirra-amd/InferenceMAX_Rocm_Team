#!/usr/bin/env bash
set -euo pipefail
export REVIEW_EVIDENCE=/work/evidence
mkdir -p "$REVIEW_EVIDENCE"
export UV_CACHE_DIR=/work/cache/uv XDG_CACHE_HOME=/work/cache
export HF_HOME=/work/cache/huggingface TRITON_CACHE_DIR=/work/cache/triton
export PYTHONPYCACHEPREFIX=/work/cache/pycache PYTHONUNBUFFERED=1 VLLM_PLUGINS=""
python3 /repo/utils/atom_attention/native_vllm_gpu_validate.py \
  --native-sha "$NATIVE_SHA" --work-dir /work 2>&1 | tee "$REVIEW_EVIDENCE/setup-kernels.log"
export PYTHONPATH=/work/vllm
export PATH="/work/vllm/.venv/bin:$PATH"
cd /work/vllm
.venv/bin/python - <<'PY'
import json, pathlib, os, xml.etree.ElementTree as ET
root=pathlib.Path(os.environ["REVIEW_EVIDENCE"])
report=json.loads((root/"native-validation.json").read_text())
assert report["status"] == "passed" and report["native_sha"] == os.environ["NATIVE_SHA"]
assert report["precompiled_base"] == os.environ["BASE_SHA"]
cases=ET.parse(root/"native-kernel-tests.xml").getroot().findall(".//testcase")
assert len(cases)==12
assert not any(c.find(t) is not None for c in cases for t in ("skipped","failure","error"))
PY
observer=/work/memory-observer
mkdir -p "$observer"
cp /repo/utils/atom_attention/review_serving_observer.py "$observer/"
cp /repo/utils/atom_attention/pr64288_memory_audit.py "$observer/"
cat > "$observer/pyproject.toml" <<'TOML'
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"
[project]
name = "pr64288-memory-observer"
version = "0.1.0"
[project.entry-points."vllm.general_plugins"]
pr64288_serving_observer = "review_serving_observer:register"
pr64288_memory_audit = "pr64288_memory_audit:register"
[tool.setuptools]
py-modules = ["review_serving_observer", "pr64288_memory_audit"]
TOML
uv pip install --python .venv/bin/python --no-deps "$observer"
.venv/bin/python /repo/utils/atom_attention/pr64288_memory_audit.py \
  2>&1 | tee "$REVIEW_EVIDENCE/memory-driver.log"
git -c safe.directory=/work/vllm diff --exit-code -- vllm tests
.venv/bin/python - <<'PY'
import json, pathlib, os
r=json.loads((pathlib.Path(os.environ["REVIEW_EVIDENCE"])/"memory-validation.json").read_text())
assert r["status"]=="passed" and r["candidate"]==os.environ["NATIVE_SHA"]
assert set(r["arms"])=={"upstream_base","candidate_on"}
assert all(a["status"]=="passed" for a in r["arms"].values())
print("MEMORY_AUDIT_PASS")
PY

"""Refresh precompiled binaries for a pinned base before native GPU validation."""
import hashlib
import importlib.util
from importlib.metadata import version
import json
import os
from pathlib import Path
import subprocess
import sys

import torch
import triton
from packaging.tags import sys_tags
from packaging.version import Version

root = Path("/work/vllm")
evidence = Path(os.environ["REVIEW_EVIDENCE"])
base = os.environ["BASE_SHA"]
candidate = os.environ["NATIVE_SHA"]
actual = subprocess.check_output(
    ["git", "-c", "safe.directory=/work/vllm", "rev-parse", "HEAD"], cwd=root, text=True
).strip()
assert actual == candidate
assert Version(torch.__version__).public == Version(version("torch")).public
assert Version(triton.__version__).public == Version(version("triton")).public
spec = importlib.util.spec_from_file_location(
    "native_prepare", "/repo/utils/atom_attention/native_vllm_gpu_validate.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
runtime = {"torch": version("torch"), "hip": torch.version.hip,
           "triton": version("triton"), "tags": list(map(str, sys_tags()))}
wheel, metadata = module.resolve_exact_rocm_wheel(base, runtime)
record = {"native_sha": candidate, "precompiled_base": base,
          "wheel": wheel, "wheel_metadata": metadata,
          "runtime": {k: v for k, v in runtime.items() if k != "tags"},
          "status": "running"}
try:
    env = dict(os.environ, VLLM_USE_PRECOMPILED="1",
               VLLM_PRECOMPILED_WHEEL_COMMIT=base,
               VLLM_PRECOMPILED_WHEEL_LOCATION=wheel,
               GIT_CONFIG_COUNT="1", GIT_CONFIG_KEY_0="safe.directory",
               GIT_CONFIG_VALUE_0=str(root))
    git_bin = evidence / "installer-bin"
    git_bin.mkdir()
    git_wrapper = git_bin / "git"
    git_wrapper.write_text('#!/bin/sh\nexec /usr/bin/git -c safe.directory=/work/vllm "$@"\n')
    git_wrapper.chmod(0o755)
    env["PATH"] = str(git_bin) + ":" + env["PATH"]
    command = ["uv", "pip", "install", "--python", str(root / ".venv/bin/python"),
               "--no-build-isolation", "--no-deps", "--force-reinstall", "--editable", "."]
    subprocess.run(command, cwd=root, env=env, check=True)
    subprocess.run(["git", "-c", "safe.directory=/work/vllm", "diff", "--exit-code"], cwd=root, check=True)
    record["binary_hashes"] = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in (root / "vllm").glob("*.so")}
    record["status"] = "passed"
except Exception as error:
    record.update(status="failed", error=str(error))
    raise
finally:
    (evidence / "native-base-binaries.json").write_text(json.dumps(record, indent=2))
    print("NATIVE_BASE_BINARIES", json.dumps(record), flush=True)

"""Validate a pinned native vLLM revision inside the ROCm diagnostic container."""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET


def run(command, *, cwd, env, capture=False):
    print("NATIVE_VALIDATION_COMMAND", json.dumps([str(x) for x in command]), flush=True)
    result = subprocess.run(
        [str(x) for x in command], cwd=cwd, env=env, text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.STDOUT if capture else None,
    )
    if capture:
        print(result.stdout, flush=True)
    result.check_returncode()
    return result.stdout if capture else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--native-sha", required=True)
    parser.add_argument("--require-causal", action="store_true")
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}", args.native_sha):
        parser.error("native-sha must be an exact 40-character commit SHA")
    root = Path("/work").resolve()
    source = root / "vllm"
    evidence = root / "evidence"
    evidence.mkdir(parents=True, exist_ok=True)
    if source.exists():
        raise RuntimeError("Use a fresh per-run validation directory")
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.pop("VLLM_PRECOMPILED_WHEEL_LOCATION", None)
    for key in list(env):
        if key.startswith("ATOM_"):
            env.pop(key)
    env.update({
        "UV_CACHE_DIR": str(root / "cache" / "uv"),
        "XDG_CACHE_HOME": str(root / "cache"),
        "HF_HOME": str(root / "cache" / "huggingface"),
        "PYTHONPYCACHEPREFIX": str(root / "cache" / "pycache"),
        "TMPDIR": str(root / "tmp"),
        "TRITON_CACHE_DIR": str(root / "cache" / "triton"),
        "VLLM_ROCM_USE_AITER": "1",
        "VLLM_ROCM_USE_AITER_LINEAR": "1",
        "VLLM_ROCM_USE_AITER_FLYDSL_FP8_PREFILL_ATTN": "1",
    })
    Path(env["TMPDIR"]).mkdir(parents=True, exist_ok=True)
    if not shutil.which("uv"):
        raise RuntimeError("The diagnostic container must provide uv")
    record = {"native_sha": args.native_sha, "require_causal": args.require_causal,
              "scope": "native kernel tests; model accuracy and serving sync checks are separate",
              "status": "running"}
    try:
        run(["git", "clone", "--filter=blob:none", "--no-checkout",
             "https://github.com/ajith-sirra-amd/vllm.git", source], cwd=root, env=env)
        run(["git", "fetch", "origin", args.native_sha], cwd=source, env=env)
        run(["git", "checkout", "--detach", args.native_sha], cwd=source, env=env)
        actual = run(["git", "rev-parse", "HEAD"], cwd=source, env=env, capture=True).strip()
        if actual != args.native_sha:
            raise RuntimeError("Native checkout does not match requested source")
        run(["git", "fetch", "https://github.com/vllm-project/vllm.git", "main"],
            cwd=source, env=env)
        base = run(["git", "merge-base", "HEAD", "FETCH_HEAD"], cwd=source,
                   env=env, capture=True).strip()
        changed = run(["git", "diff", "--name-only", base, "HEAD"], cwd=source,
                      env=env, capture=True).splitlines()
        if any(p.startswith(("csrc/", "cmake/", "rust/")) or p == "CMakeLists.txt"
               for p in changed):
            raise RuntimeError("Compiled sources differ; a full source build is required")
        record["precompiled_base"] = base
        env["VLLM_USE_PRECOMPILED"] = "1"
        env["VLLM_PRECOMPILED_WHEEL_COMMIT"] = base
        run(["uv", "venv", "--system-site-packages", "--python", sys.executable,
             source / ".venv"], cwd=source, env=env)
        python = source / ".venv" / "bin" / "python"
        run(["uv", "pip", "install", "--python", python, "setuptools>=77",
             "setuptools-scm>=8", "setuptools-rust", "wheel", "packaging",
             "pytest", "pytest-asyncio", "pytest-forked", "tblib"], cwd=source, env=env)
        run(["uv", "pip", "install", "--python", python, "--no-build-isolation",
             "--no-deps", "--editable", "."], cwd=source, env=env)
        provenance_code = (
            "import json,pathlib,torch,vllm,aiter; "
            "from vllm.platforms import current_platform; "
            "root=pathlib.Path.cwd().resolve(); "
            "assert pathlib.Path(vllm.__file__).resolve().is_relative_to(root); "
            "assert current_platform.is_rocm() and torch.cuda.is_available(); "
            "assert torch.cuda.device_count()==8; "
            "assert all('gfx950' in torch.cuda.get_device_properties(i).gcnArchName "
            "for i in range(8)); "
            "print(json.dumps({'torch':torch.__version__,'hip':torch.version.hip,"
            "'vllm_file':vllm.__file__,'aiter_file':aiter.__file__,"
            "'gpus':[torch.cuda.get_device_name(i) for i in range(8)]}))"
        )
        record["environment"] = run([python, "-c", provenance_code], cwd=source,
                                    env=env, capture=True)
        test_path = "tests/kernels/attention/test_rocm_aiter_mla_prefill_fusions.py"
        junit = evidence / "native-kernel-tests.xml"
        result = subprocess.run(
            [str(python), "-m", "pytest", "-v", "-s", test_path, "-k", "flydsl_fp8",
             "--junitxml=" + str(junit)], cwd=source, env=env,
        )
        record["pytest_exit_code"] = result.returncode
        if not junit.exists():
            raise RuntimeError("Pytest did not produce execution evidence")
        cases = ET.parse(junit).getroot().findall(".//testcase")
        record["test_names"] = [case.attrib.get("name", "") for case in cases]
        expected_tests = {
            node.name for node in ast.parse((source / test_path).read_text()).body
            if isinstance(node, ast.FunctionDef) and node.name.startswith("test_flydsl_fp8")
        }
        actual_tests = {name.split("[", 1)[0] for name in record["test_names"]}
        if not expected_tests or not expected_tests.issubset(actual_tests):
            raise RuntimeError("Some selected native GPU tests did not execute")
        if not cases or any(case.find(tag) is not None for case in cases
                            for tag in ("skipped", "failure", "error")):
            raise RuntimeError("Native GPU tests did not all execute and pass")
        if args.require_causal and not any("new_tokens_causal" in name
                                           for name in record["test_names"]):
            raise RuntimeError("Causal native test did not execute")
        result.check_returncode()
        run(["git", "diff", "--exit-code", "--", "vllm", "tests"], cwd=source, env=env)
        record["source_hashes"] = {
            p: hashlib.sha256((source / p).read_bytes()).hexdigest()
            for p in changed if p.startswith(("vllm/", "tests/")) and (source / p).is_file()
        }
        record["status"] = "passed"
    except Exception as exc:
        record.update(status="failed", error=str(exc))
        raise
    finally:
        (evidence / "native-validation.json").write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()

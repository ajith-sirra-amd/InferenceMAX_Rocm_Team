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


def resolve_exact_rocm_wheel(commit, runtime):
    """Require the requested build and matching Torch/Triton binary versions."""
    from urllib.parse import unquote, urljoin, urlsplit
    from urllib.request import urlopen

    from packaging.utils import parse_wheel_filename
    from packaging.version import Version

    hip = re.match(r"(\d+)\.(\d+)", runtime["hip"] or "")
    if hip is None:
        raise RuntimeError("Cannot identify the installed ROCm version")
    # HIP's component version can differ from the ROCm release (for example,
    # HIP 7.15 in the ROCm 10 image). The Torch wheel records the release.
    release = re.search(
        r"(?:^|\.)rocm(\d+)\.(\d+)(?:\.|$)",
        Version(runtime["torch"]).local or "",
    )
    if release is None:
        raise RuntimeError("Torch wheel does not identify its ROCm release")
    variant = "rocm" + "".join(release.groups())
    index = f"https://wheels.vllm.ai/rocm/{commit}/{variant}/"
    supported_tags = set(runtime["tags"])
    selected = {}
    for package in ("vllm", "torch", "triton"):
        metadata_url = index + package + "/metadata.json"
        print("NATIVE_WHEEL_METADATA", metadata_url, flush=True)
        with urlopen(metadata_url, timeout=30) as response:
            entries = json.load(response)
        compatible = []
        for entry in entries:
            if entry.get("package_name") != package:
                continue
            if entry.get("variant") != variant:
                continue
            tags = {
                f"{python}-{abi}-{platform}"
                for python in entry["python_tag"].split(".")
                for abi in entry["abi_tag"].split(".")
                for platform in entry["platform_tag"].split(".")
            }
            if tags & supported_tags:
                compatible.append(entry)
        if len(compatible) != 1:
            raise RuntimeError(
                f"Expected one compatible {package} wheel for {commit}/{variant}; "
                f"found {len(compatible)}"
            )
        entry = compatible[0]
        wheel_url = urljoin(index + package + "/", entry["path"])
        location = urlsplit(wheel_url)
        if (location.scheme != "https" or location.netloc != "wheels.vllm.ai"
                or commit not in location.path.split("/")):
            raise RuntimeError("Wheel metadata does not resolve to the requested commit")
        filename = unquote(location.path.rsplit("/", 1)[-1])
        name, version, _, wheel_tags = parse_wheel_filename(filename)
        metadata_version = Version(entry["version"])
        if name != package or not supported_tags.intersection(map(str, wheel_tags)):
            raise RuntimeError(f"Selected {package} wheel has incompatible name or tags")
        if version.public != metadata_version.public:
            raise RuntimeError(f"Selected {package} wheel disagrees with metadata version")
        selected[package] = dict(
            entry, resolved_wheel_url=wheel_url, wheel_version=str(version),
        )
    for package in ("torch", "triton"):
        expected = selected[package]["wheel_version"]
        if Version(runtime[package]) != Version(expected):
            raise RuntimeError(
                f"Installed {package} {runtime[package]} differs from the "
                f"requested build's {expected}; use matching dependencies"
            )
    return selected["vllm"]["resolved_wheel_url"], selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--native-sha", required=True)
    parser.add_argument("--require-causal", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--work-dir", default="/work")
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}", args.native_sha):
        parser.error("native-sha must be an exact 40-character commit SHA")
    root = Path(args.work_dir).resolve()
    if root != Path("/work") and not root.is_relative_to(Path("/outputs/.native-fp8-validation")):
        parser.error("work-dir must be a repository-mounted validation directory")
    root.mkdir(parents=True, exist_ok=True)
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
        runtime_code = (
            "import json,torch,triton; from packaging.tags import sys_tags; "
            "from importlib.metadata import version; "
            "from packaging.version import Version; "
            "assert Version(torch.__version__).public == Version(version('torch')).public; "
            "assert Version(triton.__version__).public == Version(version('triton')).public; "
            "print(json.dumps({'torch':version('torch'),'hip':torch.version.hip,"
            "'triton':version('triton'),'module_versions':"
            "{'torch':torch.__version__,'triton':triton.__version__},"
            "'tags':[str(t) for t in sys_tags()]}))"
        )
        runtime_output = run([python, "-c", runtime_code], cwd=source,
                             env=env, capture=True)
        runtime = json.loads(runtime_output.strip().splitlines()[-1])
        record["binary_runtime"] = {k: v for k, v in runtime.items() if k != "tags"}
        wheel_url, wheel_metadata = resolve_exact_rocm_wheel(base, runtime)
        env["VLLM_PRECOMPILED_WHEEL_LOCATION"] = wheel_url
        record["precompiled_wheel"] = wheel_url
        record["precompiled_metadata"] = wheel_metadata
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
        source_hashes = {
            p: hashlib.sha256((source / p).read_bytes()).hexdigest()
            for p in changed if p.startswith(("vllm/", "tests/")) and (source / p).is_file()
        }
        record["source_hashes"] = source_hashes
        record["source_root"] = str(source)
        if args.prepare_only:
            observer = Path(__file__).resolve().parent / "native_validation_plugin"
            run(["uv", "pip", "install", "--python", python, "--no-deps",
                 "--editable", observer], cwd=source, env=env)
            import shlex
            exports = {
                name: env[name] for name in (
                    "UV_CACHE_DIR", "XDG_CACHE_HOME", "HF_HOME", "PYTHONPYCACHEPREFIX",
                    "TMPDIR", "TRITON_CACHE_DIR", "VLLM_ROCM_USE_AITER",
                    "VLLM_ROCM_USE_AITER_LINEAR",
                    "VLLM_ROCM_USE_AITER_FLYDSL_FP8_PREFILL_ATTN",
                )
            }
            exports.update({
                "VIRTUAL_ENV": str(source / ".venv"),
                "VLLM_PLUGINS": "native_fp8_validation",
                "INFERENCEX_NATIVE_VLLM_SHA": args.native_sha,
                "INFERENCEX_NATIVE_MANIFEST": str(evidence / "native-validation.json"),
            })
            lines = ["unset PYTHONPATH"]
            lines.extend(f"export {key}={shlex.quote(value)}"
                         for key, value in exports.items())
            lines.append("export PATH=" + shlex.quote(str(python.parent)) + ':"$PATH"')
            (evidence / "native-env.sh").write_text("\n".join(lines) + "\n")
            record.update(status="prepared", scope="native installation only; GPU tests not run")
            return
        # Diagnose the reference BLAS failure in fresh processes before FP8 tests.
        # These observations do not replace or relax any kernel acceptance check.
        probe_code = 'import json, os, sys, torch\nkind = sys.argv[1]\ndtype = getattr(torch, kind)\nrows = 31 if kind == "float32" else 386\ntorch.manual_seed(7)\nlhs = torch.randn(rows, 512, device="cuda", dtype=dtype)\nrhs = torch.randn(3072, 512, device="cuda", dtype=dtype)\nresult = {"dtype": kind, "rocblas_use_hipblaslt": os.getenv("ROCBLAS_USE_HIPBLASLT"),\n          "shape": [rows, 512, 3072], "scope": "fresh process; no FP8 kernels imported"}\ntry:\n    out = lhs @ rhs.t()\n    torch.cuda.synchronize()\n    reference = lhs.cpu().double() @ rhs.cpu().double().t()\n    error = out.cpu().double() - reference\n    result.update(status="completed", max_absolute_error=error.abs().max().item(),\n                  normalized_rms=(error.square().mean()/reference.square().mean()).sqrt().item())\nexcept RuntimeError as exc:\n    result.update(status="runtime_error", error=str(exc))\nprint(json.dumps(result))\n'
        record["reference_blas_diagnostics"] = []
        for setting in (None, "0"):
            for dtype in ("float32", "float64"):
                probe_env = env.copy()
                if setting is not None:
                    probe_env["ROCBLAS_USE_HIPBLASLT"] = setting
                probe = subprocess.run(
                    [str(python), "-c", probe_code, dtype], cwd=source, env=probe_env,
                    text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                )
                observation = {"dtype": dtype, "requested_rocblas_use_hipblaslt": setting,
                               "exit_code": probe.returncode}
                if probe.returncode == 0:
                    observation.update(json.loads(probe.stdout.strip().splitlines()[-1]))
                else:
                    observation.update(status="process_failed", stderr=probe.stderr)
                record["reference_blas_diagnostics"].append(observation)
                print("NATIVE_REFERENCE_BLAS_DIAGNOSTIC", json.dumps(observation), flush=True)
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

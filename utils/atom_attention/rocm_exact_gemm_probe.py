"""Isolate exact GEMM layout, virtualenv and cache environment without vLLM."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

CASES = {
    "bf16_nn_worker": ("bfloat16", 16, 16, 16, False),
    "bf16_nt_control": ("bfloat16", 16, 16, 16, True),
    "fp32_nt_reference": ("float32", 31, 3072, 512, True),
    "fp64_nt_reference": ("float64", 386, 3072, 512, True),
}
ENV_KEYS = ("LD_LIBRARY_PATH", "LD_PRELOAD", "ROCM_PATH", "HIP_PATH", "HSA_XNACK",
            "ROCBLAS_USE_HIPBLASLT", "HIPBLASLT_TENSILE_LIBPATH", "TMPDIR",
            "XDG_CACHE_HOME", "PYTHONPYCACHEPREFIX", "TRITON_CACHE_DIR",
            "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES")

def loaded_libraries():
    return sorted({line.split()[-1] for line in Path("/proc/self/maps").read_text().splitlines()
                   if any(x in line for x in ("libamdhip", "libhsa-runtime", "libhipblas",
                                              "librocblas", "libtorch_hip", "librocm_kpack"))
                   and "/" in line})

def child(case):
    import torch
    kind, rows, columns, inner, transposed = CASES[case]
    dtype = getattr(torch, kind)
    lhs = (torch.arange(rows * inner).reshape(rows, inner) % 7 - 3).to(dtype)
    rhs_storage = (torch.arange(columns * inner).reshape(columns, inner) % 7 - 3).to(dtype)
    rhs = rhs_storage.t() if transposed else lhs
    reference = lhs.double() @ rhs.double()
    report = {"case": case, "dtype": kind, "shape_mnk": [rows, columns, inner],
              "transpose_rhs": transposed, "rhs_stride": list(rhs.stride()),
              "python": sys.executable, "prefix": sys.prefix, "base_prefix": sys.base_prefix,
              "torch": torch.__version__, "environment": {k: os.getenv(k) for k in ENV_KEYS},
              "scope": "fresh process; torch only; no vLLM or FP8 imports"}
    try:
        report["gpu"] = str(torch.cuda.get_device_properties(0))
        a = lhs.cuda()
        b = rhs_storage.cuda().t() if transposed else a
        output = torch.mm(a, b)
        torch.cuda.synchronize()
        actual = output.cpu().double()
        report.update(status="passed" if torch.equal(actual, reference) else "mismatch",
                      max_absolute_error=(actual - reference).abs().max().item())
    except Exception as exc:
        report.update(status="runtime_error", error=str(exc), traceback=traceback.format_exc())
    report["loaded_libraries"] = loaded_libraries()
    report["rocm_python_modules"] = sorted(k for k in sys.modules if "rocm" in k.lower())
    print("RUNTIME_GEMM_RESULT " + json.dumps(report), flush=True)

def run_probe(python, env, case):
    try:
        p = subprocess.run([str(python), __file__, "--child", case], env=env,
                           capture_output=True, text=True, timeout=90)
        lines = [s.split("RUNTIME_GEMM_RESULT ", 1)[1] for s in p.stdout.splitlines()
                 if s.startswith("RUNTIME_GEMM_RESULT ")]
        result = json.loads(lines[-1]) if lines else {"case": case, "status": "process_failed"}
        result.update(exit_code=p.returncode, stdout=p.stdout[-16000:], stderr=p.stderr[-20000:])
        return result
    except Exception as exc:
        return {"case": case, "status": "probe_failed", "error": str(exc)}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--child", choices=CASES)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.child:
        child(args.child)
        return
    root = args.output.parent / "matrix-work"
    root.mkdir(parents=True, exist_ok=False)
    subprocess.run(["uv", "venv", "--system-site-packages", "--python", sys.executable,
                    str(root / "venv")], check=True)
    native_env = os.environ.copy()
    native_env.update({
        "UV_CACHE_DIR": str(root / "cache/uv"), "XDG_CACHE_HOME": str(root / "cache"),
        "HF_HOME": str(root / "cache/huggingface"),
        "PYTHONPYCACHEPREFIX": str(root / "cache/pycache"),
        "TMPDIR": str(root / "tmp"), "TRITON_CACHE_DIR": str(root / "cache/triton"),
        "VLLM_ROCM_USE_AITER": "1", "VLLM_ROCM_USE_AITER_LINEAR": "1",
        "VLLM_ROCM_USE_AITER_FLYDSL_FP8_PREFILL_ATTN": "1",
    })
    Path(native_env["TMPDIR"]).mkdir()
    stages = [
        ("system_default", sys.executable, os.environ.copy()),
        ("system_native_env", sys.executable, native_env),
        ("venv_default", root / "venv/bin/python", os.environ.copy()),
        ("venv_native_env", root / "venv/bin/python", native_env),
    ]
    report = {"scope": "runtime diagnosis only; no model or FP8 correctness claim", "stages": []}
    for name, python, env in stages:
        stage = {"name": name, "tests": []}
        for case in CASES:
            result = run_probe(python, env, case)
            stage["tests"].append(result)
            print(json.dumps({"stage": name, "case": case, "status": result["status"],
                              "error": result.get("error")}), flush=True)
        report["stages"].append(stage)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    objects = Path("/usr/local/lib/python3.12/dist-packages/_rocm_sdk_libraries/lib/hipblaslt/library/gfx950")
    report["code_objects"] = []
    for path in objects.glob("TensileLibrary_BB_BB_HA_Bias_SAV_UA_Type_BB_HPA_Contraction_l_Ailk_Bljk_Cijk_Dijk_ID75a0_gfx950.co"):
        data = path.read_bytes()
        report["code_objects"].append({"path": str(path), "size": len(data),
            "magic_hex": data[:32].hex(), "sha256": hashlib.sha256(data).hexdigest()})
    args.output.write_text(json.dumps(report, indent=2) + "\n")

if __name__ == "__main__":
    main()

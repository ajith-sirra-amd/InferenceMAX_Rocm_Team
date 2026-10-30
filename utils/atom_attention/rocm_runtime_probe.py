"""Inspect ROCm library provenance and baseline GEMM without importing vLLM."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

def libraries():
    return sorted({
        line.split()[-1] for line in Path("/proc/self/maps").read_text().splitlines()
        if any(name in line for name in (
            "libamdhip", "libhsa-runtime", "libhipblas", "librocblas", "libtorch_hip",
        )) and "/" in line
    })

def child(kind):
    import torch
    dtype = getattr(torch, kind)
    rows, columns, inner = (16, 16, 16) if kind == "bfloat16" else (31, 3072, 512)
    left = (torch.arange(rows * inner).reshape(rows, inner) % 7 - 3).to(dtype)
    right = (torch.arange(columns * inner).reshape(columns, inner) % 7 - 3).to(dtype)
    report = {
        "dtype": kind, "shape_mnk": [rows, columns, inner],
        "scope": "fresh process; torch only; no vLLM or FP8 kernel import",
        "environment": {k: os.environ.get(k) for k in (
            "LD_LIBRARY_PATH", "ROCM_PATH", "HIP_PATH", "HSA_XNACK",
            "ROCBLAS_USE_HIPBLASLT", "HIPBLASLT_TENSILE_LIBPATH",
        )},
    }
    try:
        report.update(torch=torch.__version__, hip=torch.version.hip,
                      gpu=str(torch.cuda.get_device_properties(0)))
        lhs, rhs = left.cuda(), right.cuda()
        output = lhs @ rhs.t()
        torch.cuda.synchronize()
        reference = left.double() @ right.double().t()
        actual = output.cpu().double()
        report.update(status="passed" if torch.equal(actual, reference) else "mismatch",
                      max_absolute_error=(actual - reference).abs().max().item())
    except Exception as error:
        report.update(status="runtime_error", error=str(error), traceback=traceback.format_exc())
    report["loaded_libraries"] = libraries()
    print("RUNTIME_GEMM_RESULT " + json.dumps(report), flush=True)

def command(args):
    try:
        p = subprocess.run(args, capture_output=True, text=True, timeout=30)
        return {"command": args, "exit_code": p.returncode,
                "stdout": p.stdout[:16000], "stderr": p.stderr[:6000]}
    except Exception as error:
        return {"command": args, "error": str(error)}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--child", choices=("bfloat16", "float32", "float64"))
    args = parser.parse_args()
    if args.child:
        child(args.child)
        return
    if args.output is None:
        parser.error("--output is required")
    results = []
    for kind in ("bfloat16", "float32", "float64"):
        p = subprocess.run(
            [sys.executable, __file__, "--child", kind],
            text=True, capture_output=True, timeout=90,
        )
        print(p.stdout, flush=True)
        if p.stderr:
            print(p.stderr, file=sys.stderr, flush=True)
        entries = [line.split("RUNTIME_GEMM_RESULT ", 1)[1]
                   for line in p.stdout.splitlines() if line.startswith("RUNTIME_GEMM_RESULT ")]
        result = json.loads(entries[-1]) if entries else {"status": "process_failed"}
        result.update(exit_code=p.returncode, stderr=p.stderr[-16000:])
        results.append(result)
    import torch
    metadata = {
        "torch": torch.__version__, "hip": torch.version.hip,
        "torch_library": command(["ldd", str(Path(torch.__file__).parent / "lib/libtorch_hip.so")]),
        "kernel": command(["uname", "-a"]), "tests": results,
    }
    root = Path("/usr/local/lib/python3.12/dist-packages/_rocm_sdk_libraries")
    metadata["code_objects"] = []
    for path in list(root.glob("lib/hipblaslt/library/gfx950/TensileLibrary_BB_BB_HA*ID75a0_gfx950.co"))[:1]:
        metadata["code_objects"].append({
            "path": str(path), "size": path.stat().st_size,
            "header": command(["readelf", "-h", str(path)]),
            "notes": command(["readelf", "-n", str(path)]),
        })
    metadata["scope"] = "runtime diagnosis only; not native model accuracy or performance"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metadata, indent=2) + "\n")

if __name__ == "__main__":
    main()

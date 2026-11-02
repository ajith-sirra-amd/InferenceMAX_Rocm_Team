"""Isolate BF16 BLAS backend selection; retain the original numerical gates."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROWS = [1410, 1733, 1734, 1735, 1737, 1740, 1745, 1752,
        1754, 1760, 1773, 1776, 1788, 1077]


def worker(rank, output, expect_blas):
    import warnings
    import torch
    import torch.nn.functional as functional

    torch.set_num_threads(2)
    torch.manual_seed(7)
    selected_blas = str(torch.backends.cuda.preferred_blas_library())
    report = {"physical_rank": rank, "torch": torch.__version__,
              "hip": torch.version.hip, "gpu": torch.cuda.get_device_name(0),
              "scope": "fresh process, BF16 F.linear only; no FP8 kernel imports",
              "cases": [], "status": "running",
              "selected_blas": selected_blas, "expected_blas": expect_blas,
              "environment": {key: os.getenv(key) for key in (
                  "TORCH_BLAS_PREFER_HIPBLASLT", "TORCH_BLAS_PREFER_CUBLASLT",
                  "ROCBLAS_USE_HIPBLASLT", "HIP_VISIBLE_DEVICES", "TMPDIR")},
              "torch_git_version": torch.version.git_version}

    try:
        if expect_blas and selected_blas.lower().split(".")[-1] != expect_blas:
            raise RuntimeError(f"Expected {expect_blas}, selected {selected_blas}")
        weight = torch.randn(2880, 7168, device="cuda", dtype=torch.bfloat16) * 0.02
        for rows in ROWS:
            item = {"rows": rows, "input_dim": 7168, "output_dim": 2880,
                    "status": "running"}
            report["cases"].append(item)
            output.write_text(json.dumps(report, indent=2))
            value = torch.randn(rows, 7168, device="cuda", dtype=torch.bfloat16)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                for _ in range(3):
                    result = functional.linear(value, weight)
                    torch.cuda.synchronize()
            sample = result[:4, :16].cpu().double()
            reference = value[:4].cpu().double() @ weight[:16].cpu().double().t()
            error = sample - reference
            relative_rms = (error.square().mean() / reference.square().mean()).sqrt().item()
            messages = [str(w.message) for w in caught]
            failed = (
                not torch.isfinite(result).all().item()
                or relative_rms >= 0.02
                or any("hipblas" in text.lower() for text in messages)
            )
            item.update(status="failed" if failed else "passed",
                        sampled_relative_rms=relative_rms, warnings=messages)
            output.write_text(json.dumps(report, indent=2))
            print(json.dumps({"rank": rank, **item}), flush=True)
            del result, value
        report["status"] = (
            "passed" if all(c["status"] == "passed" for c in report["cases"])
            else "failed"
        )
    except Exception as exc:
        report.update(status="failed", error=repr(exc))
    finally:
        output.write_text(json.dumps(report, indent=2) + "\n")
    return 0 if report["status"] == "passed" else 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", type=int)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expect-blas", choices=["cublas", "cublaslt"])
    args = parser.parse_args()
    if args.worker is not None:
        return worker(args.worker, args.output, args.expect_blas)
    args.output.mkdir(parents=True, exist_ok=True)
    children = []
    for rank in range(8):
        env = os.environ.copy()
        env["HIP_VISIBLE_DEVICES"] = str(rank)
        env.pop("CUDA_VISIBLE_DEVICES", None)
        env["OMP_NUM_THREADS"] = "2"
        log = (args.output / f"rank-{rank}.log").open("w")
        command = [sys.executable, __file__, "--worker", str(rank),
                   "--output", str(args.output / f"rank-{rank}.json")]
        if args.expect_blas:
            command.extend(["--expect-blas", args.expect_blas])
        child = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        children.append((rank, child, log))
    deadline = time.monotonic() + 240
    reports = []
    for rank, child, log in children:
        timed_out = False
        try:
            child.wait(timeout=max(0.1, deadline-time.monotonic()))
        except subprocess.TimeoutExpired:
            timed_out = True
            child.kill()
            child.wait()
        finally:
            log.close()
        file = args.output / f"rank-{rank}.json"
        report = json.loads(file.read_text()) if file.exists() else {}
        report.update(physical_rank=rank, exit_code=child.returncode, timeout=timed_out)
        reports.append(report)
    passed = all(
        r.get("status") == "passed" and r["exit_code"] == 0 and not r["timeout"]
        and len(r.get("cases", [])) == len(ROWS) for r in reports
    )
    summary = {"status": "passed" if passed else "failed", "ranks": reports,
               "scope": "isolated BF16 runtime diagnostic; not serving or FP8 accuracy",
               "all_eight_ranks_required": True}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

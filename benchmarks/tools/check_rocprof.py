"""Bounded attachment diagnostic with a parent and eight small GPU workers."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def worker(device, output):
    import torch
    torch.set_num_threads(1)
    torch.cuda.set_device(device)
    x = torch.ones((128, 128), dtype=torch.bfloat16, device=f"cuda:{device}")
    y = torch.empty_like(x)
    torch.mm(x, x, out=y)
    torch.cuda.synchronize(device)
    if not bool(torch.all(y == 128).item()):
        raise RuntimeError(f"GPU {device} failed matrix check")
    (output / f"worker-{device}-ready.json").write_text(json.dumps({
        "device": device, "pid": os.getpid(), "gpu": torch.cuda.get_device_name(device)}))
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        torch.mm(x, x, out=y)
        torch.cuda.synchronize(device)
        time.sleep(0.05)


def workers(output):
    processes = []
    try:
        for device in range(8):
            processes.append(subprocess.Popen([sys.executable, __file__, "--worker", str(device),
                                               "--output", str(output)]))
        return max(p.wait() for p in processes)
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            process.wait(timeout=10)


def check(output):
    from kernel_report import aggregate
    from container import probe
    output.mkdir(parents=True, exist_ok=False)
    record = {"status": "failed", "scope": "small GPU workload; not Kimi or SA replication"}
    tree = None
    try:
        record["capabilities"] = probe(output / "capabilities")
        with (output / "workers.log").open("w") as log:
            tree = subprocess.Popen([sys.executable, __file__, "--workers", "--output", str(output)],
                                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            deadline = time.monotonic() + 120
            while len(list(output.glob("worker-*-ready.json"))) != 8:
                if tree.poll() is not None or time.monotonic() > deadline:
                    raise RuntimeError("Eight workers did not become ready; see workers.log")
                time.sleep(0.5)
            record["workers"] = [json.loads(p.read_text()) for p in sorted(output.glob("worker-*-ready.json"))]
            argv = ["rocprofv3", "--attach", str(tree.pid), "--attach-children=true",
                    "--attach-duration-msec", "5000", "--attach-sync-output=true",
                    "--kernel-trace", "--hip-trace", "--memory-copy-trace", "--marker-trace",
                    "--stats", "-f", "csv", "json", "-d", str(output / "traces")]
            record["command"] = argv
            with (output / "rocprof.log").open("w") as trace_log:
                result = subprocess.run(argv, stdout=trace_log, stderr=subprocess.STDOUT, timeout=90)
            record["profiler_returncode"] = result.returncode
            summary = aggregate(output / "traces", output / "report", "rocprof", 8)
            record["capture"] = {key: summary.get(key) for key in
                ("kernel_count", "unique_kernels", "devices", "device_coverage_complete", "errors", "coverage")}
            if result.returncode or not summary["kernel_count"] or not summary.get("device_coverage_complete") or summary["errors"]:
                raise RuntimeError("Attachment did not produce valid kernel traces on all eight GPUs")
            if tree.poll() is not None:
                raise RuntimeError("Worker tree exited during the capture")
            record["status"] = "completed"
    except Exception as error:
        record["error"] = str(error)
    finally:
        if tree is not None:
            try:
                os.killpg(tree.pid, signal.SIGTERM)
                tree.wait(timeout=10)
            except ProcessLookupError:
                pass
            except subprocess.TimeoutExpired:
                os.killpg(tree.pid, signal.SIGKILL)
                tree.wait(timeout=10)
        (output / "check.json").write_text(json.dumps(record, indent=2) + "\n")
        print(json.dumps(record, indent=2), flush=True)
    return 0 if record["status"] == "completed" else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("/output/rocprof-check"))
    parser.add_argument("--worker", type=int)
    parser.add_argument("--workers", action="store_true")
    args = parser.parse_args()
    if args.worker is not None:
        worker(args.worker, args.output)
    elif args.workers:
        raise SystemExit(workers(args.output))
    else:
        raise SystemExit(check(args.output))

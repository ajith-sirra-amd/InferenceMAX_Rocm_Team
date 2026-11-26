"""Validate complete native Torch exports against the selected measured interval."""

import argparse
import csv
import gzip
import json
import math
import re
from pathlib import Path


def validate(
    logs,
    minimum_seconds,
    require_metadata=False,
    workload="agentic",
    result_filename=None,
):
    root = logs / "profiles/torch"
    control = json.loads((root / "profile-control.json").read_text())
    if control["status"] != "exported":
        raise ValueError("Controller did not complete profile export")
    if workload in ("fixed", "vllm"):
        result = json.loads((logs / f"{result_filename}.json").read_text())
        if workload == "vllm":
            if result["completed"] != result["num_prompts"]:
                raise ValueError("Native vLLM benchmark did not complete all requests")
            starts, latencies = result["start_times"], result["latencies"]
            if (
                not starts
                or len(starts) != len(latencies)
                or len(starts) != result["num_prompts"]
            ):
                raise ValueError("Native vLLM detailed request timings are missing")
            offset = float(control["perf_counter_to_unix_seconds"])
            if not math.isfinite(offset) or any(
                not math.isfinite(value) or value <= 0 for value in starts + latencies
            ):
                raise ValueError("Invalid native vLLM request timings")
            times = [
                min(starts) + offset,
                max(start + latency for start, latency in zip(starts, latencies))
                + offset,
            ]
        else:
            times = [
                float(result[key])
                for key in ("benchmark_start_time_unix", "benchmark_end_time_unix")
            ]
        if (
            not all(math.isfinite(value) and value > 0 for value in times)
            or times[1] <= times[0]
        ):
            raise ValueError("Invalid fixed-length measurement interval")
        begin, end = [round(value * 1e9) for value in times]
    elif workload == "agentic":
        phases = re.findall(
            r"PhaseRecordsStats\(phase=CreditPhase.PROFILING,.*?start_ns=(\d+), sent_end_ns=(\d+)",
            (logs / "agentic/benchmark.log").read_text(errors="replace"),
        )
        if len(phases) != 1:
            raise ValueError("Expected one completed measured phase")
        begin, end = map(int, phases[0])
    else:
        raise ValueError("Unknown profiling workload")
    paths = sorted(root.rglob("*.trace.json*"))
    if len(paths) != 8:
        raise ValueError(f"Expected eight worker traces; found {len(paths)}")
    devices, files, kernels = {}, [], {}
    for path in paths:
        before = path.stat()
        with gzip.open(path, "rt") if path.suffix == ".gz" else path.open() as stream:
            trace = json.load(stream)
        base = int(trace.get("baseTimeNanoseconds", 0))
        seen = set()
        count = 0
        metadata = {
            "shape_events": 0,
            "input_dtype_events": 0,
            "python_frames": 0,
            "memory_events": 0,
        }
        for event in trace["traceEvents"]:
            args = event.get("args", {})
            metadata["shape_events"] += bool(args.get("Input Dims"))
            metadata["input_dtype_events"] += bool(args.get("Input type"))
            metadata["python_frames"] += event.get("cat") == "python_function"
            metadata["memory_events"] += event.get("name") == "[memory]"
            if event.get("cat") != "kernel" or event.get("dur", 0) <= 0:
                continue
            device = str(event["args"]["device"])
            seen.add(device)
            start = base + round(event["ts"] * 1000)
            stop = start + round(event["dur"] * 1000)
            name = event.get("name", "<unnamed>")
            duration = event["dur"]
            kernel = kernels.setdefault(
                name,
                {"calls": 0, "total_us": 0, "min_us": duration, "max_us": duration},
            )
            kernel.update(
                calls=kernel["calls"] + 1,
                total_us=kernel["total_us"] + duration,
                min_us=min(kernel["min_us"], duration),
                max_us=max(kernel["max_us"], duration),
            )
            d = devices.setdefault(
                device, {"calls": 0, "first_ns": start, "last_ns": stop}
            )
            d.update(
                calls=d["calls"] + 1,
                first_ns=min(start, d["first_ns"]),
                last_ns=max(stop, d["last_ns"]),
            )
            count += 1
        if len(seen) != 1 or count == 0:
            raise ValueError(
                f"{path.name}: expected one worker GPU and nonzero kernels"
            )
        if require_metadata and not all(
            metadata[key] > 0
            for key in ("shape_events", "input_dtype_events", "python_frames")
        ):
            raise ValueError(
                f"{path.name}: missing required shape, dtype or Python-call metadata: {metadata}"
            )
        if (before.st_size, before.st_mtime_ns) != (
            path.stat().st_size,
            path.stat().st_mtime_ns,
        ):
            raise ValueError(f"{path.name}: export still changing")
        files.append(
            {
                "file": path.name,
                "bytes": before.st_size,
                "kernels": count,
                "devices": sorted(seen),
                "metadata": metadata,
            }
        )
        print(json.dumps(files[-1]), flush=True)
        del trace
    if set(devices) != set(map(str, range(8))):
        raise ValueError("Expected GPU devices 0–7")
    for d in devices.values():
        if not begin <= d["first_ns"] <= d["last_ns"] <= end:
            raise ValueError("GPU capture is outside measured sending")
    common = (
        min(d["last_ns"] for d in devices.values())
        - max(d["first_ns"] for d in devices.values())
    ) / 1e9
    if common < minimum_seconds:
        raise ValueError(
            f"Only {common:.3f}s common GPU extent; require {minimum_seconds}s"
        )
    total_us = sum(k["total_us"] for k in kernels.values())
    rows = [
        dict(
            kernel=name,
            **stats,
            mean_us=stats["total_us"] / stats["calls"],
            gpu_time_share_percent=100 * stats["total_us"] / total_us,
        )
        for name, stats in sorted(
            kernels.items(), key=lambda pair: pair[1]["total_us"], reverse=True
        )
    ]
    cumulative = 0
    for row in rows:
        cumulative += row["gpu_time_share_percent"]
        row["cumulative_gpu_time_share_percent"] = cumulative
    for filename, subset in (("kernels.csv", rows), ("top10-kernels.csv", rows[:10])):
        with (root / filename).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(subset)
    return {
        "status": "passed",
        "tool": "torch",
        "common_coverage_seconds": common,
        "measurement_utc_ns": [begin, end],
        "devices": devices,
        "files": files,
        "top10_kernels": rows[:10],
        "limits": "Kernel shares use summed GPU duration across workers, not E2E wall time. "
        "Extent is not proof of gap-free recording, decode attribution or profiler overhead.",
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("logs", type=Path)
    p.add_argument("--minimum-seconds", type=float, required=True)
    p.add_argument(
        "--workload", choices=("fixed", "vllm", "agentic"), default="agentic"
    )
    p.add_argument("--result-filename")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--require-metadata", action="store_true")
    a = p.parse_args()
    try:
        result = validate(
            a.logs, a.minimum_seconds, a.require_metadata, a.workload, a.result_filename
        )
    except (OSError, ValueError, KeyError, TypeError) as error:
        result = {"status": "failed", "error": str(error)}
    a.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    raise SystemExit(0 if result["status"] == "passed" else 1)

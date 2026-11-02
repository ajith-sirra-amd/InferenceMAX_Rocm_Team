#!/usr/bin/env python3
"""Summarize worker Chrome traces without loading the complete JSON into memory."""

import argparse
import csv
import gzip
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import ijson


def union_duration(intervals):
    total = 0.0
    end = None
    for start, stop in sorted(intervals):
        if end is None or start > end:
            total += stop - start
            end = stop
        elif stop > end:
            total += stop - end
            end = stop
    return total


def summarize(events):
    categories = Counter()
    kernels = defaultdict(lambda: [0, 0.0])
    devices = defaultdict(list)
    annotations = Counter()
    for event in events:
        category = str(event.get("cat", ""))
        categories[category] += 1
        if event.get("ph") != "X":
            continue
        if category == "user_annotation":
            annotations[str(event.get("name", ""))] += 1
        if category != "kernel":
            continue
        start = float(event["ts"])
        duration = float(event["dur"])
        if not math.isfinite(start) or not math.isfinite(duration) or duration < 0:
            raise ValueError("Invalid GPU event timing")
        if duration == 0:
            continue
        name = str(event["name"])
        device = str(event.get("args", {}).get("device", event.get("pid", "unknown")))
        kernels[name][0] += 1
        kernels[name][1] += duration
        devices[device].append((start, start + duration))
    if not kernels:
        raise ValueError("Trace contains no positive-duration GPU kernels")
    device_stats = {}
    for device, intervals in devices.items():
        first = min(start for start, _ in intervals)
        last = max(stop for _, stop in intervals)
        device_stats[device] = {
            "first_kernel_ts_us": first,
            "last_kernel_ts_us": last,
            "observed_kernel_span_us": last - first,
            "kernel_busy_us": union_duration(intervals),
        }
    return {
        "event_categories": dict(categories),
        "kernels": dict(kernels),
        "devices": device_stats,
        "annotations": annotations.most_common(30),
    }


def analyze(trace_dir, proof_path, native_sha, expected_workers, output):
    control = json.loads((trace_dir / "profile-control.json").read_text())
    if control.get("status") != "captured":
        raise ValueError("Controller did not report a successful capture")
    proof = json.loads(proof_path.read_text())
    expected = set(range(expected_workers))
    workers = proof["workers"]
    if (
        proof["native_sha"] != native_sha
        or proof["phase"] != "replay"
        or {w["rank"] for w in workers} != expected
        or len(workers) != expected_workers
        or any(not {"projection", "cached_fmha"} <= set(w["kinds"]) for w in workers)
    ):
        raise ValueError("Native replay execution proof does not match")
    ranks = {}
    all_kernels = defaultdict(lambda: [0, 0.0])
    skipped = []
    for path in sorted(trace_dir.rglob("*.trace.json*")):
        match = re.search(r"_rank(\d+)(?:[._]|$)", path.name)
        if not match:
            skipped.append(str(path.relative_to(trace_dir)))
            continue
        rank = int(match.group(1))
        if rank in ranks or rank not in expected:
            raise ValueError(f"Unexpected or repeated worker rank {rank}")
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rb") as stream:
            stats = summarize(ijson.items(stream, "traceEvents.item"))
        for name, (count, duration) in stats["kernels"].items():
            all_kernels[name][0] += count
            all_kernels[name][1] += duration
        stats["trace"] = str(path.relative_to(trace_dir))
        stats["kernel_count"] = sum(v[0] for v in stats["kernels"].values())
        stats["kernel_duration_us"] = sum(v[1] for v in stats["kernels"].values())
        del stats["kernels"]
        ranks[rank] = stats
    if set(ranks) != expected:
        raise ValueError(f"Missing GPU traces for ranks {sorted(expected - set(ranks))}")
    output.mkdir(parents=True, exist_ok=True)
    total = sum(v[1] for v in all_kernels.values())
    rows = [
        {
            "kernel": name,
            "calls": count,
            "total_gpu_ms": duration / 1000,
            "mean_us": duration / count,
            "share_of_summed_kernel_time_pct": duration * 100 / total,
        }
        for name, (count, duration) in sorted(
            all_kernels.items(), key=lambda item: item[1][1], reverse=True
        )
    ]
    with (output / "kernel-times.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "status": "analyzed",
        "native_sha": native_sha,
        "control": control,
        "workers": ranks,
        "skipped_unranked_traces": skipped,
        "top_kernels": rows[:30],
        "limits": [
            "Kernel shares use summed durations across workers, not wall time.",
            "Busy time merges overlap separately on each worker/device.",
            "The observed kernel span excludes leading/trailing idle capture time.",
            "Profiled throughput is not an unprofiled performance baseline.",
            "Kernel names and annotations still need interpretation.",
        ],
    }
    (output / "profile-analysis.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["trace-dir", "proof-path", "output"]:
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--native-sha", required=True)
    parser.add_argument("--expected-workers", type=int, required=True)
    args = vars(parser.parse_args())
    if args["expected_workers"] <= 0:
        parser.error("expected-workers must be positive")
    report = analyze(**args)
    print(json.dumps({"status": report["status"], "workers": list(report["workers"])}))


if __name__ == "__main__":
    main()

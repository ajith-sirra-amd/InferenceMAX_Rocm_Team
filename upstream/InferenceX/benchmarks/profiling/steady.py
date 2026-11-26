"""Control vLLM's native profiler from the selected workload's measured-phase marker."""

import argparse
import json
import math
import os
import re
import signal
import time
import urllib.request
from datetime import UTC, datetime, timedelta
from pathlib import Path

MARKER = re.compile(r"(\d\d:\d\d:\d\d\.\d+)\s+.*Phase profiling\b.*\bstarted\b")


def marker_time(line, observed, offset):
    match = MARKER.search(line)
    if not match:
        return None
    zone = datetime.strptime(offset.strip(), "%z").tzinfo
    today = datetime.fromtimestamp(observed, zone)
    clock = datetime.strptime(match[1], "%H:%M:%S.%f").time()
    candidates = [
        datetime.combine((today + timedelta(days=d)).date(), clock, zone).timestamp()
        for d in (-1, 0, 1)
    ]
    return min(candidates, key=lambda value: abs(value - observed))


def request(base, path, body=None, timeout=30):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(
        base + path, data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return response.read().decode()


def tokens(metrics):
    values = re.findall(
        r"^vllm:generation_tokens_total(?:\{[^\n]*\})?\s+([\d.eE+-]+)",
        metrics,
        re.MULTILINE,
    )
    total = sum(map(float, values))
    if not values or not math.isfinite(total):
        raise ValueError("Missing or invalid generation counter")
    return total


def capture(
    output,
    benchmark_log,
    base_url,
    workload_pid,
    settle_seconds,
    capture_seconds,
    timeout_seconds,
    poll_seconds,
    export_timeout_seconds,
    phase_file=None,
    log_format="agentic",
):
    output.mkdir(parents=True, exist_ok=True)
    control = output / "profile-control.json"
    if control.exists() or list(output.rglob("*.trace.json*")):
        raise FileExistsError("Refusing an existing profile capture")
    report = {
        "status": "waiting_for_measurement",
        "settle_seconds": settle_seconds,
        "capture_seconds": capture_seconds,
        "base_url": base_url,
    }

    def save(status):
        report.update(status=status, updated_utc=datetime.now(UTC).isoformat())
        control.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report), flush=True)

    def alive():
        os.kill(workload_pid, 0)
        stat = Path(f"/proc/{workload_pid}/stat").read_text()
        if stat.rsplit(")", 1)[1].split()[0] == "Z":
            raise ProcessLookupError("Client exited before capture finished")

    def wait(seconds):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            alive()
            time.sleep(min(poll_seconds, max(0, deadline - time.monotonic())))

    save("waiting_for_measurement")
    attempted = False
    try:
        deadline, offset, pending = time.monotonic() + timeout_seconds, 0, ""
        start = None
        heartbeat = time.monotonic()
        while start is None:
            alive()
            if time.monotonic() >= deadline:
                raise TimeoutError("Measured phase did not start before the deadline")
            if phase_file is not None and phase_file.exists():
                start = float(json.loads(phase_file.read_text())["start_unix"])
                if not math.isfinite(start) or start <= 0:
                    raise ValueError("Invalid fixed-length measurement timestamp")
                report.update(
                    phase_marker=str(phase_file),
                    measurement_start_unix=start,
                    measurement_observed_unix=time.time(),
                )
            elif phase_file is None and benchmark_log.exists():
                with benchmark_log.open(errors="replace") as stream:
                    stream.seek(offset)
                    pending += stream.read()
                    offset = stream.tell()
                lines = pending.split("\n")
                pending = lines.pop()
                for line in lines:
                    observed = time.time()
                    if log_format == "vllm":
                        if line.strip() != "Starting main benchmark run...":
                            continue
                        start = observed
                        # Native detailed results use the host's perf_counter clock.
                        report["perf_counter_to_unix_seconds"] = (
                            observed - time.perf_counter()
                        )
                    else:
                        if not MARKER.search(line):
                            continue
                        zone = (
                            benchmark_log.parent / "agentic_power_timezone_offset.txt"
                        ).read_text()
                        start = marker_time(line, observed, zone)
                    report.update(
                        phase_marker=line,
                        measurement_start_unix=start,
                        measurement_observed_unix=observed,
                    )
                    break
            if start is None:
                if time.monotonic() - heartbeat >= 60:
                    save("waiting_for_measurement")
                    heartbeat = time.monotonic()
                wait(poll_seconds)
        target = start + settle_seconds
        report["target_start_unix"] = target
        if time.time() > target + poll_seconds:
            raise RuntimeError(
                "Measured marker arrived too late for the requested capture"
            )
        initial_tokens = tokens(request(base_url, "/metrics"))
        save("waiting_for_capture_target")
        while time.time() < target:
            wait(min(60, target - time.time()))
            save("waiting_for_capture_target")
        request(base_url, "/health")
        progress = tokens(request(base_url, "/metrics")) - initial_tokens
        if progress <= 0:
            raise RuntimeError("No generated-token progress after measurement started")
        report["generation_tokens_during_settle"] = progress
        attempted = True
        report["start_requested_unix"] = time.time()
        save("starting_capture")
        request(
            base_url,
            "/start_profile",
            {"profile_prefix": "measured"},
            export_timeout_seconds,
        )
        report["start_returned_unix"] = time.time()
        save("capturing")
        wait(capture_seconds)
    except BaseException as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        try:
            if attempted:
                report["stop_requested_unix"] = time.time()
                save("exporting")
                request(base_url, "/stop_profile", {}, export_timeout_seconds)
                report["stop_returned_unix"] = time.time()
        except BaseException as error:
            report["export_error"] = f"{type(error).__name__}: {error}"
            save("failed")
            raise
        finally:
            if "error" in report or "export_error" in report:
                save("failed")
    traces = sorted(output.rglob("*.trace.json*"))
    if len(traces) != 8 or any(p.stat().st_size == 0 for p in traces):
        report["error"] = f"Expected eight nonempty worker traces; found {len(traces)}"
        save("failed")
        raise RuntimeError(report["error"])
    report["traces"] = [
        {"file": str(p.relative_to(output)), "bytes": p.stat().st_size} for p in traces
    ]
    report["analysis_pending"] = True
    save("exported")
    return report


def interrupt(signum, frame):
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    raise InterruptedError(f"Profile controller interrupted by signal {signum}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("output", "benchmark-log"):
        p.add_argument("--" + name, required=True, type=Path)
    p.add_argument("--base-url", required=True)
    p.add_argument("--phase-file", type=Path)
    p.add_argument("--log-format", choices=("agentic", "vllm"), default="agentic")
    p.add_argument("--workload-pid", type=int, required=True)
    for name in (
        "settle-seconds",
        "capture-seconds",
        "timeout-seconds",
        "poll-seconds",
        "export-timeout-seconds",
    ):
        p.add_argument("--" + name, required=True, type=float)
    a = vars(p.parse_args())
    if any(not math.isfinite(a[k]) or a[k] <= 0 for k in a if k.endswith("_seconds")):
        p.error("Timing values must be finite and positive")
    if a["workload_pid"] <= 1:
        p.error("A managed workload PID is required")
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, interrupt)
    capture(**a)

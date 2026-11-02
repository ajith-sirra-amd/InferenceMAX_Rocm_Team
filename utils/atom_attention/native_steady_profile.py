#!/usr/bin/env python3
"""Capture a bounded vLLM profile after the agentic warm-up and settling window."""

import argparse
import json
import math
import os
import re
import signal
import time
import urllib.request
from pathlib import Path


def request(port: int, path: str, body: dict | None = None) -> str:
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=data,
        headers={"Content-Type": "application/json"} if data is not None else {},
    )
    with urllib.request.urlopen(req, timeout=120) as response:
        return response.read().decode()


def generation_tokens(metrics: str) -> float:
    values = re.findall(
        r"^vllm:generation_tokens_total(?:\{[^\n]*\})?\s+([\d.eE+-]+)(?:\s|$)",
        metrics,
        re.MULTILINE,
    )
    if not values:
        raise ValueError("Generation counter is absent from the server metrics")
    total = sum(float(value) for value in values)
    if not math.isfinite(total) or total < 0:
        raise ValueError("Invalid generation counter")
    return total


def ensure_alive(pid: int) -> None:
    if pid <= 1:
        raise ValueError("Expected the positive PID of the launched vLLM server")
    os.kill(pid, 0)


def wait_alive(pid: int, seconds: float, poll: float) -> None:
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        ensure_alive(pid)
        remaining = end - time.monotonic()
        if remaining > 0:
            time.sleep(min(poll, remaining))
    ensure_alive(pid)


def capture(
    *,
    repository: Path,
    output: Path,
    benchmark_log: Path,
    port: int,
    server_pid: int,
    settle_seconds: float,
    capture_seconds: float,
    timeout_seconds: float,
    poll_seconds: float,
    prefix: str,
) -> dict:
    repository = repository.resolve(strict=True)
    output = output.resolve()
    if output == repository or not output.is_relative_to(repository):
        raise ValueError("Profile output must be inside the allowed repository")
    if not 1 <= port <= 65535:
        raise ValueError("Invalid server port")
    timings = (settle_seconds, capture_seconds, timeout_seconds, poll_seconds)
    if any(not math.isfinite(value) or value <= 0 for value in timings):
        raise ValueError("Timing parameters must be finite and positive")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", prefix):
        raise ValueError("Invalid profile prefix")
    output.mkdir(parents=True, exist_ok=True)
    if list(output.rglob("*.trace.json*")):
        raise FileExistsError("Refusing stale profiler traces")
    control = output / "profile-control.json"
    with control.open("x", encoding="utf-8") as stream:
        json.dump({"status": "waiting"}, stream)

    owner = repository.stat()
    report = {"status": "failed", "prefix": prefix}
    try:
        deadline = time.monotonic() + timeout_seconds
        offset = 0
        marker = None
        pending = ""
        while marker is None:
            ensure_alive(server_pid)
            if time.monotonic() >= deadline:
                raise TimeoutError("Measured agentic phase did not start in time")
            if benchmark_log.exists():
                with benchmark_log.open(encoding="utf-8", errors="replace") as stream:
                    stream.seek(offset)
                    pending += stream.read()
                    offset = stream.tell()
                complete = pending.splitlines(keepends=True)
                pending = ""
                if complete and not complete[-1].endswith("\n"):
                    pending = complete.pop()
                for line in complete:
                    if re.search(r"Phase profiling\b.*\bstarted\b", line):
                        marker = line.strip()
                        break
            if marker is None:
                time.sleep(poll_seconds)

        report["phase_marker"] = marker
        report["measurement_observed_unix"] = time.time()
        before = generation_tokens(request(port, "/metrics"))
        wait_alive(server_pid, settle_seconds, poll_seconds)
        after = generation_tokens(request(port, "/metrics"))
        if after <= before:
            raise RuntimeError("No generation progress during the settling window")
        report["settle_seconds"] = settle_seconds
        report["generation_tokens_during_settle"] = after - before
        request(port, "/health")
        try:
            request(port, "/start_profile", {"profile_prefix": prefix})
            report["capture_started_unix"] = time.time()
            wait_alive(server_pid, capture_seconds, poll_seconds)
        finally:
            request(port, "/stop_profile", {})
        report["capture_stopped_unix"] = time.time()
        traces = sorted(output.rglob("*.trace.json*"))
        if not traces or any(path.stat().st_size == 0 for path in traces):
            raise RuntimeError("No complete profiler trace was exported")
        report.update(
            status="captured",
            capture_seconds=capture_seconds,
            traces=[
                {"name": str(path.relative_to(output)), "bytes": path.stat().st_size}
                for path in traces
            ],
            analysis_pending=True,
        )
        return report
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        control.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        if hasattr(os, "geteuid") and os.geteuid() == 0:
            # Only the new profile output is owned by this controller.
            for path in [*output.rglob("*"), output]:
                if path.is_symlink() or not path.resolve().is_relative_to(output):
                    raise ValueError("Unexpected link in profile output")
                if path.is_file() and path.stat().st_nlink != 1:
                    raise ValueError("Unexpected hard link in profile output")
                os.chown(path, owner.st_uid, owner.st_gid)


def interrupt_capture(signum, frame) -> None:
    signal.signal(signum, signal.SIG_IGN)
    raise InterruptedError(f"Profile controller received signal {signum}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["repository", "output", "benchmark-log"]:
        parser.add_argument(f"--{name}", type=Path, required=True)
    for name in ["port", "server-pid"]:
        parser.add_argument(f"--{name}", type=int, required=True)
    for name in ["settle-seconds", "capture-seconds", "timeout-seconds", "poll-seconds"]:
        parser.add_argument(f"--{name}", type=float, required=True)
    parser.add_argument("--prefix", required=True)
    args = vars(parser.parse_args())
    previous = {
        signum: signal.signal(signum, interrupt_capture)
        for signum in (signal.SIGINT, signal.SIGTERM)
    }
    try:
        print(json.dumps(capture(**args), indent=2))
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


if __name__ == "__main__":
    main()

"""Prepare and supervise a measured AgentX Torch capture without changing its workload."""

import argparse
import json
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from validate import validate


def prepare(
    logs, duration, delay, seconds, minimum, phase_timeout, poll, export_timeout
):
    values = (duration, delay, seconds, minimum, phase_timeout, poll, export_timeout)
    if any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError("All profiling timings must be finite and positive")
    if not minimum <= seconds <= 180:
        raise ValueError("Require minimum <= capture seconds <= 180")
    if poll >= delay:
        raise ValueError("Poll interval must be shorter than the measured-phase delay")
    if delay + seconds + 2 * export_timeout + poll + 90 >= duration:
        raise ValueError(
            "Delay, capture and start/stop export budgets must fit inside DURATION"
        )
    logs = logs.resolve()
    if (logs / "agentic/benchmark.log").exists():
        raise FileExistsError("Use fresh logs; an AgentX benchmark log already exists")
    output = logs / "profiles/torch"
    output.mkdir(parents=True, exist_ok=False)
    settings = {
        "delay": delay,
        "seconds": seconds,
        "minimum": minimum,
        "phase_timeout": phase_timeout,
        "poll": poll,
        "export_timeout": export_timeout,
        "duration": duration,
    }
    (output / "settings.json").write_text(json.dumps(settings, indent=2) + "\n")
    config = {
        "profiler": "torch",
        "torch_profiler_dir": str(output),
        "torch_profiler_activities": ["CPU", "CUDA"],
        "ignore_frontend": True,
        "torch_profiler_with_stack": False,
        "torch_profiler_record_shapes": False,
        "torch_profiler_with_memory": False,
        "torch_profiler_use_gzip": True,
        "torch_profiler_dump_cuda_time_total": False,
    }
    (output / "profiler-config.json").write_text(json.dumps(config, indent=2) + "\n")
    return config


def interrupt(signum, frame):
    # Finish stop/export on the first signal; repeated signals must not interrupt it.
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    raise InterruptedError(f"Profile supervisor interrupted by signal {signum}")


def stop_group(process, timeout):
    if process is None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        pass
    finally:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=10)


def run(logs, port, command):
    if not command or not 0 < port < 65536:
        raise ValueError("A client command and valid server port are required")
    logs = logs.resolve()
    output = logs / "profiles/torch"
    settings = json.loads((output / "settings.json").read_text())
    workload = controller = None
    result = {"status": "failed"}
    try:
        with (output / "controller.log").open("x") as stream:
            workload = subprocess.Popen(command, start_new_session=True)
            controller = subprocess.Popen(
                [
                    sys.executable,
                    str(Path(__file__).with_name("steady.py")),
                    "--output",
                    str(output),
                    "--benchmark-log",
                    str(logs / "agentic/benchmark.log"),
                    "--base-url",
                    f"http://127.0.0.1:{port}",
                    "--workload-pid",
                    str(workload.pid),
                    "--settle-seconds",
                    str(settings["delay"]),
                    "--capture-seconds",
                    str(settings["seconds"]),
                    "--timeout-seconds",
                    str(settings["phase_timeout"]),
                    "--poll-seconds",
                    str(settings["poll"]),
                    "--export-timeout-seconds",
                    str(settings["export_timeout"]),
                ],
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            print(
                f"[profile] Waiting for measured AgentX activity; control log: {output / 'controller.log'}",
                flush=True,
            )
            while workload.poll() is None:
                if controller.poll() not in (None, 0):
                    raise RuntimeError("Profile controller failed; see controller.log")
                time.sleep(min(settings["poll"], 1))
            if workload.returncode:
                raise subprocess.CalledProcessError(workload.returncode, command)
            if controller.wait(timeout=settings["export_timeout"] + 10) != 0:
                raise RuntimeError("Profile controller failed before export completed")
        result = validate(logs, settings["minimum"])
        print(
            f"[profile] Validated eight workers: {result['common_coverage_seconds']:.3f}s common GPU extent",
            flush=True,
        )
    except BaseException as error:
        result = {"status": "failed", "error": f"{type(error).__name__}: {error}"}
        raise
    finally:
        try:
            # The shell keeps the server alive until this supervisor has returned.
            if controller is not None and controller.poll() is None:
                stop_group(controller, settings["export_timeout"] + 10)
        finally:
            try:
                if workload is not None and workload.poll() is None:
                    stop_group(workload, 10)
            finally:
                (logs / "profile-validation.json").write_text(
                    json.dumps(result, indent=2) + "\n"
                )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--logs", required=True, type=Path)
    for name in (
        "duration",
        "delay",
        "seconds",
        "minimum",
        "phase-timeout",
        "poll",
        "export-timeout",
    ):
        prep.add_argument("--" + name, required=True, type=float)
    capture = sub.add_parser("run")
    capture.add_argument("--logs", required=True, type=Path)
    capture.add_argument("--port", required=True, type=int)
    capture.add_argument("command", nargs=argparse.REMAINDER)
    args = vars(parser.parse_args())
    action = args.pop("action")
    if action == "prepare":
        print(json.dumps(prepare(**args)))
    else:
        if args["command"][:1] == ["--"]:
            args["command"] = args["command"][1:]
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, interrupt)
        run(**args)


if __name__ == "__main__":
    main()

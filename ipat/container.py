"""Container-side lifecycle and separate server profiling for a frozen IPAT recipe."""
import argparse
import csv
import datetime as dt
import gzip
import importlib.metadata
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import threading
import time
from urllib.request import Request, urlopen


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def checked(argv, output, env=None, timeout=900):
    with output.open("w") as log:
        subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT, env=env, timeout=timeout, check=True)


def probe(output):
    output.mkdir(parents=True, exist_ok=True)
    facts = {"at": now(), "packages": {}}
    for name in ("torch", "vllm", "sglang", "agentx-harness"):
        try:
            facts["packages"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    try:
        import torch
        facts["rocm"] = torch.version.hip
        facts["devices"] = [{"index": i, "name": torch.cuda.get_device_name(i),
                             "bytes": torch.cuda.get_device_properties(i).total_memory}
                            for i in range(torch.cuda.device_count())]
    except Exception as error:
        facts["gpu_error"] = str(error)
    for tool, args in (("rocprofv3", ["--help"]), ("rocprofv3", ["--version"]),
                       ("rocprofv3", ["--list-avail"]), ("rocprof-compute", ["--version"])):
        name = tool + "-" + args[0].strip("-") + ".txt"
        try:
            checked([tool, *args], output / name, timeout=60)
        except Exception as error:
            facts[name] = str(error)
    if "vllm" in facts["packages"]:
        try:
            from vllm.config.profiler import ProfilerConfig
            facts["torch_profiler_fields"] = list(ProfilerConfig.__dataclass_fields__)
        except (ImportError, AttributeError) as error:
            facts["torch_profiler_error"] = str(error)
    write(output / "capabilities.json", facts)
    return facts


def request(url, body=None):
    data = json.dumps(body).encode() if body is not None else None
    with urlopen(Request(url, data=data, headers={"Content-Type": "application/json"}), timeout=90) as response:
        return response.read().decode()


def stop(process):
    if process and process.poll() is None:
        try: os.killpg(process.pid, signal.SIGINT)
        except ProcessLookupError: return
        try:
            process.wait(timeout=90)
        except subprocess.TimeoutExpired:
            try: os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError: return
            process.wait(timeout=15)


def trace_summary(directory):
    kernels, operators, devices, processes = {}, {}, set(), set()
    files, errors = [], []
    for path in sorted(directory.rglob("*")):
        if not path.is_file() or not (path.name.endswith(".json.gz") or path.name.endswith(".trace.json") or path.name.endswith("kernel_trace.csv")):
            continue
        try:
            if path.name.endswith("kernel_trace.csv"):
                with path.open() as f:
                    rows = list(csv.DictReader(f))
                for row in rows:
                    name = row.get("Kernel_Name", row.get("Name", "unknown"))
                    duration = (float(row["End_Timestamp"]) - float(row["Start_Timestamp"])) / 1000
                    bucket = kernels.setdefault(name, {"name": name, "calls": 0, "total_us": 0})
                    bucket["calls"] += 1; bucket["total_us"] += duration
                    devices.add(str(row.get("Agent_Id", "unknown")))
                    processes.add(str(row.get("Process_Id", "unknown")))
            else:
                opener = gzip.open if path.name.endswith(".gz") else open
                with opener(path, "rt") as f:
                    events = json.load(f).get("traceEvents", [])
                for event in events:
                    category, args = event.get("cat", ""), event.get("args", {})
                    if event.get("ph") != "X" or not isinstance(event.get("dur"), (int, float)):
                        continue
                    target = kernels if category == "kernel" else operators if category == "cpu_op" else None
                    if target is None:
                        continue
                    name = event.get("name", "unknown")
                    bucket = target.setdefault(name, {"name": name, "calls": 0, "total_us": 0})
                    bucket["calls"] += 1; bucket["total_us"] += event["dur"]
                    if target is operators:
                        for key in ("Input Dims", "Input type", "Input Strides"):
                            if key in args and key not in bucket: bucket[key] = args[key]
                    else:
                        devices.add(str(args.get("device", event.get("pid"))))
                        processes.add(str(event.get("pid")))
            files.append(str(path.relative_to(directory)))
        except (OSError, ValueError, KeyError, TypeError) as error:
            errors.append({"file": path.name, "error": str(error)})
    return {"kernel_count": sum(k["calls"] for k in kernels.values()), "devices": sorted(devices),
            "processes": sorted(processes), "files": files, "errors": errors,
            "kernels": sorted(kernels.values(), key=lambda k: k["total_us"], reverse=True)[:50],
            "operators": sorted(operators.values(), key=lambda k: k["total_us"], reverse=True)[:50],
            "note": "Summed dispatch time may overlap across streams/devices; it is not end-to-end elapsed time. Operator input metadata does not prove fused internal precision or full tensor flow."}


def benchmark_metrics(path, chips):
    data = json.loads(path.read_text())
    def value(key, stat="avg", unit=None):
        row = data.get(key, {})
        number = row.get(stat)
        if unit and row.get("unit") != unit: return None
        return number if isinstance(number, (int, float)) and not isinstance(number, bool) and math.isfinite(number) else None
    output = value("output_token_throughput", unit="tokens/sec")
    return {"source": "client/profile_export_aiperf.json", "output_tokens_s": output,
            "output_tokens_s_gpu": output / chips if output is not None else None,
            "p90_full_response_itl_ms": value("full_response_inter_token_latency", "p90", "ms"),
            "p90_ttft_ms": value("time_to_first_token", "p90", "ms"),
            "completed_requests": value("request_count", unit="requests"),
            "failed_requests": value("error_request_count", unit="requests"),
            "duration_seconds": value("benchmark_duration", unit="sec"),
            "was_cancelled": data.get("was_cancelled"), "quality_evaluation": "not_run"}


def profile(recipe, client, output, url, result, finished):
    settings = recipe["profile"]
    profilers = []
    try:
        marker = settings.get("start_marker")
        if marker:
            log = output / "client/logs/aiperf.log"
            deadline = time.monotonic() + recipe["client_timeout_seconds"]
            while time.monotonic() < deadline:
                if log.exists() and marker in log.read_text(errors="replace"):
                    break
                if client.poll() is not None or finished.wait(2):
                    raise RuntimeError("Client ended before the profiling phase marker")
            else:
                raise TimeoutError("Profiling phase marker did not arrive")
        if finished.wait(settings["delay_seconds"]):
            raise RuntimeError("Client ended before the requested profile window")
        result["started_at"] = now()
        result["window_basis"] = "client profiling phase marker" if marker else "client start"
        traces = output / "traces"; traces.mkdir(exist_ok=True)
        if settings["kind"] == "torch":
            body = {} if recipe["framework"] == "vllm" else {"activities": ["CPU", "GPU"], "output_dir": str(traces)}
            result["start_response"] = request(url + "/start_profile", body)
            ended_early = finished.wait(settings["seconds"])
            result["stop_response"] = request(url + "/stop_profile", {})
            if ended_early: raise RuntimeError("Client ended before the full profile window completed")
        elif settings["kind"] == "rocprof":
            help_text = (output / "capabilities/rocprofv3-help.txt").read_text()
            if "--attach-duration-msec" not in help_text or "--attach-sync-output" not in help_text:
                raise RuntimeError("Installed rocprofv3 lacks the required bounded, synchronous attachment controls")
            pids = []
            for process in Path("/proc").iterdir():
                if not process.name.isdigit() or int(process.name) == os.getpid(): continue
                try:
                    if any(os.readlink(fd) == "/dev/kfd" for fd in (process / "fd").iterdir()):
                        pids.append(int(process.name))
                except (OSError, PermissionError): pass
            if not pids: raise RuntimeError("No GPU worker process has /dev/kfd open")
            result["target_pids"] = pids
            for pid in pids:
                path = traces / str(pid); path.mkdir()
                argv = ["rocprofv3", "--attach", str(pid), "--attach-duration-msec", str(settings["seconds"] * 1000),
                        "--attach-sync-output", "--hip-trace", "--kernel-trace", "--memory-copy-trace", "--marker-trace",
                        "--output-format", "csv", "--output-directory", str(path)]
                with (path / "collector.log").open("w") as log:
                    profilers.append(subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, start_new_session=True))
            for process in profilers:
                if process.wait(timeout=settings["seconds"] + 180):
                    raise RuntimeError("A GPU worker attachment failed; see collector.log")
        else:
            raise ValueError("Choose torch or rocprof profiling")
        result["ended_at"] = now()
        result["status"] = "collected"
    except Exception as error:
        result.update(status="failed", error=str(error), ended_at=now())
    finally:
        for process in profilers: stop(process)
        write(output / "profile.json", result)


def run(recipe, output, pass_name):
    output.mkdir(parents=True, exist_ok=True)
    capabilities = probe(output / "capabilities")
    manifest = {"pass": pass_name, "started_at": now(), "status": "running", "profile": {}}
    write(output / "pass.json", manifest)
    server = client = None
    profile_thread = None
    finished = threading.Event()
    url = "http://127.0.0.1:" + str(recipe["port"])
    values = {"{url}": url, "{output}": str(output / "client"), "{port}": str(recipe["port"]),
              "{duration}": str(recipe["duration_seconds"]), "{concurrency}": str(recipe["concurrency"])}
    def expand(argv):
        result = []
        for arg in argv:
            for key, value in values.items(): arg = arg.replace(key, value)
            result.append(arg)
        return result
    env = dict(os.environ, **recipe["environment"])
    try:
        expected = recipe["physical_chips"]
        devices = capabilities.get("devices", [])
        if len(devices) != expected or any("MI355" not in d["name"].upper() for d in devices):
            raise RuntimeError(f"Expected {expected} MI355X GPUs; observed {devices}")
        argv = expand(recipe["server_argv"])
        if pass_name == "profile" and recipe["profile"]["kind"] == "torch":
            if recipe["framework"] == "vllm":
                config = {"profiler": "torch", "torch_profiler_dir": str(output / "traces"),
                          "torch_profiler_record_shapes": True, "torch_profiler_with_stack": True}
                if not set(config).issubset(capabilities.get("torch_profiler_fields", [])):
                    raise RuntimeError("This vLLM image does not expose the required torch ProfilerConfig fields")
                argv += ["--profiler-config", json.dumps(config)]
            elif recipe["framework"] == "sglang":
                env["SGLANG_TORCH_PROFILER_DIR"] = str(output / "traces")
            else:
                raise ValueError("Use rocprof for a generic server; torch endpoints require vLLM or SGLang")
        manifest["server_argv"] = argv
        # Fail on missing client dependencies before paying the model startup cost.
        client_argv = expand(recipe["client_argv"])
        if recipe["client_packages"]:
            venv = output / "client-env"
            checked(["python3", "-m", "venv", str(venv)], output / "client-setup.log", env)
            checked([str(venv / "bin/python"), "-m", "pip", "install", *recipe["client_packages"]], output / "client-install.log", env)
            client_argv[0] = str(venv / "bin" / Path(client_argv[0]).name)
            checked([str(venv / "bin/python"), "-m", "pip", "freeze"], output / "client-packages.txt", env)
        manifest["client_argv"] = client_argv
        write(output / "pass.json", manifest)
        with (output / "server.log").open("w") as log:
            server = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        deadline = time.monotonic() + recipe["startup_timeout_seconds"]
        while time.monotonic() < deadline:
            if server.poll() is not None: raise RuntimeError("Server exited before readiness; see server.log")
            try:
                request(url + recipe["health_path"])
                break
            except Exception: time.sleep(5)
        else: raise TimeoutError("Server readiness timed out")
        manifest["server_ready_at"] = now()
        manifest["client_started_at"] = now()
        write(output / "pass.json", manifest)
        with (output / "client.log").open("w") as log:
            client = subprocess.Popen(client_argv, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        if pass_name == "profile":
            profile_thread = threading.Thread(target=profile, args=(recipe, client, output, url, manifest["profile"], finished), daemon=True)
            profile_thread.start()
        deadline = time.monotonic() + recipe["client_timeout_seconds"]
        with (output / "serving-metrics.jsonl").open("w") as metrics:
            while client.poll() is None:
                if server.poll() is not None: raise RuntimeError("Server exited during the client workload")
                if time.monotonic() > deadline: raise TimeoutError("Client workload timed out")
                try: row = {"at": now(), "text": request(url + "/metrics")}
                except Exception as error: row = {"at": now(), "error": str(error)}
                metrics.write(json.dumps(row) + "\n"); metrics.flush()
                time.sleep(5)
        if client.returncode: raise RuntimeError(f"Client failed ({client.returncode}); see client.log")
        metrics_path = output / "client/profile_export_aiperf.json"
        if Path(client_argv[0]).name == "agentx" and not metrics_path.exists():
            raise RuntimeError("AgentX returned without its benchmark metrics artifact")
        if metrics_path.exists():
            manifest["metrics"] = benchmark_metrics(metrics_path, recipe["physical_chips"])
            if not manifest["metrics"]["completed_requests"] or manifest["metrics"]["was_cancelled"]:
                raise RuntimeError("Benchmark did not finish with completed requests")
        manifest["status"] = "completed"
    except Exception as error:
        manifest.update(status="failed", error=str(error))
    finally:
        finished.set()
        if profile_thread: profile_thread.join(timeout=240)
        stop(client); stop(server)
        if pass_name == "profile":
            summary = trace_summary(output / "traces")
            summary["expected_devices"] = recipe["physical_chips"]
            summary["device_coverage_complete"] = len(summary["devices"]) == recipe["physical_chips"] and "unknown" not in summary["devices"]
            write(output / "trace-summary.json", summary)
            if manifest["profile"].get("status") != "collected" or not summary["kernel_count"] or not summary["device_coverage_complete"] or summary["errors"]:
                manifest.update(status="failed", error="Profiling did not verify GPU kernel coverage on every expected device; see profile.json and trace-summary.json")
        manifest["ended_at"] = now()
        write(output / "pass.json", manifest)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pass-name", choices=("baseline", "profile", "capabilities"), required=True)
    args = parser.parse_args()
    if args.pass_name == "capabilities":
        probe(args.output)
    else:
        result = run(json.loads(args.recipe.read_text()), args.output, args.pass_name)
        raise SystemExit(0 if result["status"] == "completed" else 1)

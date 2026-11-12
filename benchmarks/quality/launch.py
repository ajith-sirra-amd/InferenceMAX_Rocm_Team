#!/usr/bin/env python3
"""Separate natural-verification server; the SA performance script is unchanged."""
import ast
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys
import time
from urllib.request import urlopen

sys.path.insert(0, str(Path(__file__).resolve().parent))
import screen


def recipe(base_path, validation):
    raw = base_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != validation["base_recipe_sha256"]:
        raise ValueError("Base SA recipe hash changed; review before natural validation")
    result = json.loads(raw)
    spec = json.loads(result["server_arguments"]["speculative-config"])
    if spec.get("rejection_sample_method") != "synthetic" or spec.get("synthetic_acceptance_length") != 3.57:
        raise ValueError("Unexpected base verifier")
    override = validation["speculative_overrides"]
    spec["rejection_sample_method"] = override["rejection_sample_method"]
    for key in override["remove"]:
        spec.pop(key, None)
    if spec["rejection_sample_method"] != "block" or any(key.startswith("synthetic_") for key in spec):
        raise ValueError("Natural configuration must use block verification without synthetic acceptance")
    result["server_arguments"]["speculative-config"] = json.dumps(spec)
    result.update(name=validation["name"], validation=validation,
                  workload=dict(screen.PARAMS, requests=48, concurrency=48,
                                dataset_sha256=validation["dataset_sha256"], seed="42 + item index"),
                  note="Separate scored natural-verification screen; not SA throughput or model-quality qualification.")
    return result


def command(config, model, port):
    argv = ["/opt/runtime-venv/bin/vllm", "serve", model, "--host", "0.0.0.0", "--port", port]
    for key, value in config["server_arguments"].items():
        if isinstance(value, bool):
            if value:
                argv.append("--" + key)
        else:
            argv.extend(["--" + key, str(value)])
    return argv


def observed_verifier(log, expected):
    for line in log.splitlines():
        if "non-default args:" in line:
            match = re.search(r"'speculative_config': (\{[^{}]*\})", line)
            if match:
                observed = ast.literal_eval(match[1])
                if observed != expected:
                    raise ValueError("Emitted vLLM speculative configuration differs")
                return dict(observed=observed, source="vLLM non-default-args startup log",
                            expected_natural_config_match=True, sa_throughput_recipe_match=False,
                            limits="Parsed serving arguments; not an independent worker-state readback.")
    raise ValueError("No emitted vLLM speculative configuration in server log")


def main():
    output = Path(os.environ["RESULT_DIR"])
    validation, _ = screen.assets()
    config = recipe(Path("/fixed-tools/baseline.json"), validation)
    if os.environ["KIMI_FIXED_OPTIMIZATION"] not in ("baseline", "moe_front_dspark"):
        raise ValueError("Natural screen supports baseline and combined only")
    os.environ.update(config["server_environment"])
    argv = command(config, os.environ["MODEL_PATH"], os.environ["PORT"])
    screen.save(output / "recipe.json", config)
    (output / "vllm_command.txt").write_text(shlex.join(argv) + "\n")
    sys.path.insert(0, "/fixed-tools")
    import audit_server
    audit_server.audit(config, output, os.environ["MODEL_PATH"], os.environ["PORT"],
                       Path("/opt/runtime-venv/lib/python3.12/site-packages"), argv)
    effective_path = output / "effective-server.json"
    effective = json.loads(effective_path.read_text())
    effective.update(sa_config_match=False, expected_natural_config_match=True,
                     sa_throughput_recipe_match=False)
    screen.save(effective_path, effective)
    server = None
    result = dict(status="failed", expected_natural_config_match=True,
                  sa_throughput_recipe_match=False, error=None)

    def stopped(signum, frame):
        raise RuntimeError(f"Natural screen interrupted by signal {signum}")

    signal.signal(signal.SIGTERM, stopped)
    signal.signal(signal.SIGINT, stopped)
    try:
        with (output / "server.log").open("xb") as log:
            server = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT,
                                      start_new_session=True)
            deadline = time.monotonic() + validation["bounds_seconds"]["readiness"]
            endpoint = "http://127.0.0.1:" + os.environ["PORT"]
            while True:
                if server.poll() is not None:
                    raise RuntimeError(f"Server exited before readiness: {server.returncode}")
                if time.monotonic() >= deadline:
                    raise TimeoutError("Server readiness exceeded 30 minutes")
                try:
                    with urlopen(endpoint + "/health", timeout=5) as response:
                        if response.status == 200:
                            break
                except OSError:
                    pass
                time.sleep(2)
            observed = observed_verifier((output / "server.log").read_text(errors="replace"),
                                        json.loads(config["server_arguments"]["speculative-config"]))
            screen.save(output / "observed-verifier.json", observed)
            client = ["timeout", "--signal=TERM", "--kill-after=15s", "375s",
                      sys.executable, str(Path(__file__).with_name("screen.py")),
                      "--endpoint", endpoint, "--model", os.environ["MODEL"],
                      "--output", str(output / "quality")]
            (output / "client-command.txt").write_text(shlex.join(client) + "\n")
            result["client_exit_code"] = subprocess.call(client)
            if result["client_exit_code"]:
                raise RuntimeError("Client screen incomplete; see saved responses/summary")
            result["status"] = "collected"
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        if server is not None:
            try:
                os.killpg(server.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                server.wait(timeout=60)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(server.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                try:
                    server.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    result.update(status="failed", error="Owned server did not stop within 75 seconds")
        screen.save(output / "natural-validation.json", result)
    return 0 if result["status"] == "collected" else 1


if __name__ == "__main__":
    raise SystemExit(main())

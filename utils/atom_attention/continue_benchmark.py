"""Watch a smoke, verify its artifacts, and queue the authorized full benchmark."""
import argparse
import datetime
import json
import os
import subprocess
import time

import requests

from inspect_results import inspect
from watch_run import watch


def continue_benchmark(session, repo, branch, smoke_id, interval, max_seconds):
    deadline = time.monotonic() + max_seconds
    if watch(session, repo, smoke_id, interval, max_seconds) != 0:
        print("SMOKE_FAILED_OR_TIMED_OUT", flush=True)
        return 1
    summary = inspect(session, repo, smoke_id)
    print("SMOKE_EVIDENCE", json.dumps(summary), flush=True)
    profiling_errors = summary["phases"].get("profiling", {}).get("errors", {})
    if not summary["engagement_smoke_pass"] or sum(profiling_errors.values()) != 0:
        print("FULL_RUN_GATE_NOT_MET", flush=True)
        return 1
    base = f"https://api.github.com/repos/{repo}"
    response = session.get(f"{base}/actions/runs/{smoke_id}", timeout=30)
    response.raise_for_status()
    smoke = response.json()
    name = "vLLM ATOM FlyDSL FP8 - Moonshot TP8 DCP8 - Full Benchmark - Warmup 10 - 3600s"
    source_sha = smoke["head_sha"]
    response = session.get(f"{base}/actions/runs", params={"per_page": 100}, timeout=30)
    response.raise_for_status()
    existing = [
        run for run in response.json()["workflow_runs"]
        if run["display_title"] == "e2e Test - " + name
        and run["head_sha"] == source_sha
        and run["created_at"] >= smoke["created_at"]
    ]
    if existing:
        full_id = existing[0]["id"]
        print("REUSE_ALREADY_DISPATCHED_FULL_RUN", full_id, flush=True)
    else:
        sent = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        response = session.post(
            f"{base}/actions/workflows/e2e-tests.yml/dispatches",
            json={"ref": branch, "inputs": {
                "generate-cli-command": (
                    "test-config --config-files upstream/InferenceX/configs/amd-master.yaml "
                    "--config-keys kimik3-fp4-mi355x-vllm-agentic-mtp"
                ),
                "test-name": name,
                "ref": source_sha,
                "runner-diagnostics-only": "false",
                "agentic-eval-only": "false",
                "duration-override": "3600",
            }}, timeout=30,
        )
        response.raise_for_status()
        print("FULL_DISPATCH_ACCEPTED", source_sha, flush=True)
        full_id = None
        for _ in range(12):
            response = session.get(f"{base}/actions/runs", params={"per_page": 30}, timeout=30)
            response.raise_for_status()
            matches = [
                run for run in response.json()["workflow_runs"]
                if run["display_title"] == "e2e Test - " + name and run["created_at"] >= sent
                and run["head_branch"] == branch
            ]
            if matches:
                full_id = matches[0]["id"]
                break
            time.sleep(5)
        if full_id is None:
            raise RuntimeError("Dispatch accepted but run ID unavailable; do not dispatch a duplicate")
    print("FULL_RUN", full_id, f"https://github.com/{repo}/actions/runs/{full_id}", flush=True)
    result = watch(session, repo, full_id, interval, max(1, deadline-time.monotonic()))
    if result == 0:
        print("FULL_EVIDENCE", json.dumps(inspect(session, repo, full_id)), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--branch", required=True)
    parser.add_argument("--smoke-run", type=int, required=True)
    parser.add_argument("--interval", type=int, required=True)
    parser.add_argument("--max-hours", type=float, required=True)
    args = parser.parse_args()
    if args.interval < 1 or args.interval > 60 or args.max_hours <= 0:
        parser.error("interval must be 1..60 seconds; max-hours must be positive")
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if not token:
        token = subprocess.run(
            ["gh", "auth", "token"], capture_output=True, text=True, check=True,
        ).stdout.strip()
    session = requests.Session()
    session.headers["Authorization"] = "Bearer " + token
    raise SystemExit(continue_benchmark(
        session, args.repo, args.branch, args.smoke_run,
        args.interval, args.max_hours*3600,
    ))

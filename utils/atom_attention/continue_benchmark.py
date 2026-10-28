
"""Watch a controlled prefix smoke and queue the authorized full agentic replay."""
import argparse
import datetime
import io
import json
import os
import subprocess
import time
import zipfile

import requests

from inspect_results import inspect
from watch_run import watch


def prefix_evidence(session, repo, run_id, source_ref):
    base = f"https://api.github.com/repos/{repo}"
    response = session.get(f"{base}/actions/runs/{run_id}", timeout=30)
    response.raise_for_status()
    run = response.json()
    if run["head_sha"] != source_ref:
        raise ValueError("Smoke workflow revision differs from the explicitly tested source")
    response = session.get(
        f"{base}/actions/runs/{run_id}/attempts/{run['run_attempt']}/jobs",
        params={"per_page": 100}, timeout=30,
    )
    response.raise_for_status()
    jobs = [x for x in response.json()["jobs"] if x["name"].startswith("agentic /")]
    if len(jobs) != 1:
        raise ValueError("Expected one GPU smoke job")
    response = session.get(f"{base}/actions/runs/{run_id}/artifacts", timeout=30)
    response.raise_for_status()
    artifacts = [
        x for x in response.json()["artifacts"]
        if x["name"].startswith("agentic_") and "aggregated" not in x["name"]
        and x["created_at"] >= jobs[0]["started_at"]
    ]
    if len(artifacts) != 1:
        raise ValueError("Expected one current-attempt raw smoke artifact")
    response = session.get(artifacts[0]["archive_download_url"], timeout=60)
    response.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        def read(suffix):
            names = [x for x in archive.namelist() if x.endswith(suffix)]
            if len(names) != 1:
                raise ValueError(f"Expected one {suffix}")
            return archive.read(names[0]).decode("utf-8", errors="replace")
        evidence = json.loads(read("path_probe_summary.json"))
        numerical_log = read("atom_attention_preflight.log")
    markers = [
        "bf16_causal_pids", "fp8_cached_context_pids",
        "disabled_asm_metadata_pids", "query_reuse_pids",
        "native_fused_fp8_gather_pids",
    ]
    passed = (
        run["conclusion"] == "success"
        and evidence.get("kernel_invocation_pass") is True
        and evidence.get("model") == "moonshotai/Kimi-K3"
        and evidence.get("framework") == "vllm"
        and evidence.get("gpu_kv_tokens") is not None
        and evidence.get("prefix_hit_tokens_increment", 0) > 0
        and all(len(evidence.get(key, [])) >= 8 for key in markers)
        and all(x["failed"] == 0 and x["completed"] == x["prompts"] for x in evidence["stages"])
        and "ATOM_ATTENTION_PREFLIGHT_PASS" in numerical_log
        and "QUERY_REUSE_PASS" in numerical_log
    )
    evidence.update(run_id=run_id, attempt=run["run_attempt"], source_ref=source_ref,
                    continuation_gate_pass=passed)
    return evidence


def continue_benchmark(session, repo, branch, smoke_id, source_ref, interval, max_seconds):
    deadline = time.monotonic() + max_seconds
    if watch(session, repo, smoke_id, interval, max_seconds) != 0:
        print("SMOKE_FAILED_OR_TIMED_OUT; inspect logs before any full dispatch", flush=True)
        return 1
    evidence = prefix_evidence(session, repo, smoke_id, source_ref)
    print("SMOKE_EVIDENCE", json.dumps(evidence), flush=True)
    if not evidence["continuation_gate_pass"]:
        print("FULL_RUN_GATE_NOT_MET", flush=True)
        return 1
    base = f"https://api.github.com/repos/{repo}"
    name = "vLLM Kimi Hybrid FP8 - Native FP8 Gather - Full Agentic Replay - Warmup 10 - 3600s"
    sent = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    response = session.get(f"{base}/actions/runs/{smoke_id}", timeout=30)
    response.raise_for_status()
    smoke_created = response.json()["created_at"]
    def matching_runs():
        response = session.get(f"{base}/actions/runs", params={"per_page": 100}, timeout=30)
        response.raise_for_status()
        return [
            x for x in response.json()["workflow_runs"]
            if x["display_title"] == "e2e Test - " + name and x["head_branch"] == branch
            and x["created_at"] >= smoke_created
        ]
    matches = matching_runs()
    if matches:
        full_id = matches[0]["id"]
        print("REUSE_ALREADY_DISPATCHED_FULL_RUN", full_id, flush=True)
    else:
        # Never repeat an uncertain POST: find the accepted run instead.
        try:
            response = session.post(
                f"{base}/actions/workflows/e2e-tests.yml/dispatches",
                json={"ref": branch, "inputs": {
                    "generate-cli-command": (
                        "test-config --config-files upstream/InferenceX/configs/amd-master.yaml "
                        "--config-keys kimik3-fp4-mi355x-vllm-agentic-mtp"
                    ),
                    "test-name": name, "ref": source_ref,
                    "runner-diagnostics-only": "false",
                    "attention-path-probe-only": "false",
                    "agentic-eval-only": "false", "duration-override": "3600",
                }}, timeout=30,
            )
            response.raise_for_status()
            print("FULL_DISPATCH_ACCEPTED", source_ref, flush=True)
        except requests.RequestException:
            print("DISPATCH_RESPONSE_UNCERTAIN; checking run list without a second POST", flush=True)
        full_id = None
        for _ in range(24):
            try:
                matches = [x for x in matching_runs() if x["created_at"] >= sent]
            except requests.RequestException:
                matches = []
            if matches:
                full_id = matches[0]["id"]
                break
            time.sleep(5)
        if full_id is None:
            raise RuntimeError("Full dispatch outcome uncertain; do not dispatch a duplicate")
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
    parser.add_argument("--source-ref", required=True)
    parser.add_argument("--interval", type=int, required=True)
    parser.add_argument("--max-hours", type=float, required=True)
    args = parser.parse_args()
    if not 1 <= args.interval <= 60 or args.max_hours <= 0:
        parser.error("interval must be 1..60 seconds; max-hours must be positive")
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if not token:
        token = subprocess.run(["gh", "auth", "token"], capture_output=True,
                               text=True, check=True).stdout.strip()
    session = requests.Session()
    session.headers["Authorization"] = "Bearer " + token
    raise SystemExit(continue_benchmark(
        session, args.repo, args.branch, args.smoke_run, args.source_ref,
        args.interval, args.max_hours*3600,
    ))

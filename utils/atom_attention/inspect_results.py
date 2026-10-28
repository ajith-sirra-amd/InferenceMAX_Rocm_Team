"""Read current-attempt artifacts and summarize the attention smoke evidence."""
import collections
import io
import json
import re
import zipfile

import requests


def summarize(server_log, records, benchmark_command):
    capacity = re.findall(r"GPU KV cache (?:size|capacity):\s*([\d,]+) tokens", server_log)
    engaged = collections.defaultdict(set)
    for pid, stage in re.findall(r"\[ATOM-FP8-ENGAGED\] pid=(\d+) stage=(\w+)", server_log):
        engaged[stage].add(int(pid))
    phases = collections.defaultdict(lambda: {"success": 0, "cancelled": 0, "errors": {}})
    for record in records:
        metadata = record.get("metadata", {})
        phase = str(metadata.get("benchmark_phase", "unknown")).lower()
        item = phases[phase]
        error = record.get("error")
        if metadata.get("was_cancelled"):
            item["cancelled"] += 1
        elif error:
            kind = str(error.get("type", "unknown"))
            item["errors"][kind] = item["errors"].get(kind, 0) + 1
        else:
            item["success"] += 1
    duration = re.search(r"--benchmark-duration\s+(\d+)", benchmark_command)
    warmup = re.search(r"--warmup-requests-per-lane\s+(\d+)", benchmark_command)
    return {
        "gpu_kv_tokens": int(capacity[-1].replace(",", "")) if capacity else None,
        "engaged_processes": {stage: sorted(pids) for stage, pids in engaged.items()},
        "phases": dict(phases),
        "requested_duration_seconds": int(duration.group(1)) if duration else None,
        "warmup_requests_per_lane": int(warmup.group(1)) if warmup else None,
    }


def inspect(session, repo, run_id):
    base = f"https://api.github.com/repos/{repo}"
    response = session.get(f"{base}/actions/runs/{run_id}", timeout=30)
    response.raise_for_status()
    run = response.json()
    response = session.get(
        f"{base}/actions/runs/{run_id}/attempts/{run['run_attempt']}/jobs",
        params={"per_page": 100}, timeout=30,
    )
    response.raise_for_status()
    jobs = response.json()["jobs"]
    gpu_jobs = [j for j in jobs if j["name"].startswith("agentic /")]
    if len(gpu_jobs) != 1:
        raise ValueError("Expected exactly one agentic GPU job")
    cutoff = gpu_jobs[0]["started_at"]
    response = session.get(f"{base}/actions/runs/{run_id}/artifacts", timeout=30)
    response.raise_for_status()
    artifacts = [
        a for a in response.json()["artifacts"]
        if a["created_at"] >= cutoff and a["name"].startswith("agentic_")
        and "aggregated" not in a["name"]
    ]
    if len(artifacts) != 1:
        raise ValueError(f"Expected one current-attempt raw artifact, found {len(artifacts)}")
    response = session.get(artifacts[0]["archive_download_url"], timeout=60)
    response.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        def read(suffix):
            names = [n for n in archive.namelist() if n.endswith(suffix)]
            if len(names) != 1:
                raise ValueError(f"Expected exactly one {suffix} in raw artifact")
            return archive.read(names[0]).decode("utf-8", errors="replace")
        records = [json.loads(line) for line in read("profile_export.jsonl").splitlines()]
        result = summarize(read("server.log"), records, read("benchmark_command.txt"))
    result.update(run_id=run_id, attempt=run["run_attempt"], conclusion=run["conclusion"])
    profiling = result["phases"].get("profiling", {})
    result["engagement_smoke_pass"] = (
        run["conclusion"] == "success"
        and result["gpu_kv_tokens"] is not None
        and result["requested_duration_seconds"] == 600
        and result["warmup_requests_per_lane"] == 1
        and len(result["engaged_processes"].get("new_tokens", [])) >= 8
        and len(result["engaged_processes"].get("cached_context", [])) >= 8
        and profiling.get("success", 0) > 0
    )
    # This establishes engagement, not GSM8K accuracy or a throughput improvement.
    return result

"""
GitHub Actions tools for the InferenceMAX agentic orchestrator.

Provides functions to:
  - list_workflow_runs     list recent runs of a workflow
  - get_run_status         get status/conclusion of a specific run
  - get_run_logs           download and return logs of a run (all jobs or one job)
  - get_job_logs           download logs for a single job
  - trigger_workflow       dispatch a workflow_dispatch event
  - interpret_logs         ask Claude to summarize/interpret log text
  - watch_run              monitor a running workflow, streaming interpreted log updates

All GitHub calls use the REST API via the `requests` library with the
GITHUB_TOKEN set in config.py / .env.
"""

from __future__ import annotations

import json
import re
import time
import zipfile
import io
from datetime import datetime, timezone
from typing import Any

import requests

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _gh_headers() -> dict[str, str]:
    if not cfg.GITHUB_TOKEN:
        raise EnvironmentError(
            "GITHUB_TOKEN is not set. Add it to .env or export it in the shell."
        )
    return {
        "Authorization": f"Bearer {cfg.GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def _gh_get(path: str, params: dict | None = None) -> Any:
    url = f"https://api.github.com/repos/{cfg.GITHUB_REPO}/{path}"
    resp = requests.get(url, headers=_gh_headers(), params=params or {}, timeout=30)
    resp.raise_for_status()
    return resp.json()


def _gh_post(path: str, payload: dict) -> requests.Response:
    url = f"https://api.github.com/repos/{cfg.GITHUB_REPO}/{path}"
    resp = requests.post(url, headers=_gh_headers(), json=payload, timeout=30)
    resp.raise_for_status()
    return resp


def _fmt_dt(iso: str | None) -> str:
    if not iso:
        return "—"
    dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
    return dt.astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")


# ---------------------------------------------------------------------------
# Public tool functions
# ---------------------------------------------------------------------------

def list_workflow_runs(
    workflow_id: str = "e2e-tests.yml",
    branch: str | None = None,
    limit: int = 10,
) -> list[dict]:
    """Return a list of recent workflow runs (most recent first).

    Args:
        workflow_id: filename or numeric ID of the workflow (e.g. 'e2e-tests.yml').
        branch:      filter by branch name (None = all branches).
        limit:       max number of runs to return (max 100).

    Returns:
        List of dicts with keys: id, name, status, conclusion, branch,
        created_at, updated_at, html_url, run_number.
    """
    params: dict = {"per_page": min(limit, 100)}
    if branch:
        params["branch"] = branch

    data = _gh_get(f"actions/workflows/{workflow_id}/runs", params)
    runs = data.get("workflow_runs", [])[:limit]
    return [
        {
            "id": r["id"],
            "run_number": r["run_number"],
            "name": r.get("display_title") or r.get("name", ""),
            "status": r["status"],           # queued | in_progress | completed
            "conclusion": r.get("conclusion"),  # success | failure | cancelled | None
            "branch": r["head_branch"],
            "created_at": _fmt_dt(r.get("created_at")),
            "updated_at": _fmt_dt(r.get("updated_at")),
            "html_url": r["html_url"],
        }
        for r in runs
    ]


def get_run_status(run_id: int) -> dict:
    """Return status and conclusion of a single workflow run.

    Args:
        run_id: numeric GitHub Actions run ID.

    Returns:
        Dict with keys: id, status, conclusion, branch, created_at, updated_at,
        html_url, jobs_summary (list of job name/status/conclusion).
    """
    run = _gh_get(f"actions/runs/{run_id}")
    jobs_data = _gh_get(f"actions/runs/{run_id}/jobs")
    jobs = [
        {
            "name": j["name"],
            "status": j["status"],
            "conclusion": j.get("conclusion"),
            "started_at": _fmt_dt(j.get("started_at")),
            "completed_at": _fmt_dt(j.get("completed_at")),
        }
        for j in jobs_data.get("jobs", [])
    ]
    return {
        "id": run["id"],
        "run_number": run["run_number"],
        "status": run["status"],
        "conclusion": run.get("conclusion"),
        "branch": run["head_branch"],
        "created_at": _fmt_dt(run.get("created_at")),
        "updated_at": _fmt_dt(run.get("updated_at")),
        "html_url": run["html_url"],
        "jobs": jobs,
    }


def get_run_logs(run_id: int, job_name_filter: str | None = None) -> str:
    """Download and return the text logs of a workflow run.

    GitHub provides logs as a ZIP archive. This function extracts and
    concatenates the log files, optionally filtered by job name.

    Args:
        run_id:           numeric GitHub Actions run ID.
        job_name_filter:  if set, only include log files whose path contains
                          this string (case-insensitive). Useful to focus on
                          a specific job (e.g. 'get-jobs', 'agentic').

    Returns:
        Plain-text log content (may be large; consider slicing before display).
    """
    url = f"https://api.github.com/repos/{cfg.GITHUB_REPO}/actions/runs/{run_id}/logs"
    resp = requests.get(url, headers=_gh_headers(), timeout=60, allow_redirects=True)
    resp.raise_for_status()

    logs_text: list[str] = []
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        names = sorted(zf.namelist())
        for name in names:
            if job_name_filter and job_name_filter.lower() not in name.lower():
                continue
            if name.endswith("/"):
                continue
            try:
                content = zf.read(name).decode("utf-8", errors="replace")
                logs_text.append(f"=== {name} ===\n{content}")
            except Exception:
                pass

    return "\n\n".join(logs_text) if logs_text else "(no logs found)"


def get_job_logs(job_id: int) -> str:
    """Download logs for a single job (lighter than the full run zip).

    Args:
        job_id: numeric GitHub Actions job ID (visible in get_run_status output).

    Returns:
        Plain-text log lines.
    """
    url = f"https://api.github.com/repos/{cfg.GITHUB_REPO}/actions/jobs/{job_id}/logs"
    resp = requests.get(url, headers=_gh_headers(), timeout=30, allow_redirects=True)
    resp.raise_for_status()
    return resp.text


def trigger_workflow(
    workflow_id: str,
    ref: str,
    inputs: dict[str, str] | None = None,
) -> dict:
    """Trigger a workflow_dispatch event.

    Args:
        workflow_id: workflow filename (e.g. 'e2e-tests.yml').
        ref:         branch or tag to run on.
        inputs:      key-value pairs matching the workflow's `on.workflow_dispatch.inputs`.

    Returns:
        Dict with 'status' and 'message'.
    """
    payload: dict = {"ref": ref, "inputs": inputs or {}}
    _gh_post(f"actions/workflows/{workflow_id}/dispatches", payload)
    return {
        "status": "triggered",
        "message": f"Workflow '{workflow_id}' dispatched on ref '{ref}'.",
    }


def _filter_log_lines(raw: str, max_chars: int = 60_000) -> str:
    """Keep log lines that are likely meaningful: errors, warnings, key events.

    Strips the GitHub Actions timestamp prefix (e.g. '2025-06-29T12:34:56.000Z ')
    and returns a deduplicated, truncated view.
    """
    # Strip GitHub Actions UTC timestamp prefix
    ts_re = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+Z\s*")
    lines = []
    for line in raw.splitlines():
        lines.append(ts_re.sub("", line))

    # Heuristic priority filter: keep error/warning/key lines + surrounding context
    important_re = re.compile(
        r"(error|warning|fail|exception|traceback|assert|critical"
        r"|benchmark|throughput|tokens/s|ttft|tpot|conc|isl|osl"
        r"|starting|finished|completed|success|exit code"
        r"|validation|pydantic|config|container)",
        re.IGNORECASE,
    )
    kept: list[str] = []
    for i, line in enumerate(lines):
        if important_re.search(line):
            # Include 1 line of context before and after
            start = max(0, i - 1)
            end = min(len(lines), i + 2)
            for j in range(start, end):
                if j not in {k for k in range(start, end) if lines[k] in kept}:
                    kept.append(lines[j])

    # If nothing matched heuristics, just return all lines
    result = "\n".join(kept) if kept else "\n".join(lines)
    if len(result) > max_chars:
        result = result[:max_chars] + "\n[... truncated ...]"
    return result


def watch_run(
    run_id: int,
    poll_interval: int = 30,
    job_filter: str | None = None,
    print_fn: Any = print,
) -> str:
    """Monitor a GitHub Actions run until it completes, printing live log updates.

    For each job that completes (or makes progress), downloads its logs, filters
    them for relevant lines, and asks Claude for a concise interpretation.
    Prints updates as they arrive. Returns a final summary when the run is done.

    Args:
        run_id:        numeric GitHub Actions run ID.
        poll_interval: seconds between polls (default 30).
        job_filter:    optional substring — only monitor jobs whose name contains
                       this string (case-insensitive).
        print_fn:      callable used for live output (default: built-in print).
                       In chat.py this is wired to the TeeConsole.

    Returns:
        A final summary string describing the run outcome.
    """
    print_fn(f"\n[watch_run] Monitoring run #{run_id} — polling every {poll_interval}s. Ctrl-C to stop.\n")

    # Track per-job state: last known line count of logs
    job_log_lines: dict[int, int] = {}   # job_id -> lines already printed
    job_status_seen: dict[int, str] = {} # job_id -> last status string

    run_url = f"https://github.com/{cfg.GITHUB_REPO}/actions/runs/{run_id}"
    client = cfg.make_anthropic_client()

    def _interpret(log_snippet: str, job_name: str, status: str) -> str:
        truncated = log_snippet[:50_000]
        msg = client.messages.create(
            model=cfg.CLAUDE_MODEL,
            max_tokens=512,
            messages=[{
                "role": "user",
                "content": (
                    f"These are recent log lines from GitHub Actions job '{job_name}' "
                    f"(status: {status}). "
                    "In 3-5 bullet points, summarize what is happening: "
                    "errors, progress milestones, benchmark results, or why it failed. "
                    "Be concise. Omit boilerplate.\n\n"
                    f"```\n{truncated}\n```"
                ),
            }],
        )
        return msg.content[0].text.strip()

    try:
        while True:
            # --- fetch current run state ---
            try:
                run_data = _gh_get(f"actions/runs/{run_id}")
                jobs_data = _gh_get(f"actions/runs/{run_id}/jobs")
            except requests.HTTPError as exc:
                print_fn(f"[watch_run] API error: {exc}. Retrying in {poll_interval}s…")
                time.sleep(poll_interval)
                continue

            run_status = run_data.get("status", "unknown")
            run_conclusion = run_data.get("conclusion")
            jobs = jobs_data.get("jobs", [])

            # --- process each job ---
            for job in jobs:
                jid = job["id"]
                jname = job["name"]
                jstatus = job.get("status", "")
                jconclusion = job.get("conclusion")

                if job_filter and job_filter.lower() not in jname.lower():
                    continue

                # Detect status change
                prev_status = job_status_seen.get(jid)
                if prev_status != jstatus:
                    job_status_seen[jid] = jstatus
                    icon = {"queued": "⏳", "in_progress": "🔄", "completed": "✅" if jconclusion == "success" else "❌"}.get(jstatus, "•")
                    print_fn(f"  {icon} [{jname}] {jstatus.upper()}" + (f" → {jconclusion}" if jconclusion else ""))

                # Download and diff logs for in_progress or just-completed jobs
                if jstatus in ("in_progress", "completed"):
                    try:
                        url = f"https://api.github.com/repos/{cfg.GITHUB_REPO}/actions/jobs/{jid}/logs"
                        resp = requests.get(url, headers=_gh_headers(), timeout=30, allow_redirects=True)
                        if resp.status_code == 200:
                            raw = resp.text
                            filtered = _filter_log_lines(raw)
                            all_lines = filtered.splitlines()
                            prev_count = job_log_lines.get(jid, 0)
                            new_lines = all_lines[prev_count:]
                            job_log_lines[jid] = len(all_lines)

                            if new_lines:
                                snippet = "\n".join(new_lines)
                                print_fn(f"\n── {jname} (new log lines) ──")
                                # Show raw new lines
                                for ln in new_lines[-40:]:  # cap at last 40 new lines shown raw
                                    print_fn(f"  {ln}")
                                # Interpret with Claude if there's enough to say
                                if len(snippet) > 200:
                                    try:
                                        interp = _interpret(snippet, jname, jstatus)
                                        print_fn(f"\n  🤖 Claude:\n{interp}\n")
                                    except Exception as exc:
                                        print_fn(f"  [interpret error: {exc}]")
                    except Exception as exc:
                        print_fn(f"  [log fetch error for {jname}: {exc}]")

            # --- check if run is done ---
            if run_status == "completed":
                icon = "✅" if run_conclusion == "success" else "❌"
                summary_lines = [
                    f"\n{icon} Run #{run_id} completed — conclusion: {run_conclusion}",
                    f"   URL: {run_url}",
                    "\nFinal job summary:",
                ]
                for job in jobs:
                    jname = job["name"]
                    jstatus = job.get("status", "")
                    jconclusion = job.get("conclusion", "")
                    if job_filter and job_filter.lower() not in jname.lower():
                        continue
                    j_icon = "✅" if jconclusion == "success" else ("❌" if jconclusion else "•")
                    summary_lines.append(f"  {j_icon} {jname}: {jstatus} / {jconclusion or '—'}")

                summary = "\n".join(summary_lines)
                print_fn(summary)
                return summary

            print_fn(f"\n  [watch_run] Run status: {run_status} — next poll in {poll_interval}s…")
            time.sleep(poll_interval)

    except KeyboardInterrupt:
        msg = f"\n[watch_run] Monitoring interrupted by user. Run #{run_id}: {run_url}"
        print_fn(msg)
        return msg


def interpret_logs(log_text: str, question: str = "Summarize errors and status.") -> str:
    """Ask Claude to interpret a log excerpt.

    Args:
        log_text:  raw log text (will be truncated to ~80 k chars to fit context).
        question:  what to ask about the logs.

    Returns:
        Claude's plain-text answer.
    """
    client = cfg.make_anthropic_client()
    truncated = log_text[:80_000]
    if len(log_text) > 80_000:
        truncated += "\n\n[... log truncated ...]"

    msg = client.messages.create(
        model=cfg.CLAUDE_MODEL,
        max_tokens=cfg.MAX_TOKENS,
        messages=[
            {
                "role": "user",
                "content": (
                    f"Below are GitHub Actions workflow logs. {question}\n\n"
                    f"```\n{truncated}\n```"
                ),
            }
        ],
    )
    return msg.content[0].text

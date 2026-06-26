"""
GitHub Actions tools for the InferenceMAX agentic orchestrator.

Provides functions to:
  - list_workflow_runs     list recent runs of a workflow
  - get_run_status         get status/conclusion of a specific run
  - get_run_logs           download and return logs of a run (all jobs or one job)
  - trigger_workflow       dispatch a workflow_dispatch event
  - interpret_logs         ask Claude to summarize/interpret log text

All GitHub calls use the REST API via the `requests` library with the
GITHUB_TOKEN set in config.py / .env.
"""

from __future__ import annotations

import json
import re
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

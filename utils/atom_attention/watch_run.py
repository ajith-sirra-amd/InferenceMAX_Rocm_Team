"""Poll a GitHub run and report current-attempt steps and failure annotations."""
import argparse
import datetime
import json
import os
import subprocess
import time

import requests

try:
    import truststore
except ImportError:
    pass
else:
    truststore.inject_into_ssl()


def watch(session, repo, run_id, interval, max_seconds):
    base = f"https://api.github.com/repos/{repo}"
    deadline = time.monotonic() + max_seconds
    while time.monotonic() < deadline:
        try:
            response = session.get(f"{base}/actions/runs/{run_id}", timeout=30)
            response.raise_for_status()
            run = response.json()
            response = session.get(
                f"{base}/actions/runs/{run_id}/attempts/{run['run_attempt']}/jobs",
                params={"per_page": 100}, timeout=30,
            )
            response.raise_for_status()
            jobs = response.json()["jobs"]
            report = {
                "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "run_id": run_id, "attempt": run["run_attempt"],
                "status": run["status"], "conclusion": run["conclusion"],
                "jobs": [
                    {
                        "id": job["id"], "name": job["name"],
                        "status": job["status"], "conclusion": job["conclusion"],
                        "active_steps": [
                            step["name"] for step in job["steps"]
                            if step["status"] == "in_progress"
                        ],
                    }
                    for job in jobs if job["conclusion"] != "skipped"
                ],
            }
            print(json.dumps(report), flush=True)
            if run["status"] == "completed":
                if run["conclusion"] != "success":
                    for job in jobs:
                        if job["conclusion"] != "failure":
                            continue
                        check_id = job["check_run_url"].rsplit("/", 1)[-1]
                        response = session.get(
                            f"{base}/check-runs/{check_id}/annotations", timeout=30,
                        )
                        response.raise_for_status()
                        for annotation in response.json():
                            print("FAILURE_REASON", annotation["message"], flush=True)
                return 0 if run["conclusion"] == "success" else 1
        except requests.RequestException as error:
            # Transient API/network errors do not end monitoring.
            print("CHECKER_RETRY", type(error).__name__, flush=True)
        time.sleep(interval)
    print("CHECKER_DEADLINE_REACHED", flush=True)
    return 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--run-id", required=True, type=int)
    parser.add_argument("--interval", required=True, type=int)
    parser.add_argument("--max-hours", required=True, type=float)
    args = parser.parse_args()
    if not 1 <= args.interval <= 60 or args.max_hours <= 0:
        parser.error("interval must be 1..60 seconds and max-hours must be positive")
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if not token:
        token = subprocess.run(
            ["gh", "auth", "token"], capture_output=True, text=True, check=True,
        ).stdout.strip()
    session = requests.Session()
    session.headers["Authorization"] = "Bearer " + token
    raise SystemExit(watch(
        session, args.repo, args.run_id, args.interval, args.max_hours * 3600,
    ))

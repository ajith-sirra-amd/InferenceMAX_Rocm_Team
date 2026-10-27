#!/usr/bin/env python3
"""Poll a GitHub Actions run. The active Codex turn consumes this output."""
import argparse
import datetime
import json
import subprocess
import sys
import time


def github_json(repo, path):
    result = subprocess.run(
        ["gh", "api", f"repos/{repo}/{path}"],
        capture_output=True, text=True, timeout=60, check=True,
    )
    return json.loads(result.stdout)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--run-id", required=True, type=int)
    parser.add_argument("--interval-seconds", required=True, type=int)
    parser.add_argument("--watch-seconds", required=True, type=int)
    args = parser.parse_args()
    if args.interval_seconds < 1 or args.watch_seconds < 1:
        parser.error("interval-seconds and watch-seconds must be positive")
    deadline = time.monotonic() + args.watch_seconds
    while time.monotonic() < deadline:
        try:
            run = github_json(args.repo, f"actions/runs/{args.run_id}")
            jobs = github_json(args.repo, f"actions/runs/{args.run_id}/jobs?per_page=100")
            print(json.dumps({
                "time": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "run_id": args.run_id,
                "url": run["html_url"],
                "status": run["status"],
                "conclusion": run["conclusion"],
                "head_sha": run["head_sha"],
                "jobs": [
                    {
                        "id": job["id"],
                        "status": job["status"],
                        "conclusion": job["conclusion"],
                        "runner": job["runner_name"],
                        "active_steps": [
                            step["name"] for step in job.get("steps", [])
                            if step["status"] == "in_progress"
                        ],
                    }
                    for job in jobs["jobs"] if job["conclusion"] != "skipped"
                ],
            }), flush=True)
            if run["status"] == "completed":
                return 0 if run["conclusion"] == "success" else 1
        except (subprocess.SubprocessError, ValueError, KeyError) as error:
            # Report transient failures and retry; credentials stay inside gh.
            print(json.dumps({
                "time": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "run_id": args.run_id,
                "poll_error": str(error),
            }), file=sys.stderr, flush=True)
        next_poll = min(time.monotonic() + args.interval_seconds, deadline)
        while time.monotonic() < next_poll:
            time.sleep(min(60, next_poll - time.monotonic()))
    print(json.dumps({"run_id": args.run_id, "watch_window_finished": True}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

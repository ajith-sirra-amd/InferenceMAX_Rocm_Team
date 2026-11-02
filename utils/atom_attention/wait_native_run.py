"""Wait for a specific GitHub Actions revision without reserving a GPU runner."""

import argparse
import json
import re
import subprocess
import time


def get_run(repo, run_id):
    return json.loads(subprocess.check_output(
        ["gh", "api", f"repos/{repo}/actions/runs/{run_id}"]
    ))


def wait_for_run(repo, run_id, sha, timeout_seconds, poll_seconds):
    if run_id <= 0 or not re.fullmatch(r"[0-9a-f]{40}", sha):
        raise ValueError("Expected a positive run ID and full source SHA")
    if timeout_seconds <= 0 or poll_seconds <= 0:
        raise ValueError("Timing values must be positive")
    deadline = time.monotonic() + timeout_seconds
    errors = 0
    while time.monotonic() < deadline:
        try:
            run = get_run(repo, run_id)
        except (subprocess.CalledProcessError, OSError, json.JSONDecodeError):
            errors += 1
            if errors >= 5:
                raise
            print(f"Run {run_id}: API request failed, retry {errors}/5", flush=True)
        else:
            errors = 0
            if run["head_sha"] != sha:
                raise RuntimeError("Predecessor run has a different source SHA")
            print(json.dumps({
                "run": run_id, "status": run["status"],
                "conclusion": run["conclusion"],
            }), flush=True)
            if run["status"] == "completed":
                if run["conclusion"] != "success":
                    raise RuntimeError(
                        f"Predecessor run ended with {run['conclusion']}"
                    )
                return
        time.sleep(min(poll_seconds, max(0, deadline - time.monotonic())))
    raise TimeoutError("Predecessor run did not complete within the queue deadline")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--run-id", type=int, required=True)
    parser.add_argument("--sha", required=True)
    parser.add_argument("--timeout-seconds", type=int, required=True)
    parser.add_argument("--poll-seconds", type=int, required=True)
    wait_for_run(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()

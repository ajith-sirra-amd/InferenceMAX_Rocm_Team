"""Stop only registered run containers mounted to this runner workspace."""
import json
import os
from pathlib import Path
import re
import subprocess


def selected(name, run_ids):
    patterns = (
        r"pr64288-(?:review|dspark-baseline)-(\d+)",
        r"bmk-server-(\d+)-\d+-mi355x-amd_[A-Za-z0-9_-]+",
    )
    return any((match := re.fullmatch(pattern, name)) and int(match[1]) in run_ids
               for pattern in patterns)


def main():
    raw = json.loads(os.environ["RESERVATION_RUN_IDS"])
    if not isinstance(raw, list) or not raw or any(type(x) is not int or x <= 0 for x in raw):
        raise RuntimeError("Expected explicit positive run IDs")
    run_ids = set(raw)
    workspace = Path(os.environ["GITHUB_WORKSPACE"]).resolve(strict=True)
    names = subprocess.check_output(["docker", "ps", "--format", "{{.Names}}"], text=True).splitlines()
    stopped = []
    for name in names:
        if not selected(name, run_ids):
            continue
        mounts = json.loads(subprocess.check_output(
            ["docker", "inspect", "--format", "{{json .Mounts}}", name], text=True))
        if not any(Path(mount["Source"]).resolve() == workspace for mount in mounts):
            raise RuntimeError(f"Refusing to stop container without the expected repository mount: {name}")
        subprocess.run(["docker", "stop", "--time", "30", name], check=True, timeout=45)
        stopped.append(name)
    remaining = subprocess.check_output(["docker", "ps", "--format", "{{.Names}}"], text=True).splitlines()
    remaining = [name for name in remaining if selected(name, run_ids)]
    report = {"registered_runs": sorted(run_ids), "stopped": stopped, "remaining": remaining}
    print(json.dumps(report), flush=True)
    Path(os.environ["GITHUB_STEP_SUMMARY"]).write_text(
        "Reservation cleanup: " + json.dumps(report) + "\n", encoding="utf-8")
    if remaining:
        raise RuntimeError("Registered containers are still running")


if __name__ == "__main__":
    main()

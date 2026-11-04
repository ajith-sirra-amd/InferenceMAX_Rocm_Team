"""Run one frozen IPAT request on a local MI355X Docker host."""
import argparse
import datetime as dt
import json
import os
from pathlib import Path
import re
import shutil
import subprocess


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")


def capture(command, target, timeout=60):
    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, timeout=timeout)
        target.write_text(result.stdout)
        return {"command": command, "returncode": result.returncode, "artifact": target.name}
    except (OSError, subprocess.TimeoutExpired) as error:
        target.write_text(str(error) + "\n")
        return {"command": command, "error": str(error), "artifact": target.name}


def preflight(request, output):
    output.mkdir(parents=True, exist_ok=True)
    facts = {"run_id": request["id"], "mode": "preflight", "status": "completed",
             "at": dt.datetime.now(dt.timezone.utc).isoformat(),
             "runner": os.environ.get("RUNNER_NAME"), "hostname": os.uname().nodename,
             "workflow_run_id": os.environ.get("GITHUB_RUN_ID"),
             "workflow_attempt": os.environ.get("GITHUB_RUN_ATTEMPT"),
             "execution_sha": os.environ.get("GITHUB_SHA"), "commands": []}
    commands = {
        "docker-version.txt": ["docker", "version", "--format", "{{json .}}"],
        "containers.txt": ["docker", "ps", "--format", "{{.ID}} {{.Names}} {{.Image}} {{.Status}}"],
        "images.txt": ["docker", "image", "ls", "--format", "{{.Repository}}:{{.Tag}} {{.ID}} {{.Size}}"],
        "gpu-list.txt": ["amd-smi", "list", "--json"],
        "gpu-static.txt": ["amd-smi", "static", "--json"],
        "rocm-smi.txt": ["rocm-smi", "--showproductname", "--showmeminfo", "vram", "--json"],
        "memory.txt": ["free", "-b"],
        "disk.txt": ["df", "-h", str(output)],
    }
    for name, command in commands.items():
        facts["commands"].append(capture(command, output / name))
    facts["paths"] = {}
    for directory in ("/data/hf_hub_cache", "/home/models", "/share_nfs/models", "/data/models",
                      "/it-share/models", "/wrk/dcgmktg_bench_xhd/ajith"):
        path = Path(directory)
        row = {"exists": path.is_dir(), "writable": os.access(path, os.W_OK)}
        if path.is_dir():
            try:
                row["entries"] = sorted(p.name for p in path.iterdir())[:100]
            except OSError as error:
                row["error"] = str(error)
        facts["paths"][directory] = row
    facts["tools"] = {name: shutil.which(name) for name in ("docker", "rocprofv3", "rocprof-compute", "amd-smi", "srun")}
    write_json(output / "manifest.json", facts)
    print(json.dumps(facts, indent=2))
    return facts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    request = json.loads(os.environ["IPAT_REQUEST"])
    if not re.fullmatch(r"[0-9a-f]{32}", request.get("id", "")):
        raise ValueError("IPAT request needs a unique hexadecimal ID")
    write_json(args.output / "request.json", request)
    if request.get("mode") == "preflight":
        preflight(request, args.output)
    else:
        raise ValueError("Unsupported IPAT request mode")


if __name__ == "__main__":
    main()

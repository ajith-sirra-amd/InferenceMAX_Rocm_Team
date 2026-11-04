"""Run one frozen IPAT request on a local MI355X Docker host."""
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import socket
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
    inventory = request.get("model_inventory")
    if inventory:
        cache = Path(inventory["cache"]).resolve()
        facts["model_inventory"] = {}
        for relative in inventory["paths"]:
            path = (cache / relative).resolve()
            if not path.is_relative_to(cache): raise ValueError("Model inventory must stay in its cache")
            row = {"path": str(path), "exists": path.is_dir()}
            if path.is_dir():
                row["entries"] = [{"name": p.name, "is_file": p.is_file(), "bytes": p.stat().st_size if p.is_file() else None} for p in sorted(path.iterdir())][:2000]
                for file in path.glob("*.json"):
                    if file.stat().st_size < 4_000_000:
                        dest = output / "model-metadata" / relative / file.name
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copyfile(file, dest)
            facts["model_inventory"][relative] = row
    if request.get("image_check"):
        assert_gpu_idle()
        settings = request["image_check"]
        image = json.loads(subprocess.check_output(["docker", "image", "inspect", settings["image"]], text=True))[0]
        facts["image"] = {"requested": settings["image"], "id": image["Id"],
                          "repo_digests": image.get("RepoDigests", []),
                          "entrypoint": image["Config"].get("Entrypoint"), "cmd": image["Config"].get("Cmd")}
        write_json(output / "manifest.json", facts)
        for part in ("home", "tmp", "cache", "hf"):
            (output / part).mkdir(exist_ok=True)
        command = ["docker", "run", "--rm", "--init", "--name", "ipat-" + request["id"],
                   "--network", "host", "--ipc", "host", "--device", "/dev/kfd", "--device", "/dev/dri",
                   "--security-opt", "seccomp=unconfined", "--ulimit", "memlock=-1",
                   "--mount", f"type=bind,src={output.resolve()},dst=/output",
                   "--mount", f"type=bind,src={Path(__file__).resolve().parent},dst=/ipat,readonly",
                   "--workdir", "/output", "--env", "HOME=/output/home", "--env", "TMPDIR=/output/tmp",
                   "--env", "XDG_CACHE_HOME=/output/cache", "--env", "HF_HOME=/output/hf",
                   "--env", "PYTHONDONTWRITEBYTECODE=1"]
        for key, value in settings.get("environment", {}).items():
            if key in ("HOME", "TMPDIR", "XDG_CACHE_HOME", "HF_HOME", "HF_TOKEN"):
                raise ValueError("Image diagnostic cannot override output or credential paths")
            command += ["--env", f"{key}={value}"]
        command += ["--entrypoint", "python3", image["Id"], "/ipat/check_image.py"]
        facts["image_check"] = capture(command, output / "image-check.log", timeout=600)
        if facts["image_check"].get("returncode") != 0:
            facts["status"] = "failed"
    write_json(output / "manifest.json", facts)
    print(json.dumps(facts, indent=2))
    return facts


def assert_gpu_idle():
    active = subprocess.check_output(["docker", "ps", "-q"], text=True).split()
    if active:
        containers = json.loads(subprocess.check_output(["docker", "inspect", *active], text=True))
        busy = [c["Name"] for c in containers if any(d.get("PathOnHost") == "/dev/kfd" for d in (c["HostConfig"].get("Devices") or [])) or c["HostConfig"].get("Privileged")]
        if busy: raise RuntimeError("GPU containers already occupy this host: " + ", ".join(busy))


def validate_recipe(recipe):
    for key in ("server_argv", "client_argv"):
        if not isinstance(recipe.get(key), list) or not recipe[key] or any(not isinstance(v, str) or "\0" in v for v in recipe[key]):
            raise ValueError(f"{key} must be a nonempty argument array")
    for key in ("physical_chips", "concurrency", "duration_seconds", "startup_timeout_seconds", "client_timeout_seconds"):
        value = recipe.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{key} must be a positive integer")
    if recipe["physical_chips"] > 8:
        raise ValueError("This runner supports one host with up to eight GPUs")
    if recipe.get("image_policy") not in ("pull", "local") or not isinstance(recipe.get("image"), str):
        raise ValueError("Specify an image and an explicit pull/local image policy")
    reserved = {"HOME", "TMPDIR", "HF_HOME", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "XDG_CACHE_HOME", "PIP_CACHE_DIR"}
    for env in (recipe.get("environment"), recipe.get("client_environment", {})):
        if not isinstance(env, dict) or any(not isinstance(v, str) for v in env.values()):
            raise ValueError("Environment values must be strings")
        if reserved.intersection(env):
            raise ValueError("Cache paths and credentials are supplied by the runner")
    if not isinstance(recipe.get("client_packages"), list) or any(not re.fullmatch(r"[A-Za-z0-9_.-]+==[A-Za-z0-9_.+-]+", p) for p in recipe["client_packages"]):
        raise ValueError("Client packages require exact package==version pins")
    if recipe.get("client_python") and not re.fullmatch(r"3\.[0-9]+(?:\.[0-9]+)?", recipe["client_python"]):
        raise ValueError("Specify the client's Python 3 version")
    setup = recipe.get("client_setup", [])
    if not isinstance(setup, list) or any(not isinstance(argv, list) or not argv or any(not isinstance(arg, str) or "\0" in arg for arg in argv) for argv in setup):
        raise ValueError("client_setup must contain argument arrays")
    profile = recipe.get("profile", {})
    if profile.get("kind") not in ("torch", "rocprof"):
        raise ValueError("Choose torch or rocprof profiling")
    for key, minimum, maximum in (("seconds", 1, 120), ("delay_seconds", 0, 3600)):
        if type(profile.get(key)) is not int or not minimum <= profile[key] <= maximum:
            raise ValueError(f"Invalid profile {key}")
    if recipe.get("framework") not in ("vllm", "sglang", "generic"):
        raise ValueError("Choose vllm, sglang or generic")
    if not isinstance(recipe.get("health_path"), str) or not recipe["health_path"].startswith("/"):
        raise ValueError("health_path must be an HTTP path")
    return recipe


def resolve_models(recipe, cache):
    """Resolve existing cache snapshots on the host without downloading weights."""
    facts = {}
    for model, relative in recipe.get("model_paths", {}).items():
        path = cache / relative
        ref = path / "refs/main"
        if ref.is_file():
            path = path / "snapshots" / ref.read_text().strip()
        path = path.resolve()
        if not path.is_relative_to(cache.resolve()) or not (path / "config.json").is_file():
            raise ValueError(f"No readable model config inside the selected cache: {model} ({relative})")
        indices = list(path.glob("*.index.json"))
        indexed = {name for index in indices for name in json.loads(index.read_text()).get("weight_map", {}).values()}
        weights = indexed or {p.name for pattern in ("*.safetensors", "*.bin", "*.pt", "*.pth", "*.gguf") for p in path.glob(pattern)}
        if not weights or any(not (path / weight).is_file() for weight in weights):
            raise ValueError(f"Model cache is missing weights or indexed shards: {model} ({relative})")
        target = "/model-cache/" + str(path.relative_to(cache.resolve()))
        for key in ("server_argv", "client_argv"):
            recipe[key] = [arg.replace("/model-cache/" + relative, target) for arg in recipe[key]]
        metadata = {}
        for file in sorted(path.glob("*.json")):
            metadata[file.name] = hashlib.sha256(file.read_bytes()).hexdigest()
        facts[model] = {"path": str(path), "snapshot": path.name if path.parent.name == "snapshots" else None,
                        "local_revision_marker": (path / ".hf-revision").read_text().strip() if (path / ".hf-revision").is_file() else None,
                        "metadata_sha256": metadata, "weight_files": len(weights), "weights_hashed": False}
    return facts


def report(output, manifest):
    lines = ["# IPAT replication and profiling", "", f"Status: {manifest['status']}",
             f"Config: {manifest.get('config_id')} · Hash: {manifest.get('config_sha256')}",
             f"Runner: {manifest.get('runner')} · Execution commit: {manifest.get('execution_sha')}", "",
             "| Pass | Status | Output tok/s/GPU | P90 full-response ITL (ms) | Completed requests | Failed requests |",
             "| --- | --- | --- | --- | --- | --- |"]
    for record in manifest["passes"]:
        metrics = record.get("metrics", {})
        values = [record["pass"], record["status"], metrics.get("output_tokens_s_gpu"), metrics.get("p90_full_response_itl_ms"),
                  metrics.get("completed_requests"), metrics.get("failed_requests")]
        lines.append("| " + " | ".join(str(v) if v is not None else "Unknown" for v in values) + " |")
    lines += ["", "Only the unprofiled baseline measures baseline serving performance. Profiling timings are diagnostic.",
              "Quality evaluation, measured HBM counters, full tensor-flow attribution and roofline calibration are not collected by this initial path.",
              "Original SA model/dataset revisions and image identity may be unresolved; see the frozen source provenance.",
              "Raw client outputs, server logs, capabilities, per-pass manifests and native traces are retained beside this report."]
    if manifest.get("error"): lines += ["", "Failure: " + manifest["error"]]
    (output / "report.md").write_text("\n".join(lines) + "\n")
    (output / "index.html").write_text('<!doctype html><meta charset="utf-8"><title>IPAT run evidence</title><h1>IPAT run evidence</h1>'
        '<ul><li><a href="report.md">Replication and profiling report</a></li><li><a href="manifest.json">Run manifest</a></li>'
        '<li><a href="request.json">Frozen request</a></li><li><a href="recipe.json">Effective recipe</a></li>'
        '<li><a href="baseline/">Unprofiled baseline</a></li><li><a href="profile/">Profile and native traces</a></li></ul>')


def cleanup(output):
    """Independent post-step: cancellation may terminate the main Python process."""
    path = output / "request.json"
    if not path.exists(): return
    request = json.loads(path.read_text())
    if not re.fullmatch(r"[0-9a-f]{32}", request.get("id", "")): raise ValueError("Invalid cleanup identity")
    subprocess.run(["docker", "rm", "--force", "ipat-" + request["id"]], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30)
    manifest = output / "manifest.json"
    image = json.loads(manifest.read_text()).get("image", {}).get("id") if manifest.exists() else None
    if image:
        subprocess.run(["docker", "run", "--rm", "--mount", f"type=bind,src={output.resolve()},dst=/output",
                        "--entrypoint", "chown", image, "-R", f"{os.getuid()}:{os.getgid()}", "/output"], timeout=120, check=True)


def recover(request, output):
    """Recover only a previous IPAT checkout's evidence after interrupted cleanup."""
    previous = request["previous"]
    run_id, attempt = str(previous["workflow_run_id"]), str(previous["attempt"])
    if not run_id.isdigit() or not attempt.isdigit(): raise ValueError("Invalid previous workflow identity")
    repo = Path(__file__).resolve().parents[1]
    old = repo.parent / f"ipat-{run_id}-{attempt}"
    origin = subprocess.check_output(["git", "-C", str(old), "config", "--get", "remote.origin.url"], text=True).strip()
    if origin.removesuffix(".git") != "https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team":
        raise ValueError("Recovery target must be a team repository checkout")
    prior_request = json.loads((old / "evidence/request.json").read_text())
    if prior_request["id"] != previous["id"]: raise ValueError("Recovery request identity differs")
    cleanup(old / "evidence")
    shutil.copytree(old / "evidence", output / "recovered", ignore=shutil.ignore_patterns("client-env", "cache", "hf", "home", "tmp"))
    write_json(output / "manifest.json", {"status": "recovered", "previous": previous})


def execute(request, output):
    recipe = validate_recipe(request["recipe"])
    repo = Path(__file__).resolve().parents[1]
    if not output.resolve().is_relative_to(repo):
        raise ValueError("Run output must be inside the InferenceMAX_Rocm_Team checkout")
    output.mkdir(parents=True, exist_ok=True)
    name = "ipat-" + request["id"]
    manifest = {"id": request["id"], "kind": "run", "status": "preparing", "started_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                "config_id": request.get("config_id"), "config_sha256": request.get("config_sha256"),
                "recipe_sha256": hashlib.sha256(json.dumps(recipe, sort_keys=True).encode()).hexdigest(),
                "runner": os.environ.get("RUNNER_NAME"), "execution_sha": os.environ.get("GITHUB_SHA"),
                "workflow_run_id": os.environ.get("GITHUB_RUN_ID"), "workflow_attempt": os.environ.get("GITHUB_RUN_ATTEMPT"),
                "passes": [], "source": request.get("source")}
    image_id = None
    def save(): write_json(output / "manifest.json", manifest)
    def interrupted(signum, frame): raise InterruptedError("IPAT run interrupted")
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    save()
    try:
        preflight(request, output / "preflight")
        assert_gpu_idle()
        if recipe["image_policy"] == "pull":
            result = capture(["docker", "pull", recipe["image"]], output / "image-pull.log", timeout=1800)
            if result.get("returncode") != 0: raise RuntimeError("Image pull failed; see image-pull.log")
        image = json.loads(subprocess.check_output(["docker", "image", "inspect", recipe["image"]], text=True))[0]
        image_id = image["Id"]
        manifest["image"] = {"requested": recipe["image"], "id": image_id, "repo_digests": image.get("RepoDigests", [])}
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0)); recipe = dict(recipe, port=sock.getsockname()[1])
        model_cache = Path(recipe["model_cache"])
        if not model_cache.is_absolute() or not model_cache.is_dir():
            raise ValueError("The selected model_cache directory does not exist on this runner")
        manifest["models"] = resolve_models(recipe, model_cache)
        mounts = ["--mount", f"type=bind,src={output.resolve()},dst=/output",
                  "--mount", f"type=bind,src={repo / 'ipat'},dst=/ipat,readonly",
                  "--mount", f"type=bind,src={model_cache},dst=/model-cache,readonly"]
        write_json(output / "recipe.json", recipe)
        for part in ("home", "tmp", "cache", "hf"):
            (output / part).mkdir(exist_ok=True)
        env = {"HOME": "/output/home", "TMPDIR": "/output/tmp", "XDG_CACHE_HOME": "/output/cache",
               "PIP_CACHE_DIR": "/output/cache/pip", "HF_HOME": "/output/hf", "PYTHONDONTWRITEBYTECODE": "1"}
        common = ["docker", "run", "--rm", "--init", "--name", name, "--label", "ipat.run=" + request["id"],
                  "--network", "host", "--ipc", "host", "--device", "/dev/kfd", "--device", "/dev/dri",
                  "--cap-add", "SYS_PTRACE", "--security-opt", "seccomp=unconfined", "--ulimit", "memlock=-1", "--ulimit", "stack=67108864", *mounts,
                  "--workdir", "/output", "--env", "HF_TOKEN"]
        for key, value in env.items(): common += ["--env", f"{key}={value}"]
        passes = request.get("passes", ["baseline", "profile"])
        if not passes or any(p not in ("baseline", "profile") for p in passes) or len(set(passes)) != len(passes):
            raise ValueError("Use baseline, profile, or baseline then profile once each")
        for pass_name in passes:
            manifest["status"] = "running_" + pass_name; save()
            argv = [*common, "--entrypoint", "python3", image_id, "/ipat/container.py", "--recipe", "/output/recipe.json",
                    "--output", "/output/" + pass_name, "--pass-name", pass_name]
            with (output / (pass_name + "-container.log")).open("w") as log:
                result = subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT,
                                        timeout=recipe["startup_timeout_seconds"] + recipe["client_timeout_seconds"] + 2100)
            path = output / pass_name / "pass.json"
            record = json.loads(path.read_text()) if path.exists() else {"pass": pass_name, "status": "failed", "error": "Container did not produce a pass manifest"}
            manifest["passes"].append(record); save()
            if result.returncode or record["status"] != "completed": raise RuntimeError(f"{pass_name} failed; retained its logs and partial evidence")
        manifest["status"] = "completed"
    except Exception as error:
        manifest.update(status="failed", error=str(error))
    finally:
        subprocess.run(["docker", "stop", "--time", "90", name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if image_id:
            result = capture(["docker", "run", "--rm", "--mount", f"type=bind,src={output.resolve()},dst=/output",
                              "--entrypoint", "chown", image_id, "-R", f"{os.getuid()}:{os.getgid()}", "/output"], output / "ownership.log")
            if result.get("returncode") != 0: manifest["ownership_error"] = result
        manifest["ended_at"] = dt.datetime.now(dt.timezone.utc).isoformat(); save()
        report(output, manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cleanup", action="store_true")
    args = parser.parse_args()
    if not args.output.resolve().is_relative_to(Path(__file__).resolve().parents[1]):
        raise ValueError("IPAT output must stay inside the team repository checkout")
    if args.cleanup:
        cleanup(args.output)
        return
    request = json.loads(os.environ["IPAT_REQUEST"])
    if not re.fullmatch(r"[0-9a-f]{32}", request.get("id", "")):
        raise ValueError("IPAT request needs a unique hexadecimal ID")
    write_json(args.output / "request.json", request)
    if request.get("mode") == "preflight":
        result = preflight(request, args.output)
        raise SystemExit(0 if result["status"] == "completed" else 1)
    elif request.get("mode") == "recover":
        recover(request, args.output)
    elif request.get("mode") == "run":
        result = execute(request, args.output)
        raise SystemExit(0 if result["status"] == "completed" else 1)
    else:
        raise ValueError("Unsupported IPAT request mode")


if __name__ == "__main__":
    main()

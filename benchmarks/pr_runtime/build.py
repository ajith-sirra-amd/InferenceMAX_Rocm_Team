"""Prepare a full-source ROCm pair with shared dependency images; publish nothing.

Uses the checked-in official bake graph. Local image IDs are host-bound; a
registry export is a separate action. No GPUs are exposed to build containers.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess

from runtime import PAIRS, probe_image, verify_image


def run(argv, cwd=None, env=None):
    subprocess.run(argv, cwd=cwd, env=env, check=True)


def capture(argv, cwd=None, env=None):
    return subprocess.check_output(argv, cwd=cwd, env=env, text=True).strip()


def dependency_inputs(source):
    return capture(["git", "ls-tree", "-r", "HEAD", "--", "docker", "requirements",
                    "pyproject.toml", "setup.py", "setup.cfg", "CMakeLists.txt", "cmake",
                    "rust", "Cargo.lock", "Cargo.toml"], cwd=source)


def build(pr, output):
    if output.exists():
        raise ValueError("Use a fresh build output directory")
    data_root = Path(os.environ["IPAT_PR_DATA_ROOT"])
    if data_root.parent.parent != Path("/share_nfs") or data_root.name != "benchmarks":
        raise ValueError("Use an explicit user's shared benchmark root")
    if not output.resolve().is_relative_to(data_root):
        raise ValueError("Use shared benchmark storage")
    output.mkdir(parents=True)
    script = Path(__file__).resolve().parent
    (output / "build-host.txt").write_text(socket.gethostname() + "\n")
    (output / "team-sha.txt").write_text(capture(["git", "rev-parse", "HEAD"], cwd=script) + "\n")
    sources = {}
    for arm, sha in zip(("baseline", "candidate"), PAIRS[pr]):
        source = output / "work" / arm / "source"
        source.mkdir(parents=True)
        run(["git", "init", str(source)])
        run(["git", "remote", "add", "origin", "https://github.com/ajith-sirra-amd/vllm.git"], cwd=source)
        run(["git", "fetch", "--depth", "1", "origin", sha], cwd=source)
        run(["git", "checkout", "--detach", "FETCH_HEAD"], cwd=source)
        if capture(["git", "rev-parse", "HEAD"], cwd=source) != sha:
            raise ValueError("Fetched source revision mismatch")
        run(["git", "submodule", "update", "--init", "--recursive"], cwd=source)
        sources[arm] = source
    inputs = dependency_inputs(sources["baseline"])
    if inputs != dependency_inputs(sources["candidate"]):
        raise ValueError("Dependency build inputs differ; review a matched build recipe before retrying")
    (output / "dependency-inputs.txt").write_text(inputs + "\n")
    prefix = "ipat-pr-" + hashlib.sha256(str(output.resolve()).encode()).hexdigest()[:16]
    # Keep a single Docker daemon: docker-image named contexts must see these
    # local tags. Export the final immutable IDs rather than mutable tags.
    inspection = capture(["docker", "buildx", "inspect"])
    driver = next((line.split(":", 1)[1].strip() for line in inspection.splitlines()
                   if line.startswith("Driver:")), "")
    if driver != "docker":
        raise ValueError("This local build path requires the Docker buildx docker driver")
    common = os.environ | dict(STACK="rocm", CI_ARCH="gfx950", PUSH="false", CACHE="",
                               CACHE_TO="", AITER_SPEC="", AITER_NIGHTLY="", CACHE_CI="false",
                               STACK_IMAGE="", STACK_WHEELS="", RUNTIME="", TEST_DEPS="",
                               CONTEXT=".", TAG=prefix + "-dependencies")
    shared = sources["baseline"]
    run(["docker", "buildx", "bake", "-f", "docker/docker-bake-rocm.hcl",
         "--set", "stack-rocm.tags=" + prefix + "-stack",
         "--set", "runtime.tags=" + prefix + "-runtime",
         "--set", "test-deps.tags=" + prefix + "-test-deps",
         "stack-rocm", "runtime", "test-deps"], cwd=shared, env=common)
    pair = dict(schema_version=1, pr=pr, source_repository="ajith-sirra-amd/vllm",
                dependency_inputs_sha256=hashlib.sha256((inputs + "\n").encode()).hexdigest())
    for arm, sha in zip(("baseline", "candidate"), PAIRS[pr]):
        source = sources[arm]
        env = common | dict(STACK_IMAGE=prefix + "-stack", RUNTIME=prefix + "-runtime",
                            TEST_DEPS=prefix + "-test-deps", TAG=prefix + "-" + arm + "-official")
        run(["docker", "buildx", "bake", "-f", "docker/docker-bake-rocm.hcl", "test"], cwd=source, env=env)
        context = source.parent
        shutil.copyfile(script / "Dockerfile.adapter", context / "Dockerfile")
        shutil.copyfile(script / "cli_entry.py", context / "cli_entry.py")
        run(["docker", "build", "--build-arg", "BUILT_IMAGE=" + env["TAG"],
             "--build-arg", "SOURCE_SHA=" + sha, "--tag", prefix + "-" + arm, str(context)])
        identity = capture(["docker", "image", "inspect", "--format", "{{.Id}}", prefix + "-" + arm])
        probe = probe_image(identity, output / (arm + "-initial-probe.json"))
        pair[arm] = dict(source_sha=sha, image=identity, dependencies_sha256=probe["dependencies_sha256"])
        verify_image(pair[arm], output / (arm + "-verification.json"))
    if pair["baseline"]["dependencies_sha256"] != pair["candidate"]["dependencies_sha256"]:
        raise ValueError("Actual paired dependencies differ; preserved builds remain unqualified")
    (output / "pair.json").write_text(json.dumps(pair, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pr", type=int, choices=PAIRS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(args.pr, args.output)

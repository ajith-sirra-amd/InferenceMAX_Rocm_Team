"""Prepare a full-source ROCm pair with shared dependency images; publish nothing.

Uses the checked-in official bake graph. Local image IDs are host-bound; a
registry export is a separate action. No GPUs are exposed to build containers.
"""
import argparse
import hashlib
import json
import os
import re
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


def source_version(source, expected_sha):
    """Require real release ancestry before the official version-calc stage."""
    sha = capture(["git", "rev-parse", "HEAD"], cwd=source)
    if sha != expected_sha:
        raise ValueError("Fetched source revision mismatch")
    if capture(["git", "rev-parse", "--is-shallow-repository"], cwd=source) != "false":
        raise ValueError("Official version calculation requires full release ancestry")
    description = capture(["git", "describe", "--tags", "--long", "--abbrev=40",
                           "--match", "v[0-9]*"], cwd=source)
    tag = description.rsplit("-", 2)[0]
    tag_sha = capture(["git", "rev-parse", tag + "^{}"], cwd=source)
    return dict(source_sha=sha, git_description=description,
                release_tag=tag, release_tag_sha=tag_sha,
                release_repository="vllm-project/vllm")


def prepare_vendor(source, destination):
    recipe = (source / "docker/Dockerfile.rocm").read_text()
    pins = re.findall(r"^ARG ROCM_TRITON_KERNELS_COMMIT=([0-9a-f]{40})$", recipe, re.M)
    if len(pins) != 1:
        raise ValueError("Require one explicit official ROCm Triton kernels pin")
    destination.mkdir(parents=True)
    run(["git", "init", str(destination)])
    run(["git", "fetch", "--depth", "1", "https://github.com/ROCm/triton.git", pins[0]], cwd=destination)
    run(["git", "checkout", "--detach", pins[0]], cwd=destination)
    if capture(["git", "rev-parse", "HEAD"], cwd=destination) != pins[0]:
        raise ValueError("Vendored reference revision mismatch")


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
    versions = {}
    for arm, sha in zip(("baseline", "candidate"), PAIRS[pr]):
        source = output / "work" / arm / "source"
        source.mkdir(parents=True)
        run(["git", "init", str(source)])
        run(["git", "remote", "add", "origin", "https://github.com/ajith-sirra-amd/vllm.git"], cwd=source)
        seed = data_root / 'pr-build-43495485285-1/runtime/work/candidate/source'
        if seed.is_dir():
            seed_sha = capture(['git', 'rev-parse', 'HEAD'], cwd=seed)
            if capture(['git', 'rev-parse', '--is-shallow-repository'], cwd=seed) != 'false':
                raise ValueError('Local Git seed must retain full history')
            run(['git', 'fetch', '--no-tags', str(seed), seed_sha], cwd=source)
            (source.parent / 'git-seed.json').write_text(json.dumps(dict(
                path=str(seed), source_sha=seed_sha, scope='Object reuse only; frozen target and official release refs fetched independently'), indent=2)+'\n')
        run(["git", "fetch", "--no-tags", "origin", sha], cwd=source)
        # GitHub forks need not carry upstream release tags. Fetch only official
        # release refs; HEAD remains the explicit frozen fork commit.
        run(["git", "fetch", "--no-tags", "https://github.com/vllm-project/vllm.git",
             "refs/tags/v*:refs/tags/v*"], cwd=source)
        run(["git", "checkout", "--detach", sha], cwd=source)
        if capture(["git", "rev-parse", "HEAD"], cwd=source) != sha:
            raise ValueError("Fetched source revision mismatch")
        run(["git", "submodule", "update", "--init", "--recursive"], cwd=source)
        versions[arm] = source_version(source, sha)
        sources[arm] = source
    (output / "source-versions.json").write_text(json.dumps(versions, indent=2) + "\n")
    inputs = dependency_inputs(sources["baseline"])
    if inputs != dependency_inputs(sources["candidate"]):
        raise ValueError("Dependency build inputs differ; review a matched build recipe before retrying")
    (output / "dependency-inputs.txt").write_text(inputs + "\n")
    vendor = output / "work" / "vendor" / "triton"
    prepare_vendor(sources["baseline"], vendor)
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
        shutil.copytree(vendor, context / "vendor" / "triton")
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

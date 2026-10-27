"""Exercise the real MI355X launcher with a recording Docker collaborator."""
import json
import os
import pathlib
import subprocess
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]


def main():
    with tempfile.TemporaryDirectory(prefix=".launcher-check-", dir=ROOT) as directory:
        task_dir = pathlib.Path(directory)
        docker = task_dir / "docker"
        docker.write_text("""#!/usr/bin/env python3
import json, os, sys
with open(os.environ["DOCKER_CALLS"], "a") as stream:
    stream.write(json.dumps(sys.argv[1:]) + "\\n")
if sys.argv[1] == "inspect":
    print("stub@sha256:1234")
if sys.argv[1] == "run":
    raise SystemExit(17)
""")
        docker.chmod(0o755)
        settings = task_dir / "benchmarks" / "runtime_settings.sh"
        settings.parent.mkdir()
        settings.write_text('INFERENCEX_RUNTIME_ENV_VARS=""\n')
        trace = task_dir / "docker_calls.jsonl"
        env = dict(os.environ)
        env.update({
            "PATH": str(task_dir) + os.pathsep + env["PATH"],
            "GITHUB_WORKSPACE": str(ROOT),
            "INFMAX_HOST_REPO": str(task_dir),
            "DOCKER_CALLS": str(trace),
            "BMK_CONTAINER_NAME": "bmk-server-101-1-test",
            "RUNNER_NAME": "mi355x-amd_test7",
            "EXP_NAME": "kimik3_test",
            "FRAMEWORK": "vllm",
            "SPEC_DECODING": "mtp",
            "SCENARIO_SUBDIR": "agentic/",
            "PRECISION": "fp4",
            "MODEL": "moonshotai/Kimi-K3",
            "HF_HUB_CACHE": "/mnt/hf_hub_cache/",
            "IMAGE": "stub-image",
            "KV_OFFLOAD_BACKEND": "none",
        })
        result = subprocess.run(
            ["bash", str(ROOT / "runners/launch_mi355x-amd.sh")],
            env=env, cwd=ROOT, capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 17, result.stderr
        calls = [json.loads(line) for line in trace.read_text().splitlines()]
        run = next(call for call in calls if call[0] == "run")
        assert "--name=bmk-server-101-1-test" in run, run
        assert "--stop-timeout=90" in run, run
        cleanup = [call for call in calls if call[0] in ("rm", "stop")]
        assert cleanup == [
            ["rm", "-f", "bmk-server-101-1-test"],
            ["stop", "bmk-server-101-1-test"],
            ["rm", "bmk-server-101-1-test"],
        ], cleanup
        trace.unlink()
        env.pop("BMK_CONTAINER_NAME")
        missing = subprocess.run(
            ["bash", str(ROOT / "runners/launch_mi355x-amd.sh")],
            env=env, cwd=ROOT, capture_output=True, text=True, timeout=30,
        )
        assert missing.returncode == 1 and not trace.exists(), missing.stdout
        print("PASS: isolated Docker name, scoped cleanup, shutdown grace, failure propagation and missing-name rejection")


if __name__ == "__main__":
    main()

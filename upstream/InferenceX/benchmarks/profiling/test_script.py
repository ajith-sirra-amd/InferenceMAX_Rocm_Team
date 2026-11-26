"""Run the actual Kimi shell with external GPU/server/client collaborators stubbed."""

import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPT = HERE.parent / "single_node/agentic/kimik3_fp4_mi355x_mtp.sh"


class ScriptTests(unittest.TestCase):
    def execute(
        self,
        enabled=True,
        cancel=False,
        ranks=8,
        kind="agentic",
        prefix=0,
        cli=None,
        validation_failure=False,
        defaults=False,
        concurrency="48",
        max_num_seqs="48",
    ):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            logs = root / "logs"
            logs.mkdir()
            (root / "model").mkdir()
            (root / "draft").mkdir()
            stub = root / "external.sh"
            stub.write_text("""
rocm-smi() { echo 'GPU Memory Allocated (VRAM%): 0'; }
curl() { "$TEST_PYTHON" "$TEST_RUNTIME" health "$@"; }
function /opt/runtime-venv/bin/vllm { exec "$TEST_PYTHON" "$TEST_RUNTIME" server "$@"; }
uv() { "$TEST_PYTHON" "$TEST_RUNTIME" bootstrap "$@"; }
""")
            binary_dir = root / "bin"
            binary_dir.mkdir()
            python_stub = binary_dir / "python3"
            python_stub.write_text("""#!/usr/bin/env bash
if [[ "$1" == -m && "$2" == infx.bench_serving.benchmark_serving ]]; then
    shift 2
    exec "$TEST_PYTHON" "$TEST_RUNTIME" fixed "$@"
fi
if [[ "$1" == */profiling/vllm_client.py ]]; then
    shift
    exec "$TEST_PYTHON" "$TEST_RUNTIME" vllm "$@"
fi
exec "$TEST_PYTHON" "$@"
""")
            python_stub.chmod(0o755)
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", 0))
                port = sock.getsockname()[1]
            env = dict(
                os.environ,
                BASH_ENV=str(stub),
                TEST_PYTHON=sys.executable,
                TEST_RUNTIME=str(HERE / "fixtures/runtime.py"),
                PATH=str(binary_dir)
                + ":"
                + str(Path(sys.executable).parent)
                + ":/usr/bin:/bin",
                MODEL="moonshotai/Kimi-K3.1-Pro",
                MODEL_PATH=str(root / "model"),
                DRAFT_MODEL_PATH=str(root / "draft"),
                TP="8",
                DCP_SIZE="8",
                CONC=concurrency,
                MAX_NUM_SEQS=max_num_seqs,
                EP_SIZE="1",
                KV_OFFLOADING="dram",
                KV_OFFLOAD_BACKEND="vllm-simple",
                TOTAL_CPU_DRAM_GB="1799",
                RESULT_DIR=str(logs),
                TEST_LOG_ROOT=str(logs),
                PORT=str(port),
                EVAL_ONLY="false",
                TEST="1" if kind == "fixed" else "0",
                BENCHMARK_MODE={
                    "fixed": "serving",
                    "vllm": "vllm",
                    "agentic": "agentic",
                }[kind],
                VLLM_BENCH_PYTHON=str(python_stub),
                PROFILE=str(int(enabled)),
                DURATION="3600",
                INFERENCEX_REPO_ROOT=str(HERE.parent.parent),
                TEST_ISL="3584",
                TEST_OSL="1024",
                TEST_PREFIX_LEN=str(prefix),
                TEST_RANGE_RATIO="1",
                TEST_NUM_PROMPTS="192",
                RESULT_FILENAME="test-result",
                INFMAX_CONTAINER_WORKSPACE=str(HERE.parent.parent),
                AIPERF_RUNTIME_DIR=str(root / "runtime"),
                AIPERF_PYTHON_VERSION="3.11",
                MODEL_PREFIX="kimik3.1",
                FRAMEWORK="vllm",
                AGENTIC_OUTPUT_DIR=str(logs),
                ENABLE_AGENTX_POWER="0",
                IS_MULTINODE="false",
                REQUIRE_POWER="0",
                AIPERF_FAILED_REQUEST_THRESHOLD="0.10",
                AIPERF_LIVE_FAILED_REQUEST_THRESHOLD="0.10",
                AIPERF_TRACE_IDLE_GAP_CAP_SECONDS="300",
                AGENTIC_WARMUP_GRACE_PERIOD="1800",
                AIPERF_DATASET_WEKA_LIVE_ASSISTANT_RESPONSES="0",
                AIPERF_DYNAMO_SESSION_TIMEOUT_SECONDS="3600",
                AIPERF_EXPERIMENTAL_FAST="0",
                AIPERF_HTTP_X_DYNAMO_SESSION_ID_FROM_CORRELATION_ID="false",
                AIPERF_UNSAFE_OVERRIDE="false",
                AIPERF_USE_DYNAMO_CONV_AWARE_ROUTING="1",
                AIPERF_WARMUP_REQUESTS_PER_LANE="10",
                FAKE_VALIDATION_FAILURE=str(int(validation_failure)),
                PROFILE_DELAY_SECONDS="0.5",
                PROFILE_SECONDS="0.1",
                PROFILE_MIN_SECONDS="0.08",
                PROFILE_PHASE_TIMEOUT_SECONDS="5",
                PROFILE_POLL_SECONDS="0.01",
                PROFILE_EXPORT_TIMEOUT_SECONDS="1",
                FAKE_TRACE_RANKS=str(ranks),
                FAKE_CLIENT_SECONDS="20" if cancel else "1.5",
                PYTHONDONTWRITEBYTECODE="1",
            )
            if defaults:
                for key in (
                    "TEST_ISL",
                    "TEST_OSL",
                    "TEST_NUM_PROMPTS",
                    "TEST_PREFIX_LEN",
                    "TEST_RANGE_RATIO",
                    "CONC",
                    "MAX_NUM_SEQS",
                    "DURATION",
                ):
                    env.pop(key)
                prefix = 4608 if kind == "vllm" else 0
            env.pop("SA_CONFIG_AUDIT", None)
            with (root / "execution.log").open("w+") as stream:
                process = subprocess.Popen(
                    ["bash", str(SCRIPT), *([cli] if cli else [])],
                    env=env,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                try:
                    if cancel:
                        deadline = time.monotonic() + 15
                        while time.monotonic() < deadline:
                            events = logs / "events.log"
                            if (
                                events.exists()
                                and "/start_profile\n" in events.read_text()
                            ):
                                process.send_signal(signal.SIGTERM)
                                break
                            if process.poll() is not None:
                                self.fail("Script exited before cancellation")
                            time.sleep(0.01)
                        else:
                            self.fail("Capture did not start before test deadline")
                    code = process.wait(timeout=20)
                finally:
                    if process.poll() is None:
                        process.terminate()
                        try:
                            process.wait(timeout=15)
                        except subprocess.TimeoutExpired:
                            os.killpg(process.pid, signal.SIGKILL)
                            process.wait(timeout=5)
                stream.seek(0)
                log = stream.read()
            controller_log = logs / "profiles/torch/controller.log"
            if controller_log.exists():
                log += "\nCONTROLLER:\n" + controller_log.read_text()
            events = (logs / "events.log").read_text().splitlines()
            self.assertIn(kind + "-client-start", events, log)
            client_argv = json.loads((logs / (kind + "-client-argv.json")).read_text())
            if kind in ("fixed", "vllm"):
                self.assertEqual(
                    client_argv[client_argv.index("--num-prompts") + 1], "192"
                )
                prefix_option = (
                    "--random-prefix-len"
                    if kind == "fixed"
                    else "--prefix-repetition-prefix-len"
                )
                self.assertEqual(
                    client_argv[client_argv.index(prefix_option) + 1], str(prefix)
                )
                self.assertEqual(
                    client_argv[client_argv.index("--max-concurrency") + 1], concurrency
                )
                if defaults:
                    input_option = (
                        "--random-input-len"
                        if kind == "fixed"
                        else "--prefix-repetition-suffix-len"
                    )
                    self.assertEqual(
                        client_argv[client_argv.index(input_option) + 1],
                        "8192" if kind == "fixed" else "3584",
                    )
                self.assertNotIn("--profile", client_argv)
                self.assertTrue((logs / "test-result-command.txt").is_file())
            else:
                self.assertEqual(
                    client_argv[client_argv.index("--benchmark-duration") + 1], "3600"
                )
                self.assertEqual(
                    client_argv[client_argv.index("--concurrency") + 1], concurrency
                )
                self.assertTrue((logs / "agentic/benchmark_command.txt").is_file())
                if not cancel and ranks == 8:
                    self.assertIn("agentic-postprocess", events, log)
            actual_profile = cli == "--profile" or (enabled and cli != "--no-profile")
            self.assertIn("server-stop", events, log)
            argv = json.loads((logs / "server-argv.json").read_text())
            self.assertEqual(argv[argv.index("--max-num-seqs") + 1], max_num_seqs)
            self.assertEqual("--profiler-config" in argv, actual_profile)
            if not actual_profile:
                self.assertEqual(code == 0, not validation_failure, log)
                self.assertNotIn("/start_profile", events)
                self.assertFalse((logs / "profiles").exists())
                return
            self.assertIn("/stop_profile", events, log)
            self.assertLess(
                events.index("/stop_profile"), events.index("export-complete")
            )
            self.assertLess(
                events.index("export-complete"), events.index("server-stop")
            )
            validation = json.loads((logs / "profile-validation.json").read_text())
            self.assertEqual(
                validation["status"],
                "failed" if cancel or ranks != 8 or validation_failure else "passed",
                log,
            )
            self.assertEqual(
                code == 0, not cancel and ranks == 8 and not validation_failure, log
            )
            if code == 0:
                self.assertEqual(validation["top10_kernels"][0]["calls"], 8)
                self.assertTrue((logs / "profiles/torch/top10-kernels.csv").is_file())

    def test_all_workload_profile_combinations(self):
        for kind in ("fixed", "vllm", "agentic"):
            for enabled in (False, True):
                with self.subTest(kind=kind, profile=enabled):
                    self.execute(kind=kind, enabled=enabled)

    def test_shared_prefix_with_and_without_profile(self):
        for enabled in (False, True):
            with self.subTest(profile=enabled):
                self.execute(kind="fixed", enabled=enabled, prefix=4608)

    def test_defaults_for_each_workload(self):
        for kind in ("fixed", "vllm", "agentic"):
            with self.subTest(kind=kind):
                self.execute(kind=kind, enabled=False, defaults=True)

    def test_concurrency_and_scheduler_limit_are_independent(self):
        self.execute(kind="fixed", concurrency="8", max_num_seqs="64")

    def test_cli_profile_switches_override_environment(self):
        self.execute(kind="fixed", enabled=False, cli="--profile")
        self.execute(kind="fixed", enabled=True, cli="--no-profile")

    def test_missing_worker_trace_fails_the_run(self):
        self.execute(ranks=7)

    def test_agentic_validation_failure_is_preserved(self):
        self.execute(enabled=False, validation_failure=True)

    def test_cancellation_exports_before_server_cleanup(self):
        self.execute(cancel=True)


if __name__ == "__main__":
    unittest.main()

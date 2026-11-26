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
    def test_profile_rejects_fixed_length_before_gpu_start(self):
        env = dict(
            os.environ,
            MODEL="example",
            MODEL_PATH="/missing",
            DRAFT_MODEL_PATH="/missing",
            TP="8",
            DCP_SIZE="8",
            CONC="48",
            EP_SIZE="1",
            KV_OFFLOADING="dram",
            KV_OFFLOAD_BACKEND="vllm-simple",
            TOTAL_CPU_DRAM_GB="1799",
            RESULT_DIR="/missing",
            PORT="8000",
            EVAL_ONLY="false",
            TEST="1",
            PROFILE="0",
        )
        result = subprocess.run(
            ["bash", str(SCRIPT), "--profile"],
            env=env,
            check=False,
            text=True,
            capture_output=True,
            timeout=5,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("--profile requires TEST=0", result.stderr)

    def execute(self, enabled=True, cancel=False, ranks=8):
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
""")
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", 0))
                port = sock.getsockname()[1]
            env = dict(
                os.environ,
                BASH_ENV=str(stub),
                TEST_PYTHON=sys.executable,
                TEST_RUNTIME=str(HERE / "fixtures/runtime.py"),
                PATH=str(Path(sys.executable).parent) + ":/usr/bin:/bin",
                MODEL="moonshotai/Kimi-K3.1-Pro",
                MODEL_PATH=str(root / "model"),
                DRAFT_MODEL_PATH=str(root / "draft"),
                TP="8",
                DCP_SIZE="8",
                CONC="48",
                EP_SIZE="1",
                KV_OFFLOADING="dram",
                KV_OFFLOAD_BACKEND="vllm-simple",
                TOTAL_CPU_DRAM_GB="1799",
                RESULT_DIR=str(logs),
                PORT=str(port),
                EVAL_ONLY="false",
                TEST="0",
                PROFILE="0",
                DURATION="3600",
                INFERENCEX_REPO_ROOT=str(HERE.parent.parent),
                SA_CLIENT_ENTRYPOINT=str(HERE / "fixtures/runtime.py"),
                PROFILE_DELAY_SECONDS="0.2",
                PROFILE_SECONDS="0.1",
                PROFILE_MIN_SECONDS="0.08",
                PROFILE_PHASE_TIMEOUT_SECONDS="5",
                PROFILE_POLL_SECONDS="0.01",
                PROFILE_EXPORT_TIMEOUT_SECONDS="1",
                FAKE_TRACE_RANKS=str(ranks),
                FAKE_CLIENT_SECONDS="20" if cancel else "1",
                PYTHONDONTWRITEBYTECODE="1",
            )
            env.pop("SA_CONFIG_AUDIT", None)
            with (root / "execution.log").open("w+") as stream:
                process = subprocess.Popen(
                    ["bash", str(SCRIPT), *(["--profile"] if enabled else [])],
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
            events = (logs / "events.log").read_text().splitlines()
            self.assertEqual(events[0], "client-check", log)
            self.assertIn("server-stop", events, log)
            argv = json.loads((logs / "server-argv.json").read_text())
            self.assertEqual("--profiler-config" in argv, enabled)
            if not enabled:
                self.assertEqual(code, 0, log)
                self.assertNotIn("/start_profile", events)
                self.assertFalse((logs / "profiles").exists())
                return
            self.assertLess(
                events.index("/stop_profile"), events.index("export-complete")
            )
            self.assertLess(
                events.index("export-complete"), events.index("server-stop")
            )
            validation = json.loads((logs / "profile-validation.json").read_text())
            self.assertEqual(
                validation["status"],
                "failed" if cancel or ranks != 8 else "passed",
                log,
            )
            self.assertEqual(code == 0, not cancel and ranks == 8, log)
            if code == 0:
                self.assertEqual(validation["top10_kernels"][0]["calls"], 8)
                self.assertTrue((logs / "profiles/torch/top10-kernels.csv").is_file())

    def test_flag_off_leaves_server_and_workload_unprofiled(self):
        self.execute(enabled=False)

    def test_flag_captures_exports_and_validates_before_server_cleanup(self):
        self.execute()

    def test_missing_worker_trace_fails_the_run(self):
        self.execute(ranks=7)

    def test_cancellation_exports_before_server_cleanup(self):
        self.execute(cancel=True)


if __name__ == "__main__":
    unittest.main()

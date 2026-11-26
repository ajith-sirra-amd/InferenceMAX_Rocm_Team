"""Exercise marker-driven control and reject incomplete/out-of-phase exports."""

import csv
import gzip
import json
import os
import sys
import tempfile
import threading
import time
import unittest
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError

sys.path.insert(0, str(Path(__file__).parent))
from run import prepare
from steady import capture, marker_time
from validate import validate


class MeasuredTests(unittest.TestCase):
    def test_prepare_reserves_fresh_output_and_rejects_infeasible_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            logs = Path(tmp)
            settings = {
                "logs": logs,
                "duration": 3600,
                "delay": 1800,
                "seconds": 130,
                "minimum": 120,
                "phase_timeout": 7200,
                "poll": 2,
                "export_timeout": 600,
            }
            with self.assertRaisesRegex(ValueError, "fit inside"):
                prepare(**{**settings, "duration": 3000})
            self.assertFalse((logs / "profiles").exists())
            with self.assertRaisesRegex(ValueError, "finite and positive"):
                prepare(**{**settings, "delay": float("nan")})
            config = prepare(**settings)
            self.assertEqual(config["profiler"], "torch")
            self.assertTrue(config["ignore_frontend"])
            self.assertEqual(
                Path(config["torch_profiler_dir"]), logs / "profiles/torch"
            )
            with self.assertRaises(FileExistsError):
                prepare(**settings)

    def test_marker_uses_log_timestamp_across_midnight(self):
        observed = datetime(2026, 11, 12, 0, 0, 2, tzinfo=UTC).timestamp()
        line = "23:59:59.125 NOTICE Phase profiling (profiling) started | target: 3600.0s duration"
        expected = datetime(2026, 11, 11, 23, 59, 59, 125000, tzinfo=UTC).timestamp()
        self.assertEqual(marker_time(line, observed, "+0000"), expected)
        self.assertIsNone(
            marker_time("Phase warmup (warmup) started", observed, "+0000")
        )

    def run_controller(self, stale=False, lost_start=False):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output, log = root / "profile", root / "benchmark.log"
            start = time.time() - (5 if stale else 0.05)
            stamp = datetime.fromtimestamp(start, UTC).strftime("%H:%M:%S.%f")
            log.write_text(
                stamp
                + " NOTICE Phase profiling (profiling) started | target: 3600.0s duration\n"
            )
            (root / "agentic_power_timezone_offset.txt").write_text("+0000\n")
            calls, metrics = [], []

            class Handler(BaseHTTPRequestHandler):
                def log_message(self, *args):
                    pass

                def do_GET(self):
                    self.send_response(200)
                    self.end_headers()
                    if self.path == "/metrics":
                        metrics.append(1)
                        self.wfile.write(
                            f"vllm:generation_tokens_total {len(metrics)}\n".encode()
                        )

                def do_POST(self):
                    calls.append((self.path, time.time()))
                    if self.path == "/stop_profile":
                        for rank in range(8):
                            (output / f"rank{rank}.trace.json").write_text("{}")
                    self.send_response(
                        500 if lost_start and self.path == "/start_profile" else 200
                    )
                    self.end_headers()

            http = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
            threading.Thread(target=http.serve_forever, daemon=True).start()
            try:
                kwargs = {
                    "output": output,
                    "benchmark_log": log,
                    "base_url": f"http://127.0.0.1:{http.server_port}",
                    "workload_pid": os.getpid(),
                    "settle_seconds": 0.3,
                    "capture_seconds": 0.03,
                    "timeout_seconds": 1,
                    "poll_seconds": 0.01,
                    "export_timeout_seconds": 1,
                }
                if stale or lost_start:
                    with self.assertRaises(RuntimeError if stale else HTTPError):
                        capture(**kwargs)
                    self.assertEqual(
                        json.loads((output / "profile-control.json").read_text())[
                            "status"
                        ],
                        "failed",
                    )
                else:
                    report = capture(**kwargs)
                    self.assertEqual(report["status"], "exported")
                    self.assertGreaterEqual(calls[0][1], start + 0.3)
                    self.assertLess(calls[0][1], start + 0.6)
                    self.assertGreaterEqual(calls[1][1] - calls[0][1], 0.03)
                self.assertEqual(
                    [c[0] for c in calls],
                    [] if stale else ["/start_profile", "/stop_profile"],
                )
            finally:
                http.shutdown()
                http.server_close()

    def test_controller_starts_after_actual_marker_and_exports(self):
        self.run_controller()

    def test_stale_marker_cannot_start_a_late_capture(self):
        self.run_controller(stale=True)

    def test_lost_start_response_still_stops_and_preserves_failure(self):
        self.run_controller(lost_start=True)

    def test_complete_eight_rank_capture_and_missing_or_warmup_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            logs = Path(tmp)
            root = logs / "profiles/torch"
            root.mkdir(parents=True)
            (logs / "agentic").mkdir()
            (root / "profile-control.json").write_text('{"status":"exported"}')
            begin = 1794380000000000000
            (logs / "agentic/benchmark.log").write_text(
                f"PhaseRecordsStats(phase=CreditPhase.PROFILING, start_ns={begin}, sent_end_ns={begin + 3600 * 10**9})"
            )
            for rank in range(8):
                doc = {
                    "baseTimeNanoseconds": begin,
                    "traceEvents": [
                        {
                            "cat": "kernel",
                            "name": "prefill",
                            "dur": 1000,
                            "ts": (2400 + rank * 0.001) * 1e6,
                            "args": {"device": rank},
                        },
                        {
                            "cat": "kernel",
                            "name": "decode",
                            "dur": 3000,
                            "ts": 2530 * 1e6,
                            "args": {"device": rank},
                        },
                    ],
                }
                with gzip.open(root / f"rank{rank}.pt.trace.json.gz", "wt") as f:
                    json.dump(doc, f)
            self.assertEqual(validate(logs, 120)["status"], "passed")
            with (root / "top10-kernels.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual([row["kernel"] for row in rows], ["decode", "prefill"])
            self.assertEqual(float(rows[0]["gpu_time_share_percent"]), 75)
            self.assertEqual(float(rows[1]["cumulative_gpu_time_share_percent"]), 100)
            self.assertEqual(int(rows[0]["calls"]), 8)
            self.assertEqual(float(rows[0]["mean_us"]), 3000)
            (root / "profile-control.json").write_text(
                json.dumps(
                    {
                        "status": "exported",
                        "perf_counter_to_unix_seconds": begin / 1e9 - 100,
                    }
                )
            )
            native_result = {
                "completed": 2,
                "num_prompts": 2,
                "start_times": [100, 101],
                "latencies": [3600, 3599],
            }
            native_path = logs / "native.json"
            native_path.write_text(json.dumps(native_result))
            native = validate(logs, 120, workload="vllm", result_filename="native")
            self.assertEqual(
                native["measurement_utc_ns"], [begin, begin + 3600 * 10**9]
            )
            native_path.write_text(json.dumps({**native_result, "latencies": [2, 2]}))
            with self.assertRaisesRegex(ValueError, "outside measured"):
                validate(logs, 120, workload="vllm", result_filename="native")
            native_path.write_text(json.dumps({**native_result, "latencies": [2]}))
            with self.assertRaisesRegex(ValueError, "timings are missing"):
                validate(logs, 120, workload="vllm", result_filename="native")
            with self.assertRaisesRegex(ValueError, "missing required shape"):
                validate(logs, 120, require_metadata=True)
            for path in root.glob("*.pt.trace.json.gz"):
                with gzip.open(path, "rt") as f:
                    doc = json.load(f)
                doc["traceEvents"] += [
                    {
                        "cat": "cpu_op",
                        "args": {
                            "Input Dims": [[32, 64]],
                            "Input type": ["c10::BFloat16"],
                        },
                    },
                    {
                        "cat": "python_function",
                        "name": "observed.py(10): observed_call",
                    },
                ]
                with gzip.open(path, "wt") as f:
                    json.dump(doc, f)
            metadata_result = validate(logs, 120, require_metadata=True)
            self.assertTrue(
                all(
                    f["metadata"]["shape_events"] == 1 for f in metadata_result["files"]
                )
            )
            with self.assertRaises(ValueError):
                validate(logs, 140)
            path = root / "rank0.pt.trace.json.gz"
            with gzip.open(path, "rt") as f:
                doc = json.load(f)
            doc["traceEvents"][0]["ts"] = -1e6
            with gzip.open(path, "wt") as f:
                json.dump(doc, f)
            with self.assertRaisesRegex(ValueError, "outside measured"):
                validate(logs, 120)
            path.unlink()
            with self.assertRaisesRegex(ValueError, "eight worker traces"):
                validate(logs, 120)


if __name__ == "__main__":
    unittest.main()

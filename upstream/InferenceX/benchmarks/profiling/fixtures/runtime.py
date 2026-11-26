"""CPU-only vLLM/AgentX collaborators for the real shell and controller tests."""

import gzip
import json
import os
import signal
import sys
import time
import urllib.request
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

logs = Path(os.environ["TEST_LOG_ROOT"])


def event(name):
    with (logs / "events.log").open("a") as stream:
        stream.write(name + "\n")


def server():
    argv = sys.argv[2:]
    (logs / "server-argv.json").write_text(json.dumps(argv))
    output = None
    if "--profiler-config" in argv:
        config = json.loads(argv[argv.index("--profiler-config") + 1])
        output = Path(config["torch_profiler_dir"])
    started = 0

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            if self.path == "/metrics":
                self.wfile.write(
                    f"vllm:generation_tokens_total {time.time() * 100}\n".encode()
                )

        def do_POST(self):
            nonlocal started
            self.rfile.read(int(self.headers.get("Content-Length", 0)))
            event(self.path)
            if self.path == "/start_profile":
                started = time.time_ns()
            if self.path == "/stop_profile":
                duration = (time.time_ns() - started) / 1000
                for rank in range(int(os.environ["FAKE_TRACE_RANKS"])):
                    with gzip.open(
                        output / f"rank{rank}.pt.trace.json.gz", "wt"
                    ) as stream:
                        json.dump(
                            {
                                "baseTimeNanoseconds": started,
                                "traceEvents": [
                                    {
                                        "cat": "kernel",
                                        "name": "fixture_kernel",
                                        "ts": 0,
                                        "dur": duration,
                                        "args": {"device": rank},
                                    }
                                ],
                            },
                            stream,
                        )
                event("export-complete")
            self.send_response(200)
            self.end_headers()

    def stop(signum, frame):
        event("server-stop")
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, stop)
    with HTTPServer(("127.0.0.1", int(os.environ["PORT"])), Handler) as http:
        http.serve_forever()


def client(kind):
    event(kind + "-client-start")
    argv = sys.argv[2:]
    (logs / (kind + "-client-argv.json")).write_text(json.dumps(argv))
    begin = time.time_ns()
    if kind == "agentic":
        stamp = datetime.fromtimestamp(begin / 1e9).strftime("%H:%M:%S.%f")
        print(stamp + " NOTICE Phase profiling (profiling) started", flush=True)
    elif "INFERENCEX_PROFILE_PHASE_FILE" in os.environ:
        phase = Path(os.environ["INFERENCEX_PROFILE_PHASE_FILE"])
        temporary = phase.with_suffix(".tmp")
        temporary.write_text(json.dumps({"start_unix": begin / 1e9}))
        temporary.replace(phase)
    time.sleep(float(os.environ["FAKE_CLIENT_SECONDS"]))
    end = time.time_ns()
    if kind == "agentic":
        print(
            f"PhaseRecordsStats(phase=CreditPhase.PROFILING, start_ns={begin}, sent_end_ns={end})",
            flush=True,
        )
    else:
        filename = argv[argv.index("--result-filename") + 1]
        (logs / filename).write_text(
            json.dumps(
                {
                    "benchmark_start_time_unix": begin / 1e9,
                    "benchmark_end_time_unix": end / 1e9,
                    "completed": 192,
                    "num_prompts": 192,
                }
            )
        )
        if kind == "vllm" and "INFERENCEX_PROFILE_PHASE_FILE" in os.environ:
            phase.write_text(
                json.dumps({"start_unix": begin / 1e9, "end_unix": end / 1e9})
            )
    event(kind + "-client-end")


def bootstrap():
    event("client-bootstrap")
    if sys.argv[2] == "venv":
        directory = Path(sys.argv[-1]) / "bin"
        directory.mkdir(parents=True)
        for executable, action in (
            ("python", "agentic-python"),
            ("aiperf", "agentic"),
            ("hf", "hf"),
        ):
            path = directory / executable
            path.write_text(
                '#!/usr/bin/env bash\nexec "$TEST_PYTHON" "$TEST_RUNTIME" '
                + action
                + ' "$@"\n'
            )
            path.chmod(0o755)


if sys.argv[1:2] == ["server"]:
    server()
elif sys.argv[1:2] == ["health"]:
    url = sys.argv[-1].replace("0.0.0.0", "127.0.0.1")
    for attempt in range(100):
        try:
            urllib.request.urlopen(url, timeout=0.1).close()
            break
        except OSError:
            time.sleep(0.01)
    else:
        raise SystemExit(1)
elif sys.argv[1:2] == ["bootstrap"]:
    bootstrap()
elif sys.argv[1:2] == ["agentic-python"]:
    event("agentic-postprocess")
    if os.environ.get("FAKE_VALIDATION_FAILURE") == "1" and sys.argv[3].endswith(
        "validate_agentic_result"
    ):
        raise SystemExit(7)
elif sys.argv[1:2] == ["hf"]:
    event("dataset-ready")
else:
    client(sys.argv[1])

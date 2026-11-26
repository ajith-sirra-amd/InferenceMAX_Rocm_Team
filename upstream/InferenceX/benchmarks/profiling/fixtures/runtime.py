"""CPU-only vLLM/AgentX collaborators for the real shell and controller tests."""

import gzip
import json
import os
import signal
import sys
import time
import urllib.request
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

logs = Path(os.environ["RESULT_DIR"])


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


def client():
    if sys.argv[1:] == ["--check"]:
        event("client-check")
        return
    event("client-start")
    directory = logs / "agentic"
    directory.mkdir(exist_ok=True)
    (directory / "agentic_power_timezone_offset.txt").write_text("+0000\n")
    begin = time.time_ns()
    stamp = datetime.fromtimestamp(begin / 1e9, UTC).strftime("%H:%M:%S.%f")
    log = directory / "benchmark.log"
    log.write_text(stamp + " NOTICE Phase profiling (profiling) started\n")
    time.sleep(float(os.environ["FAKE_CLIENT_SECONDS"]))
    with log.open("a") as stream:
        stream.write(
            f"PhaseRecordsStats(phase=CreditPhase.PROFILING, start_ns={begin}, sent_end_ns={time.time_ns()})\n"
        )
    event("client-end")


if sys.argv[1:2] == ["server"]:
    server()
elif sys.argv[1:2] == ["health"]:
    # The real readiness helper calls curl; emulate its external HTTP operation.
    url = sys.argv[-1].replace("0.0.0.0", "127.0.0.1")
    for attempt in range(100):
        try:
            urllib.request.urlopen(url, timeout=0.1).close()
            break
        except OSError:
            time.sleep(0.01)
    else:
        raise SystemExit(1)
else:
    client()

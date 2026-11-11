"""Exercise marker-driven control and reject incomplete/out-of-phase exports."""
from datetime import datetime, timezone
import gzip
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest

sys.path.insert(0, str(Path(__file__).parent / 'profile'))
from steady import capture, marker_time
from validate_torch_profile import validate


class MeasuredTests(unittest.TestCase):
    def test_marker_uses_log_timestamp_across_midnight(self):
        observed = datetime(2026, 11, 12, 0, 0, 2, tzinfo=timezone.utc).timestamp()
        line = '23:59:59.125 NOTICE Phase profiling (profiling) started | target: 3600.0s duration'
        expected = datetime(2026, 11, 11, 23, 59, 59, 125000, tzinfo=timezone.utc).timestamp()
        self.assertEqual(marker_time(line, observed, '+0000'), expected)
        self.assertIsNone(marker_time('Phase warmup (warmup) started', observed, '+0000'))

    def run_controller(self, stale=False, lost_start=False):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output, log = root / 'profile', root / 'benchmark.log'
            start = time.time() - (5 if stale else 0.05)
            stamp = datetime.fromtimestamp(start, timezone.utc).strftime('%H:%M:%S.%f')
            log.write_text(stamp + ' NOTICE Phase profiling (profiling) started | target: 3600.0s duration\n')
            (root / 'agentic_power_timezone_offset.txt').write_text('+0000\n')
            calls, metrics = [], []
            class Handler(BaseHTTPRequestHandler):
                def log_message(self, *args):
                    pass
                def do_GET(self):
                    self.send_response(200); self.end_headers()
                    if self.path == '/metrics':
                        metrics.append(1)
                        self.wfile.write(f'vllm:generation_tokens_total {len(metrics)}\n'.encode())
                def do_POST(self):
                    calls.append((self.path, time.time()))
                    if self.path == '/stop_profile':
                        for rank in range(8):
                            (output / f'rank{rank}.trace.json').write_text('{}')
                    self.send_response(500 if lost_start and self.path == '/start_profile' else 200)
                    self.end_headers()
            http = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
            threading.Thread(target=http.serve_forever, daemon=True).start()
            try:
                kwargs = dict(output=output, benchmark_log=log,
                    base_url=f'http://127.0.0.1:{http.server_port}', workload_pid=os.getpid(),
                    settle_seconds=0.3, capture_seconds=0.03, timeout_seconds=1,
                    poll_seconds=0.01, export_timeout_seconds=1)
                if stale or lost_start:
                    with self.assertRaises(Exception):
                        capture(**kwargs)
                    self.assertEqual(json.loads((output / 'profile-control.json').read_text())['status'], 'failed')
                else:
                    report = capture(**kwargs)
                    self.assertEqual(report['status'], 'exported')
                    self.assertGreaterEqual(calls[0][1], start + 0.3)
                    self.assertLess(calls[0][1], start + 0.6)
                    self.assertGreaterEqual(calls[1][1] - calls[0][1], 0.03)
                self.assertEqual([c[0] for c in calls], [] if stale else ['/start_profile', '/stop_profile'])
            finally:
                http.shutdown(); http.server_close()

    def test_controller_starts_after_actual_marker_and_exports(self):
        self.run_controller()

    def test_stale_marker_cannot_start_a_late_capture(self):
        self.run_controller(stale=True)

    def test_lost_start_response_still_stops_and_preserves_failure(self):
        self.run_controller(lost_start=True)

    def test_complete_eight_rank_capture_and_missing_or_warmup_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            logs = Path(tmp)
            root = logs / 'profiles/torch'; root.mkdir(parents=True)
            (logs / 'agentic').mkdir()
            (root / 'profile-control.json').write_text('{"status":"exported"}')
            begin = 1794380000000000000
            (logs / 'agentic/benchmark.log').write_text(f'PhaseRecordsStats(phase=CreditPhase.PROFILING, start_ns={begin}, sent_end_ns={begin+3600*10**9})')
            for rank in range(8):
                doc = dict(baseTimeNanoseconds=begin, traceEvents=[
                    dict(cat='kernel', dur=1000, ts=(2400 + rank * .001) * 1e6, args={'device':rank}),
                    dict(cat='kernel', dur=1000, ts=2530 * 1e6, args={'device':rank})])
                with gzip.open(root / f'rank{rank}.pt.trace.json.gz', 'wt') as f:
                    json.dump(doc, f)
            self.assertEqual(validate(logs, 120)['status'], 'passed')
            with self.assertRaises(ValueError):
                validate(logs, 140)
            path = root / 'rank0.pt.trace.json.gz'
            with gzip.open(path, 'rt') as f:
                doc = json.load(f)
            doc['traceEvents'][0]['ts'] = -1e6
            with gzip.open(path, 'wt') as f:
                json.dump(doc, f)
            with self.assertRaisesRegex(ValueError, 'outside measured'):
                validate(logs, 120)
            path.unlink()
            with self.assertRaisesRegex(ValueError, 'eight worker traces'):
                validate(logs, 120)


if __name__ == '__main__':
    unittest.main()

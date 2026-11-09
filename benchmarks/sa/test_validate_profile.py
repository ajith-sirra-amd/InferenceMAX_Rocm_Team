import csv
import json
from pathlib import Path
import tempfile
import unittest

from validate_profile import validate


class CaptureValidationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.logs = Path(self.temp.name)
        self.root = self.logs / 'profiles/rocprof'
        (self.root / 'traces').mkdir(parents=True)
        (self.root / 'clock-sync.json').write_text(json.dumps(dict(
            utc_before_ns=10**12, utc_after_ns=10**12, boottime_ns=0)))
        (self.logs / 'benchmark.out').write_text(
            'PhaseRecordsStats(phase=CreditPhase.PROFILING, start_ns=1000000000000, sent_end_ns=1100000000000)')

    def trace(self, devices=8, start=10**10):
        with (self.root / 'traces/15_kernel_trace.csv').open('w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Agent_Id', 'Start_Timestamp', 'End_Timestamp'])
            for i in range(devices):
                writer.writerow([f'Agent {i+2}', start, start+8*10**9])

    def test_completed_workload_without_traces_fails(self):
        result = validate(self.logs, 8)
        self.assertEqual(result['status'], 'failed')
        self.assertIn('found 0', result['errors'][0])

    def test_native_csv_without_pid_column_passes_with_all_devices(self):
        self.trace()
        self.assertEqual(validate(self.logs, 8)['status'], 'passed')

    def test_report_can_be_written_outside_container_owned_traces(self):
        report = self.logs / 'profile-capture-validation.json'
        result = validate(self.logs, 8, output=report)
        self.assertEqual(json.loads(report.read_text()), result)
        self.assertEqual(result['status'], 'failed')
        self.assertFalse((self.root / 'capture-validation.json').exists())

    def test_attach_capture_requires_confirmed_export_before_worker_exit(self):
        self.trace()
        self.assertEqual(validate(self.logs, 8, require_attachment=True)['status'], 'failed')
        status = self.root / 'attachment-status.json'
        status.write_text(json.dumps(dict(status='passed', exported_while_workers_alive=False)))
        self.assertEqual(validate(self.logs, 8, require_attachment=True)['status'], 'failed')
        status.write_text(json.dumps(dict(status='passed', exported_while_workers_alive=True)))
        self.assertEqual(validate(self.logs, 8, require_attachment=True)['status'], 'passed')

    def test_missing_rank_and_warmup_capture_fail(self):
        self.trace(devices=7)
        self.assertEqual(validate(self.logs, 8)['status'], 'failed')
        self.trace(start=-10**9)
        result = validate(self.logs, 8)
        self.assertEqual(result['status'], 'failed')
        self.assertTrue(any('outside measured' in e for e in result['errors']))

    def test_export_crash_overrides_graceful_engine_message(self):
        self.trace()
        (self.logs / 'host_agg_w0.out').write_text(
            '[shutdown] Executor: all workers exited gracefully\n'
            '!!!!!!! Segfault encountered !!!!!!!\n')
        result = validate(self.logs, 8)
        self.assertEqual(result['status'], 'failed')
        self.assertTrue(any('Segfault' in error for error in result['errors']))


if __name__ == '__main__':
    unittest.main()

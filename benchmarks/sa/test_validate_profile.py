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

    def test_minimum_requires_a_shared_interval_on_all_devices(self):
        (self.logs / 'benchmark.out').write_text(
            'PhaseRecordsStats(phase=CreditPhase.PROFILING, start_ns=1000000000000, sent_end_ns=1600000000000)')
        trace = self.root / 'traces/15_kernel_trace.csv'
        for last_device_offset, expected, shared in [(9, 'passed', 121), (15, 'failed', 115)]:
            with self.subTest(last_device_offset=last_device_offset):
                with trace.open('w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['Agent_Id', 'Start_Timestamp', 'End_Timestamp'])
                    for i in range(8):
                        start = (100 + (last_device_offset if i == 7 else 0)) * 10**9
                        writer.writerow([f'Agent {i+2}', start, start + 130 * 10**9])
                result = validate(self.logs, 130, minimum_seconds=120)
                self.assertEqual(result['status'], expected)
                self.assertEqual(result['common_coverage_seconds'], shared)

    def test_minimum_accounts_for_clock_uncertainty(self):
        (self.logs / 'benchmark.out').write_text(
            'PhaseRecordsStats(phase=CreditPhase.PROFILING, start_ns=1000000000000, sent_end_ns=1600000000000)')
        (self.root / 'clock-sync.json').write_text(json.dumps(dict(
            utc_before_ns=10**12, utc_after_ns=10**12 + 10**9, boottime_ns=0)))
        with (self.root / 'traces/15_kernel_trace.csv').open('w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Agent_Id', 'Start_Timestamp', 'End_Timestamp'])
            for i in range(8):
                writer.writerow([f'Agent {i+2}', 100 * 10**9, 220 * 10**9])
        result = validate(self.logs, 130, minimum_seconds=120)
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(result['common_coverage_seconds'], 119)

    def test_minimum_rejects_a_short_capture_that_passed_old_tolerance(self):
        (self.logs / 'benchmark.out').write_text(
            'PhaseRecordsStats(phase=CreditPhase.PROFILING, start_ns=1000000000000, sent_end_ns=1600000000000)')
        with (self.root / 'traces/15_kernel_trace.csv').open('w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Agent_Id', 'Start_Timestamp', 'End_Timestamp'])
            for i in range(8):
                writer.writerow([f'Agent {i+2}', 100 * 10**9, 210 * 10**9])
        self.assertEqual(validate(self.logs, 130)['status'], 'passed')
        self.assertEqual(validate(self.logs, 130, minimum_seconds=120)['status'], 'failed')

    def test_export_tail_is_bounded_and_must_remain_inside_measured_phase(self):
        (self.logs / 'benchmark.out').write_text(
            'PhaseRecordsStats(phase=CreditPhase.PROFILING, start_ns=1000000000000, sent_end_ns=1600000000000)')
        for span, start, expected in [(140, 100, 'passed'), (146, 100, 'failed'), (140, 470, 'failed')]:
            with self.subTest(span=span, start=start):
                with (self.root / 'traces/15_kernel_trace.csv').open('w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['Agent_Id', 'Start_Timestamp', 'End_Timestamp'])
                    for i in range(8):
                        writer.writerow([f'Agent {i+2}', start * 10**9, (start + span) * 10**9])
                self.assertEqual(validate(self.logs, 130, minimum_seconds=120)['status'], expected)

    def export(self):
        entry = dict(metadata=dict(pid=15, init_time=10, fini_time=20),
                     agents=[dict(id=dict(handle=100+i), logical_node_id=i+2) for i in range(8)],
                     buffer_records=dict(kernel_dispatch=[
                         dict(dispatch_info=dict(agent_id=dict(handle=100+i), kernel_id=42)) for i in range(8)]),
                     kernel_symbols=[dict(kernel_id=42)],
                     summary=[dict(domain='KERNEL_DISPATCH', stats=dict(count=8))])
        return {'rocprofiler-sdk-tool': [entry]}

    def test_complete_json_required_in_addition_to_valid_csv(self):
        self.trace()
        path = self.root / 'traces/15_results.json'
        self.assertEqual(validate(self.logs, 8, require_exports=True)['status'], 'failed')
        path.write_text(json.dumps(self.export()))
        result = validate(self.logs, 8, require_exports=True)
        self.assertEqual(result['status'], 'passed')
        self.assertEqual(result['exports'][0]['kernel_calls'], 8)
        path.write_text(path.read_text()[:-1])
        self.assertEqual(validate(self.logs, 8, require_exports=True)['status'], 'failed')

    def test_export_rejects_missing_finalization_counts_and_symbols(self):
        self.trace()
        for problem in ('finalization', 'device', 'symbol', 'summary'):
            with self.subTest(problem=problem):
                document = self.export()
                entry = document['rocprofiler-sdk-tool'][0]
                if problem == 'finalization':
                    entry['metadata']['fini_time'] = 0
                elif problem == 'device':
                    entry['buffer_records']['kernel_dispatch'][0]['dispatch_info']['agent_id']['handle'] = 101
                elif problem == 'symbol':
                    entry['kernel_symbols'] = []
                else:
                    entry['summary'][0]['stats']['count'] = 7
                (self.root / 'traces/15_results.json').write_text(json.dumps(document))
                self.assertEqual(validate(self.logs, 8, require_exports=True)['status'], 'failed')


if __name__ == '__main__':
    unittest.main()

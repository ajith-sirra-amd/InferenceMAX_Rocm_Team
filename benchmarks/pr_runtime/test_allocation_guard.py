import io
import json
import signal
import unittest
from unittest.mock import Mock, patch
from pathlib import Path
from allocation_guard import run, release_owned


class GuardTests(unittest.TestCase):
    def run_guard(self, verification, child):
        output = Path('/share_nfs/gguasti/benchmarks/pr-perf-test/replica-2/candidate')
        stream = io.StringIO()
        with patch('allocation_guard.verify', side_effect=verification), \
             patch('allocation_guard.Path.mkdir'), \
             patch('allocation_guard.Path.open') as opening, \
             patch('allocation_guard.subprocess.Popen', return_value=child) as launch, \
             patch('allocation_guard.time.sleep'), \
             patch('allocation_guard.os.killpg', create=True) as stop, \
             patch('allocation_guard.release_owned', return_value=[]) as release:
            opening.return_value.__enter__.return_value = stream
            result = run(['bash', 'arm.sh'], output/'allocation-guard.jsonl', output)
            return result, launch, stop, release, stream.getvalue()

    def test_refuses_missing_allocation_before_child(self):
        result, launch, stop, release, log = self.run_guard([AssertionError('cancelled')], Mock())
        self.assertEqual(125, result)
        launch.assert_not_called()
        stop.assert_not_called()
        self.assertIn('refused_before_launch', log)

    def test_stops_only_own_session_on_loss(self):
        child = Mock(pid=12345, returncode=143)
        child.poll.side_effect = [None, None]
        result, launch, stop, release, log = self.run_guard([{'job_id':'30216'}, AssertionError('cancelled')], child)
        self.assertEqual(125, result)
        stop.assert_called_once_with(12345, signal.SIGTERM)
        release.assert_called_once_with(Path('/share_nfs/gguasti/benchmarks/pr-perf-test/replica-2/candidate').resolve())
        self.assertTrue(launch.call_args.kwargs['start_new_session'])
        self.assertIn('allocation_failure_terminal', log)

    def test_passes_normal_exit_code(self):
        child = Mock(pid=12345, returncode=0)
        child.poll.return_value = 0
        result, launch, stop, release, log = self.run_guard([{'job_id':'30216'}, {'job_id':'30216'}], child)
        self.assertEqual(0, result)
        stop.assert_not_called()
        release.assert_not_called()
        self.assertIn('arm_terminal', log)

    def test_loss_at_completion_is_not_accepted(self):
        child = Mock(pid=12345, returncode=0)
        child.poll.return_value = 0
        result, launch, stop, release, log = self.run_guard([{'job_id':'30216'}, AssertionError('cancelled')], child)
        self.assertEqual(125, result)
        stop.assert_not_called()
        self.assertIn('allocation_unavailable_after_arm', log)

    def test_rejects_changed_container_binding_before_stop(self):
        output = Path('/share_nfs/gguasti/benchmarks/pr-perf-test/candidate')
        info = {'Id':'a'*64, 'Config':{'Labels':{'ipat.pr.output':'/some-other-attempt'}}}
        with patch('allocation_guard.subprocess.check_output', side_effect=['short-id\n', json.dumps([info])]), \
             patch('allocation_guard.subprocess.run') as stop:
            with self.assertRaises(AssertionError):
                release_owned(output)
            stop.assert_not_called()

    def test_container_stop_uses_inspected_immutable_id(self):
        output = Path('/share_nfs/gguasti/benchmarks/pr-perf-test/candidate')
        info = {'Id':'a'*64, 'Config':{'Labels':{'ipat.pr.output':str(output)}}}
        with patch('allocation_guard.subprocess.check_output', side_effect=['short-id\n', json.dumps([info])]), \
             patch('allocation_guard.subprocess.run', return_value=Mock(returncode=0, stdout='', stderr='')) as stop:
            release_owned(output)
            self.assertEqual(['docker','stop','--time','30','a'*64], stop.call_args.args[0])


if __name__ == '__main__':
    unittest.main()

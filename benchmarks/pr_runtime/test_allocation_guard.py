import io
import signal
import unittest
from unittest.mock import Mock, patch
from pathlib import Path
from allocation_guard import run


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


if __name__ == '__main__':
    unittest.main()

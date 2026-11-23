import datetime
import unittest
from unittest.mock import patch
from allocation import verify


class AllocationTests(unittest.TestCase):
    def output(self, argv, **kwargs):
        if argv[0] == 'hostname':
            return self.node + '\n'
        end = (datetime.datetime.now() + datetime.timedelta(hours=8)).isoformat(timespec='seconds')
        return f'JobId=30172 UserId=giovanni.guasti@amd.com JobState=RUNNING NodeList=pit2-p03-g42 EndTime={end}'

    def test_explicit_g42_and_lifetime_gate(self):
        self.node = 'pit2-p03-g42'
        with patch('allocation.subprocess.check_output', self.output):
            self.assertEqual('30172', verify(21600, job='30172', node=self.node)['job_id'])
            with self.assertRaisesRegex(AssertionError, 'needs extension'):
                verify(40000, job='30172', node=self.node)

    def test_wrong_node_and_wrong_job_rejected(self):
        self.node = 'pit2-p03-g34'
        with patch('allocation.subprocess.check_output', self.output):
            with self.assertRaisesRegex(AssertionError, 'node differs'):
                verify(job='30172', node='pit2-p03-g42')
        self.node = 'pit2-p03-g42'
        with patch('allocation.subprocess.check_output', self.output):
            with self.assertRaises(AssertionError):
                verify(job='29088', node=self.node)


if __name__ == '__main__':
    unittest.main()

"""Behavior tests for the native serving artifact validator."""
import copy
import unittest
from validate import validate_result


class NativeResultTests(unittest.TestCase):
    def setUp(self):
        self.data = dict(num_prompts=2, completed=2, max_concurrency=2,
                         total_input_tokens=16, total_output_tokens=6, duration=4,
                         benchmark_outcome=dict(status='passed', failed=0),
                         input_lens=[8, 8], output_lens=[3, 3], ttfts=[1, 2],
                         itls=[[0.5, 0.5], [1, 1]], errors=['', ''], mean_e2el_ms=3000)

    def check(self, data):
        return validate_result(data, input_tokens=8, output_tokens=3, prompts=2, concurrency=2)

    def test_metrics(self):
        result = self.check(self.data)
        self.assertEqual(result['status'], 'passed')
        self.assertEqual(result['e2e_reconstructed']['avg'], 3000)
        self.assertEqual(result['e2e_reconstructed']['std'], 1000)
        self.assertEqual(result['e2e_reconstructed']['min'], 2000)
        self.assertEqual(result['e2e_reconstructed']['max'], 4000)
        self.assertEqual(result['ttft']['avg'], 1500)
        self.assertEqual(result['tpot']['avg'], 750)
        self.assertEqual(result['output_tokens_per_second'], 1.5)
        self.assertEqual(result['native_e2e']['mean_e2el_ms'], 3000)

    def test_incomplete_or_inconsistent_requests_fail(self):
        mutations = [dict(completed=1), dict(max_concurrency=1), dict(total_input_tokens=15),
                     dict(input_lens=[7, 9]), dict(output_lens=[2, 4]), dict(errors=['', 'failure']),
                     dict(benchmark_outcome=dict(status='passed', failed=1)), dict(ttfts=[1]),
                     dict(ttfts=[1, float('nan')]), dict(itls=[[0.5, -0.5], [1, 1]]), dict(duration=0)]
        for change in mutations:
            with self.subTest(change=change):
                self.assertEqual(self.check(self.data | change)['status'], 'failed')

    def test_aggregate_only_result_requires_detailed_export(self):
        data = copy.deepcopy(self.data)
        del data['ttfts']
        result = self.check(data)
        self.assertEqual(result['status'], 'failed')
        self.assertTrue(result['errors'])


if __name__ == '__main__':
    unittest.main()

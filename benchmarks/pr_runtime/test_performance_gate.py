import copy
import json
import unittest

from performance_gate import normalize_recipe, same_layout_fixture


class PerformanceGateTests(unittest.TestCase):
    def recipe(self):
        return dict(image='sha256:example', server_arguments={'additional-config': '{}'},
                    server_environment={'VLLM_ROCM_KIMI_ATTN_RES_NORM_CACHE': '0', 'PROFILE': '0'})

    def test_dspark_removes_only_expected_flag(self):
        baseline = self.recipe()
        candidate = copy.deepcopy(baseline)
        candidate['server_arguments']['additional-config'] = json.dumps({'dspark_context_projection_sharding': True})
        self.assertEqual(normalize_recipe(baseline, 67372, 'baseline'), normalize_recipe(candidate, 67372, 'candidate'))
        candidate['server_environment']['PROFILE'] = '1'
        self.assertNotEqual(normalize_recipe(baseline, 67372, 'baseline'), normalize_recipe(candidate, 67372, 'candidate'))

    def test_attnres_binds_flag_and_retains_other_recipe_differences(self):
        baseline = self.recipe()
        with self.assertRaises(AssertionError):
            normalize_recipe(baseline, 67373, 'candidate')
        candidate = copy.deepcopy(baseline)
        candidate['server_environment']['VLLM_ROCM_KIMI_ATTN_RES_NORM_CACHE'] = '1'
        self.assertEqual(normalize_recipe(baseline, 67373, 'baseline'), normalize_recipe(candidate, 67373, 'candidate'))
        candidate['server_arguments']['additional-config'] = json.dumps({'dspark_context_projection_sharding': True})
        self.assertNotEqual(normalize_recipe(baseline, 67373, 'baseline'), normalize_recipe(candidate, 67373, 'candidate'))

    def test_fixture_revision_preserves_assertions_and_random_sequence(self):
        original = 'reference_prefix = prefix.clone()\nreference_blocks = blocks.clone()\ntorch.testing.assert_close(a, b, atol=0, rtol=0)\n'
        revised = same_layout_fixture(original)
        self.assertIn('torch.testing.assert_close(a, b, atol=0, rtol=0)', revised)
        self.assertNotIn('randn', revised)
        with self.assertRaises(AssertionError):
            same_layout_fixture(original + 'reference_prefix = prefix.clone()\n')


class TimingEnvironmentTests(unittest.TestCase):
    def test_unset_and_empty_disable_by_absence(self):
        from benchmark_check import configure_timing_environment
        for env in [{}, {'VLLM_GPU_SYNC_CHECK':'', 'PROFILE':'0'}]:
            configure_timing_environment(env)
            self.assertNotIn('VLLM_GPU_SYNC_CHECK',env)
    def test_strict_instrumentation_is_not_silently_disabled(self):
        from benchmark_check import configure_timing_environment
        for mode in ['warn','error']:
            with self.assertRaises(AssertionError):
                configure_timing_environment({'VLLM_GPU_SYNC_CHECK':mode})

if __name__ == '__main__':
    unittest.main()

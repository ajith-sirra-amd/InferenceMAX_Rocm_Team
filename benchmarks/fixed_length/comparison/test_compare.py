"""Behavior checks for serving-result comparison gates and metric direction."""

import copy
import unittest

from compare import compare_results


def result():
    return {
        "num_prompts": 192,
        "completed": 192,
        "max_concurrency": 48,
        "total_input_tokens": 192 * 8192,
        "total_output_tokens": 192 * 1024,
        "duration": 100,
        "benchmark_outcome": {"status": "passed", "failed": 0},
        "input_lens": [8192] * 192,
        "output_lens": [1024] * 192,
        "ttfts": [1] * 192,
        "itls": [[0.1] * 10 for _ in range(192)],
        "errors": [""] * 192,
        "mean_e2el_ms": 2001,
        "backend": "vllm-chat",
        "model_id": "/model",
        "tokenizer_id": "/model",
        "best_of": 1,
        "request_rate": "inf",
        "burstiness": 1.0,
    }


class ComparisonTests(unittest.TestCase):
    def test_native_e2e_and_duration_based_throughput(self):
        baseline, candidate = result(), result()
        candidate.update(mean_e2el_ms=1900.95, duration=95)
        candidate["output_throughput"] = (
            999999  # Do not trust an inconsistent convenience metric.
        )
        report = compare_results(baseline, candidate)
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["comparison_eligibility"], "screening_only")
        self.assertAlmostEqual(report["mean_request_e2e_ms"]["improvement_percent"], 5)
        self.assertAlmostEqual(
            report["output_tokens_per_second"]["improvement_percent"], 100 / 19
        )
        self.assertEqual(report["mean_request_e2e_ms"]["baseline"], 2001)
        self.assertFalse(report["significance_evaluated"])
        self.assertFalse(report["quality_evaluated"])

    def test_failures_lengths_and_config_are_gated(self):
        mutations = [
            {"completed": 191},
            {"errors": ["failure"] + [""] * 191},
            {"output_lens": [1023, 1025] + [1024] * 190},
            {"input_lens": [8191, 8193] + [8192] * 190},
            {"max_concurrency": 8},
            {"model_id": "/another-model"},
            {"tokenizer_id": None},
            {"request_rate": 48},
            {"mean_e2el_ms": float("nan")},
            {"mean_e2el_ms": 0},
        ]
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                report = compare_results(result(), result() | mutation)
                self.assertEqual(report["status"], "failed")
                self.assertNotIn("mean_request_e2e_ms", report)

    def test_complete_ordered_prompt_identity(self):
        prompts = [f"prompt {index}" for index in range(192)]
        report = compare_results(result(), result(), prompts, copy.copy(prompts))
        self.assertEqual(report["prompt_identity"]["status"], "matched")
        self.assertEqual(
            report["comparison_eligibility"],
            "workload_matched_pending_environment_review",
        )
        self.assertFalse(report["significance_evaluated"])
        reversed_prompts = list(reversed(prompts))
        self.assertEqual(
            compare_results(result(), result(), prompts, reversed_prompts)["status"],
            "failed",
        )
        self.assertEqual(
            compare_results(result(), result(), prompts[:4], prompts[:4])["status"],
            "failed",
        )
        self.assertEqual(
            compare_results(result(), result(), prompts, None)["status"], "failed"
        )

    def test_latency_and_throughput_can_disagree(self):
        report = compare_results(
            result(), result() | {"mean_e2el_ms": 2100, "duration": 90}
        )
        self.assertLess(report["mean_request_e2e_ms"]["improvement_percent"], 0)
        self.assertGreater(report["output_tokens_per_second"]["improvement_percent"], 0)


if __name__ == "__main__":
    unittest.main()

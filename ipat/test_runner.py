import copy
import gzip
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, MagicMock

from container import benchmark_metrics, trace_summary, run
from runner import validate_recipe, resolve_models, cleanup


RECIPE = {"model": "Model", "framework": "vllm", "image": "image:pin", "image_policy": "local",
    "server_argv": ["vllm", "serve", "Model"], "client_argv": ["agentx", "profile"],
    "physical_chips": 8, "concurrency": 48, "duration_seconds": 3600,
    "startup_timeout_seconds": 7200, "client_timeout_seconds": 10800, "environment": {},
    "client_packages": ["agentx-harness==1.0.12"], "health_path": "/health",
    "profile": {"kind": "torch", "seconds": 20, "delay_seconds": 10}}


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_validation_rejects_unknown_or_unbounded_execution(self):
        self.assertEqual(validate_recipe(copy.deepcopy(RECIPE))["concurrency"], 48)
        for changes in ({"physical_chips": 16}, {"concurrency": True}, {"server_argv": "vllm serve Model"},
                        {"environment": {"HF_TOKEN": "do-not-export"}}, {"client_packages": ["agentx-harness"]},
                        {"profile": {"kind": "torch", "seconds": 9999, "delay_seconds": 0}}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validate_recipe(dict(RECIPE, **changes))

    def test_trace_summary_requires_gpu_events_and_keeps_units(self):
        events = [{"ph": "X", "cat": "kernel", "name": "gemm", "dur": 1000, "pid": 10, "args": {"device": 0}},
                  {"ph": "X", "cat": "kernel", "name": "gemm", "dur": 500, "pid": 11, "args": {"device": 1}},
                  {"ph": "X", "cat": "cpu_op", "name": "aten::mm", "dur": 20,
                   "args": {"Input Dims": [[4, 8]], "Input type": ["c10::BFloat16"]}}]
        with gzip.open(self.root / "rank.trace.json.gz", "wt") as f: json.dump({"traceEvents": events}, f)
        summary = trace_summary(self.root)
        self.assertEqual(summary["kernel_count"], 2)
        self.assertEqual(summary["kernels"][0]["total_us"], 1500)
        self.assertEqual(summary["devices"], ["0", "1"])
        self.assertEqual(summary["operators"][0]["Input type"], ["c10::BFloat16"])
        self.assertEqual(trace_summary(self.root / "missing")["kernel_count"], 0)

    def test_benchmark_values_are_not_total_throughput_or_fake_zeroes(self):
        path = self.root / "profile_export_aiperf.json"
        path.write_text(json.dumps({"output_token_throughput": {"avg": 800, "unit": "tokens/sec"},
            "total_token_throughput": {"avg": 90000, "unit": "tokens/sec"},
            "full_response_inter_token_latency": {"p90": 40, "unit": "ms"}}))
        result = benchmark_metrics(path, 8)
        self.assertEqual(result["output_tokens_s_gpu"], 100)
        self.assertEqual(result["p90_full_response_itl_ms"], 40)
        self.assertIsNone(result["p90_ttft_ms"])
        self.assertEqual(result["quality_evaluation"], "not_run")

    def test_models_resolve_existing_snapshot_without_writes(self):
        cache = self.root / "models"; snapshot = cache / "models--org--Model/snapshots/revision"
        snapshot.mkdir(parents=True); (snapshot / "config.json").write_text('{}')
        ref = cache / "models--org--Model/refs/main"; ref.parent.mkdir(); ref.write_text('revision')
        recipe = dict(RECIPE, model_paths={"org/Model": "models--org--Model"},
                      server_argv=["vllm", "serve", "/model-cache/models--org--Model"], client_argv=["agentx"])
        facts = resolve_models(recipe, cache)
        self.assertEqual(facts["org/Model"]["snapshot"], "revision")
        self.assertEqual(recipe["server_argv"][2], "/model-cache/models--org--Model/snapshots/revision")
        self.assertEqual(ref.read_text(), "revision")
        recipe["model_paths"] = {"org/Model": "../escape"}
        with self.assertRaises(ValueError): resolve_models(recipe, cache)

    def test_server_failure_retains_manifest_and_does_not_launch_client(self):
        process = MagicMock(); process.poll.return_value = 1
        capabilities = {"devices": [{"name": "AMD Instinct MI355X"}] * 8}
        with patch("container.probe", return_value=capabilities), patch("container.subprocess.Popen", return_value=process) as popen:
            result = run(dict(RECIPE, port=8000, client_packages=[]), self.root / "pass", "baseline")
        self.assertEqual(result["status"], "failed")
        self.assertIn("Server exited before readiness", result["error"])
        self.assertEqual(popen.call_count, 1)
        self.assertTrue((self.root / "pass/pass.json").is_file())

    def test_post_cleanup_targets_only_the_recorded_container_and_output(self):
        (self.root / "request.json").write_text(json.dumps({"id": "a" * 32}))
        (self.root / "manifest.json").write_text(json.dumps({"image": {"id": "sha256:example"}}))
        with patch("runner.subprocess.run") as command:
            cleanup(self.root)
        self.assertEqual(command.call_args_list[0].args[0], ["docker", "rm", "--force", "ipat-" + "a" * 32])
        self.assertIn(f"type=bind,src={self.root.resolve()},dst=/output", command.call_args_list[1].args[0])
        self.assertNotIn("--device", command.call_args_list[1].args[0])
        (self.root / "request.json").write_text(json.dumps({"id": "invalid"}))
        with patch("runner.subprocess.run") as command, self.assertRaises(ValueError): cleanup(self.root)
        command.assert_not_called()


if __name__ == "__main__":
    unittest.main()

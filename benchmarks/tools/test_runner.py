import copy
import gzip
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch, MagicMock

from container import benchmark_metrics, trace_summary, run
from runner import validate_recipe, resolve_models, cleanup, ownership_command, validate_output, runtime_storage, recover, resume_profile_manifest, require_idle_devices


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
                        {"environment": {"BENCHMARK_WORK_DIR": "/elsewhere"}}, {"client_environment": {"CCACHE_DIR": "/elsewhere"}},
                        {"profile": {"kind": "torch", "seconds": 9999, "delay_seconds": 0}}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validate_recipe(dict(RECIPE, **changes))

    def test_idle_check_requires_explicit_empty_process_list_on_every_device(self):
        idle = [{"gpu": i, "process_list": [{"process_info": "No running processes detected"}]} for i in range(8)]
        require_idle_devices(idle, 8)
        for changed in (idle[:-1], idle[:-1] + [idle[0]],
                        idle[:-1] + [{"gpu": 7, "process_list": [{"pid": 123}]}],
                        idle[:-1] + [{"gpu": 7, "process_list": []}], {"error": "permission denied"}):
            with self.assertRaises(RuntimeError): require_idle_devices(changed, 8)

    def test_separate_profile_stage_retains_only_matching_successful_baseline(self):
        current = {"id": "a" * 32, "recipe_sha256": "frozen"}
        previous = dict(current, passes=[{"pass": "baseline", "status": "completed", "metrics": {"throughput": 123}}])
        result = resume_profile_manifest(previous, current)
        self.assertEqual(result["passes"], previous["passes"])
        self.assertEqual(result["status"], "preparing_profile")
        for changed in ({"id": "b" * 32}, {"recipe_sha256": "different"},
                        {"passes": [{"pass": "baseline", "status": "failed"}]}):
            with self.assertRaises(ValueError): resume_profile_manifest(dict(previous, **changed), current)

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
        (snapshot / "model.safetensors").write_bytes(b'test')
        ref = cache / "models--org--Model/refs/main"; ref.parent.mkdir(); ref.write_text('revision')
        recipe = dict(RECIPE, model_paths={"org/Model": "models--org--Model"},
                      server_argv=["vllm", "serve", "/model-cache/models--org--Model"], client_argv=["agentx"])
        facts = resolve_models(recipe, cache)
        self.assertEqual(facts["org/Model"]["snapshot"], "revision")
        self.assertEqual(recipe["server_argv"][2], "/model-cache/models--org--Model/snapshots/revision")
        self.assertEqual(ref.read_text(), "revision")
        (snapshot / "model.safetensors").unlink()
        with self.assertRaisesRegex(ValueError, "missing weights"): resolve_models(recipe, cache)
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
        self.assertEqual(command.call_args_list[0].args[0], ["docker", "rm", "--force", "benchmark-" + "a" * 32])
        self.assertIn(f"type=bind,src={self.root.resolve()},dst=/output", command.call_args_list[1].args[0])
        self.assertNotIn("--device", command.call_args_list[1].args[0])
        (self.root / "request.json").write_text(json.dumps({"id": "invalid"}))
        with patch("runner.subprocess.run") as command, self.assertRaises(ValueError): cleanup(self.root)
        command.assert_not_called()

    def test_failed_dataset_preparation_does_not_start_the_gpu_server(self):
        capabilities = {"devices": [{"name": "AMD Instinct MI355X"}] * 8}
        recipe = dict(RECIPE, port=8000, client_packages=[], client_setup=[["hf", "download", "dataset"]])
        with patch("container.probe", return_value=capabilities), patch("container.checked", side_effect=RuntimeError("dataset unavailable")), patch("container.subprocess.Popen") as popen:
            result = run(recipe, self.root / "pass", "baseline")
        popen.assert_not_called()
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["error"], "dataset unavailable")

    def test_ownership_walk_preserves_evidence_without_traversing_caches(self):
        files = ["baseline/server.log", "baseline/client/records.jsonl", "profile/traces/rank.json.gz",
                 "work/cache/compiler.bin", "cache/compiler.bin", "hf/dataset.parquet", "home/.cache/dataset.dat", "tmp/compiler.o",
                 "baseline/client-env/bin/python", "profile/client-tools/uv"]
        for name in files:
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
        (self.root / "model-link").symlink_to(self.root / "hf", target_is_directory=True)
        argv = ownership_command(self.root, "image")
        walk = argv[argv.index("/output"):argv.index("-exec")]
        walk = [arg.replace("/output", str(self.root)) for arg in walk]
        paths = subprocess.check_output(["find", *walk, "-print"], text=True).splitlines()
        selected = {str(Path(path).relative_to(self.root)) for path in paths}
        self.assertTrue(set(files[:3]).issubset(selected))
        self.assertFalse(set(files[3:]).intersection(selected))
        self.assertIn("model-link", selected)
        self.assertNotIn("model-link/dataset.parquet", selected)

    def test_storage_accepts_only_checkout_or_available_shared_root(self):
        repo = self.root / 'repo'; repo.mkdir()
        shared = self.root / 'share/ajith/benchmarks'
        with patch('runner.REPOSITORY', repo), patch('runner.SHARED_DATA_ROOT', shared):
            self.assertEqual(validate_output(repo / 'output'), repo / 'output')
            with self.assertRaisesRegex(ValueError, 'unavailable'):
                validate_output(shared / 'runs/123-1')
            self.assertFalse(shared.parent.exists())
            shared.parent.mkdir(parents=True)
            self.assertEqual(validate_output(shared / 'runs/123-1'), shared / 'runs/123-1')
            with self.assertRaises(ValueError): validate_output(self.root / 'outside')
            shared.mkdir()
            (shared / 'escape').symlink_to(self.root, target_is_directory=True)
            with self.assertRaises(ValueError): validate_output(shared / 'escape/output')

    def test_runtime_files_are_separate_from_retained_evidence(self):
        output = self.root / 'runs/123-1'
        with patch('runner.REPOSITORY', self.root):
            work, args, env = runtime_storage(output)
            self.assertEqual(work, output / 'work')
            self.assertEqual(env['BENCHMARK_WORK_DIR'], '/work')
            self.assertEqual(env['XDG_CACHE_HOME'], '/work/cache')
            self.assertEqual(env['TMPDIR'], '/work/tmp')
            self.assertIn(f'type=bind,src={output},dst=/output', args)
            self.assertIn(f'type=bind,src={work},dst=/work', args)
            self.assertEqual({p.name for p in output.iterdir()}, {'work'})
            (work / 'cache').rmdir()
            (work / 'cache').symlink_to(self.root, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, 'inside this run'):
                runtime_storage(output)

    def test_client_install_uses_work_directory_and_retains_manifest(self):
        output, work = self.root / 'evidence/baseline', self.root / 'runtime'
        process = MagicMock(); process.poll.return_value = 1
        capabilities = {'devices': [{'name': 'AMD Instinct MI355X'}] * 8}
        with patch.dict('os.environ', {'BENCHMARK_WORK_DIR': str(work)}), \
             patch('container.probe', return_value=capabilities), \
             patch('container.shutil.which', return_value='/bin/uv'), \
             patch('container.checked') as checked, \
             patch('container.subprocess.Popen', return_value=process):
            result = run(dict(RECIPE, port=8000, client_python='3.11'), output, 'baseline')
        venv_calls = [c.args[0] for c in checked.call_args_list if 'venv' in c.args[0]]
        self.assertEqual(venv_calls[0][-1], str(work / 'baseline/client-env'))
        self.assertTrue((output / 'pass.json').is_file())
        self.assertFalse((output / 'client-env').exists())
        self.assertEqual(result['status'], 'failed')

    def test_shared_recovery_checks_identity_and_excludes_runtime_files(self):
        shared = self.root / 'share/ajith/benchmarks'
        old = shared / 'runs/123-1'; old.mkdir(parents=True)
        (old / 'request.json').write_text(json.dumps({'id': 'a' * 32}))
        (old / 'server.log').write_text('retained')
        (old / 'work').mkdir(); (old / 'work/cache.bin').write_text('regenerable')
        request = {'previous': {'workflow_run_id': 123, 'attempt': 1, 'id': 'a' * 32}}
        output = self.root / 'recovery'
        with patch('runner.SHARED_DATA_ROOT', shared), patch('runner.cleanup') as cleanup_run:
            recover(request, output)
            cleanup_run.assert_called_once_with(old)
            self.assertEqual((output / 'recovered/server.log').read_text(), 'retained')
            self.assertFalse((output / 'recovered/work').exists())
            cleanup_run.reset_mock()
            request['previous']['id'] = 'b' * 32
            with self.assertRaisesRegex(ValueError, 'identity differs'):
                recover(request, self.root / 'bad-recovery')
            cleanup_run.assert_not_called()

    def test_gpu_math_failure_stops_before_client_setup_or_model_start(self):
        capabilities = {"devices": [{"name": "AMD Instinct MI355X"}] * 8,
                        "gpu_error": "HIPBLAS_STATUS_INTERNAL_ERROR"}
        with patch("container.probe", return_value=capabilities), patch("container.checked") as setup, patch("container.subprocess.Popen") as popen:
            result = run(dict(RECIPE, port=8000), self.root / "pass", "baseline")
        setup.assert_not_called()
        popen.assert_not_called()
        self.assertEqual(result["status"], "failed")
        self.assertIn("HIPBLAS_STATUS_INTERNAL_ERROR", result["error"])


if __name__ == "__main__":
    unittest.main()

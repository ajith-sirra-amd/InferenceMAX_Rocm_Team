"""Behavior checks for revision, dependency and candidate-delta gates."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
import subprocess

from runtime import candidate_recipe, load_pair, verify_probe
from check_junit import validate as validate_junit
from build import source_version


class RuntimeTests(unittest.TestCase):
    def test_release_version_requires_pinned_full_ancestry(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source"
            source.mkdir()
            def git(*args, cwd=source):
                return subprocess.check_output(["git", *args], cwd=cwd, text=True,
                                               stderr=subprocess.DEVNULL).strip()
            git("init")
            git("config", "user.name", "Runtime test")
            git("config", "user.email", "test@example.invalid")
            git("commit", "--allow-empty", "-m", "release")
            release = git("rev-parse", "HEAD")
            git("tag", "v1.0.0")
            git("commit", "--allow-empty", "-m", "candidate")
            head = git("rev-parse", "HEAD")
            result = source_version(source, head)
            self.assertEqual(result["release_tag_sha"], release)
            self.assertEqual(result["git_description"], "v1.0.0-1-g" + head)
            with self.assertRaises(ValueError):
                source_version(source, release)
            shallow = Path(tmp) / "shallow"
            git("clone", "--depth", "1", "--no-tags", source.as_uri(), str(shallow))
            with self.assertRaises(ValueError):
                source_version(shallow, head)
            git("tag", "-d", "v1.0.0")
            with self.assertRaises(subprocess.CalledProcessError):
                source_version(source, head)

    def test_junit_requires_executed_new_cases(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "junit.xml"
            path.write_text('<testsuite><testcase name="test_norm_cache_sequence"/></testsuite>')
            self.assertEqual(validate_junit(path, 67373)["status"], "passed")
            path.write_text('<testsuite><testcase name="test_norm_cache_sequence"><skipped/></testcase></testsuite>')
            self.assertEqual(validate_junit(path, 67373)["status"], "failed")
            path.write_text('<testsuite><testcase name="test_unrelated"/></testsuite>')
            self.assertEqual(validate_junit(path, 67373)["status"], "failed")

    def test_junit_rejects_failure_outside_new_cases(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "junit.xml"
            path.write_text('<testsuite><testcase name="test_norm_cache_sequence"/>'
                            '<testcase name="test_existing"><error/></testcase></testsuite>')
            self.assertEqual(validate_junit(path, 67373)["status"], "failed")

    def test_recipe_delta_preserves_settings_and_does_not_mutate_baseline(self):
        recipe = dict(server_arguments={"additional-config": '{"kda_prefill_backend":"fused","other":7}',
                                       "seed": 7}, server_environment={"AITER": "1"})
        original = copy.deepcopy(recipe)
        result = candidate_recipe(recipe, 67372, "candidate")
        self.assertEqual(json.loads(result["server_arguments"]["additional-config"]),
                         {"kda_prefill_backend": "fused", "other": 7,
                          "dspark_context_projection_sharding": True})
        self.assertEqual(result["server_arguments"]["seed"], 7)
        self.assertEqual(result["server_environment"],
                         {"AITER": "1", "VLLM_ROCM_KIMI_ATTN_RES_NORM_CACHE": "0"})
        self.assertEqual(recipe, original)
        attn = candidate_recipe(recipe, 67373, "candidate")
        self.assertEqual(attn["server_arguments"], recipe["server_arguments"])
        self.assertEqual(attn["server_environment"]["VLLM_ROCM_KIMI_ATTN_RES_NORM_CACHE"], "1")

    def test_already_enabled_recipe_is_rejected(self):
        recipe = dict(server_arguments={"additional-config": '{"dspark_context_projection_sharding":true}'},
                      server_environment={})
        with self.assertRaises(ValueError):
            candidate_recipe(recipe, 67373, "baseline")

    def test_pair_rejects_mutable_image_and_mismatched_dependencies(self):
        pair = dict(schema_version=1, pr=67372, source_repository="ajith-sirra-amd/vllm",
                    baseline=dict(source_sha="0e2871508b20815f2e5969f1697a0de5b1b90881",
                                  image="example/base@sha256:" + "a" * 64, dependencies_sha256="c" * 64),
                    candidate=dict(source_sha="c4aea1c4b27bc2a932a177ce8291ea1ba29d6365",
                                   image="example/head@sha256:" + "b" * 64, dependencies_sha256="c" * 64))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pair.json"
            path.write_text(json.dumps(pair))
            self.assertEqual(load_pair(path, 67372)["candidate"]["source_sha"],
                             "c4aea1c4b27bc2a932a177ce8291ea1ba29d6365")
            for change in ({"image": "example/head:latest"}, {"dependencies_sha256": "d" * 64},
                           {"source_sha": "0" * 40}):
                invalid = copy.deepcopy(pair)
                invalid["candidate"].update(change)
                path.write_text(json.dumps(invalid))
                with self.assertRaises(ValueError):
                    load_pair(path, 67372)

    def test_runtime_probe_rejects_wrong_image_revision(self):
        item = dict(source_sha="a" * 40, dependencies_sha256="b" * 64)
        probe = dict(source_sha="a" * 40, dependencies_sha256="b" * 64,
                     python="/opt/runtime-venv/bin/python", source_sha256={"vllm/example.py": "c" * 64})
        self.assertEqual(verify_probe(item, probe, {"org.opencontainers.image.revision": "a" * 40}), probe)
        with self.assertRaises(ValueError):
            verify_probe(item, probe, {"org.opencontainers.image.revision": "d" * 40})


if __name__ == "__main__":
    unittest.main()

"""Check natural-screen scoring, request identity and metric integrity."""
from argparse import Namespace
import contextlib
import io
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from unittest import TestCase, main
from unittest.mock import patch
from urllib.error import HTTPError

QUALITY = Path(__file__).resolve().parent
sys.path.insert(0, str(QUALITY))
import screen
import launch
import compare
from native_spec_decode import parse


def metrics(accepted=0, drafts=0, engine="0", omit=(), extra=False, running=0):
    values = {
        "vllm:spec_decode_num_accepted_tokens_total": accepted,
        "vllm:spec_decode_num_drafts_total": drafts,
        "vllm:num_requests_running": running,
        "vllm:num_requests_waiting": 0,
    }
    result = "".join(f'{name}{{engine="{engine}"}} {value}\n'
                     for name, value in values.items() if name not in omit)
    if extra:
        result += 'vllm:spec_decode_num_drafts_total{engine="1"} 0\n'
    return result


class Response(io.BytesIO):
    status = 200


class EvidenceTests(TestCase):
    def test_answer_scoring(self):
        self.assertEqual(screen.answer("worked 9\n#### 1,250.50"), "1250.5")
        self.assertIsNone(screen.answer("The answer is 9 but no marked final answer"))
        self.assertEqual(screen.answer("#### -1.25e2"), "-125")

    def test_series_integrity(self):
        before = screen.snapshot(metrics(accepted=10, drafts=5), parse)
        screen.compare_series(before, screen.snapshot(metrics(accepted=13, drafts=6), parse))
        for bad in (metrics(accepted=9, drafts=6), metrics(engine="1"), metrics(extra=True)):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                screen.compare_series(before, screen.snapshot(bad, parse))
        two = screen.snapshot(metrics(extra=True), parse)
        with self.assertRaisesRegex(ValueError, "appeared/disappeared"):
            screen.compare_series(two, screen.snapshot(metrics(), parse))
        for bad in (metrics(omit=("vllm:spec_decode_num_drafts_total",)),
                    metrics(omit=("vllm:num_requests_running",)),
                    metrics(omit=("vllm:num_requests_waiting",)),
                    metrics(accepted=float("nan")), metrics(drafts=float("inf")),
                    metrics(running=-1)):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                screen.snapshot(bad, parse)

    def run_capture(self, root, *, initial=None, final=None, invalid_answers=False):
        _, rows = screen.assets()
        calls = {"metrics": 0, "requests": 0}

        def network(request, timeout):
            if isinstance(request, str):
                calls["metrics"] += 1
                text = (initial if initial is not None else metrics()) if calls["metrics"] == 1 else (
                    final if final is not None else metrics(accepted=144, drafts=48))
                return Response(text.encode())
            body = json.loads(request.data)
            index = body["seed"] - 42
            calls["requests"] += 1
            content = "#### " + rows[index]["answer"].split("#### ")[-1]
            reason = "stop"
            if invalid_answers:
                if index == 0:
                    content = "#### -999999"
                elif index == 1:
                    reason = "length"
                elif index == 2:
                    content = "No final answer."
                elif index == 3:
                    raise HTTPError(request.full_url, 503, "mock unavailable", {}, Response(b'{"error":"unavailable"}'))
            return Response(json.dumps({"choices": [{"message": {"content": content},
                            "finish_reason": reason}], "usage": {"completion_tokens": 30}}).encode())

        output = root / "capture"
        with patch.object(screen, "urlopen", side_effect=network), patch.object(screen.time, "sleep"), contextlib.redirect_stdout(io.StringIO()):
            rc = screen.capture(Namespace(endpoint="http://test.local:8000", output=output,
                                           model="moonshotai/Kimi-K3.1-Pro"))
        return rc, json.loads((output / "summary.json").read_text()), calls, output

    def test_complete_collection_and_incomplete_answers(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            root = Path(directory)
            rc, summary, calls, output = self.run_capture(root / "good")
            self.assertEqual((rc, summary["correct"], summary["acceptance_length"]), (0, 48, 4.0))
            self.assertEqual(calls["requests"], 48)
            self.assertEqual(len((output / "responses.jsonl").read_text().splitlines()), 48)
            rc, summary, _, _ = self.run_capture(root / "bad", invalid_answers=True)
            self.assertEqual((rc, summary["correct"], summary["score_denominator"]), (1, 44, 48))
            self.assertEqual((summary["truncated"], summary["missing_or_invalid_answers"]), (1, 3))
            self.assertEqual(summary["request_errors"], 1)
            self.assertEqual(summary["status"], "incomplete")

    def test_bad_initial_metrics_prevent_requests(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            rc, summary, calls, _ = self.run_capture(Path(directory), initial=metrics(omit=("vllm:num_requests_waiting",)))
            self.assertEqual((rc, calls["requests"], summary["responses"]), (1, 0, 0))
            self.assertEqual(summary["missing_or_invalid_answers"], 48)

    def test_invalid_final_counters_preserve_answers(self):
        variants = [metrics(omit=("vllm:spec_decode_num_accepted_tokens_total",)),
                    metrics(engine="1"), metrics(extra=True), metrics(accepted=float("nan")),
                    metrics(accepted=0, drafts=0)]
        for final in variants:
            with self.subTest(final=final), tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
                rc, summary, _, output = self.run_capture(Path(directory), final=final)
                self.assertEqual((rc, summary["correct"], summary["responses"]), (1, 48, 48))
                self.assertFalse(summary["counters_valid"])
                self.assertTrue(summary["counter_errors"])
                self.assertTrue((output / "metrics-after-00.prom").is_file())

    def test_emitted_verifier_gate(self):
        expected = {"method": "dspark", "rejection_sample_method": "block"}
        result = launch.observed_verifier("non-default args: {'speculative_config': " + repr(expected) + "}", expected)
        self.assertTrue(result["expected_natural_config_match"])
        self.assertFalse(result["sa_throughput_recipe_match"])
        with self.assertRaisesRegex(ValueError, "differs"):
            launch.observed_verifier("non-default args: {'speculative_config': {'rejection_sample_method': 'synthetic'}}", expected)
        with self.assertRaisesRegex(ValueError, "No emitted"):
            launch.observed_verifier("Application ready", expected)

    def test_natural_recipe_derivation(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            path = Path(directory) / "base.json"
            path.write_text(json.dumps({"server_environment": {"CORE_SETTING": "kept"},
                "server_arguments": {"max-num-seqs": 48, "speculative-config": json.dumps({
                    "method": "dspark", "num_speculative_tokens": 6,
                    "rejection_sample_method": "synthetic", "synthetic_acceptance_length": 3.57})}}))
            validation = dict(name="test", base_recipe_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                              speculative_overrides=dict(rejection_sample_method="block", remove=["synthetic_acceptance_length"]),
                              dataset_sha256="fixture")
            result = launch.recipe(path, validation)
            spec = json.loads(result["server_arguments"]["speculative-config"])
            self.assertEqual(spec, {"method": "dspark", "num_speculative_tokens": 6,
                                    "rejection_sample_method": "block"})
            self.assertEqual(result["server_environment"], {"CORE_SETTING": "kept"})
            path.write_text("{}")
            with self.assertRaisesRegex(ValueError, "hash changed"):
                launch.recipe(path, validation)

    def test_pair_reports_scored_transitions_and_rejects_mismatched_requests(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            root = Path(directory)
            _, fixture = screen.assets()
            for mode in ("baseline", "moe_front_dspark"):
                target = root / mode
                (target / "quality").mkdir(parents=True)
                def write(name, value):
                    (target / name).write_text(json.dumps(value))
                write("effective-server.json", dict(optimization=mode, expected_natural_config_match=True,
                    sa_throughput_recipe_match=False, sa_config_match=False, environment={"core": "kept"},
                    config={"server_arguments": {"served-model-name": "moonshotai/Kimi-K3.1-Pro"}},
                    argv=["vllm", "serve", "model", "--port", "8000" if mode == "baseline" else "9000"]))
                write("optimization-validation.json", dict(status="passed"))
                write("observed-verifier.json", dict(expected_natural_config_match=True))
                write("quality/summary.json", dict(status="collected", score_denominator=48, correct=47,
                    counters_valid=True, counters_settled=True, verification_steps=100,
                    accepted_draft_tokens=300 if mode == "baseline" else 280,
                    acceptance_length=4.0 if mode == "baseline" else 3.8))
                (target / "quality/metrics-before.prom").write_text(metrics())
                for index in range(4):
                    (target / f"quality/metrics-after-{index:02d}.prom").write_text(
                        metrics(accepted=300 if mode == "baseline" else 280, drafts=100))
                write("image-digests.json", ["image@sha256:abc"])
                (target / "code-sha.txt").write_text("abc123\n")
                requests = [{"id": r["id"], "body": screen.prompt(r, "moonshotai/Kimi-K3.1-Pro")} for r in fixture]
                write("quality/requests.json", requests)
                wrong = 2 if mode == "baseline" else 1
                rows = []
                for i, request in enumerate(requests):
                    content = "#### " + ("-999999" if i == wrong else fixture[i]["answer"].split("#### ")[-1])
                    rows.append(dict(id=request["id"], request=request["body"], correct=i != wrong,
                        extracted_answer=screen.answer(content), response={"choices": [{"message": {"content": content}, "finish_reason": "stop"}]}))
                (target / "quality/responses.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
            result = compare.compare(root / "baseline", root / "moe_front_dspark")
            self.assertEqual(result["baseline_only_correct"], ["gsm8k-test-0001"])
            self.assertEqual(result["candidate_only_correct"], ["gsm8k-test-0002"])
            self.assertTrue(result["review_required"])
            self.assertFalse(result["quality_qualified"])
            summary_path = root / "moe_front_dspark/quality/summary.json"
            summary = json.loads(summary_path.read_text()); summary["correct"] = 48
            summary_path.write_text(json.dumps(summary))
            with self.assertRaisesRegex(ValueError, "Summary score differs"):
                compare.compare(root / "baseline", root / "moe_front_dspark")
            summary["correct"] = 47; summary_path.write_text(json.dumps(summary))
            summary["acceptance_length"] = float("nan"); summary_path.write_text(json.dumps(summary))
            with self.assertRaisesRegex(ValueError, "finite raw counters"):
                compare.compare(root / "baseline", root / "moe_front_dspark")
            summary["acceptance_length"] = 3.8; summary_path.write_text(json.dumps(summary))
            revision = root / "moe_front_dspark/code-sha.txt"
            revision.write_text("different\n")
            with self.assertRaisesRegex(ValueError, "settings or ordered request bodies differ"):
                compare.compare(root / "baseline", root / "moe_front_dspark")
            revision.write_text("abc123\n")
            path = root / "moe_front_dspark/quality/requests.json"
            requests = json.loads(path.read_text()); requests[0]["body"]["seed"] = 100
            path.write_text(json.dumps(requests))
            with self.assertRaisesRegex(ValueError, "Ordered requests differ from pinned"):
                compare.compare(root / "baseline", root / "moe_front_dspark")
            # Even an identical alteration on both arms is not the pinned cohort.
            canonical = [{"id": r["id"], "body": screen.prompt(r, "moonshotai/Kimi-K3.1-Pro")} for r in fixture]
            canonical[0]["body"]["messages"][0]["content"] = "Same altered question on both arms"
            for mode in ("baseline", "moe_front_dspark"):
                (root / mode / "quality/requests.json").write_text(json.dumps(canonical))
                rows_path = root / mode / "quality/responses.jsonl"
                rows = [json.loads(line) for line in rows_path.read_text().splitlines()]
                rows[0]["request"] = canonical[0]["body"]
                rows_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            with self.assertRaisesRegex(ValueError, "Ordered requests differ from pinned"):
                compare.compare(root / "baseline", root / "moe_front_dspark")

    def test_acceptance_recomputed_from_raw_samples(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            root = Path(directory)
            (root / "metrics-before.prom").write_text(metrics())
            for index in range(4):
                (root / f"metrics-after-{index:02d}.prom").write_text(metrics(accepted=300, drafts=100))
            summary = dict(accepted_draft_tokens=300, verification_steps=100, acceptance_length=4.0,
                           counters_settled=True)
            self.assertEqual(compare.acceptance(root, summary), 4.0)
            for invalid in (float("nan"), float("inf"), 5.0):
                with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, "finite raw counters"):
                    compare.acceptance(root, dict(summary, acceptance_length=invalid))
            (root / "metrics-after-03.prom").write_text(metrics(accepted=301, drafts=100))
            with self.assertRaisesRegex(ValueError, "did not settle"):
                compare.acceptance(root, summary)
            for index in range(4):
                (root / f"metrics-after-{index:02d}.prom").write_text(metrics(accepted=701, drafts=100))
            with self.assertRaisesRegex(ValueError, "Invalid raw accepted/drafts"):
                compare.acceptance(root, summary)
            (root / "metrics-before.prom").write_text(metrics(accepted=702, drafts=99))
            with self.assertRaisesRegex(ValueError, "counter reset"):
                compare.acceptance(root, summary)


if __name__ == "__main__":
    main(verbosity=2)

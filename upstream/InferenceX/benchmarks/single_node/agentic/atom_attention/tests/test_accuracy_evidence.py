"""Behavior tests for evaluation-phase FP8 execution evidence."""
import io
import json
import os
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest.mock import patch

from inferencex_atom_attention import accuracy


class AccuracyEvidenceTest(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch.dict(os.environ, {
            "ATOM_FP8_ACCURACY_STRESS": "1", "ATOM_FP8_ACCURACY_ACTIVE_FILE": "/run/active",
        }))
        accuracy._reported_pids.clear()
        self.q = SimpleNamespace(dtype="torch.float8_e4m3fn", shape=(64, 12, 192))
        self.k = SimpleNamespace(dtype="torch.float8_e4m3fn", shape=(256, 12, 192))
        self.v = SimpleNamespace(dtype="torch.float8_e4m3fn", shape=(256, 12, 128))

    def test_only_evaluation_calls_count_and_every_worker_is_required(self):
        output = io.StringIO()
        with redirect_stdout(output), patch("os.getpid", return_value=11):
            with patch("os.path.isfile", return_value=False):
                accuracy.record_accuracy_call(self.q, (self.k, self.v), fp8_projection=True)
            self.assertEqual(output.getvalue(), "")
            with patch("os.path.isfile", return_value=True):
                accuracy.record_accuracy_call(self.q, (self.k, self.v), fp8_projection=True)
                accuracy.record_accuracy_call(self.q, (self.k, self.v), fp8_projection=True)
        proof = accuracy.validate_accuracy_log(output.getvalue(), 1)
        self.assertEqual(proof["records"][0]["pid"], 11)
        self.assertEqual(len(output.getvalue().splitlines()), 1)
        with self.assertRaisesRegex(RuntimeError, "observed 1"):
            accuracy.validate_accuracy_log(output.getvalue(), 2)

    def test_bf16_projection_cannot_produce_fp8_proof(self):
        with patch("os.path.isfile", return_value=True):
            with self.assertRaisesRegex(RuntimeError, "native FP8 projection"):
                accuracy.record_accuracy_call(self.q, (self.k, self.v), fp8_projection=False)

    def test_wrong_dtype_cannot_produce_fp8_proof(self):
        self.q.dtype = "torch.bfloat16"
        with patch("os.path.isfile", return_value=True):
            with self.assertRaisesRegex(RuntimeError, "E4M3 Q/K/V"):
                accuracy.record_accuracy_call(self.q, (self.k, self.v), fp8_projection=True)


if __name__ == "__main__":
    unittest.main()

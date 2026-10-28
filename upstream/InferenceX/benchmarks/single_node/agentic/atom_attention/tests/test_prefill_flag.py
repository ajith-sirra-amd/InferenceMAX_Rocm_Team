"""CPU routing tests; vLLM, Torch, quantization and FMHA are external doubles."""
import importlib
import os
import sys
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

FLAG = "ATOM_USE_FLYDSL_FP8_PREFILL_ATTN"
PACKAGE = "inferencex_atom_attention"
BACKEND = PACKAGE + ".backend"


class PrefillFlagTest(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch.dict(os.environ))
        os.environ.pop(FLAG, None)
        os.environ.pop("ATOM_FP8_ACCURACY_STRESS", None)
        os.environ.pop("ATOM_USE_FLYDSL_FP8_CAUSAL_ATTN", None)

        class NativePrefill:
            def __init__(self, **kwargs):
                self.scale = 1.0

            def run_prefill_context_chunk(self, chunk, q, k, v, out=None):
                return "bf16_output", "fp32_lse"

            def run_prefill_new_tokens(
                self, q, k, v, return_softmax_lse, out=None, output_scale=None
            ):
                return ("bf16_output", "fp32_lse") if return_softmax_lse else "bf16_output"

        modules = {}

        def module(name, **attributes):
            result = ModuleType(name)
            result.__dict__.update(attributes)
            modules[name] = result
            return result

        module("torch", bfloat16="bf16", cuda=SimpleNamespace(
            is_current_stream_capturing=lambda: False,
            device=lambda device: nullcontext(),
        ))
        module("vllm.model_executor.layers.attention.mla_attention", MLACommonImpl=object)
        module("vllm.v1.attention.backends.mla.prefill.aiter_flash_attn",
               AiterFlashAttnPrefillBackend=NativePrefill)
        module("vllm.v1.attention.backends.mla.rocm_aiter_mla",
               AiterMLABackend=object, AiterMLAImpl=object, AiterMLAMetadataBuilder=object)
        self.quantize = Mock(return_value=("q8", "k8", "v8", 1.0, 2.0, 3.0, None, None))
        module(PACKAGE + ".quantize", fused_qkv_per_tensor_quant=self.quantize)
        self.fmha = Mock(return_value=("fp8_output", "fp32_lse"))
        self.supported = Mock(return_value=True)
        self.flydsl = module("aiter.ops.flydsl",
                             flydsl_flash_attn_fp8_func=self.fmha,
                             flydsl_flash_attn_fp8_supported=self.supported)
        self.enterContext(patch.dict(sys.modules, modules))
        # Restore any previous backend module along with the external doubles.
        sys.modules.pop(BACKEND, None)
        with patch.object(sys, "path", [str(Path(__file__).resolve().parents[1]), *sys.path]):
            self.backend_type = importlib.import_module(BACKEND).AtomFlyDSLPrefillBackend
        self.q = SimpleNamespace(
            dtype="bf16", ndim=3, device="test_gpu", shape=(512, 12, 192),
            data_ptr=lambda: 101, stride=lambda: (2304, 192, 1),
        )
        self.k = SimpleNamespace(dtype="bf16", ndim=3, shape=(8192, 12, 192))
        self.v = SimpleNamespace(dtype="bf16", ndim=3, shape=(8192, 12, 128))

    def chunk(self, max_q=512, max_k=8192):
        return SimpleNamespace(
            max_query_len=max_q, max_seq_len=max_k,
            query_start_loc=(0, max_q), cu_seq_lens=(0, max_k),
        )

    def test_disabled_values_use_bf16_without_fp8_api(self):
        for value in (None, "0", "true"):
            with self.subTest(value=value):
                if value is None:
                    os.environ.pop(FLAG, None)
                else:
                    os.environ[FLAG] = value
                with patch.dict(sys.modules, {"aiter.ops.flydsl": None}):
                    backend = self.backend_type()
                    result = backend.run_prefill_context_chunk(
                        self.chunk(), self.q, self.k, self.v,
                    )
                self.assertEqual(result, ("bf16_output", "fp32_lse"))
        self.quantize.assert_not_called()
        self.fmha.assert_not_called()

    def test_enabled_at_threshold_quantizes_and_uses_fp8(self):
        os.environ[FLAG] = "1"
        backend = self.backend_type()
        result = backend.run_prefill_context_chunk(self.chunk(), self.q, self.k, self.v)
        self.assertEqual(result, ("fp8_output", "fp32_lse"))
        self.assertEqual(self.quantize.call_count, 1)
        self.assertEqual(self.fmha.call_count, 1)

    def test_enabled_preserves_bf16_for_small_chunks_and_causal(self):
        os.environ[FLAG] = "1"
        backend = self.backend_type()
        for max_q, max_k in ((511, 8192), (512, 8191)):
            with self.subTest(max_q=max_q, max_k=max_k):
                result = backend.run_prefill_context_chunk(
                    self.chunk(max_q, max_k), self.q, self.k, self.v,
                )
                self.assertEqual(result, ("bf16_output", "fp32_lse"))
        self.assertEqual(
            backend.run_prefill_new_tokens(self.q, self.k, self.v, True),
            ("bf16_output", "fp32_lse"),
        )
        self.quantize.assert_not_called()
        self.fmha.assert_not_called()

    def test_explicit_opt_in_reports_missing_api(self):
        os.environ[FLAG] = "1"
        with patch.dict(sys.modules, {"aiter.ops.flydsl": None}):
            with self.assertRaises(ModuleNotFoundError):
                self.backend_type()

    def test_enabled_unsupported_shape_reports_error(self):
        os.environ[FLAG] = "1"
        self.supported.return_value = False
        backend = self.backend_type()
        with self.assertRaisesRegex(RuntimeError, "FP8 is unsupported"):
            backend.run_prefill_context_chunk(self.chunk(), self.q, self.k, self.v)
        self.quantize.assert_not_called()
        self.fmha.assert_not_called()

    def test_accuracy_stress_routes_small_context_but_keeps_causal_and_flag_gate(self):
        os.environ[FLAG] = "1"
        os.environ["ATOM_FP8_ACCURACY_STRESS"] = "1"
        self.q.shape = (64, 12, 192)
        self.k.shape = (256, 12, 192)
        self.v.shape = (256, 12, 128)
        backend = self.backend_type()
        self.assertEqual(
            backend.run_prefill_context_chunk(self.chunk(64, 256), self.q, self.k, self.v),
            ("fp8_output", "fp32_lse"),
        )
        self.assertEqual(backend.run_prefill_new_tokens(self.q, self.k, self.v, True),
                         ("bf16_output", "fp32_lse"))
        os.environ[FLAG] = "0"
        backend = self.backend_type()
        self.assertEqual(
            backend.run_prefill_context_chunk(self.chunk(64, 256), self.q, self.k, self.v),
            ("bf16_output", "fp32_lse"),
        )

    def test_accuracy_stress_keeps_empty_context_on_native_path(self):
        os.environ[FLAG] = "1"
        os.environ["ATOM_FP8_ACCURACY_STRESS"] = "1"
        backend = self.backend_type()
        self.assertEqual(
            backend.run_prefill_context_chunk(self.chunk(64, 0), self.q, self.k, self.v),
            ("bf16_output", "fp32_lse"),
        )
        self.quantize.assert_not_called()
        self.fmha.assert_not_called()


    def test_causal_reuses_prepared_qkv_and_original_scales(self):
        os.environ[FLAG] = "1"
        os.environ["ATOM_USE_FUSED_KV_FP8_OUTPUT"] = "1"
        os.environ["ATOM_USE_FLYDSL_FP8_CAUSAL_ATTN"] = "1"
        self.k.shape, self.v.shape = self.q.shape, (512, 12, 128)
        for value in (self.q, self.k, self.v):
            value.numel = lambda: 1000
        self.quantize.return_value = ("q8", "k8", "v8", 1.0, 2.0, 3.0, 4.0, 6.0)
        sys.modules["torch"].cat = tuple
        backend = self.backend_type()
        backend._prefill_metadata = self.chunk(512, 512)
        backend.begin_forward(direct_kv=True)
        try:
            result = backend.run_prefill_new_tokens(self.q, self.k, self.v, True)
            self.assertEqual(result, ("fp8_output", "fp32_lse"))
            self.quantize.assert_called_once()
            self.fmha.assert_called_once()
            call = self.fmha.call_args
            self.assertEqual(call.args, ("q8", "k8", "v8"))
            self.assertEqual([call.kwargs[x] for x in ("q_descale", "k_descale", "v_descale")],
                             [1.0, 2.0, 3.0])
            self.assertIs(call.kwargs["causal"], True)
            self.assertEqual(call.kwargs["cu_seqlens_q"], call.kwargs["cu_seqlens_kv"])
            self.assertEqual(backend._gather_scales, (4.0, 6.0))
        finally:
            stats = backend.end_forward()
        self.assertEqual(stats["q_quantizations"], 1)
        self.assertIsNone(backend._q_cache)

    def test_causal_opt_in_without_preparation_stays_bf16(self):
        os.environ[FLAG] = "1"
        os.environ["ATOM_USE_FUSED_KV_FP8_OUTPUT"] = "1"
        os.environ["ATOM_USE_FLYDSL_FP8_CAUSAL_ATTN"] = "1"
        backend = self.backend_type()
        self.assertEqual(backend.run_prefill_new_tokens(self.q, self.k, self.v, True),
                         ("bf16_output", "fp32_lse"))
        self.quantize.assert_not_called()
        self.fmha.assert_not_called()

    def test_oversized_direct_preparation_fails_before_quantization(self):
        os.environ[FLAG] = "1"
        backend = self.backend_type()
        self.q.numel = lambda: 2**31
        backend.begin_forward(direct_kv=True)
        try:
            with self.assertRaisesRegex(ValueError, "32-bit"):
                backend.run_prefill_new_tokens(self.q, self.k, self.v, True)
        finally:
            backend.end_forward()
        self.quantize.assert_not_called()
        self.fmha.assert_not_called()

if __name__ == "__main__":
    unittest.main()

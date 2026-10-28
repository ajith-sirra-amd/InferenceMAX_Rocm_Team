"""Keep native vLLM MLA orchestration and use ATOM's AITER FP8 FMHA API.

This first integration retains native KV expansion and DCP communication.
It deliberately quantizes each call independently: metadata can outlive a
forward pass, so caching Q without a forward-scoped lifetime would be unsafe.
"""
import os

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.attention.mla_attention import MLACommonImpl
from vllm.v1.attention.backends.mla.prefill.aiter_flash_attn import (
    AiterFlashAttnPrefillBackend,
)
from vllm.v1.attention.backends.mla.rocm_aiter_mla import AiterMLABackend, AiterMLAImpl

from .quantize import fused_qkv_per_tensor_quant

logger = init_logger(__name__)


class AtomFlyDSLMLAImpl(AiterMLAImpl):
    def forward_mha(
        self, q, kv_c_normed, k_pe, kv_c_and_k_pe_cache, attn_metadata,
        k_scale, output, output_scale=None,
    ):
        # The native AITER override can bypass the selected prefill backend
        # for uncached requests and run ASM. Use vLLM's common orchestration
        # so BOTH new tokens and cached chunks reach the selected backend.
        return MLACommonImpl.forward_mha(
            self, q, kv_c_normed, k_pe, kv_c_and_k_pe_cache, attn_metadata,
            k_scale, output, output_scale,
        )


class AtomFlyDSLMLABackend(AiterMLABackend):
    @staticmethod
    def get_name():
        return "INFERENCEX_ATOM_FLYDSL_MLA"

    @staticmethod
    def get_impl_cls():
        return AtomFlyDSLMLAImpl


class AtomFlyDSLPrefillBackend(AiterFlashAttnPrefillBackend):
    supported_dtypes = [torch.bfloat16]

    @staticmethod
    def get_name():
        return "INFERENCEX_ATOM_FLYDSL_FP8_PREFILL"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from aiter.ops.flydsl import (
            flydsl_flash_attn_fp8_func,
            flydsl_flash_attn_fp8_supported,
        )
        self._fp8_attention = flydsl_flash_attn_fp8_func
        self._supported = flydsl_flash_attn_fp8_supported
        self._logged_stages = set()

    def _run(
        self, q, k, v, *, cu_q, cu_k, max_q, max_k, causal,
        return_lse, stage, out=None,
    ):
        if any(x.dtype != torch.bfloat16 for x in (q, k, v)):
            raise ValueError("ATOM FlyDSL prefill requires BF16 expanded Q/K/V inputs")
        if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
            raise ValueError("ATOM FlyDSL prefill requires packed 3-D Q/K/V")
        if not self._supported(q.device, q.shape[1], k.shape[1], q.shape[2], v.shape[2]):
            raise RuntimeError(
                f"ATOM FlyDSL FP8 is unsupported: device={q.device}, "
                f"q_heads={q.shape[1]}, k_heads={k.shape[1]}, "
                f"qk_dim={q.shape[2]}, v_dim={v.shape[2]}"
            )
        if q.shape[0] == 0:
            empty = torch.empty((*q.shape[:2], v.shape[2]), device=q.device, dtype=q.dtype)
            if return_lse:
                return empty, torch.empty((q.shape[1], 0), device=q.device, dtype=torch.float32)
            return empty
        # Preserve the vendored ATOM quantizer's stream/device semantics on
        # each TP rank; do not rely on cuda:0 being the current device.
        with torch.cuda.device(q.device):
            q8, k8, v8, qs, ks, vs, _, _ = fused_qkv_per_tensor_quant(q, k, v)
            result = self._fp8_attention(
                q8, k8, v8,
                softmax_scale=self.scale,
                cu_seqlens_q=cu_q,
                cu_seqlens_kv=cu_k,
                max_seqlen_q=max_q,
                max_seqlen_kv=max_k,
                cross_seqlen=True,
                causal=causal,
                return_lse=return_lse,
                q_descale=qs,
                k_descale=ks,
                v_descale=vs,
                out=out,
                stream=None,
            )
            if stage not in self._logged_stages and not torch.cuda.is_current_stream_capturing():
                logger.info(
                    "[ATOM-FP8-ENGAGED] pid=%d stage=%s q=%s k=%s v=%s "
                    "input=BF16 fmha=E4M3 output=BF16 causal=%s lse=%s "
                    "api=aiter.ops.flydsl.flydsl_flash_attn_fp8_func",
                    os.getpid(), stage, tuple(q.shape), tuple(k.shape), tuple(v.shape),
                    causal, return_lse,
                )
                self._logged_stages.add(stage)
            return result

    def run_prefill_new_tokens(
        self, q, k, v, return_softmax_lse, out=None, output_scale=None,
    ):
        if output_scale is not None:
            raise ValueError("ATOM FlyDSL adapter does not support quantized attention output")
        meta = self._prefill_metadata
        return self._run(
            q, k, v, cu_q=meta.query_start_loc, cu_k=meta.query_start_loc,
            max_q=meta.max_query_len, max_k=meta.max_query_len, causal=True,
            return_lse=return_softmax_lse, stage="new_tokens", out=out,
        )

    def run_prefill_context_chunk(self, chunk, q, k, v, out=None):
        return self._run(
            q, k, v, cu_q=chunk.query_start_loc, cu_k=chunk.cu_seq_lens,
            max_q=chunk.max_query_len, max_k=chunk.max_seq_len, causal=False,
            return_lse=True, stage="cached_context", out=out,
        )

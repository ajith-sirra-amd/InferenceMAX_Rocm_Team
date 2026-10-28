"""Keep native vLLM MLA orchestration and use ATOM's AITER FP8 FMHA API.

This first integration retains native KV expansion and DCP communication.
Q reuse is restricted to one forward pass; metadata can outlive that pass.
Causal calls retain BF16, while cached-context calls use FlyDSL FP8.
"""
import os

import torch

from vllm.model_executor.layers.attention.mla_attention import MLACommonImpl
from vllm.v1.attention.backends.mla.prefill.aiter_flash_attn import (
    AiterFlashAttnPrefillBackend,
)
from vllm.v1.attention.backends.mla.rocm_aiter_mla import AiterMLABackend, AiterMLAImpl, AiterMLAMetadataBuilder

from .quantize import fused_qkv_per_tensor_quant

class AtomFlyDSLMLAImpl(AiterMLAImpl):
    def forward_mha(
        self, q, kv_c_normed, k_pe, kv_c_and_k_pe_cache, attn_metadata,
        k_scale, output, output_scale=None,
    ):
        prefill_backend = attn_metadata.prefill.prefill_backend
        prefill_backend.begin_forward()
        try:
            return MLACommonImpl.forward_mha(
                self, q, kv_c_normed, k_pe, kv_c_and_k_pe_cache, attn_metadata,
                k_scale, output, output_scale,
            )
        finally:
            prefill_backend.end_forward()


class AtomFlyDSLMLAMetadataBuilder(AiterMLAMetadataBuilder):
    _reported_pids = set()
    def _init_fp8_prefill_ps_buffers(self, *args, **kwargs):
        # Our forward override never executes native ASM prefill. Disable
        # its plan construction before allocating any PS metadata/scratch.
        # Native decode metadata and its workspaces are still initialized.
        self._fp8_prefill_enabled = False
        pid = os.getpid()
        if pid not in self._reported_pids:
            print(f"[ATOM-METADATA] pid={pid} prefill_ps=disabled decode=native", flush=True)
            self._reported_pids.add(pid)



class AtomFlyDSLMLABackend(AiterMLABackend):
    @staticmethod
    def get_name():
        return "INFERENCEX_ATOM_FLYDSL_MLA"

    @staticmethod
    def get_impl_cls():
        return AtomFlyDSLMLAImpl

    @staticmethod
    def get_builder_cls():
        return AtomFlyDSLMLAMetadataBuilder


class AtomFlyDSLPrefillBackend(AiterFlashAttnPrefillBackend):
    supported_dtypes = [torch.bfloat16]
    _reported_paths = set()
    _reported_reuse_pids = set()

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
        self._supported_configs = set()
        self._q_cache = None
        self._q_quantizations = 0
        self._q_reuses = 0

    def begin_forward(self):
        if self._q_cache is not None:
            raise RuntimeError("Nested ATOM prefill forward scopes are unsupported")
        self._q_cache = {}
        self._q_quantizations = self._q_reuses = 0

    def end_forward(self):
        stats = {"q_quantizations": self._q_quantizations, "q_reuses": self._q_reuses}
        self._q_cache = None
        pid = os.getpid()
        if stats["q_reuses"] and pid not in self._reported_reuse_pids and not torch.cuda.is_current_stream_capturing():
            print(f"[ATOM-Q-REUSE] pid={pid} q_quantizations={stats['q_quantizations']} q_reuses={stats['q_reuses']} lifetime=single_forward", flush=True)
            self._reported_reuse_pids.add(pid)
        return stats

    def _run(
        self, q, k, v, *, cu_q, cu_k, max_q, max_k, causal,
        return_lse, stage, out=None,
    ):
        if any(x.dtype != torch.bfloat16 for x in (q, k, v)):
            raise ValueError("ATOM FlyDSL prefill requires BF16 expanded Q/K/V inputs")
        if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
            raise ValueError("ATOM FlyDSL prefill requires packed 3-D Q/K/V")
        signature = (q.device, q.shape[1], k.shape[1], q.shape[2], v.shape[2])
        if signature not in self._supported_configs:
            if not self._supported(*signature):
                raise RuntimeError(f"ATOM FlyDSL FP8 is unsupported: {signature}")
            self._supported_configs.add(signature)
        if q.shape[0] == 0:
            empty = torch.empty((*q.shape[:2], v.shape[2]), device=q.device, dtype=q.dtype)
            if return_lse:
                return empty, torch.empty((q.shape[1], 0), device=q.device, dtype=torch.float32)
            return empty
        # Preserve the vendored ATOM quantizer's stream/device semantics on
        # each TP rank; do not rely on cuda:0 being the current device.
        with torch.cuda.device(q.device):
            key = (q.data_ptr(), tuple(q.shape), tuple(q.stride()))
            cached_q = self._q_cache.get(key) if self._q_cache is not None else None
            q8, k8, v8, qs, ks, vs, _, _ = fused_qkv_per_tensor_quant(
                q, k, v, quantized_q=cached_q,
            )
            if cached_q is None:
                self._q_quantizations += 1
                if self._q_cache is not None:
                    self._q_cache[key] = (q8, qs)
            else:
                self._q_reuses += 1
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
            path_key = (os.getpid(), stage, q.shape[1], q.shape[2], v.shape[2])
            if path_key not in self._reported_paths and not torch.cuda.is_current_stream_capturing():
                print(
                    f"[ATOM-FP8-ENGAGED] pid={os.getpid()} stage={stage} "
                    f"q={tuple(q.shape)} k={tuple(k.shape)} v={tuple(v.shape)} "
                    f"input=BF16 fmha=E4M3 output=BF16 causal={causal} lse={return_lse} "
                    "api=aiter.ops.flydsl.flydsl_flash_attn_fp8_func",
                    flush=True,
                )
                self._reported_paths.add(path_key)
            return result

    def run_prefill_new_tokens(
        self, q, k, v, return_softmax_lse, out=None, output_scale=None,
    ):
        result = super().run_prefill_new_tokens(
            q, k, v, return_softmax_lse, out=out, output_scale=output_scale,
        )
        path_key = (os.getpid(), "new_tokens", q.shape[1], q.shape[2], v.shape[2])
        if path_key not in self._reported_paths and not torch.cuda.is_current_stream_capturing():
            print(
                f"[ATOM-PREFILL-PATH] pid={os.getpid()} stage=new_tokens "
                f"q={tuple(q.shape)} fmha=BF16 reason=causal_complete_call_regression",
                flush=True,
            )
            self._reported_paths.add(path_key)
        return result

    def run_prefill_context_chunk(self, chunk, q, k, v, out=None):
        return self._run(
            q, k, v, cu_q=chunk.query_start_loc, cu_k=chunk.cu_seq_lens,
            max_q=chunk.max_query_len, max_k=chunk.max_seq_len, causal=False,
            return_lse=True, stage="cached_context", out=out,
        )

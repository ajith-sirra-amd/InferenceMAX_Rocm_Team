"""Keep native vLLM MLA orchestration and use ATOM's AITER FP8 FMHA API.

This first integration retains native KV expansion and DCP communication.
Q reuse is restricted to one forward pass; metadata can outlive that pass.
FlyDSL FP8 is opt-in; eligible cached-context calls use it when enabled.
Causal and small cached-context calls retain BF16.
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
        direct_kv = False
        if prefill_backend._direct_kv_enabled and self.dcp_world_size > 1:
            from .context_fp8 import eligible
            direct_kv = eligible(self, q, kv_c_and_k_pe_cache, attn_metadata)
        prefill_backend.begin_forward(direct_kv=direct_kv)
        try:
            return MLACommonImpl.forward_mha(
                self, q, kv_c_normed, k_pe, kv_c_and_k_pe_cache, attn_metadata,
                k_scale, output, output_scale,
            )
        finally:
            prefill_backend.end_forward()


    def _context_parallel_compute_prefill_context(
        self, q, cache, metadata, k_scale, dcp_world_size,
        fused_mla_kv_concat_fn=None,
    ):
        backend = metadata.prefill.prefill_backend
        if backend._direct_kv_enabled and backend._gather_scales is not None:
            from .context_fp8 import compute_context
            return compute_context(self, q, cache, metadata, k_scale)
        arguments = (q, cache, metadata, k_scale, dcp_world_size, fused_mla_kv_concat_fn)
        if getattr(self, "_dcp_route_checked", False) or torch.cuda.is_current_stream_capturing():
            return super()._context_parallel_compute_prefill_context(*arguments)
        # Pinned native fallback calls kv_b_proj as a module; its fused route
        # expands directly from the weight tensor. Observe the actual call,
        # rather than predicting the route from checkpoint/config flags.
        calls = []
        hook = self.kv_b_proj.register_forward_pre_hook(lambda *unused: calls.append(True))
        try:
            result = super()._context_parallel_compute_prefill_context(*arguments)
        finally:
            hook.remove()
        context = metadata.prefill.chunked_context
        if context is not None and context.chunks:
            route = "fallback_bf16_gather" if calls else "native_fused_fp8_gather"
            weight = getattr(self.kv_b_proj, "weight", None)
            print(
                f"[ATOM-DCP-PATH] pid={os.getpid()} route={route} "
                f"kv_b_dtype={getattr(weight, 'dtype', None)} "
                f"kv_b_module_calls={len(calls)} expanded_kv=BF16",
                flush=True,
            )
            self._dcp_route_checked = True
        return result


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
        self._fp8_enabled = (
            os.environ.get("ATOM_USE_FLYDSL_FP8_PREFILL_ATTN", "0") == "1"
        )
        if self._fp8_enabled:
            from aiter.ops.flydsl import (
                flydsl_flash_attn_fp8_func,
                flydsl_flash_attn_fp8_supported,
            )

            self._fp8_attention = flydsl_flash_attn_fp8_func
            self._supported = flydsl_flash_attn_fp8_supported
        self._direct_kv_enabled = self._fp8_enabled and (
            os.environ.get("ATOM_USE_FUSED_KV_FP8_OUTPUT", "0") == "1"
        )
        self._validate_direct_kv = self._direct_kv_enabled and (
            os.environ.get("ATOM_VALIDATE_FUSED_KV_FP8_OUTPUT", "0") == "1"
        )
        self._prepare_direct_kv = False
        self._gather_scales = None
        self._direct_validations = 0
        self._supported_configs = set()
        self._q_cache = None
        self._q_quantizations = 0
        self._q_reuses = 0

    def begin_forward(self, *, direct_kv=False):
        if self._q_cache is not None:
            raise RuntimeError("Nested ATOM prefill forward scopes are unsupported")
        self._q_cache = {}
        self._prepare_direct_kv = direct_kv
        self._gather_scales = None
        self._q_quantizations = self._q_reuses = 0

    def end_forward(self):
        stats = {"q_quantizations": self._q_quantizations, "q_reuses": self._q_reuses}
        self._prepare_direct_kv = False
        self._gather_scales = None
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
        if not self._fp8_enabled:
            raise RuntimeError(
                "FlyDSL FP8 prefill requires ATOM_USE_FLYDSL_FP8_PREFILL_ATTN=1"
            )
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
        if self._prepare_direct_kv:
            with torch.cuda.device(q.device):
                quant = fused_qkv_per_tensor_quant(q, k, v)
                key = (q.data_ptr(), tuple(q.shape), tuple(q.stride()))
                self._q_cache[key] = (quant[0], quant[3])
                self._q_quantizations += 1
                self._gather_scales = torch.cat((quant[6], quant[7]))
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
        if (
            not self._fp8_enabled
            or chunk.max_query_len < 512
            or chunk.max_seq_len < 8192
        ):
            result = super().run_prefill_context_chunk(chunk, q, k, v, out=out)
            reason = (
                "small_context_conversion_cost" if self._fp8_enabled else "flag_disabled"
            )
            key = (os.getpid(), "cached_context_bf16", reason, q.shape[1], q.shape[2], v.shape[2])
            if key not in self._reported_paths and not torch.cuda.is_current_stream_capturing():
                print(
                    f"[ATOM-PREFILL-PATH] pid={os.getpid()} stage=cached_context "
                    f"q={tuple(q.shape)} k={tuple(k.shape)} fmha=BF16 reason={reason}",
                    flush=True,
                )
                self._reported_paths.add(key)
            return result
        return self._run(
            q, k, v, cu_q=chunk.query_start_loc, cu_k=chunk.cu_seq_lens,
            max_q=chunk.max_query_len, max_k=chunk.max_seq_len, causal=False,
            return_lse=True, stage="cached_context", out=out,
        )

    def _query_fp8(self, q):
        key = (q.data_ptr(), tuple(q.shape), tuple(q.stride()))
        cached = self._q_cache.get(key) if self._q_cache is not None else None
        if cached is None:
            # The vendored quantizer handles empty K/V without reading them.
            quant = fused_qkv_per_tensor_quant(q, q[:0], q[:0, :, :128])
            cached = quant[0], quant[3]
            self._q_quantizations += 1
            if self._q_cache is not None:
                self._q_cache[key] = cached
        else:
            self._q_reuses += 1
        return cached

    def _call_projected_fp8(self, chunk, q8, qs, kv8, out=None):
        k8, v8, ks, vs = kv8
        signature = (q8.device, q8.shape[1], k8.shape[1], q8.shape[2], v8.shape[2])
        if signature not in self._supported_configs:
            if not self._supported(*signature):
                raise RuntimeError(f"Unsupported direct FP8 KV FMHA: {signature}")
            self._supported_configs.add(signature)
        return self._fp8_attention(
            q8, k8, v8, softmax_scale=self.scale,
            cu_seqlens_q=chunk.query_start_loc, cu_seqlens_kv=chunk.cu_seq_lens,
            max_seqlen_q=chunk.max_query_len, max_seqlen_kv=chunk.max_seq_len,
            cross_seqlen=True, causal=False, return_lse=True,
            q_descale=qs, k_descale=ks, v_descale=vs, out=out, stream=None,
        )

    def run_prefill_projected_fp8(self, chunk, q, kv8, out=None):
        if not self._direct_kv_enabled or self._gather_scales is None:
            raise RuntimeError("Direct FP8 KV requires this forward's supplied scales")
        if self._q_cache is None:
            raise RuntimeError("Direct FP8 KV requires an active forward scope")
        if q.dtype != torch.bfloat16 or any(
            x.dtype != torch.float8_e4m3fn for x in kv8[:2]
        ):
            raise ValueError("Direct FP8 KV requires BF16 Q and E4M3 K/V")
        with torch.cuda.device(q.device):
            q8, qs = self._query_fp8(q)
            result = self._call_projected_fp8(chunk, q8, qs, kv8, out)
        key = (os.getpid(), "cached_context_direct_fp8")
        if key not in self._reported_paths and not torch.cuda.is_current_stream_capturing():
            print(
                f"[ATOM-FP8-ENGAGED] pid={os.getpid()} stage=cached_context "
                f"q={tuple(q.shape)} k={tuple(kv8[0].shape)} v={tuple(kv8[1].shape)} "
                "input=BF16_Q_FP8_KV fmha=E4M3 output=BF16 direct_kv=1",
                flush=True,
            )
            self._reported_paths.add(key)
        return result

    def validate_projected_fp8(self, chunk, q, k, v, kv8, actual, actual_lse):
        import json

        k8, v8, ks, vs = kv8
        from vllm._aiter_ops import rocm_aiter_ops

        # Independent HIP kernel; avoids Torch reciprocal boundary rounding.
        reference8 = tuple(
            rocm_aiter_ops.per_tensor_quant(x, torch.float8_e4m3fn, scale)[0]
            for x, scale in ((k, ks), (v, vs))
        )
        for name, value, expected in zip(("K", "V"), (k8, v8), reference8):
            if not torch.equal(value.view(torch.uint8), expected.view(torch.uint8)):
                raise RuntimeError(f"Direct FP8 {name} differs from separate same-scale quantization")
        # Recompute Q from live inputs to detect stale reuse as well as KV errors.
        independent = fused_qkv_per_tensor_quant(q, k, v)
        reference, reference_lse = self._call_projected_fp8(
            chunk, independent[0], independent[3], (*reference8, ks, vs),
        )
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)
        torch.testing.assert_close(actual_lse, reference_lse, rtol=0, atol=0)
        dynamic, dynamic_lse = self._call_projected_fp8(
            chunk, independent[0], independent[3],
            (independent[1], independent[2], independent[4], independent[5]),
        )
        finite_output = torch.isfinite(actual).all() & torch.isfinite(dynamic).all()
        if not finite_output.item():
            raise RuntimeError("Nonfinite optimized attention output")
        if not torch.equal(torch.isneginf(actual_lse), torch.isneginf(dynamic_lse)):
            raise RuntimeError("Optimized attention LSE empty-row mismatch")
        finite_lse = torch.isfinite(dynamic_lse)
        if not torch.isfinite(actual_lse[finite_lse]).all().item():
            raise RuntimeError("Nonfinite optimized attention LSE")
        rms = ((actual.float() - dynamic.float()).square().mean().sqrt()
               / dynamic.float().square().mean().sqrt().clamp_min(1e-6)).item()
        lse_max = ((actual_lse[finite_lse] - dynamic_lse[finite_lse]).abs().max().item()
                   if finite_lse.any().item() else 0.0)
        bf16, bf16_lse = super().run_prefill_context_chunk(chunk, q, k, v)
        if not torch.isfinite(bf16).all().item():
            raise RuntimeError("Nonfinite BF16 attention reference")
        if not torch.equal(torch.isneginf(actual_lse), torch.isneginf(bf16_lse)):
            raise RuntimeError("BF16 reference LSE empty-row mismatch")
        finite_bf16_lse = torch.isfinite(bf16_lse)
        if not torch.isfinite(actual_lse[finite_bf16_lse]).all().item():
            raise RuntimeError("Nonfinite optimized LSE against BF16 reference")
        rms_bf16 = ((actual.float() - bf16.float()).square().mean().sqrt()
                    / bf16.float().square().mean().sqrt().clamp_min(1e-6)).item()
        lse_bf16 = ((actual_lse[finite_bf16_lse] - bf16_lse[finite_bf16_lse]).abs().max().item()
                    if finite_bf16_lse.any().item() else 0.0)
        k_clip = (k.float().abs() > 448 * ks).float().mean().item() if k.numel() else 0.0
        v_clip = (v.float().abs() > 448 * vs).float().mean().item() if v.numel() else 0.0
        if max(rms, rms_bf16) >= 0.10 or max(lse_max, lse_bf16) >= 0.20:
            from .numerical_debug import diagnose_lse
            try:
                diagnose_lse(
                    chunk, self.scale, q, k, v, independent, kv8, actual, actual_lse,
                    bf16, bf16_lse, dynamic_lse,
                )
            except Exception as diagnostic_error:
                print(f"[ATOM-FP8-LSE-DIAGNOSTIC-ERROR] {diagnostic_error}", flush=True)
            raise RuntimeError(
                f"Direct KV numerical gate failed: rms={rms}, lse={lse_max}, "
                f"rms_bf16={rms_bf16}, lse_bf16={lse_bf16}, "
                f"k_clip={k_clip}, v_clip={v_clip}"
            )
        self._direct_validations += 1
        print("[ATOM-FP8-KV-VALIDATED] " + json.dumps({
            "pid": os.getpid(), "call": self._direct_validations,
            "same_scale_kv_exact": True, "same_scale_fmha_exact": True,
            "q_tokens": q.shape[0], "kv_tokens": k.shape[0],
            "rms_vs_dynamic": rms, "lse_max_vs_dynamic": lse_max,
            "rms_vs_bf16": rms_bf16, "lse_max_vs_bf16": lse_bf16,
            "k_clipping_fraction": k_clip, "v_clipping_fraction": v_clip,
        }), flush=True)

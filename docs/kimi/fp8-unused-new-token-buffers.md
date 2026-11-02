# Follow-up: avoid unused new-token FP8 K/V buffers

[??](fp8-unused-new-token-buffers_zh.md)

Status: code-review hypothesis only; not implemented or benchmarked. Keep the currently queued PR #64288 validation at 492080779e9feea19308092b0aba0840e87c6115 unchanged.

In that revision, FP8PrefillContext.prepare calls fused_qkv_per_tensor_quant(q, k, v), then retains Q8/Q descale and the two gather descales. The generated new-token K8 and V8 tensors are unused: AiterFlashAttnPrefillBackend.run_prefill_new_tokens still sends the original BF16 Q/K/V to causal attention.

The helper allocates and writes all three FP8 outputs. For 8,192 tokens, 12 local heads, K dimension 192 and V dimension 128, the unused K8/V8 output totals 30 MiB per applicable layer forward. The large-input path also rereads BF16 K/V in its quantization pass after its separate amax pass. This is a traffic estimate, not measured speedup.

A follow-up can preserve the same amax reductions, positive floor and factor-two gather scales while writing Q8 only. Verify bit-identical Q8 and descales against the established helper for representative strided, zero/empty and changed-input graph cases before serving tests. Do not change scale selection or widen tolerances. Profile the established steady-state agentic workload first to quantify how often this work occurs.

This candidate is separate from new-token FP8 attention and does not require adding the direct-projection diff to #59124. No run has been dispatched for it.

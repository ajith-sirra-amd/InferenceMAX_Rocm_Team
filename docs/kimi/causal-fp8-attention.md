# Causal FP8 attention experiment

**English** | [中文](./causal-fp8-attention_zh.md)

The baseline is completed plugin run 40695099743: 17406.08912 total
input + output tokens/s/GPU. This experiment is separate from native vLLM PR #64288.

Reference: [vLLM #59124](https://github.com/vllm-project/vllm/pull/59124),
inspected head `132f6daa6b17cb4000ef58355943bb39339104f2`.
Use FlyDSL `causal=True`, equal query/key sequence offsets, `cross_seqlen=True`,
and independent original Q/K/V descales. The existing quantizer is unchanged.

`ATOM_USE_FLYDSL_FP8_CAUSAL_ATTN=1` enables causal FP8 only when this forward
already prepares direct-FP8 cached KV. The causal call reuses those Q/K/V tensors;
it does not introduce another quantization pass. The original causal K/V descales
remain separate from the 2x-headroom cached-projection output descales.
Forwards without direct-KV preparation retain BF16 causal attention.
Cached-context thresholds, Q-slice quantization, weight conversion, DCP,
NUMA/CCD pinning, speculative settings and the model remain unchanged.

Fifteen CPU routing/evidence tests passed. GPU diagnostics cover strided inputs,
ragged causal sequences, independent FP64 references with BF16 and identical
FP8 operands, fixed-scale future-token masking, preallocated output, LSE,
single preparation, and graph replay after inputs change.
The existing RMS <0.10 and absolute LSE <0.20 diagnostic gates are unchanged;
passing controlled inputs does not erase the earlier captured-BF16 failure.

Full GSM8K must retain the fixed task and scoring and prove both cached FP8
projection and causal FP8 calls on the same eight workers during evaluation.
The verifier uses a separate causal marker and rejects missing or pre-evaluation
evidence. Full 3600-second performance follows successful diagnostics and GSM8K.
No serving smoke or new performance/accuracy claim is made at preparation time.

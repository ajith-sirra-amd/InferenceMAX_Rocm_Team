# vLLM FP8 prefill PR implementation comparison

**English** | [中文](./vllm-fp8-prefill-pr-comparison_zh.md)

The FMHA portion substantially overlaps the existing PRs. The current experiment
adds native vLLM FP8 up-projection and direct FP8 K/V output. Both kernels already
exist in AITER; the distinct work is their vLLM integration and validation.

Inspected PR #59124 at `132f6daa6b17cb4000ef58355943bb39339104f2`, #62447 at
`cefacabced64eda821bcb49493cc970056d15429`, and the tested plugin at
`14250d6c6c17cd54fb7cd480c74c5c4eb5da2e78`. This is a source comparison; neither
PR was rerun in the current benchmark.

| Area | PR #59124 | PR #62447 | Current tested implementation |
|---|---|---|---|
| FMHA kernel | AITER FlyDSL FP8 | AITER FlyDSL FP8 | AITER FlyDSL FP8 |
| Entry point | Explicit `ROCM_AITER_FLYDSL_FP8` backend | `VLLM_ROCM_MLA_PREFILL_FLYDSL_FP8=1` inside `ROCM_AITER_FA` | InferenceMAX custom plugin; native port not yet submitted |
| Cached K/V supplied to new attention code | Already-expanded BF16 tensors, then dynamically quantized | Already-expanded BF16 tensors, then dynamically quantized | FP8 cache times native FP8 weights, FP32 accumulation, direct FP8 K/V stores |
| Native weight conversion | Unchanged by PR | Unchanged by PR | Explicit `*.self_attn.kv_b_proj: fp8_per_channel`; four native layouts validated |
| K/V scales | Per cached chunk's actual K/V maxima | Per cached chunk's actual K/V maxima | ATOM gather descales from this forward's new-token K/V: `2 * max(amax/448, 1e-6)` |
| Q reuse | Quantize full new-token Q, reuse matching slices in context calls | Re-quantize Q in each attention call | Cache by exact pointer/shape/stride within one forward; different subviews can quantize again |
| Strided quantizer input | Direct stride-aware reads | Makes Q/K/V contiguous first | Stride-aware QKV quantizer; direct cached FP8 K/V avoids separate cached-K/V quantization |
| New-token causal attention | FP8; explicitly bypasses competing ASM path | FP8 when this backend is reached; outer ASM dispatch unchanged | BF16 |
| Cached attention routing | Eligible positive Q/KV lengths, with empty/int32 fallback | Opt-in supported shapes | Normal query >=512 and context >=8192; accuracy stress overrides only positive lengths |
| GSM8K evidence | Author reports 96.21% for both arms | Author states no GSM8K/lm-eval yet | 97.119%, 1281/1319, all eight workers proved native FP8 execution |

Neither PR requires `kv_b_proj` weights to remain BF16. Weight precision may already
be selected by the model or quantization configuration. The missing change in these
PRs is specifically the direct FP8-cache/FP8-weight/FP8-output fusion and handoff;
turning on weight quantization alone does not supply that integration.

## Numerical distinctions

All three use per-tensor E4M3 quantization, but routing, tensor populations used
for scales, and rounding details differ. Their accuracy results cannot certify
this integration or establish byte-identical outputs.

- #59124 compares with BF16 on its random fixtures: cosine >0.998, max LSE
  difference <0.25 and mean <0.01. Its shared SDPA tests use FP8-specific
  `rtol=2**-4, atol=0.3`. Its full-model scores are author-reported and were
  obtained with a different serving configuration (including no spec decode).
- #62447's committed attention tests compare with an FP32 reference on the
  same dequantized FP8 inputs: LSE `atol=1e-3, rtol=0`, output relative norm
  <=0.04. This separates kernel arithmetic from input-quantization error.
- Current native projection and same-FP8-input arithmetic tests passed, as did
  full GSM8K with execution proof. The original BF16 absolute LSE <0.20
  criterion remains unpassed on the captured Kimi chunk. No limit was relaxed.

Reported speedups are not directly comparable: #59124 reports +4.5% with no
spec decode; #62447 reports +1.5% in a two-node disaggregated setup. Its cited
#62444 dependency only changes CPU-offload LRU handling, not projection. The
established 16915.72154 tokens/s/GPU result used BF16 `kv_b_proj`; native FP8
projection performance is being measured by full run 40695099743.

## Upstream scope and remaining opportunities

A second dynamic-QKV-plus-FP8-FMHA PR would substantially duplicate existing work.
The useful extension is native FP8 `kv_b_proj` plus direct FP8 output integration
on an existing FP8 backend. #59124 is the closer foundation because it already
has Q reuse, strided quantization and explicit ASM bypass; its merge conflicts
need resolution. Preserving the tested BF16 causal routing and gather-scale
policy requires explicit design, not an assertion that the paths are identical.

Two additional opportunities are visible in the current implementation:
1. Reuse full-Q slices as #59124 does. Changing from per-view Q scales to full-Q
   scales can change FP8 values, so numerical and model validation are required.
2. During BF16 new-token causal attention, scale preparation currently writes
   temporary FP8 K/V that are not consumed. A Q-output/KV-scale-only preparation
   could avoid those stores, provided it preserves the exact scales and Q output.
Neither change is installed in the active run.

The requested native flag name follows the vLLM convention:

```bash
export VLLM_ROCM_USE_AITER_FLYDSL_FP8_PREFILL_ATTN=1
```

Default: `0`. Weight quantization remains explicit in `--quantization-config`.
This is the name selected for the native port, not a flag already present in the
tested plugin. The user's vLLM fork main matches upstream main at
`0beab6c251e8046c0c6cf2fef114cd23b58cda74`; no upstream PR for this work is open.

## Sources

- [#59124 backend](https://github.com/vllm-project/vllm/blob/132f6daa6b17cb4000ef58355943bb39339104f2/vllm/v1/attention/backends/mla/prefill/aiter_flydsl_fp8.py) and [quantizer](https://github.com/vllm-project/vllm/blob/132f6daa6b17cb4000ef58355943bb39339104f2/vllm/v1/attention/ops/triton_per_tensor_fp8_quant.py).
- [#62447 backend and quantizer](https://github.com/vllm-project/vllm/blob/cefacabced64eda821bcb49493cc970056d15429/vllm/v1/attention/backends/mla/prefill/aiter_flash_attn.py).
- [#59124 tests](https://github.com/vllm-project/vllm/blob/132f6daa6b17cb4000ef58355943bb39339104f2/tests/kernels/attention/test_rocm_aiter_mla_flydsl_fp8_prefill.py) and [#62447 tests](https://github.com/vllm-project/vllm/blob/cefacabced64eda821bcb49493cc970056d15429/tests/kernels/attention/test_rocm_aiter_mla_flydsl_fp8_prefill.py).
- [Current implementation audit](./native-fp8-up-projection-audit.md) and [GSM8K proof](../../results/atom-attention/run40688077118/gsm8k_evidence.json).

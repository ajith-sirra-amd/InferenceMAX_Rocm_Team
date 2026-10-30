# vLLM FP8 prefill PR implementation comparison

**English** | [??](./vllm-fp8-prefill-pr-comparison_zh.md)

The distinct work in [PR #64288](https://github.com/vllm-project/vllm/pull/64288)
is the native handoff from FP8 cache ? FP8 projection weights to direct FP8 K/V
and cached-context FP8 FMHA. Both kernels already exist in AITER; this PR
integrates and validates their use in vLLM.

Source review on 2026-10-30: #59124 is open at
`132f6daa6b17cb4000ef58355943bb39339104f2`; #62447 is closed, unmerged, at
`cefacabced64eda821bcb49493cc970056d15429`; #64288 is draft at
`2f946cbab5000dedef23b117df991417e8917315`.
Neither reference PR has been benchmarked in the current harness.

| Area | #59124 | #62447 | #64288 |
| --- | --- | --- | --- |
| FMHA | AITER FlyDSL FP8 | AITER FlyDSL FP8 | AITER FlyDSL FP8 |
| Selection | `ROCM_AITER_FLYDSL_FP8` backend | FP8 env flag inside `ROCM_AITER_FA` | Default-off `VLLM_ROCM_USE_AITER_FLYDSL_FP8_PREFILL_ATTN` |
| Cached K/V before FMHA | BF16 expanded output, dynamically quantized | BF16 expanded output, dynamically quantized | Direct FP8 projection output |
| Weight conversion | Unchanged by PR | Unchanged by PR | Explicit native `kv_b_proj: fp8_per_channel`; no second quantizer |
| K/V scales | Actual cached-chunk maxima | Actual cached-chunk maxima | ATOM gather policy: `2 * max(new_token_amax/448, 1e-6)` |
| Q reuse | Full-Q quantization reused by slices | Re-quantizes each call | Exact pointer/shape/stride reuse within one forward; distinct slices may re-quantize |
| Strided quantization | Direct reads | Contiguous copies | Direct reads; cached K/V need no separate quantization |
| New-token attention | FP8 with ASM bypass | FP8 if this backend is reached; outer ASM unchanged | BF16 |
| Cached eligibility | Positive lengths with empty/int32 fallback | Supported shapes under opt-in | Query ?512, context ?8192; disclosed accuracy stress uses positive lengths |

Neither reference PR requires BF16 projection weights. Ordinary weight
quantization can be enabled independently. The distinguishing change is direct
FP8 projection output and its FMHA handoff.

## Validation and limits

#64288 passed eight native GPU kernel tests at `332d6c9`; the current head only
formats two test expressions and has the same executable AST. All applicable
targeted static checks passed at `2f946cb`. Full native GSM8K and serving/runtime
validation remain pending; see the [current task list](./fp8-fmha-tasklist.md).

The 97.119% GSM8K and 17,406.08912 total tokens/s/GPU results belong to the earlier
plugin prototype, not the native PR. The separate prototype new-token FP8
experiment measured 15,264.82341 tokens/s/GPU (-12.30182%) and is excluded.
The captured historical-BF16 absolute LSE <0.20 gate remains unresolved.

All three quantize to per-tensor E4M3, but scale populations and rounding differ:
- #59124's random-fixture tests compare against BF16: cosine >0.998, maximum
  LSE difference <0.25 and mean <0.01. Shared SDPA tests use FP8-specific
  `rtol=2**-4, atol=0.3`. Its author-reported 96.21% GSM8K used another
  configuration without speculative decoding.
- #62447 compares attention against FP32 using the same quantized operands:
  LSE `atol=1e-3, rtol=0`, output relative norm ?0.04.
- Our native tests retain their original tolerances. Same-operand arithmetic
  checks do not close the separate captured-BF16 discrepancy.

Author-reported performance is not a matched comparison: #59124 reports +4.5%
without speculative decoding; #62447 reports +1.5% in two-node disaggregation.

## Reviewer-requested comparisons

Run four arms on one common upstream base: #59124 alone and plus direct
projection; #62447 alone and plus direct projection. Keep model checkpoint,
native FP8 weight targets, eight MI355X, TP8/DCP8, C48, CPU/CCD binding,
offload, graph configuration, MTP and replay inputs/duration identical within
each pair. Verify actual backend and projection/FMHA execution in every arm.

Integrating direct projection must preserve the chosen reference's new-token
routing and Q reuse. The ATOM output-scale policy differs from each reference's
cached-chunk dynamic scales; document that difference and validate it rather
than calling the compared outputs identical. Report total/output tokens/s/GPU,
latency and measured errors, with raw artifacts and source identities.

#64288 has no source dependency on either PR. The requested flag discussion
remains open; native NUMA/CCD changes are postponed.

## Later opportunities

Reuse of full-Q slices changes scale populations and needs accuracy validation.
By contrast, the current preparation writes temporary FP8 new-token K/V that
BF16 causal attention does not consume. A Q-output/KV-scale-only preparation
could remove those stores while preserving exactly the same Q output and scales.
Neither change is adopted; correctness and native end-to-end evidence come first.

## Sources

- [#59124 backend](https://github.com/vllm-project/vllm/blob/132f6daa6b17cb4000ef58355943bb39339104f2/vllm/v1/attention/backends/mla/prefill/aiter_flydsl_fp8.py).
- [#62447 backend](https://github.com/vllm-project/vllm/blob/cefacabced64eda821bcb49493cc970056d15429/vllm/v1/attention/backends/mla/prefill/aiter_flash_attn.py).
- [Native audit](./native-fp8-up-projection-audit.md).

# Direct FP8 K/V output validation

**English** | [中文](./direct-kv-fp8-validation_zh.md)

The opt-in serving experiment removes the expanded BF16 K/V allocation and its
separate quantization pass for eligible cached-context chunks. It retains BF16
`kv_b_proj` weights and operands, FP32 dot accumulation, and native BF16 rounding
inside registers before the final FP8 store. Q uses the existing ATOM dynamic
quantizer. This is not FP8 projection arithmetic.

Enable both `ATOM_USE_FLYDSL_FP8_PREFILL_ATTN=1` and
`ATOM_USE_FUSED_KV_FP8_OUTPUT=1`. Unset or zero direct-output flag preserves the
existing projection path. Cached chunks require max Q >=512 and max KV >=8192,
native fused DCP eligibility, and supported BF16 weight geometry. Causal attention
and smaller cached chunks retain BF16 FMHA. New-token K/V supply ATOM's gather
descales: twice the new-token descale, with its existing minimum clamp.
Scales and quantized Q live only within one forward.

## Exact reference and rounding diagnostic

[Run 40668922297](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/40668922297)
found six Torch reciprocal/multiply rounding differences among 12,582,912 V values
at cache scale 0.137. Candidate versus native AITER HIP static quantization had
zero differences. The dynamic variant also matched the existing quantizer exactly.
For example, BF16 V=0.0001678466796875 with descale=0.02455357275903225
gives a scaled FP64 value of 0.0068359371295876326. The candidate and AITER produce
0.005859375; Torch reciprocal/multiply produces 0.0078125.
The global relative RMS difference was 3.0243e-08.

The updated exact same-scale reference is AITER HIP `per_tensor_quant`.
The Torch boundary diagnostic remains reported separately. The projection kernel
was not changed to accommodate the Torch reference, and exact byte checks remain.

## Validation sequence

1. `runner-diagnostics-only=true` runs `direct_kv_diagnostics.py` in the pinned
   image. It measures the FP8-cache -> BF16 projection -> FP8 chain at cache scales
   0.125 and 0.137 against an independent FP64 projection, then runs the actual
   backend's new-token scale preparation, cached FP8 attention, Q reuse, cleanup,
   and changed-input graph replay.
2. `attention-path-probe-only=true` runs real Kimi requests with 32K and 128K
   shared prefixes. The workflow enables `ATOM_VALIDATE_FUSED_KV_FP8_OUTPUT=1`;
   the recipe adds `--enforce-eager` because validation uses host assertions.
   Every optimized cached call compares K/V bytes and attention output/LSE exactly
   with separate same-scale AITER quantization, recomputing Q from live BF16 input.
   It also checks attention relative RMS <0.10 and LSE max error <0.20 against the
   prior dynamic-scale FP8 path, and reports K/V clipping.
3. The probe requires output and successful validation markers on all eight
   workers. Only after inspecting that evidence may graph-enabled performance run.

These tensor gates are inherited diagnostic thresholds, not a model-accuracy
percentage. The prior synthetic stress case exceeded the RMS gate when cached
ranges were four times larger than the new-token range. ATOM scale reuse therefore
requires model-derived validation; algorithm reuse alone does not establish it.

The workflow explicitly supplies both flags and forwards them through the launcher.
Normal performance execution sets validation to zero; host comparisons are absent.
The same compiled GPU operations are separately exercised under graph replay.

SA's fixed GSM8K evaluation remains unchanged. Its previous 97.1948% strict score
used BF16 attention and proves neither this path nor the original FP8-FMHA path.
No serving numerical or performance result is claimed for this integration yet.
The five existing CPU routing tests pass; Python compilation and YAML parsing pass.
Shell validation and actual GPU checks execute in the diagnostics job.

See [projection evidence](./projection-fp8-experiment.md) for the initial kernel
timings and synthetic errors. Those timings exclude serving and DCP communication.

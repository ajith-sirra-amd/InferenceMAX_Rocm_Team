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
   prior dynamic-scale FP8 path and a separate BF16 attention call, and reports K/V clipping.
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


## Completed GPU preflight

[Run 40670162135](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/40670162135), source 7e24cd6a5c7eb7d324ea106c76f7a160a840e687, passed exact same-scale K/V bytes, actual-backend Q reuse and cleanup, and changed-input graph replay. Both cache scales had zero clipping. BF16 projection relative RMS versus FP64 was 0.136–0.175%; final supplied-scale FP8 K/V versus BF16 projection was 2.645–2.753%. Backend attention RMS versus the old dynamic-scale path was 3.580% eager and 4.148% after changed-input graph replay.

[Structured evidence](../../results/atom-attention/run40670162135/summary.json). The additional independent BF16 attention comparison runs in the direct-KV preflight before model startup in the serving smoke. Real-model validation remains pending. First attempt 40669975181 stopped before numerical checks on a diagnostic import-scoping error; that was fixed without changing the kernel or thresholds.


A full performance run can be dispatched with after-run-id set to its preceding numerical run ID and ref set to the identical source SHA. The workflow waits before GPU scheduling and proceeds only on success. A failed prerequisite prevents performance execution. An empty value preserves immediate execution.


## Real Kimi LSE gate failure

Run 40670763935 passed the isolated backend preflight but failed the real-model maximum absolute LSE error gate of 0.20. Same-scale K/V bytes and attention exact comparisons passed first; failing calls reported no K/V clipping and approximately 2–3% output RMS error, but some maximum LSE errors versus BF16 were about 0.8–1.0. Full run 40671001137 was blocked before GPU allocation by this failed prerequisite.

The follow-up adds failure-only diagnostics: independent FP64 attention on the worst query/head rows separates Q quantization, K quantization, and FMHA arithmetic errors. One rank-zero compressed-cache fixture from the synthetic prefix is retained for isolated reproduction. Original gates remain unchanged; diagnostic failure is not acceptance.

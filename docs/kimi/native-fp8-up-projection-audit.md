# Native FP8 up-projection audit

**English** | [中文](./native-fp8-up-projection-audit_zh.md)

The established 16915.72154 tokens/s/GPU run used BF16 kv_b_proj.
Its cached attention path was FP8 cache -> BF16 projection -> BF16 K/V ->
dynamic FP8 K/V -> FP8 FMHA. It did not execute FP8 projection arithmetic.

The BF16 direct-output prototype still used BF16 multiplication, rounding the
FP32 accumulator to BF16 in registers before storing FP8. Its eligibility check
explicitly rejected FP8 weights. Enabling weight quantization alone would have
fallen back to native decompression instead of invoking ATOM's FlyDSL projection.

## ATOM recipe and kernel

Pinned ATOM 773eb36254e43060bfd5fabcab7d3840bf591358, recipe
recipes/Agentic-Kimi-K3.md, enables global ptpc_fp8 online quantization and does
not exclude kv_b_proj. It enables both FlyDSL gather/projection and FP8 FMHA.
Its Kimi BF16 fallback comment covers configurations without online conversion.

The adapter uses unchanged AITER bbef5a6f2d3733b293c80d2cee6a8d464e88bc01
gather_kv_b_proj_flydsl: E4M3 cache times E4M3 weight, FP32 accumulation and direct
E4M3 K/V stores with caller-supplied scalar descales. It does not round through BF16.
New-token scale preparation and cached-attention Q quantization retain the existing
ATOM policy. Causal/small-chunk attention retains the established hybrid routing.

## vLLM integration

The recipe adds "*.self_attn.kv_b_proj": "fp8_per_channel" to its existing targets.
Native vLLM Fp8PtpcOnlineLinearMethod performs conversion and packing; the adapter
only obtains views of the resulting buffers. No second weight quantizer is added.

| Native linear kernel | Native storage | Adapter |
|---|---|---|
| AiterHipbMMPerTokenFp8ScaledMMLinearKernel | Transposed shuffled [N,K], scales [1,N] | Transpose the view back; flatten scales |
| AiterPreshuffledPerTokenFp8ScaledMMLinearKernel | Shuffled [N,K] | Pass directly with preshuffle enabled |
| AiterPerTokenFp8ScaledMMLinearKernel | Plain [N,K] | Pass directly with preshuffle disabled |
| RowWiseTorchFP8ScaledMMLinearKernel | Plain transposed [K,N] | Transpose back; preshuffle disabled |

Unsupported kernel layouts fail explicitly. E4M3 dtype, contiguous [heads*256,512]
storage, FP32 per-channel scales, no bias, cache dimensions, index dtype and device
are checked. Native DCP byte gather and row ordering remain in use. Smaller chunks
in an eligible forward use FP8 projection with BF16 output for BF16 attention.
The weight change also affects new-token projection and native absorbed decode
weights; its numerical scope is broader than cached attention.

## Evidence required before serving/performance

Readiness is not invocation. [ATOM-FP8-PROJECTION-READY] reports loaded parameters;
[ATOM-FP8-UP-PROJECTION] is emitted only after the actual FlyDSL call.
The vllm bench serve prefix probe requires these and projection/attention
validation markers on all eight workers. It cannot pass by silently falling back.

The GPU preflight executes native online conversion and all four native packing
implementations with sparse, exactly quantizable weights and hand-computable
outputs. It also executes native linear selection, the actual backend weight hook,
FP8 projection, attention, scale/Q lifetime and changed-input graph replay.
FP64 dequantized-FP8 projection is an independent arithmetic reference. Its output
quantization checks require RMS <0.04, normalized max error <0.07 and zero clipping.
The previous BF16 prototype retains exact same-scale HIP byte checks.
The original attention gates, RMS <0.10 and absolute LSE <0.20, are unchanged.

CPU flag routing passed five tests and Python syntax checks passed during preparation.
Isolated GPU validation passed (details below). Real-model validation is pending. No performance
improvement or model-accuracy pass is claimed for it.

## Real-model numerical finding

Run 40678558488 failed, saving atom_fp8_failure.pt. On 64 inspected rows from
16 failing calls across eight workers, maximum FP8-FMHA LSE error against FP64 on
identical FP8 inputs was 0.00042494, while Q/K quantization shifted LSE by as much
as 1.04069066. The old dynamic-scale path also exceeded the 0.20 gate, reaching
0.93494225 versus BF16. Some per-row attention-output RMS errors against identical
FP8 inputs reached 0.13575, so LSE agreement alone does not establish FMHA correctness.

The old synthetic numerical passes remain valid for their inputs. GSM8K run
40658068830 passed with BF16 attention and does not prove this FP8 path's accuracy.
The captured-input replay investigates scales and softmax rescaling before another
serving smoke or performance run. A completed diagnostic is not numerical acceptance.

## GPU results and captured-input reference follow-up

Run [40682885241](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/40682885241)
at 58882515b58010bf1411d164c2cb44fab6e049a3 passed all four native weight
layouts, actual selected AiterPreshuffledPerTokenFp8ScaledMMLinearKernel,
FP8 x FP8 projection to FP8 FMHA, and changed-input graph replay.
Projection RMS versus FP64 was approximately 0.0262--0.0266, with zero clipping.
This is isolated GPU evidence; real Kimi serving and model accuracy remain unpassed.

Run [40682306781](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/40682306781)
completed the earlier captured-input diagnostic. Supplied scales, dynamic scales,
eager softmax rescaling, and power-of-two scales all failed the original 0.20
absolute LSE gate. Diagnostic job success is not numerical acceptance.

The next fixture replay uses attention_reference.py to compute FP64 attention for
every query and head in the saved chunk. It reports Q-only, KV-only and native
FP8-weight effects separately. Both old BF16 projection and native FP8 projection
are tested, using captured scales or diagnostic scales computed from FP64 maxima.
The latter is diagnostic only; it is not proposed as a production FP64 operation.
Both lazy-rescale and denormal-handling settings are measured. Original BF16
kernel gates remain unchanged and are reported independently from same-FP8-input
arithmetic checks. Quantizer parity against contiguous AITER HIP inputs is
reported without hiding one-ULP differences.

Related upstream PRs, inspected at their current heads:
- vllm-project/vllm#59124 (132f6daa6b17cb4000ef58355943bb39339104f2), open:
  dynamic per-tensor Q/K/V, stride-aware quantizer and Q reuse for gfx950.
  Author reports GSM8K 0.9621 for both arms; this does not validate our changes.
- vllm-project/vllm#62447 (cefacabced64eda821bcb49493cc970056d15429), open:
  opt-in ROCM_AITER_FA FP8, contiguous Q/K/V copies and dynamic per-tensor scales.
  No GSM8K reported. Identical-input references separate FMHA from quantization.
- vllm-project/vllm#50181 is a Blackwell backend-selection fix, not an MI355X
  FP8 rounding correction. No upstream code or threshold has been imported here.

## Full captured-chunk result: run 40684025796

At source 5975367b961baa7f0ce957f37bce25c0be5a18d7, all 16 same-FP8-input
arithmetic cases passed; zero passed the original BF16-difference gate.
Q/K/V quantization matched the same-scale contiguous AITER HIP quantizer exactly
over 17,694,720 / 35,389,440 / 23,592,960 elements.
Native FP8 projection passed for supplied and diagnostic dynamic scales with zero
clipping. With supplied scales its K/V RMS versus FP64 was 0.02239 / 0.02647.

On the full captured chunk, Q-only rounding changed LSE by up to 0.513783;
KV-only rounding by 0.370814; native weight conversion before final K/V quantization
by 0.053012. Native FP8 projection with supplied scales plus FP8 FMHA had output RMS
0.028700 and max LSE difference 0.934950 versus the historical BF16 kernel.
Removing the BF16 intermediate therefore does not fix the failed LSE criterion.
Disabling lazy rescale improves same-input FMHA RMS from about 0.0129 to 0.0030,
but does not fix Q/K quantization and is not applied to serving.
Denormal-handling changes were bit-identical in these tests.

The next diagnostic separates NoPE and RoPE rounding and records the worst rows.
It also tests per-request key centering: Kc = K - mean(K). Attention output is
unchanged in exact arithmetic; LSE is restored by adding scale * dot(Q, mean(K)).
The identity is checked independently before FP8 tests. The prototype uses BF16
centered K and FP64 correction, so it is neither a production optimization nor
evidence of performance. No serving thresholds or acceptance limits are changed.

## RoPE localization and next diagnostic

Run 40684733665 at f1c199b5cb3209e4ec0a43edebf55e0324585ec1 completed.
Maximum LSE shifts from quantizing one part were: Q NoPE 0.158969,
Q RoPE 0.512855, K NoPE 0.106059, K RoPE 0.370712.
At the worst row, RoPE contributed approximately -0.729 to the peak-score change,
while NoPE contributed approximately -0.00165. The original top key had probability
0.99547. The positional columns dominate this captured failure.

Key-centering's unquantized identity passed (max LSE error 1.78e-14).
The FP8 prototype lowered max LSE difference from 0.727297 to 0.479201,
still failing 0.20. It is not promoted to serving.

The next diagnostic applies the same normalized 64-column Walsh-Hadamard matrix
to Q/K RoPE, with and without one fixed random sign vector (seed 20261028).
The unquantized attention identity is verified first. Original gates are retained.
For native FP8 weights the actual ATOM projection computes NoPE and V, and only
RoPE is replaced from the original compact cache after an FP32 reference transform.
This does not claim fused-transform availability or speed. BF16 transform rounding
for Q is explicit. This is an experiment beyond the inspected ATOM/vLLM PR paths.

## Rotation result and residual-column diagnostic

Run 40685301834 at a5a4d299d8ce5e5e14759bcca02f3957fbc7f889 passed all eight
same-input arithmetic checks but failed every original BF16 gate.
Unsigned RoPE rotation with native FP8 weights reached max LSE 0.536367;
the signed variant reached 0.635032. Neither is adopted.

The next diagnostic preserves primary cached RoPE codes using a K descale aligned
to the compact cache descale by an integer power of two. It adds 64 FP8 columns:
Q carries 16 times its scaled RoPE rounding residual, while K duplicates RoPE
divided by 16. The 256-column dot product therefore includes a residual correction
without changing AITER's FP8 FMHA implementation. The original model softmax scale
is passed explicitly; it must not change to 1/sqrt(256).
All tensors consumed by FMHA remain FP8. Native FP8 projection is still real.
No claim is made that the extra packing traffic or 256-column attention is faster.
Kernel-only timings exclude packing and projection, and are labelled accordingly.
This is diagnostic work, not production integration or model-accuracy evidence.

## Residual-column result and NoPE balancing diagnostic

Run 40685906330 at c53e307afbaff423bd0d3fc16f18d3299afdc2b0 completed.
The cache-aligned primary K RoPE was exact; there was zero clipping and no
underflow in duplicated RoPE. Q RoPE reconstruction RMS was 0.0006364.
Native FP8 projection with 256-column residual FMHA reached max LSE 0.217426
and output RMS 0.024845 (lazy) or 0.021295 (eager), still failing 0.20.
Same-input arithmetic passed in all eight cases. Native residual FMHA took
0.5778 ms versus 0.5568 ms for the old 192-column FMHA, excluding packing.

Next: model-independent per-head NoPE balancing, Q'=Q/c and K'=cK, with
c=sqrt(amax(Q_NoPE)*K_scale/(amax(K_NoPE)*Q_scale)).
Native projection changes only its temporary per-channel descales; packed FP8
weights are not quantized again. RoPE residual handling is unchanged.
The diagnostic uses captured K maxima, so passing it would not establish a
production scale policy. Original numerical limits remain unchanged.

## Current outcome and next full-model validation

All six captured-input diagnostic runs completed. Native FP8 projection and
same-input FMHA arithmetic passed. The original BF16 LSE gate remains unpassed
for native FP8 weights. Run 40686549777 at 6d0f6947d8454751041e8d6a09249682a6b7528c
measured 0.198254 for the balanced BF16-weight prototype (pass on this fixture),
and 0.208252 for the native FP8-weight prototype (fail). Neither experimental
correction is installed in serving. Do not treat diagnostic job success as acceptance.

The next full GSM8K uses the existing ATOM-equivalent native FP8 projection and
192-column FP8 FMHA, without the diagnostic rotations, residuals, or balancing.
Its purpose is model-accuracy measurement after isolated arithmetic validation.
It does not erase or replace the failed 0.20 BF16-difference test.

The earlier GSM8K run never reached FP8 attention because its prompts did not
satisfy the performance thresholds. For EVAL_ONLY, this recipe now:
- sets MAX_BATCHED_TOKENS=512 and ATOM_FP8_ACCURACY_STRESS=1;
- uses FP8 cached attention for positive query/context lengths, preserving causal
  new-token BF16 routing and the main FP8 opt-in flag;
- uses eager execution for unambiguous invocation evidence;
- retains full 1319-question GSM8K, five-shot, temperature zero, real speculative
  verification, unchanged questions and scoring;
- creates a phase marker only immediately before evaluation, then requires actual
  native FP8 projection and E4M3 Q/K/V FMHA evidence from all eight worker PIDs.
  Warmup, BF16 projection, wrong dtypes, empty calls, or missing workers cannot pass.

Normal performance routing remains at query >=512 and context >=8192.
Ten CPU routing/evidence tests passed. The GPU prerequisite runs Bash syntax,
those CPU tests, native weight layouts, and large/small-context numerical plus
changed-input graph checks before the full-model evaluation can start.
A GSM8K result with execution proof is model-accuracy evidence for this stress
configuration; it is not a throughput measurement or a strict BF16 numerical pass.

## GPU prerequisite passed; GSM8K dispatched

Run [40687966988](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/40687966988)
at 14250d6c6c17cd54fb7cd480c74c5c4eb5da2e78 passed Bash syntax, all ten CPU
checks, four native weight layouts, and both large (512 Q / 8192 KV) and small
(64 Q / 256 KV) projection/FMHA chains with changed-input graph replay.
FP8_ACCURACY_PATH_PREFLIGHT_PASS and FP8_NATIVE_SMALL_CONTEXT_PASS were emitted.

Full GSM8K run [40688077118](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/40688077118)
is pinned to the same source. Its exact-source prerequisite passed and the
MI355X agentic eval job started. Score and eight-worker evaluation-phase FP8
execution proof are pending. The live checker polls this run every 60 seconds
while its Python tool session remains available.

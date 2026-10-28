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
GPU validation of the new FP8 projection integration is pending. No performance
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

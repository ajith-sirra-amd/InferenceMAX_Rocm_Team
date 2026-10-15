# Scaled FP8 context ASM validation

[??](fp8ctx-optimization_zh.md)

Keep the pinned image and installed AITER kernels. Run 38718539330 proved that
`get_ps_metadata_v1(..., need_lse=True)` is supported. This removes unwritten
LSE entries. All eight startup probes passed; the first live context failed
numerical validation (output RMS 8.8?21.6%, max LSE error 0.43?0.62).
Do not promote that run despite its green workflow conclusion.

Use installed AITER `per_tensor_quant_hip` for separate dynamic Q/K/V scales
before head replication. Pass the dequant scales to ASM. The unequal-length
packing kernel and immutable per-chunk host plan cache remain; padding 12 heads
to 16 and separate reduction remain. Dynamic quantization adds launches and
allocations; the earlier direct-cast timing is not evidence for this version.

One-time live validation compares BF16, scaled ASM with ordinary preparation,
and scaled ASM with cached preparation. Ordinary/optimized results must match
exactly. Keep finite output/LSE, RMS <= 0.1 and LSE max <= 0.5 gates against BF16.
Also compare against BF16 attention on dequantized FP8 operands: kernel RMS must
be <= 0.05 and LSE max <= 0.03. These are diagnostic smoke gates, not model-level
accuracy proof. Validation failure terminates the worker without retry/fallback.

The MI355X launcher now preserves the container exit code through host cleanup.
Recipe replay failures explicitly exit nonzero. The smoke remains C96 TP8/DCP8,
one warmup request per lane and 600-second replay with profiling disabled.
Review all eight workers and sustained benchmark completion before a full run.
Raw cache/projection and DCP changes remain on hold. Baseline FMHA accounts for
9.1293% of summed GPU kernel duration, which does not predict throughput lift.

Already-quantized FP8 operands are replicated as uint8 bytes to avoid unsupported
Triton FP8 masked-load sentinel casts. Startup compares optimized packing and
metadata against ordinary ASM preparation with identical synthetic operands.

Live diagnostics sample up to eight queries per sequence against all keys using
FP32 matmul/softmax, for both original and dequantized FP8 inputs. The numerical
gates remain unchanged. Validation-only dequant reconstruction is excluded from
timed ASM preparation. Previous scaled timings included that reconstruction
and must not be treated as steady-state overhead measurements.

# Optimized FP8 context ASM smoke

[??](fp8ctx-optimization_zh.md)

Retain the pinned Docker image and its installed AITER attention kernels.
Current upstream AITER offers fused MLA preparation, but that helper requires
identical Q/KV token counts. Cached context has different lengths. This overlay
uses an independent Triton preparation kernel for unequal lengths: head replication
and direct FP8 conversion write into reusable buffers in one launch. No AITER upgrade.

Context-chunk CPU plans are retained on the batch chunk and reused across model
layers. Immutable pinned CPU metadata prevents staging-buffer reuse races. Each
backend uploads the plan to its own device buffers on the consuming stream.
The first encounter still requires CPU readbacks; five metadata uploads remain.
Head padding from 12 to 16 and separate ASM reduction remain.

Smoke configuration: C96 TP8/DCP8, one warmup request per lane, 600 seconds of
benchmark replay, profiling disabled. One-time live comparisons run per backend:
BF16, current ASM, optimized ASM. Optimized/current output and LSE must match
exactly. Live finite values, relative RMS <= 0.1 and max LSE error <= 0.5 are
provisional smoke thresholds; they do not establish model-level correctness.
The startup probe now uses 12 heads and rejects nonfinite output/LSE.

The job refuses success if no validation log exists or any context ASM failure
was recorded. One-shot timings are diagnostic and include initialization;
repeat warmed measurements at representative shapes before declaring a win.
Only consider a full matched profile after smoke logs and throughput are reviewed.

Baseline context FMHA averages 1.9075 ms/call and accounts for 9.1293% of summed
GPU duration in the supplied 60-second traces. This is opportunity evidence,
not an end-to-end speedup prediction. The 12-to-16 padding costs 33% extra head
work, partial-result reduction adds memory traffic, and metadata can stall CPU/GPU.

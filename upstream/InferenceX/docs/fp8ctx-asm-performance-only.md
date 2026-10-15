# ASM FP8 performance-only benchmark

[??](fp8ctx-asm-performance-only_zh.md)

User explicitly authorized measuring the current ASM path before repairing
accuracy. This is a performance-only experiment, not an accuracy-qualified run.
Use direct BF16-to-FP8 casts, fused head replication/conversion and reusable
buffers, cached immutable context CPU plans, installed ASM attention/reduction,
and need_lse=True. Keep 12-to-16 head padding. Do not change raw cache/projection
or DCP behavior. Keep the original pinned Docker image and installed AITER.

Log BF16/ASM numerical comparisons once per backend. Accuracy failures warn and
continue without retry; runtime exceptions remain fatal. Mark all resulting
throughput as accuracy-unqualified. The LSE-writing fix remains applied.

C96 TP8/DCP8, 10 warmup requests per lane, measured replay duration 3600 seconds,
profiling disabled. This differs from the supplied baseline's one warmup request
per lane and 1500-second measured run, so compare cautiously or run a matched
BF16 control. A later accuracy repair can change performance. Record actual
ASM dispatch on all eight workers, command warmup/duration, and replay completion.

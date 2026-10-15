# FlyDSL FP8 chunked-context smoke

[??](fp8ctx-flydsl_zh.md)

Private FMHA overlay copied from AITER PR6266 commit
836eaba05d5a4691f3019b7d271be3a5e4be5b80. License headers retained.
Imports are relocated to a private vLLM package; shared installed AITER kernels
and dispatch remain intact. Install FlyDSL runtime 0.3.4.1 in the job container.

The context hook uses noncausal varlen FP8 attention, independent Q/KV lengths,
12 real heads, QK192/V128, dynamic HIP Q/K/V quantization, and BF16 output plus
head-major FP32 LSE. The kernel may choose split-K automatically. One split
writes output/LSE directly; multiple splits use a workspace and combine launch.
No ASM head padding or persistent-scheduling CPU metadata is required.

Before model loading, run FP32 probes for unequal lengths, ragged batches and
long KV contexts. Real per-backend comparisons require finite output/LSE,
relative RMS <= 0.1 and max LSE error <= 0.5 versus BF16. Failures terminate the
worker and propagate through the launcher. Warm both paths before event timing;
full-call timing includes quantization. These gates do not prove model accuracy.

C96 TP8/DCP8 smoke: one warmup request per lane, 600 seconds measured replay,
profiler disabled. Full benchmark is authorized only after smoke validation:
10 warmup requests per lane, 3600 seconds, profiler disabled. Raw-cache/projection
and DCP experiments remain held. Performance versus BF16 is not yet established.

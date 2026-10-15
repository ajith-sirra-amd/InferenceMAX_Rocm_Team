# ASM FP8 performance-only benchmark

[中文](fp8ctx-asm-performance-only_zh.md)

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

## Runtime dispatch evidence

Before each first optimized noncausal ASM dispatch per backend, log actual
Q/K/V shapes and FP8 dtypes, device/PID, head padding, scale, ASM entry point,
reducer and need_lse=True. The startup probe uses a separate dispatch key so it
cannot suppress the real-context evidence. Wrong operand dtypes and a failed
startup ASM self-test are fatal; context runtime failures cannot fall back.

After replay, verify FP8 optimized noncausal dispatch, successful context calls,
and exact optimized/ordinary ASM agreement on every TP worker. Numerical BF16
accuracy failures remain recorded and explicitly unqualified. Preserve
asm_dispatch_evidence.log on cleanup (including failed runs),
asm_dispatch_verification.json after replay, server.log, benchmark command,
raw results and GPU metrics in Actions artifacts. No rocprof overhead is added.

The runtime-log verifier was exercised with all-eight-worker success, missing
worker, BF16 operand, runtime failure, optimized mismatch and nonoptimized
cases; its CLI success/failure outputs and exit codes were checked locally.
GPU dispatch still requires the actual runner logs; configuration is not proof.

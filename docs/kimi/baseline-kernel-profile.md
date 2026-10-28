# C96 BF16 context-attention profile

[中文](baseline-kernel-profile_zh.md)

Use `kimik3-fp4-mi355x-vllm-agentic-mtp` with TP8/DCP8, C96 and the pinned
`nightly-rocm100-fe4df3df0eb703862c686b256e9fe44fa94c8945` image. The C96 recipe
does not enable speculative decoding. KV storage remains FP8 and expert weights
remain FP4; cached-context attention uses the original BF16 dispatch with
`APPLY_FP8CTX=0`. A one-time `[BASELINE-PRECISION]` log records actual Q/K/V shapes,
dtypes and attention scale without replacing the attention call.

The recipe collects `rocprofv3 --kernel-trace --stats` for 60 seconds starting
3300 seconds after server launch. Dispatch with a 1500-second replay duration
and warmup=1. Verify measured replay overlaps the collection interval before
interpreting statistics. Shutdown signals captured descendants from leaves to
root, then waits up to 1200 seconds for profiler output before bounded cleanup.
This shutdown change requires runtime validation.

Download `agentic_*` raw artifacts and `server_logs_*`. Kernel CSVs must exist,
contain GPU dispatches, and cover the requested interval. Report full kernel
names, calls, total/mean/min/max duration, standard deviation, device/process
identity and launch geometry where available. Compute duration shares from
validated rows; summed GPU duration is not elapsed wall time. Reject reversed
or implausible timestamps instead of allowing them to poison percentages.

Precision from the input probe is observed tensor precision. Kernel symbols
may identify input, output or accumulation types; label inference explicitly.
Do not infer attention softmax precision from BF16 inputs alone. Compare
precision claims with the exact installed AITER source when necessary.

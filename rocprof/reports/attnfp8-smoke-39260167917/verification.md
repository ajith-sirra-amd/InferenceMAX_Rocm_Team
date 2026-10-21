# AttnFP8 smoke verification

Run: https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/39260167917
Executed commit: 97fe4c6101c29dfffd7856b000f6bdab610ccf20.

Checkpoint compatibility smoke completed; native FP8 context FMHA was not achieved.
All eight devices logged BF16 Q/K/V at the context attention call. The probe
is one-time and recorded Q=(1,12,192), K=(11,12,192), V=(11,12,128); it does not
sample every later replay shape. No legacy BF16-to-FP8 context cast was installed.
Checkpoint self_attn specifies FP8 projection inputs/weights and output_tensors=null.
Engine reports quantization=quark, dtype=bfloat16, KV cache=fp8.

Actual command: C48 TP8/DCP8, DSpark k4 with synthetic acceptance3.36, max_batch8192,
max_seqs96, warmup1/lane, measured600 seconds, profiler disabled. Reported result
window624.64231seconds includes completion/drain. No kernel trace was captured;
exact GPU symbol/performance is not established by this smoke.

Measured phase432 completed requests:430 valid,2 empty-content errors (0.46%).
Aggregated all-record accounting includes warmup errors:531 records,99 warmup
records dropped,91 error records dropped; categories overlap. Do not equate91
with measured-phase errors. Recovered hipBLASLt GEMM fallback warnings present.
Accuracy not tested; MTP uses synthetic acceptance. Aggregated metadata incorrectly
labels spec_decoding=none despite actual server command/engine proving DSpark enabled.

Smoke metrics, not a matched baseline comparison:10463.79174 total tok/s/GPU,
49.70765 output tok/s/GPU, TTFT7.71108s, TPOT0.03126s. These do not demonstrate
an FP8 FMHA gain. No full benchmark dispatched as part of this verification.

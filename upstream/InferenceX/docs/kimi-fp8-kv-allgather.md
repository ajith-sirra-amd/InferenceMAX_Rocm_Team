# FP8 context-KV all-gather smoke

[中文](kimi-fp8-kv-allgather_zh.md)

Use AMD checkpoint, C48 TP8/DCP8 MTP k4, source image1b2a66639, auto loader,
warmup1 per lane,600-second replay. No context FP8 ASM overlay. Earlier
AMD smoke39260167917 installed only a dtype probe; it never installed ASM.
Baseline39239112271 loaded moonshotai/Kimi-K3, not the AMD checkpoint.

Copy stored FP8 cache bytes directly into a reusable uint8 communication
buffer, all-gather those bytes, then decode with each source rank's cache scale
into the existing BF16 workspace. Retain BF16 reconstruction/FMHA and query
communication. No additional quantization; local dequantization moves after
communication. Ordinary fp8 cache only; packed fp8_ds_mla remains unchanged.

Once per backend, compare local decode exactly with existing cache kernel and
all-rank decode with independent BF16 reference all-gather. Runtime mismatch
is fatal. Log byte payload, encoding, output dtype and exact parity. One-time
reference collectives add warmup overhead; steady-state has one byte gather.
Cache scales are assumed immutable after model initialization, as in this
checkpoint; per-rank scales gathered once. Persistent byte scratch consumes
some memory; pool recovery to23M is a measured objective, not guaranteed.

auto loader restores earlier smoke settings. Do not force a KV memory budget
before measuring peak reservations and confirming memory safety. Keep original
source branch and existing workflows unchanged. No accuracy gate is inferred
from synthetic-MTP throughput; cache decode parity covers communication only.

## Startup boundary repair

Smoke [39561981905](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/39561981905)
failed before measured replay because the installer selected a cache branch in
the non-DCP method and nested the following DCP method inside an else block.
The installer now restricts replacement to the DCP context method and rejects
unexpected method/branch layouts. Behavioral checks exercise FP8, packed FP8,
BF16 and adjacent methods; the pinned runtime source also compiles.
GPU execution, exact cache parity, KV capacity and throughput still require the
replacement smoke. This repair does not enable FP8 ASM FMHA.

## FP8 decoder compile repair

Smoke39570966330 recovered the startup GPU KV pool to **23,098,919 tokens**,
then failed during kernel warmup. Triton rejected the integer masked-load fill
value in the FP8 decoder. Omit the unused fill value; masked elements are never
stored. Use a floating-point fill for FP32 scales.

Before server launch, run the real decoder on every visible GPU using all256
FP8 byte encodings, two different source-rank scales and a partial final block.
Compare against PyTorch decoding exactly, including NaN handling. This check
fails before loading model weights if decoding cannot compile or differs.
The existing live-cache parity gates remain required. No measured throughput
or model accuracy result was produced by the failed run; the advertised KV pool
was observed before successful completion of kernel warmup.

## Native dispatch diagnostic smoke

Disable the experimental raw-byte transport installer and retain the pinned
runtime's native communication and reconstruction behavior. Install one-time
observation probes for the native DCP context eligibility decision and actual
KV collective dtype, shape and byte counts. Report reconstruction weight dtype,
layout and scale presence so the FP8-path exclusion can be verified directly.
AMD checkpoint, auto loader, no CLI quantization override, C48 TP8/DCP8,
warmup1 per lane and600second replay. Require both diagnostic markers and the
existing context-input marker before accepting the smoke. No full benchmark
until the diagnostic and any subsequent communication adaptation pass.

## Native diagnostic result39576708017

Workflow succeeded. All8 GPUs logged native eligibility=False because the
reconstruction weights are FP8 (3072,512), with weight_scale present; the
actual KV collective used BF16. KV pool23098919 tokens. Total throughput
10985.22783 tok/s/GPU; output56.97678 tok/s/GPU. Logging-only change: do not
attribute the difference from earlier AMD smoke to an optimization.
Measured phase456 valid requests and1 empty-content InvalidInferenceResultError
(0.22%). Aggregate error accounting includes warmup. No ServerDisconnectedError
in server/benchmark logs. Results do not establish model accuracy.
Evidence: rocprof/reports/kimi-model-communication-comparison/39576708017-native-diagnostic/.

## Native byte-gather adaptation smoke

Use the pinned runtime's gather_compressed_context through manager.kv_gather,
then decode gathered FP8 bytes with each source rank's scale into the original
BF16 workspace. Keep AMD's quantized kv_b_proj and BF16 context FMHA. No CLI
requantization override and no additional cache quantization. Separate persistent
byte storage prevents overlap with BF16 decode output; its capacity impact must
be measured. Preserve the native dispatch/collective probes. Before server launch,
exercise GPU decode across byte encodings, rank scales and a masked tail. Once per
backend, compare decoded all-rank cache against an independent original BF16 cache
gather exactly; fail on mismatch. Reference traffic is warmup overhead. Cache scales
are collected once and assumed immutable after initialization. Support ordinary
fp8 cache; retain original packed-cache and BF16 branches. C48 warmup1/600second
smoke only; no full run until actual byte transport and parity pass on all8 GPUs.

## Multi-GPU decode preflight device fix

Smoke39588733292 passed exact FP8 decoding on GPU0, then failed on GPU1
before server launch because the preflight allocated tensors on GPU1 without
switching the active Triton device. Explicitly select each GPU before launching
its decoder. All8-device GPU checks and live cache parity remain required; this
failed run produced no FP8 collective or benchmark evidence.

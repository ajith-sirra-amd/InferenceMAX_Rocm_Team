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

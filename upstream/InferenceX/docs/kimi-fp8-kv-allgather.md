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

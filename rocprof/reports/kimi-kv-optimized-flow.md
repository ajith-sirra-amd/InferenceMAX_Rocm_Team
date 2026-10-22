# Kimi-K3: weight quantization, KV capacity and fused attention

**English** | [中文](./kimi-kv-optimized-flow_zh.md)

## 1. Models and problem

| Checkpoint | What is quantized? | Reconstruction weight kv_b_proj |
|---|---|---|
| **Kimi-K3 by Moonshot** — moonshotai/Kimi-K3 | Expert weights: MXFP4. Attention projections and shared experts: BF16 before online overrides | BF16 |
| **Kimi-K3 by AMD, Quark quantized** — amd/Kimi-K3-Quark-MXFP4-AttnFP8 | Experts: MXFP4; attention linear weights/input arithmetic: FP8; shared experts: BF16 | FP8 with per-channel scales |

- Both use the same Kimi-K3 architecture.
- Attention weight quantization and KV-cache quantization are separate choices.
- Our benchmarks explicitly use **FP8 KV cache** for both models.
- FP8 projection weights do not guarantee FP8 Q/K/V outputs; current context FMHA receives BF16 tensors.

| Measured full run | Total tokens/s/GPU | KV-pool tokens |
|---|---:|---:|
| [Moonshot Kimi-K3 reference39239112271](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/39239112271) | 15,476.47 | 20,951,355 |
| [AMD Quark Kimi-K3 with our FP8-byte communication adaptation39601846940](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/39601846940) | 15,258.78 | 23,098,919 |
| Difference | **−1.41% throughput** | **+2,147,564 tokens / +10.25% capacity** |

- The Moonshot reference quantized only in_proj_qkvgfab online; most other attention projections remained BF16.
- “Communication adaptation” means our runtime change that gathers the existing FP8 cache bytes before decoding them. It is not part of the AMD checkpoint.
- **Problem:** the AMD setup has a larger KV pool but slightly lower measured throughput.
- **Goal:** keep compact weights/cache and recover fused context processing.
- The comparison shows the net setup difference; it does not isolate the cause of either the capacity gain or throughput gap.

## 2. Why quantizing weights matters

| Weight group | Effect of FP8 quantization | Tradeoff |
|---|---|---|
| Large attention projections | Roughly half BF16 weight bytes, plus scales; potentially faster GEMMs | Dynamic activation quantization/preparation and accuracy validation |
| kv_b_proj | Smaller reconstruction weight storage and potential weight traffic reduction | Current native fused reconstruction cannot consume its FP8 weights/scales |
| Expert weights | MXFP4 already present in both checkpoints | Export format/runtime allocations can differ |

- Less resident weight memory can leave more GPU memory for KV cache.
- The measured **+2.15M KV tokens** belongs to the complete AMD setup; it is not attributable to kv_b_proj alone.
- kv_b_proj at TP8:24 matrices/GPU ×[3072,512].
- FP8 instead of BF16 saves approximately **36MiB/GPU across these24 weights**, before scales/layout overhead.
- Other attention weights provide additional savings; runtime workspaces, graph allocations and memory profiling also determine the final pool.
- Restoring BF16 kv_b sacrifices that small weight saving to regain the currently supported fusion.

## 3. Current processing and drawback

| Step | Moonshot reference: native fused path | AMD Quark model: our adapted path |
|---|---|---|
| Compressed cache | FP8 | FP8 |
| Communication | Native FP8-byte all-gather | Adapted FP8-byte all-gather |
| Cache decode/reorder | Inside fused reconstruction | Separate processing into BF16 latent rows |
| Reconstruction weights | BF16 kv_b | FP8 kv_b plus scales |
| Reconstruction/packing | One native fused operation | Separate FP8-weight GEMM and preparation/packing |
| Context FMHA | BF16 | BF16 |
| Main tradeoff | More attention weight storage | More intermediate processing/dispatch |

- Separate processing can add HBM reads/writes and launches.
- This is a candidate performance bottleneck, not a proven explanation of the entire1.41% gap.
- Original AMD fallback, before our adaptation: **FP8 cache → decode to BF16 → BF16 all-gather → reconstruction**.

## 4. FP8 all-gather versus BF16 kv_b: the exact relationship

| Question | Answer |
|---|---|
| Does FP8 all-gather itself require BF16 kv_b? | **No.** Communication transfers cache bytes and never reads kv_b weights |
| Does the pinned native fused context path require BF16 kv_b? | **Yes:** contiguous BF16 weights, no weight scales, plus other eligibility checks |
| Direction in our experiment | **Keep kv_b BF16 → enable native fused-path eligibility → that path uses FP8-byte all-gather and fused reconstruction** |
| Does FP8 gather convert kv_b weights? | **No** |
| Can FP8 gather work with FP8 kv_b? | **Yes:** our AMD adaptation demonstrated it, with separate reconstruction processing |

## 5. Practical solution now: Moonshot with selective FP8 weights

| Component | Proposed setting | Purpose |
|---|---|---|
| Model | Moonshot Kimi-K3 | Start from BF16 attention weights |
| Expert weights | Retain MXFP4 checkpoint weights | Preserve compression |
| Other attention projections | Online fp8_per_channel | Approach AMD attention weight savings |
| kv_b_proj | **Retain BF16; omit from targets** | Use existing native fusion |
| Convolutions/shared experts/residual projections | Retain checkpoint precision | Preserve intended exclusions |
| KV cache/communication | FP8 cache; native FP8-byte collective | Compact storage and payload |
| Reconstruction | Fused reorder + dequant + BF16-weight projection + packing | Avoid separate intermediate processing |
| FMHA/output | **BF16** | Retain existing attention kernel |

```text
FP8 compressed KV cache
 → FP8-byte all-gather
 → ONE kernel: reorder + dequant + BF16 kv_b projection + K/V packing
 → BF16 full-head K/V + BF16 Q
 → BF16 FMHA
 → BF16 output
```

- Targets under self_attn: in_proj_qkvgfab, f_b_proj, g_a_proj, g_b_proj, o_proj, fused_qkv_a_proj, q_b_proj, q_proj, kv_a_proj_with_mqa, g_proj.
- First smoke selected303 attention linears; selection does not prove completed quantization.
- No AMD weight-conversion patch is installed.
- Scheme resembles AMD attention quantization except BF16 kv_b; quantized values can differ.
- **Approximately23M KV tokens and a throughput lift are targets, not established outcomes.**

## 6. Preferred kernel improvement: keep kv_b FP8 and fuse reconstruction

| Kernel choice | Benefit | Requirement |
|---|---|---|
| Existing BF16-weight fused reconstruction | Eliminates separate reorder/decode/projection/packing | BF16 kv_b weight storage |
| **Proposed fused reconstruction supporting FP8 weights/scales** | Retains AMD's compact kv_b and removes separate processing | Implement and validate FP8-weight/scaled-input support |
| Further FP8 FMHA | Potential attention compute/bandwidth gain | Scaled FP8 Q/K/V preparation, supported kernels, accuracy and complete-call performance tests |

```text
Preferred reconstruction improvement:
FP8 cache → FP8-byte gather
 → fused reorder + scaled FP8-weight reconstruction + K/V packing
 → BF16 K/V → BF16 FMHA → BF16 output
```

- **Needed to combine both benefits:** connect fused reconstruction supporting FP8 kv_b weights to the DCP path.
- Current AITER already contains per-row FP8-scale support; pinned vLLM passes no weight scale and limits eligibility to BF16.
- Prototype: `fp8_fused_reconstruction_probe.py` compares existing AITER FP8-weight and BF16-weight fusion against explicit numerical references. It is not wired into serving or GPU-validated yet.
- Serving integration must handle loaded preshuffled weights, cache scales and custom-op registration; the prototype uses unshuffled weights.
- **FP8 FMHA is optional:** it is not necessary to use FP8 all-gather or eliminate separate reconstruction preparation.
- A fused kernel may internally decode values; “FP8-weight support” does not imply every arithmetic step is FP8.
- Fully FP8 FMHA candidate: fused scaled FP8 K/V output + scaled FP8 Q → FP8 FMHA → BF16 output.
- Direct FP8 output still quantizes values; fusion can avoid a BF16 intermediate.
- FP8 FMHA may lose after conversion, padding and launch costs. Higher-precision accumulation cannot undo quantization loss.

## 7. Reconstruction dimensions and validation

| Tensor / operation | Local dimensions at TP8 |
|---|---|
| Cached latent C / RoPE key | [T,512] / [T,64] |
| Cache row | [T,576] |
| kv_b weight | [3072,512] |
| Reconstruction | C × W_kv_b^T → [T,3072] |
| Packed full-head K / V | [T,12,192] / [T,12,128] |

- Reconstruction is an up-projection, not head rearrangement; full-head FMHA cannot directly consume the512-dimensional latent.
- T: chunk token count; cache rows are distributed over DCP ranks.
- Smoke: C48 TP8/DCP8, warmup1/lane,600seconds.
- Require all8 GPUs: loaded FP8 projection weights,24 BF16 kv_b matrices, native eligibility, uint8 collective and actual fused execution.
- Measure KV pool, total/output throughput and request errors; evaluate accuracy separately with real verification.

| Smoke | Status / finding |
|---|---|
| [39717243502](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/39717243502) | Fastsafetensors: all8 numerical preflights passed; loading failed on GPU4,0MB free,nogds701 |
| [39718556543](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/39718556543) | Queue1 retry: loading failed on GPU0,198MB free,same error |
| [39719727532](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/39719727532) | Default-loader diagnostic; same selective quantization; results pending at this revision |

- No capacity/throughput result from failed loading runs; these failures do not establish faulty GPU hardware.

## Sources

- [AMD Quark Kimi-K3 model card](https://huggingface.co/amd/Kimi-K3-Quark-MXFP4-AttnFP8)
- [Moonshot Kimi-K3 checkpoint](https://huggingface.co/moonshotai/Kimi-K3)
- [Pinned native MLA backend](https://github.com/vllm-project/vllm/blob/1b2a666391f2a24228a1436b64d547c34cc6b6ce/vllm/v1/attention/backends/mla/rocm_aiter_mla.py)
- [Pinned gather/reconstruction](https://github.com/vllm-project/vllm/blob/1b2a666391f2a24228a1436b64d547c34cc6b6ce/vllm/v1/attention/ops/rocm_aiter_mla_prefill.py)

## Implementation and validation status ? 2026-10-22

- **Implemented in the perf branch; GPU validation pending.**
- AMD model: existing FP8 kv_b weights + channel scales; no BF16 weight conversion.
- Flow: FP8 cache ? uint8 all-gather ? fused row reorder / source-rank scales / FP8-weight projection / K/V packing ? BF16 K/V ? BF16 FMHA.
- Pinned AITER flat kernel is vendored in this repository; installed image AITER remains unchanged. Supports AITER preshuffled weights with register unshuffle.
- Validation: all8 GPU numerical preflight, unequal rank scales, reordered rows and masked tails; real decoded-checkpoint FP32 reference; long cached-prefix HTTP requests; full GSM8K with real DSpark verification.
- Replay warmup per lane: **0**; fast override: **0**. Serving startup profiling and graph capture still run.
- Fail the experiment if FP8 fused execution is missing, outputs violate numerical limits, or GSM8K regresses. A full throughput run needs separate successful validation.

| Moonshot diagnostic39719727532 | Observed |
| --- | --- |
| Loader | Default loader reached serving |
| KV pool | 22,534,301 tokens |
| Profiling throughput | 11,641.85674 total tokens/s/GPU; 60.77441 output tokens/s/GPU; 600s smoke only |
| Warmup errors | 85/99 records; empty streamed responses |
| Profiling errors | 4/478 records =0.84%; empty streamed responses |
| Native fusion / transport | BF16 kv_b fused reconstruction and FP8-byte gather verified on all8 |
| Verdict | Not a clean smoke pass; no full-run promotion |
| Other warning | 280 hipBLASLt internal-error fallback warnings; causal link to empty replies unproven |

- Empty response failures returned HTTP200; no server traceback observed. Investigate generation/streaming behavior with the accuracy gate.

## Moonshot-only follow-up ? 2026-10-22

- User-selected model: **moonshotai/Kimi-K3 only** for subsequent experiments.
- Keep baseline target `*in_proj_qkvgfab: fp8_per_channel`.
- Add **only** `*.self_attn.kv_b_proj: fp8_per_channel`; other attention weights remain BF16 and existing checkpoint expert quantization is retained.
- Quantize kv_b weights once during loading. Cached latent is already FP8; fused context reconstruction introduces no additional activation quantization.
- Loader probe requires24 FP8 kv_b linears per GPU and checks supported online PTPC weight layout/channel scales.
- FP8 cache ? byte all-gather ? fused FP8-weight reconstruction ? BF16 K/V ? BF16 FMHA.
- Accuracy-first validation; replay warmup per lane0. Pool and throughput improvement remain unproven.
- Already dispatched AMD run39726650098 is allowed to finish; it is not the model selected for subsequent experiments.

## Incremental o_proj accuracy experiment

- Add only `*.self_attn.o_proj: fp8_per_channel` to the two-target Moonshot experiment.
- Target set: `*in_proj_qkvgfab`, `*.self_attn.kv_b_proj`, `*.self_attn.o_proj`.
- Require postload FP8 o_proj weights/channel scales on all8 GPUs. BF16 FMHA retained.
- Run numerical/cached-context checks and GSM8K before throughput. Compare with run39729399329; accuracy and memory gains are unproven.

## Moonshot loader probe correction

- Run39729399329 failed after weight loading because the probe rejected native FP32 KDA conv1d weights. No serving or GSM8K result.
- Allow unquantized BF16/FP32 conv1d; reject FP8 convolution and attached quantization scales.
- Fused kernel and quantization targets unchanged. Run39729912013 also carries the old probe; corrected successor uses three targets including o_proj.

## AMD Quark reference for broader Moonshot quantization

- AMD checkpoint config uses `*self_attn*` FP8 per-channel weights and dynamic FP8 inputs, with explicit convolution exclusions; GSM8K strict96.59% passed with fused reconstruction.
- Add Moonshot `fused_qkv_a_proj` (346.5MiB/GPU raw) and `g_proj` (252MiB/GPU raw), alongside in_proj_qkvgfab, kv_b_proj and o_proj.
- Estimated additional savings beyond o_proj experiment:598.5MiB/GPU before scales/runtime overhead.
- Reference reduces uncertainty; Moonshot online quantization is independently validated. This does not exactly reproduce the AMD checkpoint.
- Keep q_b_proj and f_b_proj BF16 for this increment; exclude convolutions, router, lm_head, residual projections and shared experts.
- Current run39731052931 continues unchanged; queue broader accuracy experiment behind it.

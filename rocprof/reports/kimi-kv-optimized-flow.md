# Kimi-K3: problem and proposed KV flow

**English** | [中文](./kimi-kv-optimized-flow_zh.md)

## Problem

| Full benchmark | Total tokens/s/GPU | KV-pool tokens |
|---|---:|---:|
| [Moonshot reference39239112271](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/39239112271) | 15,476.47 | 20,951,355 |
| [AMD + separate FP8-gather adaptation39601846940](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/39601846940) | 15,258.78 | 23,098,919 |

- AMD gained **10.25% KV capacity**, but total throughput was **1.41% lower**.
- Goal: approach **23M KV-pool tokens** while recovering native fused context processing.
- Hypothesis: separate cache decode/reorder and reconstruction preparation add overhead.
- The benchmark comparison does not prove which operation caused the throughput gap.

## Current drawback

| Step | AMD + our separate gather adaptation | Drawback |
|---|---|---|
| Cache | FP8 compressed KV | Compact; retain |
| All-gather | Existing FP8 bytes through uint8 collective | Half BF16 payload; retain |
| Cache decode/reorder | Separate processing into BF16 latent rows | Intermediate HBM traffic and launches |
| Reconstruction | Separate FP8-weight kv_b GEMM plus required preparation | Misses native fused processing |
| K/V packing and FMHA | BF16 K/V, BF16 FMHA | FP8 weights/cache do not make FMHA FP8 |

- **FP8 all-gather does not require BF16 kv_b weights:** communication never reads those weights.
- **The pinned fused reconstruction kernel requires BF16 kv_b weights**, contiguous layout and no separate weight scales.
- AMD's FP8 kv_b weights fail that eligibility check.

## Which requirement comes first?

| Question | Answer |
|---|---|
| Does FP8 all-gather require BF16 kv_b_proj mathematically? | **No.** The collective transfers bytes and does not read projection weights |
| Does our pinned native fused context path require BF16 kv_b_proj? | **Yes**, along with the other dtype/layout/metadata checks |
| What is the direction in our experiment? | **Keep kv_b_proj BF16 → satisfy native fused-path eligibility → that path invokes FP8-byte all-gather and fused reconstruction** |
| Does selecting FP8 all-gather convert kv_b_proj to BF16? | **No** |
| Can FP8 all-gather work with FP8 kv_b_proj? | **Yes**; our AMD adaptation proved it, but reconstruction then used separate processing |

- **BF16 kv_b is necessary for this native fused implementation, but not sufficient by itself.**
- Actual uint8 collective and fused-execution logs must prove the path engaged.

## Proposed solution

| Component | Moonshot experiment | Purpose |
|---|---|---|
| Model | moonshotai/Kimi-K3 | BF16 attention starting point |
| Expert weights | Retain checkpoint MXFP4 | Keep existing compression |
| Other attention projections | Online fp8_per_channel | Save weight memory; FP8 GEMMs |
| kv_b_proj | **Retain BF16; omit from targets** | Enable native fused reconstruction |
| Convolutions/shared experts/residual projections | Retain checkpoint precision | Avoid unintended quantization |
| KV cache and collective | FP8 cache; native FP8-byte gather | Compact storage and communication |
| Reconstruction/packing | **One native fused kernel** | Avoid separate intermediate processing |
| FMHA/output | BF16 | Retain current attention kernel |

```text
FP8 compressed KV cache
 → FP8-byte all-gather
 → ONE kernel: reorder + dequant + BF16 kv_b projection + K/V packing
 → BF16 K/V + BF16 Q
 → BF16 FMHA
 → BF16 output
```

- Targets: in_proj_qkvgfab, f_b_proj, g_a_proj, g_b_proj, o_proj, fused_qkv_a_proj, q_b_proj, q_proj, kv_a_proj_with_mqa, g_proj under self_attn.
- First smoke selected **303 attention linears**; this was selection evidence, not completed quantization.
- No AMD weight-conversion patch is installed.
- Matches AMD attention precision except BF16 kv_b; online quantized values can differ.

## Benefits and tradeoffs

| Expected benefit | Remaining cost / uncertainty |
|---|---|
| Smaller attention weights than reference | Dynamic activation quantization costs work |
| Native fused reconstruction | Projection and cache dequantization remain necessary |
| Less intermediate HBM traffic/dispatch | Throughput lift must be measured |
| Potential KV pool near23M | BF16 kv_b and runtime buffers consume memory; capacity is not guaranteed |
| BF16 context FMHA retained | New projection quantization needs accuracy validation |

## Why reconstruction stays

| Tensor / operation | Local dimensions at TP8 |
|---|---|
| Cached latent C | [T,512] |
| Cached RoPE key | [T,64] |
| Cache row | [T,576] |
| kv_b weight | [3072,512] |
| Reconstruction | C × W_kv_b^T → [T,3072] |
| Packed K | [T,12,192] |
| Packed V | [T,12,128] |

- T: chunk token count; cache rows are distributed across DCP ranks.
- Reconstruction is an **up-projection**, not head rearrangement.
- Full-head FMHA cannot directly consume the compressed latent.
- Dequantization does not recover quantization loss.

## Future options

| Option | Flow change | Requirement |
|---|---|---|
| Fused FP8-weight reconstruction | FP8 cache/communication → fused FP8-weight reconstruction → BF16 K/V → BF16 FMHA | Kernel support for FP8 weights/scales |
| FP8 FMHA | Fused reconstruction outputs scaled FP8 K/V; prepare scaled FP8 Q → FP8 FMHA → BF16 output | Accuracy and complete-call performance validation |

- Direct FP8 outputs still perform quantization; fusion can remove a BF16 intermediate.
- Higher-precision accumulation cannot undo quantization loss.
- FP8 FMHA may lose after preparation, padding and launch costs.

## Validation and status

- Smoke: **C48 TP8/DCP8, warmup1/lane,600seconds**.
- Require all8 GPUs: actual FP8 projection weights,24 BF16 kv_b matrices, native eligibility, uint8 collective and fused execution.
- Measure KV pool, total/output throughput and request errors.
- Run real-verification accuracy separately; synthetic throughput does not establish accuracy.
- After fastsafetensors passes: identical smoke without the loader flag.

| Run | Finding |
|---|---|
| [39717243502](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/39717243502) | All8 numerical preflights passed. Loading failed on GPU4: HSA resource exhaustion, reported free memory0 MB, nogds error701. No pool/throughput result |
| [39718556543](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/39718556543) | Retry: fastsafetensors queue bounded to1; outcome pending at this revision |

- Loading failure does not establish faulty GPU hardware.
- Queue limit is a diagnostic change, not a proven fix.

## Sources

- [AMD model card](https://huggingface.co/amd/Kimi-K3-Quark-MXFP4-AttnFP8)
- [Pinned native MLA backend](https://github.com/vllm-project/vllm/blob/1b2a666391f2a24228a1436b64d547c34cc6b6ce/vllm/v1/attention/backends/mla/rocm_aiter_mla.py)
- [Pinned gather/reconstruction](https://github.com/vllm-project/vllm/blob/1b2a666391f2a24228a1436b64d547c34cc6b6ce/vllm/v1/attention/ops/rocm_aiter_mla_prefill.py)

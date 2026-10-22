# Kimi-K3: optimized KV communication and attention flow

**English** | [中文](./kimi-kv-optimized-flow_zh.md)

## Key facts

- FP8 all-gather transfers cache bytes; it does not use `kv_b_proj`.
- BF16 `kv_b_proj` is required by the **pinned fused reconstruction implementation**, not by the collective.
- The FP8 cache stores a compressed latent; FMHA consumes reconstructed full-head K/V.
- Reconstruction is a matrix projection, not just head rearrangement.
- FP8 attention-projection weights do not imply FP8 Q/K/V outputs.
- Faster individual kernels do not establish faster end-to-end inference.

## Tensors and dimensions

| Tensor / operation | Dimensions | Role |
|---|---|---|
| Hidden states X | [T, 7168] | Attention projection input |
| Compressed KV latent C | [T, 512] | Latent component of MLA cache |
| RoPE key component | [T, 64] | Cached positional key component |
| Compressed cache row | [T, 576] | 512 latent + 64 RoPE elements |
| Local kv_b weight, TP8 | [3072, 512] | 12 heads × (128 key + 128 value) |
| Reconstruction | C × W_kv_b^T → [T, 3072] | Produces local non-RoPE K and V |
| Packed local K | [T, 12, 192] | 128 reconstructed + 64 RoPE |
| Packed local V | [T, 12, 128] | Reconstructed values |

- T: number of tokens in the relevant chunk.
- Cache storage is distributed over DCP ranks; dimensions above describe token rows and each TP rank's reconstructed head set.
- Dequantization approximates represented values; it does not recover information lost during quantization.

## Precision and mechanism

| Stage | Practical flow in pinned runtime | Candidate optimized flow | Benefit / condition |
|---|---|---|---|
| Attention projections | FP8 weights/input arithmetic; BF16 outputs as required | Same, with producer/consumer quantization fusion where supported | Reduce GEMM cost; retain accurate normalization/RoPE |
| Compressed KV cache | FP8 | FP8 | Smaller cache and HBM traffic |
| KV all-gather | Existing FP8 bytes through uint8 collective | Same | Half BF16 payload; no new cache quantization |
| Reconstruction weights | BF16 | FP8 weights plus channel scales | Smaller weight traffic; requires supported fused kernel |
| Reorder + cache decode + projection + K/V packing | One native fused kernel with BF16 weights | One fused kernel supporting FP8 weights/scales | Avoid materialized intermediate cache and separate launches |
| Reconstruction arithmetic | Native kernel implementation | Appropriate higher-precision accumulation | Verify implementation and numerical error |
| Reconstructed K/V | BF16 | BF16 first; scaled FP8 only for validated FP8 FMHA | Avoid unnecessary precision transitions |
| FMHA | BF16 | Fastest accurate BF16 or scaled FP8 implementation | Measure complete call, including preparation |
| Attention output | BF16 | BF16 | Feed subsequent model operations |

## Three flows

| Flow | Pipeline | Status |
|---|---|---|
| Practical native fusion | FP8 cache → FP8-byte gather → fused reorder/dequant/BF16-weight reconstruction/packing → BF16 K/V → BF16 FMHA | Native path exists in pinned runtime; experiment must prove eligibility and execution |
| FP8-weight reconstruction | FP8 cache → FP8-byte gather → fused reconstruction with FP8 weights/scales → BF16 K/V → BF16 FMHA | Proposed kernel improvement; pinned native fusion lacks this weight support |
| FP8 FMHA | FP8 cache → FP8-byte gather → fused reconstruction with scaled FP8 K/V output + scaled FP8 Q preparation → FP8 FMHA → BF16 output | Candidate; requires accuracy and end-to-end performance validation |

- Reconstruction cannot be removed when full-head FMHA consumes the compressed MLA cache.
- Direct FP8 reconstruction output still quantizes reconstructed values; it can fuse that work and avoid a BF16 intermediate.
- FP32 accumulation cannot undo FP8 quantization loss.
- FP8 FMHA is not guaranteed to beat BF16 after conversion, padding, packing and launch overhead.

## Moonshot experiment

- Source: `moonshotai/Kimi-K3`; retain its compressed-tensors MXFP4 expert weights.
- Online `fp8_per_channel` attention projections; retain BF16 `kv_b_proj`.
- Explicit targets: `in_proj_qkvgfab`, `f_b_proj`, `g_a_proj`, `g_b_proj`, `o_proj`, `fused_qkv_a_proj`, `q_b_proj`, `q_proj`, `kv_a_proj_with_mqa`, `g_proj`.
- Leave convolutions, shared experts and residual projections unchanged.
- Matches AMD attention precision scheme except BF16 reconstruction; exported quantized values need not match.
- Goal: approach 23M KV-pool tokens while recovering native fused processing.
- Acceptance: all8 GPU quantization inventory, native eligibility, uint8 collective, fused execution, measured throughput/pool/errors.
- Smoke: C48, TP8/DCP8, warmup1/lane, 600 seconds.
- Accuracy: separate real-verification evaluation; synthetic throughput is not an accuracy test.

## Latest smoke finding

| Item | Evidence |
|---|---|
| Run | [39717243502](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/39717243502) |
| Loader | Explicit `fastsafetensors` |
| Numerical preflight | Passed on GPUs0–7 |
| Failure | GPU4: `HSA_STATUS_ERROR_OUT_OF_RESOURCES`; reported free memory0 MB; `nogds GPU operation failed: 701` |
| Phase | Weight loading / worker initialization; server never became ready |
| Quantization inventory | No completion marker |
| Throughput / KV pool | Not measured |
| Interpretation | Resource exhaustion during loader execution; hardware fault and exact loader cause not established |
| Next diagnosis | Inspect loading/staging memory and profiler interaction; isolate loader on same GPUs before blaming hardware |

## References

- [AMD model card](https://huggingface.co/amd/Kimi-K3-Quark-MXFP4-AttnFP8)
- [Pinned native MLA backend](https://github.com/vllm-project/vllm/blob/1b2a666391f2a24228a1436b64d547c34cc6b6ce/vllm/v1/attention/backends/mla/rocm_aiter_mla.py)
- [Pinned gather/reconstruction operations](https://github.com/vllm-project/vllm/blob/1b2a666391f2a24228a1436b64d547c34cc6b6ce/vllm/v1/attention/ops/rocm_aiter_mla_prefill.py)

# FP8 data flow

[Handbook contents](../HANDBOOK.md) · [Worked examples](11-worked-examples.md)

<a id="s-50"></a>
## Slide 50: Precision belongs to each object and each operation

“An FP8 path” does not mean every internal value uses eight bits.

- **MXFP4 weights:** 4-bit values with block scaling
- **FP8 operands:** Compact Q/K/V and projection operands
- **BF16 activations:** 16-bit storage for many outputs
- **FP32 internals:** Scales, reductions and accumulators



BF16 uses one sign bit, eight exponent bits and seven explicit fraction bits, offering a broad dynamic range but limited mantissa precision. E4M3 FP8 uses four exponent bits and three fraction bits; finite-range conventions depend on the exact format. MXFP4 is a block-scaled four-bit format, not a synonym for a scalar FP8 activation. A model can store routed-expert weights in MXFP4, dense weights in BF16 or online FP8, compressed cache in FP8, attention output in BF16 and softmax statistics in FP32 at the same time. Storage dtype, matrix-multiply operand dtype, accumulator dtype and output dtype must be stated separately.

### Sources

- [Kimi-K3 model card and checkpoint configuration](https://huggingface.co/moonshotai/Kimi-K3/tree/f831ab66814297da540d832a5235f8e904f29d06)
- [Campaign data-flow diagrams and measured stages](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/blob/b786b8cae9992c190c3d3323197c937d8ed8f8b8/docs/kimi/fp8-fmha-dataflow.md)
- [AMD ROCm GPU architecture specifications](https://rocm.docs.amd.com/en/latest/reference/gpu-arch-specs.html)

---

<a id="s-51"></a>
## Slide 51: Quantization changes numbers; dequantization rescales them

The scale preserves units, but it cannot recover rounding already lost.

- **Original X:** Higher-precision values
- **Choose s:** Use range / representable maximum
- **Quantize:** X₈ = round_FP8(clip(X/s))
- **Interpret:** X ≈ s × X₈

For E4M3FN: s = max|X| / 448, with explicit zero-range handling

To fit values into an FP8 format, divide by a scale, round to representable values and clip if necessary. A descale s converts a stored FP8 number back to its approximate real value. Some APIs instead pass its reciprocal, so always check the contract. For E4M3FN, the largest finite magnitude is 448; that value is not universal across all FP8 formats such as FNUZ variants. Dynamic quantization derives a scale from the current tensor, while static schemes reuse predetermined scales. Zero and extremely small ranges need explicit handling. Dequantizing an FP8 tensor restores its units and a wider storage dtype, but not the original BF16 information lost during quantization.

### Sources

- [Campaign data-flow diagrams and measured stages](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/blob/b786b8cae9992c190c3d3323197c937d8ed8f8b8/docs/kimi/fp8-fmha-dataflow.md)
- [Projection implementation and numerical limits](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/blob/b786b8cae9992c190c3d3323197c937d8ed8f8b8/docs/kimi/native-fp8-up-projection-audit.md)

---

<a id="s-52"></a>
## Slide 52: Scale granularity changes the accuracy/performance tradeoff

Per-tensor, per-channel and per-block scales cover different value groups.

- **Per-tensor:** One scale; simple metadata; outliers affect the whole tensor.
- **Per-channel:** Separate output-channel weight scales; tighter ranges.
- **Per-block:** Scales shared by small groups; extra packing and metadata.



A single scale is cheap to manage but must cover the widest value range in the tensor. If one value is an outlier, most other values occupy only a small part of the available FP8 range. Per-channel weight quantization uses one scale for each output channel, reducing that coupling. Block scaling such as MX formats trades finer local ranges against additional scale storage and layout constraints. The direct up-projection uses online-quantized per-output-channel FP8 weights plus the cache descale and supplied K/V output descales. These scales are not interchangeable. The new-token causal path must retain its original Q/K/V descales rather than substituting headroom scales intended for cached reconstruction.

### Sources

- [Kimi-K3 model card and checkpoint configuration](https://huggingface.co/moonshotai/Kimi-K3/tree/f831ab66814297da540d832a5235f8e904f29d06)
- [Campaign data-flow diagrams and measured stages](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/blob/b786b8cae9992c190c3d3323197c937d8ed8f8b8/docs/kimi/fp8-fmha-dataflow.md)
- [Causal FP8 implementation and accuracy evidence](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/blob/b786b8cae9992c190c3d3323197c937d8ed8f8b8/docs/kimi/causal-fp8-attention.md)

---

<a id="s-53"></a>
## Slide 53: Stage 0: before the FP8 attention optimizations

The compressed cache was already FP8; the attention operands were BF16.

- **FP8 cache:** Gather compressed history
- **BF16 projection:** Dequantize + kv_b_proj
- **BF16 K/V:** Expanded intermediate
- **BF16 FMHA:** BF16 Q joins K/V
- **Output + LSE:** BF16 / FP32

Established full configuration: 15,476.46881 total tokens/s/GPU

This is the campaign's baseline attention data flow, not a claim that the entire model was unoptimized or uniformly BF16. The compressed KV cache was already FP8. The cached-prefill reconstruction used BF16 projection operands and produced expanded BF16 keys and values. Query operands were BF16, and the fused attention kernel accepted BF16 Q/K/V. Gathering, dequantization and projection can share a kernel; the conceptual boxes do not imply a separate HBM buffer for every step. The benchmark includes all model and serving work. Its throughput label therefore describes a full configuration and is not an isolated FMHA kernel measurement.

### Sources

- [Baseline benchmark](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/39239112271/job/118015427785)
- [Campaign data-flow diagrams and measured stages](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/blob/b786b8cae9992c190c3d3323197c937d8ed8f8b8/docs/kimi/fp8-fmha-dataflow.md)

---

<a id="s-54"></a>
## Slide 54: Stage 1: dynamic operands for FP8 FMHA

Keep BF16 up-projection, then quantize the expanded Q/K/V.

- **FP8 cache:** Compressed history
- **BF16 up-project:** Expanded BF16 K/V
- **Dynamic quantize:** Q and expanded K/V
- **FP8 FMHA:** FP8 Q/K/V operands
- **Output + LSE:** BF16 / FP32

Established full configuration: 16,915.72154 total tokens/s/GPU

The first FP8 FMHA stage changes the attention operands while retaining BF16 kv_b_proj reconstruction. It dynamically quantizes Q and the expanded K/V to FP8 and passes the matching descales to the fused kernel. This introduces additional rounding and preparation work. In particular, expanded BF16 K/V are stored and then read by the quantizer before the FP8 output is stored. Q preparation can be reused within an immutable forward when the tensor identity and layout match. The measured full configuration is faster than the baseline, but the difference also reflects the complete runtime settings and cannot be claimed as a clean isolated speedup for the FMHA instruction sequence.

### Sources

- [Cached FP8 FMHA benchmark](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/40646139769)
- [Campaign data-flow diagrams and measured stages](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/blob/b786b8cae9992c190c3d3323197c937d8ed8f8b8/docs/kimi/fp8-fmha-dataflow.md)

---

<a id="s-55"></a>
## Slide 55: Stage 2: direct FP8 up-projection

FP8 cache × FP8 kv_b_proj → FP8 K/V storage → FP8 FMHA.

- **FP8 cache:** Compressed latent + shared K
- **FP8 × FP8:** Projection operands
- **FP32 accumulate:** Descale and assemble K/V
- **FP8 epilogue:** Write K₈/V₈ directly
- **FP8 FMHA:** Return BF16 O / FP32 LSE

Established full configuration: 17,406.08912 total tokens/s/GPU

The direct path uses compatible FP8 projection weights and compressed FP8 cache operands in AITER's fused gather/up-projection kernel. The matrix multiply accumulates in FP32. Its epilogue applies input and weight descales, splits the projected key/value components, appends the cached shared-key columns and rounds directly into FP8 output storage using supplied output descales. This removes the expanded BF16 K/V buffer between cached projection and attention, along with the separate cached-output quantization pass. It does not remove all quantization or all higher-precision arithmetic. Online weight conversion occurs after loading, and output scale policy still affects accuracy.

### Sources

- [FP8 projection plus cached FP8 FMHA benchmark](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/40695099743)
- [Campaign data-flow diagrams and measured stages](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/blob/b786b8cae9992c190c3d3323197c937d8ed8f8b8/docs/kimi/fp8-fmha-dataflow.md)
- [AITER fused FP8 gather and up-projection](https://github.com/ROCm/aiter/blob/bbef5a6f2d3733b293c80d2cee6a8d464e88bc01/aiter/ops/flydsl/gather_kv_b_proj.py)
- [ATOM MLA reference implementation](https://github.com/ROCm/ATOM/blob/661ef7661eef0d804d2b52d81ed41d80aeaf9fdb/atom/model_ops/attention_mla.py)

---

<a id="s-56"></a>
## Slide 56: Follow the units through the FP8 projection

Input, weight and output scales play separate roles.

- **Latent cache:** C ≈ s_C C₈
- **Weight channel j:** W_j ≈ s_W,j W₈,j
- **Accumulated projection:** Z_j ≈ s_C s_W,j Σ_i C₈,i W₈,j,i
- **FP8 output:** K₈ = round_FP8(K/s_K); similarly for V

FP8 multiplication operands → FP32 accumulation → scaled FP8 epilogue

Assume one cache descale s_C and an output-channel weight descale s_W,j for the illustrated projection contract. The raw FP8 dot product is in quantized units, so multiplying by both descales restores the physical activation units before applying the output quantization scale. Keys and values can use different output scales. The shared cached key columns must also be interpreted using their cache scale and encoded under the final K output scale. A missing scale, reciprocal-versus-descale confusion, wrong channel axis or reused scale from another forward can produce large numerical errors even when every tensor has the expected dtype. Kernel correctness must test nonunit, unequal scales.

### Sources

- [Campaign data-flow diagrams and measured stages](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/blob/b786b8cae9992c190c3d3323197c937d8ed8f8b8/docs/kimi/fp8-fmha-dataflow.md)
- [AITER fused FP8 gather and up-projection](https://github.com/ROCm/aiter/blob/bbef5a6f2d3733b293c80d2cee6a8d464e88bc01/aiter/ops/flydsl/gather_kv_b_proj.py)
- [Projection implementation and numerical limits](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/blob/b786b8cae9992c190c3d3323197c937d8ed8f8b8/docs/kimi/native-fp8-up-projection-audit.md)

---

<a id="s-57"></a>
## Slide 57: What direct output actually removes

It removes a large BF16 intermediate and its separate quantization pass.

- **Earlier path:** Write BF16 K/V → read BF16 K/V → quantize → write FP8 K/V.
- **Direct path:** Accumulate locally → scale/round → write FP8 K/V.
- **Logical sizes per token/rank:** BF16 expanded K/V: 7,680 bytes; FP8: 3,840 bytes.

12 × (192 + 128) = 3,840 expanded values per token per TP rank

At the measured local shape, each cached token expands into 12*(192+128)=3,840 values. A BF16 expanded buffer occupies 7,680 bytes per token per rank; FP8 occupies 3,840 bytes. Eliminating the BF16 materialization avoids one logical BF16 store and a later BF16 read, plus its separate quantizer's work, in that path. Those logical bytes are not a direct measurement of HBM traffic: caching, fusion, layout, workspace reuse and the chosen chunk size matter. The persistent MLA cache still stores its compressed representation; reducing an expanded temporary does not automatically increase the persistent KV cache capacity by the same amount.

### Sources

- [Campaign data-flow diagrams and measured stages](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/blob/b786b8cae9992c190c3d3323197c937d8ed8f8b8/docs/kimi/fp8-fmha-dataflow.md)
- [Projection implementation and numerical limits](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/blob/b786b8cae9992c190c3d3323197c937d8ed8f8b8/docs/kimi/native-fp8-up-projection-audit.md)

---

<a id="s-58"></a>
## Slide 58: Stage 3: reuse prepared FP8 for new-token attention

The causal call consumes existing Q₈/K₈/V₈ with their original descales.

- **Prepare once:** New-token Q/K/V → FP8 + original scales
- **Cached branch:** Direct FP8 cached projection + FP8 FMHA
- **New-token branch:** Causal FP8 FMHA on prepared tensors
- **Merge:** Use each branch's output and LSE



The new-token experiment enables causal FP8 attention only for eligible forwards that already prepare the direct-FP8 cached path. It reuses the existing new-token Q/K/V tensors rather than adding another quantization pass. The new-token causal call uses the original Q/K/V descales, while cached reconstructed K/V use separate output scales with headroom. Forwards without direct-KV preparation retain the BF16 causal path. Therefore the phrase “all three FP8 paths enabled” means configured and eligible, not a claim that every model layer and every forward is FP8. Full new-token throughput remains unestablished after the latest runner-quota failure; the completed 17,406 result is the earlier cached-only stage.

### Sources

- [Causal FP8 implementation and accuracy evidence](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/blob/b786b8cae9992c190c3d3323197c937d8ed8f8b8/docs/kimi/causal-fp8-attention.md)
- [New-token performance attempt; failed on runner quota](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/40804432321)
- [Campaign data-flow diagrams and measured stages](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/blob/b786b8cae9992c190c3d3323197c937d8ed8f8b8/docs/kimi/fp8-fmha-dataflow.md)

---

<a id="s-59"></a>
## Slide 59: Two attention branches need two different masks

The new query chunk sees all old context, plus its allowed new-token prefix.

- **Old cached prefix:** All its keys precede every new query
- **New-token block:** Use a causal triangle within the chunk
- **Same query outputs:** Merge by natural-log LSE
- **Scale discipline:** Keep cached-output and new-token descales separate



Let the old prefix contain P tokens and the new chunk N tokens. New query i sees old keys 0 through P-1 and new keys 0 through i. One efficient decomposition computes attention against the old prefix as a separate noncausal block, because all old keys are visible, and computes the new chunk using causal masking. Each branch returns normalized output and LSE. Merge those outputs using the mass-based rule. This decomposition is valid only when masks, positional/sequence offsets, softmax scale and per-tensor descales are consistent. Tests should modify future-token values under fixed scales and verify that earlier causal outputs do not change.

### Sources

- [Causal FP8 implementation and accuracy evidence](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/blob/b786b8cae9992c190c3d3323197c937d8ed8f8b8/docs/kimi/causal-fp8-attention.md)
- [FlashAttention: exact attention with IO awareness](https://arxiv.org/abs/2205.14135)

---

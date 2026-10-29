# Reference

[Handbook contents](../HANDBOOK.md) · [Worked examples](11-worked-examples.md)

<a id="s-72"></a>
## Slide 72: Tensor atlas: the cached-prefill path at TP8

Shapes shown per rank; token counts depend on the current packed call.

- **Hidden state X:** [T, 7,168]
- **Compressed cache row:** 512 latent + 64 shared K values
- **kv_b_proj weight:** [3,072, 512]
- **Query Q / key K:** [Tq,12,192] / [Tk,12,192]
- **Value V / output O:** [Tk,12,128] / [Tq,12,128]
- **LSE:** [12,Tq] in this adapter



Read each dimension by role. Tq is the packed number of query tokens and Tk the packed number of cached key/value tokens in this call. Twelve is the local head count, not the request count. The 192 key/query width includes 128 projected head-specific features and 64 shared-key layout features. Value width is 128. The weight's output dimension is 12*(128+128)=3,072. Other backends may transpose LSE or use different packed layouts, so this atlas is explicitly for the documented adapter. Logical shape does not determine contiguous layout; strides, alignment, scale granularity and pointer identity remain part of the kernel contract.

### Sources

- [Campaign data-flow diagrams and measured stages](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/blob/b786b8cae9992c190c3d3323197c937d8ed8f8b8/docs/kimi/fp8-fmha-dataflow.md)

---

<a id="s-73"></a>
## Slide 73: Glossary: model concepts

Keep the mathematical role separate from the implementation.

- **Attention / FMHA:** Weighted context retrieval / fused implementation
- **KDA / MLA:** Recurrent delta attention / latent-cache attention
- **Latent / projection:** Compressed feature space / learned matrix transform
- **MoE / router:** Expert pool / token-to-expert selection
- **Residual / gate:** Depth information path / learned modulation
- **LSE:** Logarithm of unnormalized attention mass



A model concept specifies an operation or parameterization. A kernel specifies how that operation executes on hardware. FMHA means fused multi-head attention and can accept BF16 or FP8 operands. A latent is a learned lower-dimensional representation, while quantization is a numerical representation change; the two can coexist. A gate modulates information, and different gates in KDA, MLA and MoE have different meanings. Residual mixing concerns information flow across depth. LSE is a scalar per query/head used to preserve normalization when attention is partitioned. The handbook glossary includes expanded definitions and cross-references to the equations where each term appears.

### Sources

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [Kimi-K3 model card and checkpoint configuration](https://huggingface.co/moonshotai/Kimi-K3/tree/f831ab66814297da540d832a5235f8e904f29d06)
- [Campaign data-flow diagrams and measured stages](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/blob/b786b8cae9992c190c3d3323197c937d8ed8f8b8/docs/kimi/fp8-fmha-dataflow.md)

---

<a id="s-74"></a>
## Slide 74: Glossary: execution concepts

Name the resource or mechanism before tuning it.

- **HBM / LDS:** Device memory / workgroup-shared on-chip memory
- **CU / XCD / wave:** Compute unit / GPU chiplet / 64-lane execution group
- **TP / DCP / EP:** Tensor / decode-context / expert parallelism
- **NUMA / CCD:** CPU-memory locality / CPU compute die
- **GEMM / MFMA:** Matrix multiplication / matrix instruction family
- **Quant / descale:** Round into compact format / restore numerical units



HBM capacity limits residency, while bandwidth limits how quickly large tensors move. LDS is a smaller explicitly managed memory inside the GPU execution hierarchy. CU, XCD and wavefront describe different scales of parallel execution. TP shards model calculations, DCP partitions context within a TP group and EP distributes expert ownership. NUMA and CCD concern host CPU topology and should not be confused with GPU XCDs. GEMM is an operation, MFMA an instruction family and a kernel a scheduled implementation using instructions and memory. A quantization scale may be represented as a multiplier or reciprocal depending on the API, so read the contract rather than inferring it from a variable name.

### Sources

- [AMD ROCm GPU architecture specifications](https://rocm.docs.amd.com/en/latest/reference/gpu-arch-specs.html)
- [vLLM Kimi-K3 AMD implementation](https://github.com/ajith-sirra-amd/vllm/tree/08e5d81a0d4fc76c20a97f9a517195db5a5dc5be/vllm/models/kimi_k3/amd)

---

<a id="s-75"></a>
## Slide 75: Source map and further reading

Pinned model facts, implementation references and campaign evidence.

- **Model + architecture:** Checkpoint, model card and AMD Kimi implementation
- **Algorithms + hardware:** Attention, FlashAttention, Kimi Linear and ROCm specs
- **FP8 implementation:** Campaign data flow, ATOM and AITER source
- **Evidence:** Benchmark runs, GSM8K results and native PR status



The companion source manifest provides full URLs and revision IDs, and the chapter documents link each technical claim to relevant sources. The model checkpoint and card are pinned to f831ab66814297da540d832a5235f8e904f29d06. The inspected native Kimi AMD implementation is pinned to 08e5d81a0d4fc76c20a97f9a517195db5a5dc5be. Campaign documents are pinned to b786b8cae9992c190c3d3323197c937d8ed8f8b8. Prototype serving evidence uses bb8bed732f514fd697856c5abd1d2bdd9d56a1e5 for new-token FP8 and 14250d6c6c17cd54fb7cd480c74c5c4eb5da2e78 for the established direct-projection stage. Software capabilities and measured results are dated, and should be rechecked when changing the source or runtime.

### Sources

- [Kimi-K3 model card and checkpoint configuration](https://huggingface.co/moonshotai/Kimi-K3/tree/f831ab66814297da540d832a5235f8e904f29d06)
- [vLLM Kimi-K3 AMD implementation](https://github.com/ajith-sirra-amd/vllm/tree/08e5d81a0d4fc76c20a97f9a517195db5a5dc5be/vllm/models/kimi_k3/amd)
- [AMD ROCm GPU architecture specifications](https://rocm.docs.amd.com/en/latest/reference/gpu-arch-specs.html)
- [FlashAttention: exact attention with IO awareness](https://arxiv.org/abs/2205.14135)
- [Kimi Linear: Kimi Delta Attention](https://arxiv.org/abs/2510.26692)
- [Campaign data-flow diagrams and measured stages](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/blob/b786b8cae9992c190c3d3323197c937d8ed8f8b8/docs/kimi/fp8-fmha-dataflow.md)
- [ATOM MLA reference implementation](https://github.com/ROCm/ATOM/blob/661ef7661eef0d804d2b52d81ed41d80aeaf9fdb/atom/model_ops/attention_mla.py)
- [AITER fused FP8 gather and up-projection](https://github.com/ROCm/aiter/blob/bbef5a6f2d3733b293c80d2cee6a8d464e88bc01/aiter/ops/flydsl/gather_kv_b_proj.py)
- [New-token FP8 full GSM8K evaluation](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/40712931283)
- [Native vLLM draft PR](https://github.com/vllm-project/vllm/pull/64288)

---

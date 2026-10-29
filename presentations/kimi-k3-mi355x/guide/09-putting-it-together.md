# Putting it together

[Handbook contents](../HANDBOOK.md) · [Worked examples](11-worked-examples.md)

<a id="s-70"></a>
## Slide 70: Trace one new token through the complete system

The architecture and serving machinery now form one connected picture.

- **Schedule:** Choose tokens; resolve cache/state
- **Project:** Create Q/K/V or KDA update inputs
- **Mix context:** KDA recurrence or MLA attention
- **Transform:** Latent MoE and shared branch
- **Mix depth:** Attention Residuals
- **Generate:** Logits, sampling and state commit



For a new token, the scheduler resolves which context and recurrent state are available and builds metadata. Each decoder layer performs its configured sequence-mixing operation: a KDA update or MLA attention. MLA may use latent-space decode or an expanded prefill path, depending on the workload. The feature transform is dense in the first layer and MoE thereafter. Attention Residuals combine information across depth. TP and DCP communication appear wherever distributed operands or results require it. After the final normalization and vocabulary projection, the sampler or speculative verifier selects output progress. Caches and recurrent state are updated consistently before the next step. Precision choices belong to individual edges and operations throughout this flow.

### Sources

- [Kimi-K3 model card and checkpoint configuration](https://huggingface.co/moonshotai/Kimi-K3/tree/f831ab66814297da540d832a5235f8e904f29d06)
- [vLLM Kimi-K3 AMD implementation](https://github.com/ajith-sirra-amd/vllm/tree/08e5d81a0d4fc76c20a97f9a517195db5a5dc5be/vllm/models/kimi_k3/amd)
- [Campaign data-flow diagrams and measured stages](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/blob/b786b8cae9992c190c3d3323197c937d8ed8f8b8/docs/kimi/fp8-fmha-dataflow.md)

---

<a id="s-71"></a>
## Slide 71: Six questions that check understanding

Answer these before interpreting the next performance result.

- **1–2:** Why cache K/V but usually not Q? Why does KDA need state recovery?
- **3–4:** Why are TP8 + DCP8 still eight GPUs? Why does a legacy RoPE field not prove rotation?
- **5–6:** What BF16 traffic does direct FP8 projection remove? Why is GSM8K not an LSE test?



Suggested answers: Future steps use new queries, while cached K/V preserve earlier context. KDA is recurrent, so rejected speculative tokens must not contaminate future state. DCP partitions context within the tensor-parallel group rather than creating another independent eight-device dimension. The K3 checkpoint uses NoPE in MLA even though generic layout names retain a RoPE-width field. Direct FP8 cached projection avoids materializing and rereading expanded BF16 K/V and avoids the separate cached-output quantizer, while retaining higher-precision accumulation and required scaling. GSM8K measures end-to-end task outcomes; LSE tests measure attention normalization fidelity and its effect on chunk merging. Each is necessary evidence for its own claim.

### Sources

- [Kimi-K3 model card and checkpoint configuration](https://huggingface.co/moonshotai/Kimi-K3/tree/f831ab66814297da540d832a5235f8e904f29d06)
- [Campaign data-flow diagrams and measured stages](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/blob/b786b8cae9992c190c3d3323197c937d8ed8f8b8/docs/kimi/fp8-fmha-dataflow.md)
- [Projection implementation and numerical limits](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/blob/b786b8cae9992c190c3d3323197c937d8ed8f8b8/docs/kimi/native-fp8-up-projection-audit.md)

---

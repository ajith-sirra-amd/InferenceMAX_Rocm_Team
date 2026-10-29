# Orientation

[Handbook contents](../HANDBOOK.md) · [Worked examples](11-worked-examples.md)

<a id="s-01"></a>
## Slide 01: Kimi-K3 on AMD MI355X

From attention fundamentals to FP8 execution

- **Understand:** Attention, KDA, MLA and MoE
- **Trace:** One token across eight GPUs
- **Optimize:** Precision, memory traffic and correctness



This is a detailed teaching deck for an engineer who wants to understand both the model and its execution. Read the diagrams from left to right and use click-by-click builds in PowerPoint Slide Show. Green always identifies FP8, blue BF16, purple FP32 and amber a conversion or decision. Mathematical examples are deliberately small; checkpoint dimensions are labeled separately. The Markdown handbook contains deeper explanations and worked examples. Model facts are pinned to the Kimi-K3 checkpoint revision, while optimization evidence is pinned to this campaign. The presentation does not assume that every FP8 capability runs on every forward.

### Sources

- [Kimi-K3 model card and checkpoint configuration](https://huggingface.co/moonshotai/Kimi-K3/tree/f831ab66814297da540d832a5235f8e904f29d06)
- [vLLM Kimi-K3 AMD implementation](https://github.com/ajith-sirra-amd/vllm/tree/08e5d81a0d4fc76c20a97f9a517195db5a5dc5be/vllm/models/kimi_k3/amd)
- [Campaign data-flow diagrams and measured stages](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/blob/b786b8cae9992c190c3d3323197c937d8ed8f8b8/docs/kimi/fp8-fmha-dataflow.md)

---

<a id="s-02"></a>
## Slide 02: A learning path with six stops

Build each concept before using it to explain performance.

- **01:** Attention from first principles
- **02:** Kimi-K3 architecture
- **03:** GPU execution and memory
- **04:** Serving across eight GPUs
- **05:** FP8 data flow and numerics
- **06:** Evidence and optimization



The full deck is designed for a long workshop or several shorter sessions. First learn why Q, K and V exist and how attention produces an output. Then distinguish KDA's recurrent state from MLA's compressed token cache. Next map matrix operations onto MI355X, and understand the scheduler and distributed collectives. Only after those pieces are in place do we introduce FP8 FMHA and the direct FP8 up-projection. The final section explains how to establish that a faster path is both executed and sufficiently accurate. Each slide links to its Markdown chapter. Use the glossary and tensor atlas as references during discussion.

### Sources

- [Kimi-K3 model card and checkpoint configuration](https://huggingface.co/moonshotai/Kimi-K3/tree/f831ab66814297da540d832a5235f8e904f29d06)

---

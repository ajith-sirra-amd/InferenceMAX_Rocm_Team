# Sources and evidence

Prepared 2026-10-29. All diagrams are original, editable teaching schematics.

- **model:** [Kimi-K3 model card and checkpoint configuration](https://huggingface.co/moonshotai/Kimi-K3/tree/f831ab66814297da540d832a5235f8e904f29d06)
- **code:** [vLLM Kimi-K3 AMD implementation](https://github.com/ajith-sirra-amd/vllm/tree/08e5d81a0d4fc76c20a97f9a517195db5a5dc5be/vllm/models/kimi_k3/amd)
- **activation:** [SiTU-GLU reference implementation](https://github.com/ajith-sirra-amd/vllm/blob/08e5d81a0d4fc76c20a97f9a517195db5a5dc5be/vllm/model_executor/layers/activation.py)
- **amd:** [AMD ROCm GPU architecture specifications](https://rocm.docs.amd.com/en/latest/reference/gpu-arch-specs.html)
- **flash:** [FlashAttention: exact attention with IO awareness](https://arxiv.org/abs/2205.14135)
- **attention:** [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- **kda:** [Kimi Linear: Kimi Delta Attention](https://arxiv.org/abs/2510.26692)
- **flow:** [Campaign data-flow diagrams and measured stages](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/blob/b786b8cae9992c190c3d3323197c937d8ed8f8b8/docs/kimi/fp8-fmha-dataflow.md)
- **causal:** [Causal FP8 implementation and accuracy evidence](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/blob/b786b8cae9992c190c3d3323197c937d8ed8f8b8/docs/kimi/causal-fp8-attention.md)
- **audit:** [Projection implementation and numerical limits](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/blob/b786b8cae9992c190c3d3323197c937d8ed8f8b8/docs/kimi/native-fp8-up-projection-audit.md)
- **atom:** [ATOM MLA reference implementation](https://github.com/ROCm/ATOM/blob/661ef7661eef0d804d2b52d81ed41d80aeaf9fdb/atom/model_ops/attention_mla.py)
- **aiter:** [AITER fused FP8 gather and up-projection](https://github.com/ROCm/aiter/blob/bbef5a6f2d3733b293c80d2cee6a8d464e88bc01/aiter/ops/flydsl/gather_kv_b_proj.py)
- **base:** [Baseline benchmark](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/39239112271/job/118015427785)
- **fmha:** [Cached FP8 FMHA benchmark](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/40646139769)
- **proj:** [FP8 projection plus cached FP8 FMHA benchmark](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/40695099743)
- **gsm:** [New-token FP8 full GSM8K evaluation](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/40712931283)
- **run:** [New-token performance attempt; failed on runner quota](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/40804432321)
- **pr:** [Native vLLM draft PR](https://github.com/vllm-project/vllm/pull/64288)

## Evidence boundary

The 17,406.08912 result is the completed cached FP8 projection + cached FP8 FMHA prototype. New-token FP8 full performance is not established. Run 40804432321 failed on runner disk quota. Prototype GSM8K and native-PR validation are separate.

# Patch-free Moonshot Kimi native quantization benchmark

- Model: moonshotai/Kimi-K3; draft: Inferact/Kimi-K3-DSpark.
- Pristine pinned vLLM ROCm image; no runtime vLLM/AITER source patches.
- Native --quantization-config: existing 15 explicit per-channel FP8 target patterns.
- Prior loader inventory: 513 main-model linear layers.
- Attention targets plus dense/shared gate_up_proj and down_proj; existing routed-expert MXFP4 retained.
- Convolutions, vision, embeddings and lm_head are untargeted.
- TP8, DCP8, C48; max96 sequences; batch8192; model context1048576.
- FP8 KV; SimpleCPUOffloadConnector; existing2249GB CPU allocation.
- Warmup10 per lane; benchmark3600seconds; no GSM8K.
- Automatic weight loading; no --load-format override.
- Removed source patchers, numerical probes, cached-context test requests and experimental runtime marker gates.
- Native kernel logs and loader inventory will provide execution evidence.
- ATOM uses different native kernels, activation fusions, graph settings and offload stack; similar weight coverage does not guarantee similar throughput.
- This run does not claim experimental FP8 fused KV up-projection or FP8 ASM FMHA.
- Host Docker SIGTERM stops remain a separate unresolved issue.

# Moonshot Kimi-K3 producer FP8 fusion experiment

**English** | [简体中文](ATOM-producer-fusions_zh.md)

- Main model: `moonshotai/Kimi-K3`; draft: `Inferact/Kimi-K3-DSpark`.
- Pinned vLLM: `1b2a666391f2a24228a1436b64d547c34cc6b6ce`.
- Problem: BF16 producer outputs can require an intermediate HBM buffer and a separate FP8 conversion before an FP8 linear.
- Proposal: combine selected producers with per-token FP8 conversion; feed FP8 data and scales directly to the existing FP8 GEMM, which returns BF16.
- Inspiration: [ATOM Kimi producer quantization](https://github.com/ROCm/ATOM/tree/661ef7661/atom/model_ops/kimi_k3). This is a narrow vLLM adaptation, not an ATOM backend or FP8 FMHA port.

| Operation | Existing flow | Experiment |
|---|---|---|
| KDA `f_a → f_b_proj` | Strided BF16 input → conversion → GEMM | Read strided input and produce contiguous FP8 plus scales in one kernel |
| MLA output gate → `o_proj` | Sigmoid multiply → BF16 buffer → conversion → GEMM | Sigmoid multiply and FP8 conversion in one kernel |
| Dense/shared MLP → `down_proj` | SiTU → BF16 buffer → conversion → GEMM | SiTU and FP8 conversion in one kernel |
| Dense/shared weights | BF16 | Load-time per-channel FP8 |
| Latent KV reconstruction | Existing patched FP8-weight up-projection → BF16 K/V | Retained |
| Cached latent KV all-gather | FP8 payload transported as uint8 | Retained; runtime marker required |
| Prefill FMHA | BF16 Q/K/V, 192/128 head dimensions | Retained |
| Routed experts | Checkpoint MXFP4 | Retained |
| Router/residual/convolution weights | Existing native precision | Retained |

- Additional targets: shared `gate_up_proj/down_proj` in 92 MoE layers; dense `language_model.model.layers.0.mlp.gate_up_proj/down_proj`.
- Eleven existing attention target patterns remain.
- Raw additional weight saving: approximately **1.50 GiB/GPU at TP8**, before scale storage and runtime effects. Actual KV capacity must come from the server log.
- Producer fusion retains BF16 rounding in registers before FP8 conversion; FP8 quantization still introduces rounding error.
- Existing KDA output normalization and other untargeted producers may still use separate conversion kernels.
- The rowwise Torch FP8 GEMM advertises prequantized input only for dynamic per-token scales without token padding. Other unsupported FP8 consumers fail before graph capture.
- BF16 MLP consumers retain their original activation path.

| Validation / execution | Requirement |
|---|---|
| CPU workflow | Bash and Python syntax pass before GPU scheduling |
| Installer | Exact pinned-source anchors; compile all transformed files before writing container sources |
| GPU preflight | All 8 GPUs; actual producer kernels and FP8 scaled-MM; strided, zero and empty cases; TP8 widths 128/1536/768/4224 |
| Post-load | 69 strided, 24 gated, 92 shared and 1 dense consumer groups; FP8 weights/scales and compatible activation keys |
| Path evidence | All 8 GPUs emit producer, fused KV, uint8 transport and BF16 attention markers |
| Benchmark | Warmup 10 requests/lane; measured duration 3600 seconds; fast mode disabled; automatic weight loader |
| Accuracy | No GSM8K run requested for this experiment; kernel checks do not establish model accuracy |
| Performance | Throughput gain and 23M KV capacity remain unproven |

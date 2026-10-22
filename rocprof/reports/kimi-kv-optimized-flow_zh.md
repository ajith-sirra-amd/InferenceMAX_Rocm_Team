# Kimi-K3：优化 KV 通信与注意力流程

[English](./kimi-kv-optimized-flow.md) | **中文**

## 关键事实

- FP8 all-gather 只传输缓存字节，不使用 `kv_b_proj`。
- BF16 `kv_b_proj` 是固定版本融合重建实现的要求，不是 collective 的要求。
- FP8 缓存保存压缩 latent；FMHA 使用重建后的完整多头 K/V。
- 重建是矩阵投影，不只是头排列。
- FP8 投影权重不意味着 Q/K/V 输出为 FP8。
- 单个内核更快不能证明端到端推理更快。

## 张量与维度

| 张量/操作 | 维度 | 用途 |
|---|---|---|
| Hidden states X | [T,7168] | 投影输入 |
| KV latent C | [T,512] | 压缩缓存的 latent 部分 |
| RoPE key | [T,64] | 位置分量 |
| 缓存行 | [T,576] | 512 latent + 64 RoPE |
| TP8 本地 kv_b 权重 | [3072,512] | 12头 × (128 key + 128 value) |
| 重建 | C × W_kv_b^T → [T,3072] | 本地非RoPE K及V |
| 本地 K | [T,12,192] | 128重建 + 64 RoPE |
| 本地 V | [T,12,128] | 重建 values |

- T 为相关 chunk 的 token 数；缓存分布在 DCP ranks，重建维度对应本地 TP 头。
- Dequantization 不会恢复 quantization 丢失的信息。

## 精度与机制

| 阶段 | 当前可用流程 | 优化候选 | 收益/条件 |
|---|---|---|---|
| Attention projections | FP8 weights/input arithmetic，按需 BF16输出 | 同样精度，支持时融合输入量化 | 降低GEMM成本，保留准确normalization/RoPE |
| KV cache | FP8 | FP8 | 节省容量/HBM |
| All-gather | FP8字节，uint8 collective | 相同 | BF16一半payload，无新增缓存量化 |
| 重建权重 | BF16 | FP8 + channel scales | 更少权重流量，需要融合内核支持 |
| Reorder/decode/projection/packing | BF16权重的单个原生融合内核 | 支持FP8权重/scales的单个融合内核 | 避免缓存中间张量及单独launch |
| 重建计算 | 由原生实现决定 | 合适的高精度accumulation | 验证实现与误差 |
| 重建K/V | BF16 | 优先BF16，验证FP8 FMHA后才采用scaled FP8 | 减少精度转换 |
| FMHA | BF16 | 最快且准确的BF16或scaled FP8 | 测量包括准备阶段的完整调用 |
| 输出 | BF16 | BF16 | 供后续操作使用 |

## 三种流程

| 流程 | Pipeline | 状态 |
|---|---|---|
| 当前原生融合 | FP8 cache → FP8-byte gather → fused reorder/dequant/BF16-weight reconstruction/packing → BF16 K/V → BF16 FMHA | 固定版本已有，需验证实际执行 |
| FP8重建权重 | FP8 cache → FP8-byte gather → fused FP8-weight/scales reconstruction → BF16 K/V → BF16 FMHA | 候选改进，固定版本融合内核尚不支持 |
| FP8 FMHA | FP8 cache → FP8-byte gather → fused reconstruction/scaled FP8 K/V + scaled FP8 Q preparation → FP8 FMHA → BF16 output | 需准确率及端到端性能验证 |

- 完整多头FMHA不能直接使用压缩latent，不能移除重建。
- 直接输出FP8仍执行量化，但能融合该工作并避免BF16中间张量。
- FP32 accumulation不能恢复FP8量化损失。
- 转换、padding、packing、launch开销可能抵消FP8 FMHA收益。

## Moonshot实验

- 使用 `moonshotai/Kimi-K3`，保留compressed-tensors MXFP4 experts。
- Attention采用online `fp8_per_channel`；`kv_b_proj`保留BF16。
- Targets：`in_proj_qkvgfab`, `f_b_proj`, `g_a_proj`, `g_b_proj`, `o_proj`, `fused_qkv_a_proj`, `q_b_proj`, `q_proj`, `kv_a_proj_with_mqa`, `g_proj`。
- 保留convolutions、shared experts、residual projections。
- 精度方案接近AMD，但重建为BF16，量化数值可能不同。
- 目标：KV pool接近23M tokens，同时恢复原生融合路径。
- 验收：全部8 GPU的quantization inventory、eligibility、uint8 collective、fused execution，以及throughput/pool/errors。
- Smoke：C48 TP8/DCP8，warmup1/lane，600秒。
- 准确率另行real-verification eval；synthetic throughput不是accuracy test。

## 最新Smoke结果

| 项目 | 证据 |
|---|---|
| Run | [39717243502](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/39717243502) |
| Loader | 显式fastsafetensors |
| Numerical preflight | GPU0–7全部通过 |
| 失败 | GPU4 HSA_STATUS_ERROR_OUT_OF_RESOURCES；free memory0 MB；nogds GPU operation failed:701 |
| 阶段 | Weight loading/worker initialization，server未ready |
| Quantization inventory | 没有完成标记 |
| Throughput/KV pool | 未测得 |
| 判断 | Loader执行期间资源耗尽；尚未证明硬件故障或确切loader原因 |
| 下一步诊断 | 检查loading/staging内存及profiler交互，先在相同GPU隔离loader |

## 参考

- [AMD model card](https://huggingface.co/amd/Kimi-K3-Quark-MXFP4-AttnFP8)
- [固定版本原生MLA backend](https://github.com/vllm-project/vllm/blob/1b2a666391f2a24228a1436b64d547c34cc6b6ce/vllm/v1/attention/backends/mla/rocm_aiter_mla.py)
- [固定版本gather/reconstruction](https://github.com/vllm-project/vllm/blob/1b2a666391f2a24228a1436b64d547c34cc6b6ce/vllm/v1/attention/ops/rocm_aiter_mla_prefill.py)

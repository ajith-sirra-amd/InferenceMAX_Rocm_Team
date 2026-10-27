# Moonshot Kimi-K3 producer FP8 融合实验

[English](ATOM-producer-fusions.md) | **简体中文**

- 主模型：`moonshotai/Kimi-K3`；草稿模型：`Inferact/Kimi-K3-DSpark`。
- 固定 vLLM：`1b2a666391f2a24228a1436b64d547c34cc6b6ce`。
- 问题：BF16 producer 的输出在送入 FP8 linear 前，可能需要 HBM 中间缓冲区和独立 FP8 转换。
- 方案：把部分 producer 与逐 token FP8 转换合并，将 FP8 数据及 scale 直接送入现有 FP8 GEMM，输出仍为 BF16。
- 参考：[ATOM Kimi producer quantization](https://github.com/ROCm/ATOM/tree/661ef7661/atom/model_ops/kimi_k3)。这是针对 vLLM 的小范围适配，不是 ATOM backend 或 FP8 FMHA 移植。

| 操作 | 原有流程 | 实验流程 |
|---|---|---|
| KDA `f_a → f_b_proj` | 非连续 BF16 输入 → 转换 → GEMM | 单 kernel 读取非连续输入并生成连续 FP8 与 scale |
| MLA 输出门控 → `o_proj` | sigmoid 乘法 → BF16 缓冲区 → 转换 → GEMM | 合并 sigmoid 乘法与 FP8 转换 |
| dense/shared MLP → `down_proj` | SiTU → BF16 缓冲区 → 转换 → GEMM | 合并 SiTU 与 FP8 转换 |
| dense/shared 权重 | BF16 | 加载时逐输出通道 FP8 量化 |
| latent KV 重建 | 已有补丁使用 FP8 权重上投影，输出 BF16 K/V | 保留 |
| cached latent KV all-gather | FP8 payload 以 uint8 传输 | 保留；要求运行时日志 |
| prefill FMHA | BF16 Q/K/V，192/128 head 维度 | 保留 |
| routed experts | checkpoint MXFP4 | 保留 |
| router/residual/convolution 权重 | 原有精度 | 保留 |

- 新增 target：92 个 MoE 层的 shared `gate_up_proj/down_proj`，以及 dense `language_model.model.layers.0.mlp.gate_up_proj/down_proj`。
- 保留已有 11 个 attention target pattern。
- TP8 新增原始权重节省约 **1.50 GiB/GPU**，尚未扣除 scale 与运行时影响。实际 KV 容量以 server log 为准。
- producer 在寄存器中保留 BF16 舍入，再转 FP8；FP8 量化仍有舍入误差。
- 原有 KDA 输出归一化及其他未融合 producer 仍可能需要独立转换。
- rowwise Torch FP8 GEMM 仅在逐 token 动态 scale 且无 token padding 时声明支持预量化输入；其他不兼容 FP8 consumer 会在 graph capture 前失败。
- BF16 MLP consumer 保留原激活路径。

| 验证 / 执行 | 要求 |
|---|---|
| CPU workflow | GPU 调度前通过 Bash 与 Python 语法检查 |
| installer | 精确匹配固定源码；写入容器前编译全部转换结果 |
| GPU preflight | 全 8 GPU 实际运行 producer 与 FP8 scaled-MM；覆盖非连续、全零、空输入及 TP8 宽度 128/1536/768/4224 |
| 加载后检查 | 69 strided、24 gated、92 shared、1 dense consumer group；验证 FP8 权重/scale 与输入能力 |
| 路径证据 | 全 8 GPU 输出 producer、fused KV、uint8 传输与 BF16 attention 日志 |
| benchmark | 每 lane warmup 10；测量 3600 秒；关闭 fast mode；自动 weight loader |
| 精度 | 用户未要求本实验运行 GSM8K；kernel 检查不能证明模型精度 |
| 性能 | 吞吐提升及 23M KV 容量尚未证实 |

- [run40464699186](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/40464699186) 在模型加载前停止：GPU0 通过64项数值及 FP8 scaled-MM 检查；caller 仍处于 GPU0，首个非空 GPU1 producer launch 失败。现使用输入 tensor 所在 device guard；空输入不再输出 kernel 已执行日志。全 GPU preflight 覆盖 tensor GPU 与 caller 当前 GPU 不同的情况。

# Kimi prefill：执行路径、开销与下一次测试门槛

[English](kimi-prefill-overhead-audit.md) · [基线证据](atom-attention-BASELINES.md)

## 问题与目标

- 目标：在 8×MI355X、vLLM 上超过 Moonshot Kimi-K3 基线 **15,476.47 total tokens/sec/GPU**。
- FP8 FMHA 核函数更快，不代表完整 attention 调用更快；还需计入展开、量化、通信、合并。
- 旧版全 FP8 agentic smoke：11,789.99 total tokens/sec/GPU，且存在请求错误；短测不能与完整基线直接比较。
- 当前方案：causal 和小 cached chunk 保留 BF16，较大 cached chunk 使用 FlyDSL FP8。
- 当前 smoke：[40643563020](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/40643563020)；测试 SHA：`e4cf47a7650321e4bd40c83746088a2679ec0509`。实际调用及 KV 容量已通过 artifact 验证，详见文末状态。

## 先区分模型和基线

| 测试 | 模型 | GPU KV tokens | Total tokens/sec/GPU | Output tokens/sec/GPU |
|---|---|---:|---:|---:|
| 主基线 39239112271 | Moonshot Kimi-K3 | 20,951,355 | 15,476.47 | 103.05 |
| 容量参考 39601846940 | AMD Quark Kimi-K3 | 23,098,919 | 15,258.78 | 见原始 JSON |
| 旧全 FP8 smoke 40634885348 | Moonshot Kimi-K3 | 22,695,620 | 11,789.99 | 61.13 |

- 历史文档将主基线称为“AMD baseline”，但实际 checkpoint 为 Moonshot。
- AMD Quark 与 Moonshot 的权重量化不同；当前仅使用 Moonshot，并设置在线 FP8 targets。
- 权重更小不保证 KV pool 达到 23M；运行时 scratch、graphs、draft 等分配也会影响容量。
- Total throughput 包含输入和缓存命中的输入；必须同时报告 output throughput 和 cache-hit rate。
- 基线统计窗口 3629.50 秒；旧 smoke 为 620.30 秒，并非匹配实验。

## TP8 / DCP8 精度与矩阵流程

- 96 attention heads，TP8 后每 rank **12 heads**。
- T 表示当前操作的 token 数；collective 还存在 packing/padding。

| 步骤 | 每 rank 张量或矩阵 | 精度 | 说明 |
|---|---|---|---|
| 压缩 KV cache | C:[T,512]，K_pe:[T,64] | FP8 + scale | 不是按 head 展开的 K/V |
| Native fused DCP gather | 紧凑 [T,576] | FP8 通信 | 逻辑 payload 为 576 bytes/token，不含 padding |
| 潜空间上投影 | C[T,512] × W_kv_bᵀ[512,3072] | 当前 W_kv_b 为 BF16 | Native fused path 根据权重和 scale 展开 |
| Head split + RoPE | K:[T,12,192]，V:[T,12,128] | BF16 | K 包含 128 non-RoPE + 64 RoPE |
| Prefill query | Q:[Tq,12,192] | BF16 | FP8 权重不等于 FP8 激活输出 |
| Causal / 小 cached FMHA | 相同 head 维度 | BF16 输入和输出 | 避免转换开销 |
| 大 cached chunk 量化 | Q/K/V + per-tensor scales | BF16 → E4M3 FP8 | 同一 forward 可复用同一 Q view |
| 大 cached FMHA | Q8/K8/V8，192/128 head | FP8 输入，BF16 输出，FP32 LSE | 使用 ATOM 同样的 AITER FlyDSL API |
| Chunk merge | 局部输出与 LSE | BF16 / FP32 | 保留 native 合并和归一化 |
| Decode / draft | Native vLLM AITER | 原配置 | 不由此 prefill 扩展替代 |

**当前流程：**
`压缩 FP8 cache → native fused FP8 gather + 潜空间展开 → BF16 K/V → FP8 Q/K/V 量化 → FlyDSL FP8 FMHA → BF16 输出 + FP32 LSE → native merge`。

- 当前是 **FP8 通信 + BF16 展开**，随后再次量化以进行 FP8 FMHA。
- 尚未实现直接输出 expanded FP8 K/V。
- BF16 kv_b 权重是当前固定版本 native fused 实现的适配条件，并非 FP8 all-gather 的数学要求。
- 固定版本 fallback 先 gather compact BF16 latent KV，再重排并调用 kv_b_proj 展开；通信逻辑 payload 为1,152 bytes/token，compact FP8 为576 bytes/token。当前一次性 hook 观察实际 module 调用来判定 route。
- FP8 kv_b 权重可以省显存，但不兼容的权重格式会使当前 fused 路径不可用。

## 已测开销

12 heads，共享 GPU host 上的 event timing；不是完整模型 throughput，也不是硬件计数器 roofline。

| Noncausal case | BF16 complete | FP8 量化 + FMHA complete | BF16/FP8 |
|---|---:|---:|---:|
| Q256 / KV4096 | 88.47 µs | 107.03 µs | 0.827×，退化 |
| Q1024 / KV8192 | 177.39 µs | 107.13 µs | 1.656×，改善 |
| 两个 chunk，Q1024 / KV8192 | 330.84 µs | 201.20 µs | 1.644×，改善 |

| 项目 | 成本 | 证据类型 |
|---|---|---|
| QKV quant | 约 55.3–55.7 µs | 实测 |
| Q 复用时 KV-only quant | 约 51.7 µs | 实测；转换大部分成本仍存在 |
| FP8 FMHA only | 两个单 chunk case 约 36.3 / 50.5 µs | 实测，不含量化 |
| BF16 expanded K/V | 7,680 bytes/token/rank | 逻辑张量大小 |
| FP8 expanded K/V | 3,840 bytes/token/rank | 逻辑张量大小 |
| KV 转换最少读写 | 11,520 bytes/token/rank | 估算，不含 reduction、scale、重读和分配 |
| BF16 kv_b 对比 FP8 权重 | 24 MLA layers 约增加 36 MiB/GPU | 原始权重估算，不等于 KV pool 变化 |
| 旧 smoke hipBLASLt recovery | 12 shapes 共96 warnings，m2880/k7168 | 观察到 BLAS fallback；未测耗时和实际调用次数 |
| DCP / expansion GEMM / offload / merge | 尚未独立测量 | 不可用峰值带宽替代实测 |
| 旧 smoke 模型加载 | 约 23.6 分钟，184.12 GiB/GPU | Startup，未计入 reporting window |

- 相对于未量化 FP64 reference，FP8 numerical RMS drift 约 4–5%；不是模型 accuracy 证据。
- 用户要求不跑 GSM8K。
- 当前 shape policy：max_query_len < 512 **或** max_seq_len < 8192 时 BF16，否则 FP8；来自已测 case，不是完整 shape sweep。

## 已移除或减少的开销

| 开销 | 修改 | 剩余约束 |
|---|---|---|
| Causal FP8 conversion 回退 | 保留 BF16 causal | 不宣称 causal FP8 |
| 小 context 转换不划算 | Threshold 以下 BF16 | 后续可优化阈值 |
| 多 chunk 重复量化 Q | 同一 forward 复用 Q，在 finally 清理 | KV 仍需量化，不跨 forward 复用 |
| 主模型未使用的 ASM PS scratch | 分配前禁用 | Draft native PS 仍可能存在 |
| FP8 kv_b 权重不兼容 fused 路径 | 从 targets 删除 kv_b_proj | 约 36 MiB/GPU 原始权重成本 |
| 大量诊断打印 | 一次性 marker 和一次 module hook | 无逐请求 tensor dump |
| 模型 accuracy 测试 | 独立完整 GSM8K run40658068830，真实验证 | 两项指标通过；未触发 FP8 FMHA |
| NCCL 日志刷屏 | NCCL_DEBUG=VERSION + 单节点 loopback bootstrap | 降低日志不等于修复 hang |

- 相邻层 FP8 → FP4 → FP8 **权重格式**并不必然触发激活 cast；应检查实际 GEMM 输入/输出 dtype 和 scale。
- 未设置 --load-format，使用 auto loader；SAFETENSORS_FAST_GPU=1 不会强制 fastsafetensors loader。
- 启用 --language-model-only。
- 不 patch 安装后的 vLLM/AITER source，仅通过 PYTHONPATH 注册本仓库扩展。

## 下一次 full run 门槛

| Gate | 要求 |
|---|---|
| 模型加载前 | numerical preflight PASS + QUERY_REUSE_PASS |
| Prefix 请求 | 20/20 成功，0 failed |
| 缓存 | Prefix-cache-hit counter 增长 |
| 8 workers | BF16 causal、FP8 cached、主 ASM PS 禁用、Q reuse、native fused FP8 gather marker |
| 显存 | 实际 GPU KV cache capacity 行 |
| 版本 | Full checkout 固定为通过 smoke 的 source SHA |

- Prefix stages：32K/1024/OSL16/CONC8；128K/1024/OSL8/CONC1。
- Context workspace cap 为 64K；长 prefix 触发多个 context chunk。
- Watcher 每 60 秒检查；通过后自动 dispatch full agentic replay：warmup10/lane，3600 秒，无 GSM8K。
- Controller 防止重复 dispatch；失败先检查日志并修复。
- Full 结果必须报告 pool、total/output throughput、错误、时长和 cache-hit rate。

## 代码入口

| 路径 | 作用 |
|---|---|
| upstream/InferenceX/benchmarks/single_node/agentic/atom_attention/inferencex_atom_attention/backend.py | Hybrid policy、Q reuse 生命周期、metadata 和 route marker |
| 同目录 quantize.py | QKV quant 与 KV-only specialization |
| 同目录 register.py | vLLM 注册 |
| 上级 atom_attention/path_probe.py | Prefix smoke 与 artifact gate |
| upstream/InferenceX/benchmarks/single_node/agentic/kimik3_fp4_mi355x_mtp.sh | Serve command、targets、preload checks 和 replay |
| utils/atom_attention/preflight.py | Numerical、strided-input、graph replay 验证 |
| utils/atom_attention/continue_benchmark.py | Artifact gate、固定 SHA dispatch、监控 |

- 使用已有 perf/kimi-attnfp8-kv-allgather-20261021 分支。
- 不修改 chore/sa-agentx-v1.0；代码仅写入本仓库。

## 已验证的前缀 smoke 与恢复监控的完整 replay  2026-10-28

- Smoke [40643563020](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/40643563020) 通过：**20/20 请求成功、0 失败**，prefix-hit tokens 增加 **620,544**。
- 实际 GPU KV 容量为 **22,665,373 tokens**。8 个 worker 均确认 BF16 causal、FP8 cached FMHA、主模型 ASM PS 禁用、同一 forward 内 Q 复用和 native fused FP8 gather。
- Numerical preflight 与 QUERY_REUSE_PASS 均通过。最新 Q1024/KV8192 完整调用计时：BF16 **172.28 µs**，hybrid **111.99 µs（1.538）**；两个 cached chunk 为 **337.45 / 204.27 µs（1.652）**。这些是单 GPU 诊断，不包含 KV 重建和 collective。
- 每个 worker 在线量化 **489 层**，kv_b 权重保留 BF16。模型加载 **439.4440.3 秒**、占用 **184.08 GiB/GPU**，比旧测试的 23.6 分钟更短。
- 请求的 FULL_AND_PIECEWISE 实际降为 **FULL_DECODE_ONLY**，原因是模型没有 compiled submodule 且关闭了 breakable graphs。较早的 graph-sizing capture 占用 **9.209.64 GiB/GPU**；最终 capture 报告 **6.67 GiB/GPU**。这是两个阶段，不能相加为常驻显存。
- Attention block 实际对齐为 **1536 tokens**，Mamba page padding 为 **3.97%**。Planner 还报告新增 18 个 padding layers，相关 cache grouping 最多可能浪费 **26.09%**；这是上界，不是全局显存浪费的实测值。
- 此次受控前缀 smoke 未出现 hipBLASLt recovery warning；完整 AgentX 的形状不同，仍需检查。
- 完整性能 run40646139769 已成功：总吞吐 16,915.72154、输出吞吐 117.84290 tokens/s/GPU，相对 baseline39239112271 提升 9.30% 和 14.35%。这是完整候选配置的结果，不能归因于 FMHA 单项。详见[结果与证据](fp8-fmha-results-2026-10-28_zh.md)。GSM8K run40658068830 已通过：strict-match 为 1282/1319（97.1948%），flexible-extract 为 1281/1319（97.1190%），均超过 90%。没有 FP8 FMHA engagement marker；验证的是 GSM8K 使用 BF16 路径的服务配置，并非 cached FP8 kernel 的模型准确率。
- Full 配置：Moonshot、TP8/DCP8、C48、DSpark k4、warmup10/lane、3600 秒、无 GSM8K。已完成的 setup-job 日志确认 checkout 为 **e4cf47a7650321e4bd40c83746088a2679ec0509**；workflow head 是后续文档提交。
- 完整吞吐、正式测量/warmup 错误及 cache-hit rate 已记录于结果页。GSM8K run40658068830 已通过：strict-match 为 1282/1319（97.1948%），flexible-extract 为 1281/1319（97.1190%），均超过 90%。没有 FP8 FMHA engagement marker；验证的是 GSM8K 使用 BF16 路径的服务配置，并非 cached FP8 kernel 的模型准确率。

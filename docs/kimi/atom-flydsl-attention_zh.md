# vLLM Kimi：BF16 / FlyDSL FP8 混合 prefill

[English](atom-flydsl-attention.md)  [开销审计](kimi-prefill-overhead-audit_zh.md)  [基线](atom-attention-BASELINES.md)

## 插件开关

启动 vLLM 服务前，在服务环境中设置 `ATOM_USE_FLYDSL_FP8_PREFILL_ATTN=1`。插件创建 backend 时读取该开关；修改后需要重启服务。

| 开关 | Prefill 行为 |
|---|---|
| 未设置或 `0` | 所有 prefill attention 使用 BF16，不导入 FlyDSL FP8 API |
| `1` | 最大 query 长度 >= 512 且最大 KV 长度 >= 8192 的缓存块使用 FP8；causal 和较小缓存块使用 BF16 |

只有字符串 `1` 启用 FP8。benchmark 脚本要求明确传入 `0` 或 `1`；可复用 benchmark workflow 设置 `'1'`，MI355X launcher 将其传入容器。仅消融 attention 时，将 workflow 中该环境变量改为 `'0'`。

插件必须在 `PYTHONPATH` 中，并通过 `--attention-backend CUSTOM --attention-config '{"mla_prefill_backend":"CUSTOM"}'` 选择。插件的 `vllm.general_plugins` entry point 注册两个自定义 backend。如果 `VLLM_PLUGINS` 使用允许列表，需要包含 `inferencex_atom_attention`。只在原版 vLLM 中设置开关不会自动加载此扩展。

此改动不需要修改已安装的 vLLM/AITER 源码。权重量化仍由 `--quantization-config` 独立控制；`kv_b_proj` 保持 BF16。关闭开关后，插件的原生通信、KV 展开集成和 metadata 策略仍然保留，因此这是 FMHA 消融，并非完整的原版 vLLM 基线。

CPU 路由检查命令：`python -m unittest discover -s upstream/InferenceX/benchmarks/single_node/agentic/atom_attention/tests -p 'test_*.py'`。测试运行真实插件代码，仅替代外部 GPU/vLLM 依赖，覆盖关闭/未设置、启用阈值、causal/小块回退、缺少 API 和不支持的形状。GPU 数值及模型准确率需要单独的 GPU 运行验证。

- Native vLLM、固定 ROCm image；Moonshot Kimi-K3。
- 使用 ATOM 同样的 AITER FlyDSL FP8 FMHA API，不替换 vLLM 框架。
- 不 patch 安装的 vLLM/AITER source；使用本仓库扩展。
- 当前 smoke40643563020，source SHA e4cf47a7650321e4bd40c83746088a2679ec0509；已验证全部 8 workers 的实际调用、20/20 请求成功和 22,665,373 KV tokens。Full replay [40646139769](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/40646139769) 已完成，详见[完整结果](fp8-fmha-results-2026-10-28_zh.md)。GSM8K run40658068830 已通过：strict-match 为 1282/1319（97.1948%），flexible-extract 为 1281/1319（97.1190%），均超过 90%。没有 FP8 FMHA engagement marker；验证的是 GSM8K 使用 BF16 路径的服务配置，并非 cached FP8 kernel 的模型准确率。

| 步骤 | 当前路径 |
|---|---|
| Loader / scheduler / weight quant / KV storage | Native vLLM + Moonshot + online FP8 targets |
| DCP / latent expansion | BF16 kv_b 权重适配 native fused compact FP8 gather；smoke 已确认全部 8 workers 的 marker |
| Expanded K/V | BF16；没有直接输出 FP8 expanded KV |
| Causal FMHA | BF16 |
| Cached max Q < 512 或 max KV < 8192 | BF16 |
| 更大 cached chunk | QKV quant + FlyDSL FP8 FMHA |
| Q reuse | 仅同一 forward 相同 Q view；finally 清理 |
| 输出 / LSE / merge | BF16 / FP32 / native |
| Main ASM PS scratch | 分配前禁用；draft native PS 仍可能存在 |
| Decode / draft | Native AITER |

- 删除 FP8 kv_b weight override，原始权重成本约36 MiB/GPU。
- TP8 每 rank12 heads；Q/K head192，V head128。
- 完整 cached call diagnostic：Q1024/KV8192 改善1.656；Q256/KV4096 退化0.827。
- QKV quant 约55µs，Q reuse 后 KV quant 仍约52µs。
- Numerical preflight 已通过其误差阈值，不是模型 accuracy 证据。

## 验证与自动继续

- Kernel run40642162788 通过 numerical、strided、empty-KV、Q reuse、changed-input graph 检查。
- 当前 prefix smoke 在模型加载前运行最新 preflight。
- vllm bench serve prefix_repetition：32K/128K prefix、suffix1024、短输出。
- Gate：20/20 请求成功、缓存命中增长、KV capacity 行、所有8 workers 的BF16 causal / FP8 cached / disabled main PS / Q reuse / native fused FP8 gather。
- 完整性能 replay 已完成：warmup10/lane、3600秒、固定通过 smoke 的 SHA。用户后续要求的 GSM8K 已独立提交为 run40658068830。
- 避免重复 dispatch；gate 失败先检查日志并修复。

## 已有结果与限制

- 旧全 FP8 smoke40634885348：KV22,695,620；total11,789.99、output61.13 tokens/sec/GPU。
- 测量阶段476成功/2empty-response errors；warmup85empty-response errors。
- 主基线39239112271 实际为Moonshot：15,476.47 total tokens/sec/GPU。
- 未单独测量communication、expansion GEMM和offload；roofline 暂停。
- 详见[开销审计](kimi-prefill-overhead-audit_zh.md)。

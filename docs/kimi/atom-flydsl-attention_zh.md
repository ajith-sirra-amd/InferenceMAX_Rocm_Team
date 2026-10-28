# vLLM Kimi：BF16 / FlyDSL FP8 混合 prefill

[English](atom-flydsl-attention.md)  [开销审计](kimi-prefill-overhead-audit_zh.md)  [基线](atom-attention-BASELINES.md)

- Native vLLM、固定 ROCm image；Moonshot Kimi-K3。
- 使用 ATOM 同样的 AITER FlyDSL FP8 FMHA API，不替换 vLLM 框架。
- 不 patch 安装的 vLLM/AITER source；使用本仓库扩展。
- 当前 smoke40643563020，source SHA e4cf47a7650321e4bd40c83746088a2679ec0509；已验证全部 8 workers 的实际调用、20/20 请求成功和 22,665,373 KV tokens。Full replay [40646139769](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/40646139769) 已自动启动，结果待完成。

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
- Numerical drift 约45% RMS，不是模型 accuracy 证据。

## 验证与自动继续

- Kernel run40642162788 通过 numerical、strided、empty-KV、Q reuse、changed-input graph 检查。
- 当前 prefix smoke 在模型加载前运行最新 preflight。
- vllm bench serve prefix_repetition：32K/128K prefix、suffix1024、短输出。
- Gate：20/20 请求成功、缓存命中增长、KV capacity 行、所有8 workers 的BF16 causal / FP8 cached / disabled main PS / Q reuse / native fused FP8 gather。
- Controller 每60秒检查；通过后 full agentic replay：warmup10/lane、3600秒、无GSM8K、固定通过 smoke 的 SHA。
- 避免重复 dispatch；gate 失败先检查日志并修复。

## 已有结果与限制

- 旧全 FP8 smoke40634885348：KV22,695,620；total11,789.99、output61.13 tokens/sec/GPU。
- 测量阶段476成功/2empty-response errors；warmup85empty-response errors。
- 主基线39239112271 实际为Moonshot：15,476.47 total tokens/sec/GPU。
- 未单独测量communication、expansion GEMM和offload；roofline 暂停。
- 详见[开销审计](kimi-prefill-overhead-audit_zh.md)。

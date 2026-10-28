# Kimi 混合 FP8 prefill：完整性能结果

[English](fp8-fmha-results-2026-10-28.md) · [插件与开关](atom-flydsl-attention_zh.md)

[Run 40646139769](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/40646139769) 已成功完成。与基线 [39239112271](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/39239112271) 相比，完整候选配置的总吞吐提升 **9.30%**，输出吞吐提升 **14.35%**。

## 实测结果

| 指标 | 基线 | 候选配置 | 变化 |
|---|---:|---:|---:|
| 总 tokens/s/GPU | 15476.46881 | 16915.72154 | +9.30% |
| 输出 tokens/s/GPU | 103.05251 | 117.84290 | +14.35% |
| TTFT p90，秒 | 4.11628 | 3.18346 | -22.66% |
| ITL p90，秒 | 0.04778 | 0.04205 | -11.99% |
| 正式测量成功请求 | 3357 | 3598 | |
| 正式测量错误 | 6 个空响应 | 6 个空响应 + 1 个 ClientOSError | |
| GPU KV 池，tokens | 20951355 | 22665373 | |
| GPU cache hit | 81.165% | 85.205% | |
| CPU/external cache hit | 11.884% | 8.192% | |
| 总 cache hit | 93.050% | 93.396% | |

Warmup 单独统计：基线成功 101 次、空响应错误 430 次；候选配置成功 105 次、空响应错误 426 次。workflow 成功并不表示这些错误不存在。

## 配置与实际路径

服务代码 checkout 为 **`e4cf47a7650321e4bd40c83746088a2679ec0509`**，已由 setup-job 日志确认。workflow head 包含后续文档提交，不能作为服务代码版本。

- 模型 `moonshotai/Kimi-K3`；TP8/DCP8，并发 48，max sequences 96，max batched tokens 8192。
- ROCm image：`vllm/vllm-openai-rocm:nightly-1b2a666391f2a24228a1436b64d547c34cc6b6ce`。
- 原生 DSpark k4；性能 replay 使用 synthetic acceptance 3.36。
- 每 lane warmup 10 次，请求 replay 时长 3600 秒，实际报告时长 3627.79603 秒。
- 全部 8 个 worker PID 均记录 BF16 causal attention、FP8 cached FMHA、BF16 `kv_b_proj` 对应的原生 compact-FP8 gather、同次 forward 的 Q 复用，以及主模型 ASM prefill metadata 禁用。
- 最终 graph capture 为 6.67 GiB/GPU。模型加载占 184.08 GiB/GPU，耗时约 765 秒，不在 benchmark 窗口内。较早的 sizing capture 是独立阶段，不能与最终 capture 相加。
- server 日志没有 traceback、HIP error、EngineDeadError、hipBLASLt 或 NaN 匹配项；这不构成准确率证据。

原始 aggregate 的部分身份字段不完整，包括 `spec_decoding: none`。实际启动命令确认使用 DSpark；保留原始 aggregate，不改写其内容。

## 解释与下一组对照

此结果**不能归因于 FMHA 单项**。候选配置还改变了在线权重量化、metadata 分配和 Q 准备。缓存驻留比例及请求组合也不同。总吞吐包含已缓存输入 tokens，不能视为纯新增 prefill 计算吞吐。按输入长度分组后，ITL p90 约下降 11–13%；TTFT p90 除小于 8192 的组（+1.06%）外均改善。这些是描述性分组，并非完全相同请求调度的配对实验。

插件开关由后续 [2a656fd](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/commit/2a656fdc330761bb076b5c9115307dff4cbadc3a) 加入。5 项 CPU 路由测试通过；此版本尚未执行新的 GPU 服务验证。使用插件不需要原生 vLLM patch。

准确率审查后，建议在同一版本对比 `ATOM_USE_FLYDSL_FP8_PREFILL_ATTN=0` 与 `1`，固定模型、image、权重量化 targets、trace、speculation、拓扑和 warmup，并确认启用组实际触发 cached FP8。这样可以更接近隔离 attention 开关的影响；仍需报告 replay 和 cache 差异。

## 准确率后续与证据

[GSM8K run 40658068830](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/40658068830) 在 GPU 性能任务结束后立即提交，仍固定到 `e4cf47a`。使用完整 GSM8K、five-shot、temperature 0、真实 block rejection，无评测数量限制。**运行已成功完成：strict-match 97.1948%（1282/1319），flexible-extract 97.1190%（1281/1319），两项 validator 检查均通过。**仓库 Kimi validator 要求两个 exact-match 提取指标都至少达到 0.90。

**FP8 FMHA 与 Q-reuse marker 均为零。**全部 8 个 worker 记录 BF16 causal 和 cached attention。样本文件共 2638 行，对应相同 1319 个问题的两种提取过滤器。这证明已测服务配置通过 GSM8K，不能证明 cached FP8 kernel 的模型准确率，也未验证后续开关或原生移植版本。详见[准确率证据](../../results/atom-attention/run40658068830/accuracy_proof.json)及[lm-eval 结果](../../results/atom-attention/run40658068830/lm_eval_results.json)。

证据：[原始 aggregate](../../results/atom-attention/run40646139769/benchmark_summary.json)、[runtime 与命令来源](../../results/atom-attention/run40646139769/runtime_evidence.json)、[基线比较](../../results/atom-attention/run40646139769/baseline_comparison.json)、[workload/错误明细](../../results/atom-attention/run40646139769/workload_evidence.json)、[GSM8K dispatch](../../results/atom-attention/run40658068830/dispatch.json)。原始 artifacts：raw `12802705904`、benchmark `12802277146`、server `12802601010`。

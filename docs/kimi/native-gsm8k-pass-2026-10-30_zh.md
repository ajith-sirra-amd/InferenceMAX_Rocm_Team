# 原生 FP8 验证已通过 - 2026-10-30

[English](./native-gsm8k-pass-2026-10-30.md)

PR 版本：`492080779e9feea19308092b0aba0840e87c6115`。

[完整 GSM8K 运行 41064573708](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/41064573708) 已成功完成。

- 严格匹配：**1278/1319（96.8916%）**。
- 灵活提取：**1277/1319（96.8158%）**。
- 完整 1319 题、five-shot、temperature=0、C48、TP8/DCP8、真实 MTP block 验证。
- 八个独立 worker 均在评估期间执行原生 FP8 上投影和缓存上下文 FMHA。
- 保留 `VLLM_GPU_SYNC_CHECK=error`。服务器日志没有运行时错误、traceback、ERROR 行或磁盘配额错误。
- 安装源码哈希与[静态检查 41065468482](https://github.com/ajith-sirra-amd/vllm/actions/runs/41065468482) 一致，静态检查已通过。

[归档证据](../../results/atom-attention/run41064573708/gsm8k-evidence.json) 包含源码版本、构件编号、指标及执行路径。本文替代此前任务笔记中“原生 GSM8K 待完成”的状态。

两处元数据传输现在使用 `async_tensor_h2d`，先将整数元数据放入固定 CPU 内存。该修复不改变 FP8 量化或运算。

本次使用 eager 模式及 512-token 精度压力路由，新 token 注意力保持 BF16。正常图模式的原生性能、配对比较及历史绝对 LSE <0.20 门限仍待完成。PR #64288 保持草稿。尚无同版本原生 BF16 对照，不能把历史原型分数差异直接归因于 FP8。

评估后的可选 lm-eval Git 元数据收集提示 `/outputs` 所有权不匹配，但未导致评估失败；原生源码来源已独立核验。这是单独的验证脚本问题。

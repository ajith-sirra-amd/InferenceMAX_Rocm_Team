# 原生 FP8 FMHA 一致性验证草案

[English](README.md)

[补丁](native-fp8-fmha-parity.patch) 将现有插件移入 vLLM 原生模块，基于已完成性能与 GSM8K 运行所使用的 vLLM 提交 **`1b2a666391f2a24228a1436b64d547c34cc6b6ce`**。尚未应用到运行中的服务，也未宣称原生 GPU 或端到端验证通过。

参考插件版本为 `2a656fdc330761bb076b5c9115307dff4cbadc3a`。开启开关后的计算对应此前 `e4cf47a` 实验路径。原生 backend 仅替换导入路径、类名和 backend 名称；量化代码完整复制，保留 launch 参数、reciprocal 指令、strided 输入支持及当前未使用的 KV-only helper，并附 MIT 许可。这是为了先验证一致性，后续提交审查还需消除重复代码。

## 保留的行为

- 仅 `ATOM_USE_FLYDSL_FP8_PREFILL_ATTN=1` 启用 cached FP8 FMHA。
- Causal 和较小缓存块使用 BF16；FP8 要求 max Q >= 512 且 max KV >= 8192。
- 相同 Q view 仅在一次 forward 内复用，并在 `finally` 中清理。
- 保留原生 DCP gather/reconstruction、BF16 expanded K/V 和原生 decode。
- 为选中的 backend 禁用主模型 ASM prefill metadata。
- 不引入权重量化配置改动。
- 两个原生 backend 替代插件的两个 CUSTOM 注册；必须同时选择，单独设置开关不会更改默认 backend。

## 应用与验证

在固定提交的干净 vLLM checkout 及匹配的 ROCm 环境中操作。补丁路径相对 vLLM 仓库根目录。具体 `git apply --check`、`git apply` 和 GPU 比较命令见同目录英文页；两种语言使用相同命令。

`utils/atom_attention/compare_native_plugin.py` 使用真实 vLLM prefill metadata，在同一组张量上比较插件和原生实现，覆盖 8 组开关/形状组合、strided Q/V、causal 输出、两个缓存块，以及跨两次 forward 复用存储但改变 Q 的情况。要求 FP8 字节、descale、BF16 输出和 FP32 LSE 完全一致，并核对 Q 量化/复用次数。**脚本尚未运行。**

该比较不覆盖原生 backend 选择、DCP collective、graph capture、完整模型加载及服务 forward。宣称等价前，仍需原生受控前缀 smoke、graph 验证、模型评测及 flag-off/on 性能对照。

原生服务选择：

```bash
export ATOM_USE_FLYDSL_FP8_PREFILL_ATTN=1
vllm serve "$MODEL" \
    --attention-backend ROCM_AITER_FLYDSL_MLA \
    --attention-config '{"mla_prefill_backend":"ROCM_AITER_FLYDSL_FP8"}'
```

比较时保持原模型、TP8/DCP8、权重量化、speculation 及其他参数，仅用以上两个 backend 名称替换插件的两个 CUSTOM 选项。

## 上游状态

尚未创建 vLLM PR。此补丁是受控的一致性验证草案，并非可直接提交的上游变更。[#59124](https://github.com/vllm-project/vllm/pull/59124) 和 [#62447](https://github.com/vllm-project/vllm/pull/62447) 已提供原生 FlyDSL FP8 prefill。当前实验的 causal/小块 BF16 策略及量化实现存在差异，应围绕这些差异缩小最终贡献范围。

[vLLM 规则](https://github.com/vllm-project/vllm/blob/1cb31be7028457c7eae7b6fabca1941d5d3c4730/AGENTS.md) 禁止重复解决已有开放 PR 覆盖的问题，并要求提交人逐行审查、运行相关测试。[PR checklist](https://github.com/vllm-project/vllm/blob/1cb31be7028457c7eae7b6fabca1941d5d3c4730/.agents/skills/pr-checklist/SKILL.md) 要求行为测试、可见文档、验证证据及人工审查，draft PR 也不例外。

审查中仍需处理：合并重复 runtime 代码、适配原生格式和诊断日志、补齐真实服务回归测试与 ROCm CI 选择。尚未运行原生 GPU 测试、pre-commit 或 `git apply --check`；本地 Windows Bash 及嵌套文件写入遭拒。补丁构造检查不能替代上述验证。

已有[性能与 GSM8K 证据](../../docs/kimi/fp8-fmha-results-2026-10-28_zh.md)属于插件版本。GSM8K 为 97.19%/97.12%，但未触发 FP8 FMHA。

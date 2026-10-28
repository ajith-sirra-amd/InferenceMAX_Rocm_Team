# 因果 FP8 注意力实验

[English](./causal-fp8-attention.md) | **中文**

基线为已完成插件运行 40695099743：输入加输出吞吐量为
17406.08912 tokens/s/GPU。本实验与原生 vLLM PR #64288 分离。

参考 [vLLM #59124](https://github.com/vllm-project/vllm/pull/59124)，
已审查提交 `132f6daa6b17cb4000ef58355943bb39339104f2`。
使用 FlyDSL `causal=True`、相同 Q/KV 序列偏移、`cross_seqlen=True`
及独立的原始 Q/K/V descale。现有量化器不变。

`ATOM_USE_FLYDSL_FP8_CAUSAL_ATTN=1` 仅在当前 forward 已准备直接 FP8 缓存 KV 时
启用因果 FP8，复用这些 Q/K/V tensor，不增加量化调用。
因果 K/V 原始 descale 与缓存投影的 2 倍余量输出 descale 分开使用。
未准备直接 KV 的 forward 保留 BF16 因果注意力。
缓存阈值、Q 切片量化、权重量化、DCP、NUMA/CCD 绑定、投机参数和模型均保持原状。

15 项 CPU 路由及证据测试已通过。GPU 诊断覆盖非连续输入、变长因果序列、
独立 FP64 参考（BF16 及相同 FP8 输入）、固定 scale 下未来 token 屏蔽、
预分配输出、LSE、一次准备及输入变化后的图回放。
RMS <0.10、绝对 LSE <0.20 的诊断门限不变；
受控输入通过不代表之前捕获的 BF16 数值失败已解决。

完整 GSM8K 保持固定任务及评分，要求同一组八个 worker 在评测阶段
实际执行缓存 FP8 投影和因果 FP8。独立因果标记拒绝缺失或评测前证据。
诊断及 GSM8K 成功后再运行完整 3600 秒性能测试；不做服务冒烟测试，
准备阶段不宣称新的性能或准确率结果。

## GPU 诊断通过；完整运行已排队

源码：`bb8bed732f514fd697856c5abd1d2bdd9d56a1e5`。
[GPU 诊断](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/40712781470)通过四组因果输入、
固定 scale 的未来 token 屏蔽、预分配输出、单次量化及输入变化后的图回放。
受控 BF16 参考最大 RMS 为 0.04442、LSE 误差为 0.10964；
相同 FP8 参考最大 RMS 为 0.01858、LSE 误差为 0.00005395。
这些结果不代表之前真实模型捕获的 BF16 数值失败已解决。

[完整 GSM8K](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/40712931283)以诊断成功为前置条件。
[完整性能](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/40713116775)要求 GSM8K 及因果调用证明成功。
两者执行上述固定源码。检查器和证据收集器每 60 秒轮询。
[提交及诊断证据](../../results/atom-attention/causal-fp8-20261028.json)。

## 按用户要求取消完整性能运行

用户要求：**不要提交完整性能运行**。
已排队的运行 40713116775 在 GPU benchmark 启动前取消，agentic benchmark 作业已跳过。
仅继续 GSM8K 运行 40712931283。没有新的用户指示，不得提交或重新启动完整性能运行。
准确率检查器和产物收集器继续工作。

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

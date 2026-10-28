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

## GSM8K 已完成；用户重新授权完整性能测试

用户之后明确要求派发完整测试，因此上述取消指令不再阻止新测试。
原测试保持取消状态；新的[完整性能测试 40719654583](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/40719654583)
使用与准确率测试相同的执行代码 `bb8bed732f514fd697856c5abd1d2bdd9d56a1e5`。
工作流版本为 `a6ac3e9550b5df48ddd52238d91165952a5a8580`；其间仅修改了文档和证据记录。
测试为标准 3,600 秒 AgentX 回放，并发 48，八张 MI355X，TP8/DCP8。
对比基线为 17,406.08912 输入加输出 tokens/s/GPU，同时检查请求错误和实际 FP8 执行。
状态检查器和产物收集器每 60 秒检查一次。

GSM8K 测试 40712931283 已成功完成全部 1,319 道题，five-shot，temperature 0：

| 评分 | 正确数 | 准确率 | 相对仅缓存上下文 FP8 原型 |
| --- | ---: | ---: | ---: |
| Strict | 1,279 / 1,319 | 96.96740% | 少 2 题 |
| Flexible | 1,278 / 1,319 | 96.89158% | 少 3 题 |

已直接检查产物 `12828229398`（结果和样本）及 `12828099867`（服务器日志）。
原验证器使用 `require_causal=True`，确认同一组八个 worker 在评估阶段均执行了缓存 FP8 投影/FMHA 和因果 FP8 注意力。
这是 eager/512-token 准确率路由下的插件证据，并非原生 PR 的 GPU 验证，也未解决已捕获 BF16 参考的 LSE 门槛问题。

性能结果出来后，只有测得性能提升才将新 token FP8 加入原生 PR #64288。
随后对最终原生代码版本完成 GPU kernel 数值、graph replay、优化路径 GSM8K、同步检查、服务性能及必需 CI 验证。
评审者另要求分别比较 #59124 和 #62447 各自单独运行、以及加入直接 FP8 上投影后的性能，每组保持权重和运行设置一致。
这些对照测试仍待执行，与测试 40719654583 分开。
用实际原生验证证据更新 PR，全部必需检查通过后再标记为可评审。
新增环境变量的讨论保持开放，稍后再决定。

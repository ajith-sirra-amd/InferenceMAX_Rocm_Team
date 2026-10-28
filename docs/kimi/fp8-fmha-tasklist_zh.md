# Kimi FP8 任务清单

[English](./fp8-fmha-tasklist.md) | **中文**

## 已完成的证据

- [x] 已确认吞吐量：基线 **15,476.46881**、BF16 投影加缓存 FP8 FMHA **16,915.72154**、直接 FP8 投影加缓存 FP8 FMHA **17,406.08912** 输入加输出 tokens/s/GPU。
- [x] 仅缓存上下文 FP8 原型 GSM8K：**1,281/1,319（97.119%）**，已验证八个 worker 实际执行优化路径。
- [x] 新 token 因果 FP8 受控 GPU 诊断通过四组用例、因果 mask 和更改输入后的 graph replay。
- [x] 新 token FP8 原型 GSM8K [40712931283](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/40712931283)：strict **1,279/1,319**，flexible **1,278/1,319**；同一组八个 worker 均实际执行缓存和因果 FP8。
- [x] PR 描述已有三阶段性能表及用户指定的准确基线 job 链接。

以上为原型结果。最终原生 PR 的 GPU 验证仍待完成；GSM8K 通过并未解决历史捕获 BF16 参考的 LSE 差异。

## 后续任务顺序

1. [ ] **当前完整性能测试：** 跟进 [40719654583](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/40719654583)，执行代码 `bb8bed732f514fd697856c5abd1d2bdd9d56a1e5`，C48 + MTP。对比 **17,406.08912** 基线，检查吞吐、延迟、测量阶段请求错误和实际缓存/因果 FP8 调用。不要重复派发。
2. [ ] **有条件地加入原生 PR：** 只有新 token 测得性能提升才加入 PR #64288；否则验证已确立的缓存上下文实现。候选代码位于 fork 的 `checks/fp8-mla-causal-20261028` 分支，PR 分支尚未改变。
3. [ ] **原生 GPU 正确性：** 验证最终实现的投影/FMHA 数值、因果 mask（若包含）、非单位 scale、真实 FP8 执行、更改输入后的 graph replay，以及服务运行中的 `VLLM_GPU_SYNC_CHECK=error`。不得仅为通过测试而放宽捕获 BF16 参考的 LSE 门槛。
4. [ ] **原生完整 GSM8K：** 1,319 题，five-shot，temperature 0，真实 speculative verification。记录最终原生 commit，并证明评估阶段八个 worker 执行预期的缓存/因果路径。
5. [ ] **原生完整性能：** 最终原生代码以 C48 + MTP 运行既定工作负载，报告吞吐、TTFT、TPOT 和测量阶段请求错误，与插件原型结果分开。
6. [ ] **评审要求的对照：** 比较 #59124 单独运行及加直接 FP8 上投影后，再比较 #62447 的同类组合。每组使用共同 vLLM 基础版本、相同权重和运行设置，并检查实际执行；简单叠加 patch 不是执行证据。
7. [ ] **完成 PR：** 完成必需静态/CI 检查，用实际原生准确率、性能和 GPU 证据更新标题、描述及简短评审回复。必需检查全部通过后才标记可评审；环境变量讨论保持开放，稍后决定。
8. [ ] **稳态 profiling：** 验证任务完成后，对最终优化代码单独运行 agentic profiling。确认吞吐和 cache 稳定，排除启动和预热；短时间采集八个 worker 的 trace，分析主要 kernel、内存流量、通信、CPU 发射开销和 GPU 空闲间隙。带 profiling 的吞吐不与普通基准混为同一结果。
9. [ ] **profiling 后并发 sweep：** 完整测试 **C56 + MTP**、**C64 + MTP**、**C96 不启用 MTP**。固定代码、模型、权重、TP8/DCP8、CCD 绑定、offload、graph、数据集和时长。C96 必须移除 speculative 配置及模拟接受机制，并核实实际服务参数。
10. [ ] **确定下一项优化：** 与 C48 + MTP 比较总吞吐、输出吞吐、TTFT、TPOT、请求错误及 cache/GPU 利用率。结合稳态 profile 排序后续机会，提出具体下一项实验。

11. [ ] **NUMA/CCD 绑定上游贡献评估：** 检查 Kimi 启动脚本及 vLLM 现有 CPU affinity 支持，确认可移植的改动、拓扑和线程生命周期缺口、重复 PR 及独立贡献所需证据。此项是评估，尚未提交绑定 PR。

## 原生评估的实际执行证据

当前 recipe 通过 `PYTHONPATH` 加载原型插件并选择 `CUSTOM` attention。原生验证必须安装并导入 fork 的精确版本、选择原生 backend，并排除原型插件。原生缓存路径目前要求 query 长度 >=512 且 context 长度 >=8192；普通 GSM8K prompt 可能只执行 BF16 fallback。若使用仅用于测试的 routing override，必须明确披露，并与生产路由的性能证据分开。固定 GSM8K 任务和评分保持不变，必须证明评估阶段八个 worker 均成功调用优化路径。

## 执行约束

- 保持当前 turn 持续工作，每 **五分钟**检查运行状态。独立 poller 只收集状态，不能唤醒已结束的模型 turn。未启用 Codex 唤醒触发器或系统调度器。
- 写入范围仅限 **InferenceMAX_Rocm_Team** 和 **ajith-sirra-amd/vllm**。不修改 Codex 状态文件，不安装系统调度器。
- 不运行 serving smoke；保持固定 GSM8K 任务、评分和测试门槛。
- 使用精确代码版本和实际执行证据，不把插件结果当作原生 PR 验证。
- 保存证据，同步中英文文档；基准行为变化仅追加到性能变更日志。

## 原生 GPU 验证工具

e2e 工作流在 `runner-diagnostics-only=true` 时可接受精确 fork commit 参数 `native-vllm-sha`。
[验证脚本](../../utils/atom_attention/native_vllm_gpu_validate.py) 检出指定版本，使用仓库内的 `.venv` 和对应基础版本的预编译库，执行原生 `flydsl_fp8` kernel 测试。
它记录源码哈希、软件/GPU 信息和 JUnit 结果；选中的测试缺失或跳过均判为验证失败。
这只是 kernel 验证。模型 GSM8K、服务同步检查、完整性能及历史捕获 BF16 门槛仍需分别验证。
等待当前完整性能结果并选定最终原生版本后再派发；当前性能运行期间未启动原生 GPU 测试。

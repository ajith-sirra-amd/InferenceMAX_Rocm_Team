# Kimi FP8 任务清单

[English](./fp8-fmha-tasklist.md) | **中文**

## 下一次调度：2026-10-29 用户决定

用户确认更换后的服务器配置相同，并明确不需要额外的服务器/对照验证运行。
不要为此调度已准备的 `checks/kimi-causal-fp8-control-20261029` 分支。
将当前完整 causal-FP8 运行与已建立的 **17,406.08912 total tokens/s/GPU**
参考结果比较，同时报告延迟和错误。检查结果后选择原生候选实现，继续原生
GPU 数值检查、完整 GSM8K 和原生性能测试。与其他 FP8 PR 的配对比较仍是
独立的后续任务。

新的 GPU 作业已进入 `Launch job script`，尚无吞吐量结果。

## 2026-10-29 当前运行

原重试作业 `123215353998` 未发布任何 setup 步骤。操作者确认没有 Docker
容器或 GPU 活动，随后 runner 断开连接。取消于 02:06:10 UTC 完成。
这属于 runner 启动失败，不是已完成的性能测量。

新的完整运行 [40804432321](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/40804432321)
使用工作流版本 `75a52838ee8fa89461dc812913eb71b370b839bd`，服务源码仍为
`bb8bed732f514fd697856c5abd1d2bdd9d56a1e5`：C48 + MTP、TP8/DCP8、
原型固定镜像和 3600 秒回放。前置验证及配置生成已通过。
在 2026-10-29T02:09:49.605677+00:00，GPU 作业状态为 **queued**，runner 为 **offline**。
尚无新 token FP8 吞吐量结果。已启动每 60 秒检查一次的前台 checker，
该进程不能唤醒已结束的模型回合。

此前 [健康检查 40802919527](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/40802919527)
在 `pit2-p03-g15` 上通过：8 张 MI355X 空闲、固定镜像存在、文件系统约 4.4 TiB
可用空间、工作区为空。Runner 名称仍以 `g07` 结尾。
既有 17,406.08912 结果的日志也标识同一实际主机。恢复流程未恢复或删除旧性能文件。

原生安装修正在 `75a52838ee8fa89461dc812913eb71b370b839bd` 通过
[CPU CI 40804336409](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/40804336409)：
**18 项测试，零失败、错误或跳过**，launcher 和 Bash 检查也通过。
Artifact 为 `12867865400`。原生 GPU 验证仍待执行。

## 已完成的证据

- [x] 已确认吞吐量：基线 **15,476.46881**、BF16 投影加缓存 FP8 FMHA **16,915.72154**、直接 FP8 投影加缓存 FP8 FMHA **17,406.08912** 输入加输出 tokens/s/GPU。
- [x] 仅缓存上下文 FP8 原型 GSM8K：**1,281/1,319（97.119%）**，已验证八个 worker 实际执行优化路径。
- [x] 新 token 因果 FP8 受控 GPU 诊断通过四组用例、因果 mask 和更改输入后的 graph replay。
- [x] 新 token FP8 原型 GSM8K [40712931283](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/40712931283)：strict **1,279/1,319**，flexible **1,278/1,319**；同一组八个 worker 均实际执行缓存和因果 FP8。
- [x] PR 描述已有三阶段性能表及用户指定的准确基线 job 链接。

以上为原型结果。最终原生 PR 的 GPU 验证仍待完成；GSM8K 通过并未解决历史捕获 BF16 参考的 LSE 差异。

## 后续任务顺序

1. [ ] **恢复并完成新 token 性能测量：** [40719654583](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/40719654583) 的 GPU job 122927005395 于 16:50:55 UTC 因 runner 磁盘配额错误失败。未上传基准 artifact，尚无新 token 性能提升证据。[恢复任务 40730725939](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/40730725939) 在执行任何步骤前因 runner 与 GitHub 失联而失败。未删除文件，也未恢复结果。GPU 工作需要恢复 runner 存储/连接，或提供已请求的 SSH hostname/user。恢复证据后再决定仓库范围内清理或完整重跑。目标对照仍为代码 `bb8bed732f514fd697856c5abd1d2bdd9d56a1e5`、C48 + MTP，对比 **17,406.08912**。
2. [ ] **有条件地加入原生 PR：** 只有新 token 测得性能提升才加入 PR #64288；否则验证已确立的缓存上下文实现。候选代码位于 fork 的 `checks/fp8-mla-causal-20261028` 分支，PR 分支尚未改变。
3. [ ] **原生 GPU 正确性：** 验证最终实现的投影/FMHA 数值、因果 mask（若包含）、非单位 scale、真实 FP8 执行、更改输入后的 graph replay，以及服务运行中的 `VLLM_GPU_SYNC_CHECK=error`。不得仅为通过测试而放宽捕获 BF16 参考的 LSE 门槛。
4. [ ] **原生完整 GSM8K：** 1,319 题，five-shot，temperature 0，真实 speculative verification。记录最终原生 commit，并证明评估阶段八个 worker 执行预期的缓存/因果路径。
5. [ ] **原生完整性能：** 最终原生代码以 C48 + MTP 运行既定工作负载，报告吞吐、TTFT、TPOT 和测量阶段请求错误，与插件原型结果分开。
6. [ ] **评审要求的对照：** 比较 #59124 单独运行及加直接 FP8 上投影后，再比较 #62447 的同类组合。每组使用共同 vLLM 基础版本、相同权重和运行设置，并检查实际执行；简单叠加 patch 不是执行证据。
7. [ ] **完成 PR：** 完成必需静态/CI 检查，用实际原生准确率、性能和 GPU 证据更新标题、描述及简短评审回复。必需检查全部通过后才标记可评审；环境变量讨论保持开放，稍后决定。
8. [ ] **稳态 profiling：** 验证任务完成后，对最终优化代码单独运行 agentic profiling。确认吞吐和 cache 稳定，排除启动和预热；短时间采集八个 worker 的 trace，分析主要 kernel、内存流量、通信、CPU 发射开销和 GPU 空闲间隙。带 profiling 的吞吐不与普通基准混为同一结果。
9. [ ] **profiling 后并发 sweep：** 完整测试 **C56 + MTP**、**C64 + MTP**、**C96 不启用 MTP**。固定代码、模型、权重、TP8/DCP8、CCD 绑定、offload、graph、数据集和时长。C96 必须移除 speculative 配置及模拟接受机制，并核实实际服务参数。
10. [ ] **确定下一项优化：** 与 C48 + MTP 比较总吞吐、输出吞吐、TTFT、TPOT、请求错误及 cache/GPU 利用率。结合稳态 profile 排序后续机会，提出具体下一项实验。

11. [x] **NUMA/CCD 绑定上游贡献评估：** [源码评估已完成](./numa-ccd-upstream-assessment_zh.md)。现有原生按 GPU 配置的 CPU 列表可表达该选择；独立 AMD CCD 自动策略仍需拓扑处理和匹配测量。未提交新的绑定 PR，也不声称性能提升。

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

原生验证脚本为精确 upstream merge-base 解析明确的 wheel URL，并要求 ROCm、Python ABI、Torch 和 Triton 元数据匹配。构建缺失或不兼容时停止验证，不回退到其他 wheel。解析器已检查正常接受、Torch 不匹配、Python ABI 不匹配及构建缺失拒绝；这些只是环境准备检查，不是 GPU 数值结果。

## 原生服务准备

验证脚本新增独立 `--prepare-only` 模式，安装精确原生版本并准备仓库内环境；结果标记为 `prepared`，绝不作为 GPU 测试通过。独立 `native_fp8_validation` observer 校验原生源码哈希，并在明确标记的阶段按 rank 和进程记录成功的 FP8 投影、缓存及因果 dispatch，不替换原生数值 kernel。显式启用 accuracy-stress 时，仅将缓存 eligibility 改为 query/context 长度大于零；eager 评估及 512-token scheduler chunk 仍由调用方配置。observer 记录该 override，且拒绝将其当作正常 replay 证据。

recipe 现已接入 e2e 工作流的精确 `native-vllm-sha`，选择原生 ROCM_AITER_MLA / ROCM_AITER_FA backend，并保存安装来源和调用证据。尚未执行 GPU 观测及原生模型评估，派发前必须通过 CPU 集成检查。当前调用方与现有 PR 一致，仅要求缓存 FP8 证据（`INFERENCEX_NATIVE_REQUIRE_CAUSAL=0`）；若因果 FP8 测得提升并加入 PR，必须显式改为 1。CPU 测试拒绝不完整、BF16、仅预热、版本不符、worker 重启、仅 graph capture 以及把 stress 当 replay 的证据。`replay` 标记覆盖所启动的 replay（包含预热），不能证明稳态 kernel 驻留。

原生环境/执行证据 CPU CI [40729902162](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/40729902162) 在代码 `0e40f469f1623c09336a034fa3d130b3b294fa4c` 通过 **15 项测试，零失败/错误/跳过**；JUnit artifact 为 `12834722522`。这不是 GPU 数值验证。早先 setup 阶段失败来自错误的 artifact action pin；两个新增工作流均已改用仓库中经验证的现有 pin。

原生服务集成 CI [40732677087](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/40732677087) 在 `bf9b27bd72d9a46814721c6d33811991a142afc8` 通过：15 项原生环境/证据测试、recipe/launcher Bash 语法及现有容器启动器验证。未派发原生 GPU 测试或额外完整基准。PR #64288 仍为 draft，代码为 `08e5d81a0d4fc76c20a97f9a517195db5a5dc5be`。

## 原生依赖修正（2026-10-29）

安装依赖的完整版本以发布的 wheel 文件名为准。当前原生基线的 Torch 元数据为
`2.13.0`，但 wheel 版本为 `2.13.0+rocm10.0.0`；直接追加
`rocm100` 会错误拒绝匹配环境。解析器现从 wheel 文件名获取版本，
仍拒绝 Torch、Triton、ABI 或 ROCm 不匹配。

仅原生诊断和原生服务使用固定 ROCm 10 镜像
`nightly-rocm100-edad8b03c5285a1c12e7e4f83c1af709b319a01b`，
digest 为 `sha256:009169c114af5b15e25eddc808afb3c1fc3f12c94e746472b16f39fcf39cd716`。
发布的 Torch 2.13.0 和 Triton 3.8.0+git669b31ac wheel 与原生基线依赖匹配。
实际容器及 GPU 兼容性仍待验证。原型性能测试保留原矩阵镜像（Torch 2.12.0）。
原生源码仍固定到完整提交，并使用自身 merge-base 的预编译 wheel。

## 2026-10-29 请求增加绑定方式测试

在 C56 + MTP、C64 + MTP 和 C96 不使用 MTP 的测试中，增加原生 vLLM NUMA
绑定并关闭脚本 CCD pinning 的方案（`PIN_CCD=0`，由调用方明确传递）。
当前 FP8 性能运行及原生正确性验证配置保持原状。绑定方式在相同并发下比较，
不能将并发变化的效果归因于绑定。

已合并的 vLLM #61258 通过 AMD SMI 支持 ROCm GPU 到 NUMA 节点的自动检测；
其报告的 GPU serving 验证未覆盖 AMD。使用最终验证后的原生源码，并确认
`numactl`、启动条件、worker CPU affinity 和内存策略。原生自动模式使用
GPU 本地内存软偏好；显式 `--numa-bind` 使用严格内存绑定，属于另一种策略。
Intel PCT 核心选择并非 AMD CCD 选择。ROCm 打包 PR #55433 仍未合并，
不能将实际未绑定的运行计为该实验。


# PR #64288 评审跟进 - 2026-11-02

[English](pr64288-review-follow-through-2026-11-02.md)

## 当前实现与验证

PR 版本保持为 `492080779e9feea19308092b0aba0840e87c6115`。直接使用 FP8 缓存和原生逐通道 FP8 权重完成 K/V 上投影，再进入缓存上下文 FP8 FMHA；新 token 注意力仍为 BF16。目标为 gfx950、DCP > 1、MLA 维度 512/128/64/128。额外环境变量的讨论保持开放。

- 当前版本 GPU 测试：运行 41401317715，8 项通过，没有失败或跳过。
- 完整 GSM8K：运行 41426504427，严格匹配 1275/1319（96.6641%），宽松匹配 1274/1319（96.5883%）。使用真实 MTP、eager、512-token 分块和显式准确率压力路由；八个 worker 均执行直接 FP8 上投影和缓存 FMHA。
- 静态检查与文档：运行 41442476770 通过 Ruff、格式、适用的 pre-commit/mypy、源文件一致性检查和严格 MkDocs 构建，并检查了渲染后的使用说明。
- 上游 CI 在执行前受贡献者授权门禁阻止；独立 fork 检查不能替代上游 CI。不请求授权标签，也不绕过门禁。
- 历史 BF16 绝对 LSE 最大差值约 0.935，超过调查用的 0.20 阈值；该阈值并非上游强制验收规则。GSM8K 通过不能消除此差异。失败的旋转/缩放实验不在 PR 中。
- 18,244.04135 tokens/s/GPU 来自较早的 edad8b03 镜像和运行 41413372088，并非 e0d00e4b。该运行有 11/3834 个测量请求错误，另有 8 个截止时取消的请求；不能声称已获得隔离变量的原生性能提升。

## 对照测试及用户限制

只在用户 fork 的基准测试分支集成参考实现。不得把直接 FP8 上投影代码推入 #59124 或其作者分支。

参考 PR 为 132f6daa6b17cb4000ef58355943bb39339104f2；集成版本为 2eb0ada53e6d0fcd658010d9869957026a229adb；执行脚本版本为 190f103aad8911bff3a31a2ee0cbc12ed9070ce9。运行 41443627369 的 GPU 集成测试 26 项通过，0 失败、0 跳过，11 项无关测试未选；CPU 观察器/安装器测试 21 项通过。

- 参考运行 41443951889 已失败。八个 rank 的 DCP KV all-gather（序号 232455）及 logits all-gather（序号 2668）出现 600 秒超时，随后引擎退出。服务器日志未发现更早的 BLAS 异常；根因尚未确定。
- 热身期已有 434/531 个请求错误；测量期仅 63 条记录，其中 52 条错误。记录时长约 1440 秒，未完成要求的 3600 秒。因此该运行不提供有效吞吐对照。
- 八个 worker 均记录参考 FP8 causal/cached FMHA 和 BF16 展开 K/V，没有直接 FP8 上投影事件。
- #59124 加直接上投影的组合运行 41444880140 已在 GPU 作业开始前取消，未经新的用户授权不得恢复。
- 原生 PR 性能运行 41447686785 正在执行，使用精确 PR 版本 4920807、新镜像 e0d00e4b、C48/TP8/DCP8、合成 MTP、正常路由及图执行。
- 完整 GSM8K 运行 41447950940 等待性能运行成功。工作流版本 94f5efa4a61aae4f4a09d54e2c30d9b4dd56216d，执行版本仍为 190f103；等待上限 240 分钟。

两个 PR 的新 token 路由、Q 复用和缩放策略不同，即使均成功也不能把差异全部归因于上投影。

## 后续工作

持续跟进原生性能与 GSM8K，检查源版本、运行覆盖率、请求与服务器错误、八个 worker 的优化路径证明，并及时更新 PR。完成用户要求的验证后再标记为 ready for review；上游 CI 和人工评审仍是合并要求。

已验证的 BLAS 配置为 TORCH_BLAS_PREFER_HIPBLASLT=0 与 TORCH_BLAS_PREFER_CUBLASLT=0 的组合；尚未分别隔离测试，也未证明该配置本身带来性能提升。它属于运行配置，不属于 FMHA PR 的修复。

后续待办为稳态 profiling、C56/C64 加 MTP 和 C96 不加 MTP。原生 NUMA 改动暂缓。Kimi-K3.1 尚未验证，需确切 checkpoint/config。当前排队版本不加入“只写 Q8、消除未使用新 token K8/V8 缓冲区”的后续优化。

热身期对照：成功的原生运行 41413372088 同样有 437/531 个空内容热身错误（参考运行为 434/531）。脚本包含单 token 热身请求，该计数不能单独证明量化数值错误。测量阶段失败和 collective 超时仍是需要调查的问题。

## 新镜像性能完成；按用户要求停止参考 PR 工作

原生运行 41447686785 已成功：**18,276.23099 total tokens/s/GPU**。使用 PR 版本 492080779e9feea19308092b0aba0840e87c6115 和镜像 e0d00e4b，C48/TP8/DCP8、合成 MTP k=4、正常路由和图执行，完整 3600 秒测量回放。TTFT/ITL 覆盖率均为 100%，八个 worker 均执行直接 FP8 上投影及缓存 FMHA，新 token 注意力仍为 BF16。

测量期共 3840 条记录，3834 条成功，6 条错误（5 条空内容响应、1 条请求体写入错误）；截止及宽限期结束时另有 9 个请求取消。未检测到 BLAS 警告或服务器运行时错误。相比旧镜像的 18,244.04135，单次结果高 0.1764%；不能据此证明镜像升级或 BLAS 配置带来的性能提升。

GSM8K 运行 41447950940 的第 1 次尝试在 GPU 调度前因 GitHub API HTTP 502 失败，未执行准确率测试。性能运行成功后已重启失败作业，第 2 次尝试的前置条件检查已通过；继续跟进完整准确率结果、精确源版本、真实 MTP 和八个 worker 的执行证明。

用户明确要求：**不修复 #59124（包括参考基准集成），不再次运行，只报告已观察到的失败。** 组合运行保持取消。本 PR 直接把 FP8 latent cache 与 FP8 权重上投影为 FP8 K/V，消除展开的 BF16 K/V 缓冲区及独立的缓存 K/V 量化步骤。#59124 的新 token 注意力为 FP8，本 PR 保持 BF16。失败的参考运行不能作为性能提升的基线。

PR 描述和已有评审状态评论已更新，源代码未改变；所要求的后续准确率测试完成前仍保持 draft。

## CI 复查与性能分析准备

当前 PR head 仍为 `492080779e9feea19308092b0aba0840e87c6115`。DCO、Mergify Summary 和 Meta 检查通过。上游 `pre-run-check` 因贡献者尚无已合并 PR 且缺少授权标签而失败；pre-commit 被跳过，Read the Docs 的前置检查也阻止构建。这些状态未发现源码错误。日志明确禁止 agent 请求授权标签，因此未请求、添加标签或绕过 CI 策略。独立 fork 中针对当前源码的静态检查和严格文档构建已通过。

公开说明聚焦直接 FP8 cached-K/V up-projection，仅简短说明 #59124 比较运行失败，不单列比较章节或强调精度差异。New-token FP8 留待后续独立工作，前提是准确率通过且性能显著提升。额外环境变量的讨论保持开放。

GSM8K 运行 `41447950940` 的 attempt 2、job `125363224586` 在最近检查时仍在进行。需检查分数、精确源码、真实 MTP 和全部 worker 的优化路径执行证据后，再标记为可评审。

稳态性能分析接线已单独发布到 `checks/native-steady-profile-20261102`，commit 为 `9da21ee27541876ead88045459ecbf0e74e7c670`。CPU 运行 `41476491034` 的 **30 个测试全部通过，无失败或跳过**，真实 launcher 和语法检查也通过。已检查 JUnit artifact `13157862094`，确认包括 SIGTERM 清理和启动响应丢失在内的全部九个控制器用例均执行。仅分析模式延长容器关闭时间，允许停止服务前导出 trace。

性能分析默认关闭，仅允许独立原生完整 replay。等待测量阶段标记和 300 秒生成进展后，采集 20 秒 worker CPU/GPU 活动。输出和 artifact 均保存在本仓库。GPU 分析尚未派发。参见[性能分析指南](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/blob/checks/native-steady-profile-20261102/docs/kimi/native-steady-profile_zh.md)。

## 原生验证完成并进入评审

完整 GSM8K 运行 [41447950940](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/41447950940) 的 attempt 2 成功。Strict：**1276/1319 = 96.7399%**；flexible：**1275/1319 = 96.6641%**。样本包含 1319 个独立题目，每种评分 filter 各一行，共 2638 行。

已确认使用固定镜像 e0d00e4b，digest 为 `b1374c3814b6e7706c49968ba17ca3b35484382f7b9fc7115dd3e66068d53f2f`。真实 MTP k=4 采用 block 验证，日志记录了非零接受率。配置为 eager、512-token chunk、C48/TP8/DCP8、five-shot、temperature zero 和明确的准确率压力路由。此结果验证所述准确率配置下的优化算术路径；匹配的 BF16 对照仍属另一项验证。

全部八个 worker 在评估阶段记录了非空 E4M3 直接投影和 cached FMHA，native SHA 为 `492080779e9feea19308092b0aba0840e87c6115`，BLAS backend 均为 `_BlasBackend.Cublas`。provenance 清单中的八个已安装源文件 hash 均与 fork commit 独立比对一致。未检测到服务运行时错误、BLAS 失败或磁盘配额错误。独立 GPU kernel 测试仍对应运行 41401317715；此次评估安装器使用 prepare-only 模式。

Artifact：分数与样本 **13157663920**，执行证明、来源和服务日志 **13157794027**，独立服务日志 **13157993168**。[结构化报告](../../results/atom-attention/run41447950940/native-gsm8k-summary.json)。

已更新 PR 描述和现有状态评论。**PR #64288 现为 OPEN、非 draft，可供评审**，源码保持已验证版本。上游 CI 授权和人工评审仍阻止合并，未请求或添加授权标签。

下一项已授权稳态分析为 [41478604149](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/41478604149)，workflow/execution SHA 为 `9da21ee27541876ead88045459ecbf0e74e7c670`，使用相同原生 PR 和镜像，完整 C48/MTP replay 并启用 profile。前置 CPU 运行 41476491034 已通过，仍需跟踪准备、采集及最终 trace 分析。

后续 sweep：当前配置仅列出 C48，需明确添加 C56/C64。当前 launcher 指向的非 MTP Kimi script 不存在，C96 无 MTP 运行前需修正路由和 metadata。尚未派发这些 sweep。

## 并发扫描前置检查完成

独立分支 `checks/native-concurrency-sweep-20261102`、SHA `d67152dac62f6f19a42a49c72bc2ae98b7747de1` 已加入 C56/C64 MTP 和 C96 无 MTP 的独立配置，原 C48 配置不变。原生非 MTP Kimi 选择现有共用 recipe，并向容器传入 `SPEC_DECODING`；metadata 与实际参数不一致时会报错。

CPU 运行 [41479440563](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/41479440563) 通过：30 个 pytest 用例无失败或跳过，launcher 行为和 shell 检查通过。已检查 artifact **13159250994**：实际 matrix generator 仅生成 C56/MTP、C64/MTP 和 C96/none，均为 TP8/DCP8、2249 GB CPU DRAM 预算、3600 秒。原生 workflow 将旧 matrix 镜像替换为固定的 e0d00e4b 镜像。

计划参数：C56 为 k=4/AL=3.36/chunk=8192/max-seqs=112；C64 为 k=3/AL=3.00/chunk=24576/max-seqs=80；C96 无 MTP，chunk=24576/max-seqs=144。这是完整配置点的比较，不是仅改变并发的对照。尚未派发 sweep GPU job。

当前 profile 运行 **41478604149**、GPU job **125390981568** 仍处于服务/replay 步骤。workflow 和执行版本固定为 **9da21ee27541876ead88045459ecbf0e74e7c670**，并发扫描准备不会改变正在执行的任务。需继续跟踪至结束，检查 worker trace 后再安排 sweep。

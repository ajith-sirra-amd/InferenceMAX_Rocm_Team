# 原生 FP8 上投影审查

[English](./native-fp8-up-projection-audit.md) | **中文**

## ???????2026-10-30

?? [GPU ?? 41044413189](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/41044413189)
? PR ?? `332d6c99acd5281098f1fc3e2a00cd06ef866a0d` ???
**8 ????0 ???0 ???0 ??**???????????????? FMHA?
???????????????????????? GPU 0 ??????
????????????????
[??????](../../results/atom-attention/run41044413189/native-kernel-evidence.json)?

???? 41037832511 ?? 5/8?41042897918 ?? 7/8?
????????????????? GPU BLAS ?????? CPU FP64?
?????????? GPU ?????????????
???????? FP8 ??????? FP32/FP64 BLAS ???
?? `ROCBLAS_USE_HIPBLASLT=0` ??????????????????

?? PR ?? `2f946cbab5000dedef23b117df991417e8917315` ????????????
Python ??? AST ? GPU ???????????????
???? 41044549904 ???????????????????
[????? 41046262627](https://github.com/ajith-sirra-amd/vllm/actions/runs/41046262627) ?????

??????? [GSM8K ?? 41046329246](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/41046329246)?
?? 1319 ??????????C48 + MTP ?????
eager ???????? 512-token ????????
??????????? FP8 ????? FMHA????
`VLLM_GPU_SYNC_CHECK=error`?? token ??????? BF16?

?? BF16 ????? LSE <0.20 ???????
? token FP8 ???? 12.30182%???? PR?PR ?????
??????????/??????????????????????
??????????????

已测得的 16915.72154 tokens/s/GPU 使用 BF16 kv_b_proj。缓存注意力路径为
FP8 cache -> BF16 projection -> BF16 K/V -> dynamic FP8 K/V -> FP8 FMHA，
并未执行 FP8 投影乘法。

BF16 直接输出原型仍执行 BF16 乘法，FP32 累加结果先在寄存器内舍入到 BF16，
再保存为 FP8。其资格检查明确排除 FP8 权重。单独开启权重量化会回退到原生
解压路径，并不会调用 ATOM 的 FlyDSL 投影。

## ATOM 配方与内核

固定 ATOM 版本 773eb36254e43060bfd5fabcab7d3840bf591358 的
recipes/Agentic-Kimi-K3.md 开启全局 ptpc_fp8 在线量化，未排除 kv_b_proj，
并开启 FlyDSL gather/projection 和 FP8 FMHA。代码中的 Kimi BF16 回退注释
适用于未开启在线转换的配置。

适配器调用未修改的 AITER bbef5a6f2d3733b293c80d2cee6a8d464e88bc01
gather_kv_b_proj_flydsl：E4M3 缓存乘以 E4M3 权重，FP32 累加，使用调用者提供
的标量 descale 直接写 E4M3 K/V，不经过 BF16 舍入。新 token scale 准备和
缓存注意力 Q 量化保留现有 ATOM 策略；因果及小块注意力保留原混合路由。

## vLLM 集成

配方在现有 targets 中新增 "*.self_attn.kv_b_proj": "fp8_per_channel"。
转换和打包由原生 Fp8PtpcOnlineLinearMethod 完成，适配器仅获取缓冲区视图，
不添加第二套权重量化器。

| 原生线性内核 | 原生存储 | 适配方式 |
|---|---|---|
| AiterHipbMMPerTokenFp8ScaledMMLinearKernel | shuffled [N,K] 的转置视图，scale [1,N] | 先转置视图还原，再展平 scale |
| AiterPreshuffledPerTokenFp8ScaledMMLinearKernel | shuffled [N,K] | 直接传入并启用 preshuffle |
| AiterPerTokenFp8ScaledMMLinearKernel | 普通 [N,K] | 直接传入并关闭 preshuffle |
| RowWiseTorchFP8ScaledMMLinearKernel | 普通转置 [K,N] | 转置还原，关闭 preshuffle |

未审查的布局明确报错。检查 E4M3 dtype、连续 [heads*256,512] 存储、FP32
逐通道 scale、无 bias、缓存维度、索引 dtype 和设备。保留原生 DCP 字节 gather
及行顺序。同一符合条件的 forward 中，小块使用 FP8 投影输出 BF16，随后执行
BF16 注意力。权重变化也影响新 token 投影和原生吸收后的 decode 权重，影响范围
不限于缓存注意力。

## 服务和性能运行前的证据

加载就绪不等于实际调用。[ATOM-FP8-PROJECTION-READY] 记录加载参数；
[ATOM-FP8-UP-PROJECTION] 仅在实际 FlyDSL 调用后输出。vllm bench serve
prefix probe 要求八个 worker 都具备这些标记及投影、注意力验证标记，不能
通过静默回退获得通过。

GPU 预检使用稀疏且可精确量化的权重和可手算输出，执行原生在线转换及四种
原生打包实现；同时执行原生线性内核选择、真实 backend 权重 hook、FP8 投影、
注意力、scale/Q 生命周期和改变输入后的 graph replay。FP64 反量化 FP8 投影
作为独立算术参考，输出量化要求 RMS <0.04、归一化最大误差 <0.07 且无截断。
此前 BF16 原型继续执行同 scale HIP 字节精确比较。原注意力门限不变：
RMS <0.10、绝对 LSE 误差 <0.20。

准备阶段五项 CPU flag 路由测试及 Python 语法检查通过。新 FP8 投影集成的
独立 GPU 算术验证已通过，完整模型验证仍待执行；目前不声称性能提升或模型准确率通过。

## 真实模型数值发现

运行 40678558488 失败并保存 atom_fp8_failure.pt。八个 worker 的 16 次失败
调用共检查 64 行：FP8 FMHA 相对相同 FP8 输入的 FP64 LSE 最大误差为
0.00042494，而 Q/K 量化引起的 LSE 偏移最高为 1.04069066。原动态 scale
路径相对 BF16 的 LSE 最大误差为 0.93494225，同样超过 0.20 门限。
部分行相对相同 FP8 输入的注意力输出 RMS 误差达 0.13575，因此仅 LSE 一致
不能证明 FMHA 正确。

原合成输入数值测试的通过结论对其输入仍然有效。GSM8K 40658068830 使用
BF16 注意力通过，不能证明此 FP8 路径准确率。后续先用保存的输入检查 scale
和 softmax rescaling，再进行服务 smoke 或性能运行。诊断完成不等于数值验收。

## GPU 结果与捕获输入的参考计算

运行 [40682885241](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/40682885241)
使用提交 58882515b58010bf1411d164c2cb44fab6e049a3，通过四种原生权重布局、
实际选中的 AiterPreshuffledPerTokenFp8ScaledMMLinearKernel、FP8 x FP8 投影到
FP8 FMHA，以及输入变化后的 graph replay 检查。投影相对 FP64 的 RMS 约为
0.0262--0.0266，未发生截断。这仅证明独立 GPU 测试通过；真实 Kimi 服务路径
及模型精度尚未通过验证。

运行 [40682306781](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/40682306781)
完成了之前的捕获输入诊断。提供的 scale、动态 scale、每轮 softmax 重缩放及
二次幂 scale 均未通过原有绝对 LSE 误差小于 0.20 的检查。诊断任务成功不等于数值验收通过。

下一次回放使用 attention_reference.py，为保存的 chunk 中所有 query 和 head
计算 FP64 参考注意力，分别报告仅 Q 量化、仅 K/V 量化和原生 FP8 权重量化的影响。
同时检查 BF16 与 FP8 投影，分别使用捕获的 scale 和根据 FP64 最大值计算的诊断
scale；后者仅用于诊断，不是生产路径。测试 lazy-rescale 和 denormal 处理组合，
单独保留原 BF16 kernel 验收结果与相同 FP8 输入的算术检查。与连续 AITER HIP
输入的量化结果差异会明确报告，包括一个 ULP 的差异。

已查看的上游 PR：
- vllm-project/vllm#59124，提交 132f6daa6b17cb4000ef58355943bb39339104f2，仍开放：
  gfx950 动态逐张量 Q/K/V 量化、支持 stride 的量化器及 Q 复用。
  作者报告两组 GSM8K 均为 0.9621，但不能证明本分支改动的精度。
- vllm-project/vllm#62447，提交 cefacabced64eda821bcb49493cc970056d15429，仍开放：
  ROCM_AITER_FA 可选 FP8 路径，先创建连续 Q/K/V 再动态逐张量量化。
  尚无 GSM8K 结果；相同输入参考用于区分 FMHA 与量化误差。
- vllm-project/vllm#50181 修复 Blackwell 后端选择，不是 MI355X FP8 舍入修复。
  本次未引入上游代码，也未改变验收阈值。

## 完整捕获 chunk 结果：运行 40684025796

提交 5975367b961baa7f0ce957f37bce25c0be5a18d7 的 16 个相同 FP8 输入算术检查
全部通过；原 BF16 差异检查全部未通过。Q/K/V 共 17,694,720 / 35,389,440 /
23,592,960 个元素与同 scale、连续输入的 AITER HIP 量化结果逐字节一致。
原生 FP8 投影在提供的 scale 和诊断动态 scale 下均通过，未发生截断。
提供的 scale 下 K/V 相对 FP64 的 RMS 为 0.02239 / 0.02647。

整个捕获 chunk 中，仅 Q 量化引起的最大 LSE 变化为 0.513783，仅 K/V 量化为
0.370814；最终 K/V 量化之前的原生权重转换影响为 0.053012。
原生 FP8 投影使用提供的 scale 并接 FP8 FMHA 时，相对原 BF16 kernel 的输出
RMS 为 0.028700，最大 LSE 差异为 0.934950，因此移除 BF16 中间结果未修复该问题。
关闭 lazy rescale 将相同输入下的 FMHA RMS 从约 0.0129 降至 0.0030，但未修复
Q/K 量化误差，也未用于服务路径。denormal 处理开关在本测试中结果逐位一致。

下一诊断将区分 NoPE 与 RoPE 量化影响，并记录误差最大的行。同时测试按请求对
K 减去均值：Kc = K - mean(K)。精确运算下注意力输出保持不变，LSE 加上
scale * dot(Q, mean(K)) 即可还原。先独立验证此恒等式，再测 FP8。
原型使用 BF16 居中 K 和 FP64 修正，不能作为生产优化或性能证据。
服务路由阈值与数值验收阈值均未改变。

## RoPE 误差定位与下一诊断

运行 40684733665（提交 f1c199b5cb3209e4ec0a43edebf55e0324585ec1）已完成。
分别量化单个部分时的最大 LSE 变化为：Q NoPE 0.158969、Q RoPE 0.512855、
K NoPE 0.106059、K RoPE 0.370712。最差一行的峰值分数变化中，RoPE 贡献约
-0.729，NoPE 约 -0.00165；原最大 key 的概率为 0.99547。
该捕获失败主要来自位置编码列。

未量化的 K 减均值恒等式通过（最大 LSE 误差 1.78e-14）。
FP8 原型将最大 LSE 差异从 0.727297 降至 0.479201，仍未通过 0.20，
因此没有用于服务路径。

下一诊断将相同的归一化 64 列 Walsh-Hadamard 矩阵应用到 Q/K RoPE，
分别测试无符号变换和固定随机符号向量（seed 20261028）。先验证未量化的注意力
恒等式，再检查原验收阈值。原生 FP8 权重分支的 NoPE 和 V 由实际 ATOM 投影计算，
仅 RoPE 使用原 compact cache 的 FP32 参考变换结果覆盖。
这不代表已有融合变换内核或性能收益；Q 的 BF16 变换舍入明确计入。
该实验超出已查看的 ATOM/vLLM PR 实现，不能视为上游精度保证。

## 旋转结果与残差列诊断

运行 40685301834（提交 a5a4d299d8ce5e5e14759bcca02f3957fbc7f889）的八个
相同输入算术检查均通过，但所有原 BF16 验收均失败。原生 FP8 权重配合无符号
RoPE 旋转的最大 LSE 差异为 0.536367，带符号变换为 0.635032，均未采用。

下一诊断使 K 的 descale 与 compact cache 的 descale 保持整数二次幂关系，
以保留缓存 RoPE 的主要 FP8 编码。增加 64 个 FP8 列：Q 携带其 scale 域内
RoPE 舍入残差的 16 倍，K 在对应列复制除以 16 的 RoPE。
这样 256 列点积包含残差修正，无需改变 AITER FP8 FMHA 实现。
必须显式传入原模型 softmax scale，不能改为 1/sqrt(256)。
FMHA 所有输入仍为 FP8，并实际调用原生 FP8 投影。
不能据此断言额外打包流量或 256 列注意力更快；仅 kernel 计时明确排除打包与投影。
本次仅为诊断，不是生产集成或模型精度证明。

## 残差列结果与 NoPE 平衡诊断

运行 40685906330（提交 c53e307afbaff423bd0d3fc16f18d3299afdc2b0）已完成。
对齐 cache scale 后的 K RoPE 主项完全一致，无截断，复制的 RoPE 也无下溢。
Q RoPE 重构 RMS 为 0.0006364。原生 FP8 投影与 256 列残差 FMHA 的最大 LSE
差异为 0.217426，输出 RMS 为 0.024845（lazy）或 0.021295（eager），仍未通过
0.20。八个相同输入算术检查均通过。原生残差 FMHA 耗时 0.5778 ms，
原 192 列 FMHA 为 0.5568 ms；均未计打包成本。

下一诊断按 head 平衡 NoPE：Q'=Q/c、K'=cK，其中
c=sqrt(amax(Q_NoPE)*K_scale/(amax(K_NoPE)*Q_scale))。
原生投影只调整临时逐通道 descale，不再次量化已打包的 FP8 权重。
RoPE 残差方式不变。本诊断使用捕获的 K 最大值，即使通过也不能证明已有生产
scale 选择策略。原数值验收阈值保持不变。

## 当前结果与下一次完整模型验证

六次捕获输入诊断均已完成。原生 FP8 投影及相同输入的 FMHA 算术检查通过，
但原生 FP8 权重仍未通过原 BF16 LSE 差异验收。运行 40686549777（提交
6d0f6947d8454751041e8d6a09249682a6b7528c）中，平衡后的 BF16 权重原型最大
LSE 差异为 0.198254（该输入通过），原生 FP8 权重原型为 0.208252（失败）。
这些修正均未用于服务路径。诊断任务成功不能当作数值验收通过。

下一次完整 GSM8K 使用现有 ATOM 等效的原生 FP8 投影与 192 列 FP8 FMHA，
不启用诊断中的旋转、残差或平衡。目的为独立算术验证后的模型精度测量，
不会取消或替代已失败的 0.20 BF16 差异检查。

之前 GSM8K 的提示较短，未达到性能路径阈值，因此未执行 FP8 注意力。
现在 EVAL_ONLY 路径将：
- 设置 MAX_BATCHED_TOKENS=512 与 ATOM_FP8_ACCURACY_STRESS=1；
- 对非空 query/context 使用 FP8 缓存注意力，保留新 token 的因果 BF16 路由及主 FP8 开关；
- 使用 eager 执行，明确记录实际调用；
- 保持完整 1319 题、five-shot、temperature=0、真实 speculative verification，
  不改问题或评分；
- 仅在评测开始前创建阶段标记，评测期间必须获得全部八个 worker PID 的
  原生 FP8 投影与 E4M3 Q/K/V FMHA 调用证据。
  warmup、BF16 投影、错误 dtype、空调用或缺少 worker 均不能通过。

常规性能路由仍为 query >=512、context >=8192。十项 CPU 路由及证据检查通过。
完整模型运行前的 GPU 前置任务将检查 Bash 语法、CPU 测试、原生权重布局，
以及大/小 context 数值和输入变化后的 graph replay。
附带调用证明的 GSM8K 结果只代表该压力配置的模型精度，不是吞吐或严格 BF16 数值通过证明。

## GPU 前置检查已通过；GSM8K 已提交

运行 [40687966988](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/40687966988)
使用提交 14250d6c6c17cd54fb7cd480c74c5c4eb5da2e78，通过 Bash 语法、十项 CPU
检查、四种原生权重布局，以及大 context（512 Q / 8192 KV）与小 context
（64 Q / 256 KV）投影/FMHA 链及输入变化后的 graph replay。
已输出 FP8_ACCURACY_PATH_PREFLIGHT_PASS 与 FP8_NATIVE_SMALL_CONTEXT_PASS。

完整 GSM8K 运行 [40688077118](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/40688077118)
固定到同一提交。相同源版本的前置检查已通过，MI355X agentic eval job 已启动。
评分及八个 worker 在评测阶段的 FP8 调用证明尚待完成。
Python 工具会话保持可用期间，检查器每 60 秒查询一次该运行。

## 完整 GSM8K 通过，并验证实际 FP8 执行

运行 [40688077118](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/40688077118)
使用提交 `14250d6c6c17cd54fb7cd480c74c5c4eb5da2e78`，已成功完成。
完整 1319 道题、五样本提示、temperature=0 的评估采用真实推测解码验证，
执行原生 FP8 `kv_b_proj` 和未修改的 192 列 AITER FP8 FMHA。
`GSM8K_FP8_EXECUTION_PROOF_PASS` 记录了全部八个 worker 在评估阶段
实际执行原生 FP8 上投影及 E4M3 Q/K/V FMHA；下载的服务日志也独立通过同一验证器。

| GSM8K 指标 | 之前走 BF16 路径的运行 | 原生 FP8 压力评估 | 差异 |
|---|---:|---:|---:|
| Strict match | 1282/1319 (97.1948%) | 1281/1319 (97.1190%) | -1 题 / -0.0758 个百分点 |
| Flexible extract | 1281/1319 (97.1190%) | 1281/1319 (97.1190%) | 0 题 |

全部样本的题目、提示和目标答案哈希与基线一致。Flexible 评分中，
14 道原本答对的题变错，14 道原本答错的题变对；Strict 评分分别为 14 和 13 道。
总体准确率相同不代表逐题答案相同。两个分数均超过未修改的仓库 GSM8K 阈值 0.90。

[机器可读证据](../../results/atom-attention/run40688077118/gsm8k_evidence.json)
包含 artifact 标识、逐题变化、精确分数及全部八个执行记录。
原始评估 artifact 为 `12818760410`，服务日志为 `12818805276`。

这是 eager/512-token 压力配置下的模型准确率证据。原有 BF16 绝对 LSE
标准仍未通过；本次分数没有通过修改验收阈值或服务内核来获得。
下一次性能冒烟测试采用相同源码，恢复正常路由和图执行设置。
目前仍不宣称原生 FP8 上投影带来性能提升。

## GSM8K 通过后已提交性能冒烟测试

[运行 40694030981](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/40694030981)
执行源码 `14250d6c6c17cd54fb7cd480c74c5c4eb5da2e78`，并以成功完成的 GSM8K 运行为
相同源码的前置条件。GPU 作业 `122836131583` 已启动。本次请求 600 秒回放，
采用正常性能路由（query >=512、context >=8192）、8192 batched tokens 和图执行，
关闭 accuracy stress。这是诊断运行，不能直接作为相对既有 3600 秒、
16915.72154 tokens/s/GPU 结果的最终性能证据。完整性能运行前应检查吞吐量、
错误情况以及全部八个 worker 实际执行原生 FP8 路径的证据。

[提交记录](../../results/atom-attention/run40694030981/dispatch.json)。
持久检查器每 60 秒轮询一次。配套收集器会在产物出现后下载 benchmark 和服务日志，
提取吞吐量及实际投影/FMHA worker 记录；收集器不会自动批准或提交完整性能运行。

## 按用户要求改为完整运行并取消冒烟测试

用户要求直接完整运行，不做冒烟测试。已通过 GitHub Actions 取消 40694030981，
GPU 作业已清理。这次取消不是数值或性能失败。

完整[运行 40695099743](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/40695099743)
使用相同已验证源码 `14250d6c6c17cd54fb7cd480c74c5c4eb5da2e78`、正常路由及图执行，
按配置进行 3600 秒、并发 48、TP8/DCP8 的 agentic 回放。GSM8K 前置条件通过，
GPU 作业 `122839857513` 已启动。检查器和完整运行产物收集器每 60 秒检查一次。

[完整运行提交记录](../../results/atom-attention/run40695099743/dispatch.json)。
[现有 vLLM PR 实现比较](./vllm-fp8-prefill-pr-comparison_zh.md)记录重叠范围、
不同的 FP8 投影集成及用户指定的原生开关 `VLLM_ROCM_USE_AITER_FLYDSL_FP8_PREFILL_ATTN`
（默认 0）。现已发布[原生 vLLM 草稿 PR #64288](https://github.com/vllm-project/vllm/pull/64288)，来自 fork 分支 `perf/rocm-fp8-kv-proj-fmha`。正在运行的 benchmark 仍使用已测试插件和旧开关；原生补丁尚未在 GPU 上验证。

## 原生 vLLM 草稿 PR 已发布（2026-10-28）

[PR #64288](https://github.com/vllm-project/vllm/pull/64288) 修改八个文件，
提交为 `d7f52a37e4e38d893b70d09bbd7a94b5d80cd555`，基于上游
`ed6b8f4cd2b47e4240e7ff5feb5105444245b0c2`。
补丁集成原生逐通道 FP8 权重布局适配、直接 FP8 K/V 输出、FlyDSL 缓存上下文 FMHA、
单次 forward 的 Q/scale 生命周期，以及默认关闭的
`VLLM_ROCM_USE_AITER_FLYDSL_FP8_PREFILL_ATTN`。
目标模型新 token 因果注意力保持 BF16；draft 模型路由保持原状。

[静态检查](https://github.com/ajith-sirra-amd/vllm/actions/runs/40700944665)通过：
适用的 pre-commit hooks、Ruff、mypy、Markdown、SPDX、设备 API、
Buildkite 测试绑定、Python 语法和 Buildkite YAML。
最终文件 blob 与验证产物完全一致。QKV 量化器可执行 AST 与已测试原型一致；
仅调整 import、格式及许可证说明。

按用户明确要求，GPU kernel、图执行、同步检查、原生补丁 GSM8K 和端到端性能验证均待完成，
先创建草稿 PR。PR 将原型 97.119% GSM8K 证据单独列出，
不把它视为原生补丁验证。历史 BF16 绝对 LSE <0.20 条件仍未通过。

[发布与验证记录](../../results/atom-attention/vllm-native-pr-20261028.json)。

## 独立后续实验：新 token 因果 FP8 注意力

参考 [PR #59124](https://github.com/vllm-project/vllm/pull/59124)，使用相同 FlyDSL API，
设置 `causal=True`、相同 Q/KV 序列偏移、`cross_seqlen=True` 和独立 Q/K/V descale，
并保护空输入和 32 位索引限制。

当前直接上下文准备已计算新 token 的 FP8 Q/K/V，下一候选可以直接复用这些 tensor，
不再增加一次量化。因果注意力必须使用原始新 token descale；缓存投影仍使用独立的
2 倍余量 gather descale。保留当前缓存上下文 Q 切片的量化范围，
不要同时采用参考 PR 的整段 Q 切片复用。

该后续实验已审查，尚未实现或派发；与 #64288 分离。
需要验证因果 mask、相同 FP8 输入的输出/LSE、模型准确率及完整性能。
完整运行 40695099743 及每 60 秒检查的 checker/collector 仍在运行。

## 完整性能运行已完成（2026-10-28）

[运行 40695099743](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/40695099743)
成功完成，执行源码为 `14250d6c6c17cd54fb7cd480c74c5c4eb5da2e78`。
配置为 3600 秒 AgentX 回放、并发 48、八张 MI355X、TP8/DCP8 和正常图执行及路由。
实际报告的测量窗口为 3580.06515 秒，既有运行为 3627.79603 秒。

| 指标 | 既有 40646139769 | 当前 40695099743 |
|---|---:|---:|
| 输入加输出 tokens/s/GPU | 16915.72154 | 17406.08912（+2.8989%） |
| 输出 tokens/s/GPU | 117.84290 | 120.16691（+1.9721%） |
| 平均 TTFT，秒 | 1.87909 | 1.76809 |
| p95 TTFT，秒 | 4.61033 | 4.36319 |
| 平均 TPOT，毫秒 | 24.68 | 23.15 |
| 平均端到端延迟，秒 | 19.29718 | 18.04654 |
| 测量阶段成功请求 | 3598 | 3644 |
| 测量阶段错误 | 7/3605（0.1942%） | 5/3649（0.1370%） |

当前原始记录包含 425 个预热错误和 5 个测量错误（4 个空内容响应和 1 个客户端连接错误）。
预热与错误排除存在交集，430 个原始错误不代表 430 个测量错误。
八个 worker（PID 1921-1928）均实际调用 FP8 缓存乘 FP8 权重、直接 FP8 输出和 FP8 FMHA。
工作负载配置相同，但完成的请求及缓存命中不同；这是单次完整运行观察，不能归因于单个 kernel 或 CCD。

性能运行与既有运行均使用合成投机接受率。独立完整 GSM8K 使用真实验证，结果仍为
1281/1319；性能运行不是准确率测试。原生 PR #64288 仍待独立 GPU 验证，
历史 BF16 绝对 LSE 门限仍未通过。

[机器可读结果](../../results/atom-attention/run40695099743/performance_evidence.json)。
用户要求的下一实验是在独立候选中使用已准备的新 token FP8 Q/K/V 及原始 descale 执行因果注意力。

## 因果实验已启动

独立因果实现及 GPU 诊断已完成；完整 GSM8K 和性能运行按成功前置条件排队。
参见[因果 FP8 证据](./causal-fp8-attention_zh.md)。

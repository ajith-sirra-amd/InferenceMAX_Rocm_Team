# FP8 KV all-gather ??

[English](kimi-fp8-kv-allgather.md)

AMD checkpoint?C48 TP8/DCP8 MTP k4?auto loader???1??600?????????? FP8 cache ? uint8 ????????? rank ??? scale ??? BF16??? BF16 ??/FMHA ? query ??????????? backend ??????????? BF16 all-gather ???????????? scales ????????????????????23M KV pool ??????????????? AMD ????? dtype probe????? ASM?39239112271 ? Moonshot ???????????????????????????????????

## 启动补丁边界修复

Smoke [39561981905](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/39561981905)
在正式回放前失败：安装器从非 DCP 方法中的缓存分支开始替换，导致后续 DCP 方法嵌套在 else 内。
修复将替换范围限制为 DCP 上下文方法，并拒绝意外的方法或分支布局。
行为检查覆盖 FP8、packed FP8、BF16 和相邻方法；固定版本的运行时源码也通过编译。
GPU 执行、缓存精确一致性、KV 容量和吞吐量仍需替代 smoke 验证。
本修复不会启用 FP8 ASM FMHA。

## FP8 解码器编译修复

Smoke39570966330 启动时恢复了 **23,098,919 tokens** 的 GPU KV pool，随后在内核预热阶段失败。
Triton 拒绝将 masked load 的整数填充值转换为 FP8。修复移除未使用的填充值；被掩码的元素不会写入输出。
FP32 scale 的填充值改为浮点数。

服务器启动前，在每个可见 GPU 上运行真实解码器，覆盖全部256种 FP8 字节编码、两种源 rank scale 和末尾不足一个 block 的输入。
与 PyTorch 解码进行精确比较，并处理 NaN 一致性。编译或数值错误将在模型权重加载前失败。
保留实际缓存的精确一致性检查。失败运行未产生正式吞吐量或模型准确性结果；KV pool 是内核预热完成前的启动观测值。

## 原生调度诊断 smoke

禁用实验性的原始字节传输安装器，保留固定版本运行时的原生通信和重建行为。
添加一次性观测日志，记录 DCP 上下文路径的 eligibility，以及真实 KV collective 的 dtype、shape 和字节数。
记录重建权重 dtype、布局和 scale 是否存在，以直接验证 FP8 路径被排除的原因。
使用 AMD checkpoint、auto loader，不传 CLI 量化覆盖；C48 TP8/DCP8，每 lane 预热1次，回放600秒。
要求两种诊断日志和现有上下文输入日志都存在才能接受 smoke。
诊断及后续通信适配验证通过前不运行完整 benchmark。

## 原生诊断结果39576708017

Workflow成功。全8个 GPU 记录 eligibility=False：重建权重为 FP8 (3072,512)，存在 weight_scale。实际 KV collective 使用 BF16。KV pool23098919 tokens。总吞吐10985.22783 tok/s/GPU，输出56.97678 tok/s/GPU。本次只增加日志，吞吐差异不代表优化收益。
正式阶段有456个有效请求和1个空内容 InvalidInferenceResultError (0.22%)。汇总错误计数包含预热。server/benchmark日志没有 ServerDisconnectedError。结果不证明模型准确性。
证据：rocprof/reports/kimi-model-communication-comparison/39576708017-native-diagnostic/.

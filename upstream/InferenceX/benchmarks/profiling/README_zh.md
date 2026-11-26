# Kimi 实测阶段性能分析

[English](README.md) | **中文**

[Kimi C48 直接启动脚本](../single_node/agentic/kimik3_fp4_mi355x_mtp.sh)
支持 **`--profile`**；不传此标志时不启用性能分析。它使用固定版本 vLLM
镜像的原生 Torch profiler，采集八个 worker。SA 模型、精度、图配置、预热
以及 3600 秒 AgentX 回放保持不变。分析运行的结果应与吞吐基线分别保存。

先提供已有的服务端、模型和客户端环境，包括挂载到 `/data` 下的本地快照、
新的可写 `RESULT_DIR`，以及与所选 SA 源码匹配且已准备完成的
`SA_CLIENT_ENTRYPOINT`。在团队仓库根目录执行：

```bash
TEST=0 PROFILE=0 DURATION=3600 \
PROFILE_DELAY_SECONDS=1800 \
PROFILE_SECONDS=130 \
PROFILE_MIN_SECONDS=120 \
PROFILE_PHASE_TIMEOUT_SECONDS=7200 \
PROFILE_POLL_SECONDS=2 \
PROFILE_EXPORT_TIMEOUT_SECONDS=600 \
bash upstream/InferenceX/benchmarks/single_node/agentic/kimik3_fp4_mi355x_mtp.sh --profile
```

六个时间参数均须显式提供。示例在 **AgentX 实测阶段的时间戳之后**等待
1800 秒，采集 130 秒，并要求所有 GPU 的公共 trace 时间跨度至少为 120 秒。
采集时间不以模型启动为起点。`PROFILE_PHASE_TIMEOUT_SECONDS` 限制客户端
启动后等待实测阶段的时间，不改变服务启动超时。
`PROFILE_EXPORT_TIMEOUT_SECONDS` 分别限制启动和停止 API 的响应等待时间。
采集最多 180 秒；延迟、采集、两次 API 时间预算、轮询以及 90 秒的健康和
指标请求预算，必须全部放入 `DURATION`。参数无效或分析目录已存在时，
在 GPU 启动前失败。

此选项支持原生 AgentX（`TEST=0`），拒绝 `TEST=1` 及旧的 `PROFILE=1`
机制，后者会改变定长测试请求数。客户端须在 `RESULT_DIR` 下生成原生
`agentic/benchmark.log` 和 `agentic/agentic_power_timezone_offset.txt`。
控制器检查实测阶段及生成 token 的进展，然后调用 `/start_profile` 和
`/stop_profile`。导出完成前服务保持运行；取消时也会尝试有时间上限的停止
和导出。不增加 CPU 绑核或第二个 GPU profiler。此时间采集关闭形状、调用栈
和内存跟踪；HBM 计数器需要单独采集。

`RESULT_DIR` 下的输出：

| 文件 | 内容 |
| --- | --- |
| `vllm_core_command.txt` | 加入诊断 profiler 前的 SA 命令，供可选 SA 审计检查 |
| `vllm_command.txt` | 包含 `--profiler-config` 的实际服务启动命令 |
| `profiles/torch/settings.json`、`profiler-config.json` | 采集参数及原生 profiler 配置 |
| `profiles/torch/controller.log`、`profile-control.json` | 阶段检测、token 进展、启停时间、导出状态及错误 |
| `profiles/torch/*.trace.json.gz` | 完整 worker trace，也可能位于原生输出的子目录中 |
| `profile-validation.json` | 八个 worker 的覆盖情况，以及实际采集是否位于实测发送区间内 |
| `profiles/torch/kernels.csv`、`top10-kernels.csv` | 全部内核符号及按 GPU 总耗时排列的前十项，包括调用数、平均/最小/最大耗时和 GPU 时间占比 |

验证会完整解析 trace，要求设备 0–7 均有内核记录，并检查读取时文件未变化。
缺少 worker、导出不完整或采集超出实测发送区间都会使运行失败。仅验证通过
后生成 CSV。GPU 时间占比累加了各 worker 重叠执行的时间，不等于端到端
延迟占比，也不能证明优化收益。为未压缩导出临时文件和压缩 trace 预留足够
可写空间，并保留失败运行的原目录。

本地验证以模拟的 HTTP/GPU/客户端运行真实 shell、控制器和验证器：

```bash
python3 -m unittest discover -s upstream/InferenceX/benchmarks/profiling -p 'test_*.py'
```

这验证控制流程和产物；仍须在选定 runner 上进行新的完整模型采集，才能证明
真实 GPU 覆盖情况。

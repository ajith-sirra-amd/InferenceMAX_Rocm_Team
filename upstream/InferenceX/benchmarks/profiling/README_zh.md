# Kimi 基准测试与性能分析

[English](README.md) | **中文**

[Kimi 脚本](../single_node/agentic/kimik3_fp4_mi355x_mtp.sh) 支持三种工作负载，
每种均可使用 `PROFILE=0`（默认）或 `PROFILE=1`。
`--profile` 和 `--no-profile` 优先于环境变量。

| `BENCHMARK_MODE` | 工作负载 | 兼容旧参数 |
| --- | --- | --- |
| `serving` | `run_benchmark_serving`，仓库内的随机请求客户端 | `TEST=1` |
| `vllm` | 原生 `vllm bench serve`，使用 `prefix_repetition` 数据集 | 设置 `BENCHMARK_MODE=vllm` |
| `agentic` | `run_agentic_replay_and_write_outputs`，轨迹回放及结果校验 | `TEST=0`，也是默认模式 |

`BENCHMARK_MODE` 优先于 `TEST`。三种模式使用同一套源自 SA 的服务端配置，
并启用服务端前缀缓存。开启分析不改变请求数、并发、预热或回放时长。
本脚本使用独立控制器，避免旧共享函数的 `PROFILE=1` 将请求数限制为并发数。

## 默认值与覆盖

以下默认值是用户明确要求为此直接启动脚本提供的。环境变量优先；
显式传入空值或无效的必需参数会校验失败。

| 参数 | `serving` | `vllm` | `agentic` |
| --- | --- | --- | --- |
| `TEST_ISL`（随机输入或独立后缀 token 数） | 8192 | 3584 | 由轨迹决定，忽略该参数 |
| `TEST_PREFIX_LEN`（共享前缀 token 数） | 0 | 4608 | 由轨迹决定，忽略该参数 |
| `TEST_OSL`（输出 token 数） | 1024 | 1024 | 由轨迹决定，忽略该参数 |
| `CONC`（客户端并发） | 48 | 48 | 48 |
| `MAX_NUM_SEQS`（服务端调度上限） | 48 | 48 | 48 |
| `TEST_NUM_PROMPTS` | 192 | 192 | 不使用 |
| `DURATION`（实测工作负载秒数） | 不使用，按请求数结束 | 不使用，按请求数结束 | 3600 |
| `PROFILE_DELAY_SECONDS` | 10 | 10 | 1800 |
| `PROFILE_SECONDS`（采集秒数） | 10 | 10 | 130 |
| `PROFILE_EXPORT_TIMEOUT_SECONDS`（每次启停确认等待） | 120 | 120 | 600 |

公共默认值为 `PROFILE_PHASE_TIMEOUT_SECONDS=7200`、`PROFILE_POLL_SECONDS=0.5`、
`PROFILE_MIN_SECONDS=0.8 * PROFILE_SECONDS`。
`TEST_RANGE_RATIO=1` 使仓库客户端使用固定长度。
`VLLM_NUM_PREFIXES=1` 表示原生 vLLM 数据集只有一个共享前缀；增加该值可创建多个前缀组。
原生客户端使用 `VLLM_BENCH_PYTHON`，默认 `/opt/runtime-venv/bin/python`。

客户端并发与 `MAX_NUM_SEQS` 相互独立。任一值不为 48 即构成新的工作点，
不能视为复现 SA C48。这些分析默认值是起始设置，并非实测最优配置。

## 示例

先设置服务端及模型环境：`MODEL`、`MODEL_PATH`、`DRAFT_MODEL_PATH`、`TP=8`、
`DCP_SIZE=8`、`EP_SIZE=1`、`KV_OFFLOADING=dram`、`KV_OFFLOAD_BACKEND=vllm-simple`、
`TOTAL_CPU_DRAM_GB=1799`、`EVAL_ONLY=false`、`PORT`、`RESULT_FILENAME`，
以及全新、可写的 `RESULT_DIR`。模型快照挂载到 `/data`，使用
[SA 参考指南](../../../../benchmarks/agentic/README.md) 中的匹配镜像。

在团队仓库根目录执行，以上参数须已导出：

```bash
SCRIPT=upstream/InferenceX/benchmarks/single_node/agentic/kimik3_fp4_mi355x_mtp.sh

# 仓库客户端，普通随机请求。
BENCHMARK_MODE=serving PROFILE=0 RESULT_DIR=/data/results/serving-plain bash "$SCRIPT"
BENCHMARK_MODE=serving PROFILE=1 RESULT_DIR=/data/results/serving-profile bash "$SCRIPT"

# 仓库客户端，共享前缀：3584 + 4608 = 8192 输入 token。
BENCHMARK_MODE=serving TEST_ISL=3584 TEST_PREFIX_LEN=4608 PROFILE=0 \
RESULT_DIR=/data/results/shared-plain bash "$SCRIPT"
BENCHMARK_MODE=serving TEST_ISL=3584 TEST_PREFIX_LEN=4608 PROFILE=1 \
RESULT_DIR=/data/results/shared-profile bash "$SCRIPT"

# 原生 vLLM prefix_repetition，默认同样以 8192 输入 token 为目标。
BENCHMARK_MODE=vllm PROFILE=0 RESULT_DIR=/data/results/vllm-plain bash "$SCRIPT"
BENCHMARK_MODE=vllm PROFILE=1 RESULT_DIR=/data/results/vllm-profile bash "$SCRIPT"

# 轨迹回放，需先完成下述额外设置。
BENCHMARK_MODE=agentic PROFILE=0 RESULT_DIR=/data/results/agentic-plain bash "$SCRIPT"
BENCHMARK_MODE=agentic PROFILE=1 RESULT_DIR=/data/results/agentic-profile bash "$SCRIPT"

# 覆盖示例：低并发、独立调度上限及采集窗口。
BENCHMARK_MODE=serving CONC=8 MAX_NUM_SEQS=64 TEST_ISL=4096 TEST_OSL=512 \
PROFILE=1 PROFILE_DELAY_SECONDS=5 PROFILE_SECONDS=5 \
RESULT_DIR=/data/results/serving-c8-profile bash "$SCRIPT"
```

共享前缀长度会**加到** `TEST_ISL` 上，而非包含在其中。仓库客户端应用聊天模板，
原生 vLLM 的 `prefix_repetition` 使用 completion 请求。分词及前缀块对齐会影响
实际缓存复用；此模式用于测试前缀缓存，不保证固定命中率，也不等同于 AA 完整工作负载。

Agentic 模式需先 source `upstream/InferenceX/benchmarks/runtime_settings.sh`，
再设置自己的覆盖值。将 `INFMAX_CONTAINER_WORKSPACE` 导出为本次 checkout 的
`upstream/InferenceX` 路径，导出 `MODEL_PREFIX=kimik3.1` 和 `FRAMEWORK=vllm`
以正确选择轨迹及记录结果元数据，并设置 `IS_MULTINODE=false`、`REQUIRE_POWER=0`、
`ENABLE_AGENTX_POWER=0`、`AIPERF_EXPERIMENTAL_FAST=0`，以及可写的 `AIPERF_RUNTIME_DIR`。
`AGENTIC_OUTPUT_DIR` 指定聚合输出位置。现有 `resolve_trace_source` / `build_replay_cmd`
会在 GPU 启动前安装客户端并准备数据集，需要 `uv`、数据集访问权限及足够磁盘空间。
此模式使用**当前共享函数所对应的客户端**，并非最新 SA 实测的精确 AgentX 依赖集。
直接启动脚本不再使用 `SA_CLIENT_ENTRYPOINT`。历史对比仍使用归档脚本及固定版本客户端。

## 采集时间与校验

延迟从**预热后的实测阶段开始**计算，不从服务端启动计算，也不推迟客户端启动。
`PROFILE_SECONDS` 控制采集窗口，不控制合成基准测试总时长。合成模式完成全部
`TEST_NUM_PROMPTS` 后结束；Agentic 使用 `DURATION`。阶段等待超时从客户端启动计时，
不会改变服务端启动超时。

仓库客户端写入原子更新的实测时间标记。原生 vLLM 启动器调用已安装的
`vllm bench serve` CLI，并在当前进程的实测请求迭代器及请求完成处记录独立时间区间，
不修改已安装文件，不传入客户端 `--profile`。Agentic 使用原生实测阶段标记及完成记录。

控制器验证生成 token 持续增加后调用 `/start_profile`，等待窗口结束，再调用
`/stop_profile`。采集时长必须为正、不超过 180 秒，且不小于 `PROFILE_MIN_SECONDS`；
轮询间隔须小于延迟。Agentic 的延迟、采集、两次 API 等待预算、轮询和 90 秒
健康/指标预算必须能容纳在回放时长内。合成模式须提供足够请求覆盖窗口。
工作负载过早结束会明确失败，不会减少请求、重复工作负载或静默缩短采集。
请使用新的输出目录。

服务端会存活到导出结束，取消时也会在有限时间内尝试导出。
此采集关闭形状、调用栈及内存跟踪。带分析的结果应与无分析的吞吐对比结果分开保存。

| `RESULT_DIR` 下的输出 | 内容 |
| --- | --- |
| `vllm_core_command.txt` | 添加分析参数前的服务端命令 |
| `vllm_command.txt` | 实际服务端命令 |
| `<RESULT_FILENAME>-command.txt` 或 `agentic/benchmark_command.txt` | 所选客户端命令 |
| `profiles/torch/settings.json`、`profiler-config.json` | 采集及分析器设置 |
| `profiles/torch/measurement-start.json` | 合成工作负载的实测时间标记 |
| `profiles/torch/controller.log`、`profile-control.json` | 阶段、进度、启停及导出状态 |
| `profiles/torch/*.trace.json.gz` | worker trace，可能位于原生嵌套目录 |
| `profile-validation.json` | 八个 worker 的覆盖及实测区间包含关系 |
| `profiles/torch/kernels.csv`、`top10-kernels.csv` | 内核排序、调用次数、时间及 GPU 耗时占比 |

校验会完整解析稳定的导出文件，要求设备 0–7 均有非零内核，并拒绝不完整或超出
实测活动区间的采集。仅校验通过后写出 CSV。GPU 时间包含并行重叠活动，不能直接视为
端到端延迟占比或优化收益。HBM 计数器需要另行采集。

CPU 检查（Shell 测试替换外部 GPU、HTTP 和客户端依赖）：

```bash
python3 -m unittest discover -s upstream/InferenceX/benchmarks/profiling -p 'test_*.py'
PYTHONPATH=upstream/InferenceX python3 -m pytest upstream/InferenceX/utils/test_benchmark_serving.py
```

这些检查验证控制流程、默认值及覆盖值传递、标记位置和产物。本次更新没有运行 GPU
基准测试，也没有生成新的性能测量。

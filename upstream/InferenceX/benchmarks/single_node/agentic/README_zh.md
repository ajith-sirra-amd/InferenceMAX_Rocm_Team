# 单节点 AgentX 基准

[English](README.md) | **中文**

本目录为实验性 AgentX 基准启动脚本，结果并非 inferencex.com 的官方发布结果。
脚本用于验证 agentic-coding 场景，具体模型和配置仍需执行验证。

## DRAM KV 卸载策略

AgentX 配置使用 `kv-offloading` 指定资源层级（`none` 或 `dram`），
启用卸载时必须通过 `kv-offload-backend.name` 指定实现，如 `vllm-native`、
LMCache 或 Mooncake；仅有独立版本的软件后端需要 `version`。
AgentX 默认持续 3600 秒，调用工作流可显式覆盖。主配置必须使用精确的
`cluster:<name>` 标签。`configs/runners.yaml` 声明每节点可用内存及 GPU 数。

矩阵生成器按以下公式给出总预算：
`floor(min(available MiB, 2,861,022) * 1,048,576 * utilization * tp / gpus-per-node / 1,000,000,000)`。
内存上限为十进制 3 TB。基准脚本必须使用调用方传入的 `TOTAL_CPU_DRAM_GB`，
按实际 rank 数分配，不能用模型常量替代。DSv4 SGLang 仅提供
`--hicache-ratio`，由启动器将比例限制在预算之内。

## Kimi 定长对比

`kimik3_fp4_mi355x_mtp.sh` 显式固定 SA run 42638222682、job 129550719685
实际执行的 C48 吞吐配置：25 个引擎环境变量、TP8/DCP8、设备 0–7、10240
batched tokens、48 sequences，以及 DSpark 六个草稿 token、合成接受长度 3.57。
历史调优分支已删除。本地模型和草稿路径、端口由调用方提供。
`TOTAL_CPU_DRAM_GB=1799` 对应每 rank 224875000000 字节的原始卸载预算。
吞吐配置要求 `EVAL_ONLY=false`；质量评估需要自然验证配置。

`TEST=1` 在服务就绪后调用现有 `run_benchmark_serving`，替代 AgentX 回放。
显式传入 `TEST_ISL`、`TEST_OSL`、`TEST_RANGE_RATIO`、`TEST_NUM_PROMPTS` 和
`RESULT_FILENAME`。比例 1 请求固定长度；原生 helper 执行 `2 * CONC` 次预热，
保存详细指标及命令。`TEST=0` 要求调用方提供 `SA_CLIENT_ENTRYPOINT`，
在启动服务前检查已准备的客户端，服务就绪后再交给该入口执行回放。
团队调用方使用 SA 实测版本 `903a3023682c6b27683839309cbd4b4d9153afcb`
的原生 AgentX 客户端，并在使用 GPU 前准备固定版本的运行环境和数据集。
两种模式共用同一个服务命令。
可选的 `SA_CONFIG_AUDIT` 将实际命令与 `SA_REFERENCE_CONFIG` 比较，
不会设置环境变量或构建服务命令。参见团队的
[定长指南](../../../../../benchmarks/fixed_length/README.md)。
完整的先基线、后组合优化对比见团队
[AgentX 指南](../../../../../benchmarks/agentic/README.md)。

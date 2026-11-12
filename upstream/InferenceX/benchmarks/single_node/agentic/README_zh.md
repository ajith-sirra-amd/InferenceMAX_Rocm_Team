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

`kimik3_fp4_mi355x_mtp.sh` 在 `TEST=1` 时等待服务就绪后调用现有的
`run_benchmark_serving`，替代 AgentX 回放。显式传入 `TEST_ISL`、
`TEST_OSL`、`TEST_RANGE_RATIO`、`TEST_NUM_PROMPTS` 和 `RESULT_FILENAME`。
比例 1 请求固定长度；原生 helper 执行 `2 * CONC` 次预热，保存详细指标及命令。
`TEST=0` 保留回放。可选的 `SA_SERVER_CONFIG` 与 `SA_CONFIG_PREPARE`
在启动前载入冻结的服务环境和命令参数，绕过历史调优默认值。两种客户端
使用相同的服务生命周期。参见团队的
[定长指南](../../../../../benchmarks/fixed_length/README.md)。

# 原生 Kimi 并发扫描

[English](native-concurrency-sweep.md)

检查 C48 稳态 profile 后再运行以下点。使用根目录 e2e workflow、精确的原生 PR 版本、native comparison 模式、关闭 profiling，并运行完整 3,600 秒 replay。保留固定 e0d00e4b 镜像、TP8/DCP8、DRAM offload 和当前 CCD 绑定。

| 配置 key | 并发 | MTP | 计划 draft token 数 | Synthetic acceptance length | Scheduler chunk | 最大 sequence 数 |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| kimik3-fp4-mi355x-vllm-agentic-mtp-c56 | 56 | 启用 | 4 | 3.36 | 8192 | 112 |
| kimik3-fp4-mi355x-vllm-agentic-mtp-c64 | 64 | 启用 | 3 | 3.00 | 24576 | 80 |
| kimik3-fp4-mi355x-vllm-agentic-c96 | 96 | 关闭 | 0 | 不适用 | 24576 | 144 |

以上沿用现有 recipe 针对各并发值的配置，不是仅改变并发的对照实验。比较前需从每次 artifact 中核实实际服务命令。C48 配置保持不变。

原生 Kimi 的推测和非推测调度共用现有 recipe。launcher 转发 `SPEC_DECODING`；若 metadata 与实际 `--speculative-config` 参数不匹配，recipe 会报错。C96 不得包含推测配置。

每个点均需确认准备成功、完整测量 replay、全部八个 worker 的直接 FP8 投影和 cached FMHA 执行、指标覆盖，以及请求和服务错误。带 profiler 的吞吐量不能作为普通基线。目前尚无这些 sweep 的 GPU 结果。New-token FP8 和原生 NUMA 绑定继续推迟。

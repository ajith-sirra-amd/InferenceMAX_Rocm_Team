# NUMA/CCD 绑定上游贡献评估

[English](./numa-ccd-upstream-assessment.md) | **中文**

评估日期为 2026-10-28，依据 vLLM 基础版本 `a1cfd2c1b584c4ac197db9cc9ac84a6d973d39df`。尚未修改绑定行为，也未提交新的 PR。

## 现有支持和剩余机会

Kimi recipe 使用 `rocm-smi` 发现 GPU 所在 NUMA node，按共享 L3 cache 对 CPU 分组，为同一 NUMA node 的各 GPU 分配一个组，并在服务就绪后对 worker 线程执行 `taskset`。此代码设置 CPU affinity，本身并未设置 NUMA 内存分配策略。

vLLM 已有 `--numa-bind`、`--numa-bind-nodes` 和按可见 GPU 配置的 `--numa-bind-cpus`。multiprocessing wrapper 在 worker 启动前设置绑定，后续创建的线程继承该绑定。显式 CPU 列表使用 `--physcpubind` 并请求严格 `--membind`；自动绑定则使用软内存偏好。因此，用这些选项替换 recipe 的启动后 `taskset`，还会改变绑定时机及潜在内存策略，不能视为只改变 CPU affinity 的对照。

有意义的独立贡献可以复用现有绑定机制，为每个可见 AMD GPU 自动选择合适的 L3/CCD CPU 组。检查的实现已有 Intel PCT 选择策略，但未发现对应的 AMD CCD 选择器。应依据 cache 的 `level` 和 `type` 发现 L3 拓扑，而不是假设所有机器都使用 `index3`。共享 L3 分组只是拓扑输入；其与 CCD 的对应关系应在目标 CPU 上验证。

## 范围和所需证据

- 复用现有 CPU 列表解析、GPU/rank 映射、worker 启动和 affinity 报告。
- 尊重允许的 CPU 集合、cgroup、可见设备顺序、GPU 重映射及 TP/DP rank 映射。
- 明确 GPU 数量超过可用 L3 组或拓扑缺失时的行为，不猜测 CPU ID。
- 明确内存策略，避免无意改变 offload pool 的放置和内存容量。
- 在模型/通信初始化前及 worker 线程创建后验证 affinity，同时检查通信、offload 吞吐和 CPU 发射延迟。
- 先对比当前 recipe 与现有原生显式 CPU 列表绑定，再评估新的自动策略。凡声称隔离 CCD 选择收益的对照，都应尽可能保持 FP8 代码、工作负载、并发、MTP 和内存策略一致。
- 遵守用户不运行 serving smoke 的要求，以完整 agentic 运行提供性能证据。

本次源码评估不声称 CCD 性能提升。FP8 验证和并发对照期间保持当前绑定配置固定。

## 重复工作检查

开放的 [PR #55433](https://github.com/vllm-project/vllm/pull/55433) 已处理 ROCm 镜像缺少 `numactl` 的问题，不应重复提交该修复。NUMA 和 CCD 贡献搜索未发现等价的 AMD CCD 自动选择器；正式提出贡献前仍需再次检查。

来源：

- [Kimi recipe](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/blob/b4858ca17798690f913b02dc6d43f8c2c61b858d/upstream/InferenceX/benchmarks/single_node/agentic/kimik3_fp4_mi355x_mtp.sh)
- [vLLM NUMA 配置](https://github.com/vllm-project/vllm/blob/a1cfd2c1b584c4ac197db9cc9ac84a6d973d39df/vllm/config/parallel.py)
- [vLLM NUMA 工具](https://github.com/vllm-project/vllm/blob/a1cfd2c1b584c4ac197db9cc9ac84a6d973d39df/vllm/utils/numa_utils.py)
- [Worker 启动集成](https://github.com/vllm-project/vllm/blob/a1cfd2c1b584c4ac197db9cc9ac84a6d973d39df/vllm/v1/executor/multiproc_executor.py)

## 请求扩展并发测试（2026-10-29）

用户要求在 FP8 验证和稳定态 profiling 后，在 C56/C64 使用 MTP、C96 不使用
MTP 的测试中尝试原生 NUMA 绑定并关闭脚本 CCD pinning。这更新了此前在所有
后续并发实验中固定绑定方式的安排；当前 FP8 验证配置保持不变。

原生 ROCm 检测通过 `amdsmi_topo_get_numa_node_number` 获取每个可见 GPU 的
NUMA 节点。CPU 拓扑路径适用于 AMD CPU，可选 PCT 核心选择仅针对 Intel。
已合并 #61258 提供带本地内存软偏好的自动绑定，并明确说明 AMD serving
尚未验证。显式绑定仍使用严格内存策略。测量前检查 `numactl`（打包修复
#55433 尚未合并）、允许的 CPU/内存节点、实际 worker affinity 和分配策略。
不同绑定策略在相同并发、MTP、源码、权重和负载下比较。


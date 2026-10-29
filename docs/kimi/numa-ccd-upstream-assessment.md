# NUMA/CCD binding upstream assessment

**English** | [中文](./numa-ccd-upstream-assessment_zh.md)

Assessment as of 2026-10-28, against vLLM base `a1cfd2c1b584c4ac197db9cc9ac84a6d973d39df`. No binding change or new PR has been submitted.

## Existing support and the remaining opportunity

The Kimi recipe discovers GPU NUMA nodes with `rocm-smi`, groups CPUs by their shared L3 cache, assigns one group per GPU on each NUMA node, then uses `taskset` on worker threads after server readiness. This is CPU affinity; that code does not itself establish a NUMA memory-allocation policy.

vLLM already provides `--numa-bind`, `--numa-bind-nodes`, and per-visible-GPU `--numa-bind-cpus`. Its multiprocessing wrapper binds before the worker starts, so later worker threads inherit the binding. Explicit CPU lists use `--physcpubind` and request strict `--membind`; automatic binding uses soft memory preference. Replacing the recipe's late `taskset` with those options therefore also changes launch timing and potentially memory policy. It is not a clean CPU-affinity-only comparison.

A useful separate contribution could automatically select a suitable L3/CCD CPU group for each visible AMD GPU through the existing binding infrastructure. The inspected implementation contains an Intel PCT selection policy but no equivalent AMD CCD selector. L3 topology should be discovered through cache `level` and `type`, rather than assuming `index3` on every machine. Shared L3 groups are a topology input; their correspondence to CCDs should be verified on the target CPU.

## Scope and evidence needed

- Reuse existing CPU-list parsing, GPU/rank mapping, worker launch, and affinity reporting.
- Respect the allowed CPU set, cgroups, visible-device order, GPU remapping, and TP/DP rank mapping.
- Define behavior when GPUs outnumber available L3 groups, or topology is unavailable; do not invent CPU IDs.
- Keep memory policy explicit so offload-pool placement and memory capacity do not silently change.
- Verify affinity before model/communication initialization and after worker threads are created. Check communication and offload throughput as well as CPU launch latency.
- Compare the current recipe with existing native explicit CPU-list binding, then any new automatic policy. Keep FP8 code, workload, concurrency, MTP, and memory policy matched wherever the comparison claims to isolate CCD selection.
- Use full agentic runs for performance evidence, following the user's no-serving-smoke constraint.

No CCD uplift is claimed from this source review. Keep the current binding configuration fixed during the FP8 validation and concurrency comparisons.

## Duplicate-work check

Open [PR #55433](https://github.com/vllm-project/vllm/pull/55433) addresses missing `numactl` in ROCm images. Do not open another PR for that fix. Searches for NUMA and CCD contributions did not identify an equivalent automatic AMD CCD selector; repeat the check before proposing one.

Sources:

- [Kimi recipe](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/blob/b4858ca17798690f913b02dc6d43f8c2c61b858d/upstream/InferenceX/benchmarks/single_node/agentic/kimik3_fp4_mi355x_mtp.sh)
- [vLLM NUMA configuration](https://github.com/vllm-project/vllm/blob/a1cfd2c1b584c4ac197db9cc9ac84a6d973d39df/vllm/config/parallel.py)
- [vLLM NUMA utilities](https://github.com/vllm-project/vllm/blob/a1cfd2c1b584c4ac197db9cc9ac84a6d973d39df/vllm/utils/numa_utils.py)
- [Worker launch integration](https://github.com/vllm-project/vllm/blob/a1cfd2c1b584c4ac197db9cc9ac84a6d973d39df/vllm/v1/executor/multiproc_executor.py)

## Requested sweep extension (2026-10-29)

The user requested native NUMA binding without the recipe CCD pinning for
C56/C64 with MTP and C96 without MTP, after FP8 validation and steady-state
profiling. This supersedes the earlier instruction to keep binding fixed for
all later concurrency experiments; current FP8 validation stays fixed.

Native ROCm topology detection calls `amdsmi_topo_get_numa_node_number` for each
visible GPU. The CPU topology path is applicable to AMD CPUs; the optional PCT
core selection is Intel-specific. Merged #61258 provides best-effort automatic
binding with soft local-memory preference and explicitly reports AMD serving
as untested. Explicit binding retains strict memory placement. Check installed
`numactl` (open packaging fix #55433), allowed CPU/memory nodes, actual worker
affinity and allocation policy before measuring. Keep policy comparisons at
matched concurrency, MTP, source, weights and workload.


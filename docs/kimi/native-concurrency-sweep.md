# Native Kimi concurrency sweep

[中文](native-concurrency-sweep_zh.md)

Run these points after inspecting the C48 steady-state profile. Use the root e2e workflow with the exact native PR revision, native comparison mode, profiling disabled and a full 3,600-second replay. Keep the pinned e0d00e4b image, TP8/DCP8, DRAM offload and current CCD binding.

| Configuration key | Concurrency | MTP | Planned draft tokens | Synthetic acceptance length | Scheduler chunk | Maximum sequences |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| kimik3-fp4-mi355x-vllm-agentic-mtp-c56 | 56 | Enabled | 4 | 3.36 | 8192 | 112 |
| kimik3-fp4-mi355x-vllm-agentic-mtp-c64 | 64 | Enabled | 3 | 3.00 | 24576 | 80 |
| kimik3-fp4-mi355x-vllm-agentic-c96 | 96 | Disabled | 0 | Not applicable | 24576 | 144 |

These are the existing recipe's concurrency-specific settings, not a concurrency-only ablation. Confirm the actual server command in each artifact before comparing results. The C48 configuration is unchanged.

Native Kimi uses the existing common recipe for both speculative and non-speculative schedules. The launcher forwards `SPEC_DECODING`; the recipe rejects a mismatch between that metadata and its `--speculative-config` arguments. C96 must have no speculative configuration.

Each point needs successful setup, a complete measured replay, all-eight-worker direct FP8 projection/cached-FMHA proof, metric coverage, and request/server error inspection. Do not treat profiled throughput as an unprofiled baseline. No sweep GPU result is claimed yet. New-token FP8 and native NUMA binding remain deferred.

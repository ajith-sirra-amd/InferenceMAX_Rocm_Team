# Kimi-K3 文档

[English](README.md) | **中文**

- 当前基线：run39239112271，Moonshot Kimi-K3，15,476.47 total tokens/sec/GPU，20,951,355 KV tokens。
- 容量参考：run39601846940，AMD Quark Kimi-K3，23,098,919 KV tokens。
- 先阅读当前开销分析；旧交接记录对应历史配置。
- 代码和原始证据保留在原路径。

| 文档 | 主题 |
|---|---|
| [基线证据](atom-attention-BASELINES.md) | 已核实的模型、吞吐和容量 |
| [ATOM 注意力](atom-flydsl-attention_zh.md) | 适配器、内核验证和限制 |
| [KV 流程](kimi-kv-optimized-flow_zh.md) | 潜在展开、精度和通信 |

- [当前开销审计](kimi-prefill-overhead-audit_zh.md)：问题、精度与矩阵流程、实测开销、下一次测试门槛。

- [ATOM 对比与下一步候选](atom-vllm-opportunities-2026-10-28_zh.md)：已验证差异、基线对照和实验建议。

## 全部记录

- [39726650098-validation.md](39726650098-validation.md)
- [ATOM-producer-fusions.md](ATOM-producer-fusions.md)
- [ATOM-producer-fusions_zh.md](ATOM-producer-fusions_zh.md)
- [CHANGED-FILES.md](CHANGED-FILES.md)
- [DCP-TRIALS-LOG.md](DCP-TRIALS-LOG.md)
- [EXPERIMENT-QUEUE.md](EXPERIMENT-QUEUE.md)
- [HANDOFF-2026-09-09.md](HANDOFF-2026-09-09.md)
- [HANDOFF-FMHA-PR54494.md](HANDOFF-FMHA-PR54494.md)
- [HANDOFF.md](HANDOFF.md)
- [IMAGE-RECIPE.md](IMAGE-RECIPE.md)
- [KIMI_vLLM_vs_Atom.md](KIMI_vLLM_vs_Atom.md)
- [Kimi-DCP-Experiemnts-Summary.md](Kimi-DCP-Experiemnts-Summary.md)
- [Kimi-K3-Concurrency-Sweep.md](Kimi-K3-Concurrency-Sweep.md)
- [Kimi-K3-Where-The-Time-Goes.md](Kimi-K3-Where-The-Time-Goes.md)
- [Kimi-vLLM-Learnings.md](Kimi-vLLM-Learnings.md)
- [LMCACHE.md](LMCACHE.md)
- [NEXT-PERF-LEVERS-POST-54494.md](NEXT-PERF-LEVERS-POST-54494.md)
- [OFFLOAD-LOAD-RACE-UPSTREAM-REPORT.md](OFFLOAD-LOAD-RACE-UPSTREAM-REPORT.md)
- [OVERLAY-ABLATION.md](OVERLAY-ABLATION.md)
- [PR54494-ROOT-CAUSE-AND-FIX.md](PR54494-ROOT-CAUSE-AND-FIX.md)
- [REPRODUCE.md](REPRODUCE.md)
- [SESSION-HANDOFF.md](SESSION-HANDOFF.md)
- [UPSTREAM-STATUS.md](UPSTREAM-STATUS.md)
- [atom-attention-BASELINES.md](atom-attention-BASELINES.md)
- [atom-flydsl-attention.md](atom-flydsl-attention.md)
- [atom-flydsl-attention_zh.md](atom-flydsl-attention_zh.md)
- [atom-producer-fp8-NATIVE-QUANT-RUN.md](atom-producer-fp8-NATIVE-QUANT-RUN.md)
- [atom-producer-fp8-RUNNER-TERMINATION.md](atom-producer-fp8-RUNNER-TERMINATION.md)
- [attnfp8-smoke-39245144606-failure-analysis.md](attnfp8-smoke-39245144606-failure-analysis.md)
- [attnfp8-smoke-39245144606-session-log.md](attnfp8-smoke-39245144606-session-log.md)
- [attnfp8-smoke-39260167917-verification.md](attnfp8-smoke-39260167917-verification.md)
- [baseline-kernel-profile.md](baseline-kernel-profile.md)
- [baseline-kernel-profile_zh.md](baseline-kernel-profile_zh.md)
- [kimi-attnfp8-smoke.md](kimi-attnfp8-smoke.md)
- [kimi-attnfp8-smoke_zh.md](kimi-attnfp8-smoke_zh.md)
- [kimi-fp8-kv-allgather.md](kimi-fp8-kv-allgather.md)
- [kimi-fp8-kv-allgather_zh.md](kimi-fp8-kv-allgather_zh.md)
- [kimi-kv-optimized-flow.md](kimi-kv-optimized-flow.md)
- [kimi-kv-optimized-flow_zh.md](kimi-kv-optimized-flow_zh.md)
- [profiles-readme.md](profiles-readme.md)

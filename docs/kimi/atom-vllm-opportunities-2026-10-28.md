# ATOM Kimi review and vLLM performance candidates — 2026-10-28

[中文](atom-vllm-opportunities-2026-10-28_zh.md) · [Current audit](kimi-prefill-overhead-audit.md) · [Baselines](atom-attention-BASELINES.md)

## Scope and current evidence

Reviewed the repository's Kimi ATOM launch script, ATOM source at `661ef7661eef0d804d2b52d81ed41d80aeaf9fdb`, and the serving vLLM source at `1b2a666391f2a24228a1436b64d547c34cc6b6ce`. ATOM source capability is not evidence that a historical image enabled that path.

The hybrid vLLM prefix smoke40643563020 passed 20/20 requests and all-eight-worker path checks. Full AgentX run40646139769 is being monitored; its throughput result is pending. Keep Moonshot, the existing perf branch, warmup10/lane,3600seconds and no GSM8K.

**Important correction:** ATOM's `ATOM_USE_FLYDSL_FP8_PREFILL_ATTN` defaults to **0** at the inspected revision; this Kimi script does not set it. Its FlyDSL gather switch defaults to1. FP8 FMHA exists as an opt-in capability; do not attribute an ATOM benchmark's throughput to it without the image environment and actual invocation evidence.

## Differences worth testing

| Candidate | ATOM evidence | Current vLLM evidence | Next useful action |
|---|---|---|---|
| C48 without speculative decoding | C48/56/72 explicitly use no draft | C48 uses DSpark k4, also confirmed in the primary baseline | A controlled C48 no-draft ablation can test whether draft/verification cost still pays at the resident batch size. Its gain is unknown. |
| Dedicated cached-KV quantizer | Attention dispatch uses a dedicated contiguous K/V quantizer after Q has been prepared | Our copied module contains `fused_kv_per_tensor_quant`, but the adapter and timing probe use generic QKV quantization with `SKIP_Q` | Benchmark the two implementations at real chunk sizes, then consider the dedicated route only for contiguous K/V; retain the generic strided route. |
| Producer-plus-quantization fusion | SiTU→down_proj, sigmoid gate→o_proj, Q-normalization→q_b_proj, and strided KDA f_a→f_b_proj can forward quantized data/scales | Pinned model call sites still expose separate activation and FP8-consumer boundaries. Native input AttnRes+quant already exists, so do not duplicate it. | Start with one producer and the actual FP8 consumer interface; compare complete producer+GEMM time and numerical outputs. Earlier repository producer experiments are useful evidence, not a serving validation of this candidate. |
| Expanded FP8 K/V directly from reconstruction | Supported FlyDSL gather epilogue returns K8/V8 and descales to FMHA | Native fused gather communicates compact FP8 but outputs BF16 expanded K/V, then requantizes | Larger kernel integration: prove installed API, weight layout/scales, row mapping, cached-value scaling and numerical behavior before serving. |
| Mixed/prefill graph coverage | ATOM recipe requests FULL | vLLM resolves FULL_AND_PIECEWISE to FULL_DECODE_ONLY because there is no compiled submodule and breakable graphs are disabled | Investigate supported breakable/piecewise execution. Simply copying FULL does not establish backend support or a gain. |
| CPU offload locality | ATOM binds workers and preferred NUMA memory before allocation; LMCache NUMA is enabled | Native offload allocates and zeroes CPU tensors, then pins them. Recipe CCD binding happens after readiness; smoke pinned1562threads. | Inspect actual page placement before changing it; bind to measured GPU-local topology before allocation if placement is wrong. Do not copy another host's rank-to-socket list. |
| Recurrent-state storage/checkpoint policy | Script selects FP16 state, disables demand checkpoints and uses prompt-end anchors | Pinned Kimi state getters use FP32 recurrent state; hybrid planning resolves1536-token attention blocks | Architectural candidate requiring state-kernel and checkpoint validation. The pinned getters omit the SSM-dtype override, so a CLI flag alone does not reproduce ATOM. |
| Graph-memory and cache grouping overhead | Different state/cache organization | Final smoke graph capture6.67GiB/GPU; earlier sizing capture9.20–9.64GiB. Planner pads Mamba pages3.97% and reports an upper-bound grouping loss. | Measure actual allocations and graph dispatch coverage. Do not call the planner's26.09% bound a measured global loss or disable memory accounting to manufacture capacity. |

The current large-chunk diagnostic measures roughly58.7µs for QKV preparation,52.9µs for K/V preparation with reused Q, and51.6µs for FP8 FMHA alone. These costs motivate reducing preparation. They do not measure the full-model time share.

Direct FP8 reconstruction is not an interchangeable replacement for the current BF16-weight native fused path: the inspected FlyDSL implementation gates FP8-weight/scaled layouts and supported output views. Its cached output scales also need validation; a scale derived from new tokens is not a universal bound for cached tensors.

The recipes both request TP8/DCP8 at C48,8192 batched tokens,96 sequence seats,FP8 KV cache,AITER SiTUv2/stage2 settings and broad online FP8 projection/shared-expert weights. Both request INT4 quick reduction, but vLLM explicitly sets a16MiB minimum; matching the quantization name does not establish identical size thresholds or routing. Check actual collective message sizes before changing that threshold. ATOM communicator reuse is enabled only from C56 in this script, so it is not a missing C48 toggle.

## Compare against the actual baseline

The primary baseline39239112271 used DSpark k4, FULL_DECODE_ONLY,1536-token blocks and the same padding warning. These shared costs cannot by themselves explain a difference between it and the hybrid run.

| Runtime evidence | Baseline39239112271 | Hybrid prefix smoke40643563020 |
|---|---:|---:|
| Loaded model memory/GPU |187.43GiB|184.08GiB|
| GPU KV tokens |20,951,355|22,665,373|
| Final graph capture/GPU |6.96GiB|6.67GiB|
| Model-load time |239.5–240.4s|439.4–440.3s|

The smoke is an invocation test, not a full throughput comparison. Loading time is outside the measurement window. Baseline raw records contain **101 successful warmups/430 warmup errors**, and **3357 successful measured requests/6 measured errors**. Do not classify its436 total error records as436 benchmark-phase failures.

Legacy aggregate JSON says `spec_decoding: none` even when the server log proves DSpark k4; use server commands/engine configuration for speculation provenance.

## Decision after the full run

1. Confirm the tested checkout, model, actual command, warmup10,3600-second profile and all required paths.
2. Compare total **and output** tokens/sec/GPU against15476.46881 and103.05251, plus TTFT/ITL, completed requests, phase-specific errors, GPU/CPU hit rates, actual KV pool and reporting duration.
3. If healthy, select one isolated next experiment. C48 no-draft is the simplest ATOM-inspired system ablation; the dedicated K/V quantizer is a narrow code candidate after numerical and complete-call timing checks.
4. If request errors or runtime failure dominate, diagnose those before ranking a throughput gain. If the workload mix/cache hits differ materially, report that limitation.
5. Do not combine speculation, NUMA, graph and kernel changes in one comparison. Fullrun40646139769 is already dispatched.

The recipe currently rejects `SPEC_K=0`; a no-draft experiment needs an explicit caller/recipe path that omits speculative configuration and adjusts graph sizing. It is not a supported zero-token flag today.

## Source anchors

- [ATOM recipe](../../upstream/InferenceX/benchmarks/single_node/agentic/kimik3_fp4_mi355x_atom_mtp.sh), especially concurrency selection, NUMA setup, state settings and server arguments.
- [Current vLLM recipe](../../upstream/InferenceX/benchmarks/single_node/agentic/kimik3_fp4_mi355x_mtp.sh).
- [ATOM model producer fusions](https://github.com/ROCm/ATOM/blob/661ef7661eef0d804d2b52d81ed41d80aeaf9fdb/atom/models/kimi_k3.py).
- [ATOM MLA dispatch and gather epilogue](https://github.com/ROCm/ATOM/blob/661ef7661eef0d804d2b52d81ed41d80aeaf9fdb/atom/model_ops/attention_mla.py#L1681).
- [ATOM environment defaults](https://github.com/ROCm/ATOM/blob/661ef7661eef0d804d2b52d81ed41d80aeaf9fdb/atom/utils/envs.py#L346).
- [Pinned vLLM model](https://github.com/vllm-project/vllm/blob/1b2a666391f2a24228a1436b64d547c34cc6b6ce/vllm/models/kimi_k3/amd/linear.py).
- [Pinned vLLM offload allocation](https://github.com/vllm-project/vllm/blob/1b2a666391f2a24228a1436b64d547c34cc6b6ce/vllm/v1/simple_kv_offload/worker.py#L239).

# PR #64288: code paths, upstream history and regression audit

Audit snapshot: 2026-11-05. PR remains **Draft**, tested head `a03796b8e3bfe536e22f9db741109dbf80170079`. No GPU work was started during this audit: the reservation ended, and the earlier cleanup completed.

## Findings

1. **The tested OFF arm does not already implement our complete path.** Both arms use FP8 latent KV storage and native per-channel FP8 projection weights. OFF dequantizes the gathered latent, calls the native dynamically quantized FP8 linear, writes expanded BF16 K/V, and runs cached-context BF16 AITER attention. ON gathers FP8 bytes, derives cached-data output bounds, projects directly into FP8 K/V, and calls FlyDSL FP8 attention.
2. **Much of the historical improvement is outside the attention flag.** The original run online-quantized only 69 KDA projection layers. The later recipe also quantizes attention and shared-expert projections. The current OFF comparison retains those optimizations.
3. **A mixed forward changes more than eligible chunks.** At the tested head, if any chunk qualifies, smaller chunks use our fused projection with BF16 output instead of OFF's native linear path. This is a verified code difference, not yet a proven cause of regression.
4. **No stage profile identifies a single root cause yet.** Safe scale preparation, shape-dependent FMHA scheduling, mixed-chunk routing and compilation/measurement effects must be separated. Existing measurements support a full-agentic gain and synthetic-serving regressions; neither result should be generalized.
5. **Latest upstream adds overlapping attention functionality, but still no direct FP8 projection output.** #59124 is opt-in, uses expanded BF16 K/V followed by quantization, and merged after the tested base. It cannot explain the measured OFF result.

## Runtime provenance

| Runtime | Revision / evidence |
| --- | --- |
| Original and prototype | `1b2a666391f2a24228a1436b64d547c34cc6b6ce` |
| Earlier native | source `492080779`, compiled base `a1cfd2c1b584c4ac197db9cc9ac84a6d973d39df` |
| Tested current | source `a03796b8e3bfe536e22f9db741109dbf80170079`, compiled base `69d27bf8002a4634de1699c95fb0aeb22fb86ca5` |
| Prepared main integration | `54f39769d0e17a8f2874695e849c4e6626e71b56` on `ee85faa1280c3f9ef877c7b17d1a3ef113133e57`; not GPU validated |
| Latest main inspected | `14d147a7a05d29d011a67ed1f1f819c03546dafe`; not the tested nightly runtime |

Pinned tested image:
`vllm/vllm-openai-rocm:nightly-rocm100-e0d00e4b8eb0c89ec30921d0695cfdd409e9c211@sha256:b1374c3814b6e7706c49968ba17ca3b35484382f7b9fc7115dd3e66068d53f2f`.

Identical image tags/digests do not imply identical installed source and binaries. The serving harness checks out each entire source revision and starts a fresh process. Its editable-install version string can still display the original installation's version after checkout; the recorded git SHA and source hashes establish the arm, not that string.

Artifact `13415946292` records Torch 2.13.0+rocm10.0.0, HIP 7.15.26333, Triton 3.8.0+git669b31ac and exact vLLM wheel URLs. It does not record an exact AITER source hash. The image build can select a nightly AITER wheel independently; a vLLM AITER-pin commit alone is not proof that a particular AITER implementation executed.

## Actual routing at a037 / 69d27

| Operation | OFF | ON |
| --- | --- | --- |
| Persistent compressed KV | FP8, 576 values per token | Same |
| Projection weights | Native FP8 per channel | Views of the same native storage; no extra weight copy |
| Eligible cached gather / DCP communication | Dequantized BF16 latent workspace / gather | FP8 bytes / gather |
| Cached projection | Native FP8 linear, dynamic activation quantization, BF16 expanded output | FP8 operands, FP32 accumulation, direct FP8 expanded output |
| Eligible cached FMHA | BF16 Q/K/V AITER FlashAttention | Dynamically quantized FP8 Q, direct FP8 K/V, FlyDSL FMHA |
| Entire forward below cutoff | Native gather / linear / BF16 attention | Same routing |
| Small chunk within a forward that has an eligible chunk | Native gather / linear / BF16 attention | Custom fused projection emitting BF16, then BF16 attention |
| Context-free prefill | Compatible native FP8 ASM route | Same route preserved |
| New-token portion of a cached prefill | Common BF16 causal backend | Same in this tested revision |
| Output and context merge | BF16 attention output; FP32 natural-log LSE | Same contract |

Thus “OFF already uses FP8” is true for weights, cache storage and eligible **uncached ASM** attention, but false for the full **cached-context direct-output** optimization.

Source:
- [Native DCP gather and projection](https://github.com/ajith-sirra-amd/vllm/blob/a03796b8e3bfe536e22f9db741109dbf80170079/vllm/model_executor/layers/attention/mla_attention.py#L3094)
- [ON scope and routing](https://github.com/ajith-sirra-amd/vllm/blob/a03796b8e3bfe536e22f9db741109dbf80170079/vllm/v1/attention/backends/mla/rocm_aiter_mla.py#L2950)
- [ON per-chunk loop](https://github.com/ajith-sirra-amd/vllm/blob/a03796b8e3bfe536e22f9db741109dbf80170079/vllm/v1/attention/ops/rocm_aiter_mla_fp8_prefill.py#L231)
- [BF16 cached attention call](https://github.com/ajith-sirra-amd/vllm/blob/a03796b8e3bfe536e22f9db741109dbf80170079/vllm/v1/attention/backends/mla/prefill/aiter_flash_attn.py#L135)

## Historical recipe changes verified in server artifacts

| Run | Total tokens/s/GPU | Online-quantized target-model layers | Meaning |
| --- | ---: | ---: | --- |
| [39239112271](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/39239112271) | 15,476.46881 | 69 | Only `*in_proj_qkvgfab`; not the current OFF recipe |
| [40646139769](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/40646139769) | 16,915.72154 | 489 | Broad additional weight quantization; BF16 kv_b_proj output feeding prototype FP8 FMHA |
| [40695099743](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/40695099743) | 17,406.08912 | 513 | Adds 24 FP8 kv_b_proj layers and direct FP8 output |
| [41447686785](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/41447686785) | 18,276.23099 | 489 | Earlier native feature-enabled run; 24 separate gate projections fused upstream |
| Current matched OFF / ON | 18,266.89472 / 19,000.97537 | 489 / 489 | Same broad weight configuration, attention flag isolated |

Artifact IDs re-read for the first four server logs: `12191793463`, `12802601010`, `12825383260`, `13154932461`. All report load format `auto`. Original and later commands keep GMU 0.90, TP8/DCP8, 8192 batched tokens and max sequences 96.

The layer-count change 513 to 489 is consistent with #59634 folding 24 gate projections into the fused QKV projection; it is not removal of FP8 kv_b_proj. No matched weight-only/runtime-only ablation exists to divide the historical throughput gain into exact percentages.

## Upstream changes

All pages were collected: **1,291 commits** prototype to tested base, **547 commits** earlier native base to tested base, and **119 commits** tested base to the latest-main snapshot.

Full revision ranges:
- [Prototype to tested base](https://github.com/vllm-project/vllm/compare/1b2a666391f2a24228a1436b64d547c34cc6b6ce...69d27bf8002a4634de1699c95fb0aeb22fb86ca5)
- [Earlier native base to tested base](https://github.com/vllm-project/vllm/compare/a1cfd2c1b584c4ac197db9cc9ac84a6d973d39df...69d27bf8002a4634de1699c95fb0aeb22fb86ca5)
- [Tested base to latest-main snapshot](https://github.com/vllm-project/vllm/compare/69d27bf8002a4634de1699c95fb0aeb22fb86ca5...14d147a7a05d29d011a67ed1f1f819c03546dafe)

Relevant changes below were checked against diffs and recipe guards. “Applicable” establishes execution relevance, not a measured contribution to throughput.

| Merged PR | Change | Relevance |
| --- | --- | --- |
| [#59634](https://github.com/vllm-project/vllm/pull/59634), Oct 20 | Fused MLA gate/QKV projection; fused uncached FP8 preparation and head padding | Applicable model optimization after prototype; present before the earlier native base |
| [#63342](https://github.com/vllm-project/vllm/pull/63342), Oct 27 | KDA prefill convolution block_m 8 to 128 with matching metadata | Applicable prefill optimization; present before earlier native base |
| [#53166](https://github.com/vllm-project/vllm/pull/53166), Oct 27 | Fused cached gather/up-projection | BF16/FP16 **weight** guard excludes current FP8-weight recipe; does not make OFF identical |
| [#60110](https://github.com/vllm-project/vllm/pull/60110), Oct 27 | Fold q_b_proj and W_UK for decode | Requires unquantized q_b_proj; current FP8 q_b_proj excludes it |
| [#62371](https://github.com/vllm-project/vllm/pull/62371), Oct 19 | AITER static quantization into MLA output | Recorded `fuse_attn_quant=False`; cannot simply credit this recipe |
| [#62444](https://github.com/vllm-project/vllm/pull/62444), Oct 29 | Refresh eager CPU-cache LRU on GPU prefix hits | Applicable to full replay with eager SimpleCPUOffload; absent from no-offload serving matrix |
| [#59322](https://github.com/vllm-project/vllm/pull/59322), Oct 29 | Avoid repeated offload hash scans | Changes OffloadingConnector scheduler, not this SimpleCPUOffload implementation |
| [#64008](https://github.com/vllm-project/vllm/pull/64008), Oct 29 | Cache compiled FlyDSL host launchers | Touches specialized decode/other attention paths; not our normal FlyDSL FMHA adapter |
| [#63297](https://github.com/vllm-project/vllm/pull/63297), Oct 31 | Retain KDA prefix checkpoints with Gluon; use fused path for checkpoint steps | Changes cache reuse, KDA prefill execution and per-request capacity accounting |
| [#61950](https://github.com/vllm-project/vllm/pull/61950), Nov 2 | Correct loading/zeroing of online-quantized padded KDA projection | Applicable correctness change; not evidence of a quantified speedup |
| [#56492](https://github.com/vllm-project/vllm/pull/56492), Nov 2; [#65418](https://github.com/vllm-project/vllm/pull/65418), Nov 3 | Post-warmup extensible cache sizing; larger ROCm safety margin | Explains changed memory budgeting, not FP8 feature memory savings |
| [#61435](https://github.com/vllm-project/vllm/pull/61435), Nov 3; [#64029](https://github.com/vllm-project/vllm/pull/64029), Nov 2 | Specialized Kimi decode/M1a/M1b and persistent kernels | Explicit latency-specialized decode path, not selected by the tested command |
| [#63730](https://github.com/vllm-project/vllm/pull/63730), Nov 5 08:43 UTC | Kimi tuned-GEMM recipe/online latent projection support and optional warmup | **After tested base**; cannot explain current results |
| [#59124](https://github.com/vllm-project/vllm/pull/59124), Nov 5 09:21 UTC | Explicit `ROCM_AITER_FLYDSL_FP8` prefill backend, Q/K/V quantization and Q reuse | **After tested base**; expanded BF16 K/V still materialized; no direct FP8 projection |
| [#62276](https://github.com/vllm-project/vllm/pull/62276), Oct 19 | ROCm 10 image Torch 2.13 update | Stack change between generations; exact installed versions matter |

Do not equate every merged “Kimi/ROCm performance” title with an enabled optimization. Latest main is also not proof that a compatible nightly wheel/image has been published or tested.

## Measured regression and limits

The matched full C48+MTP agentic replay improved **4.02% total throughput** and **4.49% output throughput**. This is one pair with synthetic acceptance 3.36; request errors/cutoff cancellations remain documented in the PR.

The separate random-prompt serving matrix uses live MTP, 128 output tokens, 64K max length and no CPU offload:

| Eligible case | OFF output tokens/s | ON output tokens/s | Throughput change | Mean TTFT OFF / ON |
| --- | ---: | ---: | ---: | ---: |
| 16K prefix, C8 | 23.83573 | 22.11723 | -7.21% | 11.1622 / 11.0072 s |
| 16K prefix, C48 | 33.89550 | 32.34014 | -4.59% | 35.1491 / 37.9162 s |
| 32K prefix, C48 | 19.61760 | 18.81913 | -4.07% | 58.6498 / 65.2935 s |

Rates here are whole-server **output** tokens/s, not full-replay input-plus-output tokens/s/GPU. C8 TTFT did not regress; its TPOT did. Mean C48 TTFT increased 7.87% / 11.33%.

Evidence: [42006336780](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/42006336780), artifact `13415946292`. First observed eligible shapes: Q512/KV16896 and Q1024/KV33792. These are observations, not a frequency distribution. Smaller BF16 chunks were also observed; the logs do not establish how often a mixed forward occurred.

Potential contributors and required discrimination:

- Scale preparation performs two launches and scans selected cached rows; native OFF also has activation quantization, so profile **both full pipelines**.
- Existing standalone preparation timings are about 44–45 microseconds at sampled shapes, including launch overhead. Removing runtime-count specialization reduced first-call costs but produced no steady-state gain. That candidate remains separate.
- FP8 FMHA tile height and KV splits depend on sequence lengths, heads and batch. Current AITER source exposes these choices; the tested artifact lacks the exact AITER hash, so do not present current-main tuning choices as measured dispatch.
- ON changes small-chunk projection inside mixed forwards. The isolated candidate below removes that confound.
- The harness performs a phase-file read at every common prefill entry. It is shared across arms, but is not free; ON also has additional observed wrappers. Future performance timing should disable the observer entirely and collect routing proof separately.
- One warmup per case did not cover every scale-kernel specialization: ON variants grew during the matrix. Warm every measured shape, then confirm no JIT growth before collecting steady-state latency.
- Below-cutoff cases varied by +3.70% and +14.40% despite retaining BF16. Repeat counterbalanced arm orders; do not attribute that variation to FP8.
- Recorded MTP acceptance was close across arms. This makes it a weaker current suspect, but the logged windows include warmup and are not exact benchmark-only counters.

## ATOM comparison and numerical boundary

Inspected ATOM `atom/model_ops/attention_mla.py`, blob `9f63654e2405bfd66f1a302f10b60734559f50c3`, especially `_forward_prefill_chunked`: it computes new-token K/V, calls `fused_qkv_per_tensor_quant`, and reuses `gather_ks/gather_vs` for direct cached projection. This avoids a cached-data scan. New-token statistics alone cannot bound arbitrary cached outliers.

Our cached-data bound uses latent row norms, native weight row norms and RoPE maxima. It passed the earlier 8x/32x outlier tests. Keep that numerical protection. A future optimization can fuse bound collection into cache gathering, but must retain the bound over all selected rows and consistent DCP scale handling. Do not restore new-token-only scale estimates just to reproduce prototype speed.

## Concrete candidate and next decision

Prepared isolated fork candidate:
[9840536e6d53ea1552be8c2da5950d29b7377924](https://github.com/ajith-sirra-amd/vllm/commit/9840536e6d53ea1552be8c2da5950d29b7377924),
branch `perf/fp8-native-chunk-fallback-20261105`, parent a037.

- Extract the existing native DCP gather/linear/packing body into a reusable per-chunk helper.
- Route small chunks in mixed forwards through that helper, before FP8-byte gathering.
- Keep eligible FP8 projection, quantization, thresholds and attention unchanged.
- Keep the main PR head unchanged. This is a routing ablation, with no measured performance claim.
- CPU structural and lint validation is prepared in `.github/workflows/pr64288-code-audit.yml`; GPU execution, graph replay and accuracy remain pending.

Recommended solution direction is **select the fastest validated full pipeline by workload shape**, retaining native fallback. The current two max-length thresholds ignore total rows, request count, raggedness and split-K overhead. Do not hardcode new thresholds from one noisy serving pass.

Next GPU sequence, only after a new reservation:
1. Capture full shape/routing counts separately; record AITER/FlyDSL source hashes and resolved kernel tile/split settings. Include tail chunks, ragged lengths, zero-context rows and all eight ranks.
2. At those exact shapes compare native OFF, direct projection with BF16 output/FMHA, and direct FP8 output/FMHA. Time gathering/communication, scales/Q, projection, FMHA and merging; also time the complete pipeline. Do not sum isolated timings as a substitute for pipeline measurement.
3. Test the native-fallback candidate against a037, both with flag OFF and ON. Require unchanged OFF outputs and eligible FP8 results, correct mixed/empty-chunk merging, graph replay and unchanged numerical tolerances.
4. Run observer-free, fully warmed serving pairs in alternating OFF/ON then ON/OFF order for C8 and C48 cases. Preserve GMU 0.90, load format auto, quantization, graph settings and dataset. Report variability, throughput, TTFT, TPOT and errors.
5. Only after identifying the limiting stage, implement the appropriate kernel/dispatch change. Validate GSM8K, cached-path NIAH and the full C48+MTP replay. Broader concurrency sweeps follow the PR gates.

There is no validated fix that guarantees higher throughput across every configuration yet. A defensible result is better eligible-case throughput with measured native fallback for cases that do not amortize the FP8 overhead.

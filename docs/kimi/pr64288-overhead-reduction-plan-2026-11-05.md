# FP8 MLA overhead reduction: lessons from established implementations

Date: 2026-11-05  
Status: proposed modifications and validation plan; no new GPU throughput result  
Scope: Kimi-K3 on eight MI355X, TP8/DCP8, PR #64288

## Recommendation

Yes: established ATOM, AITER and vLLM techniques offer credible ways to reduce overhead and potentially outperform the current baseline. The first new implementation candidate should be **reuse of expanded K/V and temporary buffers**, followed by measured reduction of adapter launch overhead.

The target is lower **complete cached-context pipeline time**, including gathering, communication, scales, projection, attention and merging. FP8 arithmetic alone does not establish a win. Preserve native fallback wherever the optimized path does not amortize its overhead.

The [code and upstream-history audit](pr64288-regression-code-audit-2026-11-05.md) contains the full routing analysis, historical recipe differences and merged-PR review.

## What the current comparison actually measures

Tested PR source: `a03796b8e3bfe536e22f9db741109dbf80170079`.  
Compiled vLLM base: `69d27bf8002a4634de1699c95fb0aeb22fb86ca5`.

| Stage | Flag OFF | Flag ON |
| --- | --- | --- |
| Persistent latent KV and kv_b_proj weights | FP8 | Same FP8 storage |
| Cached latent gathering / DCP communication | BF16 after cache dequantization | FP8 bytes |
| Projection arithmetic | Native FP8 linear, after dynamic activation quantization | FP8 fused projection |
| Expanded K/V output | BF16 | Direct FP8 for eligible chunks |
| Eligible cached FMHA | BF16 Q/K/V | FP8 Q/K/V |
| Attention output / merge contract | BF16 output, FP32 natural-log LSE | Same contract |

Our distinct opportunity is avoiding the latent conversion round trip, reducing communication/expanded-output traffic and improving cached attention. Both arms already benefit from FP8 weights; there is no additional weight-memory saving from the attention flag.

The matched full C48+MTP replay improved total throughput by 4.02%. The separate synthetic serving matrix regressed by 4.07–7.21% in its eligible cached cases. These are different workloads. Neither establishes performance across all configurations.

## Established techniques: what is new and what is already present

| Technique | Reference | Current state | Decision |
| --- | --- | --- | --- |
| Reuse K/V output workspace across cached chunks | ATOM supplies slices of shared K/V workspaces to the gather/project kernel | Our `project()` allocates K and V tensors on every call | First new candidate |
| Reuse reduction scratch and output-scale storage | Established workspace ownership pattern; extend it to our cache-bound preparation | Our preparation allocates partials and scale tensors per call | Combine with K/V workspace candidate |
| Quantize Q once and reuse it | ATOM and upstream #59124 | Already implemented for matching query slices in `FP8PrefillContext.queries` | Preserve; do not claim a new saving |
| Fuse preparation operations into fewer launches | ATOM fused quantization; upstream fused prefill preparation | Our Q/cache preparation already uses two fused launches | Further fusion requires identifying a remaining dependency-safe opportunity |
| Cache compiled launchers | AITER `_run_compiled` and cached kernel builders | Present in the inspected AITER projection wrapper | Verify the installed version; do not add a duplicate cache |
| Keep weight views and fixed norms | Our native weight adapter | Already uses views; weight norms are computed at load time | Preserve; no second weight copy or repacking |
| Avoid specialization on changing counts | Existing isolated runtime-count candidate | Reduced sampled first-call totals from 5.69 s to 1.95 s; no steady-state gain | Treat as a cold/JIT-cost option, not a proven throughput improvement |
| Choose kernels for the actual workload shape | AITER tile/split selection; native fallbacks | Our outer eligibility uses only maximum query/context lengths | Replace only after measuring complete-pipeline crossover |

Buffer reuse does not imply that every current `torch.empty` calls the GPU allocator directly: PyTorch already caches device allocations. The expected opportunity is fewer Python/allocator bookkeeping operations and better control of temporary lifetimes. It must be measured.

## Execution order

### 1. Validate the prepared native-fallback ablation

At a037, a forward with any eligible FP8 chunk enters our context loop. Smaller chunks within that forward then use our fused projection with BF16 output, rather than the OFF native projection sequence.

A separate [candidate at 436f902](https://github.com/ajith-sirra-amd/vllm/commit/436f9022967b26f4cb19f712063a63defe6a6cff) extracts the native gather/project/pack helper and uses it for those smaller chunks. Eligible FP8 operations remain unchanged.

[CPU syntax, structural comparison, lint and formatting passed](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/42043465344). GPU execution, mixed-chunk merging, graph replay and performance are pending. This is an ablation to isolate a path difference, not a measured optimization.

### 2. Reuse K/V outputs and preparation scratch

Adapt ATOM's explicit output-buffer approach:

- Let the internal projection path receive K/V output buffers instead of always allocating them.
- Obtain capacity from supported workspace ownership, or allocate bounded scratch once per prefill forward and reuse slices across its chunks.
- Reuse cache-bound partials and K/V-scale storage where their lifetimes permit.
- Keep each cached query's FP8 tensor and Q descale alive until all consumers finish.
- Preserve the same quantization formulas, FP32 reductions, clamping and FMHA inputs.

Use per-forward or properly separated workspace lanes. A single mutable module-global buffer is unsafe when forwards or microbatches overlap. Reuse must respect stream ordering, graph capture and the final consumer of each tensor. Do not alias K/V outputs with compressed cache or other still-live inputs.

Start with the eligible FP8 path only. Keep native fallback unchanged. Measure allocation calls, host enqueue time, GPU pipeline time and peak/reserved memory. Larger persistent reservation could consume KV capacity, so unchanged GMU alone is not an adequate memory check.

Expected benefit: reduced host preparation and allocation overhead, particularly with multiple cached chunks. This change does not remove the cached-data scan or an attention kernel launch by itself.

### 3. Remove measured adapter overhead

The current path enters device contexts in `FP8PrefillContext.project()`, `project()` and `FP8PrefillContext.run()`, and performs overlapping tensor checks in preparation and projection.

Profile the host path first, then:

- Consolidate redundant device/context handling inside a trusted internal entry point while preserving the public function's device contract.
- Reuse one resolved query-cache entry between projection preparation and attention where practical.
- Keep immutable capability/layout validation at initialization when it is not already there.
- Retain per-call checks for changing shapes, devices, strides and index limits.

Do not bypass AITER argument checks through private kernel entry points merely to shorten the call stack. AITER already caches compiled launchers in the inspected source. Any additional wrapper cache must have a demonstrated miss/cost and a correct key.

Expected benefit: lower host dispatch overhead without changing mathematical operations. Multi-device behavior and graph replay still require validation.

### 4. Select between complete projection/attention pipelines

Compare three paths using identical native weights and cached inputs:

| Path | Projection output | Attention | Question answered |
| --- | --- | --- | --- |
| A: native baseline | BF16 | BF16 | Current control |
| B: fused projection | BF16 | BF16 | Do cheaper gather and projection win without FP8 attention preparation? |
| C: fused projection | FP8 directly | FP8 | Does the extra attention/output saving exceed scale and dispatch costs? |

Path B can avoid the latent dequantize/requantize sequence while avoiding FP8 output-scale preparation. Its numerical output is not assumed bit-identical to A, because A requantizes the dequantized latent before multiplication. Validate it explicitly.

Measure request count, total and maximum Q/KV lengths, raggedness, heads, DCP layout, projection tiles and FMHA split count. C48 request concurrency is not the same as the number of sequences in each attention call.

Use those measurements to choose supported shape regions. Do not raise the current 512/8192 thresholds solely to exclude failing benchmark cases. Any selection that changes collective data types or buffers must be consistent across ranks.

The desired outcome is faster eligible cases and a measured native fallback elsewhere. It is not a promise that FP8 FMHA will be faster for every shape.

### 5. Fuse cached-bound collection only if it is still a bottleneck

This is a larger kernel experiment, after workspace and dispatch measurements.

One possible design is to collect latent norm/RoPE maxima while gathering cached rows, reducing a separate read of the compressed cache. DCP makes this nontrivial: local statistics must cover the exact valid rows used by the eventual projection, and the output scales must account for all relevant ranks.

A design that adds a new collective or synchronization may cost more than the scan it removes. Include that cost. Do not assume a single GEMM can derive a global dynamic output scale and use it in every tile's epilogue without an explicit dependency-safe design.

The current two-launch Q/cache preparation is already fused. Replacing it must show a reduction in full-pipeline work, rather than moving the same work into another kernel.

## Numerical protection to retain

ATOM's inspected cached path obtains `gather_ks/gather_vs` from new-token Q/K/V preparation and reuses them for cached projection. That is useful performance guidance, but new-token statistics alone do not bound arbitrary cached outliers.

Our cached-data calculation bounds projected channels using:

```text
|latent_row · weight_row| <= norm(latent_row) * norm(weight_row)
```

It also accounts for cached RoPE magnitude, cache descale and rounding margin. Keep protection equivalent to the validated 8x/32x cached-outlier cases.

Do not use fixed scales, new-token-only estimates, relaxed tolerances or silent clipping as a performance shortcut. Buffer/dispatch changes should preserve existing results; algorithm or projection-path changes require the existing numerical and model-level checks.

## Measurement and acceptance gates

For the next GPU reservation:

1. Record routing and shape counts separately from performance. Include all eight ranks, mixed/tail chunks, zero-context rows and ragged batches. Record installed AITER/FlyDSL versions and source hashes.
2. Warm every measured shape and verify that no new compilation occurs during steady-state timing. Run performance with the serving observer disabled; its current phase-file reads and wrappers are not free.
3. Compare full A/B/C pipelines at observed shapes, including Q512/KV16896 and Q1024/KV33792. Time components for diagnosis, but also measure the complete pipeline; isolated times do not capture overlap or contention.
4. Test one change at a time. Repeat serving comparisons in counterbalanced OFF/ON and ON/OFF order. Report throughput, TTFT, TPOT, errors and variability.
5. Validate outputs/LSE, changed-input graph replay, mixed/empty chunks and cached outliers. For a candidate proceeding toward the PR, run GSM8K, cached-path NIAH and the full C48+MTP replay.
6. Compare allocator peaks/reservation and reported KV capacity at GMU 0.90 and load format `auto`. Keep weight quantization, graph settings and model revision fixed.
7. Proceed to broader concurrency sweeps after the PR gates. Reject candidates whose apparent benefit disappears after warmup or whose regressions exceed the observed measurement variation.

No GPU job was started for this document. The previous reservation ended. PR #64288 remains Draft at a037; this plan does not change its implementation.

## Integration scope

The long-term PR should focus on its distinct direct-output projection capability and reuse established attention interfaces where practical. Upstream #59124 now supplies an optional FlyDSL FP8 prefill backend, but it still materializes expanded BF16 K/V before quantizing them.

Combining direct projection with that backend requires an explicit already-quantized-input/scales contract to avoid quantizing K/V again. Keep that integration separate from the first overhead ablations. New-token FP8 attention remains deferred, and the additional-environment-variable discussion remains open.

## Source references

- [ATOM shared workspace and cached projection](https://github.com/ROCm/ATOM/blob/64412f1093b3a388131590996debf222745282e6/atom/model_ops/attention_mla.py#L1735): inspected blob `9f63654e2405bfd66f1a302f10b60734559f50c3`.
- [ATOM Q reuse and new-token-derived gather scales](https://github.com/ROCm/ATOM/blob/64412f1093b3a388131590996debf222745282e6/atom/model_ops/attention_mla.py#L1835).
- [AITER projection wrapper and cached launch](https://github.com/ROCm/aiter/blob/3dffe637860dcc35cb95ba23790eae3fc099ddef/aiter/ops/flydsl/gather_kv_b_proj.py#L416): inspected blob `85233e73b9fe843f9c4e2023f66e6904965b9b21`. This reference is not proof of the exact AITER version installed in the earlier GPU runs.
- [Our current projection allocations and Q cache](https://github.com/ajith-sirra-amd/vllm/blob/a03796b8e3bfe536e22f9db741109dbf80170079/vllm/v1/attention/ops/rocm_aiter_mla_fp8_prefill.py#L103).
- [Our current preparation allocations and cache bounds](https://github.com/ajith-sirra-amd/vllm/blob/a03796b8e3bfe536e22f9db741109dbf80170079/vllm/v1/attention/ops/rocm_aiter_mla_fp8_quant.py#L302).
- [Native vLLM ASM workspace ownership example](https://github.com/ajith-sirra-amd/vllm/blob/a03796b8e3bfe536e22f9db741109dbf80170079/vllm/v1/attention/backends/mla/rocm_aiter_mla.py#L2864).
- [Merged upstream FP8 attention backend](https://github.com/vllm-project/vllm/blob/7187c01de54aadcf28367f53284eda95c28778fd/vllm/v1/attention/backends/mla/prefill/aiter_flydsl_fp8.py).

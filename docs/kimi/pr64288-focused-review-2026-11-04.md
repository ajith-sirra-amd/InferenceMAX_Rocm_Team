# PR64288 focused review work - 2026-11-04

## Source and execution

- Attention PR: https://github.com/vllm-project/vllm/pull/64288
- Published PR head is now `e2393d702a3736dffdfbfc7eb25abbabe484a1df`.
- Review candidate: `e2393d702a3736dffdfbfc7eb25abbabe484a1df`, branch
  `checks/pr64288-asm-sync-review-20261104` in the vLLM fork.
- Separate synchronization candidate: `c894899149c43bae0873ec8e3b8b76213d267f8f`,
  branch `fix/mla-metadata-async-copy-20261104`. No separate upstream PR opened yet.
- Both use base `44d9f92e05f45ef45ec11d762011a2179be9d9cc`.
- [Static validation](https://github.com/ajith-sirra-amd/vllm/actions/runs/41819725072):
  passed. Artifact `13316801035` verified the exact candidate source, Ruff,
  formatting, applicable pre-commit hooks and retained quantizer calls.
- [Focused GPU validation](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/41820076564):
  eight existing cases passed; additional diagnostics failed as detailed below.
- Execution harness: `63e4ed42d83ac11b3b940cabe1c87880b0dded65`.
- Pinned container remains `e0d00e4b`, digest
  `sha256:b1374c3814b6e7706c49968ba17ca3b35484382f7b9fc7115dd3e66068d53f2f`.
- No full performance runs dispatched. The previously cancelled OFF/ON pair stays cancelled.

## Candidate changes

1. Restore ASM eligibility in both the metadata builder and attention implementation.
   Existing dispatch already limits ASM to prefills without cached context. Cached
   FlyDSL routing remains separate. User documentation describes that behavior.
2. Remove the two default-path synchronization changes from the attention diff and
   preserve them on a branch containing only those changes.

The synchronization candidate stages the cache pointer list and context row indices
in pinned host memory using `async_tensor_h2d`. Its reproducer and serving validation
remain necessary before opening a separate upstream PR. Older GSM8K evidence included
these fixes; it must not be presented as evidence for a revision without them.

## Focused diagnostic scope

- Run the existing eight FP8 kernel cases on the exact candidate.
- Construct a real MLA layer, projection and metadata builder; check uncached
  `forward_mha` against the existing FP8 reference tolerance.
- Run the same uncached diagnostic with the previous backend source and require
  the specific missing-ASM-metadata failure. Restore and verify candidate source
  afterward. This is a single-rank layer regression; serving coverage is separate.
- Construct cached outliers at four magnitudes in plain and preshuffled projection
  layouts. Compare the inherited new-token scale estimate against a cached-amax
  diagnostic, and report clipping, attention output and LSE error.

The cached-amax diagnostic uses a BF16 projection to measure the output range and
then a second projection to FP8. It is **not** a proposed serving implementation,
a performance measurement, or a fix for every source of FP8 error.

## Cache scale finding and next decision

ATOM `58ba50180312cc2ba57f8c21d28598352db1a561` uses
`2 * max(new_token_amax / 448, 1e-6)`. AITER v0.1.27 explicitly documents that
its fused projection computes neither a BF16 intermediate nor a cached amax.
Copying ATOM's estimate does not guarantee cached outliers fit.

After recording the outlier result, evaluate a cache-derived scale without
materializing expanded BF16 K/V. One possible experiment is a conservative
Cauchy-Schwarz bound from the maximum cached latent row norm and the maximum
projection-weight row norm, with the cached RoPE maximum included for K.
This is a new scale-selection experiment, not established ATOM behavior. It can
prevent clipping mathematically but can lose FP8 precision for small values;
numerical, graph-replay, long-context and performance checks are required before
any integration. No such scale change is in the PR or review candidate.

## Remaining order

1. Inspect focused GPU artifacts and fix any diagnostic or implementation failure.
2. Finish the small review changes and separate sync-fix validation.
3. Resolve cached outlier behavior; preserve existing numerical tolerances.
4. Matched short `vllm bench serve`: main, same-source flag OFF, flag ON; include
   more than one concurrency and cases below the routing cutoffs.
5. GSM8K plus long-context accuracy above 8K using normal routing and proof of
   actual optimized-path execution.
6. Freeze final source, image and settings.
7. Full matched OFF then ON runs with MTP, last. Confirm whether the user's latest
   `C448` meant `C48` before dispatch.
8. Update PR claims, comments and evidence with exact revision scope.

Seungrok's environment-variable discussion remains open. New-token FP8,
native NUMA changes and further concurrency sweeps remain deferred.

## Completed focused results

The small fixes are published in PR64288 at `e2393d70`.
The current head passed all eight existing native FP8 kernel cases in
[41820076564](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/41820076564),
artifact `13316522295`. That workflow failed in additional diagnostics, not
in those eight kernel cases.

The corrected real-layer ASM regression in
[41821548931](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/41821548931),
artifact `13317306322`, failed before the fix with `ASM_METADATA_MISSING`
and passed after it. The log records the ASM kernel load; maximum absolute
error versus the existing reference was 0.0215493441.

The initial ASM diagnostic needed its configuration context set before
distributed initialization. A reuse attempt,
[41821244716](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/41821244716),
then failed on Git's ownership check before running tests. Git trust was
restricted to the known `/work/vllm` checkout for each command; the next
attempt completed all diagnostics. Neither issue changed production code.

Constructed outliers gave identical results in both weight layouts:

| Cached multiplier | Current-scale output error | Cached-amax output error | Cached-amax projection error | Cached-amax strict gate |
| ---: | ---: | ---: | ---: | --- |
| 1 | 0.00097656 | 0.00097656 | 0 | pass |
| 2 | 0.00184940 | 0.00380253 | 0 | fail |
| 8 | 0.02530534 | 0.00186784 | 0 | pass |
| 32 | 0.20291615 | 0.00020885 | 0 | pass |

Cached-amax removed clipping in every constructed case. Its two failed
cases (one per layout) remain failed against the unchanged additional
`atol=2e-3, rtol=2e-3` output check. This is not a serving scale fix.

For the multiplier-2 case, AITER rounds unnormalized softmax probabilities
to FP8 before P*V, while the denominator sums the unrounded probabilities.
The relevant probability is 0.9732996356757996, which rounds to 1 in E4M3.
The resulting BF16 output is predicted as 0.1328125 versus the high-precision
0.12900997184323335: error 0.00380252815676665, matching the measured error.
Source: AITER v0.1.27
`aiter/ops/flydsl/kernels/fmha_gfx950/flash_attn_fp8_gfx950.py`
(`_softmax_part`) and `op_gemm.py` (`cast_p_fp8_direct`).
This explains that case's inherited FMHA approximation; it does not excuse
clipping from the integration's cached-scale estimate.

An isolated cache-derived norm-bound experiment passed in
[41822676794](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/41822676794).
It tests actual plain/shuffled projection against an independent FP64 oracle
with the existing projection gates (relative RMS <0.04, normalized peak <0.07)
and zero clipping. Its Torch intermediates are diagnostic; no scale change
or performance claim has been added to the PR.

Current-head upstream DCO passed. The
[contributor-authorization gate](https://github.com/vllm-project/vllm/actions/runs/41822182813/job/126547776319)
failed before pre-commit execution; the account has zero merged PRs and none
of the required authorization labels. No label was requested or added.

## Cache-derived scale prototype results

The norm-bound diagnostic in run 41822676794 passed all six cases
(cached multipliers 1/8/32, plain and preshuffled weights), with zero clipping.
Worst relative RMS was 3.4176%, below the unchanged 4% limit; normalized peak
errors remained below the unchanged 7% limit. Both storage layouts matched.
The bound was about 3.7-7.55 times the actual maximum, which motivates model
accuracy checks before integration.

The [GPU cache scan](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/41823896651)
passed for 0, 1, 386, 8192 and 32768 selected rows. Artifact: 13318372088.
GPU scales matched the Torch oracle (rtol=1e-6, atol=1e-8), including graph
replay after modifying cache values and the nonbinary cache scale.

| Selected cached rows | Scan (us) | Projection alone (us) | Scan + projection (us) | Added projection-stage cost |
| ---: | ---: | ---: | ---: | ---: |
| 386 | 4.406 | 8.765 | 13.091 | 49.35% |
| 8192 | 5.387 | 29.863 | 35.861 | 20.08% |
| 32768 | 9.718 | 87.137 | 98.273 | 12.78% |

These are one-GPU synthetic CUDA-graph measurements, not serving performance
or evidence of uplift. The scale rule in the attention PR is unchanged.

The next isolated experiment compares the current Q/K/V preparation plus
projection against Q-only quantization, cache scanning and direct projection.
It reuses the current Q kernel arithmetic, checks bitwise Q agreement and
changed-input graph replay, and tests an index beyond the signed int32 byte
offset range. The scan now explicitly converts row indices to int64 before
multiplying by the cache stride. No prototype is promoted to the attention PR.

## Current-head external checks

DCO and independent fork static checks passed. Upstream pre-commit did not
execute because of the contributor-authorization gate. Read the Docs build 35640024 also
stopped in docs/pre_run_check.sh because the same authorization check failed,
before installing dependencies or rendering documentation. These checks are not
reported as passing, and authorization labels have not been requested.

## Complete preparation comparison

[Run 41824901365](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/41824901365)
passed; artifact 13319470276. All six cases produced bitwise-identical Q and
Q scales, including changed-input graph replay. The selected row at byte offset
2,147,484,096 also passed, exercising the int64 address calculation.

| Cached rows | Query rows | Current preparation + projection (us) | Q-only + scan + projection (us) | Time change |
| ---: | ---: | ---: | ---: | ---: |
| 8192 | 512 | 38.160 | 41.411 | 8.52% |
| 8192 | 2048 | 43.634 | 44.042 | 0.93% |
| 8192 | 8192 | 70.839 | 59.066 | -16.62% |
| 32768 | 512 | 95.352 | 103.565 | 8.61% |
| 32768 | 2048 | 100.691 | 106.351 | 5.62% |
| 32768 | 8192 | 132.386 | 121.073 | -8.55% |

Medians of three rotated-order graph timing samples are shown. One GPU,
12 heads, one cached chunk, synthetic inputs; no FMHA or serving measurement.
Replacing K/V quantization repays scan cost at 8K query tokens, but the 512-token
cases regress by about 8.5%. The prototype remains outside the attention PR.

Next, test fusing the cache partial scan into Q's partial-maximum launch and
its scale reduction into Q's quantization launch. This keeps the same norm-bound
math and Q arithmetic while reducing launch overhead. Require unchanged Q,
cached-scale oracle agreement and replay after changing Q, cache, indices and
cache scale before drawing any performance conclusion.

## Fused preparation results

[Run 41825671539](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/41825671539)
passed; artifact 13318951976. Q and Q scales were bitwise identical to the current
quantizer. Cached scales matched the oracle, including graph replay after changing
Q, cache values, row indices and cache scale.

| Cached rows | Query rows | Current preparation + projection (us) | Fused preparation + projection (us) | Time change |
| ---: | ---: | ---: | ---: | ---: |
| 8192 | 512 | 37.831 | 37.160 | -1.78% |
| 8192 | 2048 | 43.647 | 39.154 | -10.29% |
| 8192 | 8192 | 70.766 | 53.058 | -25.02% |
| 32768 | 512 | 95.423 | 99.368 | 4.13% |
| 32768 | 2048 | 100.703 | 100.499 | -0.20% |
| 32768 | 8192 | 132.161 | 110.817 | -16.15% |

The two fused preparation launches improve the tested larger-query cases.
The 512-query/32K-cache case still costs 4.1% more for this component.
These are synthetic single-chunk component timings, not serving throughput.
Multiple-chunk amortization and actual-model behavior remain unvalidated.

Next run checks the fused preparation with real projection and FMHA, plain and
preshuffled weights, and changed-input graph replay at 20/386 and 512/8192
query/cache rows. Existing projection limits (4% RMS, 7% normalized peak), zero
clipping, FP8-input attention output (atol=2e-3, rtol=2e-3) and absolute LSE
(atol=1e-3, rtol=0) are retained. Pre-quantization output/LSE differences are
reported separately and do not substitute for model accuracy.

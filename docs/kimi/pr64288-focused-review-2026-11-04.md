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
  dispatched; result pending when this note was written.
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

An isolated cache-derived norm-bound experiment is running in
[41822676794](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/41822676794).
It tests actual plain/shuffled projection against an independent FP64 oracle
with the existing projection gates (relative RMS <0.04, normalized peak <0.07)
and zero clipping. Its Torch intermediates are diagnostic; no scale change
or performance claim has been added to the PR.

Current-head upstream DCO passed. The
[contributor-authorization gate](https://github.com/vllm-project/vllm/actions/runs/41822182813/job/126547776319)
failed before pre-commit execution; the account has zero merged PRs and none
of the required authorization labels. No label was requested or added.

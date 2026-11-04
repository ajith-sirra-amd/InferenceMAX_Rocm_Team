# PR64288 review closure checklist

## Current source and evidence

- Published PR: `e2393d702a3736dffdfbfc7eb25abbabe484a1df`.
- Validation candidate: `a03796b8e3bfe536e22f9db741109dbf80170079`.
- Pinned upstream main: `69d27bf8002a4634de1699c95fb0aeb22fb86ca5`.
- Active serving/accuracy run: [41840577273](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/41840577273), harness `64bf0998`.
- Final replay harness checked at `dfea3045471efc0033c4f832a1eb8936e9162ac4`:
  [41841997094](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/41841997094), 45 tests passed, no failures/skips; launcher checks passed.
- Recovered GPU evidence: [41838441209](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/41838441209), artifact `13329885256`.
  Twelve kernel tests and eight numerical/graph-replay cases passed at the candidate.
  Zero clipping; five installed source hashes independently match the candidate.
  The workflow was cancelled for a later observer correction and is not an overall pass.

## Design fit

- The candidate's upstream diff contains eight attention, test, configuration and
  documentation files. Neither metadata synchronization change is in that diff.
- Compressed FP8 cache and native per-channel FP8 projection weights feed AITER's
  direct projection, with FP32 accumulation and FP8 K/V output for cached FMHA.
- The scale estimate is replaced by cached-latent/weight norm bounds and cached
  RoPE range. This is new scale-selection work, not established ATOM behavior.
  Conservative bounds can reduce precision, so model accuracy and serving
  performance are required before promotion.
- Existing uncached ASM and below-cutoff routing are preserved. The additional
  environment-variable discussion remains open as requested.
- The implementation is substantial; synthetic component timings alone do not
  justify an end-to-end uplift claim. Final matched workload evidence is pending.

## Reviewer concerns

| Concern | Status and remaining evidence |
| --- | --- |
| Remove unused quantizer arguments/branches | Published in `8a55b8eac`. |
| Remove AITER signature introspection | Published in `8a55b8eac`; pinned API retained. |
| Remove prototype wording | Published in `8a55b8eac`. |
| Reuse cached enablement flag | Published in `8a55b8eac`. |
| Preserve uncached ASM | Published in `e2393d70`; real-layer failure-before/pass-after evidence recorded. |
| Separate default-path synchronization changes | Removed from attention PR. Context-row helper overlaps upstream #63935/#64548. A separate DSpark public-entry reproducer is still needed. |
| Cached K/V larger than new-token values | Candidate derives bounds from each cached chunk; plain/shuffled outlier and changed-input replay checks passed. Model accuracy and matched performance remain pending. |
| Same-head main/OFF/ON serving at multiple concurrencies and below cutoffs | Active run includes C8/C48 and both cutoff cases. BF16 fallback observation was corrected without changing dispatch or numerical gates. |
| Normal-routing long-context accuracy | Active run includes 16K/32K retrieval, three depths, two repeats, and requires FP8 projection/FMHA on every worker in the enabled arm. |
| GSM8K model evaluation | Active run includes all 1,319 examples for main and candidate ON with live MTP. Actual routing coverage must be reported separately. |
| Existing related PR comparison | #59124's reference replay failed; do not repair or add the direct-projection implementation to it. No measured comparison is claimed. |
| Enablement environment variable | Leave open pending the author's decision with Seungrok. |

## Testing and validation

The original projection, attention-output and LSE tolerances remain unchanged.
The recovered candidate kernel evidence is not a replacement for model accuracy.
The active run must produce complete request counts, correct source manifests,
valid per-rank execution evidence, retrieval answers, full GSM8K scores and live
speculative acceptance. Compare main/candidate GSM8K outcomes rather than treating
the repository's minimum score alone as proof of equivalent accuracy.

For the separate synchronization question, the existing public server test
`tests/models/kimi_k3/test_prefix_cache.py::test_turns_reuse_prefix_and_match_recompute[plain-partial-dspark]`
is a possible minimal reproducer with `VLLM_GPU_SYNC_CHECK=error`. It uses the
repository's small Kimi/DSpark model configuration and dummy weights; it is a
behavior test, not a model-accuracy test. Run on the unchanged base first and
inspect the exact failure. Do not infer a model-path defect solely from the
already recorded isolated Torch transfer error, or propose a fix without a
matching before/after result.

## Code quality and documentation

Candidate static checks and strict rendered docs passed in
[41836553476](https://github.com/ajith-sirra-amd/vllm/actions/runs/41836553476).
The new cache-bound behavior is documented in the candidate's rendered usage
section. Candidate promotion can fast-forward the existing PR branch; its ancestry
includes the published head. No force push is needed for the candidate as checked.

Upstream authorization-gated checks are not passing CI. Do not request/add labels
or represent independent fork checks as upstream approval.

## Final performance and PR contents

After serving/model checks and final source review, freeze the source, exact
container digest and harness. Run the existing full C48/TP8/DCP8 recipe with MTP,
flag OFF then ON. The replay uses synthetic DSpark acceptance, while accuracy
uses real verification; label both correctly. Keep CCD binding, CPU KV offload,
quantization targets, BLAS settings and measurement duration identical across the
two performance arms. Report request errors and timing/coverage, not throughput
alone. Historical prototype numbers are not the current matched baseline.

Update the PR description and existing evidence comments only with results that
have been inspected, retaining their exact revision scope. Do not mark unresolved
review questions or external CI as completed.

New-token FP8, NUMA, concurrency sweeps and steady-state profiling remain after
the current review-validation work. IPAT stays untouched.

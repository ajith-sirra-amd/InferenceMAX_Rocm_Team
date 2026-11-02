# PR #64288 review follow-through ? 2026-11-02

[??](pr64288-review-follow-through-2026-11-02_zh.md)

## Current source and scope

PR #64288 remains at 492080779e9feea19308092b0aba0840e87c6115. It implements direct FP8 cached projection/FMHA and keeps new-token attention BF16. The user requested comparison only against #59124, then review readiness. No benchmark integration is pushed into the actual PR.

## Review checklist

- Design fit: the implementation adds 803 production lines, mostly in isolated quantization/projection helpers, and reuses AITER kernels and native compressed DCP gathering/merging. Target scope is gfx950, DCP >1 and MLA dimensions 512/128/64/128. Maintainer consideration of complexity versus measured gain remains necessary. The additional environment-variable thread remains open as requested.
- Behavioral validation: eight current-head MI355X tests passed in run 41401317715. Full GSM8K run 41426504427 passed 1275/1319 strict (96.6641%) and 1274/1319 flexible (96.5883%), with real MTP and direct FP8 projection/FMHA on eight distinct workers. Accuracy-stress routing, eager and 512-token chunks are disclosed.
- Quality/docs: run 41442476770 passed applicable pre-commit/mypy, Ruff/format, exact changed-file comparison and full strict MkDocs rendering. The rendered FP8 usage section and configuration were inspected. Artifacts: 13140797249 static; 13140323711 rendered HTML.
- PR contents: description updated with completed accuracy, current validation and historical/native result separation. No upstream CI approval is claimed. The upstream pre-run-check denied contributor authorization, and RTD stopped before rendering; no workflow bypass or label request was made.
- Numerical limits: the historical BF16 absolute-LSE result (~0.935) remains disclosed. The extra 0.20 diagnostic is not a mandated upstream threshold. Failed rotation/scaling experiments are excluded. No tolerance was widened.

## Matched comparison

Reference PR head: 132f6daa6b17cb4000ef58355943bb39339104f2.
Integration source: 2eb0ada53e6d0fcd658010d9869957026a229adb on the common base a1cfd2c1b584c4ac197db9cc9ac84a6d973d39df.
Harness/workflow source: 190f103aad8911bff3a31a2ee0cbc12ed9070ce9.

- Run 41443401220: CPU harness checks passed; two overlong integration-test lines failed Ruff, so GPU was skipped. Fixed in the next pinned source.
- Run 41443627369: CPU passed (21 observer/installer tests plus launcher/source checks); GPU integration passed: 26 passed, no failures or skips (11 unrelated tests deselected). Artifact 13141920953; reference Q reuse/direct projection passed changed-input graph replay. Includes original native arithmetic tests, reference BF16/quantization tests, and shared-Q/direct-projection graph replay.
- Run 41443951889: full reference C48/MTP replay cleared successful prerequisite 41443627369 and is starting. Direct projection OFF, ROCM_AITER_FLYDSL_FP8 ON, 3600 seconds. The prerequisite must succeed at the exact harness SHA before GPU scheduling.
- Direct-ON replay: dispatch after inspecting the preceding results. Keep source, weights, image, workload, CCD binding and BLAS settings identical.

The observer requires FP8 causal/cached FMHA on all eight workers. Reference mode requires BF16 expanded K/V and forbids direct native projection calls. Direct mode requires FP8 projection. Both use reference Q reuse; direct output scales remain new-token-derived times two, unlike reference actual-context maxima.

## Remaining

Inspect GPU test reports and failures immediately. Complete matched full runs, check all-worker execution, coverage and request/server errors, update the reviewer with concise evidence, complete the checklist, then mark the PR ready for review under the user's authorization. Upstream CI and human review remain merge requirements.

Broader profiling and C56/C64 MTP, C96 without MTP remain backlog after this PR task. Native NUMA changes are postponed.

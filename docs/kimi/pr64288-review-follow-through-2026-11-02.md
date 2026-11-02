# PR #64288 review follow-through ? 2026-11-02

> **Status correction, 2026-11-02 17:13 UTC:** PR head is now `2c01e1ce21a262f6108861a831eae5a7d2a2ca8e` after merging `main`. Completed GPU/GSM8K/performance/static results below apply to `492080779e9feea19308092b0aba0840e87c6115`; post-merge validation remains outstanding. Latest GSM8K strict accuracy rounds to **96.7400% (1276/1319)**. See [the factual audit](pr64288-factual-audit-2026-11-02.md). Earlier status entries below are historical.

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
- Run 41443951889: full reference C48/MTP replay cleared successful prerequisite 41443627369 and subsequently failed (see failure evidence below). Direct projection OFF, ROCM_AITER_FLYDSL_FP8 ON, 3600 seconds. The prerequisite must succeed at the exact harness SHA before GPU scheduling.
- Run 41444880140: cancelled before the GPU job started at the user's request. Do not resume without new authorization.

The observer requires FP8 causal/cached FMHA on all eight workers. Reference mode requires BF16 expanded K/V and forbids direct native projection calls. Direct mode requires FP8 projection. Both use reference Q reuse; direct output scales remain new-token-derived times two, unlike reference actual-context maxima.

## Remaining

Inspect GPU test reports and failures immediately. Complete the standalone native replay and its gated GSM8K, check all-worker execution, coverage and request/server errors, update the reviewer with concise evidence, complete the checklist, then mark the PR ready for review under the user's authorization. Upstream CI and human review remain merge requirements.

Broader profiling and C56/C64 MTP, C96 without MTP remain backlog after this PR task. Native NUMA changes are postponed.

## User steering: standalone PR run and follow-up GSM8K

The user explicitly prohibited adding the direct FP8 up-projection diff to PR #59124 or its author's branch. No such write occurred; integration code is only in the user's fork benchmark branch. Do not resume the combined benchmark without new authorization.

- Combined run 41444880140: cancelled before the GPU job started.
- Reference #59124 run 41443951889: failed; no valid throughput comparison.
- Standalone native PR #64288 run 41447686785: active, exact PR SHA 492080779e9feea19308092b0aba0840e87c6115, November 1 image e0d00e4b, full C48/TP8/DCP8/MTP replay. Native comparison mode keeps BF16 new-token attention.
- Full native GSM8K run 41447950940: dispatched with a success prerequisite on 41447686785. Workflow SHA 94f5efa4a61aae4f4a09d54e2c30d9b4dd56216d extends the prerequisite wait to 240 minutes; execution source remains 190f103aad8911bff3a31a2ee0cbc12ed9070ce9. Real MTP, eager 512-token stress routing, exact native source and eight-worker execution proof remain required.
- Standalone vs reference results cannot isolate projection-only gain because causal routing/Q reuse/scale policy also differ. Update the PR with that distinction.
- The historical 18,244.04135 native throughput is on earlier pinned image edad8b03, not e0d00e4b. The hipBLAS workaround has no isolated performance uplift measurement; both TORCH_BLAS_PREFER_HIPBLASLT=0 and TORCH_BLAS_PREFER_CUBLASLT=0 were tested together.
- Kimi-K3.1 compatibility is unconfirmed. Native gating is configuration-based (512/128/64/128, gfx950, DCP >1, FP8 E4M3/native per-channel weights), but guessed checkpoint config URLs returned HTTP 401. An optional request for the exact K3.1 checkpoint/config URL is pending; do not claim validation.

## Reference failure evidence

Run 41443951889 failed. The first watchdog report at 13:33:13 UTC was DCP KV all-gather in dcp.py:1649: PG 3, sequence 232455, 4,755,456 input elements and 38,043,648 output elements, after 600 seconds. A second timed-out group was logits all-gather: PG 2, sequence 2668, 1,064,960 input elements and 8,519,680 output elements. All eight ranks reported matching sequence/sizes within each group. No preceding server ERROR/Traceback/BLAS exception was found. Root cause remains unestablished; these traces do not assign fault to reference attention, integration, hardware or BLAS.

Client errors were already present during warm-up: 434/531 records. Measurement produced 63 records, 52 errors, and only 1440.19 recorded seconds. The failed-request threshold aborted the replay after the engine died. This is not a valid performance baseline. Reference FP8 causal/cached attention and BF16 expanded K/V were observed on all eight workers; no direct projection events occurred.

Full raw evidence remains in run artifacts 13146026833 and 13146571024; a structured failure summary is at results/atom-attention/run41443951889/failure-summary.json. Our standalone PR run continues and GSM8K remains gated behind its success. The cancelled combined arm stays cancelled.

Warm-up control: successful native run 41413372088 had 437/531 empty-content warm-up errors, versus 434/531 for the failed reference. The harness uses extra one-token warm-up requests; these counts alone do not establish a numerical defect. The reference's measured-phase failure and collective timeout remain the actionable evidence.

## Newer-nightly performance completed; reference work stopped by user

Native run 41447686785 succeeded at **18,276.23099 total tokens/s/GPU**, using PR SHA 492080779e9feea19308092b0aba0840e87c6115 and image e0d00e4b. C48/TP8/DCP8, synthetic MTP k=4, normal routing/graphs, full 3600-second measured replay; aggregation includes 3625.89578 seconds of request activity. TTFT and ITL coverage both 100%. Eight workers proved direct FP8 projection and cached FMHA; new-token attention stayed BF16.

Measured records: 3840 total, 3834 successful, 6 errors (five empty-content responses and one request-body write failure). Nine additional credits were cancelled at the duration/grace-period cutoff. No BLAS warnings or detected server runtime errors. The result is 0.1764% above the earlier-image 18,244.04135 result in one run; no causal nightly-image/workaround uplift is established.

GSM8K run 41447950940 attempt 1 failed before GPU scheduling because its prerequisite watcher received GitHub API HTTP 502 at 14:56:41 UTC. No accuracy evaluation executed. Failed jobs were restarted as attempt 2 after the native performance run succeeded; the prerequisite passed. Keep following the full evaluation and inspect score, exact source, real MTP and all-worker execution.

User instruction: **Do not attempt to fix #59124, including the reference benchmark integration, and do not rerun it. Report its observed failure only.** The combined arm remains cancelled. This PR differs by direct FP8 latent-cache/FP8-weight projection into FP8 K/V, eliminating the BF16 expanded K/V materialization and separate cached K/V quantization. #59124 uses FP8 new-token attention; this PR retains BF16. The failed reference is not a valid speedup baseline.

The PR description and existing reviewer comment now include this difference and the completed newer-nightly result. Native source remains unchanged and PR stays draft pending the requested follow-up evaluation.

## User decisions on PR presentation and later work

The PR summary, comparison table and current reviewer comment no longer emphasize unchanged BF16 new-token attention or contrast its precision with #59124. Public claims remain explicitly scoped to cached-context attention and direct FP8 cached-K/V up-projection; the source and full engineering evidence are unchanged.

New-token FP8 is deferred to a later experiment. Prepare a separate future PR only after accuracy validation and a significant measured performance uplift. Do not add it to the current PR, and do not resume or fix the #59124 reference run. Current priority remains full native GSM8K attempt 2, review readiness, then the already-requested steady-state profile and concurrency sweep.

## CI recheck and profiling preparation

Current PR head remains `492080779e9feea19308092b0aba0840e87c6115`. DCO, Mergify Summary and the Meta check passed. Upstream `pre-run-check` failed because this contributor has zero merged PRs and none of the accepted authorization labels. Pre-commit is skipped, and Read the Docs' pre-run gate refuses the build. These statuses do not identify a source failure. The gate explicitly says agents must not request authorization labels; none was requested or added. No CI policy was bypassed. Independent current-source static checks and strict rendered documentation passed.

Public presentation stays focused on direct FP8 cached-K/V up-projection. Include only one brief #59124 mention with its failed comparison outcome; no dedicated comparison section or precision contrast. New-token FP8 remains future separate work conditional on accuracy and significant uplift. The environment-variable discussion remains open.

GSM8K run `41447950940`, attempt 2, job `125363224586`, is still active at the last check. Do not mark review readiness until its score, exact source, real MTP and all-worker optimized execution evidence have been inspected.

Steady-state profiling wiring is published separately on `checks/native-steady-profile-20261102` at `9da21ee27541876ead88045459ecbf0e74e7c670`. CPU run `41476491034` passed **30 tests, zero failures/skips**, plus real launcher/syntax checks. JUnit artifact `13157862094` was inspected and contains all nine controller cases, including SIGTERM cleanup and lost start response. A profile-only shutdown allowance permits trace export before stopping the server.

Profiling is default-off, restricted to a full standalone native replay, and waits for the measured-phase marker plus 300 seconds of generation progress before a 20-second worker CPU/GPU capture. Output and artifacts remain in this repository. GPU profiling has not yet been dispatched. See [profiling guide](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/blob/checks/native-steady-profile-20261102/docs/kimi/native-steady-profile.md).

## Completed native validation and review readiness

Full GSM8K run [41447950940](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/41447950940), attempt 2, succeeded. Strict: **1276/1319 = 96.7399%**. Flexible: **1275/1319 = 96.6641%**. The sample artifact contains 1319 unique examples, with one row per scoring filter (2638 rows total).

Validated on pinned image e0d00e4b, digest `b1374c3814b6e7706c49968ba17ca3b35484382f7b9fc7115dd3e66068d53f2f`. Real MTP k=4 used block verification and logged nonzero acceptance. Eager execution, 512-token chunks, C48/TP8/DCP8, five-shot, temperature zero and explicit accuracy-stress routing were confirmed. This validates the optimized arithmetic under the disclosed accuracy settings; a matched BF16 control remains separate.

All eight workers recorded nonempty E4M3 direct projection and cached FMHA during evaluation, at native SHA `492080779e9feea19308092b0aba0840e87c6115`. Their BLAS backend was `_BlasBackend.Cublas`. All eight installed source hashes in the provenance manifest independently matched the fork commit. No server runtime errors, BLAS failure signatures or disk-quota errors were detected. The standalone GPU kernel suite remains run 41401317715; this evaluation's installer used prepare-only mode.

Artifacts: scores/samples **13157663920**, raw proof/provenance/server log **13157794027**, standalone server log **13157993168**. [Structured report](../../results/atom-attention/run41447950940/native-gsm8k-summary.json).

The PR description and existing status comment were updated. **PR #64288 is now OPEN, non-draft, ready for review**, at the unchanged validated source. Upstream CI authorization and human review still block merging. No authorization label was requested or added.

Next authorized steady-state profile: [41478604149](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/41478604149), workflow/execution SHA `9da21ee27541876ead88045459ecbf0e74e7c670`, same native PR and image, full C48/MTP replay, profile enabled. Prerequisite CPU run 41476491034 passed. Follow through setup, capture and final trace analysis; GPU trace evidence is pending.

Future sweep preparation: the current master config lists only C48, so C56/C64 require explicit entries. The launcher's non-MTP Kimi script path does not exist at this revision; fix routing and metadata before C96 without MTP. These sweeps have not been dispatched.

## Concurrency sweep preflight completed

Separate branch `checks/native-concurrency-sweep-20261102`, SHA `d67152dac62f6f19a42a49c72bc2ae98b7747de1`, now contains separate C56/C64 MTP and C96 non-MTP configurations. The original C48 entry remains unchanged. Native non-MTP Kimi selects the existing shared recipe, passes `SPEC_DECODING` into the container, and rejects metadata/argument mismatches.

CPU run [41479440563](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/41479440563) passed: 30 pytest cases, zero failures/skips, launcher behavior and shell checks. Artifact **13159250994** was inspected: the actual matrix generator emitted exactly C56/MTP, C64/MTP and C96/none, each TP8/DCP8, 2249 GB CPU DRAM budget and duration 3600. The native workflow overrides the legacy matrix image with the pinned e0d00e4b image.

Planned recipe settings: C56 uses k=4/AL=3.36/chunk=8192/max-seqs=112; C64 uses k=3/AL=3.00/chunk=24576/max-seqs=80; C96 has no MTP, chunk=24576/max-seqs=144. This compares complete configuration points, not concurrency alone. No sweep GPU job has been dispatched.

Active profile run **41478604149**, GPU job **125390981568**, continues at the server/replay step. Its workflow and execution remain pinned to **9da21ee27541876ead88045459ecbf0e74e7c670**; sweep preparation does not change the running job. Keep following it through completion and inspect worker traces before scheduling sweeps.

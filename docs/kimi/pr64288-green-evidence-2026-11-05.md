# PR64288 validation evidence replacement ? 2026-11-05

The user requested that every passing validation claim in the PR link to a genuinely successful workflow. Passing components inside failed or cancelled enclosing workflows need corresponding fresh successful validation. Do not relabel failures or weaken acceptance criteria.

## Frozen revisions and reservation

- Candidate: `a03796b8e3bfe536e22f9db741109dbf80170079`.
- Published PR head: `e2393d702a3736dffdfbfc7eb25abbabe484a1df`; promotion is pending validation and performance assessment.
- Upstream base: `69d27bf8002a4634de1699c95fb0aeb22fb86ca5`.
- Full replay harness: `6b4f3553391063b1060d39cf16b154e7d26585b3`.
- Replacement validation harness: `ba6150507c57f89740881aaee6969843f7729f47`.
- Pinned image: `nightly-rocm100-e0d00e4b8eb0c89ec30921d0695cfdd409e9c211@sha256:b1374c3814b6e7706c49968ba17ca3b35484382f7b9fc7115dd3e66068d53f2f`.
- Physical host: `pit2-p03-g40`; GitHub runner name is stale (`mi355x-amd_pit2_p03_g07`).
- Reservation 12214 ends **2026-11-05 13:52:35 UTC / 19:22:35 IST**.
- Replacement GPU workflows enforce a **13:42:35 UTC** execution cutoff and stop only their own container.
- Monitor run status every two minutes, with setup failures checked promptly.

## Dispatched work, as of 04:32 UTC

| Scope | Run | State |
| --- | --- | --- |
| Full C48 + MTP, flag OFF | [41964736019](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/41964736019) | Warmup processing requests; not completed |
| Matched full C48 + MTP, flag ON | [41967738227](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/41967738227) | Dispatched; waiting for successful OFF |
| 12 kernel cases + 8 numerical/graph cases + uncached ASM | [41968261662](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/41968261662) | Hosted preflight passed; waiting for successful ON |
| NIAH, main and candidate ON | [41968261654](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/41968261654) | Hosted preflight passed; waiting for successful ON |
| Main/OFF/ON C8/C48 and cutoff serving cases | [41968261826](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/41968261826) | Hosted preflight passed; waiting for successful ON |

Each standalone run creates a fresh environment under its own repository-mounted directory; the old persistent environment was deleted during replay cleanup. GPU jobs use the same exclusive runner sequentially. Their relative order after ON depends on GitHub scheduling.

## Acceptance and interpretation

- Original numerical tolerances, prompts and serving cases are unchanged.
- GPU kernel JUnit must show exactly 12 executed cases, with no failure, error or skip.
- Eight numerical cases must complete; zero clipping and original projection/FP8-input FMHA thresholds remain enforced.
- The existing real-layer uncached ASM regression must pass.
- NIAH retains all 12 cases per arm: 16K/32K, depths 10/50/90%, two repeats; matching prompt hashes and FP8 projection/FMHA coverage on all eight candidate ranks are required.
- Serving retains all five cases and 512 completed requests per arm. Report latency regressions if present; workflow success does not prove a speedup.
- Full replays use synthetic DSpark acceptance; NIAH/serving/GSM8K use live MTP. Do not combine these as equivalent performance evidence.
- GPU/model evaluations do not establish passage of the separately disabled baseline synchronization checker.
- Do not rerun or alter the explicitly failed PR59124 reference experiment.
- Upstream contributor-authorization gating remains external; do not request restricted labels or bypass it.

## Already green evidence

- [Full GSM8K 41954676681](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/41954676681): inspected artifact 13386927365, all 1319 requests HTTP 200, zero request/response failures. Main and candidate both 1270/1319 strict (96.28506444%) and 1269/1319 flexible (96.20924943%). Candidate accepted 112527 MTP draft tokens. Equal aggregate scores do not mean identical responses. GSM8K only exercised ordinary prefill; use NIAH for optimized cached-path coverage.
- [Candidate static and rendered docs 41836553476](https://github.com/ajith-sirra-amd/vllm/actions/runs/41836553476): artifacts 13327540790 and 13325893159 downloaded and inspected. Both record candidate a03796b8. Applicable pre-commit checks passed, including Ruff, formatting, mypy, test wiring and SPDX. Strict MkDocs build completed; changed-file list contains exactly eight intended files. Nonapplicable hooks were skipped.

## PR update procedure

Inspect each replacement artifact before citing it. Replace failed/cancelled component-pass links in the PR body and authored comments, including the ASM and cached-scale replies. Keep evidence tied to the tested source. Promote the validated candidate without force only after assessment; then revise the body/checklist and tag the relevant reviewer concisely. Audit all links again after publication.

## 04:59 UTC live observation

OFF completed warmup and began its 3600-second measured interval at approximately 04:45:55 UTC. At 13:45 into measurement the reporter showed **1054 successful measured requests and one error**. Do not describe this as an error-free replay.

The measured failure is request `2fcd44b3-3c05-4dc5-abc4-d73c159d5bf9`, conversation `279424d0e53192cde462be8d39c23b52b977`, turn 2, 60329 input tokens. It started 04:55:58.959 UTC and completed after 11.8315 seconds. AIPerf reports `InvalidInferenceResultError`: no response with actual content. It was not cancelled. No ERROR/Traceback matches were found in the inspected server log. The cause is not established. The native FP8 flag is OFF.

The raw profile separately contains 531 warmup records, of which 440 carry the same empty-content classification. The warmup uses max_tokens=1, and the warmup progress reporter printed errors=0. Keep raw warmup classifications separate from measured errors; do not use that reported zero as proof that every warmup response contained text.

All five workflows remain in progress/waiting. Their replacement validation GPU jobs have not yet started.

The complete hyperlink audit found four failed/cancelled enclosing workflows supporting component-pass claims; the three fresh workflows cover their kernel/numerical/ASM, NIAH and serving scope. All eight non-run GitHub links returned HTTP 200. Every currently cited successful run has unexpired artifacts. Historical total throughput values were rechecked against archived JSON: 15476.46881, 16915.72154, 17406.08912, 18244.04135 and 18276.23099 tokens/s/GPU. Their raw error-accounting fields include warmup classifications; measured failure counts require phase-aware accounting.

## Completed OFF control and matched ON startup

[OFF run 41964736019](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/41964736019) completed successfully. Inspected artifacts:
- 13391229164: normalized benchmark JSON.
- 13391229174: raw records, commands, source manifest and execution proof.
- 13391578130: server log.

Normalized total throughput: **18266.89472 input-plus-output tokens/s/GPU**; output-only: **126.61731 tokens/s/GPU**. Aggregator duration: **3628.05955 seconds**. AIPerf separately reports **3624.56 seconds**; preserve the metric-specific time basis.

Measured records: **3835 successes + 6 empty-content response errors**. All six are InvalidInferenceResultError with the same no-content message. TTFT and inter-token-latency metrics are present for all **3835/3835** successful requests. Raw records contain 531 warmup entries (440 with no-content classification), excluded from measured error counts.

**Cutoff limitation:** the client cancelled **12 in-flight requests** at the configured end, then exhausted its 10-second cancellation-return grace period and forced phase completion. The client log includes that ERROR and later terminates its records-manager process after shutdown grace. The enclosing workflow passed. Do not describe this run as having no errors or perfectly clean shutdown. No ERROR/Traceback/hipError/hipBLAS error matches were found in the server log.

Execution proof records candidate a03796b8, normal routing, replay phase, and **flag_off on all eight ranks**. The native manifest pins binaries to base 69d27bf8 and the expected Torch/Triton/ROCm stack. The job log records both TORCH_BLAS_PREFER flags=0 and **[pin-ccd] pinned 1050 threads**. The CCD map itself was not included in the raw artifact; it was inspected live earlier.

ON run **41967738227**, GPU job **127076572077**, started automatically. At 05:52:20 UTC its container `bmk-server-41967738227-1-mi355x-amd_pit2_p03_g07` was cloning/checking out the same candidate. New runtime directory: `/outputs/.native-fp8-validation/serve.jnGneVEQ/vllm`.

The user asked whether OFF equals ON and whether FP8 has no benefit. Answered candidly: current OFF 18266.89 and historical ON 18276.23 differ by only **0.05%**, but use different source revisions. **No current end-to-end benefit has been demonstrated.** Await matched ON at a03796b8; if similar, report no measurable C48+MTP benefit and do not present the PR as a demonstrated speedup. Profiling becomes the next investigation into any lost kernel savings, after the authorized review validation.

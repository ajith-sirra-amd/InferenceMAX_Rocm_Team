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

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

## OFF routing audit and ON startup, 06:05 UTC

The OFF raw server log has exactly eight NATIVE_FP8_EXECUTION events: one flag_off event per rank, flag disabled, BF16 incoming query, no graph capture or accuracy-stress routing, candidate a03796b8. The observer is installed in OFF mode and raises if the native direct projection, FlyDSL FP8 FMHA, or enabled native forward is invoked. Its final NATIVE_FP8_FLAG_OFF_PROOF_PASS covers all eight ranks. This is stronger than merely finding no FP8 log messages.

**OFF does not disable all FP8 attention.** The existing upstream ASM prefill uses FP8 Q/K/V and BF16 output when there is no cached context. With cached context, that ASM path is bypassed; the ordinary AITER backend uses BF16 for both cached-context attention and its new-token causal portion. The PR flag changes eligible cached-context projection/FMHA only. Sources at candidate a03796b8: rocm_aiter_mla.py lines 2840-2848, 3009-3021 and 3027-3050; prefill/aiter_flash_attn.py lines 103-158; mla_attention.py lines 3239-3315. Do not infer kernel input precision solely from the dtype logged at entry to forward_mha.

ON actual server environment verified native flag=1, candidate a03796b8, normal routing, native validation plugin and both BLAS selection flags=0. Its shell serve command exactly matches OFF. At 06:04:48 UTC the container is initializing, no server error matches or native execution events yet; benchmark has not started. Confirm projection and cached_fmha events on all ranks once requests run.

The primary PR comment, body and ASM reply were updated at 05:58 UTC to remove failed enclosing workflows from passing-validation claims. Explicit unresolved/historical failure references remain labeled as failures. The published head remains e2393d70.

The merge-conflict bot reported the published head at 04:35 UTC. Candidate a03796b8 already merges pinned main 69d27bf8. Current upstream main a2bcb565 is 77 commits ahead of that pin; the compare API shows no changes to the candidate's eight feature files in those commits. Check mergeability after any authorized candidate publication; do not change the frozen revision during the matched runs.

## ON attempt 1 startup failure; identical retry active

At 06:06:01 UTC, Triton failed while reading compiled `_dcp_a2a_unpack_combine_kernel` code during decode warmup: `OSError: [Errno 116] Stale file handle`. The cache was `/outputs/.native-fp8-validation/serve.jnGneVEQ/cache/triton`, backed by `/home` on `10.12.62.2:/amd-home` (NFS4). No native FP8 projection/FMHA events occurred. The matching 24 DCP cache files were all readable during follow-up inspection; transient NFS access or concurrent cache replacement is plausible, but the underlying cause is not proven.

The failed server was sent SIGTERM. When shutdown remained stuck, only its named benchmark container was stopped; GitHub uploaded the server log successfully and completed cleanup. Inspected server-log artifact: **13393465085**, five stale-file-handle text matches, zero native execution events. This attempt produced no performance result.

The same ON workflow's failed jobs were rerun without source, image, workload, routing or test changes. **Attempt 2 started 07:27:11 UTC**, GPU job **127111065570**. Container: `bmk-server-41967738227-2-mi355x-amd_pit2_p03_g07`; fresh runtime: `/outputs/.native-fp8-validation/serve.57QiqL2e/vllm`. At 07:30:35 UTC candidate installation completed and startup continued. Keep proof monitors pointed to attempt 2; the attempt-1 container/runtime no longer exist.

All three replacement validation workflows stopped at their predecessor gate when ON attempt 1 failed. Their failed jobs were also rerun: kernels, NIAH and serving are now on **attempt 2**, waiting for ON success. Their kernel/test acceptance remains unchanged. Reservation cutoff remains **13:42:35 UTC**, not extended by retries. If the stale-cache error repeats, isolate compiler caches before another retry rather than repeating indefinitely.

## Retry recovered through startup and entered workload warmup

At 07:40 UTC, attempt 2 had passed kernel warmup and reached 72% graph capture with zero server error matches. By 07:43 UTC the server was healthy (HTTP 200), graph capture and CPU KV offload initialization had completed, and the benchmark client was preparing the trace workload.

Workload warmup started at **07:44:51 UTC** with 51 mandatory snapshot primers and 10 additional requests per lane, **531** expected warmup completions. At **07:45:44 UTC**, the server log contained eight `projection` and eight `cached_fmha` events, one each for ranks 0-7. The installed validation function accepted their exact candidate SHA, FP8 types, nonempty inputs, non-capture execution and normal replay routing. This confirms real warmup execution of the optimized cached path; the one-hour measured phase had not started yet.

Actual server PID **1827**: flag=1, native candidate a03796b8, accuracy stress=0, native observer enabled, both BLAS preference flags=0. The retry's full serve command again compared equal to OFF. The fresh Triton cache is under `serve.57QiqL2e/cache/triton`.

CCD binding was inspected live at 07:46 UTC: ranks 0-7 (PIDs 2885,2887,2888,2886,2891,2884,2890,2889) have main-thread CPU masks **0-7, 8-15, 16-23, 24-31, 64-71, 72-79, 80-87, 88-95**, matching `results/ccdmap.txt`; memory-node masks are 0-1. No performance result is available yet. All three standalone review workflows remain waiting on attempt-2 ON success.

## ON measured interval started

Attempt 2 completed all **531** warmup requests at **08:02:51.715 UTC** in **1080.49 seconds** (no cutoff cancellations). The progress reporter's zero errors is not a statement about raw max_tokens=1 empty-content classifications; inspect raw records separately at completion, as for OFF.

The configured **3600-second measured interval started 08:02:51.848 UTC**, expected to finish sending around **09:02:52 UTC / 14:32:52 IST**, followed by drain, upload and cleanup. At 08:03:47 the reporter showed 28 successful measured requests and zero measured errors. No final throughput result yet.

For comparison only of setup timing, the inspected OFF benchmark log records warmup 04:24:38.350-04:45:55.066 UTC (**1276.72 seconds**). Do not treat warmup duration as the reported throughput benchmark. ON had a KDA Triton JIT warning and one-minute pauses with subsequent progress; a read-only GPU check showed all eight GPUs busy. No ERROR/Traceback recurrence was observed through measurement startup. CPU-offload stale-pin warnings are separate from the earlier NFS file-handle exception; the scheduler reports releasing those pins.

The live benchmark command preserves agentic trace timing (`--trace-idle-gap-cap-seconds 300`), C48, seed 42, 393 traces, 10 warmup requests per lane, synthetic DSpark acceptance and the 3600-second duration. Total input throughput includes cached prompt tokens; performance interpretation must retain that definition. Do not infer kernel speedup from total-token throughput alone.

## Current inspected status, 2026-11-05 09:49 UTC

- Full OFF and ON workflows are green at a03796b8. Matched total throughput is **18266.89472 -> 19000.97537 tokens/s/GPU (+4.01864%)**; output-only **126.61731 -> 132.29955 (+4.48773%)**. One pair only; variability unmeasured; total includes cached prompt tokens.
- ON final artifacts inspected: 13402275321 (normalized), 13401956368 (raw/proof), 13402420033 (server). 3984 measured successes; 7 InvalidInferenceResultError empty-content responses and 1 ClientOSError (request-body write on closing transport). Seven cutoff cancellations; cancellation-return grace expired. TTFT/ITL present on 3984/3984 successes. Zero server ERROR/Traceback matches. Final proof verifies nonempty FP8 projection and cached FMHA on ranks 0-7, normal routing, exact source a03796b8.
- NIAH 41968261654 green. Artifact 13403891073 inspected: main and candidate 12/12, matching case IDs/prompt hashes, 16K/32K contexts at depths 10/50/90%, two repeats, live MTP. Every candidate case has FP8 projection/FMHA on all eight ranks.
- Kernel/numerical/ASM 41968261662 green. Artifact 13403681986 inspected: JUnit exactly 12 passed, no skips/errors/failures; all 8 unchanged numerical/graph cases passed with zero projection clipping. Real-layer uncached ASM regression passed (12 heads, 128 tokens, max absolute error 0.0215493441).
- Serving 41968261826 in progress, GPU job 127151749910. Container pr64288-green-41968261826-2. At 09:47 UTC main arm was starting. Follow through; no replacement performance result yet.
- Updated primary reviewer comment 6473435649 and ASM/cache-scale replies 4466935267/4466935928 with inspected green evidence. Explicitly retain candidate-versus-published-head distinction. Published head still e2393d70; serving comparison and publication assessment pending.
- Optional four-line runtime-count optimization 768a5da2 remains isolated on perf/fp8-cache-runtime-counts-20261105, not the PR candidate. Static workflow 41993124155 green, artifacts not yet inspected. No GPU validation dispatched for this optimization.

## Cache-capacity investigation and correction

The user specifically challenged the historical **22M+ -> 19.8M** change. Do not confuse it with ON/OFF or NIAH/performance comparisons.

| Run | Source / compiled base | Reported capacity | Cache budget |
| --- | --- | ---: | ---: |
| Prototype 40695099743 | image 1b2a66639 + prototype | 22,695,620 | 53.81 GiB initial |
| Earlier native 41447686785 | 492080779 / a1cfd2c1 | 22,524,219 | 53.39 GiB |
| Current OFF 41964736019 | a03796b8 / 69d27bf8 | 19,834,746 | 2024 blocks, 48.36 GiB logical / 48.42 GiB physical |
| Current ON 41967738227 attempt 2 | a03796b8 / 69d27bf8 | 19,834,746 | same 2024 blocks |
| NIAH candidate | a03796b8 / 69d27bf8 | 5,021,999 | 2069 blocks, 49.44 GiB logical |

**Verified identical container digest** for earlier native, current OFF and current ON:
vllm/vllm-openai-rocm:nightly-rocm100-e0d00e4b8eb0c89ec30921d0695cfdd409e9c211@sha256:b1374c3814b6e7706c49968ba17ca3b35484382f7b9fc7115dd3e66068d53f2f.
Their installed vLLM sources/binaries differ. The model weights per worker remain approximately 184.49 vs184.48 GiB. Old and new recorded startup free memory agree (e.g. rank0 273.53/287.98 GiB). Same TP8/DCP8/C48 and FP8 stored KV.

Capacity is int(max_model_len * num_blocks / sum(per-request group blocks)). Arithmetic reproduces:
- old native: int(1048576 * 2234 / 104) = 22524219;
- current full: int(1048576 * 2024 / 107) = 19834746;
- current NIAH: int(65536 * 2069 / 27) = 5021999.

For full attention the per-request requirement is ceil(1048576 / 8 / 1536)=86 blocks (NIAH:6). Three KDA groups account for18 old versus21 current. Upstream **#63297**, e3eec515, enables prefill checkpoint blocks on the gluon KDA backend; old log has gluon without checkpoint enabled, new log explicitly prints KDA prefill checkpoint enabled. New KDA file is byte-identical to pinned upstream 69d27bf8. The extra checkpoint changes the per-request denominator from104 to107; it is not our attention flag.

Upstream **#56492** introduced extensible cache sizing from actual post-warmup memory, now used in the current runs. The earlier native run used the older pre-allocation profiling path. The new allocator keeps measured transient headroom (2.51 GiB in both ON and OFF) and selects the minimum rank block budget. Current FP8 flag does not modify cached latent storage or native weight quantization, so equal ON/OFF capacity is consistent. Temporary expanded K/V savings did not increase the observed startup block budget. Do not assert that every byte of the historical delta is causally isolated: a matched startup-only old/new allocator ablation has not been run.

Important older-log detail: after final graph capture the earlier native run itself advised **46.18 GiB** KV to fit requested utilization, while it actually allocated **53.39 GiB**. Logged actual usage:188.98 GiB weights/non-Torch,16.82 GiB peak activation,7.06 GiB graphs. New final physical allocation is48.42 GiB. The old/new reported available-memory lines are from different sizing stages; initial67.90 GiB current is not the final cache allocation.

Upstream #65418 raises the ROCm margin floor to3 GiB, but the formula applies it to physical headroom and only the smaller budget binds; **do not attribute3 GiB of the loss to that margin without checking which constraint binds**.

User asked why vLLM was reinstalled. Explained: PR source installation tests unmerged code; rebasing to newer upstream was a separate choice for reviewer current-main comparison/merge-conflict resolution, and matching binaries avoid source/extension mismatch. Updating upstream is not inherently required by the FP8 optimization. Acknowledged this should have been explicit before comparing historical results. Need keep runtime revisions explicit in PR and future reports.

Reservation remains unchanged: stop GPU work by13:42:35 UTC.

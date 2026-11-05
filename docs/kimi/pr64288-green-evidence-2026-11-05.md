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

## Serving setup failure and compiler-cache isolation, 09:56 UTC

Serving run41968261826 attempt2 failed during **unmodified upstream main** startup. Worker TP5 reported Inductor/Triton compilation failure for triton_poi_fused__to_copy_mul_sigmoid_slice_tanh_0 at09:52:09 UTC; the first exception is OSError [Errno116] Stale file handle while CompiledKernel reads compiled binary bytes. No serving cases had completed. Evidence persisted in artifact13403514638. The failed container was stopped by exact name; no other container was touched.

This repeated the class of NFS compiler-cache failure seen in the ON replay. Changed **only the validation harness** in commit49552a38912be4551bfb9986b54a005c9661c745:
- New optional REVIEW_COMPILER_CACHE_ROOT is set to /tmp/pr64288-compiler-cache for the serving workflow.
- Each arm gets separate local TRITON_CACHE_DIR, TORCHINDUCTOR_CACHE_DIR and VLLM_CACHE_ROOT, with filesystem type and paths saved in compiler-cache.json. NFS is rejected.
- Startup polling now fails promptly upon this exact fatal exception, rather than waiting for the API server's health timeout.
- Python AST checks verified unchanged CASES, bench, bench_command and execution_proof.
- Native candidatea037, binary/base69d27, containerdigest, model config, prompts, concurrencies and tolerances remain unchanged.

Replacement serving run **42006336780** started automatically from the harness push. Current container name will be pr64288-green-42006336780-1. Monitor that run; old serving monitors are obsolete. Do not cite failed41968261826 as passing validation. Cutoff remains13:42:35 UTC.

User has repeatedly asked about the historical cache decline and runtime reinstall. Responded with the original document's timeline: prototype vLLM1b2a66639 at17406TPS/22695620tokens; earlier native492080779+a1cfd2c1 at18276TPS/22524219tokens; currenta037+69d27 at18267OFF/19001ON and19834746tokens. Same e0d00e4b image digest for the two native generations, different installed runtime. The historical 17,406-to19,001 difference must not be attributed only to this PR. Current matched OFF/ON is the observed4.02% flag comparison.

All five inspected files (cache utils/interface, extensible allocator, gpu_worker, AMD KDA) are byte-identical between candidatea037 and upstream69d27. This attributes the cache-policy code changes to upstream, but does not isolate every byte of memory difference with a GPU allocator ablation.


## Published candidate, memory audit and current-main conflict, 10:24 UTC

Published PR64288 head is now a03796b8e3bfe536e22f9db741109dbf80170079. Description and reviewer replies reflect its inspected green accuracy/numerical/static/full-performance evidence. Main/OFF/ON serving remains pending. The PR is open, not draft; upstream CI authorization remains external.

Serving42006336780 at harness49552a38912be4551bfb9986b54a005c9661c745 is healthy on container-local overlayfs compiler caches. Main has completed short_context_c8 (32/32), short_query_c8 (32/32), cached_c8 (64/64), and is running cached_c48. No stale-file-handle recurrence. The first three results alone are not a comparison or a final workflow pass.

Memory audit42008641826 at harness2919dbdb42f430b62f4f228e3436d2d41e5e96ea passed code checks and waits on serving. Startup-only main OFF/candidate ON, same a037/69d27, fixed GMU0.90, explicit load-format auto, TP8/DCP8, full1M max length and CPU KV offload. Read-only per-rank allocator/storage snapshots at load, memory profiling and final warmup. No allocator resets or GPU arithmetic. Do not sum virtual KV storage as physical memory.

Both current OFF and ON use FP8 latent cache AND native per-channel FP8 kv_b_proj weights. Original historical baseline used BF16 projection weights. Current OFF runs native FP8 linear with dynamic activation quantization and emits BF16 K/V; ON directly emits FP8 K/V. The current pair isolates the attention flag, not weight quantization.

GitHub PR base.sha remained69d27 even while actual main had advanced toee85faa1280c3f9ef877c7b17d1a3ef113133e57 (107 commits ahead). Always inspect git/ref/heads/main as well as pull.base.sha. The earlier compare69d27...44d9 was behind19, explaining its zero added commits/files; it was not current main.

Upstream PR59124 merged at2026-11-05T09:21:40Z, commit7187c01de54aadcf28367f53284eda95c28778fd. Its added imports/forward routing plus another upstream initialization change cause three textual conflicts in rocm_aiter_mla.py. All other feature files merge cleanly.

Prepared separate integration candidate54f39769d0e17a8f2874695e849c4e6626e71b56 on integrate/pr64288-main-20261105, parentsa037 andee85. Exactly8 files differ fromee85. Keeps both sides of additive conflicts and excludes the upstream superseding FlyDSL backend from our two direct-projection routing guards. Native FP8 kernels and numerical test file are byte-identical toa037. No GPU validation yet; do not publish this revision or transfer old result claims to it.

Static/docs harness ea6bc5839134953ab9dc43e69aff725868f53175 on checks/pr64288-integrate-static-20261105 targets54f39769 againstee85. It runs independently on hosted CPU runners, leaving the active GPU source unchanged. Inspect result and artifacts.

PR description and related-PR comment6473674684 now state that59124 merged after the validated baseline and current-main integration/revalidation remains pending. No current-main gain is claimed.

Keep GPU cutoff13:42:35UTC (19:12:35IST), reservation12214 ends13:52:35UTC. User explicitly requires GMU0.90 and preserving load-formatauto. Next: finish serving; inspect memory audit; validate backend coexistence for integration candidate; publish only after applicable checks. Profiling and C56/C96 remain later.


## Memory audit inventory fix and integration static results, 10:33 UTC

Replaced queued memory run42008641826 before any GPU job began (now cancelled). Audit traversal previously skipped callable torch.nn.Module instances, missing model weights/buffers, and omitted the global workspace manager. Corrected in harnessf8f707ea2aea2d633aea7506915717dd833eacff:
- Traverse real torch.nn.Module instances plus vLLM object dictionaries, with explicit depth-limit accounting.
- Include current_workspace_manager as a separate root; deduplicate aliased storages.
- Label peak counters since last allocator reset correctly.
- Record true CPU-offload/synthetic-acceptance/GMU/load-format settings even if the audit fails.
CPU validation with a real Sequential/Linear model and shared buffer found exactly three unique storages (weight48bytes, bias12bytes, shared workspace64bytes), without initializing CUDA.

Replacement audit42010811107 passed code checks and waits on serving42006336780. Sourcea037, base69d27, recipe and GPU cutoff unchanged.

Integration static/docs run42010320935 on fork passed for54f39769. Inspected artifact13406900964: exact8-file scope, applicable pre-commit/Ruff/mypy/CI tethering passed. Inspected rendered-docs artifact13406259759: strict build completed, source matches54f39769. This is CPU/static evidence only.

Exact matching ROCm100 wheel metadata for integration baseee85faa1280c3f9ef877c7b17d1a3ef113133e57 returns HTTP404 at10:32UTC. Do not silently use69d27 compiled binaries with the new source. Current pinned serving and memory runs remain valid and unchanged. Recheck matching wheel availability before any integration GPU dispatch.

Audited all workflow hyperlinks in the PR body and author comments at10:30UTC: all11 referenced runs currently have conclusion success. The short related-PR note now mentions the merge without linking the failed historical reference run.

Potential later memory experiment (not dispatched): full recipe captures every even graph size2..480 with max-num-seqs96. C48 plus4draft tokens schedules at most240 decode tokens for48 live requests; a bounded graph-capture-size reduction may reclaim memory while retaining GMU0.90. First inspect audit evidence and fallback behavior; do not change current comparisons or claim a saving without measurement.


## Serving main completed; OFF controls in progress, 11:17 UTC

Serving42006336780 remains active. Main69d27 completed all5 cases,512/512 requests, with prefill/context execution proof on ranks0-7 and zero server ERROR/Traceback matches. Main shutdown emitted a shared_memory resource-tracker warning; no request failure was observed.

| Case | Main duration (s) | OFF duration (s), if completed | Requests per arm |
| --- | ---: | ---: | ---: |
| short_context_c8 | 190.690330 | 195.536790 | 32 |
| short_query_c8 | 126.133210 | 133.309624 | 32 |
| cached_c8 | 351.133431 | 343.685766 | 64 |
| cached_c48 | 769.919844 | pending | 192 |
| long_cached_c48 | 1284.139677 | pending | 192 |

OFF at publisheda037 started its server around10:57UTC and is now beginning cached_c48. Its first3 cases total128/128 successful requests. Both below-cutoff proofs explicitly show BF16 Q/K/V context attention on all8 ranks; short-query proof has max_query_len<512 and cached context>=8192. Full OFF/ON comparison and enclosing workflow pass are still pending. Do not link this active workflow as passing PR evidence yet.

Live OFF server PID10627 was checked: native attention flag0; sourcea037; both BLAS preference flags0; its Triton cache is separate under/tmp/pr64288-compiler-cache/candidate_off. Runtime source and test cases unchanged. Current-case HTTP200 log counts mark started streaming responses, not completed requests; use result JSON completed counts for final success claims.

At11:09UTC integration-baseee85 exact wheel metadata still returned404. The public ROCm100 nightly alias points toa2bcb5657a881a002e79d0adab3c6b5e39436121, which predates59124. Do not use it as matching binaries for integration54f39769. No new human reviewer comments were found in checks through11:12UTC.

Replacement memory audit42010811107 atf8f707ea remains waiting behind serving. GPU cutoff13:42:35UTC unchanged. After serving, inspect artifacts and update PR replies, then follow the memory audit. Any graph-size experiment remains separate, conditional on evidence and available reservation time; preserve existing performance recipe until validated.

## Serving progress and follow-up scope, 12:34 UTC

Serving42006336780 is on its final ON long_cached_c48 case. Main and OFF each completed512/512 requests; ON completed320/320 in its first four cases. Final case is progressing, without inspected server errors. HTTP200 stream starts are not completion counts.

| Case | Main output tokens/s | OFF output tokens/s | ON output tokens/s | ON vs OFF |
| --- | ---: | ---: | ---: | ---: |
| short_context_c8 | 21.47985 | 20.94746 | 21.72310 | +3.70% |
| short_query_c8 | 32.47360 | 30.72546 | 35.14914 | +14.40% |
| cached_c8 | 23.33016 | 23.83573 | 22.11723 | -7.21% |
| cached_c48 | 31.92021 | 33.89550 | 32.34014 | -4.59% |
| long_cached_c48 | 19.13811 | 19.61760 | pending | pending |

Numbers are for the whole server, not per GPU. One pass only. Below-cutoff cases execute BF16 and cannot establish FP8 speedup. Eligible ON cases have real FP8 projection/FMHA proof on ranks0-7. Do not cite the unfinished workflow as passing. Keep these mixed results alongside the separate full-replay +4.02% result, whose CPU offload/synthetic DSpark workload differs.

User questioned two-hour GPU duration. Matrix has three startups and fifteen benchmark cases. Each control's final case took about21minutes; ON final started around12:24UTC. Do not shorten current cases or relax acceptance.

Periodic DSpark counts (including warmup, not exact measured-only counters) yield cached_c8 approximate acceptance length OFF1.36773/ON1.36681 and cached_c48 OFF1.37399/ON1.37672. No evidence of a large acceptance collapse. Cause of serving regression remains unproven.

ON Triton cache had8 variants each of _q_and_cache_partials/_q_quant_and_cache_reduce after cached_c8,15 after cached_c48,17 during the long case. This motivates isolated runtime-count candidate768a5da2 but does not measure compilation overhead or establish causality. GPU validation remains pending.

Memory42010811107 still waits for serving; no memory GPU job has begun. Integration54f39769 has inspected passing static/docs evidence, but no GPU validation. Exact ee85 ROCm100 wheel still404 at12:33UTC; nightly stilla2bcb5657 (pre59124). Do not pair incompatible sources/binaries or transfer results to unvalidated integration. PR head remainsa037. GPU cutoff13:42:35UTC, GMU0.90, load-formatauto unchanged.

## Serving completed and PR updated, 12:50 UTC

Serving42006336780 completed successfully. Inspected artifact13415946292: all1536 requests completed (512 each for pinned base69d27, candidatea037 OFF, candidatea037 ON), all12 setup kernel tests passed, identical serve commands, expected source manifests, no server ERROR/Traceback matches, and genuine non-capture FP8 projection/FMHA on ranks0-7 in all3 eligible ON cases. Both below-cutoff cases retained BF16.

Final long_cached_c48 ON duration1305.905344s, output18.819128 tokens/s for the whole server; OFF19.617601, base19.138105. ON vs OFF output throughput: cached_c8 -7.21%, cached_c48 -4.59%, long_cached_c48 -4.07%. One pass per arm; variability unmeasured. Cause remains unproven. This is different from the full agentic replay (+4.02% total), which uses CPU offload/synthetic acceptance.

Updated PR64288 description, primary validation comment6473435649 and ASM/cache-scale replies4466935267/4466935928. Description now contains all5 throughput rows plus mean/P99 TTFT and mean TPOT, prominently discloses serving regressions, and distinguishes execution success from performance improvement. Tagged seungrokj/frida-andersson in the updated primary comment as previously authorized. Rechecked all12 workflow links used in the body/primary comment: completed/success. Current-main integration and upstream CI remain pending; published head remainsa037.

Memory42010811107 now has active GPU job127253927549, containerpr64288-green-42010811107-1. At12:49UTC it is installing the exact vLLM environment before model startup. Use memoryMonitorCommand, not the obsolete serving container monitor.

Queued isolated runtime-count diagnostic42028735939 behind memory success. Harness545886933c95499c50efe81d2ff4fe09d59543ce on checks/pr64288-runtime-counts-20261105; native768a5da2 vs parenta037, same compiled base69d27. Hosted code checks passed. Runs unchanged12 kernel tests,8 numerical/graph cases and ASM regression, then a single-GPU parent/candidate exact-output comparison across21 cache-size cases (empty/tail/power-of-two boundaries) and exploratory timings. New script utils/atom_attention/pr64288_runtime_counts_diagnostic.py; wrapper pr64288_runtime_counts_validation.sh; workflow pr64288-green-runtime-counts.yml. GPU step capped900seconds and requires1200seconds of reservation headroom to start. No model serving run and no PR-head update for this optional candidate. Timing is exploratory, not a serving claim.

The serving watcher stopped normally when its container disappeared after successful cleanup. No watcher is currently active. Resume two-minute monitoring of memory, then runtime-count diagnostic. Cutoff13:42:35UTC remains binding.

## Memory baseline measured and graph-cap experiment queued, 13:04 UTC

Memory audit42010811107 upstream69d27 startup reproduced19834746KV tokens,2024blocks/48.42GiB physical, at GMU0.90/loadauto without the PR. Model load184.48GiB/rank. All24 base snapshots present, traversal untruncated with zero depth-limit hits. After warmup Torch allocated188.2071GiB, reserved196.4785GiB, peak-since-reset190.3434GiB; device free28.4883-30.6758GiB. Graph capture84seconds,7.24GiB/rank. Final sizing retains2.51GiB transient headroom and takes the minimum rank's2024blocks. Candidate ON started13:03UTC; wait for its final snapshot.

The snapshot workspace_storages heuristic matches any occurrence of metadata/buffer/workspace in an alias path. Some model parameter aliases under metadata_builders are therefore listed as workspaces and fill its100-entry limit. Do NOT treat that list as a memory-category total. Allocator counters and graph/cache logs remain valid; no false workspace attribution was published. The graph-cap follow-up retains the full storage inventory and matches only owning tensor fields or the workspace-manager root.

Queued startup-only C48 graph-cap experiment42032344547, harness6cfc07b7451b8aec2c7104aa9a590426acef4eee on checks/pr64288-memory-graphcap-20261105. It waits for runtime-count diagnostic42028735939 at harness545886. Native source remains publisheda037 and compiled base69d27. Only compilation capture cap480->240 and associated even sizes2..240 change. Max-num-seqs96, max-context1M, CPU offload, synthetic MTP, GMU0.90/loadauto retained. C48 with4draft tokens needs at most48*5=240 verify tokens. CPU AST test compared old/new command transforms: every other command argument and compilation field identical. This measures startup capacity only, not serving performance or accuracy. Step cap900seconds, minimum1200seconds headroom; stop by13:42:35UTC.

Exact integration-baseee85 ROCm100 wheel remained404 at12:55UTC. No new human review comments then.

## Memory audit verified and kernel diagnostic running, 13:17 UTC

Memory42010811107 completed successfully; inspected artifact13418055772. Both arms have24snapshots (3phases x8ranks), untruncated traversal and zero depth-limit hits. Identical serve commands, GMU0.90/loadauto, exacta037/69d27 provenance and no server ERROR/Traceback matches. Both report19834746KV tokens and7.24GiB graph memory.

Every rank at every phase has ON-minus-base allocated and peak-since-reset memory of exactly12288bytes, reserved difference0, and free-device-memory difference-10485760bytes. This is a startup measurement, not steady-state serving allocation. It shows no GiB-scale retained-weight difference or startup-memory saving from the flag. Workspace labels in this audit remain heuristic/alias-contaminated; counters and graph/cache logs are the reliable evidence.

PR description now links this passing fixed-GMU audit in its existing no-capacity-gain statement. Primary serving update remains unchanged; no unvalidated experimental code was published to the PR.

Runtime-count diagnostic42028735939 started its GPU job at roughly13:15UTC on768a5da2. Native parenta037, compiled base69d27, step cap900seconds. Monitor runtimeMonitorCommand and runtimeWatchCode. Graph-cap experiment42032344547 has passing hosted code checks and waits for runtime diagnostic success. Its GPU job requires>1200seconds reservation headroom and caps execution900seconds. If insufficient time remains, do not bypass the cutoff.

Serving artifact first observed rank0 FP8 shapes: cached_c8 and cached_c48 use query512/context16896; long_cached_c48 uses query1024/context33792. Proof also records BF16 fallback chunks (context7680). These are first-seen examples, not execution-frequency distributions. They are useful for a future targeted pipeline profile; do not claim the entire cases executed exclusively in FP8.

Stop GPU work by13:42:35UTC. Current-main integration still needs matching binaries and GPU validation; published PR heada037 remains unmergeable against the newly advanced main. Serving regressions remain unresolved, despite successful accuracy/execution checks.

## Runtime-count diagnostic verified; final memory experiment running, 13:23 UTC

Runtime-count workflow42028735939 is green. Inspected artifact13417907202: native768a5da2/compiled69d27,12kernel tests,8numerical/graph cases, ASM regression (max absolute0.0215493441),21boundary cases exactly match parenta037. First-call totals across the diagnostic shapes: parent5688.47ms, candidate1948.45ms; includes compilation/allocation and both preparation helpers. This is not pure compilation time or serving throughput.

Exploratory median steady prepare times in ms:
- Q512/cache8192: parent0.044680, candidate0.044617 (-0.14% time).
- Q8192/cache16384:0.044496 ->0.045431 (+2.10% time).
- Q8192/cache32768:0.044345 ->0.044489 (+0.32% time).
- Q8192/cache131072:0.045409 ->0.047564 (+4.74% time).

No steady-state improvement established. Keep768a5da2 isolated, do not publish it as the PR head or claim it fixes the serving regressions. Numerical acceptance remained unchanged. A future targeted profile should use the observed512/16896 and1024/33792 query/context shapes and compare the full native-OFF/ON pipeline.

Graph-cap42032344547 started GPU job127271019241 before its minimum-headroom gate closed. Containerpr64288-green-42032344547-1. It uses publisheda037/69d27, capture cap240, max-num-seqs96, full1M context and CPU offload at GMU0.90/loadauto. At13:22UTC environment setup was in progress. Step capped900seconds. This is the final GPU experiment in the reservation; stop by13:42:35UTC.

All earlier three workflows (serving, fixed-GMU memory, runtime-count diagnostic) completed successfully. PR description contains serving and baseline-memory evidence; experimental768 code is not published. Use graphCapMonitorCommand/graphCapWatchCode to follow the final experiment and inspect its artifact before claiming a capacity change.

## Reservation work stopped; final status, 13:37 UTC

Cancelled graph-cap experiment42032344547 at13:35UTC while checkpoint prefetch was still delaying model load. A rank reported330.46seconds of prefetch; no load-model snapshot or final capacity result was available. Host inspection at13:34 showed abundant available RAM (~3072945284kB), negligible memory pressure, /dev/shm1%used, and I/O pressure avg60~31%. This supports an I/O delay observation, not a kernel failure or disk-quota diagnosis. The remaining900-second step budget could not reasonably cover loading, warmup and CPU-offload initialization. No timeout or acceptance gate was weakened.

GitHub cancellation accepted and the exact containerpr64288-green-42032344547-1 was explicitly stopped. At13:36:42UTC, remote Docker reported no running containers and ps reported no VLLM worker processes. All four workflows are terminal:
- Serving42006336780: success; artifact13415946292 inspected.
- Fixed-GMU memory42010811107: success; artifact13418055772 inspected.
- Runtime-count diagnostic42028735939: success; artifact13417907202 inspected.
- Graph-cap42032344547: cancelled during startup; no capacity/performance/accuracy result. It is not cited as passing PR evidence.

All13 workflow links in PR64288 description and author comments were rechecked at13:36UTC: completed/success. PR heada03796b8 remains open, not draft, and unmergeable against current main. Description and reviewer replies include measured serving regressions and the passing baseline-memory audit. Runtime-count candidate768a5da2 remains isolated; no steady-state speedup was demonstrated.

Remaining work, in order:
1. Profile/diagnose the cached-serving regressions on the observed Q/context shapes (512/16896 and1024/33792), then validate any change with matched measurements.
2. Finish current-main integration54f39769 with matching ROCm binaries and GPU checks; upstream CI authorization and env-interface discussion remain unresolved. Do not request prohibited CI labels.
3. Retry the isolated graph-cap240 startup experiment in a new reserved GPU window; follow with serving/accuracy validation before adopting the configuration. GMU0.90 and loadauto remain fixed.
4. Broader steady-state agentic profiling and C56/C64+MTP, C96 without MTP sweeps remain later.

No polling cell remains active and no GPU job is queued. Reservation cutoff13:42:35UTC was respected. Do not dispatch more server work in this reservation or rerun these workflows after their cutoff. Future GPU work requires an authorized reservation window and updated explicit cutoff. Local independent code review can proceed.

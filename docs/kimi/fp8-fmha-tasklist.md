# Kimi FP8 task list

**English** | [中文](./fp8-fmha-tasklist_zh.md)


## Runtime fix and native accuracy retry: 2026-10-30

The native PR remains draft at `2f946cbab5000dedef23b117df991417e8917315`.
Eight native GPU kernel tests and current static checks passed. There is still
no native GSM8K score or native serving-performance result.

Run [41051436780](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/41051436780)
confirmed the source-import and IPC-path fixes, then failed before model execution
at the unchanged worker's 16x16 BF16 `torch.mm(x, x)). ROCm reported
`device kernel image is invalid` and `HIPBLAS_STATUS_INTERNAL_ERROR`.

Three completed diagnostics isolate the validation harness's `TMPDIR` override:

- [41052806908](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/41052806908):
  BF16/FP32/FP64 GEMM passed in both images on the current runner. Its BF16
  transposed-operand test was not identical to worker initialization.
- [41054279799](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/41054279799):
  exact worker BF16 GEMM, transposed BF16, FP32 and FP64 pass with default
  settings in both system Python and a virtual environment. All four fail with
  native harness environment overrides; loaded ROCm library paths match.
- [41054684302](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/41054684302):
  `TMPDIR` alone reproduces the failure. The other eight overrides individually
  pass. Removing only `TMPDIR` from the full native environment makes all four
  cases pass; pre-creating cache directories does not fix it.

Harness fix [941b262](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/commit/941b262404f15d5c55874304dfc54e655cf53010)
preserves the container's temporary directory and records the resolved directory.
It does not alter the PR, ROCm libraries, worker initialization, or kernel routing.
The precise filesystem/runtime mechanism behind the failing temporary path has
not been established. No host or driver change was made.

[Harness CI 41055874246](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/41055874246)
passed 20 tests with zero failures/errors/skips and the applicable shell checks.
Full native [GSM8K retry 41055998426](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/41055998426)
is dispatched with this exact harness revision and PR `2f946cb`: fixed 1319
questions, five-shot, temperature zero, C48 + real MTP verification,
eager/512-token accuracy stress, GPU synchronization checks, and optimized-path
execution proof required on all eight workers. New-token attention remains BF16.
Dispatch is not an accuracy pass.

No `Disk quota exceeded`, `No space left on device`, or `EDQUOT` text was found
in the job logs of 41046329246, 41048209579, 41051436780, 41052672056 or 41052806908.
This is a log search, not an account-quota measurement.

The earlier new-token FP8 performance regression remains separate and unexplained.
#59124 reports +4.5% for new-token plus cached-context FP8 without MTP; our
17,406.08912 to 15,264.82341 comparison adds causal FP8 to an already optimized
projection/cached-FMHA path with MTP. Our causal call reuses prepared FP8 Q/K/V,
uses `cross_seqlen=True` like #59124, and introduces no extra quantization pass.
Actual shape frequencies and matched per-call timings are still needed.
The historical captured-BF16 absolute LSE <0.20 gate remains unresolved.

[Runtime evidence](../../results/atom-attention/run41054684302/runtime-isolation.json)
and [retry record](../../results/atom-attention/run41055998426/dispatch.json).

## Earlier native validation: 2026-10-30

- Native [GPU run 41044413189](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/41044413189): **8 passed, 0 failed, 0 errors, 0 skips**, at PR `332d6c9`. Quantization, plain/shuffled projection and FMHA, changed-input graph replay, native weight loading and state cleanup passed. This is GPU-0 kernel evidence, not eight-worker model execution.
- Current PR `2f946cbab5000dedef23b117df991417e8917315` formats two test expressions only; executable AST matches the GPU-tested test and production files are unchanged.
- [Static run 41046262627](https://github.com/ajith-sirra-amd/vllm/actions/runs/41046262627): all applicable targeted pre-commit hooks, Ruff, mypy, Markdown, SPDX, Buildkite registration, AST/YAML and exact-source checks passed. Artifact `12973200163`.
- Upstream pre-commit requires contributor approval. ReadTheDocs build 35521392 stopped on this failed prerequisite before building the docs. Fork checks do not replace upstream CI.
- Full native [GSM8K run 41051436780](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/41051436780): dispatched at PR `2f946cb`, harness `415ea45b8a8a177c01c14f4559b6f40cc6f4fcb0`. Fixed 1,319 questions, five-shot, temperature zero, C48 + MTP with real verification; eager/512-token accuracy stress routing; `VLLM_GPU_SYNC_CHECK=error`; requires FP8 projection/cached-FMHA proof on eight workers. New-token attention remains BF16. No native GSM8K score yet.

[Kernel evidence](../../results/atom-attention/run41044413189/native-kernel-evidence.json)
and [current dispatch](../../results/atom-attention/run41051436780/dispatch.json).

### Serving setup fixes, without changing native kernels

[41046329246](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/41046329246)
failed before model execution: the image package shadowed editable native vLLM
outside the source checkout. Fix `cf766e9` sets only the pinned checkout on
`PYTHONPATH`, checks vLLM/CLI/FP8 imports from outside the checkout, and exports
that source path to workers. [Failure evidence](../../results/atom-attention/run41046329246/setup-failure.json).

[41048209579](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/41048209579)
confirmed that import fix and resolved Kimi/DSpark configuration, then failed before
model execution because the IPC path exceeded Linux's 107-byte socket-path limit.
Fix `415ea45` uses a short unique repository-local working directory.
The installation manifest now uploads with the artifacts. Neither attempt produced
a GSM8K result. [Failure evidence](../../results/atom-attention/run41048209579/setup-failure.json).
[Final harness CI 41051172001](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/41051172001)
passed 20 tests with no failures/errors/skips, plus recipe/launcher and workflow Bash checks.

Earlier kernel runs passed 5/8 and 7/8. CPU FP64 reference calculations and the
existing distributed fixture fixed those reference/setup failures; GPU kernels,
graph replay, cases and tolerances are preserved. Fresh-process FP32/FP64 BLAS
errors also reproduce before importing FP8 kernels, including with
`ROCBLAS_USE_HIPBLASLT=0`; native serving compatibility remains to be established.

The historical captured-BF16 absolute LSE <0.20 gate is unresolved. New-token FP8
is excluded after its 12.30182% regression. PR #64288 stays draft. Required next
work: native model/runtime validation, numerical closure, native performance and
matched reviewer comparisons; then steady-state profiling and C56/C64 + MTP,
C96 without MTP. #62447 is now closed/unmerged; its pinned implementation remains
the requested comparison reference. Native NUMA changes remain postponed.

## Result and native candidate: 2026-10-30

### Native validation setup corrected

Both setup fixes passed [GitHub CPU CI 41037738077](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/41037738077):
20 tests, no failures/errors/skips, plus native-workflow Bash syntax checks.
The second GPU attempt, 41037273374, failed at checkout because the first
diagnostic container left root-owned scratch files. Workflow revision
`57cb67a739dd869e8f54e559bbb56541f741f5d5` restores runner ownership before
checkout and after validation, only inside repository-local `.native-fp8-validation`,
without following symlinks or deleting files.

[Fresh native GPU run 41037832511](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/41037832511) targets the unchanged PR revision
`08e5d81a0d4fc76c20a97f9a517195db5a5dc5be`, with causal FP8 excluded.
The ownership restoration step passed; native numerical results are pending.
[Dispatch record](../../results/atom-attention/run41037832511/dispatch.json).

[Attempt 2](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/40804432321/job/123956986964) completed successfully with no quota annotations.
Full C48 + MTP new-token FP8 measured **15,264.82341 total tokens/s/GPU**,
a **12.30182% regression** against **17,406.08912**.

| Metric | Established cached FP8 | With new-token FP8 |
| --- | ---: | ---: |
| Total tokens/s/GPU | 17,406.08912 | 15,264.82341 |
| Output tokens/s/GPU | 120.16691 | 100.53104 |
| Mean TTFT, seconds | 1.76809 | 2.01387 |
| p95 TTFT, seconds | 4.36319 | 5.43159 |
| Mean TPOT, milliseconds | 23.15 | 28.61 |
| Successful measured requests | 3,644 | 3,319 |
| Measured errors | 5 / 3,649 | 8 / 3,327 |

Server logs verify FP8 projection, cached FMHA and causal FMHA on all eight
workers. Causal attention reuses prepared FP8 Q/K/V; it adds no second quantization
pass. Actual server configuration confirms DSpark MTP, despite the summary's
inconsistent `spec_decoding=none` field. The current host is `pit2-p03-g25`;
the established host was `pit2-p03-g15`. The user confirmed identical servers
and declined an extra server/control run. Request populations, output lengths
and cache hits differ, so this is not an isolated causal-kernel measurement.

**Decision:** do not promote new-token FP8. Keep cached-only native PR #64288
at `08e5d81a0d4fc76c20a97f9a517195db5a5dc5be`.
[Native kernel validation 41036211097](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/41036211097) passed recipe/launcher preflight
but failed during wheel resolution, before numerical tests. HIP component
version `7.15.26333` was incorrectly used to request `rocm715`; the ROCm 10
Torch package identifies the correct `rocm100` release. The exact-base
`a1cfd2c1b584c4ac197db9cc9ac84a6d973d39df` rocm100 metadata exists.
Harness fix `e7b3309859056b3bd39d7c54bd347f0df4e77244` derives the release
from the Torch package and checks exact installed distribution versions.
The old resolver reproduced HTTP 404; the corrected resolver passed that case.
Full CPU checks precede the next GPU attempt; native kernel/model results
remain pending. The original captured-BF16 LSE gate remains unresolved.

[Performance evidence](../../results/atom-attention/run40804432321/attempt2-performance-evidence.json).

## Resumed retry: 2026-10-30

The user restarted [run 40804432321, attempt 2](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/40804432321/job/123956986964).
At **2026-10-30T08:17:46.011600+00:00**, GPU job **123956986964** is **in progress**;
runner 35 is online and busy. Cleanup and checkout passed; `Launch job script`
is active. The retry currently has no failure annotations. GitHub's downloadable
job log is not available yet, so this status alone does not prove GPU execution.

The rerun preserves workflow source `75a52838ee8fa89461dc812913eb71b370b839bd`
and serving source `bb8bed732f514fd697856c5abd1d2bdd9d56a1e5`: full 3600-second
C48 + MTP replay with FP8 up-projection, cached FP8 FMHA and new-token causal FP8.
No duplicate benchmark was dispatched. The prior attempt failed with quota errors
in check-run annotations; it produced no accepted new-token performance result.

Next: collect attempt-2 benchmark, raw-request and server artifacts; verify execution
and compare throughput, latency and measured errors against **17,406.08912**
total tokens/s/GPU. Select the native candidate only after this comparison, then
run native GPU kernel validation, full GSM8K and full performance in order.
The already-passed prototype GSM8K is not being repeated. No native NUMA change
or serving smoke is planned.

A session-bound poller checks every five minutes and ends on completion or after
three hours. It does not wake a finished model turn or install a system scheduler.
The exact native PR remains draft at `08e5d81a0d4fc76c20a97f9a517195db5a5dc5be`;
its upstream pre-commit stopped at the contributor approval gate before code checks.
Native GPU evidence and the captured-BF16 LSE criterion remain outstanding.

## Next dispatch: user decision on 2026-10-29

The user confirmed the replacement servers are identical and explicitly declined
an additional server/control validation run. Do not dispatch the prepared
`checks/kimi-causal-fp8-control-20261029` branch for that purpose.
Compare the current full causal-FP8 run against the established
**17,406.08912 total tokens/s/GPU** reference, with latency and errors reported.
After its result is reviewed, select the native candidate and proceed with native
GPU numericals, full GSM8K and native performance. Required matched comparisons
against the other FP8 PRs remain a separate later task.

The fresh GPU job reached `Launch job script`; no throughput result is available yet.

## 2026-10-29 current run

The original retry job `123215353998` never published a setup step. The operator
confirmed no Docker container or GPU activity, and the runner disconnected.
Cancellation completed at 02:06:10 UTC. This was a runner startup failure,
not a completed performance measurement.

Fresh full run [40804432321](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/40804432321)
uses workflow source `75a52838ee8fa89461dc812913eb71b370b839bd` and unchanged
serving source `bb8bed732f514fd697856c5abd1d2bdd9d56a1e5`: C48 + MTP,
TP8/DCP8, original prototype image, 3600-second replay. The prerequisite and
configuration jobs passed. At 2026-10-29T02:09:49.605677+00:00, the GPU job is **queued**
and the runner is **offline**. No new-token throughput is established.
A 60-second foreground checker was started for this run; it does not wake a finished model turn.

Earlier [health check 40802919527](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/40802919527)
passed on `pit2-p03-g15` (runner name still ends in `g07`), with eight idle MI355X,
the pinned image present, 4.4 TiB filesystem space available, and an empty
workspace. The established 17,406.08912 run log also identifies `pit2-p03-g15`.
No old performance files were recovered or deleted by the recovery workflow.

Native setup fixes at `75a52838ee8fa89461dc812913eb71b370b839bd` passed
[CPU CI 40804336409](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/40804336409):
**18 tests, zero failures/errors/skips**, plus launcher and Bash checks.
Artifact `12867865400`. Native GPU validation remains pending.

## Completed evidence

- [x] Established throughput: baseline **15,476.46881**, cached FP8 FMHA with BF16 projection **16,915.72154**, and direct FP8 projection + cached FP8 FMHA **17,406.08912** total input + output tokens/s/GPU.
- [x] Cached-only FP8 prototype GSM8K: **1,281/1,319 (97.119%)**, with actual optimized execution on eight workers.
- [x] New-token causal FP8 controlled GPU diagnostics: four cases, masking and changed-input graph replay passed.
- [x] New-token causal FP8 prototype GSM8K [40712931283](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/40712931283): strict **1,279/1,319**, flexible **1,278/1,319**; cached and causal FP8 execution verified on the same eight workers.
- [x] PR description includes the three-stage performance table and the exact baseline job supplied by the user.

These are prototype results. Exact native-PR GPU validation is still pending; the historical captured-BF16 LSE discrepancy is not resolved by GSM8K.

## Remaining work in order

1. [x] **New-token full performance completed:** 15,264.82341 total tokens/s/GPU, -12.30182%; evidence and limits are recorded above.
2. [x] **Native candidate selected:** retain cached-only PR revision `08e5d81a0d4fc76c20a97f9a517195db5a5dc5be`; do not promote causal FP8. Native GPU validation continues after the setup correction.
3. [ ] **Native GPU correctness:** run the final native implementation through projection/FMHA numerical checks, causal masking if included, nonunit scales, exact FP8 execution checks, changed-input graph replay, and `VLLM_GPU_SYNC_CHECK=error` in serving. Resolve the captured-BF16 LSE issue without loosening the gate merely to pass.
4. [ ] **Native full GSM8K:** 1,319 questions, five-shot, temperature 0, real speculative verification. Record the exact native commit and prove the intended cached/causal paths execute on all eight workers during evaluation.
5. [ ] **Native full performance:** run the final native source at C48 with MTP under the established workload and runtime settings. Report throughput, TTFT, TPOT, and measured request errors separately from prototype results.
6. [ ] **Reviewer comparisons:** compare #59124 alone versus #59124 plus direct FP8 up-projection, and #62447 alone versus #62447 plus direct FP8 up-projection. Use a common vLLM base and identical weights/runtime settings within each pair. Validate actual invocation; simple patch stacking is not execution proof.
7. [ ] **PR completion:** finish required static/CI checks, update title/description and concise reviewer replies with actual native accuracy/performance/GPU evidence, then mark ready for review only after the required checks pass. Keep the additional-environment-variable discussion open for the later decision.
8. [ ] **Steady-state profiling:** after the validation tasks, run a separate agentic profile on the final optimized source. Confirm throughput/cache stability and exclude startup/warmup. Capture a short trace across all eight workers; analyze top kernels, memory traffic, communication, CPU launch overhead and GPU idle gaps. Do not mix instrumented throughput with unprofiled benchmark claims.
9. [ ] **Concurrency sweep after profiling:** full runs at **C56 with MTP**, **C64 with MTP**, and **C96 without MTP**. Keep the code, model, weights, TP8/DCP8, CCD binding, offload, graph settings, dataset and duration fixed. For C96, omit speculative configuration and synthetic speculative acceptance and verify actual server settings.
10. [ ] **Select the next optimization:** compare the sweep against C48 with MTP using total and output tokens/s/GPU, TTFT, TPOT, request errors, and cache/GPU utilization. Rank further opportunities from the steady-state profile and propose a concrete next experiment.

11. [x] **NUMA/CCD binding upstream assessment:** [source review completed](./numa-ccd-upstream-assessment.md). Existing native per-GPU CPU-list binding can express the selection; a separate automatic AMD CCD policy needs topology handling and matched measurements. No new binding PR or performance claim.

## Native evaluation execution proof

The current recipe loads the prototype through `PYTHONPATH` and selects `CUSTOM` attention. Native validation must install and import the exact fork revision, select the native backend, and exclude the prototype. Native cached eligibility currently requires query length >=512 and context length >=8192; ordinary GSM8K prompts may otherwise exercise BF16 fallback. Any test-only routing override must be disclosed and kept separate from production-routing performance evidence. Keep the fixed GSM8K task and scoring unchanged, and require successful optimized calls during evaluation on all eight workers.

## Operating constraints

- Keep this active turn working; check run status every **five minutes**. The independent poller collects status; it does not wake a finished model turn. No Codex wake-up trigger or OS scheduler has been enabled.
- Writes are restricted to **InferenceMAX_Rocm_Team** and **ajith-sirra-amd/vllm**. Do not modify Codex state or install a system scheduler.
- No serving smoke runs. Preserve fixed GSM8K task/scoring and test thresholds.
- Use exact code revisions and actual execution proof; do not present plugin evidence as native-PR validation.
- Store evidence and keep English/Chinese documentation synchronized. Append benchmark behavior changes to the performance changelog.

## Native GPU validation harness

The e2e workflow now accepts `native-vllm-sha` with an exact fork commit when `runner-diagnostics-only=true`.
The [runner](../../utils/atom_attention/native_vllm_gpu_validate.py) checks out that revision, uses a repository-local `.venv` and matching-base precompiled libraries, and runs the native `flydsl_fp8` kernel tests.
It records source hashes, package/GPU provenance, and JUnit results; missing or skipped selected tests fail validation.
This is kernel validation only. Model GSM8K, serving synchronization, full performance, and the historical captured-BF16 criterion remain separate checks.
Dispatch follows the current full performance result and final native source selection; no native GPU job was launched during the active performance run.

The native harness resolves an explicit wheel URL for the exact upstream merge-base and requires matching ROCm, Python ABI, Torch, and Triton metadata. Missing or incompatible builds stop validation rather than falling back to another wheel. Resolver checks covered acceptance, Torch mismatch, Python ABI mismatch, and missing-build rejection; these are setup checks, not GPU numerical results.

## Native serving preparation

The validation runner has a separate `--prepare-only` mode that installs the exact native source and prepares a repository-local environment. Its result is `prepared`, never a GPU-test pass. A standalone `native_fp8_validation` observer checks native source hashes and records successful FP8 projection/cached/causal dispatches by rank and process during an explicitly marked phase. It does not replace native numerical kernels. An explicitly selected accuracy-stress mode only changes the cached eligibility predicate to positive query/context lengths; eager evaluation and 512-token scheduler chunks remain caller responsibilities. The observer records this override and rejects it as normal replay evidence.

The recipe now accepts the exact `native-vllm-sha` through the e2e workflow and selects the native ROCM_AITER_MLA / ROCM_AITER_FA backends. It records installation provenance and invocation proof in artifacts. GPU observation and native model evaluation have not run; CPU integration checks are required before dispatch. The current caller requires cached FP8 proof only (`INFERENCEX_NATIVE_REQUIRE_CAUSAL=0`), matching the current PR; set this explicitly to 1 if causal FP8 is promoted after measured uplift. CPU tests reject incomplete, BF16, warmup-only, wrong-revision, restarted-worker, graph-capture-only, and stress-as-replay evidence. The `replay` marker covers the launched replay including warmup and does not establish steady-state kernel residency.

Native setup/evidence CPU CI [40729902162](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/40729902162) passed **15 tests, zero failures/errors/skips** at source `0e40f469f1623c09336a034fa3d130b3b294fa4c`; JUnit artifact `12834722522`. This is not GPU numerical validation. The earlier setup-only failure used a malformed artifact action pin; both new workflows now use the verified existing repository pin.

Native serving integration CI [40732677087](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/40732677087) passed at `bf9b27bd72d9a46814721c6d33811991a142afc8`: 15 native setup/evidence tests, Bash syntax for recipe/launcher, and the existing container-launcher validation. No native GPU run or additional full benchmark was dispatched. PR #64288 remains draft at `08e5d81a0d4fc76c20a97f9a517195db5a5dc5be`.

## Native dependency correction (2026-10-29)

The published wheel filename is the source of the installed dependency version.
For the current native base, Torch metadata says `2.13.0` while the wheel is
`2.13.0+rocm10.0.0`; appending `rocm100` to the metadata version incorrectly
rejects a matching runtime. The resolver now parses the actual wheel version
and still rejects Torch, Triton, ABI, or ROCm mismatches.

Native-only diagnostics and serving select the immutable ROCm 10 image
`nightly-rocm100-edad8b03c5285a1c12e7e4f83c1af709b319a01b`
at digest `sha256:009169c114af5b15e25eddc808afb3c1fc3f12c94e746472b16f39fcf39cd716`.
Its published Torch 2.13.0 and Triton 3.8.0+git669b31ac wheels match the native
base dependencies. Actual container/GPU compatibility still requires validation.
Prototype runs retain their original matrix image (Torch 2.12.0).
Native source remains exact and uses the precompiled wheel for its own merge-base.

## Deferred: native NUMA binding (user decision 2026-10-29)

The user postponed native vLLM NUMA binding. Do not implement or dispatch the
native-NUMA / CCD-disabled sweep variant now. Keep the existing recipe CCD
pinning for current validation, profiling and the planned C56/C64 + MTP and
C96 without-MTP sweeps. Revisit native NUMA only when the user reopens this task.


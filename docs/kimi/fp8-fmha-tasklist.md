# Kimi FP8 task list

**English** | [中文](./fp8-fmha-tasklist_zh.md)

## 2026-10-29 recovery and restart

The runner is online again. Recovery attempt 2 failed on disk quota; attempt 3
reached the inspection step but found no Git checkout. The revised read-only
[health job 40802919527](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/40802919527)
passed: host `pit2-p03-g15`, eight idle MI355X GPUs, the pinned image present,
4.4 TiB reported filesystem free space, and an empty repository workspace.
The runner label/name still ends in `g07`; use the recorded hostname for provenance.
No old performance files were recovered and no files were deleted by recovery.

[Full new-token performance attempt 2](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/40719654583/job/123215353998)
was dispatched at 2026-10-29 01:52 UTC. It reruns the failed GPU job using the
original workflow and source `bb8bed732f514fd697856c5abd1d2bdd9d56a1e5`,
C48 + MTP, TP8/DCP8, and the full configured replay. Dispatch is not startup or
a completed result. Native GPU validation remains the next gated stage.

## Completed evidence

- [x] Established throughput: baseline **15,476.46881**, cached FP8 FMHA with BF16 projection **16,915.72154**, and direct FP8 projection + cached FP8 FMHA **17,406.08912** total input + output tokens/s/GPU.
- [x] Cached-only FP8 prototype GSM8K: **1,281/1,319 (97.119%)**, with actual optimized execution on eight workers.
- [x] New-token causal FP8 controlled GPU diagnostics: four cases, masking and changed-input graph replay passed.
- [x] New-token causal FP8 prototype GSM8K [40712931283](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/40712931283): strict **1,279/1,319**, flexible **1,278/1,319**; cached and causal FP8 execution verified on the same eight workers.
- [x] PR description includes the three-stage performance table and the exact baseline job supplied by the user.

These are prototype results. Exact native-PR GPU validation is still pending; the historical captured-BF16 LSE discrepancy is not resolved by GSM8K.

## Remaining work in order

1. [ ] **Recover and complete the new-token performance measurement:** GPU job 122927005395 in [40719654583](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/40719654583) failed on runner disk quota at 16:50:55 UTC. No benchmark artifact was uploaded; no new-token uplift is established. [Recovery job 40730725939](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/40730725939) failed before executing any steps: the runner lost communication with GitHub. No files were deleted and no results recovered. GPU work needs restored runner storage/connectivity or the requested SSH hostname/user. Recover evidence before repository-scoped cleanup or a full rerun. The intended comparison remains source `bb8bed732f514fd697856c5abd1d2bdd9d56a1e5`, C48 + MTP, against **17,406.08912**.
2. [ ] **Conditional native integration:** if the new-token result shows uplift, add the prepared new-token FP8 implementation to PR #64288. Otherwise retain the established cached-context implementation for native validation. The candidate is isolated in the fork branch `checks/fp8-mla-causal-20261028`; its PR branch has not changed.
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

# Kimi FP8 task list

**English** | [中文](./fp8-fmha-tasklist_zh.md)

## Completed evidence

- [x] Established throughput: baseline **15,476.46881**, cached FP8 FMHA with BF16 projection **16,915.72154**, and direct FP8 projection + cached FP8 FMHA **17,406.08912** total input + output tokens/s/GPU.
- [x] Cached-only FP8 prototype GSM8K: **1,281/1,319 (97.119%)**, with actual optimized execution on eight workers.
- [x] New-token causal FP8 controlled GPU diagnostics: four cases, masking and changed-input graph replay passed.
- [x] New-token causal FP8 prototype GSM8K [40712931283](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/40712931283): strict **1,279/1,319**, flexible **1,278/1,319**; cached and causal FP8 execution verified on the same eight workers.
- [x] PR description includes the three-stage performance table and the exact baseline job supplied by the user.

These are prototype results. Exact native-PR GPU validation is still pending; the historical captured-BF16 LSE discrepancy is not resolved by GSM8K.

## Remaining work in order

1. [ ] **Current full performance run:** follow [40719654583](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/40719654583), source `bb8bed732f514fd697856c5abd1d2bdd9d56a1e5`, C48 with MTP. Compare throughput, latency, measured request errors, and actual cached/causal FP8 calls against **17,406.08912**. Do not duplicate this run.
2. [ ] **Conditional native integration:** if the new-token result shows uplift, add the prepared new-token FP8 implementation to PR #64288. Otherwise retain the established cached-context implementation for native validation. The candidate is isolated in the fork branch `checks/fp8-mla-causal-20261028`; its PR branch has not changed.
3. [ ] **Native GPU correctness:** run the final native implementation through projection/FMHA numerical checks, causal masking if included, nonunit scales, exact FP8 execution checks, changed-input graph replay, and `VLLM_GPU_SYNC_CHECK=error` in serving. Resolve the captured-BF16 LSE issue without loosening the gate merely to pass.
4. [ ] **Native full GSM8K:** 1,319 questions, five-shot, temperature 0, real speculative verification. Record the exact native commit and prove the intended cached/causal paths execute on all eight workers during evaluation.
5. [ ] **Native full performance:** run the final native source at C48 with MTP under the established workload and runtime settings. Report throughput, TTFT, TPOT, and measured request errors separately from prototype results.
6. [ ] **Reviewer comparisons:** compare #59124 alone versus #59124 plus direct FP8 up-projection, and #62447 alone versus #62447 plus direct FP8 up-projection. Use a common vLLM base and identical weights/runtime settings within each pair. Validate actual invocation; simple patch stacking is not execution proof.
7. [ ] **PR completion:** finish required static/CI checks, update title/description and concise reviewer replies with actual native accuracy/performance/GPU evidence, then mark ready for review only after the required checks pass. Keep the additional-environment-variable discussion open for the later decision.
8. [ ] **Steady-state profiling:** after the validation tasks, run a separate agentic profile on the final optimized source. Confirm throughput/cache stability and exclude startup/warmup. Capture a short trace across all eight workers; analyze top kernels, memory traffic, communication, CPU launch overhead and GPU idle gaps. Do not mix instrumented throughput with unprofiled benchmark claims.
9. [ ] **Concurrency sweep after profiling:** full runs at **C56 with MTP**, **C64 with MTP**, and **C96 without MTP**. Keep the code, model, weights, TP8/DCP8, CCD binding, offload, graph settings, dataset and duration fixed. For C96, omit speculative configuration and synthetic speculative acceptance and verify actual server settings.
10. [ ] **Select the next optimization:** compare the sweep against C48 with MTP using total and output tokens/s/GPU, TTFT, TPOT, request errors, and cache/GPU utilization. Rank further opportunities from the steady-state profile and propose a concrete next experiment.

11. [ ] **NUMA/CCD binding upstream assessment:** review the Kimi launch script and existing vLLM CPU-affinity support; identify portable changes, topology/thread-lifecycle gaps, duplicate PRs, and the evidence needed for a separate contribution. This is an assessment, not a submitted binding PR.

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

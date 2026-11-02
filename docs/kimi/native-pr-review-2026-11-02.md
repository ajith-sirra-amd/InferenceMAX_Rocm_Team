# Native FP8 PR review - 2026-11-02

Reviewed PR [#64288](https://github.com/vllm-project/vllm/pull/64288) at
`492080779e9feea19308092b0aba0840e87c6115`.
This report supersedes stale native-GSM8K-pending statements in earlier notes.

## Findings

No new serving-code defect was confirmed in this review. This is not a claim
that all numerical and performance criteria have passed.

- **Design:** The direct FP8 projection is distinct from #59124/#62447.
  It reuses AITER kernels and native compressed DCP gathering/context merging.
  It adds substantial code; matched native performance must justify the cost.
- **Routing:** Default off, target-model opt-in, draft routing preserved.
  gfx950, DCP >1, BF16 activations, E4M3 cache, supported FP8 weight layouts and
  512/128/64/128 MLA dimensions are enforced. Other prefill backends do not
  enter the direct FP8 context scope. Opt-in disables target ASM routing.
- **Weights:** Native per-channel FP8 layouts are adapted by views. Weights are
  neither quantized again nor repacked by the adapter. Cache/weight descales
  reach projection; output descales reach FMHA.
- **Arithmetic:** FP8 cache x FP8 weights, FP32 accumulation, direct FP8 K/V.
  Q uses per-tensor dynamic quantization. FMHA returns BF16 output and
  head-major natural-log LSE for the native merge.
- **State:** Q/scales live within one prefill forward. The context manager
  clears them on normal return and exceptions. Graph replay with changed
  inputs is covered by GPU tests.
- **Numerical limitation:** Cached K/V ranges are estimated from current
  new-token maxima with 2x headroom. Cached outliers can saturate. The
  historical BF16-reference absolute-LSE <0.20 criterion remains unpassed;
  GSM8K and same-quantized-input kernel checks do not close it.
- **Efficiency opportunity:** Preparation writes FP8 new-token K/V although
  causal attention remains BF16. Keep this out of an image-only comparison.
- **Tests:** Eight native GPU cases passed at 332d6c9. Current-head static
  checks and full native GSM8K passed. The subsequent metadata fixes do not
  alter quantization/kernel arithmetic. Normal-routing full replay failed its coverage gate after a serving stall.
- **Docs/review:** Refresh the PR's stale GSM8K/static status. Keep the
  environment-variable discussion open as requested. Four matched reviewer
  comparison arms and upstream contributor-gated CI remain outstanding.

## Established native model evidence

[Full GSM8K 41064573708](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/41064573708):
strict 1278/1319 (96.8916%); flexible 1277/1319 (96.8158%).
Five-shot, temperature zero, C48, TP8/DCP8, real MTP block verification,
eager execution, 512-token accuracy-stress routing. FP8 projection and cached
FMHA verified on all eight workers. VLLM_GPU_SYNC_CHECK=error; no server
runtime/traceback/quota errors. Configured score threshold: 0.90.

[Current-head static checks 41065468482](https://github.com/ajith-sirra-amd/vllm/actions/runs/41065468482):
Ruff, format, applicable pre-commit including mypy, AST/YAML and exact ten-file
source comparison passed. Source hashes match the GSM8K installation.

An optional lm-eval Git metadata lookup reported dubious ownership of
/outputs after evaluation. It did not invalidate the independently checked
native source or the score; the warning has not been fixed.

## Failed current-image reference

- Run: [41388129540](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/41388129540), attempt 1.
- GPU job: 125093482424; runner mi355x-amd_pit2_p03_g07.
- Dispatched: 2026-11-02T04:22:48Z.
- Workflow SHA: 5997c7c005f91914ac0256c7ec3ca05cf28c5d3a.
- Execution SHA: 941b262404f15d5c55874304dfc54e655cf53010.
- Native PR SHA: 492080779e9feea19308092b0aba0840e87c6115.
- Image: vllm/vllm-openai-rocm:nightly-rocm100-edad8b03c5285a1c12e7e4f83c1af709b319a01b@sha256:009169c114af5b15e25eddc808afb3c1fc3f12c94e746472b16f39fcf39cd716.
- Full 3600-second agentic replay, C48, TP8/DCP8, MTP, normal graphs/routing.
- Cached-context FMHA + direct projection FP8; new-token causal attention BF16.
- Synthetic speculative acceptance matches the historical performance recipe.
- This is an optimized current-image reference, not a BF16 baseline.
- Completed with failure at 2026-11-02T06:12:03Z. Serving stopped making
  progress near 06:05Z, followed by shared-memory broadcast timeouts.
- TTFT coverage 95.9%, ITL 96.2%; both below the unchanged 98% gate.
  3,657 valid and 9 error records out of 3,666 profiled requests.
- Reported 18,236.327 tokens/s/GPU uses only 3,463.91877 seconds of the
  configured 3,600-second interval. It is an invalid performance result.
  Using the same token count over 3,600 seconds gives 17,546.988 only as
  an illustration of the denominator effect, not a validated benchmark.
- Exact installed native source hashes match static evidence; all eight
  workers exercised direct FP8 projection and cached FP8 FMHA.
- 104 hipBLASLt errors were logged in ordinary BF16 F.linear. Serving
  continued after those errors; their relation to the later stall is unproven.
  No server traceback or disk-quota error was found.
- Evidence: ../../results/atom-attention/run41388129540/failure-evidence.json.

## BLAS isolation and next diagnostic

[Run 41400490712](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/41400490712)
tested 14 BF16 shapes, three repetitions, on all eight GPUs in fresh processes
without FP8 imports. BOTH old and November 1 images failed: 13/14 shapes
produced hipBLASLt errors on each rank (312 warnings per image). All outputs
were finite; maximum sampled relative RMS was 0.00200604, below 0.02.
No isolated process timed out. The old-image step's continue-on-error setting
made GitHub show success; artifact status is authoritative. Native FP8 tests
were skipped in this diagnostic. A nightly bump alone does not fix the issue.

[Run 41401317715](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/41401317715),
source 2b03ea66409449b4c20ba6ee952572027964e76c, tests the same shapes with
TORCH_BLAS_PREFER_HIPBLASLT=0 and TORCH_BLAS_PREFER_CUBLASLT=0. It asserts
PyTorch selected Cublas (the HIP backend on ROCm) and retains all warning and
numerical gates. Both candidates passed 112/112 cases with no warnings/timeouts; the unchanged
current-head native FP8 tests on the newer image passed 8/8, zero failures/skips. This is a runtime workaround experiment, not a proven
serving-stall fix or a production-code change.

Four reviewer comparison patches passed preparation in
[run 41401194581](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/41401194581):
AST, Ruff, format and whitespace checks. Artifact 13120734226 records exact
file hashes. These checks do not validate runtime routing or numerical
equivalence; direct arms retain new-token-derived K/V scales while reference
arms use actual-context maxima.

## Remaining review gates and next runs

1. Resolve the runtime diagnostic and investigate the serving stall; the current-image reference is invalid.
2. Select and pin a newer compatible ROCm nightly; verify source/wheel/library
   compatibility before consuming GPU time.
3. Validate kernels, serving synchronization, accuracy and matched performance
   on the new image; do not relax numerical thresholds.
4. Prepare #59124 alone / + direct projection and #62447 alone / + direct
   projection on a common base. Preserve each reference's causal routing and Q
   reuse; #59124 overrides the hook currently used to prepare projection scales.
   The four arms require explicit integration and execution proof.
5. Close numerical and review gaps before marking ready. Human review and
   upstream CI remain required; fork checks are separate evidence.
6. After PR validation: steady-state profiling, C56/C64 with MTP and C96 without
   MTP. Native NUMA work remains postponed.

## Full runtime-workaround replay dispatched

[Run 41402358706](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/41402358706):
C48, TP8/DCP8, MTP synthetic acceptance, 3600 seconds, normal graphs/routing.
Same current image and native PR SHA as the failed replay. Execution SHA
2693f50611a445f3f0f2ad91f69b16350f6629c3 sets explicit hipBLAS selection
and records worker stacks every 60 seconds in native_worker_stacks.
Actual BLAS selection is checked when each worker emits FP8 execution proof.
No native serving arithmetic or offload patch is applied. Stack sampling
makes this an instrumented stability diagnostic, not an uninstrumented
performance baseline. Coverage remains 98%.

Prerequisite harness run 41402131374: 20 tests passed, none failed/skipped.
BLAS/current-head native GPU evidence is archived at
../../results/atom-attention/run41401317715/stack-evidence.json.
Follow this replay to completion and inspect the evidence before declaring
the full serving stall fixed or dispatching the matched newer-image replay.
## Startup failure and corrected diagnostic — 08:17Z

Run 41402358706 failed before serving. Checkpoint loading stopped at 80/96
shards, then EngineCore shut down workers at 07:06:47Z. The TCPStore
connection resets are secondary. No native FP8 execution event, hipBLAS
warning, quota error, or benchmark result exists in the archived server log.

The new periodic stack files were not included in artifact paths, and the
existing post-run cleanup deleted the workspace. Recovery 41411453288
confirmed they were unavailable. This was a diagnostic collection mistake,
fixed by workflow commit 715ba4532db9a2b79496fcf9a8173d024ce9b091.
Execution commit 16260f819be89b47defeecbd377fd4e762e5501e also keeps fatal
tracebacks in server.log while periodic samples go to separate files.

Host diagnostic 41411641720 found all eight GPUs idle with reported RAS
counters zero. It could not read kernel logs due runner-account permissions;
this does not rule out a kernel or memory fault. Docker event check
41411820242 records exit code 1 and no retained OOM event; it does not identify
the first worker's exit cause.

Corrected full diagnostic:
[41411886545](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/41411886545),
execution 16260f819be89b47defeecbd377fd4e762e5501e, workflow
d4f6823f4c954a9974c737259f5ab8d3b49090ed, unchanged PR SHA/current image,
C48/MTP/3600 seconds. Harness 41411714298 passed 20/20 tests. The next result
must establish startup, capture any first fatal event, and then pass the
unchanged serving/coverage gates. Neither startup failure nor original
late-run stall has a confirmed root-cause fix yet.
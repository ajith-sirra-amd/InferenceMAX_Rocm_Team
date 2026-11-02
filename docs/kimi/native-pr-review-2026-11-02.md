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
  alter quantization/kernel arithmetic. Normal-routing full replay is active.
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

## Active current-image reference

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
- Last observed: Launch job script active; no result artifacts yet. This alone
  is not proof of GPU activity or progress inside the container.

Accept only after inspecting aggregated throughput, request errors, server
logs, exact installed source/image provenance, and FP8 execution on all eight
workers. Preserve the result before updating the nightly image.

## Remaining review gates and next runs

1. Finish and inspect the current-image reference.
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

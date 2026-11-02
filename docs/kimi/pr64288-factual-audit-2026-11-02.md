# PR #64288 factual audit — 2026-11-02

The PR description and three existing author comments were audited and updated at approximately 17:13 UTC. No source commit, PR state, environment-variable discussion or GPU queue was changed.

## Revision scope

- PR: https://github.com/vllm-project/vllm/pull/64288
- Current head: `2c01e1ce21a262f6108861a831eae5a7d2a2ca8e`.
- Validated source: `492080779e9feea19308092b0aba0840e87c6115`.
- Current head is a merge of the validated source and upstream `3b6eeb060621eba88deb74a146039291a1afbc42`.
- The two FP8 implementation modules, prefill backend, existing gather helper, kernel tests and usage documentation are byte-identical between the validated and merged revisions.
- Upstream changes affect the surrounding ROCm MLA backend, DSpark integration, envs and Buildkite configuration. The merged source is not covered by the earlier end-to-end validation.
- Current-head GPU and equivalent static/docs checks remain outstanding. The PR stays open for human review.

## Confirmed measurements

Each throughput was re-read from its run's benchmark JSON artifact. All rows use Kimi-K3, C48, eight GPUs, TP8/DCP8.

| Run | Stage | Total tokens/s/GPU | Output tokens/s/GPU | Benchmark artifact |
| --- | --- | ---: | ---: | --- |
| 39239112271 | Historical baseline | 15,476.46881 | 103.05251 | 12191288569 |
| 40646139769 | Prototype FP8 FMHA, BF16 projection | 16,915.72154 | 117.84290 | 12802277146 |
| 40695099743 | Prototype direct FP8 projection + FP8 FMHA | 17,406.08912 | 120.16691 | 12825966400 |
| 41413372088 | Validated source, edad8b03 container | 18,244.04135 | 126.79649 | 13132141935 |
| 41447686785 | Validated source, e0d00e4b container | 18,276.23099 | 126.80246 | 13154102668 |

Total throughput includes successful-request input tokens, including cached prompts, plus generated tokens. It is not output-token throughput or fresh-prefill compute throughput. Server logs establish DSpark synthetic acceptance with four draft tokens and synthetic acceptance length 3.36; the benchmark JSON's spec_decoding=none metadata is not the actual execution configuration.

Historical baseline online quantization targeted only *in_proj_qkvgfab. The subsequent FMHA recipe targeted additional attention and MLP projections. The direct projection recipe also targeted kv_b_proj. The 2.89888657% difference between the two prototypes is an observed result, not an isolated native-PR improvement. Native measurements differ by 0.17643920%; one replay per configuration cannot establish container or BLAS uplift.

## Accuracy and diagnostics

- Latest GSM8K: 1276/1319 strict = 96.73995451099317%, conventionally rounded to **96.7400%**; flexible 1275/1319 = **96.6641%**.
- Earlier displayed 96.7399% was truncated rather than rounded. Updated description and comments use 96.7400%.
- Latest evaluation: 41447950940 attempt 2; real DSpark block verification, nonzero accepted tokens, eager execution, accuracy-stress routing, 512-token chunks, GPU sync check enabled, direct projection and cached FMHA observed on eight workers, source hashes matched 492080779.
- First attempt failed in the prerequisite watcher before GPU evaluation.
- Native GPU test artifact 13121751260 confirms eight passed, no failures/skips, at 492080779.
- Fork static artifact 13140797249 confirms passing Ruff/format, applicable pre-commit/mypy and exact-source comparison at 492080779. Rendered docs artifact: 13140323711.
- The historical absolute BF16 LSE discrepancy remains disclosed; passing GSM8K and same-quantized-input tests do not establish BF16 equivalence.
- Six measured request errors in the latest replay and eleven in the earlier replay remain disclosed. Cutoff cancellations are separate. No server runtime errors or BLAS warnings were detected in the previously inspected logs.

## PR corrections

- Scoped elimination of the BF16 intermediate to eligible cached chunks.
- Described smaller-chunk BF16 output fallback precisely.
- Disclosed target-model ASM-prefill routing/metadata bypass and the DSpark asynchronous cache-pointer initialization.
- Separated tested vLLM source/precompiled binary revision from container tags.
- Kept the harness's additional weight quantization and BLAS settings distinct from the PR flag.
- Removed current-head validation claims after the main merge.
- Updated current CI link and left checklist validation/quality unchecked pending merged-head checks.
- Corrected stale combined-reference-test promises in comment 6473674684.
- Updated comments 6473435649 and 6577154974 with the tested revision and post-merge validation gap.
- Preserved the unresolved additional-environment-variable review thread.

The current authorization gate is https://github.com/vllm-project/vllm/actions/runs/41485679652/job/125415263854 . Its log confirms the contributor label/merged-PR policy; no source check ran. DCO passed; pre-commit and Check format were skipped. No authorization label was requested or added.

## Ongoing work

As last checked during this audit:
- 41478604149: C48 steady-state profile replay in Launch job script, validated source 492080779.
- 41485661453: C56 with MTP waiting for the profile run.
- 41485729121: C96 without MTP waiting for C56.
- These runs remain pinned to the previously validated source; they do not validate the new PR merge.
- Dispatch the prepared CPU profile analyzer after the profile succeeds, alongside C56.
- Native AMD NUMA work is separate; preserve CCD binding in the queued runs.

# Inspected review evidence from run 41861899047

Candidate: a03796b8e3bfe536e22f9db741109dbf80170079. Pinned main: 69d27bf8002a4634de1699c95fb0aeb22fb86ca5.

The overall workflow failed in candidate GSM8K. The completed checks below retain their original scope and do not make the workflow an overall pass.

## Matched synthetic serving

TP8/DCP8, live MTP, C8/C48, normal routing, graphs enabled, CCD binding, no CPU KV offload. One warmup request and one measurement per case. Total throughput includes input plus output tokens across all eight GPUs; these values are not per-GPU AgentX throughput. Instrumentation was enabled.

| Case | Main tokens/s | OFF tokens/s | ON tokens/s | ON vs OFF | ON vs main |
| --- | ---: | ---: | ---: | ---: | ---: |
| short_context_c8 | 891.71 | 913.88 | 862.15 | -5.66% | -3.31% |
| short_query_c8 | 6120.17 | 6990.07 | 5252.38 | -24.86% | -14.18% |
| cached_c8 | 3177.48 | 3162.98 | 3675.10 | +16.19% | +15.66% |
| cached_c48 | 4518.52 | 4350.97 | 4404.93 | +1.24% | -2.51% |
| long_cached_c48 | 5212.73 | 5203.82 | 5323.39 | +2.30% | +2.12% |

Each arm completed 512/512 requests across five cases, with zero reported failures and no nonempty error records. Below-cutoff results regressed; their cause is not established. These are single measurements without variability estimates. Full final C48+MTP OFF/ON replay remains pending.

## Long-context retrieval

Both main and candidate passed all 12 cases: 16K/32K contexts, depths 10/50/90%, two repeats. Prompt hashes match between arms. Every candidate case records FP8 projection and FMHA on all eight ranks. This small constructed retrieval suite is not a broad long-context accuracy guarantee.

## GSM8K

Main completed 1319 examples: 96.28506444% strict / 96.20924943% flexible. Candidate stopped at 1042/1319 after a 1800-second HTTP timeout, then lm-eval raised UnboundLocalError while reporting that exception. There is no completed candidate score. The server was still generating before shutdown.

Main/candidate mean sampled generation throughput was 33.80/33.02 tokens/s and sampled MTP acceptance was approximately 71.1% in both. This aggregate behavior does not explain the specific timeout. GSM8K execution records contain prefill events but no cached FP8 projection/FMHA events. NIAH provides the latter coverage.

## Follow-through

Retry 41952495827 stopped before model launch because the runner host has no gh executable. Its unrelated whole-runtime ownership cleanup was cancelled after the model step had been skipped and test-container cleanup completed.

Replacement 41953335199 uses the pinned GitHub artifact action and scopes ownership cleanup to its own evidence directory. Harness revision 1af4b30f1e70d2270c71c365b0eb3438a509a4e7. It retains the 1800-second request and 7200-second evaluation timeouts, full dataset, prompts, C48, live MTP, source and scoring. Observational HTTP tracing records request identity, payload, timings, usage and primary failures. No inference source changes were made for the retry.

The user confirmed a new reservation and later reported clearing the GPUs. No inference about the cause of the earlier timeout is drawn from that report.

## Superseding retry audit

Run 41953335199 ended with GitHub success, but artifact 13381852933 contains
only three setup files. GSM8K did not execute. A single-quoted echo inside the
single-quoted container command split its script into two shell arguments.
The bad command reproduced that split; the corrected command passes one
complete script containing the evaluation invocation. A separate failing
GSM8K evidence-audit status records that the successful workflow is invalid.

Replacement run 41954472581 uses harness 1484593279c55e86827b1b3e31b7f8c55e7ae706.
Hosted preflight now validates the complete shell argument, and a post-run
check requires a passed 1319-example score, exact candidate/base revisions,
nonzero live MTP acceptance, and 1319 matching HTTP starts/completions with
zero request errors. The 30-minute HTTP timeout remains unchanged.
No score or performance result from either November 5 setup retry is claimed.

## Current attempt

Run 41954472581 failed in hosted preflight because its new argument audit
indexed the run field of action-only steps. No GPU job executed. The selector
was corrected and exercised against action-only, preflight and container steps.

Active replacement: 41954676681, workflow revision
5f4d3674c2af1288042daa812e0d116ca844f86a, GPU job 127006701542.
Hosted checks and baseline-artifact recovery passed. The diagnostic step is
running. The candidate remains a03796b8; no accuracy pass is claimed yet.

After inspecting a complete candidate result, the prepared final replay uses
tag pr64288-final-replay-20261104, immutable harness
6b4f3553391063b1060d39cf16b154e7d26585b3, whose CPU tests passed in 41877883215.
Use e2e-tests.yml with the kimik3-fp4-mi355x-vllm-agentic-mtp config, C48,
native-vllm-sha=a03796b8e3bfe536e22f9db741109dbf80170079,
native-comparison=native, duration-override=3600. Run native-fp8-enabled=0,
then 1, without profiling. This selects the pinned e0d00e4b image digest and
binaries for the exact 69d27bf8 base. Preserve CPU offload, CCD placement,
quantization targets, BLAS settings and synthetic speculative acceptance.
Do not label those replay results as live-MTP accuracy evaluations.

## Upstream documentation check diagnosed

Read the Docs build 35640024 at published head e2393d70 did not reach rendering.
The public build-command log shows docs/pre_run_check.sh stopping because
pre-commit/pre-run-check concluded failure. Its log links the existing
contributor-authorization gate, 41822182813/job/126547776319. This is not a
reported MkDocs/source error. Candidate independent rendered docs passed in
41836553476; that does not replace upstream CI authorization.
Evidence: https://app.readthedocs.org/api/v2/build/35640024/.
Do not request labels or bypass the upstream gate as an agent.

## Live runner observation at 03:06 UTC

The retry is executing on physical host `pit2-p03-g40`. GitHub still displays
runner name `mi355x-amd_pit2_p03_g07`; do not use that stale label to identify
the server. Verified SSH access is through local alias `pit-login`, then the
login host's existing verified connection to `pit2-p03-g40`.

Container `pr64288-review-41954676681` is alive. The request trace at
`/work/review-reruns/41954676681/candidate_on/gsm8k-request-events.jsonl`
records 497 completed requests, 48 in flight and no request/response failures.
The oldest in flight is 213 seconds; the slowest completed request is 444.5
seconds. These are progress observations, not an accuracy pass. Keep this
progressing attempt unchanged and inspect the final evidence.

The new reservation end time has been requested and is pending. Continue the
already-running validation under the renewed authorization; establish the
cutoff before scheduling the later multi-hour OFF/ON sequence. Current local
checkout writes are denied; the source and workflow have not been changed.
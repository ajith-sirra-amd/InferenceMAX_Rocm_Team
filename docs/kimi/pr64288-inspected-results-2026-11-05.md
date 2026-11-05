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

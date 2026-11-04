# C56 short control and C48 profile follow-through ? 2026-11-04

[??](./pr64288-short-control-status-2026-11-04_zh.md)

## C56 short control

- Run: https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/41775886780
- GPU job: 126396136906; runner mi355x-amd_pit2_p03_g07.
- Native PR: 8a55b8eacf9020a0b421a3c50a927c4a40e61604.
- Harness: fcd87b661cc34c2b74202d458b2700e9d7fd49d9, checks/pr64288-c56-short-control-20261104.
- CPU prerequisite: 41775841600, successful; artifact 13297460260, 45 tests passed, no failures/errors/skips.
- C56 + MTP, TP8/DCP8, native-comparison=native, native-fp8-enabled=0.
- duration-override=300: 300-second measurement, one extra warmup request per lane after required cache primers.
- Pinned e0d00e4b image retained. Current source merge base 3b6eeb060621eba88deb74a146039291a1afbc42 has matching Torch 2.13.0+rocm10.0.0 and Triton 3.8.0+git669b31ac metadata.
- The native installer still checks the actual binary runtime and source hashes before serving.
- Preserve quantization configuration, CCD binding, BLAS settings and chunk size.
- Completed successfully; see the result and coverage limitation below.
- Flag-off proof checks all eight workers and rejects native FP8 projection/FlyDSL FMHA calls. This control retains other existing vLLM routing, including eligible ASM; it is not an all-BF16 arm.
- If successful, run the same source/config with the flag enabled. Do not compare this updated source directly to historical 492080779 as an isolated flag experiment.
- A five-minute pass cannot clear the historical C56 stall at approximately 50 minutes.

## Completed analysis of existing C48 steady-state profile

- Capture run: 41478604149; capture artifact: 13165722638.
- Native source: 492080779e9feea19308092b0aba0840e87c6115 (older than the current PR).
- CPU analysis: https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/41776530466
- Analysis commit: 93b7624d154f38ddfa2e02bcb65654877e2ae11a.
- Analysis artifact: 13296594944, native-profile-analysis-41478604149.
- Associated CPU checks: 41776530461, successful.
- Confirmed all eight worker traces and native projection/cached-FMHA replay proof.
- Capture started after 300 seconds of measured-phase settling, with 203,023 generated tokens observed during settling. Approximately 20.12 seconds of GPU kernel span per rank.

| Observed kernel/group | Share of summed kernel durations |
| --- | ---: |
| FP8 FMHA main kernel | 12.1792% |
| MLA multi-token decode kernel, qh32/qseqlen4 | 8.6147% |
| Direct FP8 gather/up-projection, all three variants | 4.8846% |
| Fused Q/K/V quantizer | 0.0979% |
| Communication kernels matched by name | 21.7501% |

Communication grouping includes quick_allreduce_int4, ncclDevKernel_Generic, two cross_device_reduce_2stage variants, and ddaAllGatherIpc.
Per-worker GPU-kernel busy union covers 93.58?97.93% of the first-to-last-kernel span.
These are summed kernel durations and within-trace interval unions, not wall-time speedup estimates, GPU compute utilization, or bandwidth measurements. Communication kernels may include waiting. Identify their call sites and critical-path overlap before choosing an optimization.
The analysis establishes that FMHA remains the largest individual kernel in this captured optimized C48 workload. It does not validate the current PR or explain the C56 stall.

## Next priorities

1. C56 short control completed; source/path proof verified, but coverage validation was automatically skipped.
2. Matching FP8-enabled short diagnostic 41783313151 completed; see the final comparison.
3. Address remaining review requests: preserve uncached ASM, separate unrelated synchronization fixes, test cached outliers and long-context accuracy, and run matched bench-serve comparisons.
4. Use the captured profile to prioritize further kernel/communication experiments after correctness and review work.


## Completed short-control result and matched follow-up

Run 41775886780: GPU job completed successfully at 05:38:58 UTC.
Artifacts: aggregate 13298423058; raw/provenance 13299180749; server log 13299365403.

- Total throughput: 11,103.89603 tokens/sec/GPU, including input and output.
- Output throughput: 59.92731 tokens/sec/GPU.
- Reported accounting interval: 326.78755 seconds; configured sending window: 300 seconds.
- Warmup: 115 requests, 1001.62 seconds. Overall GPU job approximately 45 minutes.
- Measured records: 251 valid, one empty-response error; three cutoff cancellations.
- Client drain/credit timeout at cutoff, including a forced phase-completion message.
- No collective timeout, BLAS, OOM, illegal-memory or disk-quota error detected in the full server log.
- Exact current source, matching binary runtime and flag-off proof for all eight workers confirmed.

Correction to the earlier planned coverage claim: no threshold was changed, but the
AIPerf scenario automatically bypasses profiling metric coverage validation for
durations below 900 seconds. The log explicitly reports this skip. Do not label
the control as coverage-validated or canonical performance, and do not replace
the PR's established results with it.

The matching FP8-enabled diagnostic has been dispatched:
https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/41783313151
Same harness fcd87b661 and native PR 8a55b8eac; only native-fp8-enabled changes to 1.
Completed; see the final comparison below.


## Final five-minute diagnostic comparison

Both workflows succeeded. Exact PR 8a55b8eac, harness fcd87b661, binary base
3b6eeb060, pinned e0d00e4b image, C56 + MTP and TP8/DCP8 match.
Source hashes, server command and benchmark command were compared and are identical.
The native FP8 flag is 0 for the control and 1 for the optimized arm.

| Metric | Flag off: 41775886780 | Flag on: 41783313151 |
| --- | ---: | ---: |
| Total tokens/sec/GPU, input plus output | 11,103.89603 | 11,752.32188 |
| Output tokens/sec/GPU | 59.92731 | 63.84768 |
| Valid measured requests | 251 | 263 |
| Measured empty-response errors | 1 | 1 |
| Cutoff cancellations | 3 | 3 |
| Reported accounting interval, seconds | 326.78755 | 319.83543 |
| Warmup time, seconds | 1001.62 | 794.24 |

Observed total-throughput increase: 5.8396%; output-throughput increase: 6.5419%.
These compare the same artifact fields; they are not a long steady-state result
or a statistical-significance claim. All eight ranks in the enabled arm recorded
direct FP8 projection and cached FP8 FMHA. Neither full server log contains the
searched collective-timeout, BLAS, OOM, illegal-memory or disk-quota errors.

Both clients hit the cutoff drain/credit timeout. Both automatically skipped
coverage validation because measurement was shorter than 900 seconds. This is
not an accuracy test and does not clear the historical roughly 50-minute stall.
Keep these diagnostics separate from the PR's established performance and GSM8K data.

Enabled-run artifacts: aggregate 13302542486; raw/provenance 13302462835;
server log 13302627213. GPU job 126419769821 completed successfully at 06:59:17 UTC.
Next review work: preserve uncached ASM routing, separate the synchronization
changes, and complete the remaining performance/accuracy requests. No additional
GPU run was dispatched as part of this pair.

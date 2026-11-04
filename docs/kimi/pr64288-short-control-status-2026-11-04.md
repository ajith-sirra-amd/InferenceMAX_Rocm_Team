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
- Status when recorded: launcher running; no GPU result yet.
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

1. Complete the C56 short control; inspect result coverage, server errors and disabled-path proof.
2. On success, repeat with the same current source and FP8 flag enabled.
3. Address remaining review requests: preserve uncached ASM, separate unrelated synchronization fixes, test cached outliers and long-context accuracy, and run matched bench-serve comparisons.
4. Use the captured profile to prioritize further kernel/communication experiments after correctness and review work.

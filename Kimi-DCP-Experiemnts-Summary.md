# Kimi-K3 on 8x MI355X — Performance

Kimi-K3 (2.8T MoE, 1M context, MXFP4) · vLLM ROCm · agentic replay · TP8.
**Target: 12,500 tok/s/GPU.**

---

# Where we are

**Both targets matter equally. Neither is met.**

| | target | best measured | gap |
|---|---|---|---|
| **throughput** | 12,500 tok/s/GPU | **10,632** (T195, C72, nightly+overlay+PRs) | **−14.9%** |
| **C1 interactivity** | as low as possible | **7.71 ms** ITL p50 (T123) | — |
| SA reference | — | C52 **8,296** · C1 **8.64** ms | we trail C52 by 4.2% |

## T180 C1 — the C1 engine is HEALTHY. Nineteen "C1 failures" were the workload, not the engine.

First run of the `TEST=1` fixed-len probe, and it inverts the C1 story.

| | value |
|---|---|
| successful requests | **10 / 10** |
| Mean TPOT | **7.41 ms** |
| Median TPOT | 7.41 ms · P99 7.50 ms |
| Mean ITL | 29.59 ms |
| Mean TTFT | 1,185.81 ms (median 639.41) |
| Total token throughput | 1,031.73 tok/s |
| isl/osl | 8192 / 1024, ratio 0.8 (uniform 6,830-7,936 in) |
| node | mi355x-amd_b23_07 |
| HSA errors | 0 |
| memory access faults | **0** |

No `Memory access fault`, no `EngineDeadError`, no `ProfileAborted`. Server came
up, captured graphs 1..16, served 10/10 and shut down clean with
`benchmark_exit_code=0`.

**This falsifies the reading I have carried for nineteen trials.** I recorded C1
as a broken engine on a bad node. It is not: at C1 config on b23_07 the engine
serves fixed-length traffic perfectly, at a TPOT better than the 7.57 ms T147
record. What kills C1 is the **agentic workload specifically** -- p50 80k-token
prompts, 735k max, 93.3% prefix-cache reads -- not the engine, not the config,
and not (on this evidence) the node.

That reopens the C1 root cause. The `Memory access fault ... Write access to a
read-only page` is real and still unexplained, but it is now bounded to the
agentic path rather than being a general property of C1 on this node. PR #37682
(zero-init ROCm MLA output buffers for cudagraph padding) stays plausible --
long prompts pad graphs differently than 8k fixed prompts do.

**The job still reported `failure`, and that was my bug.** `run_benchmark_serving`
was pointed at `--result-dir "$RESULT_DIR"` (`/workspace/results`) while
`benchmark-tmpl.yml:289` checks for `$RESULT_FILENAME.json` in the mounted host
workdir. The canonical fixed-len scripts use `--result-dir /workspace/`. Fixed to
`${INFMAX_CONTAINER_WORKSPACE:-/workspace}`. The measurement above is valid --
it was printed in full before the wrapper looked for the file.

## T180 C52 — fixed-len also passes. Both engines are healthy.

| | C1 | C52 |
|---|---|---|
| successful requests | **10 / 10** | **520 / 520** |
| benchmark duration | 80.72 s | 739.00 s |
| Mean TPOT | 7.41 ms | 75.77 ms |
| Mean ITL | 29.59 ms | 75.94 ms (median 37.83) |
| Mean TTFT | 1,185.81 ms | 2,344.64 ms (P99 24,260) |
| Mean E2EL | 8,070 ms | 72,573 ms |
| Total token throughput | 1,031.73 tok/s | 5,840.48 tok/s |
| HSA / memfault | 0 / 0 | **0 / 0** |
| isl/osl/ratio | 8192 / 1024 / 0.8 | 8192 / 1024 / 0.8 |

`graphs: dense ladder 1..80 (mns=80 x 1 rows), DCP=8` -- independent
confirmation that **SPEC_ROWS=1 at C52**, i.e. MTP is off, as established from
the config dumps.

Both jobs reported `failure` purely from the result-dir plumbing bug (writing to
`/workspace/results` instead of the host workdir). Fixed in 88a7ecd3; both runs
predate it. The measurements above printed in full and are valid.

**Do not compare 5,840 tok/s to the agentic 8,342.** Different metric and
different workload -- agentic serves 93.3% of prompt tokens from prefix cache,
this does full prefill every request.

**Calibration gained:** C52 at 8k/1k runs **72.57 s per request** (mean E2EL),
not the 13 s I guessed when sizing the perf mode. 520 prompts x 72.57 / 52 =
725 s, against the 739 s measured. The perf-mode default is corrected to 72.57,
so a 900 s target now sizes to 645 prompts rather than 3,600.

## T184 — SA stack reproduction attempt 1: overlay applied, but NCCL hang

Run 33365015813, job 99403791804, C52 only, agentic replay.

| gate | value |
|---|---|
| `[k3-overlay]` | **applied=1** (`vllm_nightly_46638857_k3_c16_c52_current.patch`) |
| engine | `v0.26.1rc1.dev1219+g46638857f` (correct nightly base) |
| `[chunk]` | 16384 · `[load]` fastsafetensors · `[mns]` 80 offload=dram |
| `graphs:` | dense ladder 1..80 (mns=80 x 1 rows), DCP=8 |
| result | **ProfileAborted — 0 successful / 94 total** (94 warmup, 87 error dropped) |

GitHub reported `success`. It produced no measurement. Green != passed, again.

**Root cause: NCCL collective timeout, NOT HSA.**
```
[rank1] Watchdog caught collective operation timeout: WorkNCCL(SeqNum=36817, OpType=_ALLGATHER)
[rank6] Received a dump signal due to a collective timeout from rank 1
c10::DistBackendError -> Worker proc VllmWorker-3 died unexpectedly
```
Died in warmup at `num_running_reqs=1, num_waiting_reqs=17` -- almost no load.

**Diagnosis: our five DCP env vars are an aigmkt workaround we carried onto a
stack that does not need them.** T167 pinned
`VLLM_USE_DIRECT_DCP_{A2A,Q_GATHER,KV_GATHER}=0` because aigmkt lacks the
compiled op `direct_dcp_a2a_lse_reduce`. **The K3 overlay adds that op.** By
forcing them off we pushed DCP onto a fallback gather path SA never exercises,
and added `VLLM_ALLOW_DCP_FULL_CUDAGRAPH=1` + `VLLM_DCP_Q_REPLICATE=1` on top.
SA's launcher and their 8,953 run set **none of these five**.

Fix: skip all five when `K3_OVERLAY_APPLIED=1`, keep them on aigmkt.

Comm backend was NOT the deviation -- SA's 8,953 run used `a2a`, same as ours.
(Their current launcher defaults to `ag_rs`, but that change came later, after an
unrelated `HSA_STATUS_ERROR_EXCEPTION 0x1016` in aiter moe_sorting on a2a.)

## T187 — nightly stack SERVES. Two of my own bugs blocked the measurement.

Run 33377340922, job 99441726069, C52.

| gate | value |
|---|---|
| `[k3-overlay]` | **applied=1** |
| `[dcp-direct]` | **overlay applied -- left at engine defaults** |
| `[pr-stack]` | **applied=0** |
| FUNC pass (214k/874, 104 prompts) | **5 successful**, 314.45 s, TTFT mean **43,623 ms** |
| PERF pass (8k/1k, 645 prompts) | **0 successful**, 0.48 s |

**The DCP env fix worked.** No `_ALLGATHER` hang, no HSA, no memory fault -- the
nightly+overlay engine came up and served real traffic. T184's failure is
resolved: our five `VLLM_USE_DIRECT_DCP_*` / `ALLOW_DCP_FULL_CUDAGRAPH` /
`Q_REPLICATE` exports were an aigmkt workaround that broke the overlay's DCP path.

**Bug 1 -- the PR stack was vetoed by one hunk.**
`cudagraph_utils.py Hunk #2 FAILED at 362` (Hunk #1 succeeded). #54095 is cut
against a newer tree than 46638857. `patch` is all-or-nothing per invocation, so
that single hunk blocked all five files including #53940, which had no conflict.
Fixed: patches split per-file under `k3_patches/pr_stack/`, applied
independently, gate line reports applied-count and skipped names.

**Bug 2 -- my FUNC pass was an overload test, not a health gate.**
104 concurrent x 214k tokens = ~22M input tokens with no prefix cache. It got
5/104 through at 43.6 s TTFT and left the server wedged, so the perf pass
returned 0 successful in 0.48 s. I flagged this exact risk when building it and
shipped it anyway. Dropped: `TEST_MODE` now defaults to `perf` (8k/1k), which is
directly comparable to T180's C52 baseline of 5,840.48 tok/s / TPOT 75.77 ms.

**Still no throughput number on the nightly stack.** Three attempts (T184 hang,
T185/T186 leaked VRAM, T187 self-inflicted) and the headline is unchanged at
8,342 on aigmkt against SA's 8,953.

## T188 — FIRST GREEN ON THE NIGHTLY STACK. +13.3% throughput, -13.0% TPOT.

Run 33379616710, job 99448806930, C52, fixed-len 8k/1k perf pass, 645 prompts.

| gate | value |
|---|---|
| `[k3-overlay]` | applied=1 |
| `[pr-stack]` | **applied=4 files**, skipped `cudagraph_utils.py` (#54095 hunk 2 stale) |
| `[dcp-direct]` | overlay applied -- engine defaults |
| Available KV cache | **51.29 GiB** (chunk 16384, nightly) |
| HSA / memfault / ProfileAborted | **0 / 0 / 0** |

### vs T180 (aigmkt, chunk 8192) -- same harness, same ISL/OSL, same conc

| metric | T180 aigmkt | **T188 nightly** | delta |
|---|--:|--:|--:|
| Successful requests | 520/520 | **645/645** | both clean |
| Total token throughput | 5,840.48 | **6,616.31** | **+13.3%** |
| Output token throughput | 651.44 | **735.15** | **+12.9%** |
| Mean TPOT (ms) | 75.77 | **65.89** | **-13.0%** |
| P99 TPOT (ms) | 93.13 | **68.77** | **-26.2%** |
| Mean ITL (ms) | 75.94 | **65.89** | -13.2% |
| Median ITL (ms) | 37.83 | 42.36 | +12.0% (worse) |
| P99 ITL (ms) | 501.02 | **1,008.91** | **+101% (worse)** |
| Mean TTFT (ms) | 2,344.64 | 3,329.95 | +42% (worse) |
| Available KV cache | 59.81 GiB | 51.29 GiB | -8.52 GiB |

**Throughput and mean/P99 TPOT all improve substantially. TTFT and the ITL tail
get worse** -- exactly the chunk 8192 -> 16384 trade predicted, and the -8.52 GiB
KV pool is the measured cost of the bigger token budget at fixed gmu 0.9.

**NOT attributable to one change.** T188 moves image (aigmkt -> nightly-46638857),
K3 overlay (none -> applied), chunk (8192 -> 16384), load_format (auto ->
fastsafetensors), CCD pinning (on -> off), DCP env (forced -> defaults), and adds
4 PR-stack files. Any of these could carry the +13.3%.

**Caveat: this is fixed-len, NOT comparable to SA's agentic 8,953** or to our
agentic 8,342. Different harness, different workload (agentic serves 93.3% of
prompt tokens from prefix cache). The agentic run is next and is the only number
that lands against 8,953.

## T189 — NEW BEST: 8,685 tok/s/GPU at C52 on the nightly stack

Run 33382611653, job 99458189050, C52, agentic replay.

| | value |
|---|---|
| **Throughput per GPU** | **8,685 tok/s** |
| Requests | 2,090 successful / 2,200 total (107 warmup, 95 error dropped) |
| Request Error Rate | **0.14%** |
| Output token throughput | 456.37 tok/s |
| ITL mean / p50 / p90 | 80.84 / 74.84 / 105.57 ms |
| Time to Second Token mean | 128.04 ms |
| HSA / memfault / ProfileAborted | **0 / 0 / 0** |
| `[k3-overlay]` / `[pr-stack]` | applied=1 / applied=4 files |

### Where this puts us

| | tok/s/GPU | vs T189 |
|---|--:|--:|
| ours C52, aigmkt (T163) | 8,115 | **+7.0%** |
| ours C60, aigmkt (best, n=2) | 8,342 | **+4.1%** |
| **T189 C52, nightly** | **8,685** | — |
| SA C52, nightly (33324464095) | 8,953 | **-3.0%** |
| target | 12,500 | **-30.5%** |

**First time past 8,342.** Also the first agentic run on the nightly stack to
produce a number at all -- T184 hung in `_ALLGATHER`, T185/T186 died on leaked
VRAM, T187 was killed by my own overload FUNC pass.

**Still 3.0% behind SA at the same concurrency**, with two known config gaps:
cudagraph capture 80 (SA 4096) and five unaudited aigmkt-era env vars
(`VLLM_ROCM_USE_AITER_MLA`, `VLLM_ROCM_USE_AITER_MOE`, `AITER_DISABLE_FMHA_OPUS`,
`GPU_ARCHS`, `VLLM_ROCM_QUICK_REDUCE_QUANTIZATION`) that SA does not set. Node
also differs (b23_07 vs amds_01).

**Not attributable to one change** -- image, overlay, chunk 16384, fastsafetensors,
no CCD pinning, DCP env defaults and 4 PR-stack files all moved together.

## T190 — 9,482 tok/s/GPU at C60. WE ARE NOW AHEAD OF SA.

Run 33390742917, job 99483607390, C60, agentic replay, nightly stack.

| | value |
|---|---|
| **Throughput per GPU** | **9,482 tok/s** |
| Requests | 2,076 successful / 2,202 total (123 warmup, 112 error dropped) |
| Request Error Rate | **0.14%** |
| Output token throughput | 471.49 tok/s |
| ITL mean / p50 / p90 | 95.82 / 84.89 / 124.06 ms |
| Time to Second Token mean | 163.31 ms |
| HSA / memfault / ProfileAborted | **0 / 0 / 0** |
| `[k3-overlay]` / `[pr-stack]` | applied=1 / applied=4 files |

### Standings

| | tok/s/GPU | vs T190 |
|---|--:|--:|
| ours C60, aigmkt (old best) | 8,342 | **+13.7%** |
| ours C52, nightly (T189) | 8,685 | **+9.2%** |
| SA C52, nightly (33324464095) | 8,953 | **+5.9%** |
| **T190 C60, nightly** | **9,482** | — |
| target | 12,500 | **-24.1%** |

**First time past SA.** The C60-over-C52 gain is +9.2% on this stack, much larger
than the +2.8% the same step gave on aigmkt (8,342 vs 8,115) -- the concurrency
curve is steeper here, so the old peak location may not hold either.

Latency cost of C60 over C52: ITL mean 80.84 -> 95.82 ms (+18.5%), TTST
128.04 -> 163.31 ms (+27.5%). Error rate identical at 0.14%.

**n=1.** Not replicated, and the old stack's C60 spread (0.20%) was measured on a
different image. Attribution still unresolved -- image, overlay, chunk,
fastsafetensors, pinning, DCP env and 4 PR files all moved together back at T188.

## T192 — GSM8K PASSES on the nightly stack: exact_match 0.995

Run 33404440358, job 99528474415, EVAL_ONLY=true EVAL_LIMIT=200, T190 config.

```
|gsm8k| 3|flexible-extract| 5|exact_match|^ |0.995|+- |0.005|
|     |  |strict-match    | 5|exact_match|^ |0.995|+- |0.005|
```

**T190's 9,482 tok/s/GPU is now accuracy-validated.** The job reports `failure`
only because the eval-only path still trips the benchmark-result-json check in
`benchmark-tmpl.yml` -- a harness bug, not an accuracy failure.

This closes the gap flagged earlier: T188/T189/T190 all ran `RUN_EVAL=false`, so
a large numerics change (264 KB kernel overlay + #53940 a4w4 MoE kernels + chunk
16384) had gone to throughput with no accuracy gate. It holds.

**Image saved:** `kimi-k3-vllm-v2:latest` (local only, no registry push) =
nightly-46638857 + K3 overlay + #53940. Dockerfile at
`k3_patches/Dockerfile.kimi-k3-vllm-v2`.

## T193 — 9,775 tok/s/GPU at C64. The aigmkt cliff at 64 does NOT exist here.

Run 33407055937, job 99537198147, C64 agentic, nightly stack.

| | value |
|---|---|
| **Throughput per GPU** | **9,775 tok/s** |
| Requests | 2,160 successful / 2,293 total (131 warmup, 116 error dropped) |
| Request Error Rate | **0.09%** (best yet) |
| Output token throughput | 495.36 tok/s |
| ITL mean / p50 / p90 | 98.43 / 87.51 / 128.61 ms |
| `[pr-stack]` | **applied=5 files, skipped: none** |
| HSA / memfault / ProfileAborted | 0 / 0 / 0 |

### The concurrency curve inverted between stacks

| conc | aigmkt | nightly stack |
|---|--:|--:|
| 52 | 8,115 | 8,685 |
| 60 | 8,342 | 9,482 |
| 64 | **7,976 (-4.4% CLIFF)** | **9,775 (+3.1%, still climbing)** |

The C64 cliff was a property of the OLD image, not of the model or the node.
On the nightly stack throughput is still rising at 64 with the *lowest* error
rate we have recorded. C72 is now worth testing -- on aigmkt it died outright.

### TWO variables moved, not one

T193 differs from T190 by concurrency (60 -> 64) AND by `#50813` (SiTUv2 A8W4
routed MoE), which landed in this run -- `[pr-stack] applied=5, skipped: none`
where T190 had 4. So the +3.1% cannot be attributed to concurrency alone.

**#50813 changes MoE quantisation math and has NOT been GSM8K-validated.**
T192's 0.995 covers the 4-file stack, not this one. Eval next, before the number
is quoted anywhere.

## T194 — GSM8K on the 5-file stack: 0.99. #50813 validated, 9,775 stands.

Run 33416822894, job 99569435360, EVAL_ONLY=true EVAL_LIMIT=200, C64 config,
`[pr-stack] applied=5 skipped:none`.

```
|gsm8k| 3|flexible-extract| 5|exact_match|^ |0.99|+- |0.0071|
|     |  |strict-match    | 5|exact_match|^ |0.99|+- |0.0071|
```

| stack | GSM8K | run |
|---|--:|---|
| 4 files (overlay + #53940) | 0.995 +- 0.005 | T192 |
| **5 files (+ #50813)** | **0.99 +- 0.0071** | T194 |

**Not a regression.** At limit 200 one question is worth 0.005, so 0.995 -> 0.99
is exactly one more wrong answer, and the intervals overlap heavily
([0.983, 0.997] vs [0.990, 1.000]). #50813's MoE quant change is accuracy-safe
at this resolution. A larger limit would be needed to separate them, and there
is no reason to think there is anything to separate.

**T193's 9,775 tok/s/GPU is now accuracy-validated.**

## T195 — 10,632 tok/s/GPU at C72. Still climbing. aigmkt DIED here.

Run 33418755100, job 99575730290, C72 agentic, nightly stack.

| | value |
|---|---|
| **Throughput per GPU** | **10,632 tok/s** |
| Requests | 2,239 successful / 2,392 total (148 warmup, 138 error dropped) |
| Request Error Rate | 0.22% |
| Output token throughput | 499.32 tok/s |
| ITL mean / p50 / p90 | 108.42 / 101.49 / 142.76 ms |
| `[pr-stack]` | applied=5, skipped: none |
| HSA / memfault / ProfileAborted | 0 / 0 / 0 |

### The concurrency curve, both stacks

| conc | aigmkt | nightly stack | delta |
|---|--:|--:|--:|
| 52 | 8,115 | 8,685 | +7.0% |
| 60 | 8,342 | 9,482 | +13.7% |
| 64 | 7,976 (cliff) | 9,775 | +22.6% |
| **72** | **DIED** | **10,632** | -- |

Every aigmkt-era conclusion about the top of the curve was an artifact of that
image. There is still no peak: 52 -> 60 -> 64 -> 72 is +9.2%, +3.1%, +8.8%.

**Gap to 12,500 is now -14.9%**, from -33% this morning.

Costs at C72: ITL mean 108.42 ms (from 98.43 at C64, +10%), error rate 0.22%
(from 0.09%) -- still far under the 10% abort threshold. Output token throughput
per-GPU is flattening (471 -> 495 -> 499 across C60/C64/C72) while total
throughput rises, i.e. the gain is coming from more concurrent streams, not
faster ones.

## T196 — C80 = 9,864, DOWN 7.2% from C72. The peak is C72.

Run 33428881369, job 99609156264. 2,253 successful / 2,420, err 0.13%, zero
faults, `[pr-stack] applied=5`.

### Full concurrency curve, nightly stack

| conc | tok/s/GPU | step |
|---|--:|--:|
| 52 | 8,685 | -- |
| 60 | 9,482 | +9.2% |
| 64 | 9,775 | +3.1% |
| **72** | **10,632** | **+8.8%** |
| 80 | 9,864 | **-7.2%** |

**Peak = C72 at 10,632.** The curve is unimodal on this stack, same shape the
aigmkt curve had but shifted right by ~12 concurrency points and ~28% higher.

**Prime suspect for the C80 drop: mns.** `MAX_NUM_SEQS` is pinned at 80 for
DCP>1, so at conc 80 mns == conc with zero headroom, while the agentic harness
spawns lanes *past* nominal concurrency (`1 additional request on each of 80
lanes` in the warmup line). At C72 there were 8 slots of slack; at C80 there are
none, so requests queue instead of batching.

That is a testable claim, and it is the next run: C80 with mns 96. If throughput
recovers, mns was the limiter and the peak moves right again. If it does not,
C72 is the real operating point and the remaining gap to 12,500 (-14.9%) needs
kernel work rather than tuning.

Note N5 ("mns 96 KILLS THE ENGINE", T165) is an aigmkt-era finding. Every
aigmkt conclusion about the top of this curve -- the C64 cliff, the C72 death --
has already been falsified on this stack. It should not block the experiment.

## T197 — C80 + mns 96 = 10,159 (+3.0%). mns WAS a limiter. N5 falsified.

Run 33439502712, job 99644048477. `[mns] max_num_seqs=96 conc=80 offload=dram`,
`graphs: dense ladder 1..96 (mns=96 x 1 rows)`, `[pr-stack] applied=5`.
2,354 successful / 2,525, err 0.30%, zero HSA, zero memfault.

| C80 config | tok/s/GPU |
|---|--:|
| mns 80 (T196) | 9,864 |
| **mns 96 (T197)** | **10,159 (+3.0%)** |
| C72 mns 80 (T195) | **10,632** -- still the peak |

**Two findings.**

1. **mns headroom is real but partial.** Giving C80 16 slots of slack recovered
   +3.0%, which supports the "conc == mns starves the agentic lanes" reading --
   but it recovered only ~40% of the C72->C80 drop. Something else also degrades
   past 72.

2. **N5 is falsified.** "mns 96 KILLS THE ENGINE" (T165, aigmkt) is wrong on this
   stack: mns 96 ran clean at err 0.30%. That is the fourth aigmkt-era
   conclusion about the top of the curve to fall, after the C64 cliff, the C72
   death, and the mutable-tag story.

**Next: C72 + mns 96.** If mns headroom is a general win rather than a C80
rescue, the peak should rise above 10,632. One variable against T195.

## T198 — C72 + mns 96 = 10,630. Identical to mns 80. Peak CONFIRMED n=2.

Run 33448907210, job 99674166270. `[mns] max_num_seqs=96 conc=72`,
`graphs: dense ladder 1..96`, `[pr-stack] applied=5`. 2,240 successful / 2,394,
err 0.27%, zero faults.

| C72 | tok/s/GPU |
|---|--:|
| mns 80 (T195) | 10,632 |
| mns 96 (T198) | 10,630 |
| **spread** | **0.02%** |

**1. mns is neutral at C72.** The +3.0% mns 96 bought at C80 was specifically
because conc == mns there; once concurrency is below mns, extra headroom does
nothing. That is a clean confirmation of the C80 diagnosis rather than a general
tuning win, and it closes mns as a lever.

**2. C72 is replicated at n=2 with a 0.02% spread.** 10,632 / 10,630 across two
independent runs an hour apart. That is by far the tightest replication in this
ledger -- the old aigmkt C60 pair was 0.20% and C64 was 1.6%. The peak is real
and the number is solid.

### Final concurrency curve, nightly stack (mns 80 unless noted)

| conc | 52 | 60 | 64 | **72** | 80 | 80 (mns 96) |
|---|--:|--:|--:|--:|--:|--:|
| tok/s/GPU | 8,685 | 9,482 | 9,775 | **10,632 / 10,630** | 9,864 | 10,159 |

**Concurrency tuning is now exhausted.** Remaining gap to 12,500 is -14.9%.

## C52 — every lever tried is flat or negative

| run | change vs T103 | tok/s/GPU |
|---|---|--:|
| **T103** | baseline: DCP=8, mns 80, dram, aigmkt | **7,950.6** |
| T156 | nightly + rebased #51705 | 7,906 (−0.6%) |
| T158 | `NCCL_MIN_NCHANNELS=32` | 7,656 (−3.2%) |
| T157 | `gmu 0.95` | **0 — engine hung** |
| **T160** | nightly + #51705 + **CCD pinning**, offload **dram** | **7,968** — best to date |
| T161 | pin after ready, offload **none** | 7,824 (−1.8%) |
| T162 | **async scheduling** (offload none) | 7,686 (−1.8% vs T161) |
| T164 | **chunk 4096** (offload dram) | 7,528 (−7.4% vs T163) |
| T165 | **mns 96** | **engine died** — 5,090 partial, aborted |
| T166 | **gmu 0.92** | **0/103 — hung in warmup** |
| T167 | **direct DCP a2a** | **workers died at capture — op absent** |
| **T168** | **repeat of T163, nothing changed** | **8,103** — noise floor 0.30% |
| T169 | **C48** (best config, conc swept) | 7,771 — **−4.2% vs C52** |
| **T169** | **C56** (best config, conc swept) | **8,326 — NEW BEST, beats SA** |
| T170 | **C72** (best config, conc swept) | **engine died** — aborted, 43/284 = 15.1% |
| T170 | **C64** (best config, conc swept) | 8,040 — **−3.4% vs C56** |
| T170 | C1 (unchanged) | **aborted, tenth straight** — 15/146 = 10.274% |
| T171 | **C60** (best config, conc swept) | **aborted early — INCONCLUSIVE**, not a cliff |
| T171 | **C56 repeat** (peak, 2nd sample) | **aborted — 10/59 = 16.9%; warmup 3,194 s** |
| T171 | C1 (unchanged) | **aborted, eleventh straight** — 15/146 = 10.274% |
| T172 | C1 (unchanged) | **aborted, twelfth straight** — 15/145 = 10.345% |
| T172 | **C56 repeat #2** | **0 successful / 56 — HSA out-of-resources killed the engine** |
| T173 | **C56 repeat #3** | **aborted — 29/191 = 15.2%; 3 HSA errors again** |
| T173 | C1 (unchanged) | **aborted, thirteenth** — 15/146; **HSA = 0** |
| T174 | C1 (unchanged) | **aborted, fourteenth** — 15/146, identical |
| **T174** | **C64 repeat** (node-health probe) | **7,912 — clean, HSA = 0. Node recovered.** |
| T175 | **C56 repeat #4** | **0 successful / 70 — warmup failed, HSA = 3 again** |
| T175 | C1 (unchanged) | **aborted, fifteenth** — 15/146 |
| T176 | C1 (unchanged) | **aborted, sixteenth** — 15/145 |
| **T176** | **C60** (hypothesis test) | **8,350 — NEW BEST, clean, HSA = 0** |
| **T177** | **C60 repeat** | **8,333 — REPRODUCED, clean, HSA = 0** |
| T177 | C1 (unchanged) | **aborted, seventeenth** — 15/146 |
| T178 | C1 (unchanged) | **aborted, eighteenth** — 15/146 |
| T179 | C1 (unchanged) | **aborted, nineteenth** — 15/146 |
| **T178** | **C62** (cliff-edge test) | **7,966 — clean, HSA = 0. Cliff is at 60→62.** |

## Phase E: the concurrency curve

Config space at C52 is exhausted, so the remaining question is whether 52 is the
right operating point. Best config held fixed; only concurrency varies.

| conc | tok/s/GPU | vs C52 mean (8,115) |
|--:|--:|--:|
| **48** | 7,771 | **−4.2%** |
| **52** | 8,127 / 8,103 (mean 8,115) | baseline |
| **56** | **8,326** | **+2.6%** |
| **64** | 8,040 | **−0.9%** |
| **72** | **aborted — engine died** | — |

Every delta clears the 0.30% noise floor (14×, 8.7×, 3.1×), so the shape is
real: **the curve peaks at 56 and turns over.** 64 gives back −3.4% against 56
and lands *below* the C52 mean; 72 is past the cliff entirely.

### T171 C60: aborted early, and it is NOT the C72 cliff

C60 tripped `ProfileAborted` at **18/176 = 10.2%**, harness validation
**43/206 = 20.874%**. The 4,628 tok/s/GPU printed is a partial, not a
measurement. Gate lines all correct (dcp 8/a2a/1, mns 80, chunk 8192, gmu 0.9,
ladder 1..80, 1,562 threads pinned, init 447 s).

**The failure signature is different from C72's, and the difference matters:**

| | C60 (T171) | C72 (T170) |
|---|--:|--:|
| HTTP 500 `EngineCore` | **0** | 16 |
| `InvalidInferenceResultError` | 54 | 30 |
| Effective concurrency, max | **48** | 85 |
| Requests before abort | ~206 | ~284 |

C72 died: the engine threw 500s and the empty streams followed from that. C60
shows **no engine death at all** — only empty-content responses — and effective
concurrency never even reached 48, well under both the offered 60 and mns 80.
There was no queueing pressure to speak of.

The same error class appears at C64 at **3/1808 = 0.166%**. So C60 is that
background rate spiking 18× and crossing aiperf's threshold inside the first
~200 requests, not a new physical limit.

**A cliff at 60 that disappears again at 64 is not physical.** C56 (8,326) and
C64 (8,040) both completed cleanly on either side. C60 is recorded as
**inconclusive and pending one re-run** — it is not evidence about the shape of
the curve, and the peak-at-56 conclusion neither gains nor loses from it.

### T171 C56 also aborted — the whole run is compromised, not the operating point

The C56 repeat, the same concurrency that measured **8,326 cleanly in T169**,
aborted at **10/59 = 16.949%** (aiperf 10/56 = 17.9%), 49 successful of 174.
The 1,489 tok/s/GPU printed is a partial. Same error class as C60: **10
`InvalidInferenceResultError`, zero HTTP 500s.**

The smoking gun is the warmup:

| | init engine (profile + KV + warmup) | eff. concurrency max | outcome |
|---|--:|--:|---|
| T170 C72 | 531 s | 85 | engine died |
| T170 C64 | 447 s* | — | **clean, 8,040** |
| T171 C60 | 447 s | 48 | aborted |
| **T171 C56** | **3,194 s** | **28** | **aborted** |

\*C64's own init; weight-load time is not the discriminator — it swung 169 s to
717 s across these runs and the 717 s one (C64) is the clean measurement.

**3,194 s of warmup is 6–7× every other run on the identical config.** Combined
with effective concurrency never passing 28 against mns 80, the node was
degraded while C56 ran.

**This retracts the framing I gave C60 one cycle earlier.** I called C60 an
isolated inconclusive point. Two consecutive jobs in the same run now show the
same signature, one of them at a concurrency that had already measured clean.
That is not a per-concurrency effect — **T171 as a whole is compromised**, and
neither of its throughput numbers is usable. The Phase E curve stands on T169
and T170 alone, which is where it stood before.

**Consequence: C56 = 8,326 is still n=1.** The one attempt to replicate it
landed on a bad node. It needs a clean re-run before the "settled peak" label
is fully earned.

### T172 C56: the sentinel trace is a SYMPTOM. The root cause is HSA out-of-resources.

The second C56 attempt failed harder — **0 successful of 56 profiling requests,
100%**. Warmup completed *cleanly* (115/115, `errors=0`), then every profiling
request got `ClientConnectorError` on `localhost:8888`: the server was already
gone. `init engine` was **564.61 s**, i.e. normal, so this is not the T171 C56
slow-warmup failure either.

I pulled `server.log` from the artifact rather than stopping at the runner blob,
and it changes the picture. The first error in the run is **not** the sentinel:

```
:0:rocdevice.cpp :3582: Callback: Queue 0x74ea64600000 Aborting with error :
    HSA_STATUS_ERROR_OUT_OF_RESOURCES: The runtime failed to allocate the
    necessary resources.        (×3, three different queues)
```

and *then*, downstream of it:

```
engine_core_sentinel.py:179 in run_with_fault_tolerance
    status, result = mq.dequeue(timeout=dequeue_timeout)
shm_broadcast.py:797 in acquire_read
    raise RuntimeError("cancelled")
→ vllm.v1.engine.exceptions.EngineDeadError
```

**This tempers the N8 story, which I should state plainly.** I have been
reading `engine_core_sentinel` → `mq.dequeue` → `EngineDeadError` as *the* fault
and attributing it to an RPC `dequeue_timeout` that needs raising. In this run
that trace is clearly **downstream**: three ROCm queues aborted with
`HSA_STATUS_ERROR_OUT_OF_RESOURCES` first, and the sentinel then observed a
worker that could no longer answer. Raising a timeout would not have saved it.

What this does and does not establish, kept separate on purpose:

- **Does:** at least one instance of the sentinel trace is caused by GPU-runtime
  resource exhaustion, not by a timeout being too short. The sentinel trace
  alone is therefore **not** sufficient evidence for N8.
- **Does not:** prove the twelve C1 aborts share this cause. I have not seen an
  HSA line in a C1 log. C1 fails at a fixed 15 failures with a drifting
  denominator, which still looks like a distinct, deterministic fault.

**Action: every future sentinel/`EngineDeadError` diagnosis must check
`server.log` for an HSA line before N8 is invoked.** Cheap, and it is the step
that would have caught this earlier.

### The HSA count across every run since the last clean number — one cause, not four

The rule above paid off on its first use. T173 C56 aborted at 29/191 = 15.2%,
and `server.log` again showed **three** `HSA_STATUS_ERROR_OUT_OF_RESOURCES`. So
I went back and grepped the archived `server_logs_*` artifacts for every run
since the last clean measurement:

| run | job | HSA errors | outcome |
|---|---|--:|---|
| T170 | C64 | **0** | **clean — 8,040 tok/s/GPU** |
| T171 | C60 | 1 | aborted, 20.9% |
| T171 | C56 | 2 | aborted, 16.9% (3,194 s warmup) |
| T172 | C56 | 3 | **0/56**, server dead before profiling |
| T173 | C56 | 3 | aborted, 15.2% |

**This overturns my "three different failure modes" framing from two cycles
ago.** I described T171 C60 (empty streams), T171 C56 (slow warmup) and T172
C56 (HSA) as three separate faults and concluded "node health is degraded" in a
vague way. They are **one fault at increasing severity**: ROCm queue resources
failing to allocate, with the count rising 0 → 1 → 2 → 3 → 3 in chronological
order and sitting at exactly zero on the last run that produced a number.

The symptom the benchmark reports — empty content streams, `EngineDeadError`,
connection-refused — varies with *when* in the run the queue abort lands. The
cause does not vary.

**What this means practically:** this is not a config property of C56, and no
amount of re-dispatching C56 will produce a number. The node is accumulating
unreleased GPU queue resources across runs. **That needs a runner reset, which
is outside my bounds** — I can dispatch jobs, not recycle the machine.

**C56 = 8,326 therefore stays n=1, and I am stopping the replication attempts
at three.** Four runs have now been spent on this; a fifth on the same node
would burn another hour to reproduce the same HSA abort. The Phase E curve and
the peak-at-56 conclusion rest, as they have throughout, on T169 and T170.

### T174: the node recovered, and C64 n=2 forces a noise-floor correction

The C64 probe came back **clean: 7,912 tok/s/GPU**, 1,783 successful of 1,919,
validated error rate **5/1788 = 0.280%**, and — the point of the run —
**`HSA_STATUS` count = 0**. The queue-exhaustion fault that killed four
consecutive throughput attempts is gone. `init engine` 909.77 s, wall 113 min,
both in line with T170 C64's 115 min.

Choosing C64 over a fourth C56 was the right call: it answered the node
question *and* produced a real second sample.

**And that second sample corrects something I have been leaning on.**

| C64 | tok/s/GPU |
|---|--:|
| T170 | 8,040 |
| T174 | **7,912** |
| mean | 7,976, **spread 1.6%** |

The noise floor I established from T163/T168 (8,127 vs 8,103) was **0.30%**, and
I have been scoring every delta against it. That figure was measured **at C52**.
At C64 the same-config spread is **1.6% — 5.3× larger.** Noise is not a single
number for this benchmark; it grows with concurrency, which makes sense given
how much more the scheduler is juggling.

**What this costs me:** I called C64's −3.4% against C56 a real turnover backed
by "3.1× the noise floor". Against the *correct* C64 spread of 1.6% it is about
**2×**, which is suggestive but not decisive. The peak-at-56 claim is weaker
than I stated. What survives is the shape — 7,771 at 48 and an abort at 72
bracket the peak firmly — but **56-vs-64 specifically is now within shouting
distance of run-to-run variation** and should not be quoted as settled.

Every per-concurrency delta scored against 0.30% needs re-reading with this in
mind; the C52 ledger is unaffected since 0.30% was measured there.

### T175 falsifies my "node degradation" model. The HSA fault tracks C56, not time.

C56 failed a fourth time: **0 successful of 70**, aiperf reason *"A root AgentX
warmup request failed, so profiling was not"* started. `init engine` 1233 s.
Per the standing rule I checked `server.log` first: **HSA = 3.**

One cycle ago I said the node had recovered, on the strength of T174 C64 coming
back clean with HSA = 0. That was too fast. Laying every run out by *config*
instead of by *time* gives a different and much cleaner picture:

| conc | runs | HSA counts | outcome |
|---|--:|---|---|
| **C1** | 2 | 0, 0 | no HSA ever (aborts, but for the N8 reason) |
| **C64** | 2 | **0, 0** | **both clean — 8,040 and 7,912** |
| C60 | 1 | 1 | aborted |
| **C56** | 4 | **2, 3, 3, 3** | **all four failed** |

**My accumulation model predicted T174 C64 would show HSA ≥ 3, since it ran
after three HSA-3 runs. It showed 0.** The model is falsified: the counts do not
climb with time and do not reset with a "recovery". They sort by concurrency.

What actually correlates: **C56 triggers `HSA_STATUS_ERROR_OUT_OF_RESOURCES`
and C64 does not** — five C56 attempts, one success (T169) and four failures,
against two clean C64 runs bracketing them in time. A *lower* concurrency
failing where a higher one succeeds is counter-intuitive, which is presumably
why I reached for the time-based story instead. But the C64-at-zero data point
sits in the middle of the bad C56 runs and rules it out.

I am not going to invent a mechanism for it. What I can say is what the data
constrains: it is reproducible, config-linked, and not a monotone node decay.

**Practical consequence, stated plainly: C56 = 8,326 is a one-off that has
resisted four replication attempts, and it should not be reported as the best
config.** The defensible best is **C64, n=2, mean 7,976** (8,040 / 7,912) —
lower than 8,326 but actually reproducible, and it sits below SA's 8,296 rather
than above it. The earlier "parity with SA reached" line rested on 8,326 and no
longer holds.

### T176 C60 = 8,350: the discriminating test paid off, and it is the new best

C60 came back **clean: 8,350 tok/s/GPU**, 1,840 successful of 1,967, validated
error rate **4/1844 = 0.217%**, and **HSA = 0**. `init engine` 1575 s, wall
122 min.

**The hypothesis test resolved in the "boundary between 56 and 60" direction.**
Updated by config:

| conc | attempts | HSA counts | best clean result |
|---|--:|---|---|
| C1 | 3 | 0, 0, 0 | — (aborts, N8) |
| **C56** | 5 | 2, 3, 3, 3 (+1 old clean) | 8,326, **1 clean in 5** |
| **C60** | 2 | **1, 0** | **8,350 — 1 clean in 2** |
| **C64** | 2 | **0, 0** | 8,040 / 7,912 — **2 clean in 2** |

So "low concurrency triggers HSA" is **too strong**. C60 can run clean. What the
data supports now is narrower: **C56 is the reliably-bad point (1 in 5), C60 is
intermittent (1 in 2), C64 is reliable (2 in 2)** — a gradient in failure rate
across 56 → 60 → 64 rather than a sharp boundary. That is a weaker claim than
last cycle's, and it is the one the evidence carries.

**8,350 is the best measured number on this stack** and, unlike 8,326, it comes
from a run with a validated error rate and a clean HSA log. It clears SA's 8,296
by **0.65%**.

Honest limits on it, since I over-claimed on 8,326 before:
- **n=1 clean** (1 of 2 attempts). Not yet reproduced.
- The C64 spread is 1.6%, so 8,350-vs-8,326 (0.3%) and 8,350-vs-SA (0.65%) are
  both **inside plausible run-to-run variation**. Treat 8,350 as "best observed
  and cleanly measured", **not** as a demonstrated margin over either.
- The concurrency curve is now 7,771 / 8,115 / 8,326 / **8,350** / 7,976 at
  48 / 52 / 56 / 60 / 64. Peak sits somewhere in 56–60; the exact location is
  not resolved at this noise level.

### T177 C60 = 8,333: the best number now has n=2 and it is tight

The repeat came back **clean: 8,333 tok/s/GPU**, 1,836 successful of 1,961,
validated error rate **2/1838 = 0.109%**, **HSA = 0**, init 518.61 s, wall
116 min.

| C60 | tok/s/GPU |
|---|--:|
| T176 | 8,350 |
| T177 | **8,333** |
| mean | **8,342, spread 0.20%** |

**This is the first best-config claim in the whole ledger that rests on two
clean runs**, and the spread is 0.20% — tighter than the 0.30% C52 floor and 8×
tighter than C64's 1.6%. The two C60 runs also differ enormously in `init engine`
(1575 s vs 519 s) and still land 0.20% apart, which is reassuring: the
throughput number is not tracking warmup weather.

**Settled position, and the caveats that survive:**

- **Best config = C60, 8,342 mean (n=2 clean).** Replaces the C56 8,326 one-off,
  which stays withdrawn.
- **It clears SA's 8,296 by 0.55%.** With C60's own spread at 0.20% that is
  ~2.8× — the first time a lead over SA has been outside its own measurement
  noise. Still a *small* lead, and SA's number is n=1 from my side, so I would
  call it "ahead, modestly" and not more.
- **C60 reliability is 2 clean in 3** (HSA 1, 0, 0). Better than it looked one
  cycle ago, but not free: roughly a third of C60 attempts still die on HSA.
- Curve with the best data now: 7,771 / 8,115 / 8,326* / **8,342** / 7,976 at
  48 / 52 / 56 / 60 / 64 (*one-off). Peak at 60.
- Gap to the 12,500 target: **−33%**. The honest position stated at the top of
  the queue is unchanged — no stack of remaining config levers closes it.

### T178 C62 = 7,966: the cliff is immediately past 60, and it is a step not a slope

C62 came back **clean: 7,966 tok/s/GPU**, 1,777 successful of 1,906, error rate
**2/1779 = 0.112%**, **HSA = 0**, init 623.53 s.

The test resolved in the **second** branch I named when dispatching it — the
fall starts immediately past 60, not at 64:

| conc | tok/s/GPU | n |
|--:|--:|--:|
| 48 | 7,771 | 1 |
| 52 | 8,115 | 2 |
| 56 | 8,326* | 1 (4 failed) |
| **60** | **8,342** | **2 clean, spread 0.20%** |
| 62 | **7,966** | 1 clean |
| 64 | 7,976 | 2 clean |

**Two things worth stating plainly:**

1. **C60 sits on a knife edge.** Going from 60 to 62 — a 3% change in offered
   concurrency — costs **−4.5%** throughput. That is 22× C60's own spread, so it
   is not noise. Anyone adopting C60 as the operating point must know there is
   no headroom above it.
2. **It is a step, not a slope.** C62 (7,966) and C64 (7,976) agree to 0.13% —
   inside C64's own 1.6% spread. Throughput does not decay progressively past
   the peak; it drops once, between 60 and 62, and then sits flat. That shape
   suggests something discrete switches over rather than a gradual
   queueing/contention effect. I do not have the mechanism and am not going to
   guess one from benchmark output alone.

**This partially walks back my dispatch-time framing.** I wrote that a C62
around 8,3xx would mean "a production point at 60 has real margin either side".
It does not. The margin exists below 60 only, and how far below is now the open
question.

### The 8,342 headline comes from runs whose C1 arm produced nothing

Both C60 runs report `conclusion: success` at the GitHub level, run and jobs
alike. **That green tick is misleading.** aiperf aborted C1 in both
(15/145 and 15/146 = 10.3%, over its 10% threshold) and the harness still exits
0, so CI status cannot be used to tell whether a concurrency point yielded a
number.

What survives and what does not:

- **Survives:** C60 = 8,342 (n=2, 0.20%). C1 and C60 are separate jobs with
  separate server launches and materially different configs (C1: dcp=1, mns=8,
  offload=none). C60's own validated error rates were 0.217% / 0.109%, HSA = 0.
  The C1 abort does not contaminate the C60 measurement.
- **Does not:** any claim to a complete curve. **C1 has failed 18 consecutive
  times across every config in this campaign** — it is the N8 fault, not a
  property of C60 or of any operating point I chose.

**Practical consequence:** a single-operating-point result exists and is solid;
a full-curve submission does not. Closing that gap is N8 and nothing else.

### ROOT CAUSE FOUND, AND N8 IS WITHDRAWN: C1 dies on a GPU memory-access fault

I pulled `server.log` and read *above* the traceback instead of stopping at it.
The first event is not a timeout:

```
Memory access fault by GPU node-5 ... Reason: Write access to a read-only page
Memory access fault by GPU node-6 ... (same)
   ... nodes 2,3,4,5,6,7,8,9 — all eight ranks, same second
GPU coredump: execvp failed        (handler binary absent, no dump written)
Worker proc VllmWorker-7 died unexpectedly (exit code: None)
[shutdown] Executor: SIGTERM count=7 -> SIGKILL count=6
core.py:1370  mq.dequeue(timeout=dequeue_timeout) -> RuntimeError("cancelled")
-> EngineDeadError -> HTTP 500 -> 15 failed requests -> ProfileAborted
```

**An illegal write to a read-only page, on all 8 ranks simultaneously,
mid-decode.** The engine was healthy at that instant: 3 running requests, KV
usage 11.4%, MTP accepting normally at AL 4.00. Nothing was saturated.

Reproduced in three independently pulled logs — T173, T175, T178 — each with
8 memory faults, 8 read-only-page reasons, 1 worker death, and **HSA = 0**,
which also separates it cleanly from the C56 `HSA_STATUS_ERROR_OUT_OF_RESOURCES`
fault.

**N8 is withdrawn.** I named it "raise the executor RPC dequeue timeout" and
carried it as the top blocked item for many cycles, on the strength of the
`mq.dequeue(timeout=dequeue_timeout)` frame in the traceback. That frame is the
**last** link in the chain, not the first: the dequeue is cancelled *because*
the executor is already killing dead workers. **Raising any timeout would have
changed nothing.** The mistake was reading a traceback as a cause when it was a
consequence, and not applying to C1 the same "check server.log first" rule I had
already written down for the C56 HSA fault.

**What it actually is:** a defect — an illegal write in a kernel, firing on
every rank at the same step, which points at replicated code (MTP/DSpark or the
DCP path) rather than at scheduling, capacity, or any tunable. It is something
to file against the image, not a knob to turn. No amount of benchmark dispatch
will fix it.

**Consequence for the mns ceiling.** The comment in `kimik3_fp4_mi355x_mtp.sh`
justifying the mns<=80 cap says a 96-row batch "makes the step longer than the
executor's RPC dequeue timeout". That rests on the diagnosis just withdrawn and
is **not trustworthy**. T165's mns=96 failure needs re-reading against
`server.log` before mns 80 is treated as a real ceiling — it may be the same
memory fault, in which case the mns axis was never actually capacity-limited.

### C1 is a genuinely different fault — confirmed, not assumed

When the HSA cause emerged I wrote that it does **not** follow that the C1
aborts share it, and kept the two separate pending evidence. T173 supplies the
evidence. Same run, same node, same hour — the C56 job and the C1 job:

| T173 job | HSA errors | sentinel | EngineDeadError | outcome |
|---|--:|--:|--:|---|
| C56 | **3** | yes | yes | aborted 15.2% |
| **C1** | **0** | yes (×2) | yes (×4) | aborted 15/146 |

**C1 hits the sentinel → `mq.dequeue` → `acquire_read` → `RuntimeError:
cancelled` → `EngineDeadError` chain with zero HSA lines anywhere in
`server.log`.** The C56 job on the same node in the same run had three.

So the two faults are distinct, and the caution was right:

- **C56/C60:** HSA queue exhaustion; sentinel is downstream. Needs a node reset.
- **C1:** sentinel fires with no HSA precursor. **N8 remains the live
  hypothesis here** — and this is the first time it has been isolated rather
  than merely assumed. The thirteen C1 aborts, at a fixed 15 failures with a
  drifting denominator, are not the node.

That also means the N8 investigation is still worth doing; T172 weakened the
*evidence chain* I had been using, not the hypothesis itself for C1.

**Phase E is closed. The settled operating point is C56 = 8,326 tok/s/GPU**,
1,814 successful, error rate 0.220% — the best measured number on this stack and
0.4% past SA's 8,296 (1.3× noise, so parity rather than a decisive win).

C64 itself was a clean run — 1,805 successful of 1,939, validated error rate
**3/1808 = 0.166%**, all gate lines correct — so 8,040 is a real measurement of
a worse operating point, not a degraded one. Note the wall time though: 115 min
against ~50–70 min for the other points. At 64 the engine is already spending
most of its time in the queueing regime that kills it outright at 72.

### T170 C72: the cliff is the same engine death, reached by load instead of mns

C72 ran to the benchmark and then died: **16× HTTP 500 `EngineCore encountered
an issue`** plus 30 `InvalidInferenceResultError` (empty streams from the dead
engine) → **43/284 = 15.141% > 10%** → `ProfileAborted`. The 4,275 tok/s/GPU
printed before the abort is a partial and is **not a measurement**.

Gate lines were all correct — this was not a misconfiguration:
`[dcp] size=8 backend=a2a interleave=1`, `[mns] max_num_seqs=80 conc=72
offload=dram`, `[chunk] 8192`, `[gmu] 0.9`, `[pin-ccd] pinned 1562 threads`,
`graphs: dense ladder 1..80`, init 531.26 s.

What the aiperf tables show at the moment of death:

| | value |
|---|--:|
| Effective concurrency | avg 53.96, **max 85** (mns is 80) |
| Effective decode throughput per user | **4.49 tok/s/user** avg |
| Effective latency (CO-aware) | avg 114.6 s, **max 524.7 s** |
| Tokens in flight | avg 1.95 M, **max 4.63 M** |
| Prompt tokens/request | avg 111,750, 93.32% served from cache |

**This is the T165 failure with a different trigger.** T165 raised `mns` 80→96
and the engine died; here `mns` stayed at 80 and the *offered load* pushed
effective concurrency to 85, past the batch the ladder was captured for.
Requests queued, per-user decode collapsed to 4.5 tok/s, CO-aware latency ran to
524 s, and the executor RPC blew its timeout exactly as before.

So the two knobs are the same knob. Batch pressure — whether supplied by `mns`
or by concurrency — hits **one** ceiling, and that ceiling is the executor RPC
`dequeue_timeout` (**N8**), not memory: KV was never the binding resource in
either run. N8 was already the top blocked item for C1; it now also caps the
throughput curve. Unblocking it is worth more than any remaining config lever.

**8,326 also passes SA's 8,296 for the first time** — by 0.4%, which is only
1.3× the noise floor, so call it *parity reached*, not a decisive win.

The lesson worth keeping: C52 was never a tuned choice, it was inherited. Nine
config experiments (N1–N9) moved C52 from 7,906 to 8,127, about +2.8%. Simply
running at 56 instead added another +2.6% in one run. **The operating point was
worth as much as the entire config search.** C64 and C72 come next.

## T165: the C1 crash is not C1-specific, and my aigmkt prediction was wrong

T165 C52 died mid-replay at mns 96 with the **identical** trace to every C1 abort:

```
engine_core_sentinel.py:179 run_with_fault_tolerance
  -> mq.dequeue(timeout=dequeue_timeout) -> shm_broadcast acquire_read
  => vllm.v1.engine.exceptions.EngineDeadError
```

Two things this settles:

1. **I predicted aigmkt would not have this crash. It does.** The sentinel is in
   `v0.26.1rc1.dev1133+gf94666b60` too, so "the nightly's new fault-tolerance
   subsystem" was the wrong framing. The prediction is withdrawn.
2. **It is not memory.** The engine's own dump at death reads
   `num_running_reqs=45`, `kv_cache_usage=0.28`. No OOM, no HSA fault. The
   worker simply did not answer the executor's RPC inside the dequeue timeout.

**mns 96 is a settled negative and reverted.** mns 80 completed twice on this
exact image (T163 8,127, T164 7,528). The mechanism that fits: a 96-row batch
makes the step long enough to exceed the timeout, and the sentinel turns a slow
step into a fatal error. Same mechanism as C1, where k=8 MTP over ~430k-token
prompts produces the long step.

**Consequence for the target:** the resident-sequence axis is capped by an RPC
timeout, not by hardware. Raising that timeout is now the highest-value unblock
— it would reopen mns and very likely fix C1 as well. `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1200`
is already set and is **not** the binding one; the actual `dequeue_timeout` on
the `multiproc_executor.get_response` path still needs to be located in source.
| **T163** | **dram restored** + pin after ready | **8,127 — best to date** |

**T163 is the new best: 8,127 tok/s/GPU**, 1,955 successful, error rate 0.102%.
The offload is worth **+3.9%** (vs T161's 7,824 on the identical stack) — larger
than the +1.8% I estimated from T160, and it also clears T160's 7,968 by +2.0%
with pinning applied after ready rather than during the load. **SA's 8,296 is
now only 2.0% away.** Note the connector allocated 226.89 GB/rank here, not the
243.6 GB of earlier runs, so the exact offload size is not held constant across
the ledger.

**CCD pinning is worth +0.78% at C52** (T160 7,968 vs T156 7,906, same stack, pinning
the sole variable), and edges past the old aigmkt baseline T103 by +0.2%. Clean run:
1,899 successful, aiperf error rate **0.105%** (2/1901), full 3,640 s profiling phase.

The gain is small but it is real and it is the first C52 lever that is not negative.
Cost: T160's weight load took **2008.62 s** against 576–681 s unpinned, because the
pre-pin background loop fires during loading and confines ~190 loader threads per
worker to one CCD's 8 physical cores. That is wall-clock only — it does not touch the
serving-window throughput — but it adds ~23 min to every run. Fixed for the next run:
pinning is now one-shot **after `wait_for_server_ready`**, background loop deleted.

## The noise floor is 0.30%, and it mostly vindicates the ledger

T168 re-ran T163's exact config and got **8,103** against **8,127** — a spread
of **0.30%**. Every delta in the ledger can now be read against that:

| claim | delta | vs noise | verdict |
|---|--:|--:|---|
| chunk 4096 (T164) | −7.4% | 25× | real |
| DRAM offload (T163 vs T161) | +3.9% | 13× | real |
| async scheduling (T162) | −1.8% | 6× | real |
| dram+pin-after-ready vs T160 | +2.0% | 7× | real |
| **CCD pinning (T160 vs T156)** | **+0.78%** | **2.6×** | **weakest — treat as provisional** |

**Caveat on the caveat:** n=2 gives a point estimate of spread, not a confidence
interval. 0.30% is one observation of run-to-run difference, not a standard
deviation. The +0.78% pinning claim is the only one close enough to the floor
that a third sample could overturn it; everything else clears by 6× or more.

Best C52 stands at **8,127** (T163), replicate 8,103, mean 8,115.

**The direct DCP a2a is not available on this image.** T167 flipped
`VLLM_USE_DIRECT_DCP_A2A` 0 → 1 and every worker died during cudagraph capture:

```
AttributeError: '_OpNamespace' '_C' object has no attribute
                'direct_dcp_a2a_lse_reduce'
```

I had read the script's `=0` as *force-disabling a fast path we'd never
measured*. Wrong: it disables a **compiled C++ op that `aigmkt/kimi-k3-vllm`
does not ship**. #51705 adds the Python plumbing, not the kernel, to this build.
Re-enabling needs a rebuilt image, which is out of bounds. Reverted to 0 with
the error recorded next to it.

**gmu > 0.90 is settled: the headroom is not usable.** T166 at 0.92 got 0
successful out of 103. The server started and KV grew **59.8 → 65.6 GiB
(+9.7%)** — the memory really is there — then it hung in warmup and never
served a request. T157 at 0.95 hung identically (0/57). Two points above 0.90
hang, 0.90 works. Reverted, and the fallback ladder's "try gmu first" advice is
wrong in the upward direction.

Settled negatives: QuickReduce FP −8.39% · EP=8 −4.7% · **async −1.8% on the nightly** (was −9.2% on the old engine; retested T162, still the wrong sign) · **chunk: 8192 is the peak** — 4096 is −7.4% (T164), 16384 is −2.5% · FP16 GEMM loses 6/8 shapes.

**The async result matters beyond its own sign.** The profile's biggest single
item — ~150 s of 403.9 s idle attributed to host/Python batch prep — is the
thing async scheduling exists to overlap, and overlapping it makes throughput
*worse*. So either that attribution is wrong, or the host work is already off
the critical path and the idle has another cause. Treat the "37% of idle is
host" figure as unconfirmed until something else moves it.
**The nightly is a C1 lever, not a C52 one** — #53942 is explicitly an m=1/m=2 change and cannot apply at batch 52.

## C1 — all nightly numbers are RETRACTED

Every nightly C1 run ends in **one EngineCore 500 crash** → 12 connection-refused → aiperf cancels at its 10% threshold. Always 17/148, always ~1980 s of a 3600 s window.

| run | quoted | status |
|---|--:|---|
| T123 (aigmkt) | **7.71 ms**, 1,288.2 | **valid** — 190/193 over 3609 s |
| T156 / T158 / T160 | 7.89 / 8.06 / 7.71 | **withdrawn — engine crashed** |

The CCD-pinning "−2.3%" is withdrawn with them. **Root cause found** (detail below): a
`sample_tokens` collective RPC exceeds its dequeue timeout and the nightly's new
fault-tolerance sentinel turns that into a fatal `EngineDeadError`. No worker fault
precedes it. Prediction: **aigmkt does not crash** — which the sa.sh run tests.

## T161 — pinning after server-ready, and a rule I had backwards

**Pinning after `wait_for_server_ready` works.** Weight load **176.83 s** against
T160's 2008.62 s — 11.4× — with the pin still landing (1,554 threads). Every
future run gets that back. `bash -n` clean, gate lines all correct.

**The C1 crash is not the pinning.** T161 C1 aborted identically (`ProfileAborted`,
15/146 = 10.274%, 131/148) with the pre-pin loop gone. The nightly's
`engine_core_sentinel` remains the suspect.

**mns 80 + `kv-offloading: none` no longer OOMs on our node.** It completed
1,856 requests, error rate 0.161%. The 3/3 `HSA_STATUS_ERROR_OUT_OF_RESOURCES`
history for that combination did not reproduce. The fallback ladder stays
written down but is not currently needed.

**Correction — I had the offload rule backwards.** T161 is *not* a clean
pin-timing A/B: it also flipped the offload `dram` → `none`, because the working
tree had been replaced by the sa.sh copy. Read as an offload A/B instead, every
direct throughput comparison we have favours **dram**:

| | dram | none | delta |
|---|--:|--:|--:|
| ours, mns 80 | **7,968** (T160) | 7,824 (T161) | **−1.8%** |
| ours, T103 vs T133 (mns 80 vs 65) | 7,950.6 | 7,725.96 | −2.8% |
| SA, mns 80 | 8,296 | 8,204 | −1.1% |

The T116/T124 idle finding was real — idle 44.3% → 28.2%, >10 ms stalls −57% —
but **it did not convert into throughput**. Removing the offload removes stalls
*and* removes the KV capacity that keeps the batch full; the second effect is
larger. Restore `dram` for the best-config runs.

## Rules learned the hard way

- **DRAM offload ON at C52** (+1.1–2.8% in three independent A/Bs). The earlier
  "offload OFF" rule was inferred from idle, not throughput, and is withdrawn.
- **DCP OFF at C1** (+36.5% TPOT), **ON at C52**.
- `mns` 80 + no offload OOMs on **our node only** — SA gets 8,204 with it.
- Read the aiperf error summary **before** quoting any throughput number.

---

# Detail — 2026-08-28/29 session

## ROOT CAUSE of the C1 engine crash: an RPC timeout promoted to a fatal error

From `results/server.log` (T160 C1 artifact; the runner console log stops at
`Application startup complete` and never sees this):

```
EngineCore.step
  -> model_executor.sample_tokens
  -> collective_rpc -> get_response
  -> mq.dequeue(timeout=dequeue_timeout)
  -> shm_broadcast.py:797 acquire_read
  -> RuntimeError("cancelled")
=> vllm.v1.engine.exceptions.EngineDeadError
```

**Nothing failed on the worker side.** The last worker line before the crash is
an ordinary aiter GEMM (`M:2247 N:3072 K:512`, a prefill chunk). No HSA fault,
no CUDA error, no OOM, no traceback from any rank. A worker simply did not
answer the shared-memory queue within the dequeue timeout, and the engine
concluded it was dead.

Note the frame `vllm/v1/fault_tolerance/engine_core_sentinel.py:179
run_with_fault_tolerance`. That subsystem is **new in the nightly**. Likely
mechanism:

> a single very long step -- the agentic replay's p95 input is **430,904
> tokens** -- exceeds the RPC dequeue timeout, and where the older engine simply
> waited, the sentinel promotes it to a fatal `EngineDeadError`.

That fits every observation: it fires **once**, always at the same point in the
replay (hence the constant 17/148), it kills the server so the following 12
requests are connection-refused, and it never appeared on `aigmkt` -- T123
completed 190/193 over 3609 s on that image.

**Testable prediction:** sa.sh on `aigmkt/kimi-k3-vllm:latest` will *not* abort
at C1. If it does abort, the cause is the config or the replay rather than the
engine version, and this explanation is wrong.

If confirmed, the fix is a longer RPC timeout, not a performance change.
`VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1200` is already set by the script, so the
binding timeout is a *different* one on the `sample_tokens` path and still needs
locating.



## C1 interactivity sweep, 2026-08-28 (T138–T151) — CURRENT

Fixed-length screen: ISL 122k (effective ~63.8k after the BPE round-trip),
OSL 500, 100 requests, DCP off, k=8 @ AL 4.00, draft KV fp8.
Chosen to match the agentic replay's 122,657-token mean input, because DCP's
value is context-length dependent and an 8k screen would mis-rank it.

| run | image | DCP | mns / ladder | TPOT mean | p50 | p90 | ITL |
|---|---|--:|---|--:|--:|--:|--:|
| [T147](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/33171360827) | **nightly 6f7df92a8e** | 1 | 8 / 1..72 | **7.57** | 7.47 | **7.91** | 29.47 |
| [T148](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/33183801155) | nightly | 1 | 8 / 1..72 | 7.57 | 7.49 | 7.93 | — |
| [T150](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/33185946417) | nightly | 1 | **1 / 1..9** | 7.58 | 7.48 | 7.93 | — |
| [T151](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/33187965065) | aigmkt | 1 | 8 / 1..16 | 7.93 | 7.77 | 8.20 | — |
| [T145](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/33168461313) | aigmkt | 1 | 8 / 1..72 | 8.77 | 8.63 | 9.06 | 34.16 |
| [T146](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/33169840000) | aigmkt | 1 + comm flags | 8 / 1..72 | 8.79 | 8.64 | 9.07 | 34.29 |
| [T144](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/33166811023) | aigmkt | **8** | 8 / 1..72 | 11.97 | 11.91 | 13.71 | 46.72 |
| [T143](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/33165123199) | aigmkt | **8** | 8 / 1..72 (draft bf16) | 11.93 | 11.84 | 13.70 | 46.54 |

**Four results, in order of size.**

1. **DCP=8 costs +36.5% TPOT at C1** (T144 vs T145: 11.97 → 8.77) and widens the
   tail (13.71 → 9.06 p90). It buys a 6.4× KV pool — 20,580,438 vs 3,218,642
   tokens — which is the one resource that does not bind with a single resident
   request. DCP adds a per-layer a2a + KV gather + LSE merge to a step that is
   already barrier-bound, to parallelise attention work that is negligible at
   batch 9. **SA reached the same conclusion independently: their C1 runs
   `decode_context_parallel_size=1`.** The doc's older "SA has DCP on" claim is
   about their **C52** arm.
2. **The nightly is worth −13.7%** (T145 → T147, 8.77 → 7.57, sole variable =
   175 vLLM commits). Most likely #53942 (`eh_proj` low-latency GEMM, explicitly
   "enabling m=1 and m=2", which is exactly the C1 batch regime, 12.9–25.2%
   kernel improvement) and #53818 (ROCm graphs were being captured on a stream
   that never ran warmup).
3. **Neither ladder size nor scheduler headroom matters at C1** on the nightly:
   mns 8/ladder 72, mns 8/ladder 16 and mns 1/ladder 9 all land 7.57–7.58 mean,
   7.91–7.93 p90. But ladder 9 cuts graph capture **44 s → 7 s** and graph
   memory 1.46 → 0.83 GiB/GPU, and grows the KV pool 2.3%. Free; take it.
4. **`--dcp-comm-backend a2a` and `--cp-kv-cache-interleave-size 1` are inert at
   DCP size 1** (T145 vs T146: identical TPOT, byte-identical KV pool
   3,218,642). They only do work when there is a CP group.

**Unexplained, stated rather than buried:** on the *aigmkt* image, ladder 72 → 16
plus fastsafetensors → auto gave −9.6% (8.77 → 7.93), yet both deltas measured
**inert** on the nightly. The image digest was identical across both runs
(`sha256:7f0dfe6304c9`, build `dev1133+gf94666b60`), so the tag did not move.
Working hypothesis, not established: #53818 fixes graph capture on a stream that
never ran warmup, so a 72-graph ladder had more exposure to that bug on the old
engine. The same pair also leaves the old T123 (6.70) vs T133 (7.18) gap
unexplained — both of its candidate causes are now measured inert.


## CCD pinning matched zero threads — and why

T159 logged `[pin-ccd] pinned 0 threads`, so the run is not a test of pinning at
all; 1,511 tok/s/GPU simply reproduces T158's 1,515.

**Cause:** the matcher was `pgrep -f "VLLM::Worker_TP${_g}_"` — with a trailing
underscore. That form only exists when DCP is on, where the process title is
`Worker_TP0_DCP0`. At C1 DCP is off and the title is `Worker_TP0`, with nothing
after it, so the pattern matched nothing on every one of the 8 ranks.

This is the *second* defect in the same 20 lines. The archived note already
recorded the first: `taskset -pc <pid>` sets affinity for one TID, so with ~197
threads per worker it pinned 1 and left ~196 roaming — which is why pinning
"never showed a win" historically. Both bugs share a shape: **the code ran, printed
a plausible-looking line, and silently did nothing.** `pinned 0 threads` is the
only reason this one was caught rather than logged as another null result.

Fixed to `pgrep -f "VLLM::Worker_TP${_g}([^0-9]|$)"`, which matches both
`Worker_TP0` and `Worker_TP0_DCP0` and does not match `Worker_TP1`. Verified
against all three strings. The same stale pattern in the wait-for-worker loop
was fixed too.

**CCD pinning remains UNMEASURED.** Do not record T159 as evidence either way.


## CCD pinning slows model load 3.6x -- it is applied DURING the load

Measured from T160 C1 against the unpinned runs:

| run | pinning | weight load | serve -> ready |
|---|---|--:|--:|
| **T160 C1** | **pinned** | **2133.9 s (35.6 min)** | **39.3 min** |
| T158 C1 | none | 600.7 s | 14.0 min |
| T156 C1 | none | 681.4 s | 15.3 min |
| T151 C1 | none, aigmkt | 155.2 s | 7.9 min |

The background pre-pin loop fires as soon as the workers appear -- first
`[pin-ccd] pinned 1418 threads` at 01:48:19, well before loading finishes -- and
confines each worker's ~190 threads to **one CCD = 8 physical cores**. Weight
loading is embarrassingly parallel (safetensors decode plus H2D staging), so
capping it at 8 cores serialises it.

**The pre-pin loop is also unnecessary.** The post-`wait_for_server_ready` call
walks `/proc/<pid>/task/*` and catches every thread, including any spawned
during load. If CCD pinning is retried, **delete the background loop and pin
only after the server is ready** -- otherwise every pinned run pays ~25 extra
minutes of load and the comparison is confounded by it.

Separately worth noting: the *unpinned* nightly runs load 4x slower than the
aigmkt run (600-681 s vs 155 s). That is a different effect -- image or storage
contention -- and is not explained by pinning.


## RETRACTION: every nightly C1 number came from a run where the ENGINE CRASHED

The aiperf error summary from T160 C1 explains the 17/148 and invalidates the
numbers built on it:

    N/A  InvalidInferenceResultError   2
    500  Internal Server Error         1     <- EngineCore encountered an issue
    N/A  ClientConnectorError         12     <- ConnectionRefused on 127.0.0.1:8888

    Run aborted (failed_request_threshold): 15/146 profiling requests failed
    (10.3%), exceeding the --failed-request-threshold limit of 10.0%.

**This is ONE engine crash, not 15 request failures.** EngineCore dies, the
server stops listening, and the following 12 requests get connection-refused.
aiperf then trips its 10% threshold and **cancels the run**. That is why the
count is always ~17/148 and the window is always ~1980 s instead of the 3600 s
DURATION -- it is deterministic because the crash lands at the same point in the
replay.

**Consequently these numbers are withdrawn as measurements:**

| run | quoted | status |
|---|--:|---|
| T156 C1 | 1,509 tok/s/GPU, ITL p50 7.89 | engine crashed mid-run |
| T158 C1 | 1,515, ITL p50 8.06 | engine crashed mid-run |
| T160 C1 | 1,521, ITL p50 7.71 | engine crashed mid-run |

They describe a truncated prefix of the replay that ends at an engine crash, and
they are **not comparable to T123's 1,288.2 / 7.71 ms**, which completed 190/193
requests over 3609 s. In particular the CCD-pinning "-2.3%" I reported from T160
rests on a crashed run and should not be treated as evidence.

The engine's own stack trace is written to `results/server.log`, which the
runner console log does not capture -- the runner log ends at
`Application startup complete`. Root-causing the crash requires that artifact.

**The crash is now the top priority.** It invalidates C1 measurement entirely,
and it is present on the nightly + rebased #51705 stack. Whether it also occurs
on `aigmkt/kimi-k3-vllm:latest` is the immediate question, since T123 on that
image did not crash.


## First real CCD-pinning measurement: C1 ITL p50 7.89 -> 7.71 ms

T160 C1 is the first run where the pin actually applied -- `[pin-ccd] pinned
1418 threads` then `1490` -- after fixing all three defects.

All nightly + rebased #51705, C1, DCP off, k=8, mns 8, ladder 1..16:

| run | pinning | ITL p50 (ms) | tok/s/GPU |
|---|---|--:|--:|
| T156 | none | 7.89 | 1,509 |
| T158 | none (+nccl32) | 8.06 | 1,515 |
| **T160** | **1,490 threads** | **7.71** | **1,521** |

**-2.3% on ITL p50 against T156. Do not over-read it.** T156 and T158 differ by
2.2% (7.89 -> 8.06) from an RCCL change that is independently known to be
*negative*, which puts run-to-run spread at C1 in the same 2% band as the effect
being claimed. One run cannot separate them.

The C52 arm is the better test and is still running: 8 workers of ~190 threads
each on a 2-socket box where GPU0-3 threads were observed executing on node1
cores, so the cross-socket traffic pinning removes is far larger there than at
C1 with a single resident request.

All four nightly C1 runs have now dropped **exactly 17/148**.


## The pinning diagnostic killed a C52 run -- third bug in the same block

T159 C52 reached `Application startup complete`, logged `[pin-ccd] pinned 1562
threads`, and then went straight to `Stopping vLLM server`. **The replay never
ran.** No throughput, no requests, ~50 min of GPU time lost.

**Cause, mine:** the stray-affinities diagnostic I appended does

    _stray=$(for _t in /proc/$_p/task/*; do taskset -pc "${_t##*/}" 2>/dev/null | sed ...; done | ... | wc -l)

Under `set -euo pipefail`, `taskset` failing on any thread that has already
exited makes the pipeline non-zero, the command substitution non-zero, the
assignment non-zero, and `set -e` terminates the script -- firing
`cleanup_agentic_services`, which stops the server.

**The C1/C52 asymmetry is the proof.** C1 pinned 0 threads, so the loop body and
the diagnostic never executed and the run completed. C52 pinned 1,562, executed
it, and died. Same script, opposite outcomes, explained entirely by whether the
pgrep matched.

**Fix:** the diagnostic is deleted outright -- it was never load-bearing -- and
both `pin_workers_to_ccd` call sites are now `|| true`, with the inner
`taskset` guarded by `|| true` as well. Verified: a `set -euo pipefail` shell
running the same loop against a nonexistent pid now survives.

**Three defects have now been found in these ~20 lines**: (1) `taskset -pc <pid>`
pinned 1 TID of ~197, (2) the pgrep pattern required a trailing underscore that
only exists under DCP, (3) the diagnostic aborted the run under `pipefail`. Each
one silently produced a plausible-looking log. **CCD pinning is still
UNMEASURED.**


## The C1 11.5% drop rate is SYSTEMATIC, not noise

T156 C1 and T158 C1 both dropped **exactly 17 of 148** requests (11.5%), on
different engine settings, both finishing at ~1983 s having served 131. T123 on
the same replay dropped 3/193 (1.6%) over 3609 s.

Identical counts across two independent runs rules out randomness. Something in
the current C1 configuration fails a fixed subset of the replay. Candidates, in
order: the 1M `max-model-len` against requests whose context exceeds what the
DCP-off KV pool (3.3M tokens) can hold; a client timeout on the longest
trajectories; or a spec-decode path that errors on particular inputs.

**Consequence: no C1 tok/s/GPU from this configuration is comparable to T123's
1,288.2.** Dropped requests consume time but are excluded from the success
accounting, and the window is 1983 s against 3609 s. Report ITL p50 for C1
until this is fixed — that metric is per-token and robust to the drops:

| | ITL p50 (ms) | tok/s/GPU | dropped |
|---|--:|--:|--:|
| T123 (aigmkt) | **7.71** | 1,288.2 | 1.6% |
| T156 (nightly) | 7.89 | 1,509 | 11.5% |
| T158 (nightly + nccl32) | 8.06 | 1,515 | 11.5% |

Diagnosing the drops is now worth more than the next config knob.


## RCCL: more channels is worse at C52

T158, sole variable `NCCL_MIN_NCHANNELS=32`:

| | tok/s/GPU |
|---|--:|
| T156 (default channels) | **7,906** |
| T158 (`NCCL_MIN_NCHANNELS=32`) | **7,656** |

**−3.2%.** KV pool identical (31,924,580), requests comparable (1801/1911 vs
1879/1989), input throughput 60,576 vs 62,665 tok/s. Reverted.

More channels splits each collective across more parallel streams, which helps
bandwidth-bound transfers but adds per-channel launch and synchronisation
overhead. At C52 the collectives are already large enough to saturate, so the
extra channels are pure overhead. **Settled — do not raise the channel count.**
`NCCL_PROTO` is a different axis and remains untested, but this result lowers
the prior on RCCL tuning generally.


## gmu 0.95 hangs the engine at C52 — do not retry

T157, C52, sole variable `gpu-memory-utilization` 0.90 -> 0.95:

- KV pool grew **31,924,580 -> 40,222,007 tokens (+26%)**, so the setting took.
- Server reached `Application startup complete`.
- Then **every one of 55 warmup requests hung**: `returned=0/107, sent=55,
  in_flight=55, errors=0` for **1200 s**, and the client dropped all 57.

**0 successful requests.** It is not an OOM — there is no `HSA_STATUS_*`, no
allocation failure, and the engine reported no errors. It accepted work and
produced nothing, which points at exhausted activation/workspace headroom
stalling the step rather than a clean allocation failure.

vLLM itself flagged the margin in the same log: *"--gpu-memory-utilization=0.9500
is equivalent to 0.9349 without CUDA graph memory profiling"*. At mns 80 +
DCP=8 + the DRAM offload, 0.9349 effective leaves too little.

**Settled: gmu stays 0.9.** The weak prior in the queue was right — KV usage is
only ~28% at C52, so the pool was never the constraint, and buying more of it
cost the run entirely.


## C1 on the nightly: 1,509 tok/s/GPU, but the run is dirty

T156 C1, agentic, nightly + rebased #51705, DCP off, k=8, mns 8, ladder 1..16:

| | T123 (aigmkt) | T133 (aigmkt) | **T156 (nightly)** |
|---|--:|--:|--:|
| tok/s/GPU | 1,288.2 | 1,237.2 | **1,509** |
| TPOT/ITL p50 (ms) | **7.71** | 8.69 | 7.89 |
| requests | 190/193 | 184/187 | **131/148** |
| error-dropped | 1.6% | 1.6% | **11.5%** |
| duration (s) | 3609 | 3556 | **1985** |

**Do not quote the +17% throughput.** The run served 131 requests in 1985 s
against T123's 190 in 3609 s, and dropped 11.5% of requests versus 1.6%. A
higher error rate inflates tok/s/GPU because dropped requests cost time but are
excluded from the numerator's accounting, and the shorter window means a
different slice of the replay. On the metric that is robust to this — **ITL
p50 — the nightly is 7.89 vs T123's 7.71, i.e. slightly WORSE**, not better.

Also note `Inter Token Latency` min = **0.00 ms** again, the same degenerate
-request artifact found in SA's logs, so p95 (2,701 tok/s/user) and p99 (13,009)
are junk. p50 only.

**Needs a clean re-run before it goes in any summary.** The 11.5% error rate is
itself the finding worth chasing.


## The nightly does NOT help C52 throughput

T156, agentic, nightly `6f7df92a8e` + rebased #51705, DCP=8, mns 80, DRAM
offload, load-format auto — the same recipe as T103 and SA:

| | tok/s/GPU | image |
|---|--:|---|
| SA C52 | **8,296** | aigmkt |
| T103 | **7,950.6** | aigmkt |
| **T156** | **7,906** | **nightly + #51705** |

**−0.6% against T103 — flat, arguably a hair worse.** So the **−13.7% TPOT the
nightly bought at C1 does not transfer to C52 throughput.**

That is consistent with the mechanism rather than a surprise: the headline
candidate, **#53942, is explicitly an "enabling m=1 and m=2 for low latency
gemm" change**. m=1/m=2 is the C1 batch regime. At C52 the decode batch is 52
and the GEMMs are already large enough that the low-latency dispatch does not
apply. #53818 (graphs captured on a stream that never ran warmup) likewise
matters most where per-step launch overhead dominates, which is C1.

**Consequence for the plan:** the nightly is a C1 lever, not a C52 lever. C52
throughput still sits at ~7,900–7,950 and the remaining gap to SA's 8,296 is
~4.9%, unexplained by anything in the engine version. The C52 levers worth
spending GPU time on are the ones that attack idle and dense GEMM directly —
AITER tuned configs (45,250 misses), #52190 torch.compile (currently zero
post-grad fusion), CCD pinning — not newer vLLM.

Run detail: 1879/1989 requests (93 error-dropped), KV pool 31,924,580 tokens,
input 62,665 tok/s, 3629.7 s window, GSM8K 0.99 on this exact config.


## GSM8K with MTP on scores 0.14 — and no baseline ever covered this

T154, nightly + rebased #51705, GSM8K limit 200:

| arm | DCP | k | GSM8K |
|---|--:|--:|--:|
| C52 | 8 | **0** | **0.99** strict + flexible |
| C1 | 1 | **8** | **0.14** flexible / **0.13** strict |

**Every accuracy baseline this project has ever recorded was taken with MTP
OFF** — 98.5% (T97, the T103 config), 0.9659/0.9644 (trial 23), 96.82/96.89.
There is no prior GSM8K with speculative decoding enabled. So 0.14 is not a
regression against a known-good number; it is the first measurement of a
configuration nobody had ever gated.

**Leading explanation:** `rejection_sample_method: "synthetic"` with
`synthetic_acceptance_length: 4.00` **imposes** the accept length rather than
verifying draft tokens against the target. If drafts are accepted without real
verification, the emitted text is corrupt by construction and 0.14 is expected
behaviour of the measurement methodology, not a defect in the engine or in the
#51705 rebase. The alternative is that the rebase broke the draft path.

T155 separates them: same stack, C1, **k=0**, everything else identical.
~0.99 confirms the synthetic explanation; ~0.14 indicts the rebase.

**If the synthetic explanation holds, it has a consequence worth stating
plainly: every C1 TPOT number in this document — ours and SA's, since SA uses
the same `synthetic` setting — is measured on a configuration that does not
produce usable output.** That is fine for ranking engine changes, which is what
the golden-AL methodology is for, but it means "best TPOT at C1" is not by
itself a shippable result. A shippable C1 number needs real rejection sampling,
which would change the accept length and therefore the TPOT.


## Draft model: constraints found while optimising it

- **The DSpark draft cannot leave `TRITON_MLA`.** It is the only ROCm MLA backend
  declaring `supports_non_causal_multi_token_decode`, which the non-causal draft
  requires; `flashinfer_mla` and `tokenspeed_mla` also declare it but are NVIDIA.
  `ROCM_AITER_MLA` inherits the base default `False` and `mla_attention.py`
  raises `"Non-causal multi-token MLA requires an explicitly supported attention
  group"`.
- **Flipping that ClassVar to `True` would be unsafe**, not merely ineffective:
  the flag is a declaration, not an implementation, the aiter ASM path has no
  gqa=64 kernel past qseqlen 1, and because `rejection_sample_method` is
  `synthetic` the accept length is **imposed**, so wrong drafts would still
  report AL 4.00. That whole class of change is untestable on this harness and
  needs real rejection sampling + GSM8K.
- **`TritonMLAMetadataBuilder._cudagraph_support` must stay `UNIFORM_BATCH`.**
  Cudagraph capability is the **minimum across attention groups**, so with
  upstream's `UNIFORM_SINGLE_TOKEN_DECODE` the draft demotes the whole engine to
  PIECEWISE and runs eager: **14.05 → 77.65 tok/s, ITL 71.16 → 12.88 ms**.
  Acceptance test: the server log must show `Capturing model for DSpark
  speculator...`.
- **Draft KV `auto` → `fp8` is TPOT-neutral but grows the KV pool 36.5%**
  (15,077,972 → 20,580,438 tokens in the same 53.84 GiB). The draft was holding
  KV at 2 bytes/element while the target held 1. Keep fp8.


## SA comparison, corrected twice

Reading SA's logs directly (read-only) retired two claims in this document.

| SA run | conc | offload | mns | result |
|---|--:|---|--:|---|
| [32968517728](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/32968517728) | 52 | **dram** | 80 | **8,296** tok/s/GPU |
| [33062469329](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/33062469329) j98484457387 | 52 | **none** | 80 | **8,204** tok/s/GPU |
| 33062469329 j98484457440 | 1 | none | 8 | ITL p50 **8.64** ms |
| [33083417848](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/33083417848) | 1 | none | 8 | ITL p50 **10.21** ms |

- **"SA C52 runs no offload" was wrong** — the 8,296 reference runs `dram`. Both
  their C52 configs are the same shape as our T103 (7,950.6), so we are **3–4%
  behind on an identical recipe**, not blocked on something we cannot run.
- **"`mns` 80 without the offload died 3/3 including both SA C16/C52" was wrong.**
  SA ran exactly that and got 8,204 on `mi355x-amds_01`. The OOM
  (`HSA_STATUS_ERROR_OUT_OF_RESOURCES`) is specific to `mi355x-amd_b23_07`.
  **It is a node limit, not a config limit.**
- **The 1.56× "SA is faster at C1" gap is a client metrics artifact.** Their two
  runs of the *same* config differ only in `RUNNER_NAME`, and one records
  `Decode Duration` min = **0.00 ms**, which makes `1/tpot` explode: `intvty`
  p95 3,465 and p99 22,987 tok/s/user (0.043 ms/token — not physical), mean
  output 1,280.95 tok/s/user with max 65,065, against the clean node's 100.24 /
  207.66. Aggregate throughput is identical (1,236 vs 1,239). **Treat any SA C1
  percentile above p50 as suspect unless `Decode Duration` min is non-zero.**
  Like-for-like we are at parity: ours T133 = 1,237.2 tok/s/GPU, TPOT mean 7.18,
  p50 8.69, p90 11.04.


> **Where the time goes — now MEASURED,** by rocprofv3 on
> [T116](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/32964875218)
> (2.3 GB trace, 6.36M dispatches). See
> [Kimi-K3-Where-The-Time-Goes.md](Kimi-K3-Where-The-Time-Goes.md).
> **The GPU is idle 44.1% of the serving window.** Of the 55.9% it is busy:
> collectives **34.3%**, MLA attention **21.8%**, dense BF16 GEMM **15.3%**,
> MoE (already FP8×MXFP4) 12.5%, KDA 5.2%.
> This retires two earlier claims from this document: attention is **21.8%, not
> 5.6%**, and dense BF16 GEMM is **15.3%, not 94%**.
> **Biggest lever is the idle, then collectives — not any GEMM.**

## FP8 attention weights: tested and rejected (T117–T119)

`amd/Kimi-K3-Quark-MXFP4-AttnFP8` converts exactly the 15.3% dense-BF16 block
(`q/k/v/o_proj`, MLA `q_a/q_b/kv_a/kv_b`) to F8_E4M3, keeping routed experts
MXFP4. Two real blockers were found and **both fixed**:

1. **Load OOM at shard 11/12, 272 MB free.** Not size — it needs 188.2 GB/GPU
   against moonshotai's 195.1. Cause: `fastsafetensors` stages a whole shard
   batch on device (8 × ~15.7 GB ≈ 117 GiB). Fix: `LOAD_FORMAT=auto`.
2. **Wrong MoE kernel.** AMD's config declares MXFP4 *activations*, so
   `OCP_MX_Scheme` resolved to `w_mxfp4_a_mxfp4` → `AITER_MXFP4_MXFP4`
   ("W4A4: CK kernel"), not the `AITER_MXFP4_BF16` + situv2 a8w4 path every
   working run uses. Fix: `global_quant_config.input_tensors = null` in the
   downloaded checkpoint (weights bit-identical; MXFP4 activation quant is
   `is_dynamic`, so nothing stored is invalidated; `*self_attn*` keeps its own
   fp8 spec so FP8 attention survives).

**It still loses, on memory:**

| | moonshotai | AttnFP8 |
|---|---:|---:|
| Model loading took | 192.56 GiB | **240.75 GiB** |
| Available KV cache | 54.41 GiB | **9.38 GiB** |
| GPU KV cache | ~32.7M tok | **5,391,048 tok** |
| Prefix cache hit | 93.8% | **0.0%** |

**+48 GiB/GPU more resident while 7 GB/GPU smaller on disk** — unexplained, and
*not* MoE tile padding (K3's AITER A16W4 SiTU kernel takes native intermediate
size). At conc 52: `Running: 1 reqs, Waiting: 18`. Cancelled rather than measure
KV starvation. **Verdict: trading a 93.8% prefix cache for a block worth +3–5%
is a bad trade.** Parked until the +48 GiB is explained.

## Interactivity: TPOT 22.43 → 10.30 ms at concurrency 1 (2.18×)

Goal was 3×. We got **2.18×**, and the shortfall is structural, not a missing knob.

| Metric | T106 base | T121 DCP off + no offload | **T122 + MTP** | SA c1 |
|---|---:|---:|---:|---:|
| **TPOT mean** | 0.02243 | 0.02165 | **0.01030** | 0.02156 |
| TPOT p90 | 0.02851 | 0.02732 | **0.01357** | 0.02666 |
| **Interactivity** tok/s/user | 44.58 | 46.20 | **97.08** | 46.37 |
| **tok/s/GPU** | 940.0 | 973.3 | **1,210.4** | 980.6 |
| e2e mean | 19.32 s | 18.56 s | **16.48 s** | 18.45 s |
| TTFT mean | 1.429 | 1.367 | **1.964** | 1.241 |
| TTFT p95 | 5.043 | 4.821 | **6.277** | 3.394 |

**Decomposition — this is the useful part:**

- **DCP off + offload off = −3.5% only** (22.43 → 21.65 ms). It closed the gap to
  SA c1 exactly (973.3 vs 980.6), so the ~4% deficit *was* the DRAM offload. But
  it proves TPOT at conc 1 is **barrier-latency-bound**: ~118 global 8-rank syncs
  per decode step, which no config change removes.
- **MTP = the whole 2.18×** (21.65 → 10.30 ms). It doesn't make a step cheaper —
  it amortises the step over ~2.5 accepted tokens.

**Measured acceptance was real, not assumed:** mean acceptance length **2.48–2.55**,
draft acceptance rate **77%**, which validates the 2.51 synthetic constant.

**Why 2.18× and not 2.54×:** MTP costs draft forward passes plus verification, and
its weights and draft KV shrank the pool 4.57M → 2.94M tokens, dropping prefix hit
94.7% → 89.5%.

**The cost, not hidden: TTFT got worse** — mean +37%, p95 +24%. The draft model runs
on prefill steps too and at conc 1 there is nothing to hide it behind. MTP is a
TPOT-for-TTFT trade, not a free win.

**To go further:** `num_speculative_tokens` 2→3 is cheap to try but acceptance decays
with depth. The real ceiling is the ~118 barriers/step — the remaining 10.3 ms lives
there, which is the collectives problem (34.3% of GPU busy), not a knob.
Raising `synthetic_acceptance_length` would hit 3× on paper, but measured acceptance
is 2.48–2.55 and inflating it would be manufacturing the number.

## SA is ahead of us by ~4–5%, and the cause is the DRAM KV offload

SA run [32968517728](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/32968517728)
uses **our own image** (`aigmkt/kimi-k3-vllm:latest`) and beats us at both ends:

| | SA | ours | Δ |
|---|---:|---:|---:|
| **c52** tok/s/GPU | **8,295.9** | 7,914.8 (T120) | **−4.8%** |
| c52 TPOT | 0.0828 | 0.0892 | +7.7% worse |
| **c1** tok/s/GPU | **980.6** | 940.0 (T106) | **−4.1%** |
| c1 TPOT | 0.02156 | 0.02243 | +4.0% worse |
| c1 TTFT p95 | 3.394 s | 5.043 s | **+48.6% worse** |

Their server log confirms **DCP is on** (`decode_context_parallel_size=8`) — the
`dcp1` in their artifact names is just the matrix label, exactly as our own
artifacts say `ep1` while the script hardcodes DCP=8. Diffing the two logs, every
material setting matches: dcp_comm_backend a2a, ROCM_AITER_MLA, max_num_seqs 80,
max_num_batched_tokens 8192, FULL_AND_PIECEWISE, speculative_config None,
`Model loading took 192.56 GiB`, GPU KV 32,756,602 tokens.

**One difference remains: `kv-offloading` — they run `none`, we run `dram`.**

T116's idle attribution independently points at the same thing. `copyBuffer`
(the DRAM offload's host↔device transfer) is the **single largest idle cause**:
**114.2 s of idle immediately before it (17.6% of all idle) and 67.3 s after**,
while contributing only 0.84% of actual GPU work across 697,367 calls.

The offload earned its place when the GPU pool was small (T92 vs T94, non-DCP,
3.3×). DCP now gives us 32.7M tokens on-device, so it has become pure host
traffic. **Next run: conc 52 with `kv-offloading: none`.**

## SOLVED: GPU idle 44.3% -> 28.2% by removing the DRAM KV offload

T116 profiled C52 **with** the offload; T124 re-profiled the same point **without**
it. Both traced identically, so rocprof's overhead cancels.

| | T116 offload ON | **T124 offload OFF** |
|---|---:|---:|
| GPU busy | 55.7% | **71.8%** |
| **Idle** | **44.3%** | **28.2%** |
| Collectives (% busy) | 34.31% | **29.44%** |
| dispatches | 6.36M | 8.47M |
| tok/s/GPU *(traced, not quotable)* | 3,146.0 | **6,821.5** |
| requests | 375/524 | **1123/1232** |

**Idle fell 16.1 points, and precisely the predicted stalls vanished:**

| gap size | T116 | T124 | change |
|---|---:|---:|---|
| 10-200 us | 115.2 s (7.9%) | 124.0 s (8.7%) | unchanged - pure launch overhead |
| 0.2-1 ms | 147.5 s (10.1%) | 104.8 s (7.3%) | -29% |
| 1-10 ms | 118.7 s (8.1%) | 57.6 s (4.0%) | -51% |
| **>10 ms** | **265.7 s (18.1%), n=4,104** | **114.6 s (8.0%), n=877** | **-57%, 4.7x fewer** |

The multi-millisecond stalls were the offload's host<->device traffic. The
sub-200 us launch gaps did not move at all, which is the expected signature:
those are dispatch overhead, not memory traffic.

**What remains, in order:** 10-200 us launch gaps 8.7% of serving (2.06M events,
a host-bound floor); >10 ms stalls 8.0% (877 events at ~130 ms each); 0.2-1 ms
per-step scheduler work 7.3%; 1-10 ms 4.0%. `copyBuffer` is still the top
idle-follower at 71.9 s (down from 114.2 s) since it is a generic HIP copy used
beyond the offload. `merge_attn_states` at 28.0 s is DCP's partial-attention
merge and is intrinsic to DCP.

## Collectives: the DCP group never gets the fast all-reduce

From the server log:

    group 'tp:0'  -> ['AITER_CUSTOM', 'PYNCCL']
    group 'dcp:0' -> ['PYNCCL']
    group 'ep:0'  -> ['PYNCCL']

Generic `ncclDevKernel_Generic_1` is **22.55%** of GPU busy against **3.63%** for
`cross_device_reduce_2stage`, the tuned AITER path TP uses. The cost is
concentrated in the group that did not get the fast backend. T116 detail, per
decode step (~290 collective launches per step on one GPU):

| %busy | calls | us/call | per step | grid | family |
|---:|---:|---:|---:|---:|---|
| 16.93 | 415,346 | 334.5 | 152.3 | 28672 | nccl |
| 3.63 | 80,682 | 369.0 | 29.6 | 40960 | cross_device_reduce (AITER custom) |
| 3.43 | 247,288 | 113.8 | 90.7 | 1792 | nccl |
| 2.19 | 47,312 | 379.0 | 17.4 | 28416 | nccl |
| ~5.9 | few | 5-9 ms | <1 | 4096-16384 | nccl (prefill) |

## Superseded: GPU idle is 44.3% (measured with the offload ON)

From T116 (1467 s serving window, 649.5 s idle across 2,724,405 gaps):

| kernel **after** the gap | idle_s | % serving | % of idle |
|---|---:|---:|---:|
| **`copyBuffer`** (DRAM KV offload) | **114.2** | **7.78%** | **17.6%** |
| `Cijk_` (dense GEMM) | 69.6 | 4.75% | 10.7% |
| `elementwise_manual_unroll` | 52.7 | 3.59% | 8.1% |
| `ncclDevKernel` | 43.5 | 2.97% | 6.7% |
| `merge_attn_states` (DCP partial-attn merge) | 41.9 | 2.86% | 6.5% |
| `fillBufferAligned` | 35.2 | 2.40% | 5.4% |

`fillBufferAligned` is only 6,069 gaps but averages **5.7 ms** each — an
allocator/memset path worth investigating separately.

---

**Best: 7,950.6 tok/s/GPU — DCP=8 + full graphs at concurrency 52 (T103).**
**148% of the SA reference (5,388)** · **64% of the 12,500 target**.
Clean full window, 0 faults, 0 aborts.
**Accuracy verified: GSM8K 98.5%** on the identical config
([T97](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/32819265466)).

**The DCP crash is fixed.** Cause was our **sparse capture ladder**: decode
batches were padded up to the next captured size, and the padded rows read out
of bounds. A dense ladder (every size 1…40, as the SA reference uses) runs
clean. DCP now beats non-DCP *and* keeps 31× the KV pool at ~94% prefix hit.

**Concurrency is where DCP pays.** Doubling to 40 gave **+57%** (4,585 → 7,206)
while KV usage stayed at **11.3%** and prefix hit held at 93.7%. Non-DCP cannot
follow — it is already at 54.3% hit on a 4.4M pool with a 243 GB/rank DRAM
crutch. **But concurrency 72 is past the knee.** T98 aborted at 1,210 s — 29/287
requests failed on timeout, TTFT 20.22 s. Not a capacity problem: **zero
preemptions, KV plateaued at 57.7%**. The cause is our `max_num_seqs = CONC × 2`
convention, which let **91 sequences decode concurrently**. Each step then
attends over ~5× the KV, decode eats the step budget, prefill starves (input
58,000 → 32,000/s), and latency blows out.

**Conc 40 is the sweet spot so far.** The ceiling is decode time per step, not
KV capacity — so the lever is `max_num_seqs`, not more offered load.

| Config | DCP | MTP | Offload | Graphs | Conc | Tok/s/GPU | TPOT | TTFT | Status | Run |
|---|---|---|---|---|---|---:|---:|---:|---|---|
| **SA c52 — ahead of us** | **8** | No | **No** | Full | **52** | **8,295.9** | **0.0828** | **4.37 s** | OK 3,629 s | [SA](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/32968517728) |
| SA reference | No | Yes | Yes | — | 20 | 5,388 | 0.0382 | 12.2 s | OK | [SA](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/31993981851/job/95282381888) |
| **T103 best** | **8** | No | **Yes** | Full | **52** | **7,950.6** | 0.0913 | 4.76 s | **OK 3,628 s** | [T103](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/32855763638) |
| T120 parity, new image | **8** | No | **Yes** | Full | 52 | 7,914.8 | 0.0892 | 4.85 s | OK 3,629 s | [T120](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/32976534358) |
| T105 | **8** | No | **Yes** | Full | 56 | 7,844.0 | 0.1041 | 7.03 s | OK 3,630 s | [T105](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/32876995670) |
| T104 | **8** | No | **Yes** | Full | 64 | 7,650.7 | 0.1264 | 8.99 s | OK 3,627 s | [T104](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/32866152287) |
| T96 | **8** | No | **Yes** | Full | **40** | **7,206.4** | 0.0722 | 3.73 s | **OK 3,627 s** | [T96](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/32812667740) |
| T102 | **8** | **Yes** | **Yes** | Full | 40 | 1,075.4 | 0.1900 | 262.6 s | OK 3,612 s | [T102](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/32846271434) |
| T98 | **8** | No | **Yes** | Full | **72** | 2,039.9 | 0.1542 | 20.22 s | **Aborted 1,210 s** | [T98](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/32821348618) |
| T95 | **8** | No | **Yes** | Full | 20 | **4,585.3** | **0.0445** | **2.24 s** | **OK 3,630 s** | [T95](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/32806967290) |
| T94 non-DCP | No | No | **Yes** | Full | 20 | 4,537.0 | 0.0454 | 2.82 s | OK 3,627 s | [T94](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/32801140212) |
| T64 | No | No | **Yes** | Full | 20 | **4,622.8** | 0.0461 | 2.14 s | **OK 3,612 s** | [T64](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/32436403856) |
| T86 | **8** | No | No | Full | 40 | 4,621.5 | 0.0736 | 4.42 s | Fault 420 s | [T86](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/32754837021) |
| T87 | **8** | No | No | Full | 40 | 4,611.6 | 0.0734 | 5.75 s | Fault 421 s | [T87](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/32758545737) |
| T74 | **8** | No | No | Full | 20 | 4,551.0 | 0.0499 | — | Fault 2,771 s | [T74](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/32699967765) |
| T81 | **8** | No | No | Full | 20 | 4,489.3 | 0.0501 | 2.18 s | Fault 2,997 s | [T81](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/32729114956) |
| T83 | **8** | No | No | Full | 40 | 4,457.1 | 0.0672 | 5.16 s | Fault 425 s | [T83](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/32745214998) |
| T77 | **8** | No | No | Full | 20 | 4,451.9 | 0.0503 | — | Fault 3,013 s | [T77](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/32708108429) |
| T79 | **8** | No | No | Full | 20 | 4,421.6 | 0.0488 | 2.25 s | Fault 2,992 s | [T79](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/32716127572) |
| T73 | **8** | No | **Yes** | Full | 20 | 4,421.2 | 0.0497 | — | Fault 3,033 s | [T73](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/32694320223) |
| T80 | **8** | No | No | Full | 20 | 4,298.4 | 0.0444 | — | Fault 2,987 s | [T80](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/32722526689) |
| T67b DP2/TP4 | No | No | Yes | Full | 20 | 2,998.6 | 0.1140 | 8.57 s | OK | [T67b](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/32458502570) |
| T82 Triton MLA | **8** | No | No | Full | 20 | 2,210.0 | 0.1508 | — | OK 3,608 s | [T82](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/32736571541) |
| T65 | No | **Yes** | Yes | Piecewise | 20 | 2,045.4 | 0.1003 | 67.7 s | OK | [T65](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/32444354043) |
| T66c | **8** | No | **Yes** | **Piecewise** | 20 | 1,574.5 | 0.2174 | 12.4 s | **OK 3,628 s** | [T66c](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/32451498395) |
| T92 | No | No | **No** | Full | 20 | 1,397.0 | 0.2332 | 35.0 s | OK 3,629 s | [T92](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/32789572478) |
| T93c | No | No | No* | Full | 20 | 1,343.0 | 0.2292 | 40.3 s | OK 3,613 s | [T93c](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/32795749613) |
| T88 | No | **Yes** | No | Full | 20 | 982.4 | 0.1659 | 212.7 s | OK 3,605 s | [T88](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/32763413965) |
| T91 Async sched | No | No | No | Full | 20 | 963.5 | 0.2498 | 137.5 s | OK 3,467 s | [T91](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/32782978418) |
| T90 | No | **Yes** | No | Full | 20 | 724.2 | 0.1374 | 374.8 s | OK 3,629 s | [T90](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/32774606161) |

\* T93c asked for offload; the script silently ignored it. See below.

---

# The five things that matter

- **DCP + full graphs crashes.** GPU page fault, 10 runs out of 10. Nothing
  else does. Full graphs alone are fine (T64). DCP alone is fine (T66c).
- **That crash is where the headroom is.** DCP holds 32.8M KV tokens vs 4.4M,
  used only 10%, 94% cache hit. T64 is 14x oversubscribed at 53% hit and needs
  a 243 GB/rank DRAM crutch. Only DCP can take more concurrency.
- **MTP loses 5x.** Cache hit falls 53% -> 10%. Not fixable by pool size or
  scheduling; both tried.
- **`--async-scheduling` loses 4.8x.** It defers block frees, pool fills to
  96%, cache evicts. The reference uses it — do not copy that flag.
- **KV offload is worth 3.3x** on non-DCP. It was silently missing from T74
  onward, so DCP-vs-non-DCP is **not yet a fair comparison**.

---

# The DCP + full graphs crash — SOLVED

**Cause:** our capture ladder was sparse (`1,2,4,8,16,24,32,40`). Every decode
batch was padded up to the next captured size, and the padded rows read out of
bounds under DCP. Fix: capture **every** size, `1…max_num_seqs`.

- Crashed 10 of 10 runs with the sparse ladder. Clean on the first dense-ladder
  run (T95, 3,630 s, 7.35M unique tokens — past every previous crash point).
- Only DCP + full graphs was affected. Full graphs alone (T64) and DCP with
  piecewise (T66c) were always fine.
- Timing scaled with concurrency, not time or tokens: conc 20 crashed ~2,900 s,
  conc 40 ~420 s. More concurrency means more distinct batch sizes, so more
  padding events. `max_num_seqs` made no difference (40 and 80 both ~421 s).
- Signature fit padding: all 8 ranks faulted at the **same low offset** over
  unrelated bases — one buffer overrun by a fixed amount on every rank.

**What confirmed it:** a SemiAnalysis run
([32746060058](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/32746060058))
ran DCP8 + full graphs for 2 hours with **zero faults** using a dense ladder
(every integer 2…60). That run used TRITON_MLA, which first suggested the
backend was the difference — the ladder turned out to be the real variable, and
AITER MLA works fine with a dense ladder while being ~2× faster
(4,585.3 vs ~2,210–2,476).

**Cost of finding it:** six runs were spent varying DCP knobs — offload, opus
rows, comm backend, query replication, `max_num_seqs`, images — while the ladder
sat unexamined. The SA reference script was available throughout and its ladder
was never diffed against ours.

---

**Images** (`vllm/vllm-openai-rocm:nightly-<sha>`)

| Tag | Date | Used by | PR #51705 |
|---|---|---|---|
| `ac7509e2` | 08-13 | T64 | — |
| `5a4c8d99` | 08-19 | T66c | — |
| `d626108b` | 08-20 | T73–T85 | 4 failed hunks |
| `f94666b6` | 08-24 | T86–T94 | 0 failed hunks |

- `ac7509e2` is where our best result was measured — three images back.
- `f94666b6` is current. Still no PR #51705 upstream, gate still present.
  Kept only because our patch applies cleanly on it.
- Image is **not** the cause of the DCP crash: T86 reproduced it on
  `f94666b6` exactly as T85 did on `d626108b`.
- T65 and T67b images were not verified from logs.

**Accuracy**

| Check | Config | Result |
|---|---|---|
| GSM8K 5-shot | T96 config (DCP=8, full graphs, AITER MLA, dense ladder 80, offload) | **98.5%** exact_match ±0.0086 |

Flexible-extract and strict-match agree exactly, so the score is not an
extraction artifact. Run: [T97](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/32819265466), `EVAL_LIMIT=200`.
Set `EVAL_ONLY=true` in the `.sh` to re-run it; `EVAL_LIMIT=full` for the whole
dataset.

**Applied upstream PRs**

| PR | What it gives us |
|---|---|
| [#51705](https://github.com/vllm-project/vllm/pull/51705) | DCP for Kimi-K3; `VLLM_ALLOW_DCP_FULL_CUDAGRAPH` hatch; softmax-LSE for AITER MLA decode. **Vendored** as `pr51705_vllm.diff` — still unmerged upstream, and DCP cannot run without it |
| [#52188](https://github.com/vllm-project/vllm/pull/52188) | Kimi-K3 DCP with DSpark (`prepare_dcp_local_seq_lens`, `cp_local_slot`). **Merged upstream, already in the image** — nothing to apply |

Everything else considered (#51040, #51171, #50619, #50791, #50883, #52269,
#48392, #51203, ROCm/aiter#4915) was either inert, conflicting or inapplicable —
see `archive/README-patches.md`.

**Applied patches** (`apply_kimi_k3_patches.sh`, 3 of them)

| Patch | What it does |
|---|---|
| `pr51705` | Vendored vLLM PR — DCP support and the full-graph hatch |
| `pr51705-rejects` | Adds `enable_dcp_q_replicate` to `MultiHeadLatentAttention.__init__`; the PR's own hunk rejects on this image |
| `kv-blockpool` | Clamps a negative block count |

- `pr51705` is **vendored in-repo** since T76. Before that it was fetched live
  and pinned by SHA — the PR was force-pushed twice in one day and killed two
  runs, hence the vendoring.
- `pr51705-rejects` fixes `enable_dcp_q_replicate` missing from
  `MultiHeadLatentAttention.__init__`. Silent with speculation off; kills every
  rank at init with MTP on. Needed on `d626108b`, not on `f94666b6`.
- All three are **pre-applied** in `aigmkt/kimi-k3-vllm:latest`, so the runtime
  patch step is commented out in the launcher.
- Archived patches and the reasoning for each removal:
  `archive/README-patches.md`.

---

# What to do next

1. ~~GSM8K accuracy gate~~ — **done, passed.** 98.5% exact_match (±0.0086),
   5-shot, flexible-extract and strict-match identical, run on the exact T96
   config (DCP=8, dense ladder 80, AITER MLA, offload). This mattered because
   the bug the dense ladder fixed was an out-of-bounds *read*, which can return
   garbage instead of crashing. It does not. T96's 7,206.4 is real.
2. **Decouple `max_num_seqs` from concurrency.** Conc 72 with `max_num_seqs`
   144 aborted (T98); the in-flight decode batch, not the pool, is the limit.
   Retry high concurrency with `max_num_seqs` held near 80.
3. ~~Re-test MTP~~ — **done, MTP is closed.** T102 ran DCP+MTP cleanly and lost
   ~85%: 1,075.4 vs 7,206.4. Dense ladder, clean draft, DCP working — no
   confounds left. Cause is memory: draft + graphs halve the KV pool
   (32.8M → 14.2M), prefix hit falls 93.7% → 28.6%, workload thrashes. DCP's
   advantage *is* KV capacity; speculation spends it.

4. **Concurrency 1 and 4 — interactivity runs.** Different axis from everything
   above: these measure **TPOT/TTFT, not throughput**, and will look poor on
   tok/s/GPU by design (~340 and ~1,300 predicted). Expect TPOT ~19–24 ms.
   This is where DCP should look best — one request's KV is sharded over all 8
   GPUs, so attention is 8-way parallel at 1/8 the KV read per rank.

Reminder: the capture ladder must be dense and must track `max_num_seqs`
exactly. Sparse ladders caused the crash.

**Concurrency sweep — complete.** Throughput peaks at 52; latency is best at 4.

| conc | n | Tok/s/GPU | TPOT | TTFT p50 | Prefix hit | Run |
|---:|---:|---:|---:|---:|---:|---|
| **1 + MTP** | 8 | **1,210.4** | **0.0103** | 1.09 s | 89.5% | [T122](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/32992548212) |
| 1 (DCP off, no offload) | 8 | 973.3 | 0.0217 | 0.93 s | 94.7% | [T121](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/32986150759) |
| 1 | 4 | 940.0 | 0.0224 | 0.96 s | 95.5% | [T106](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/32887252465) |
| **4** | 6 | 1,525.1 | 0.0256 | **0.87 s** | — | [T107](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/32895236167) |
| 40 | 43 | 7,206.4 | 0.0722 | — | 93.7% | [T96](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/32812667740) |
| **52** | 54 | **7,950.6** | 0.0913 | — | 93.7% | [T103](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/32855763638) |
| 56 | 66 | 7,844.0 | 0.1041 | — | 92.7% | [T105](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/32876995670) |
| 64 | 80 | 7,650.7 | 0.1264 | — | 90.0% | [T104](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/32866152287) |

- **Best throughput: 7,950.6 tok/s/GPU @ conc 52** — 148% of the SA reference,
  64% of target, GSM8K 98.5%.
- **Best interactivity: conc 4** — TPOT 25.6 ms, TTFT p50 0.87 s. Beats conc 1 on
  TTFT *and* throughput; conc 1's `max_num_seqs 4` was too tight and queued the
  replay's branched sub-requests.
- `max_num_seqs` needs **branching headroom** (~1.25× conc). Too tight raises
  TTFT (T106); too loose lets residents run away and abort (T98, `mns` 144 → 91
  residents → TTFT 20.2 s).

**Throughput peaks at concurrency 52 — it does not keep climbing.**

| conc | n | tok/s/GPU | TPOT | TTFT | prefix hit |
|---:|---:|---:|---:|---:|---:|
| 40 | 43 | 7,206.4 | 0.0722 | 3.73 s | 93.7% |
| **52** | 54 | **7,950.6** | 0.0913 | 4.76 s | 93.7% |
| 56 | 66 | 7,844.0 | 0.1041 | 7.03 s | 92.7% |
| 64 | 80 | 7,650.7 | 0.1264 | 8.99 s | 90.0% |

Flat-topped between 52 and 56 (1.3% apart), then falls. Cause is prefix-hit
erosion (93.7% → 90.0%) as concurrent contexts multiply, plus TPOT rising faster
than resident count.

**The model below is superseded and kept only for its TPOT/TTFT fits.** Its
throughput term `T(n)=16,823·n/(n+48)` is monotonic and predicted conc 64 would
beat 52 by ~20%; it fell 3.8% instead. The earlier "~10,000–10,500 ceiling"
followed from that form and is **wrong** — measured peak is ~7,950.

**Scaling model** (fitted on T95 n=18, T96 n=36, T98 n=91; `n` = resident requests)

```
TPOT(n) = 0.01745 + 0.001503*n     s     (n=36 predicted 0.0716 vs 0.0722 measured)
TTFT(n) = -7.06   + 0.300*n        s
T(n)    = 16,823 * n/(n + 48)      tok/s/GPU
```

- Best realistic landing **~10,000–10,500 tok/s/GPU** at conc ~64–80 — 80–84% of target.
- **Latency binds first, not cache.** Throughput asymptote is 16,823, above
  target, but 12,500 needs n≈145 → TTFT ~36 s. T98 aborted at 20.2 s.
- **KV offload adds capacity DCP cannot use**: 11.3% pool usage at conc 40 and
  `ext_cache_hit` 0.0–0.6%. It was worth 3.3× for *non-DCP* only.
- Passing 12,500 needs the **0.0015 s/request decode slope** cut — a faster MLA
  decode kernel or shorter effective context. Not more concurrency, not more cache.

# Known traps

| Trap | Cost |
|---|---|
| **Sparse capture ladder under DCP** | **GPU page fault** — use every size 1…N |
| `--async-scheduling` on | 4.8x |
| KV offload block missing from the `.sh` | 3.3x |
| Capture ladder below `max_num_seqs x (1+spec)` | Decode silently drops to piecewise |
| `spec-decoding` matrix field | Also picks the script filename |
| `per_gpu.total_tput_tps` | Divides by `tp`, not GPU count — 2x wrong on DP |
| `agg_bmk.json` `spec_decoding` | Reports the matrix label, not engine state |
| Workflow "completed success" | Says nothing about whether the benchmark aborted |

---

# Appendix — detail, evidence and history

Everything below is the working record: mechanisms, profiling, falsified
hypotheses, corrections and the full trial ledger.

## 2. Findings, in priority order

### 2.0 DCP's penalty was a ROCm-only cudagraph downgrade, not DCP itself

For most of this investigation DCP looked structurally slow — roughly a third of
plain TP8. It was not. `vllm/platforms/rocm.py` silently downgrades the
cudagraph mode whenever DCP is on:

```python
if compilation_config.cudagraph_mode.has_full_cudagraphs():
    # decode context parallel does not support full cudagraphs
    if parallel_config.decode_context_parallel_size > 1:
        compilation_config.cudagraph_mode = CUDAGraphMode.PIECEWISE
```

This gate exists **only on the ROCm path** — `cuda.py` has zero occurrences, so
NVIDIA DCP keeps full graphs. Piecewise splits the graph at every attention op
and hands each decode step back to the host to relaunch. Host-launch latency was
already the profiled bottleneck (3.4), so this downgrade lands directly on the
limiter, and every DCP number recorded before this point was measuring the
downgrade rather than DCP.

Lifting it (vLLM PR #51705, which gates the same code on
`VLLM_ALLOW_DCP_FULL_CUDAGRAPH`):

| | piecewise | full graphs | |
|---|---:|---:|---|
| tok/s/GPU | 1,574.5 | 4,551.0 | +189% |
| TPOT | 0.2174 | 0.0499 | −77% |
| prefix cache hit | 71.0% | 94.3% | |

DCP now sits within 2% of the best non-DCP run while carrying **31× the KV
capacity** (32.8M vs 4.4M tokens). That combination — reference-class TPOT at
32× cache — is the only configuration on this page with a credible path past the
reference, because it is the only one where concurrency can rise without
evicting. KV usage at concurrency 20 was 10.5% with zero preemptions.

**Blocker.** Both full-graph runs die on a GPU page fault before the window
closes (2,771 s and 3,033 s). Until that closes these are not results. See 2.0b.

### 2.0b The full-graph page fault — evidence and open question

All eight ranks fault simultaneously with `VM_L2_PROTECTION_FAULT_STATUS`,
`PERMISSION_FAULTS: 0x3`, IH client `0x1b (UTCL2)`, TCP client — a
vector-memory out-of-bounds **read**.

What the addresses say: the eight faulting addresses share the same low offset
(six at `...ce000`, two at `...cc000`) across completely unrelated bases. That is
one logical per-rank buffer overrun by a *fixed* amount on every rank. A KV or
block-table walk running off the end would scatter across ranks instead.

**The trigger is DCP, not full graphs.** T64 ran non-DCP with the *same*
`FULL_AND_PIECEWISE` mode and completed 3,612 s cleanly. So full graphs alone
are safe; DCP + full graphs is the failing combination.

**Every DCP configuration knob has now been falsified.** Six runs, one variable
each, all landing in the same band:

| Run | variable moved | duration | unique tokens | tok/s/GPU |
|---|---|---:|---:|---:|
| T73 | KV offload on (no opus patch) | 3,033 s | 6.36M | 4,421.2 |
| T74 | offload off | 2,771 s | 6.06M | 4,551.0 |
| T77 | opus rows present | 3,013 s | 6.51M | 4,451.9 |
| T79 | capture ladder 40 | 2,992 s | 6.34M | 4,421.6 |
| T80 | `ag_rs` comm backend | 2,987 s | 6.42M | 4,298.4 |
| T81 | `VLLM_DCP_Q_REPLICATE=0` | 2,997 s | 6.35M | 4,489.3 |

Falsified: opus rows, KV offload, capture ladder, comm backend, query
replication. The fault is invariant to configuration, so it lives in the DCP +
full-graph decode path itself, not in how we drive it.

Two corrections to earlier reasoning on this page, both mine:

- The capture-ladder theory was wrong *on its own terms*. `FULL_AND_PIECEWISE`
  already routes mixed prefill+decode batches to piecewise and reserves full
  graphs for uniform decode batches, so the 48/64 slots were never the
  mixed-batch path claimed.
- T80 was not the clean single-variable test it was described as. PR #51705
  redefines `VLLM_DCP_Q_REPLICATE` as *auto — on when `dcp_comm_backend=="a2a"`*,
  and T80 ran `ag_rs` while still pinning it on. `ag_rs` was tested in an
  off-design pairing and is not cleanly refuted (it was also ~3% slower).

**The trigger scales with concurrency, not with elapsed time or accumulated
work.** Two results pin this down, and the second corrects the first:

- T82 ran **3,607.6 s** clean — longer than every faulting run — while reaching
  only 4.12M unique tokens. So it is not elapsed time, and a run can look clean
  purely by not doing enough work.
- T83 raised concurrency 20 → 40 and faulted at **424.5 s** with only
  **1,629,728** unique tokens. So it is not accumulated work either.

| | concurrency 20 (6 runs) | concurrency 40 (T83) |
|---|---|---|
| fault at | ~2,900 s | **424.5 s** |
| unique tokens | ~6.3M | **1.63M** |
| kv_usage | ~11% | 25.9% |
| requests completed | ~924–958 | 243 |

Concurrency brings the fault **7× sooner** while the request rate rose only
~1.8×. The tight ~6.3M clustering across the six earlier runs was an artifact of
holding concurrency fixed at 20, **not** a property of the bug — anything built
on that correlation should be discarded. The evidence now points at something
scaling with the number of *concurrent* sequences.

Practical consequence: the fault reproduces in **~7 minutes** at concurrency 40
instead of ~50, which makes previously unaffordable diagnostics (kernel
serialization, component bisects) cheap.

**Piecewise safety is real, not an artifact.** T66c reached **14,317,525**
unique tokens clean, well past the ~6.3M trigger, so it was not merely too slow
to get there.

**Conclusion: this is an upstream bug, not a misconfiguration.** The fault is
invariant to everything available to us:

| Dimension varied | Result |
|---|---|
| KV offload on/off | faults |
| aiter opus rows present/removed | faults |
| capture ladder 64 / 40 / 80 | faults |
| comm backend `a2a` / `ag_rs` | faults |
| `VLLM_DCP_Q_REPLICATE` 1 / 0 | faults |
| decode backend AITER / TRITON | inconclusive (T82/T84) |
| concurrency 20 / 40 | faults, 7× sooner at 40 |
| `max_num_seqs` 40 / 80 | faults, **no effect on timing** |
| image `d626108b` / `f94666b6` | faults identically |
| `AMD_SERIALIZE_KERNEL=3` | faults, no attribution |

**It scales with in-flight decode batch size, not with buffer sizing.** T87 held
`max_num_seqs=40` while running concurrency 40 and faulted at 420.6 s — the same
as `max_num_seqs=80`. That eliminates the buffer-sizing hypotheses outright,
since `paged_kv_indices` (`max_num_seqs × max_model_len`) and the aiter MLA
metadata workspace (`max_num_reqs = max_num_seqs`) are both sized by the
quantity held constant.

Corroborating: `VLLM_ALLOW_DCP_FULL_CUDAGRAPH` defaults to **`False`** upstream —
the PR author opted not to enable this combination by default.

**Reproduction for an upstream report:** concurrency 40, DCP=8,
`FULL_AND_PIECEWISE`, `VLLM_ALLOW_DCP_FULL_CUDAGRAPH=1` → GPU page fault on all
8 ranks in **~420 s**, 10 of 10 runs. All ranks fault at one shared low offset
over unrelated bases. No kernel attribution available: `Reason: Unknown`, and
the GPU coredump handler fails with `execvp` ENOENT.

**The image is the one variable never changed.** Every run T64–T83 used
`nightly-d626108b` (2026-08-20). Checked against the newest nightly
`f94666b6` (2026-08-24, 139 commits later): PR #51705 is **still not merged**
upstream and the ROCm DCP→PIECEWISE gate is **still present**, so a newer image
does not fix this by itself. It is a better base for one concrete reason —
the vendored diff applies with **0 failed hunks** there versus **4** on
`d626108b`, which removes the silent `enable_dcp_q_replicate` reject that killed
T78. **Now confirmed end-to-end (T86): the newer image faults identically**
(419.9 s), and it genuinely ran the new build (`dev1133+gf94666b60` vs
`dev994+gd626108b1`), so this is not a cache artifact. We stay on it for the
cleaner patch application, not for any fix.

**The aiter opus-rows patch is exonerated**, on both fault and performance:

| Run | opus rows | offload | duration | tok/s/GPU | outcome |
|---|---|---|---:|---:|---|
| T73 | present | yes | 3,033 s | 4,421.2 | fault |
| T74 | removed | no | 2,771 s | 4,551.0 | fault |
| T77 | **present** | no | 3,013 s | 4,451.9 | fault |

With opus rows left fully intact the run still faults, at the same point, with
the same throughput. An earlier reading here — that untuned-GEMM fallbacks rose
~480× (88 → 42,320) because of the opus patch — was **wrong**. T77 logged
**46,056** of those messages *with* opus rows present. That gap was piecewise vs
full graphs changing batch shapes, not opus. Adjacency of untuned-GEMM lines to
the fault was never evidence either: they run 42,320 deep from line 202 onward.

What the fault correlates with:

| Run | unique tokens at fault | kv_usage | prefix hit |
|---|---:|---:|---:|
| T74 | 6,063,765 | 11.8% | 94.2% |
| T77 | 6,508,095 | 11.8% | 94.1% |
| T73 | 6,359,477 | 10.5% | 94.3% |
| **T64 (non-DCP, clean)** | **62,697,249** | 41.1% | 53.4% |

T64 ran ten times past the DCP fault point without incident, so the unique-token
count is not itself the trigger. Note that at fault time the DCP runs have
**never evicted anything** — 94% prefix hit against a 31× pool, so sequences
grow monotonically. That fits a per-request length limit frozen into a captured
graph, which only DCP's pool is large enough to let sequences reach.

Ruled out by inspection: `paged_kv_indices`, the persistent page-index buffer,
is sized `max_num_seqs × max_model_len` (167 MB) against at most ~8.4M entries
in use.

Demoted: `min_kv_seq_len` frozen into the captured graph at
`rocm_aiter_mla.py:1268`, where Gluon derives `NUM_KV_SPLITS` from it as launch
geometry. A stale split count is hazardous when replay sequences are *shorter*
than capture, but these faults arrive only after context *growth*.

Closest upstream analogue: vLLM **#50791**, which sizes the FlashInfer sparse
MLA workspace for DCP. Not applicable to us (B200/FlashInfer), but it documents
this exact bug class — DCP forces LSE during decode, the LSE slab is carved from
a DCP-unaware workspace, and every DCP worker dies at once. The ROCm-side
equivalent is the place to look next.

**DCP is parked meanwhile.** At concurrency 20 it does not win anyway
(4,451–4,551 vs non-DCP 4,622.8); its value is capacity headroom for higher
concurrency, which is worth nothing until a run survives the window.

### 2.1 The decode attention backend is worth 2.4×
Forcing `--attention-backend TRITON_MLA` on the target model costs **72% of
throughput and 60% of TPOT** versus leaving it unset and letting ROCm select
`ROCM_AITER_MLA`.

| | forced TRITON_MLA | AITER MLA | Δ |
|---|---:|---:|---|
| tok/s/GPU | 2,685.0 | **4,622.8** | **+72.2%** |
| TPOT mean | 0.1161 | **0.0461** | **−60.3%** |
| TTFT mean | 3.450 s | **2.143 s** | −37.9% |
| requests OK | 569/611 | 1,046/1,090 | +84% |

Single variable: same image, patches, KV pool (byte-identical 4,376,929 tokens),
concurrency and spec setting. The reference sets **no** target backend; the
`TRITON_MLA` string in its config belongs to `--speculative-config`, i.e. the
**draft** model. AMD documents AITER MLA at 1.2–1.6× TRITON_MLA; measured here
it is 2.4× on this workload.

### 2.2 All three parallelism/speculation features lose against plain TP8

| Feature | Δ vs TP8 | Mechanism |
|---|---:|---|
| **DCP=8** | **−66%** | Collective-bound. Delivered its full capacity promise — 31.26× KV, 71.0% prefix hit, ≤15% utilised, `ext_cache_hit` 0.0% — and still lost 3×. |
| **MTP** | **−56%** | Accelerates decode, which is **0.6% of scored tokens**, while consuming 36% of the KV pool serving the other 99.4%. TTFT 2.14 s → 67.7 s. |
| **DP attention** | **−35%** | Best of the three. TPOT p50 0.0652 vs mean 0.1140 — a spread no other configuration shows. |

### 2.3 KV capacity is not the constraint
DCP=8 supplies 7.5× the KV pool (32,779,397 vs 4,376,929 tokens), raises prefix
hit from ~52% to 71.0%, never exceeds 15% utilisation and never spills to the
DRAM tier — and is still 3× slower. **99.4% of scored tokens are prefill**
(input 21,360 tok/s vs output 119), and prefill throughput is set by compute and
synchronisation, not by spare decode KV.

### 2.4 The workload is prefill-dominated
Any optimisation targeting decode addresses 0.6% of the scored tokens. This
single fact explains why MTP loses, why DCP's decode-side capacity is worthless
here, and why TTFT regressions matter more than TPOT regressions.

### 2.5 DCP + MTP — no longer blocked upstream (finding below is superseded)

**Superseded as of the current image.** vLLM **#52188 "[Spec decode] Support
Kimi-K3 DCP with DSpark"** merged 2026-08-17 and is present in
`nightly-d626108b`. It landed `prepare_dcp_local_seq_lens` and `cp_local_slot`
in `cp_utils.py` — including at `spec_decode/dflash/cudagraph.py:45`, which is
precisely the "local sequence lengths are not advanced between draft steps"
gap described below. Together with vendored PR #51705, which supplies
`supports_non_causal_multi_token_dcp` on Triton MLA and `supports_dcp_with_varlen`
on AITER MLA, both capability flags the validator looks for are now satisfied.

Related, still open upstream: **#52269** (Kimi-K3 DSpark under DCP, draft — not
worth vendoring yet) and **#48392** (dense GQA/MHA drafts; Kimi-K3 is MLA, so
#52188 already covers it).

Untested here so far, because DCP + full graphs still faults (2.0b). The
original analysis is kept below for the record:

`mla_attention.py::_validate_dspark_dcp_support` requires one of two capability
flags. Across the entire image:
```
supports_non_causal_multi_token_dcp = True  ->  tokenspeed_mla.py  (only)
supports_dcp_with_varlen            = True  ->  flashinfer_mla, flashattn_mla,
                                                flashinfer_mla_sparse, flashmla_sparse
```
Neither `triton_mla.py` nor `rocm_aiter_mla.py` declares either. `TOKENSPEED_MLA`
is absent from `platforms/rocm.py` and not installed. `triton_mla.py` also
disables its draft path under DCP in code:
```python
# DCP local sequence lengths are not advanced between draft steps.
self.supports_draft_decode_metadata_update = self.dcp_world_size == 1
def update_draft_decode_metadata(self, _metadata): pass   # no-op
```
Setting the flag alone would let drafting proceed on stale per-rank KV ranges —
wrong output, no crash. This is an implementation gap, not a configuration gap.

### 2.6 The reference's advantage is speculation, not parallelism
The MI355X reference runs **no DCP and no EP** — plain TP8 + MTP. Its command
carries no `--decode-context-parallel-size`, no `--max-num-batched-tokens` and no
`--attention-backend`. Its 5,388 is measured with `rejection_sample_method:
synthetic` at `synthetic_acceptance_length: 2.51`.

### 2.7 Chunked-prefill sizing controls the KV pool
`--max-num-batched-tokens` feeds
`mla_attention.py::determine_chunked_prefill_workspace_size`. Pinning it reserves
memory the KV pool cannot then have. With MTP enabled, omitting the flag moved
the pool from 1,385,293 → 2,646,059 tokens (1.32× → 2.52×), exactly matching the
reference. It is a memory knob, not only a scheduling knob.

### 2.8 DP attention is memory-feasible at intermediate layouts
Measured from the checkpoint: experts 1,446.46 GB (92.67%), non-expert 114.40 GB
(7.33%). With EP sharding experts across all ranks, non-expert weights shard
within the TP group and replicate only across DP:

| Layout | per-GPU | fits (259 GB budget) |
|---|---:|---|
| DP1/TP8 | 195.1 GB | yes — current |
| **DP2/TP4/EP8** | **209.4 GB** | yes |
| DP4/TP2/EP8 | 238.0 GB | yes |
| DP8/TP1/EP8 | 295.2 GB | **no** — exceeds the 288 GB card |

Only the TP1 extreme fails. Any DP layout requires `ep > 1` or the experts do not
shard.

---

## 3. Limiting kernels and where the time goes

Torch profiler, decode-only windows, 8 ranks. **Caveat: all existing traces were
captured with the forced TRITON_MLA backend** — the 2.4× handicap. Traces of the
corrected stack are in progress.

### 3.1 Whole-trace breakdown (rank 0, 15 s decode window)

| | DCP=8 | non-DCP |
|---|---:|---:|
| Total GPU kernel time | 14.801 s | 5.033 s |
| Kernels launched | 121,030 | 235,224 |
| `aiter::cross_device_reduce_2stage<bf16,8,false>` | **78.37%** / 10,602 calls | 55.48% / 22,599 |
| `ncclDevKernel_Generic_1` | 8.22% / 950 | — |
| `mscclKernel_Sum_hip_bfloat16_Simple` | 6.06% / 912 | — |
| `mfma_moe1_silu_mul_afp8_wfp4` | 1.20% / 3,496 | 8.37% / 7,452 |
| `mfma_moe2_afp8_wfp4_bf16` | 0.71% / 3,496 | 4.96% / 7,452 |
| `_gemm_a16_w16` | — | 3.24% / 7,533 |
| `_attn_res_kernel` | — | 2.34% / 15,147 |
| **collectives / compute** | **92.65% / ~7%** | 56.05% / ~44% |

### 3.2 The decisive measurement

```
aiter::cross_device_reduce_2stage   SAME kernel, SAME 287 KB payload
  non-DCP :     8 us per call  (p50, every rank)
  DCP=8   : 1,094 us per call  (11.599 s / 10,602 calls)
  ratio   : 137x
```

Per-step kernel density is unchanged (DCP 1,345 vs non-DCP 1,232 kernels/step)
and compute per step is identical (11.6 vs 11.5 ms). **The kernel did not become
slower; it was made to wait.**

Per decode step, ~118 all-reduces in **both** arms:

| | all-reduce | compute | steps/15 s | TPOT |
|---|---:|---:|---:|---:|
| non-DCP | 14.6 ms | 11.5 ms | ~191 | 43 ms |
| DCP=8 | **129 ms** | 11.6 ms | ~90 | 167 ms |

### 3.3 Root cause on the profiled stack: host-launch starvation

An all-reduce returns when the **last** rank arrives, so the rank spending the
least time inside it arrives late.

```
DCP=8 all-reduce seconds per rank
  r0 11.72  r1 11.71  r2 10.13  r3 9.32  r4 10.59  r5 11.69  r6 8.47  r7 2.07  <- straggler
  p50 ~1000 us on 7 of 8 ranks -> waiting on EVERY call, not a tail
  non-DCP for contrast: p50 8-9 us on every rank
```
```
Host timeline over the same 15.20 s window
  straggler  gpu_busy  5.15 s  idle 10.05 s (66%)  123,900 launch gaps
             gap p50 62 us  p90 190 us  max 1.2 ms
  normal     gpu_busy 14.86 s  idle  0.35 s ( 2%)  116,678 gaps
             gap p50 0 us  p90 0 us
  host op totals IDENTICAL: 96,452 ms vs 97,114 ms
```
The straggler's GPU waits ~62 µs after each kernel for the host to launch the
next, ~124k times ≈ 10 s idle. Host *work* is identical between ranks, so this is
launch **delay**, not extra work. The other seven block inside the all-reduce,
which is why the cost is *attributed* to collective time.

### 3.4 Limiters, ranked

| # | Limiter | Evidence |
|---|---|---|
| 1 | **~118 global barriers per decode step** | Identical count in both arms. Each all-reduce is a hard sync across 8 ranks, so per-rank jitter is amplified 118× per step — this converts 62 µs of host delay into 129 ms/step. |
| 2 | **Host launch latency on the critical path** | Straggler idle 66%, ~124k gaps, identical host work, `max_concurrent_batches=1` leaves nothing to overlap. |
| 3 | **DCP adds synchronisation while sharding no work** | compute/step 11.5 → 11.6 ms. Only 24 of 93 layers are full-attn MLA, so there is little to shard. |
| 4 | **Workload shape** | 99.4% prefill; decode-side capacity is not the constraint. |

### 3.5 Which kernels would need acceleration

**No kernel needs to be made faster.** The dominant 78.37% is *blocking time*
charged to a kernel that completes in 8 µs when not waiting. Substituting the
implementation (PYNCCL for AITER custom) produced **−23% throughput, +31% TPOT** —
replacing the kernel makes it worse.

Addressable work, by measured share and tractability:

| Target | Share / status | Notes |
|---|---|---|
| **Fusion to cut barrier and launch count** | attacks limiters #1 and #2 | NVIDIA ships `VLLM_ENABLE_K3_LATENT_MOE_TAIL_FUSION`; **absent from the ROCm build**. The only lever that hits both limiters. |
| **DCP's own collectives** | `ncclDevKernel_Generic_1` 8.22% + `mscclKernel_Sum` 6.06% ≈ **14.3%** | Genuine work (gather + LSE merge), but a minority — perfect execution caps the win near 14%. |
| `cp_lse_ag_out_rs` direct path | ported, **−0.9%** | `direct_dcp_a2a_lse_reduce` hand-ported to HIP. Functional, no gain. |
| `q_gather` / `kv_gather` direct path | **not portable** | `multimem.st.*` PTX (NVLink hardware multicast, `__CUDA_ARCH__ >= 900`). No AMD equivalent. |
| Query all-gather elimination | `VLLM_DCP_Q_REPLICATE` | Requires `DCPGroupColumnParallelLinear`; Kimi-K3 builds plain `ColumnParallelLinear` (`models/kimi_k3/amd/linear.py:376,384`). Addresses part of the 14.3%. |
| Remove host from the launch path | untested | Full-step graph capture. Async scheduling is the cheap version and did not help. |

**Summary:** the cost is *serialisation*, not arithmetic. Kernel-level work can
address at most the ~14.3% that DCP's own collectives occupy. The other 78% is
seven GPUs idling in a barrier waiting on one host thread, fixable only by
removing barriers or removing the host from the launch path.

---

## 4. Recommended direction

1. **Keep the target attention backend unset.** Largest single effect measured.
2. **Do not enable DCP or MTP** on this workload at c20 with this request mix.
3. **Pursue fusion** — `VLLM_ENABLE_K3_LATENT_MOE_TAIL_FUSION` has no ROCm
   equivalent, and barrier/launch count is the binding constraint.
4. **Investigate DP attention's latency tail** — TPOT p50 0.0652 against mean
   0.1140 suggests the all-reduce removal works and DP-group imbalance costs the
   difference. Load balancing across DP groups is untested.
5. **Upstream request, precisely scoped:** implement
   `update_draft_decode_metadata` for a ROCm MLA backend so DCP local sequence
   lengths advance between draft steps, then declare
   `supports_draft_decode_metadata_update` and
   `supports_dcp_with_varlen` / `supports_non_causal_multi_token_dcp`.

---

## 5. Measurement hazards in this harness

Two reported fields produce incorrect conclusions if taken at face value.

| Field | Behaviour | Impact |
|---|---|---|
| `agg_bmk.json` → `spec_decoding` | Reports the **matrix label**, not engine state. Read `'none'` while the engine logged `speculative_config=SpeculativeConfig` and 364 SpecDecoding samples. | Caused a run to be recorded as speculative when it was not, and a non-existent "2.7× decode regression" (a spec-on vs spec-off comparison). |
| `agg_bmk.json` → `per_gpu.total_tput_tps`, and the job footer `Throughput per GPU` | Divides TOTAL by **`tp`**, not the GPU count. Correct whenever `tp` == GPUs; wrong by `8/tp` for any DP layout. | DP2/TP4 reported **5,997.2 tok/s/GPU** — apparently beating the reference — against an actual **2,998.6**. Confirmed from the same run's `world_size=8`, `rank=0..7`, `local_rank=0..7`, `EngineCore_DP0/DP1`. |

**Any DP or DP+EP result published from this harness is overstated 2× unless the
divisor is corrected.** Aggregates should be reconciled against the engine log.

A third hazard is reproducibility rather than measurement. vLLM PR #51705 was
fetched at runtime and pinned by sha256; the PR was then force-pushed **twice in
one day**, and each time the pin correctly refused to apply — which silently
removed the DCP patches and killed the run at engine init with `Decode Context
Parallelism (DCP) requires attention implementations to return the softmax LSE
during decode`. The pin behaved exactly as designed, but a hard pin against an
actively developed PR guarantees repeated breakage. The diff is now **vendored
in-repo** (`pr51705_vllm.diff`, `vllm/`-only), so the exact bytes that produced a
result sit next to the script that applies them, with no network fetch. Any
comparison spanning that change must confirm which revision was actually in the
image.

---

## 6. Configuration reference

### 6.1 In-container patches

| # | Name | Purpose |
|---|---|---|
| [1] | aiter pybind11 | Standalone pybind11 (internals v12) outranks torch's bundled copy (v11) via `-I` vs `-isystem`. Separate type registry per internals id → JIT module cannot see `aiter_tensor_t` → `TypeError` at warmup. Unblocks `ROCM_AITER_FA` prefill. |
| [2] | TritonMLA cudagraph | Raises `_cudagraph_support` to `UNIFORM_BATCH` so the DSpark drafter keeps FULL cudagraphs. Measured 14.05 → 77.65 tok/s single-stream (5.52×). **Incompatible with speculation on current nightlies** — it permits full capture, which then trips `assert m.max_query_len <= reorder_batch_threshold`. Skipped when spec is on. |
| [3] | KV block-pool clamp | `allocate_external_computed_blocks()` can pass a negative count to `get_new_blocks`, which is silently destructive (`num_free_blocks -= n` increases it; `range(n)` iterates zero times). A later pop walks past the tail. Load-dependent: c10 died at 3612 s, c12 at 487 s, c16 at 354 s. Requires `--kv-transfer-config`. |
| [4] | DCP-LSE plumbing | Plumbs softmax LSE + round-robin CP through AITER MLA decode. aiter's LSE is natural-log with sm_scale folded, `[B,H]` fp32. Superseded by [5]. |
| [5] | vLLM PR #51705 | Upstream DCP for Kimi-K3 DSpark, **and** the `VLLM_ALLOW_DCP_FULL_CUDAGRAPH` gate that unlocks 2.0. Now **vendored** in-repo as `pr51705_vllm.diff` and applied from disk. Does **not** fix the 0x1016 fault. |
| [6] | DCP block-table sizing | **The 0x1016 fix.** Tables sized `cdiv(max_model_len, block_size × dcp_size)` but indexed with the undivided `max_model_len`. Boundary tracks the ratio exactly: DCP8/bf16 faulted at 134,400; DCP4/bf16 at 262,656; DCP8/fp8 at 135,168 — block **count**, not bytes. |
| [7] | direct DCP a2a (ROCm port) | vLLM compiles the kernel only under `VLLM_GPU_LANG == CUDA`. The a2a combine ported to HIP (`st/ld.global.{release,acquire}.sys.u32` → `__hip_atomic_*` at SYSTEM scope). Gather kernels use `multimem.st.*`, not ported. **−0.9%.** |
| [8] | DCP gathered-head sizing | Supplies PR #51705's failing hunk (`_decode_num_heads = num_heads × dcp_world_size`). Requires image `5a4c8d99`; on `ac7509e2b` the builder predates the plumbing and lacks `dcp_world_size`. |

Patches [5], [6] and [8] are DCP-only and are skipped on non-DCP arms.

### 6.2 Layout constraints

```
TP × DP must equal 8 (the GPU count)
DP > 1 requires EP > 1, or the experts do not shard
tp_size % dcp_size == 0        -> at TP=4, legal DCP is 1, 2, 4
weights/GPU = non_expert/TP + experts/(EP ? ranks : TP)   must be < 288 GB
DCP patches require image 5a4c8d99; ac7509e2b lacks PR #51705 plumbing
```

---

## 7. Trial ledger

| # | Run | Configuration | Result |
|---|---|---|---|
| 138 | 33158510878 | fixed-len 8k, C1, k=6, ladder 56 | ITL 32.70, TPOT 8.70 — 10 requests, percentiles unusable |
| 139 | 33159693747 | k=8, ladder 72 | ITL 33.63, TPOT 8.42 |
| 140 | 33160495127 | k=8, ladder 9 | TPOT 8.37; capture 65s→26s, 1.46→0.83 GiB |
| 141 | — | (superseded) | |
| 142 | 33164549184 | 1000 prompts @ 8k | CANCELLED — wrong ISL for a DCP screen |
| 143 | 33165123199 | **122k**, 100 req, DCP=8, draft bf16 | TPOT 11.93 / p90 13.70 |
| 144 | 33166811023 | DCP=8, draft **fp8** | TPOT 11.97 / p90 13.71; KV pool +36.5% |
| 145 | 33168461313 | **DCP off**, draft fp8 | **TPOT 8.77 / p90 9.06** — DCP costs +36.5% |
| 146 | 33169840000 | DCP=1 + a2a + interleave | 8.79 / 9.07 — flags **inert** at size 1 |
| 147 | 33171360827 | **nightly 6f7df92a8e** + 1-line cg patch | **TPOT 7.57 / p90 7.91** — nightly worth −13.7% |
| 148 | 33183801155 | nightly + load-format auto | 7.57 / 7.93 — load-format **inert** |
| 149 | 33185690716 | (ladder cap 16) | CANCELLED to run mns=CONC first |
| 150 | 33185946417 | nightly, **mns=1 ladder 1..9** | 7.58 / 7.93; capture 44s→**7s**, KV +2.3% |
| 151 | 33187965065 | aigmkt, mns 8, ladder 1..16 | 7.93 / 8.20 — this is `sa.sh`'s C1 config |
| 152 | 33190157834 | nightly + #51705 (bad rebase) | FAIL — `MultiHeadLatentAttention.__init__() got an unexpected keyword argument 'enable_dcp_q_replicate'`; I wrongly discarded `kimi_k3/nvidia/mla.py` as "NVIDIA-only" |
| 153 | 33191059734 | nightly + #51705 (fixed rebase) | cancelled for the accuracy gate |
| 154a | 33191753746 | **GSM8K C52**, limit 200, nightly + rebased #51705, DCP=8 | **99.0%** strict + flexible (±0.71). Gate PASSES — rebase sound on the DCP path. Job red only on the expected "Benchmark result not found" (eval-only makes no benchmark JSON; the runner cannot skip that check for agentic scenarios) |
| 154b | 33191753746 | **GSM8K C1**, limit 200, DCP off, **MTP k=8**, synthetic AL 4.00 | **0.14 / 0.13** — vs **0.99** on C52 (k=0) on the identical stack |
| 155 | 33197400117 | GSM8K C1 k=0 control | CANCELLED — superseded; C52's 0.99 already clears the rebase, and C1 accuracy is not a gate under `synthetic` |
| 156a | 33197613253 | **C52 PERF, nightly 6f7df92a8e + rebased #51705**, DCP=8, mns 80, dram, load auto | **7,906 tok/s/GPU**, 1879/1989 requests, KV 31,924,580, input 62,665 tok/s, 3629.7 s |
| 156b | 33197613253 | **C1 PERF**, nightly + rebased #51705, DCP off, k=8, mns 8, ladder 1..16 | **1,509 tok/s/GPU**, ITL p50 **7.89** ms (err-adj 8.07), intvty p50 126. **CAVEAT: 17/148 error-dropped (11.5%) vs T123's 1.6%**, only 131 served in 1984 s vs T123's 190 in 3609 s — not a clean comparison |
| 157 | 33209544438 | C52, **gmu 0.95** | **HARD FAIL — 0 successful / 57, all error-dropped.** Server started, KV pool grew 31.9M -> **40.2M tokens (+26%)**, then all 55 warmup requests HUNG: 0 returned, 0 errors, 1200 s. Not an OOM — the engine accepted work and never produced output. Reverted to 0.9 |
| 158 | 33212429374 | C52, **NCCL_MIN_NCHANNELS=32** | **7,656 tok/s/GPU — a 3.2% LOSS** vs T156's 7,906 on the identical config. 1801/1911 requests, KV 31,924,580 (unchanged), input 60,576 tok/s. Reverted |
| 158b | 33212429374 | C1, NCCL_MIN_NCHANNELS=32 | 1,515 tok/s/GPU, ITL p50 **8.06** ms vs T156's 7.89 — RCCL neutral-to-slightly-worse at C1 too. **17/148 error-dropped again, identical to T156** |
| 159a | 33222609872 | **CCD pinning C1** | **`[pin-ccd] pinned 0 threads` — the pin did NOT apply.** 1,511 tok/s/GPU, 17/148 dropped (3rd identical) — i.e. just a repeat of T158. Cause found: the pgrep pattern required a trailing underscore |
| 159b | 33222609872 | CCD pinning C52 | **FAILED — my own bug.** Pin DID apply (1,458 then 1,562 threads) but the script exited immediately after, before the replay. `set -euo pipefail` + a `taskset \| sed` pipeline in my stray-affinities diagnostic. C1 survived only because it pinned 0 threads so that code never ran |
| 160a | 33227244303 | **CCD pinning C1, all 3 bugs fixed** | Pin APPLIED: **1,418 then 1,490 threads**. ITL p50 **7.71** ms, 1,521 tok/s/GPU, 17/148 dropped (4th identical) |
| 160b | 33227244303 | CCD pinning C52 | in flight |
| 1 | 32025696861 | patch [4], DCP8, bf16 | FAIL 0x1016 @134,400 |
| 2 | 32039650984 | [4], DCP4, bf16 | FAIL 0x1016 @262,656 |
| 3 | 32042030173 | PR#51705 only, DCP8 | FAIL 0x1016 @135,168 |
| 4 | 32043813560 | #51705 + **[6]** | **0x1016 fixed**, errors=0 |
| 5 | 32049134216 | + AIPERF_FAST | 742.4, TPOT 663 ms; 113/183 dropped |
| 6 | 32055440757 | GSM8K gate | timeout — 3.5–6.8 tok/s, ≈15 h needed |
| 7 | 32060082326 | + PIECEWISE cudagraph | 780.6, TPOT 0.687 |
| 8 | 32066978737 | DCP off, [1]–[3] off | FAIL — `AITER_DISABLE_FMHA_OPUS` only set in the DCP branch |
| 9 | 32068474469 | DCP off, [1]–[3] on | FAIL @25 min — RCCL watchdog |
| 10 | 32070778181 | DCP off, [1][2][3] | FAIL @21 min — same signature, [5] not the cause |
| 11 | 32073039787 | DCP off, [2] off | 1,014.9, TPOT 0.170 |
| 12 | 32077536567 | + full profile | 999.4, TPOT 0.1698 |
| 13 | 32084553677 | chunk 8192→32768 | lost to the workflow concurrency rule |
| 14 | 32089974051 | yukiozzz image, c64, MTP | FAIL — `mla_gluon requires batch_size=1, got 64` |
| 15 | 32090860356 | yukiozzz, c64 | FAIL — same, got 3 |
| 16 | 32093774227 | c64, chunk 32768 | cancelled @20 min |
| 17 | 32094907936 | DCP8 c20 DRAM | FAIL — leftover `MAX_NUM_BATCHED_TOKENS=32768` |
| 18 | 32096487055 | DCP8 c20 **DRAM offload** | **1,990.8**, TPOT 0.1742 — 781 → 1,991 (2.55×) |
| 19 | 32101946357 | conc 64 | 1,041.1, TTFT 501 s |
| 20 | 32110276088 | TRITON_MLA + MTP | cancelled @55 min |
| 21 | 32114847961 | TRITON_MLA, spec off | 1,948.4 |
| 22 | 32123047671 | DCP off, c1 + c20, MTP | c20 **3,340.5**, TPOT 0.0429 |
| 23 | 32138970163 | GSM8K gate | **passed** 0.9659 / 0.9644 vs 0.9651 baseline |
| 24 | 32143146154 | DCP=2 | FAIL @init — `is_valid_num_heads(24)` |
| 25 | 32143877066 | DCP=4 c20 | **2,033.7**, TPOT 0.1674 — best DCP pre-correction |
| 26 | 32154159649 | DCP=4 **c8** | 969.4 |
| 27 | 32162233477 | direct symm-mem | FAIL @9 min — op absent, CUDA-only |
| 28 | 32163743641 | **ag_rs** combine | 1,978.4 |
| 29 | 32172937044 | interleave 1→16 | 1,977.2 |
| 31a | 32269094879 | local image tag | FAIL @6 min — harness uses `--pull always` |
| 31b | 32270805303 | **ported direct a2a** | 2,014.7 — **−0.9%** |
| 35e | 32333672290 | DCP=8 profiling | 92.65% collectives |
| 37 | 32340149740 | non-DCP profiling | 56.05% collectives |
| 38–41 | — | per-rank straggler analysis | straggler moves 7→0→1→5 |
| 45/46 | 32368684074 | NUMA pinning, node + per-rank | no effect / worse |
| 47 | 32372517517 | non-DCP, full window | 2,655.7, TPOT 0.1176 |
| 48 | 32381243949 | DCP=8 + **PYNCCL** | 1,531.6, TPOT 0.2271 — **−23%** |
| 50 | 32390477829 | non-DCP + MTP, FULL cudagraphs | FAIL @init — MLA full-capture assert |
| 51 | 32392005995 | non-DCP + MTP, c20 | cancelled — KV-starved at 1.32× |
| 54 | 32396466979 | non-DCP + MTP, **c8** | 541.0, TPOT 0.2921 |
| 55 | 32404418920 | EP=8 attempt | FAIL — `spec-decoding` also selects the script filename |
| 57 | 32406159156 | **EP=8** verified | 2,619.2, TPOT 0.1177 — neutral |
| 58 | 32414712217 | image bisect on ac7509e2b | 2,685.0, TPOT 0.1161 |
| 59 | 32424123796 | DCP + MTP on ac7509e2b | FAIL — config gate rejects the pair |
| 60 | 32425822552 | DCP + MTP on 5a4c8d99 | FAIL — no ROCm backend declares the capability |
| **64** | **32436403856** | **TP8, AITER MLA, spec off** | **4,622.8, TPOT 0.0461** |
| 65 | 32444354043 | + MTP | 2,045.4, TPOT 0.1003 |
| 66c | 32451498395 | + DCP=8 | 1,574.5, TPOT 0.2174 |
| 67b | 32458502570 | **DP2/TP4/EP8** | 2,998.6 corrected (5,997.2 reported), TPOT 0.1140 |
| 68 | 32466225062 | DP2/TP4 + MTP | cancelled |

---

## 8. Investigation history

### 8.1 Levers tested against the DCP decode penalty — all negative

| Lever | Result |
|---|---|
| DRAM KV offload | 2.55× as a feature; no effect on the straggler |
| Concurrency (c8 / c20 / c64) | optimum at c20: 969 / 2,034 / 1,041 |
| World size 8 → 4 | +2%; DCP=2 illegal (24 heads) |
| Combine algorithm a2a vs ag_rs | 2,034 vs 1,978 |
| Shard granularity interleave 1 → 16 | 1,977 |
| Attention backend (within DCP) | within noise |
| CUDA graphs | no TPOT effect |
| Async scheduling on/off | no effect; straggler persists |
| NUMA pinning, node-level | no effect |
| NUMA pinning, per-rank slices | worse (188 → 218 ms, rising) |
| Hand-ported direct P2P collective | works, −0.9% |
| All-reduce implementation (PYNCCL) | −23% throughput, +31% TPOT |
| MTP under DCP | refused at engine init |

### 8.2 Hypotheses the profiling eliminated

| Hypothesis | Evidence against |
|---|---|
| KV offload causes the straggler | Removing offload entirely; straggler persisted |
| A specific bad GPU or link | Straggler identity moves: rank 7 → 0 → 1 → 5 |
| GPU load imbalance | Compute per rank 1.05–1.11 s, ≤5% spread |
| DCP shards decode work | 11.5 vs 11.6 ms/step — no reduction |
| Async scheduling | Enabling it left the straggler intact (rank 5, 89% idle) |
| The collective implementation | PYNCCL substitution measurably worse |
| NUMA placement | No effect at node level; worse per-rank |
| A clean 1-vs-7 split | Intermediate ranks appear (265/272 µs) — a gradient |

### 8.3 Corrections made during the investigation

| Earlier claim | Correction |
|---|---|
| "The reference decodes on TRITON_MLA" | It sets no target backend; TRITON_MLA is the **draft's**. Forcing it cost 2.4× and was present in every non-DCP trial. |
| "The reference runs DCP=8 + MTP" | It runs **no DCP**. That configuration came from the B300 (NVIDIA) script — a different platform and file. |
| "MTP starves this workload" | The KV shortfall came from a pinned `--max-num-batched-tokens`. With it omitted the pool matched the reference exactly. MTP still loses, for a different reason. |
| "There is a 2.7× decode regression" | An artefact of comparing spec-on TPOT (per accepted token) against spec-off: `0.1161 / 2.509 = 0.0463` vs 0.0429. |
| "DCP's ceiling is ~2,000 tok/s/GPU" | Measured against a handicapped control. Re-baselined, DCP is −66%, not −24%. |
| "DP attention is infeasible on 8 GPUs" | Only the DP8/TP1 extreme fails. DP2/TP4 fits with ~50 GB spare and runs. |
| "The DCP decode cost is irreducible" | The cost is real but mostly **not DCP's** — it is the TP all-reduce any TP=8 decode pays, which DCP worsens by adding sync points. |

### 8.4 Script defect class — five instances

A value that appears set at one site is decided or discarded at another.

| Defect | Effect |
|---|---|
| Shadowed `DCP_SIZE` | An assignment inside the auto-concurrency block won; runs used DCP=4 / 17× KV when 8 / 32× was intended. |
| `DISABLE_SPEC` in a concurrency-gated block labelled DCP | Non-DCP arms silently ran without speculation while labelled `spec-mtp`. |
| Dead `CUDAGRAPH_CAPTURE_SIZES` | A sparse ladder overwritten by a later assignment in the same branch. |
| **Orphaned `EP_ARGS`** | Computed but never referenced in `VLLM_CMD` — **expert parallelism was unreachable for 55 trials**. |
| Split `DCP_SIZE` default | The patch gate read `1` while the run set `8`, so DCP ran without its patches. |

Structural remedies applied: `DCP_SIZE` resolves once before any consumer;
layout guards compute from measured weights rather than remembered rules; an
orphan check parses `VLLM_CMD` and diffs it against defined `*_ARGS` arrays.

### 8.5 Other notable failures

- A locally built image tag cannot resolve — the harness uses `docker pull --pull always`.
- `spec-decoding` in the matrix also selects the **script filename**; setting it to `none` pointed the harness at a non-existent file.
- `VLLM_TORCH_PROFILER_DIR` is obsolete; the knob is `--profiler-config`.
- A `-p2` patch dry-run reported success where a correct `-p1` showed only one nightly applies (23/24 hunks).
- Patch [8]'s guard matched *uses* of `_decode_num_heads` rather than its assignment, reporting "already patched" against a partial application.
- Acceptance length was misread from `sort -u | head` (the bottom five values) rather than the distribution; true mean 2.706 under `standard` rejection.
- Kimi-K3 fp8 KV decode requires a backend accepting an fp8 query input, so the draft-causal rewrite is a consequence of the fp8 KV choice, not an independent option.

# Autonomous run queue — Kimi-K3 / 8× MI355X

Owner away 2026-08-28 → 2026-08-30. This file is the single source of truth for
what runs next. Every wake-up: read **Current state**, act, update this file.

## Targets

| | target | best today | gap |
|---|---|---|---|
| Throughput | **10,632 (T195, C72, nightly+overlay+5 PR files, err 0.22%)** | **12,500 tok/s/GPU** · SA 8,953 | **−14.9%** |
| C1 interactivity | as low as possible | **7.57 ms** TPOT (T147, nightly) | — |

**T180 (2026-08-31): C1 engine is healthy.** First `TEST=1` fixed-len probe:
**10/10 requests, TPOT 7.41 ms, ITL 29.59 ms, zero faults** on b23_07. The
nineteen prior "C1 failures" were the *agentic workload*, not a broken engine or
a bad node. Job reported `failure` only because my TEST branch wrote the result
json to `/workspace/results` instead of the host workdir -- fixed to
`${INFMAX_CONTAINER_WORKSPACE:-/workspace}`.

**Two distinct faults, now separated:**
- **C1 memory access fault** -- agentic path only; never reproduced under
  fixed-len. Still unexplained. PR #37682 remains the candidate.
- **HSA_STATUS_ERROR_OUT_OF_RESOURCES** -- NOT node-specific. Seen on our
  b23_07 (C56) and on SA's mi355x-amds_02 (C52, run 33355794530). **No config
  variable explains it.** Verified from `non-default args` in each log:

  | | mns | MTP | offload | node | ladder | HSA |
  |---|---|---|---|---|---|---|
  | ours C56 | 80 | off | dram | b23_07 | 80 | yes |
  | SA C52 2026-08-31 (33355794530) | 80 | off | none | amds_02 | 80 | yes |
  | SA C52 2026-08-26 (32968517728) | 80 | off | none | amds_00 | 80 | **no -- 8,296** |

  Between SA's pass and SA's fail the only differences are node, the 5-day gap
  on the mutable `latest` image tag, and `load_format` (fastsafetensors vs
  safetensors). Our own C60 completes clean at mns=80 on b23_07 while C56 fails
  there, so it is not purely the node. **HSA is intermittent; no fix
  identified.** `MAX_NUM_SEQS=65` (script line 202) HAS been run and completed:
  **T133 = 7,725.96** at gmu 0.90 / offload **none**.

  **But there is NO clean mns 80-vs-65 measurement.** The often-quoted -2.8%
  against T103's 7,950.6 is **confounded**: T103 is mns 80 + **dram**, T133 is
  mns 65 + **none**. Two variables. The offload effect alone measures -1.1% to
  -1.8% in adjacent rows, so most or all of the 2.8% may be the offload, not
  mns. Do not quote "mns 65 costs 2.8%" -- it is unmeasured.
  (Two prior revisions of this line were wrong: first claiming nobody had run
  65, then attributing the full -2.8% to mns.)

  RETRACTED 2026-08-31: this entry previously claimed the fault tracked
  `mns=80 + offload=none`, then `mns=80 + MTP`. Both wrong. MTP is OFF at every
  throughput point on both sides, so SPEC_ROWS=1 and the "mns x 9 rows"
  mechanism does not exist.

- **MTP never runs above C4.** `kimik3_fp4_mi355x_mtp.sh:140-143` sets
  `SPEC_NUM_TOKENS=8` only for CONC in {1,2,4}; everything else gets 0.
  Confirmed empirically: our C60 (job 99326249467, 8,333 tok/s/GPU) logs
  `speculative_config = ABSENT`. **The entire C48-C72 ledger is
  non-speculative**, and `spec-decoding: mtp` in the yaml is only a
  script-filename selector. Speculative decoding at high concurrency is
  UNTESTED; `SPEC_NUM_TOKENS` is env-overridable, so it is reachable.

- **SA comparison is apples-to-apples.** Both 8,342 (ours) and 8,296 (SA) are
  spec-none. The 0.55% lead stands; remaining deltas are conc (60 vs 52),
  offload (dram vs none), and CCD pinning.

## C72 = 10,632 / 10,630 (n=2, spread 0.02%). PEAK CONFIRMED. mns closed.

mns 96 at C72 is identical to mns 80 -- the C80 gain was purely the conc==mns
starvation, not a general win. Concurrency and mns are both exhausted.

Curve: 52=8,685 / 60=9,482 / 64=9,775 / **72=10,632** / 80=9,864 (mns96: 10,159).

Next lever: **chunk 8192 vs 16384 at C72**. N4 said 8192 was optimal, but that was
measured on aigmkt and every aigmkt top-of-curve finding has since fallen. Chunk
has never been A/B'd on the nightly stack. One variable against T195.

## mns 96 helps (+3.0% at C80) but C72 still peaks. Next: C72 + mns 96.

C80: mns80=9,864 -> mns96=**10,159**. Recovers ~40% of the C72->C80 drop, so
mns headroom is real but is not the whole cause. **N5 ("mns 96 kills the
engine") falsified** -- clean run, err 0.30%.

Curve (mns 80): 52=8,685 / 60=9,482 / 64=9,775 / **72=10,632** / 80=9,864.

## PEAK FOUND: C72 = 10,632. C80 drops to 9,864 (-7.2%).

Curve: 52=8,685 / 60=9,482 / 64=9,775 / **72=10,632** / 80=9,864.

Next: C80 with **mns 96**. At conc 80 mns==conc (pinned 80 for DCP>1) leaving no
headroom for the agentic lanes that spawn past nominal concurrency. If mns is the
limiter the peak moves right; if not, C72 is the operating point and the -14.9%
to 12,500 needs kernel work.

## NEW BEST 10,632 (T195, C72) -- +18.8% over SA, -14.9% from target

C72 agentic, err 0.22%, zero faults. **aigmkt died outright at C72**; here it is
the best point yet and the curve has still not peaked (52->60->64->72 gives
+9.2%, +3.1%, +8.8%).

Next: C80. Note mns is fixed at 80 for DCP>1, so at C80 conc == mns and there is
no headroom for the agentic lanes that spawn past nominal concurrency. If C80
flattens, mns is the next variable -- and N5's "mns 96 kills the engine" is an
aigmkt-era finding that should not be trusted on this stack.

## NEW BEST 9,775 (T193, C64) -- +9.2% over SA, -21.8% from target

C64 agentic, err 0.09%, `[pr-stack] applied=5 skipped:none`. The aigmkt C64
cliff (-4.4%) does not exist on this stack -- throughput is still climbing at 64.

CAVEAT: two variables moved (conc 60->64 AND #50813 SiTUv2 A8W4 MoE landed).
#50813 changes MoE quant math and is NOT GSM8K-validated -- T192's 0.995 covers
the 4-file stack only. Running eval now before pushing to C72.

## GSM8K VALIDATED: 0.995 (T192). Image saved as kimi-k3-vllm-v2.

Now PRUNING the patch set. Target: minimal patches that keep GSM8K passing AND
throughput >= 9,482. Ablation plan, one variable each:

| run | stack | question |
|---|---|---|
| A | overlay only (`APPLY_PR_STACK=0`) | is #53940 doing anything? |
| B | overlay + #53940 (= v2, 9,482) | baseline, already measured |
| C | overlay + #53940 + #50813 | does SiTUv2 A8W4 MoE add? |

The 264 KB overlay is NOT ablatable at reasonable cost -- 34 files would need 34
runs. Only the PR-stack files are prunable at this budget.

## NEW BEST 9,482 (T190, C60, nightly stack) -- AHEAD OF SA

C60 agentic, 2,076/2,202 successful, error rate 0.14%, zero HSA/memfault.
**+5.9% over SA's 8,953**, +13.7% over our old aigmkt best 8,342, +9.2% over
T189's C52 on the same stack. Gap to 12,500 now **-24.1%** (was -33%).

C60-over-C52 is +9.2% here vs +2.8% on aigmkt -- the conc curve is steeper on
this stack, so the old peak at 60 may not be the peak any more. Next: C64.

## NEW BEST 8,685 (T189, C52, nightly stack) -- 2026-08-31

Agentic replay, 2,090/2,200 successful, error rate **0.14%**, zero HSA/memfault.
Beats our old best 8,342 (C60, aigmkt) by **+4.1%** and our C52 8,115 by +7.0%.
Still **-3.0% vs SA's 8,953** and **-30.5% from 12,500**.

Remaining known gaps vs SA at C52: cudagraph capture 80 vs 4096; five aigmkt-era
env vars SA does not set; node b23_07 vs amds_01.

Next: C60 on the nightly stack -- the old curve peaked at 60, and that peak was
never re-measured after the image change.

## SA is 7.3% AHEAD, on a nightly image (read 2026-08-31)

SA run 33324464095 ("Kimi K3 current findings baseline", 2026-08-30, branch
`amd/kimi-k3-current-baseline-20260831`), all three jobs green:

| job | conc | tok/s/GPU | node | config |
|---|---|---|---|---|
| c1 | 1 | 1,229 | amds_03 | MTP ON k=6, mns 2, gmu 0.875, no DCP |
| c16 | 16 | 4,208 | amds_02 | spec-none, DCP8, mns 80, gmu 0.86, chunk 8192, capture 80 |
| c52 | 52 | **8,953** | amds_01 | spec-none, DCP8, mns 80, gmu 0.9, dram vllm-simple |

**8,953 beats our best 8,342 (C60) by 7.3%, and our own C52 8,115 by 10.3%** --
at the same concurrency, so this is not an operating-point difference.

### The image is the story

```
IMAGE: vllm/vllm-openai-rocm:nightly-46638857fdbb30e0c232c9e8f9cb1ff6d6f545c3
```

| | ours | SA c52 |
|---|---|---|
| image | `aigmkt/kimi-k3-vllm:latest` | **nightly 46638857** |
| max_num_batched_tokens | 8,192 | **16,384** |
| max_cudagraph_capture_size | 80 (dense 1..80) | **4,096** |
| load_format | auto | fastsafetensors |
| CCD pinning | yes (1,562 threads) | none |

**Every negative result in this file was measured on aigmkt.** N4 ("8192 is the
optimum, 16384 is -2.5%") and the LADDER_MAX<=80 cap are both contradicted by a
faster SA run on the nightly. They are image-specific, not general.

### HSA fault tracks the IMAGE, not the node (2026-08-31)

Third sighting, SA run 33360219789 job 99390061271, node **amds_03**, aigmkt:
`HSA_STATUS_ERROR_OUT_OF_RESOURCES` -> `hipErrorUnknown` -> `VllmWorker-6 died
unexpectedly` (multiproc_executor.py:314) after 51 min; aiperf 88/872 failed
(10.09%) -> ProfileAborted. Full run: 784 successful / 979 total.

| run | image | node | mns | offload | HSA |
|---|---|---|---|---|---|
| ours C56 | aigmkt | b23_07 | 80 | dram | yes |
| SA C52 08-31 04:05 | aigmkt | amds_02 | 80 | none | yes |
| SA C52 08-31 05:21 | aigmkt | amds_03 | 80 | none | yes |
| SA C52 08-26 | aigmkt | amds_00 | 80 | none | no -- 8,296 |
| SA C52 08-30 | **nightly+overlay** | amds_01 | 80 | dram | no -- **8,953** |

**Three failures on aigmkt across three different nodes. Zero on the nightly.**
The node hypothesis (b23_07 is bad) is dead.

CORRECTED 2026-08-31: this section previously blamed mutable-tag drift -- that
`aigmkt:latest` was rebuilt between 08-26 and 08-31. **Wrong.** The registry
manifest for `aigmkt/kimi-k3-vllm:latest` is dated **2026-08-26T06:38:27Z**, six
hours BEFORE SA's passing 08-26 run, and unchanged since. The pass and the three
failures ran the SAME build. So aigmkt is **1 pass / 3 fails on one fixed
image** -- intermittent, not drifted. The 134 s -> 2,557 s weight-load blowup is
therefore environmental (node I/O, or a stale per-node enroot squashfs cache --
SA's own launcher comments warn that a squashfs of nominally the same tag can
differ from the registry image).

Also corrected: the image config history shows
`RUN ... bash /w/apply_kimi_k3_patches.sh`, so **the patches are BAKED INTO
aigmkt at build time**. The earlier claim that the whole ledger is "unpatched
stock image" was wrong. Entrypoint is `["vllm","serve"]`.

Caveat: the nightly has 0 failures but on n=1. T182 is the test.

### Correction to a bound I asserted wrongly

I repeatedly said a different image was "out of bounds". That was wrong. The
hard bound is **NO Docker Hub push** -- publishing. The `image:` field is
`configs/amd-master.yaml:1961`, inside the kimi block, which IS on the editable
list. Pulling an existing nightly is a one-line yaml edit and always was allowed.
Several cycles were spent treating "needs a different image" as a dead end.

Related limit that is real: `apply_kimi_k3_patches.sh:156` is `patch -p1` with no
build step (no cmake/ninja/setup.py). It can patch Python in site-packages but
cannot produce a compiled HIP op -- which is exactly why N9 (#51705's
`direct_dcp_a2a_lse_reduce`) failed. Compiled-kernel PRs need an image that
already ships them, not runtime patching. Also note the patch script is never
invoked by the live launcher (only `archive/..configurable.sh:220`), so the whole
ledger is unpatched stock image.

### Next: reproduce SA's C52 wholesale

Deliberately breaks ONE-VARIABLE. Change image + chunk 16384 + capture 4096 +
fastsafetensors together and target 8,953. Rationale: swapping only the image
while keeping knobs tuned for aigmkt tests an unmeasured combination and would
likely read as a regression for the wrong reason. Decompose after matching.

**An image swap invalidates the ledger as a comparison set.** 8,342 must be
re-established on the new base before any further claim means anything.

## T188: nightly stack is GREEN and faster (fixed-len)

645/645, **6,616.31 tok/s (+13.3%)**, **TPOT 65.89 ms (-13.0%)** against T180's
aigmkt baseline on the identical fixed-len harness. Zero HSA, zero memfault.
`[pr-stack] applied=4 files` (only `cudagraph_utils.py` skipped -- #54095's hunk 2
is cut against a newer tree). KV pool 51.29 GiB at chunk 16384.

Costs: TTFT +42%, ITL P99 +101%. The chunk 8192->16384 trade, now measured.

Perf gate PASSED -> next is the agentic replay, the only run comparable to SA's
8,953 and our 8,342.

## Three-stage gate (2026-08-31, per owner)

Run in order. Each stage only runs if the previous one passed.

| # | stage | how | C1 | C52 |
|---|---|---|---|---|
| 1 | **Functionality** | `TEST=1 TEST_MODE=func` -- agentic-band lengths, few iterations | 4 prompts | 104 prompts |
| 2 | **Perf fixed-len** | `TEST=1 TEST_MODE=perf` -- 8k/1k fixed, ~15 min | 112 prompts | **645 prompts** |
| 3 | **Perf agentic** | `TEST=0` -- the real agentic replay | full | full |

Stage 1 lengths: ISL 214,000 / OSL 874 / ratio 0.37 -> uniform[79,180, 214,000]
in and [323, 874] out, i.e. the agentic p50..p90 band. Stage 2 is ratio 1.0,
exactly fixed, so it is comparable run to run.

**Caveat on stage 2's "15 minutes":** `benchmark_serving.py` has no duration
flag, only `--num-prompts`. The counts above are derived from an assumed
per-request latency, both now MEASURED in T180: C1 8.07 s, C52 72.57 s (mean
E2EL over 520 requests). The earlier C52 figure of 13 s was a guess and was
wrong by 5.6x. Read the reported
`Benchmark duration (s)` and set `TEST_EST_REQ_SECONDS` to calibrate.

**Caveat on stage 1 as a workload proxy:** it matches agentic *lengths* but not
its 93.3% prefix-cache hit rate, so it does several times the prefill work per
token. It is a functionality gate, not a throughput number -- do not compare its
tok/s to the agentic ledger.

**Why stage 1 exists at all:** T180's 8k probe passed C1 10/10 while the agentic
replay has failed 19 straight times. A probe that passes when the real workload
fails is not a gate.

**Honest position on 12,500, restated because it drives priorities:** the T124
profile puts GPU idle at 28.2% of e2e wall. Eliminating idle *entirely* yields
~11,050 tok/s/GPU. Every remaining kernel lever is single-digit percent. So
12,500 is **not reachable by stacking the levers currently identified** — it
needs either a kernel-level step change (the nightly's #53942 class of work) or
a different operating point. I will keep pushing and report the real number
rather than a flattering one.

**CAVEAT THAT MUST TRAVEL WITH THE 8,342 HEADLINE — there is no complete run.**
The two C60 runs (T176 `33329440318`, T177 `33337236325`) are **green at the
GitHub level** — run and both jobs report `conclusion: success`. That status is
misleading: **the C1 arm produced no measurement in either.** aiperf hit
`ProfileAborted` at 15/145 and 15/146 (10.3% > the 10% threshold) and the
harness still exits 0.

- The **C60 number is unaffected**: separate job, separate server launch,
  different config (C1 is dcp=1 / mns=8 / offload=none). C60's own error rates
  were validated at 0.217% and 0.109% with HSA = 0.
- But **the curve has a hole at C1, and has for 18 consecutive attempts across
  every config tried.** This is not a C60 property — no config in this campaign
  has ever produced a passing C1.
- **If the deliverable is the full concurrency curve rather than one operating
  point, we do not have a submittable result.** The entire gap is N8 (executor
  RPC `dequeue_timeout`), which needs vLLM source. Blocked on access, not ideas.

## Bounds — never cross

- Dispatch only to `ajith-sirra-amd/InferenceMAX_Rocm_Team`. `SemiAnalysisAI/InferenceX` is **read-only**.
- **No Docker Hub pushes.** Everything via runtime patches.
- **No git push anywhere except `InferenceMAX_Rocm_Team`.** Not to `ajith-sirra-amd/vllm`.
- Edit only: `kimik3_fp4_mi355x_mtp.sh`, `kimik3_fp4_mi355x_mtp.sa.sh`, `apply_kimi_k3_patches.sh`, the `pr51705_nightly.diff`, the kimi block in `amd-master.yaml`, and root docs.
- **No code changes anywhere else.** vLLM changes live in the vendored diff only.
- Only cancel runs I started.
- **Accuracy gate before throughput** whenever numerics could move.

## Current state

**Config tuning is CLOSED (T195/T198/T199).** Three independent knobs measured
flat at the C72 operating point:

| knob | runs | result |
|---|---|---|
| concurrency | 52/60/64/72/80 | peak 72 = **10,632**, n=2 (10,632 / 10,630) |
| max-num-seqs | 80 vs 96 @ C72 | 0.02% spread -- neutral |
| max-num-batched-tokens | 16384 vs 8192 @ C72 | 0.07% spread -- neutral |

Headline **10,632 tok/s/GPU** = **-14.9%** from 12,500, **+18.8%** over SA's
8,953. GSM8K 0.995 (4-file pr-stack) / 0.99 (5-file). The remaining gap is not
in the launcher's argument space.

**C1 chunk A/B is DONE and is the first non-flat knob on this stack.**

| C1, 214k agentic-band | chunk 16384 (T200) | chunk 8192 (T201) |
|---|--:|--:|
| Mean TPOT | 9.70 ms | **8.92 ms** |
| Median TPOT | 8.97 ms | 8.96 ms |
| P99 TPOT | 12.31 ms | **9.18 ms** |
| Mean TTFT | **13,663.8 ms** | 15,772.4 ms |

Median identical, p99 −25.4%, TTFT +15.4%. Chunk 8192 does not speed decode up,
it stops decode stalling behind an oversized prefill chunk. Opposite sign to
C72, where chunk was flat both ways (T199) because 72 concurrent requests keep
the batch full regardless of slice size.

**Standing recommendation: chunk 8192 at C1, 16384 at C72.** Per-concurrency,
which matches the guidance that chunk/mns/ladder need not be bound across
C1 and C52+.

**Now running: T202 -- `PROFILE=1` hotspot run on the C72 config.** First
execution of the profile path. `TEST_MODE=perf` (8k/1k lengths), 72 prompts =
one concurrency wave, `TEST_OSL=256` to keep eight per-rank traces inside the
artifact upload limit. Traces land in `$RESULT_DIR/torch_profile` and the whole
RESULT_DIR is uploaded as the `agentic_*` artifact (confirmed: that artifact is
exactly RESULT_DIR).

Caveat to read it with: 8k fixed-len gets none of the agentic 93.3% prefix-cache
reuse, so prefill will be over-represented relative to the real workload.
Steady-state decode at batch 72 is the part that transfers.

**Queue after T202:**
1. Act on the profile -- kernel or scheduler work. This is the only remaining
   path to the last 14.9%; config tuning is exhausted (conc, mns, chunk all flat
   at C72).
2. Attribution: C64 without #50813 (T193 moved two variables).

## Queue — REPRIORITISED 2026-08-29 against the T124 profile

The old order led with tuned GEMM. That does not match our own measurements.
Ranked by **% of e2e wall the lever can actually touch**
([Where-The-Time-Goes](Kimi-K3-Where-The-Time-Goes.md)):

| rank | target | % e2e wall | queue item |
|---|---|--:|---|
| 1 | idle, host launch/prep in decode | **28.2%** (37% of it = host) | **N2** async sched |
| 2 | collectives, `dcp:0` on generic PYNCCL | **21.3%** | **N3** dcp comm backend |
| 3 | MLA prefill attn | 16.9% | no lever (needs FP8 FMHA) |
| 4 | BF16 dense GEMM | 11.9% | **N4** tuned configs |

### N1 — CCD pinning after `wait_for_server_ready` — **DONE (T161), kept**

T160 measured **2008.62 s** to load weights (vs 576–681 s unpinned): the pre-pin
loop confines the ~190 loader threads per worker to one CCD's 8 physical cores
during weight load and cudagraph capture. Pinning must be **one-shot, after the
server reports ready** — steady-state locality is the thing we want; load and
capture are one-time and must run across all cores. This also makes C3
measurable for the first time (T160's number is confounded by the load penalty).
Remove the background pre-pin loop entirely.

### N2 — async scheduling — **DONE (T162): −1.8%, SETTLED NEGATIVE**

The profile attributes **~150 s of 403.9 s idle (37%) to host/Python**: batch
tensor build + **~127 H2D `copyBuffer` per step** (71.9 s alone), sampling
elementwise (38 s), allocator memsets (28 s). Async scheduling is the one lever
that overlaps that host work with the GPU step. It was **−9.2% on the old
engine**; the scheduler has moved 175 commits and the profile says this is where
the time is. Safe at C52 (`k=0`, no spec decode). No numerics change.

### N3 — the `dcp:0` group never gets the fast all-reduce

    group 'tp:0'  -> ['AITER_CUSTOM', 'PYNCCL']
    group 'dcp:0' -> ['PYNCCL']

Generic `ncclDevKernel_Generic_1` = **22.55% of GPU busy**; the tuned
`cross_device_reduce_2stage` TP uses = **3.63%**. 21.3% of wall sits in the group
that did not get the fast backend. Try `--dcp-comm-backend` alternatives to
`a2a`; check what the nightly's enum accepts before dispatching.

### N4 — chunked prefill size — **DONE (T164): 4096 is −7.4%, 8192 is the peak**

### N5 — `max_num_seqs` 80 → 96 — **IN FLIGHT (T165)**

Capacity is the only axis that has moved this number (T163's +3.9%). Push it.

### N6 — AITER tuned GEMM configs

T160 C52 still logs `not found tuned config ... will use default config! using
torch solution:0` — the miss falls all the way back to **plain torch**, not an
aiter kernel. Miss shapes confirmed in T160: decode `M:19..80 N:6288 K:7168` and
`N:3584 K:7168`; prefill `M:8192` × {8448, 7168, 6288, 3584, 2304, 2112, 1536}.
Needs an **offline tuning job**, not a benchmark dispatch. No numerics change.

### RETIRED — C2 (#52190 torch.compile silently disabled)

**Does not apply on the nightly.** T160 C52 logs
`Enabled custom fusions: norm_quant, act_quant, allreduce_rms, mla_dual_rms_norm`
with `mode=VLLM_COMPILE`, populated `splitting_ops`, and **no**
`does not support it` line. Fusion is already on — and already inside T156's
7,906 (−0.6%). So post-grad fusion is worth ~nothing here. Closed.

Also confirmed live at C52: `cudagraph_mode=FULL_AND_PIECEWISE`, sizes 1..80,
`VLLM_ALLOW_DCP_FULL_CUDAGRAPH=1`. Cudagraphs are not the missing piece; the
residual launch gaps are in *mixed* prefill+decode steps and in host prep, which
is what N2 targets.

### Remaining, unchanged
- **C4** `gpu-memory-utilization` 0.90 → 0.95.
- **C5** RCCL env: `NCCL_PROTO=LL/LL128`, `NCCL_ALGO`, `NCCL_MIN_NCHANNELS`.
- **C6** Custom all-reduce `max_size` 8 MiB → larger. Prefill AR is ~117 MB and
  always falls back to RCCL; that is where 16.93% of GPU busy lives.
- **C7** Re-test async scheduling **on the nightly**. It was −9.2% on the old
  engine; the scheduler has moved 175 commits.
- **C8** KV offload on/off at the best C52 config (SA: dram 8,296 vs none 8,204,
  so ~1.1% to them).

### Phase D — C1 TPOT
- **D1** TP=4. Step is 87.5% fixed overhead (~118 8-rank barriers/step); halving
  world size halves participants. Weights 192.63 GiB → 48 GiB/GPU, fits 288.
  **Caveat: idles 4 GPUs, so tok/s/GPU halves under the harness's /8 divisor.**
  Valid for a TPOT question only.
- **D2** k > 8. Blocked: needs a golden AL entry beyond k=8. The two-point fit
  (`step = 29.44 + 0.465·(k+1)` ms) says TPOT is still falling at k=8.
- **D3** torch.compile (C2) re-measured at C1.

### Phase E — final sweep (only once C52 is maximised)
`C1, C4, C8, C16, C32, C40, C48, C52, C54, C56, C64, C72` on the best config.
Expect the peak between 40 and 56; we have 40 and 52 and nothing between.

## Rules for each run

1. One variable per run. Log the gate lines (`[dcp]`, `graphs:`, `[mns]`, `[eval]`).
2. Numerics-affecting change → GSM8K (limit 200) **before** the perf run.
3. Record result in `Kimi-DCP-Experiemnts-Summary.md` and update **Current state** here.
4. If a run fails, diagnose from the log before re-dispatching. Never re-dispatch blind.
5. Never leave the GPUs idle — if nothing is running, dispatch the next queue item.

## Ledger — what is already settled (do not re-run)

| | |
|---|---|
| DCP=8 at C1 | **+36.5% TPOT.** DCP off at CONC ≤ 4. |
| DCP comm flags at size 1 | inert — identical TPOT, byte-identical KV pool |
| load-format auto vs fastsafetensors | inert on nightly (7.57 vs 7.57); auto gives +4.7% KV pool |
| draft KV fp8 | TPOT-neutral, **+36.5% KV pool** — keep |
| ladder / mns at C1 | inert on nightly (72, 16, 9 all 7.57–7.58); ladder 9 cuts capture 44 s → 7 s |
| draft on ROCM_AITER_MLA | impossible — TRITON_MLA is the only ROCm backend with `supports_non_causal_multi_token_decode` |
| QuickReduce FP | −8.39% |
| EP=8 | −4.7% |
| FP16 GEMM | loses 6 of 8 shapes |
| MTP above CONC 4 | −85% at C40 |

- **T174 DISPATCHED: C64, not C56.** Deliberate choice. C64 is the *only*
  recent config that produced a clean number (T170: 8,040, **HSA = 0**), so this
  run does double duty:
  - **Node-health probe.** If C64 comes back with HSA > 0, the node is still
    accumulating queue resources and no throughput number from it is usable —
    diagnosed without pretending it is a config result.
  - **A real second sample of C64** if the node has recovered, which the ledger
    needs since every point in Phase E except C52 is n=1.
  Re-running C56 a fourth time would have produced neither.

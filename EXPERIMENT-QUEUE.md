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

## ROADMAP (user-set, 2026-09-02) — four phases, in order

**Goal: land on upstream nightly + a minimal patch set that still does 10,600,
then push for 12,500, then profile and optimize kernels.**

### Phase 1 — prune to the minimum that preserves ~10,600 on upstream nightly

The **overlay axis (A/B/C/D/E) is CLOSED**: only A+B are detachable (8 KB, both
neutral at C72); C+D+E fail on imports and are one coupled 256 KB unit
(T221-T226). There is no "F..Z" left on that axis -- the patch does not divide
further by subsystem.

The **live axis is the PR axis**:

| step | what | status |
|---|---|---|
| 1.1 | `pronly` GSM8K gate | **DONE** — 0.99 (T231) |
| 1.2 | `pronly` C72 throughput | **T232 running** — prices the overlay |
| 1.3 | rebase the 5 conflicting PRs onto `7c5dc571` | pending 1.2 |
| 1.4 | leave-one-out over `pr_split` buckets | pending 1.3 |

**1.2 is the decision point.** If `pronly` lands in the v4 band we are already
done and the overlay is prunable in full. If it lands below, the gap is owned by
the five PRs that conflict with `7c5dc571` itself -- **#53166** (1 of 8 hunks),
**#51437** (1 of 6), **#53301** (2), **#52190** (1 of 1), **#54163** (1 of 1).
Each fails standalone against the pristine base, so this is context drift, not
design conflict: a rebase, not new engineering.

Order to attack in 1.3, by expected value on this workload:
**#53166** (MLA prefill fusion; ISL p50 ~87k, in:out ~195:1) → **#51437**
(decode all-reduce overlap; owns TPOT/ITL) → **#53301** (per-step metadata,
6 MLA + 14 KDA groups at TP8) → #52190 → #54163.

Prune target for 1.4: drop every bucket that costs nothing, keeping the result
inside the 10,607 +/-1.2% band. Expect-free first: P51392 (NVIDIA path),
P54165 (spec off at C72), P52033 (multi-stream off), P50618 (isolated guard).

### Phase 2 — LMCache, target 12,500

Not wired for K3 today: `kimik3_fp4_mi355x_mtp.sh` has zero LMCache references;
we run `SimpleCPUOffloadConnector`. Port the server block from
`minimaxm3_fp4_mi355x.sh` (the only live AMD script with one), confirm it starts
on ROCm, then a single C72 arm.

The case for it is the **prefix-cache gap**: the trace offers
`theoretical_prefix_cache_hit=95.7%` and we capture **73.4%**, with
`ext_cache_hit` ~78-79% and `kv_usage` only ~49%. Twenty-two points of available
reuse are being left on the floor. Caveat to hold: the published LMCache wins
are B200/B300 CUDA and lean on CUDA-IPC export of KV buffers, which does not
port to ROCm for free. Also worth ruling out first that the gap is a
geometry/eviction defect rather than a backend-capacity limit -- #53598, #53917
and #54163/#54165 are all bugs in the family "prefix-cache hits structurally
dropped under DCP/hybrid/spec".

### Phase 3 — profiling, then kernel optimization

**Both in-tree profiling paths are currently DEAD** and this needs solving before
Phase 3 can start:
- T202: `nightly-46638857` has no `VLLM_TORCH_PROFILER_DIR` and no
  `start_profile` handler.
- T203: `rocprofv3` deadlocks the engine during capture
  (`queue_interposition.cpp:374 Async signal handler still waiting`, RCCL
  collective timeout).

Moving to `7c5dc571` may fix the first -- worth re-checking on `pronly`, since
that base is five weeks newer.

Standing lead for when this opens, from live aiter shape logs (2,304 lines):

| N, K | count | what |
|---|--:|---|
| N:6288, K:7168 | 744 | KDA fused multi-output projection (TP8) |
| N:3584, K:7168 | 744 | attention projection shard |
| **N:1536, K:128** | 744 | **KDA `f_b_proj`** — skinny, memory-bound |

`N:1536, K:128` is the standout target: K=128 is bandwidth-limited with almost
no reuse, every logged call shows `bpreshuffle=False`, and #50618 independently
documents a stride bug on this exact call (`stride=(6288, 1)` under TP8 -- note
6288 matches the top shape). **Caveat: these are capture-phase enumerations, not
runtime frequency** -- M sweeps 1..96, mirroring the cudagraph ladder. Real
attribution needs a working profiler, which is why this is Phase 3 and not now.

**Parked at user request: the GEMM microbenchmark. Do not queue it.**

## Current state

**Config tuning is CLOSED (T195/T198/T199), and the headline now carries an
error bar (T228).** Four runs on a materially identical C72 stack give
**10,607 tok/s/GPU +/-1.2% (n=4)**: 10,632 / 10,630 / 10,646 / 10,518. The
cleanest pair (T206 vs T228 -- same image, same script, 26 h apart) differs by
1.20%, so **every C72 delta below ~1.2% in this ledger is unrankable noise**.
That includes the mns and chunk results below: they are confirmed neutral, but
none of them identified a winner.

**T230 corrects T229 and re-opens mns.** C76 + mns 96 = **10,624**, +2.91% over
C76 + mns 80, and inside the C72 band. The C76 dip was mns starvation, not a
concurrency limit: the replay ran **81-83 lanes at conc 76**, which does not fit
under mns 80. TTFT mean fell 33.7% and p99 42.8% while ITL barely moved --
queueing, not compute. The peak is a **plateau across 72-76 at ~10,61x**, and
the rule is *mns must exceed the replay's lane count (which runs above CONC)*,
not "mns is neutral" as T198 was generalised to say. Shipping number unchanged
at 10,607 +/-1.2%.

**T231: GSM8K 0.99 on `kimi-k3-vllm:pronly` -- the fully-mergeable stack is
numerically sound.** No overlay, base `nightly-7c5dc571`, 12 of 18 PRs. 0.99 vs
v4's 0.995 at +/-0.0071 is indistinguishable. The job shows `failure` only
because EVAL_ONLY writes no benchmark JSON and the post-step waits for one --
`eval_exit=0`, 200/200 served. Not a defect.

## PRUNING PARKED — bare-nightly C52 is the active thread

PR-prune (Phase 1.4, leave-one-out over `pr_split`) is **parked** at user
request until bare-nightly C52 resolves. Phase 1's goal is already met
(T232: pronly 10,692 = v4), so nothing is blocked by the pause.

## QUEUED — T234: bare nightly C52 agentic, gmu 0.85

**Dispatch when T233 clears.** Ref config: `kimik3_fp4_mi355x_mtp.old.sa.sh` —
mns 80, chunk 16384, DCP 8, `kv-offloading: none`, conc 52.
**ONE variable: gmu 0.9 → 0.85.**

### Root cause (corrected)

SA run 33596998428, `Worker_TP2_DCP2`, 07:03:06 — the line *before* the
traceback:

```
:0:rocdevice.cpp:3715: Callback: Queue 0x70e817200000 Aborting with error :
HSA_STATUS_ERROR_OUT_OF_RESOURCES: The runtime failed to allocate the necessary resources
-> hipErrorUnknown
-> segfault in TensileLite::GetDevice / torch.cuda.current_device
-> Worker proc died -> EngineDeadError
```

**The segfault frames are corpse frames.** Once the HIP context is dead, any
device-property query faults — which is why two unrelated call sites (Triton
autotune in `chunk_delta_h.py`, and hipBLASLt Tensile) both crashed in device
lookups. I initially diagnosed those frames as the cause and staged a hipBLASLt
workaround; that was wrong and has been reverted.

Warmup had completed **107/107, errors=0**; it died at the warmup→profiling
handoff. With offload disabled the whole KV working set sits on GPU, which is
what exhausted the queue allocation.

### Why gmu 0.85, and why it is gated

gmu is the direct lever on the queue/scratch headroom the runtime failed to
allocate. Direction matters: the ledger only ever pushed gmu **up** at C52 and
it broke both times — T157 gmu 0.95 hung the engine, T166 gmu 0.92 gave 0/103
in warmup. **Downward is untested.**

Gated on the absence of `/etc/k3-image-manifest`, so **v4 and pronly keep gmu
0.9**. Changing it under them would move every number in the ledger, and T211
measured gmu as violently non-neutral (0.92 at C1 → 2.4× worse TPOT).

Gate lines to verify: `[gmu] bare image -- override 0.85 …` and
`[cfg] … gmu=0.85 …`.

### If 0.85 is not enough

Next lever, one variable at a time: **ladder/mns 65 → 52** at gmu 0.9. Both are
resource-side; do not change them together or the result is unattributable.

---

**IMAGE CHANGED MID-FLIGHT — read before interpreting T233.** `kimi-k3-vllm:pronly`
was rebuilt at 2026-09-02 ~07:4x to **drop #54165** (closed-unmerged upstream;
author closed it as superseded by #54163; spec decode is off at C72 so it was
inert there). The tag now means **11 PRs (4 merged + 7 open), nothing closed**.

**T232 and T233 both ran the OLD 12-PR image** — they were already dispatched.
Their numbers are valid for that stack. The first run on the 11-PR image will be
T234. Expect no difference at C72 (spec off), but it is n=0 until measured, and
at C1 it could matter because MTP is on at CONC <= 4.

**T232 RESULT: pronly C72 = 10,692 — the overlay is worth NOTHING at C72.**
The fully-mergeable stack (12 upstream PRs on `nightly-7c5dc571`, zero vendor
patch) matches the 264 KB overlay: +0.6% over the matched-mns overlay run
(T198, 10,630), inside the +/-1.2% band, error rate 0.18% (best in ledger).
**Phase 1's goal is met.** My pre-registered prediction that it would land
below the band was falsified -- missing #53166/#51437/#53301/#52190/#54163 and
the 26 orphan hunks costs nothing here. The 5-PR rebase (step 1.3) is therefore
**optional, not blocking**. Caveat: n=1 against a +/-1.2% band, C72 agentic only;
C1 TPOT on pronly is untested. **T233 = n=2 replication** before this is relied on.

*(superseded)* T232 dispatched: C72 agentic on `pronly`. This is the run that prices the
overlay -- v4 is 10,607 +/-1.2%, and the gap is what the 264 KB of unpublished
patch plus the 5 conflicting PRs are worth. Prediction on record: pronly lands
below the v4 band, because #53166 (MLA prefill fusion) is missing and this
workload is prefill-dominated (ISL p50 ~87k, in:out ~195:1).

**T229 filled the last hole in the conc curve: C76 = 10,324**, −2.7% below the
C72 mean and outside the noise band. Curve is now 60/64/72/76/80 =
9,482 / 9,775 / **10,607** / 10,324 / 9,864 — peak 72, gradient fall-off, no
cliff. Caveat: mns is pinned at 80, so C76 has only 4 slots of slack and sits
partway into the C80 starvation regime (T196/T197). **T230 = C76 + mns 96** is
dispatched to separate conc from mns.

Three independent knobs measured flat at the C72 operating point:

| knob | runs | result |
|---|---|---|
| concurrency | 52/60/64/72/80 | peak 72 = **10,632**, n=2 (10,632 / 10,630) |
| max-num-seqs | 80 vs 96 @ C72 | 0.02% spread -- neutral |
| max-num-batched-tokens | 16384 vs 8192 @ C72 | 0.07% spread -- neutral |

Headline **10,607 tok/s/GPU +/-1.2% (n=4)** = **-15.1%** from 12,500, **+18.5%**
over SA's 8,953. GSM8K 0.995 (4-file pr-stack) / 0.99 (5-file). The remaining gap is not
in the launcher's argument space.

**T202 FAILED its objective: no profile was produced.** `nightly-46638857` has
no `VLLM_TORCH_PROFILER_DIR` and no `start_profile` handler in
`entrypoints/openai/api_server.py` -- verified in the image. The server logged
`Unknown vLLM environment variable detected` and ignored it; `--profile` never
reached the client either (`profile=False`). Zero traces, 52 KB artifact. The
torch-profiler design is dead on this stack and has been deleted, not retried.

**Now running: T203 -- PROFILE=1 rewired to rocprofv3.** `/opt/rocm/bin/rocprofv3`
is in the image and the tree carries `VLLM_NVTX_SCOPES_FOR_PROFILING` hooks for
it. The server process is now wrapped in
`rocprofv3 --kernel-trace --stats -d $RESULT_DIR/rocprof -o k3 --output-format csv`.
Kernel trace only -- the question is which GPU kernels own the decode step, not
which host calls happened. Same workload cap: 72 prompts (one wave), OSL 256.

Known risk being taken: cudagraph capture of the 1..80 ladder generates a very
large kernel count, so the CSV may be big. If it is unmanageable the next
iteration attaches with `rocprofv3-attach` after `wait_for_server_ready` instead
of wrapping the whole process.

**C1/C52 config split, settled:**

| | C1 | C72 |
|---|--:|--:|
| chunk | **8192** (p99 TPOT 12.31 -> 9.18 ms) | 16384 (flat either way) |
| mns | 8 | 80 |
| dcp | off | 8 |

**`kimi-k3-vllm:v4` is built** (35.6 GB, 0 failed hunks): base nightly +
Hyukjoon's c16_c52 overlay + the 5-file pr-stack, all baked, one overlay for
every concurrency, no runtime patching. Launcher short-circuits on
`/etc/k3-image-manifest`. BLOCKED from CI use: `runners/launch_mi355x-amd.sh`
line 36 `{{index .RepoDigests 0}}` aborts on a local-only tag, and that file is
outside the standing edit bounds -- needs explicit approval.

**Queue (image-recipe validation, then registry push):**

1. **T204 -- C1 GSM8K limit 200 on the single c16_c52 overlay.** RUNNING.
   Numerics gate for retiring the c1 cut.
2. **C1 perf on `kimi-k3-vllm:v4`** -- fixed-len, chunk 8192, gmu 0.9. Compare
   to T201: TPOT 8.92 ms mean / 9.18 ms p99.
3. **C72 on `kimi-k3-vllm:v4`** -- the T195 config with zero runtime patching.
   Expect `[k3-overlay] baked into image`, `[pr-stack] baked into image`, and
   10,632 +- noise.
4. **Then push `aigmkt/kimi-k3-vllm:v4`** -- APPROVED by the user, overriding
   the standing no-Docker-Hub-push bound. Tag matches the local tag and the
   baked labels (`k3.overlay=c16_c52-all-conc`, `k3.pr-stack=53940,50813`).
   Do NOT reuse the `v2` name: local `kimi-k3-vllm:v2` (`484756f4...`) is a
   different, older build (overlay + #53940 only, no manifest), and
   `aigmkt/kimi-k3-vllm:latest` is a third thing again. Mutable/mismatched tags
   already cost this effort several wrong conclusions.
   Needs `docker login` for the `aigmkt` org on b23_07.

**Deferred / dropped:**
- C72 GSM8K: DROPPED as redundant. T194 (GSM8K 0.99) and T195 have byte-identical
  server config -- same image, overlay, 5-file pr-stack, mns 80, ladder 1..80,
  chunk 16384, dcp 8, gmu 0.9. Only CONC differs, and mns clamps to 80 at both
  64 and 72, so even the capture geometry matches. There is no numerics
  difference for GSM8K to find.
- Profiling: PARKED. Both in-tree paths are dead (T202 torch profiler absent,
  T203 rocprofv3 queue interposition deadlocks RCCL during capture).
- Attribution: C64 without #50813 (T193 moved two variables).

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

## Image validation COMPLETE. Push pending docker login.

| gate | result | run |
|---|---|---|
| C1 latency | TPOT 9.69 ms mean / 11.70 p99 | T205 |
| C72 throughput | **10,646 tok/s/GPU**, err 0.31% | T206 |
| GSM8K @ C72 | **0.995** flexible + strict | T207 |

`kimi-k3-vllm:v4` = `nightly-46638857` + c16_c52 overlay + 4-file pr-stack
(#53940), baked, no runtime patching. Push as **`aigmkt/kimi-k3-vllm:v4`**
(user-approved, overrides the standing no-Docker-Hub-push bound). Blocked only
on `docker login` for the `aigmkt` org on b23_07.

Next while that is pending: the C1 tail-variance hunt (below).

## Overlay + PR ablation plan (A/B/C/D/E) — prune to minimum

Goal: find the smallest subset that still delivers 10,632 at C72, so there is
less to carry forward and less to upstream.

**Mechanically ready.** `k3_patches/overlay_split/` partitions the 264 KB
overlay by concern. Recombined it is byte-identical (264,116 B, 199 hunks, 34
files). All five groups verified to apply **independently** on a pristine
46638857 tree and in sequence, so any subset is a legal experiment.

| grp | scope | files | bytes |
|---|---|--:|--:|
| A | dcp a2a buffer pool (`v1/attention/ops/dcp.py`) | 1 | 3,555 |
| B | spec-decode cudagraph (`dflash/cudagraph`, `speculator`, `config/speculative`) | 3 | 4,434 |
| C | kv-offload + cache manager (`v1/core/*`, `simple_kv_offload`, `kv_cache_*`) | 9 | 76,944 |
| D | ROCm AITER MLA (`backends/mla/*`, `rocm_aiter_mla_reduce`, `mla_attention`) | 5 | 76,806 |
| E | Kimi-K3 model path (`models/kimi_k3/**`, MoE runner, `envs`, `platforms/rocm`) | 16 | 102,377 |

Driver: `K3_OVERLAY_SPLIT=1 OVERLAY_GROUPS=ABCDE`. Gate line
`[overlay-split] groups=... applied=N failed:...`. `ABCDE` == the monolith.

**Sequence, all at C72 (the peak), one variable each:**

| # | groups | pr-stack | tests |
|---|---|---|---|
| 0 | ABCDE | 5 | control — must reproduce 10,632 via the split path |
| 1 | _BCDE | 5 | is A (dcp buffer pool) needed for throughput? |
| 2 | A_CDE | 5 | is B needed at C72? (spec is OFF above C4 — expect droppable) |
| 3 | AB_DE | 5 | is C needed? largest generic group |
| 4 | ABC_E | 5 | is D needed? |
| 5 | ABCD_ | 5 | is E needed? |
| 6 | best | 4 | drop #50813 — also the T193 attribution gap |
| 7 | best | 0 | drop the pr-stack entirely |

Then recombine the neutral groups into one minimal set and re-verify at C72,
plus a GSM8K limit-200 gate, since dropping any group is numerics-affecting.

**Expected, to be falsified by measurement:** B should drop free at C72 (MTP is
off above C4). A is a correctness fix for a fault we have not hit on this stack,
so it may measure neutral but should still be kept. D and E are the kernel path
and are unlikely to be droppable. C is the interesting one — 9 files, generic,
and the offload it manages is worth ~1-2% by three earlier A/Bs.

Cost: ~8 runs at C72. Every prior top-of-curve intuition on this stack has been
wrong, so no group is dropped on reasoning alone.

## Decision: C1 is CLOSED. v4 is the single image for all concurrencies.

User call: accept the v4 C1 result as-is and do no further C1 work.
**C1 final: TPOT 9.69 ms mean / 9.13 median / 11.70 p99** (T205, v4, chunk 8192,
gmu 0.9, dcp off, mns 8, ladder 1..72). The `c1` overlay cut is retired.

This supersedes the earlier plan to keep a per-concurrency overlay. The ~8.6%
mean / 27% p99 TPOT the `c1` cut would have bought is knowingly given up in
exchange for one image, one recipe, one thing to validate and ship. Recorded so
nobody re-derives the regression later and treats it as a bug:

| | TPOT mean | p99 |
|---|--:|--:|
| c1 cut, runtime patch (T201) | 8.92 ms | 9.18 ms |
| **c16_c52, baked v4 (T205) -- ACCEPTED** | **9.69 ms** | **11.70 ms** |

Context worth keeping: Hyukjoon confirmed the two cuts are **deliberately
different PR selections tuned for high vs low concurrency**, not old-vs-new
snapshots. So the regression is explained, not mysterious -- we are running the
high-conc selection at C1 by choice.

No `v4-c1` variant. No further C1 runs.

Open question for Hyukjoon: the PR list is **not** in the patch file we hold
(`90f975fa...f64dcc0`, 264,116 B) -- grepped for `#NNNNN`, `PR NNNNN`, `pull/`,
`github.com/`, `cherry-pick`, `backport`, `Signed-off-by`, zero hits. The only
references anywhere are in the c1 cut: `ATOM#1752` (a port source) and
`vLLM PR 40710` (cited as a do-not-use warning). Need either the manifest that
accompanies the patch, or a newer revision whose header carries the list.

### Parked: C1 chunk-size sweep on v4 (user: "experiment that later")

Not queued. Recorded so the prior work is not repeated.

**Known, on the retired `c1` cut** (T200 vs T201, identical workload):

| chunk | TPOT mean | p99 | TTFT mean |
|--:|--:|--:|--:|
| 16384 | 9.70 ms | 12.31 ms | 13,664 ms |
| **8192** | **8.92 ms** | **9.18 ms** | 15,772 ms |

Smaller chunk stops decode stalling behind an oversized prefill slice; the cost
is prefill throughput. Mechanism is real at C1 because there is exactly one
request, so slice size *is* the interleaving policy (unlike C72, where T199
measured chunk flat in both directions).

**Unexplored:** 4096 and 2048, and the whole sweep on the v4 / `c16_c52` stack
where C1 currently sits at 9.69 ms mean / 11.70 ms p99 with chunk 8192. If the
same tail mechanism holds, a smaller chunk may recover part of the 27% p99 gap
accepted when the c1 cut was retired -- without needing a second image.

Cost: ~15 min per point (C1 fixed-len, 4 prompts).

## C1 tail-variance hunt on v4 (reopened by user)

Target: kill the p50->p99.9 spread. v4 spans **2.64 ms** (9.13 -> 11.77); the
retired c1 cut spanned **0.22 ms**. Typical tokens are already fine (median
9.13 vs 8.96, +1.9%) -- this is purely a tail problem.

Current C1 config on v4: mns 8, ladder 1..72, gmu 0.9, chunk 8192, dcp off,
offload dram, SPEC_ROWS 9.

**Runs, one variable each, ~15 min per point (C1 fixed-len, 4 prompts):**

| # | change | why this is a tail suspect |
|---|---|---|
| 1 | **mns 8 -> 1** (ladder 72 -> 9) | At C1 the real batch is **one sequence = 9 spec rows**, every step. We capture 72 graph sizes and use ~9. Sizes 10..72 are dead capture, and any step that lands on a mismatched bucket pays a re-dispatch. Making the ladder exactly the used shape is the cleanest way to remove graph-selection jitter. Ladder invariant still holds: 1 x 9 = 9. |
| 2 | **offload dram -> none** | At C72 the dram offload is worth 1-2% because it keeps the batch full. At C1 there is no batch to keep full, so the CPU<->GPU KV traffic on a 214k prompt is pure added latency, and it is **bursty** -- exactly the shape that produces a tail rather than a uniform slowdown. |
| 3 | **chunk 8192 -> 4096** | Proven mechanism on the c1 cut (16384 -> 8192 took p99 12.31 -> 9.18). Never swept below 8192, never swept at all on v4. |
| 4 | **gmu 0.9 -> 0.92** | Weakest prior: KV is tiny at C1 so headroom should not bind. Cheap to test, and it is the one knob whose C1 value was never actually measured (0.92 was assumed, then dropped). |

Run 1 and 2 first -- they are the two with a mechanism that predicts a *tail*
specifically rather than a uniform shift. 3 and 4 are follow-ups.

If none of them close it, the remaining explanation is that the `c16_c52` cut
simply lacks a low-conc code path the `c1` cut had, and the gap is structural at
9.69 ms mean unless we carry two overlays.

Blocked until T206 (C72 on v4) finishes -- GPUs busy.

## Bare new-nightly evaluation (no overlay, no pr-stack)

Base: `vllm/vllm-openai-rocm:nightly-7c5dc571cbd1064ecc8a9b1045637ff647aa22cb`
(built 2026-09-01 05:31). **This is the first nightly containing #51705** --
merged 2026-08-31 17:42, five days after our v4 base `46638857` was cut.
Ancestry-checked against merge commit `dbb7fffddb`: 46638857 is *behind*,
7c5dc571 is *ahead*.

`REQUIRE_K3_OVERLAY=0`, `APPLY_PR_STACK` inert (overlay never applies, so the
pr-stack gate is skipped). Nothing external is added.

**Why this matters:** if a stock public nightly reaches useful throughput with
zero patches, the distribution problem in UPSTREAM-STATUS.md largely dissolves --
no carried overlay, no self-built image, reproducible by anyone.

**Order (cheap-fail-first, per user):**

| # | run | what it answers |
|---|---|---|
| 1 | **C1 functional** (TEST=1, agentic-band lengths, 4 prompts) | does it serve at all with MTP + no patches? |
| 2 | **C52 functional** | does it serve at high conc with DCP=8? |
| 3 | **GSM8K limit 200 @ C52 ONLY** | numerics without the overlay. C<=4 is structurally invalid for GSM8K (synthetic acceptance), so C52 is the only valid accuracy point. |
| 4 | **Agentic perf @ C52** | vs aigmkt ~8,204 and nightly+overlay 8,685 (T189) |
| 5 | **Agentic perf @ C1** | vs the C1 ledger; MTP on, DCP off |

Perf comparison is at **C1 and C52**, not C72 -- those are the points with
aigmkt-era history to compare against:

| config | tok/s/GPU | stack |
|---|--:|---|
| aigmkt C52 | ~8,204 | aigmkt image |
| aigmkt C60 (aigmkt best) | 8,342 | aigmkt image |
| nightly-46638857 + overlay, C52 | 8,685 | T189 |
| SA C52 | 8,953 | nightly + overlay |

If bare 7c5dc571 lands near 8k at C52 it matches the aigmkt image with **zero**
external patches, which is the result that would matter.

T212 (agentic C72 on this base) was dispatched first by mistake and cancelled
~20 min in; it is superseded by step 4.

**Known gaps vs the overlay, expect these to cost something:** `amd/mla.py` and
`v1/attention/ops/rocm_aiter_mla_reduce.py` are new files the overlay adds that
do not exist upstream at any ref, and the `_DCPA2ABufferPool` fix is in neither
nightly (`dcp.py` is byte-identical, 44,932 B, in both).


## Bare-nightly starvation bisect (T218+)

C1 agentic passes (1,222 tok/s/GPU, T217). C52 agentic starves (T216). The
threshold is somewhere between, and locating it says which mechanism is at
fault:

- **If it starves at low conc too (C8/C16)** -> the prefix-cache path is broken
  outright under DCP, and the failure is not about load.
- **If it degrades gradually** -> it is a capacity/scheduling limit, and the
  overlay's KV-offload group is buying headroom rather than correctness.

Config held at T216's (mns 80, chunk 16384, dcp 8) so **concurrency is the only
variable**. Note mns 80 at C16 is deliberately *not* the `old.sa.sh` value (20);
matching T216 matters more here than tuning.

| run | conc | status |
|---|---|---|
| T218 | 16 | dispatched |
| next | 32 or 8 | bisect from T218's result |

Reference: C52 starved with 38/107 warmup returned in 19.5 min, errors=0.

## Bare-nightly degradation curve (mns 80, chunk 16384, dcp 8)

| conc | tok/s/GPU | err | run |
|---|--:|--:|---|
| 1 | 1,222 | 0.54% | T217 |
| 16 | 3,591 | 10.05% | T218 |
| **32** | **?** | **?** | **T220 running** |
| 52 | starves | 100% | T216 |

C52 is not being re-run -- T216 already measured it with mns 80, and T219
eliminated the mns confound in the protective direction. C32 locates the usable
ceiling for a stock upstream image, which is the practical question behind
"can we ship without the overlay".


## Bare-nightly evaluation: CLOSED

| conc | tok/s/GPU | err | verdict |
|---|--:|--:|---|
| 1 | 1,222 | 0.54% | clean; C1 fixed-len TPOT 8.52 ms BEATS the patched stack |
| 16 | 3,591 | 10.05% | peak, but at aiperf's abort threshold |
| 32 | 1,843 | 16.92% | throughput HALVES, errors climb |
| 52 | starves | 100% | no result |

GSM8K 0.985 at C52 (T215) -- numerics are fine; the failure is throughput/
scheduling, not correctness.

**Conclusion: stock upstream is not shippable for the agentic workload.** Usable
ceiling ~C16. The overlay's value = (clean C72 @ 10,646) vs (C16 @ 3,591 with
10% errors).

**Open, in priority order:**
1. Why does it break between C16 and C32? Candidate: hybrid-cache geometry
   (#53917, still open) under DCP with many lanes.
2. v4 remains the shipping image. `aigmkt/kimi-k3-vllm:v4` is pushed.
3. Overlay A/B/C/D/E ablation on the v4 base -- still unrun, and now better
   motivated: it would say which group buys the C16->C72 range.


## Overlay ablation: CLOSED (T221-T226)

Control ABCDE = 10,756 tok/s/GPU err 0.18%.

| grp | leave-one-out | detachable |
|---|---|---|
| A | 10,719 (-0.34%) | yes -- keep anyway, correctness fix |
| B | 10,747 (-0.08%) | yes, at C>4 only |
| C | won't start | no (E imports from it) |
| D | won't start | no (DCP needs its AiterMLAImpl) |
| E | won't start | no (defines fused_sigmoid_gate) |

**Nothing meaningful to prune.** C+D+E = 256 KB of 264 KB and are one coupled
unit. A+B = 8 KB and are free either way.

Shipping set: all five. `kimi-k3-vllm:v4` already carries exactly this.

**Open items:**
1. C1 chunk sweep on v4 (parked, user's call) -- 4096/2048 vs 8192.
2. Upstreaming: A and B filable standalone; C/D/E must go as one series.
   Needs Hyukjoon's sign-off.
3. Why bare upstream collapses above C16 on the agentic workload (T216-T220)
   remains unexplained at the code level. The ablation says the overlay is
   needed but not which line fixes it.


## C1 chunk sweep: CLOSED (8192)

| chunk | mean TPOT | p99 | TTFT |
|--:|--:|--:|--:|
| 16384 | 9.69 | 11.70 | 14,971 |
| **8192** | **9.06** | **9.31** | 15,310 |
| 4096 | 9.04 | 9.31 | 19,460 |
| 2048 | 9.18 | 9.70 | 27,333 |

8192 is the floor. 4096 ties on TPOT but costs 27% TTFT; 2048 is worse on both.
No further chunk runs needed at either end of the range (C72 was flat, T199).

**All config levers are now closed.** conc, mns, chunk, gmu, offload at both C1
and C72; overlay ablation A-E; bare-upstream curve C1-C52.

**Remaining open, none of which are config tuning:**
1. Upstreaming (needs Hyukjoon): A and B filable standalone, C/D/E as one series.
2. Code-level reason bare upstream collapses above C16 on agentic.
3. Profiling for the last 14.9% -- both in-tree paths dead (T202 torch profiler
   absent, T203 rocprofv3 deadlocks capture). Would need rocprofv3-attach or an
   external tool.

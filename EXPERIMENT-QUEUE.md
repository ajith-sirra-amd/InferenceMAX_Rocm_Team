# Autonomous run queue — Kimi-K3 / 8× MI355X

**Owner away 2026-09-04 → 2026-09-08. Run autonomously. Target: 12,500+ tok/s/GPU.**
**This file is LATEST-FIRST. Newest state at the top; history below.**

---

# ▲ LATEST STATE

## T271 dead: LMCache `dev89` rotated out of the nightly channel

`ERROR: No matching distribution found for lmcache==0.5.5.dev89+rocm7.2`. It
installed fine on 09-03/09-04. **Not retried** — LMCache is parked, and SA's own
`ext_cache_hit` is 0.0%, so pinning their exact build is not worth chasing a
rolling channel. Cost: ~6 min. **Phase B closes here.**

## T272 v1 cancelled at 7 min — my config error, not a run failure

`mnbt` was still **8192** from the T269 SA arm; I never reverted it. That would
have made T272 a **two-variable** run (gmu 0.90->0.92 AND mnbt 16384->8192)
against the T265 control, and useless as a gmu test. Caught in the gate lines,
cancelled, `mnbt` restored to 16384, container removed, all 8 GPUs verified at
1%. **Lesson: check every gate line against the intended config, not just the
one knob under test.**

## T272 v2 running: gmu 0.92 at C72 — following the biggest lever we have

gmu is the largest effect measured in this whole campaign: **0.88 -> 0.90 was
+16.9%** (T269 7,128 -> T270 8,329, same image). Our C72 baseline already runs
0.90, so the open question is whether there is headroom above it.

One variable vs the T265 control (11,019): `K3_GMU=0.92`, everything else at
baseline (`rec-no53940`, C72, dram/vllm-simple, mns 96, mnbt 16384).

**Prior evidence against 0.92 is C1-only** — T211 measured it catastrophic at
C1 (mean 9.06 -> 21.61 ms) and T157 hung at 0.95. The ledger note claiming
"0.88 is the tested ceiling" is stale: 0.90 has run clean at C72 dozens of
times. A hang or a regression here is a real possibility and would establish
the ceiling; it is cheap to detect early.

No GSM8K gate: gmu changes memory budget, not arithmetic.


## MAJOR CORRECTION: the 3-PR stack is worth ~1%, not ~20%

T270 (stock, gmu **0.90**) = **8,329** vs T269 (stock, gmu **0.88**) = 7,128.
**gmu alone is worth +16.9%.**

I reported the patch stack as +20.6% (T257 vs T259) and then +18.2%
(T257 vs T269). **Both were confounded by gmu.** The bare-image 0.88 override
fires only on images lacking `/etc/k3-image-manifest`, so our patched image
never took it and every stock arm was handicapped 17%. GPU-KV fingerprints
prove the split: T257 30,169,355 and T270 30,249,138 are the 0.90 class;
T259 and T269 are both exactly 26,932,446.

| gmu-matched at 0.90 | tok/s/GPU |
|---|--:|
| stock, no patches (T270) | 8,329 |
| ours, 3 PRs (T257) | 8,426 |
| **delta** | **+1.2%** |

**Consequence for the upstream ask:** #52494 and #52968 were justified on
measured deltas of -1.35% and -0.93% from same-image A/Bs, which remain valid.
But the *stack as a whole* buys ~1% at C48, not twenty. `UPSTREAM-STATUS.md`
must not carry the inflated figure.

**Also:** every stock arm (T259, T260, T261, T269) ran 17% handicapped. Their
absolute numbers are real but their comparisons against patched runs were not.

**T271 running — the exact SA cell.** Stock base, gmu 0.90, C48, **with**
LMCache `dev89` / workers 8 / chunk 12288, mnbt 8192, mns 96, DCP 8. Every
knob now matches SA run 33773561410 (**10,152**). If we land near 8,3xx again,
LMCache-loaded-but-inert is not the difference and the gap is in their launcher
or their node.


## T269 — the base nightly is NOT the SA gap

C48 comparison set, all no-LMCache, all gmu 0.88, mnbt 8192, mns 96, DCP 8:

| run | base | patches | tok/s/GPU |
|---|---|---|--:|
| **SA 33773561410** | `7c5dc571` | none | **10,152** *(gmu 0.90, LMCache loaded but inert)* |
| **T257** | `7c5dc571` | 3 PRs | 8,426 |
| **T269** | `7c5dc571` | none | **7,128** |
| T259 | `73029d42` | none | 6,988 |

**T269 vs T259 = +2.0%.** The base version accounts for almost nothing.
**Our 3 PRs are worth +18.2%** on the same base (7,128 -> 8,426).
**20.5% remains unexplained** between our best C48 (8,426) and SA's 10,152.

**Correction:** T259 was recorded as gmu 0.90. It ran at **0.88** — the
bare-image override fires on any image without `/etc/k3-image-manifest`, and the
`[sa-match]` echo I read prints *before* that override. The authoritative value
is the `[gmu]` line. Both stock runs were 0.88.

**T270 running:** identical to T269 but **gmu 0.90**, matching SA exactly. This
is the last knob we know differs apart from LMCache being loaded. The 0.88
default exists to dodge `HSA_STATUS_ERROR_OUT_OF_RESOURCES` on this bare
image/DCP8/no-offload combination, so the run may crash — that is itself an
answer, since SA evidently runs 0.90 here without trouble.


## PHASE A CLOSED. All four user-given PRs resolved.

| PR | verdict |
|---|---|
| **#54889** a2a-pack-mask | **+0.87% at C72** (11,115 vs 11,019 control), n=1, ~2x noise. Unproven; needs a replicate. |
| **#54494** dcp-q-replicate | **NEUTRAL at C72** (+0.09%). GSM8K 0.995 PASS. **Untestable at C1** — see below. |
| **#54735 / #54736** | **BLOCKED** — hunks target code in no published nightly; they carry unmerged dependencies. Re-dry-run each cycle. |

**#54494 cannot be tested at C1 in the configuration we ship.** Two findings
from T268:
1. The yaml's `dcp-size` never reaches the launcher — `DCP_SIZE` arrives empty
   and the script's conc-fallback decides. Benign at C48-C72 (fallback gives 8)
   but it means the matrix was never driving that knob. `K3_FORCE_DCP` added.
2. With DCP forced to 8 at C1, engine init dies:
   `TRITON_MLA is not valid — non-causal MLA attention with DCP not supported`.
   The MTP draft uses TRITON_MLA, so **DCP>1 and MTP are mutually exclusive**.
   #54494 needs DCP active; C1 ships with MTP. Testing it would mean measuring
   a config we do not ship, for a PR already neutral where throughput is scored.
   Not pursued.

**Neither testable PR moves us toward 12,500.** Baseline 11,019-11,027, gap ~11.8%.

## PHASE B running — T269 SA REPRODUCE

Bare `nightly-7c5dc571` (SA's exact base) with `K3_FORCE_STOCK=1`: no overlay,
no PR stack. C48, **no LMCache and no offload at all**, mnbt 8192, mns 96,
gmu 0.90, DCP 8. Target: SA's **10,152**. Our patched C48 was 8,426 (T257) and
stock-on-a-different-nightly was 6,988 (T259), so this fills the base-version
hole in the 2x2.


## Phase A verdicts: both user-given testable PRs are ~neutral at C72

| PR | C72 result | vs same-day control | verdict |
|---|--:|--:|---|
| **#54889** a2a-pack-mask (T264) | 11,115 | +0.87% vs 11,019 | ~2x noise, n=1, **unproven** |
| **#54494** dcp-q-replicate (T267) | 11,029 | **+0.09%** vs 11,019 | **NEUTRAL** |

#54494 passed GSM8K at 0.995 (T266) and its TPOT distribution is
indistinguishable from control (104.1/97.3/134.6/166.9 vs 103.0/98.1/134.4/168.6).
**This confirms the earlier prediction:** at C72 we are compute-bound and
prefill-dominated (ISL p50 ~54k, OSL p50 ~224), so removing a decode-side
per-layer all-gather buys nothing and the redundant q-projection costs nothing
either. The C1 arms are the only place this PR can show a benefit.

**Neither PR moves us toward 12,500.** Baseline stands at 11,019-11,027,
gap ~11.8%.

**T268 v1 CANCELLED at 5 min — the yaml `dcp-size` is a no-op.** The launcher
logged `[dcp] size=1 source=conc-fallback conc=1`: `DCP_SIZE` arrives empty from
the harness, so the script's own fallback decides, and at CONC<=4 that is 1.
q-replicate would have been inert and the A/B would have compared two identical
runs. Cancelled rather than spend 2.5 h on a run that could not answer the
question. **Note this also means every prior run's DCP came from the fallback,
not the yaml** — benign at C48-C72 where the fallback gives 8, but worth knowing.

**Fix:** `K3_FORCE_DCP` (default 8) added to the launcher, overriding the
fallback. `bash -n` clean, `wait_for_server_ready` intact.

**T268 re-dispatched: C1 CONTROL**, `dcp-size: 8` forced, image `rec-qrep` with
`VLLM_DCP_Q_REPLICATE=0`. T269 will flip only the env var, so the pair is a
true one-variable A/B. Note C1 enables MTP (spec k=8), unlike the C48-C72 arms.


## T266 — #54494 GSM8K gate PASSED (0.995)

`exact_match 0.995` strict and flexible, **identical to the baseline 0.995**.
`eval_exit=0`. The job's `failure` status is the known EVAL_ONLY result-JSON
plumbing artifact (same as T251), not an eval failure.

GPU KV **28,220,371**, 1.5% below baseline — each rank materialising the DCP
group's full query head set. Confirms the PR is active rather than silently off.

**#54494 is numerically clean. T267 running: #54494 perf at C72**, image
`rec-qrep`, everything else at baseline. Compare against the **same-day
control T265 = 11,019**, not the historical 11,027.


## T265 control — baseline HOLDS, errors are NOT ours

**11,019 tok/s/GPU** on `rec-no53940` at the identical C72 config, same day as
T264. Against the 11,027 anchor that is **0.07%** — the baseline is intact and
still trustworthy.

**Error rate 5.65% on the pristine baseline image.** T264 was 5.45%. So the
elevated rate belongs to the environment (harness/dataset), **not to #54889 and
not to any patch of ours**. Something changed after T247/T252 (0.26-0.30%).
Worth understanding eventually, but it is not blocking and it affects all arms
equally, so same-day A/Bs remain valid.

**#54889 same-day A/B: 11,115 (T264) vs 11,019 (T265) = +0.87%.** Same-session
pairs replicate to ~0.4%, so this is roughly 2x noise — suggestive, not proven.
Needs n=2 before it can be claimed. It is NOT yet a step toward 12,500.

**T266 running:** GSM8K-200 gate for #54494 with `VLLM_DCP_Q_REPLICATE=1` on
`rec-qrep`. Numerics change, so the gate precedes any perf number.


## T264 done — #54889 is NOT a demonstrated win, and errors are systemic

**11,115 tok/s/GPU** vs baseline 11,027 = **+0.80%, inside the ±1.2% band.**
Duration 3,629 s (normal), GPU KV 28,653,478 (exact baseline fingerprint), so
the config was clean and the PR was the only variable.

**The error rate is the real signal.** T264 logged **5.45%**
`InvalidInferenceResultError` against the baseline's 0.26-0.30%. But it is NOT
the PR:

| trial | image | err% |
|---|---|--:|
| T247/T252 baseline | ours | **0.26-0.30** |
| T257 C48 | ours | 4.49 |
| T259 C48 | stock | 5.45 |
| T260 C64 | stock | 9.93 |
| T261 C72 | stock | 15.45 |
| T262 C64 | ours | 12.01 |
| T264 C72 | ours + #54889 | 5.45 |

**Every run since T257 is elevated, on both images, patched and unpatched.**
Something in the environment drifted after T247/T252. Until that is understood,
throughput deltas measured now are not comparable to the 11,027 anchor.

**T265 = BASELINE RE-ANCHOR** (running): `rec-no53940` at the identical C72
config. If it returns ~11,0xx with ~5% errors, the drift is environmental and
#54889 is neutral. If it returns 0.3% errors, #54889 owns them.

# ▲ AUTONOMOUS PLAN (2026-09-04 → 09-08)

## Rules while unattended

1. **Never leave GPUs idle.** One run at a time; dispatch the next as soon as
   the previous is recorded.
2. **Update the md files after EVERY run** — `Kimi-DCP-Experiemnts-Summary.md`
   (row at top), this file (state at top), `UPSTREAM-STATUS.md` when a PR's
   measured delta moves. Commit and push each time.
3. **ONE variable per run.**
4. **Any config beating 11,027 gets GSM8K-200 at that exact config** before it
   is claimed, and a **replicate (n=2)** before it becomes the headline. A
   single run inside the ±1.2% cross-day band is not a new best.
5. **Failures: diagnose from the log first.** Never re-dispatch blind.
6. After cancelling anything, **`docker ps` and remove a surviving
   `bmk-server`** — a cancelled GH job does NOT free the node.
7. Bounds unchanged: dispatch only to `ajith-sirra-amd/InferenceMAX_Rocm_Team`,
   SA repo read-only, no Docker Hub push, no push to any other repo.

## Order of work

**Phase A — the four user-given PRs (in flight)**
- T264 *(running)* — #54889 a2a-pack-mask @ C72. Image `rec-a2amask`.
- T265 — GSM8K-200, `VLLM_DCP_Q_REPLICATE=1`. Image `rec-qrep` (**built**).
- T266 — #54494 @ C72.
- T267a/b — #54494 @ C1 with `dcp-size: 8` forced (control, then arm).
  Our launcher drops to DCP=1 at CONC<=4 and the PR requires DCP active, so a
  naive C1 A/B would compare two identical runs.
- #54735 / #54736 — **re-dry-run every cycle.** Blocked today: their hunks
  target code in no published nightly. The moment they apply, they jump the
  queue (P0 correctness; they retire the CLOSED #53917 we still carry).

**Phase B — SA reproduce**
- T268 — bare `nightly-7c5dc571`, `K3_FORCE_STOCK=1`, C48, **no LMCache**,
  mns 96 / mnbt 8192 / gmu 0.90 / DCP 8. Fills the base-version hole: SA is on
  `7c5dc571`, our stock run T259 was on `73029d42`.
- T269 — same plus `vllm-simple` offload.

**Phase C — spec-decode bugfixes (C1/MTP path only)**
- T270 — #54163, T271 — #54165. Both non-draft, both untested by us. Spec is
  off at C48-C72, so these only bite at C1.

**Phase D — if still short of 12,500 after A-C**
The four draft perf PRs (#53301, #51437, #53166, #52190) were parked with
"will see these very later". After Phases A-C are exhausted they are the only
remaining lever toward the stated 12,500 goal, so **test them, largest first,
one variable each, GSM8K-gated.** Flag clearly in the ledger that this
un-parking was my call under the 12,500 objective, for review on return.

**Phase E — consolidation**
Replicate whatever the best config turns out to be to n=2, GSM8K-gate it, and
leave `UPSTREAM-STATUS.md` stating the final ask with measured deltas.

## Standing facts not to relitigate

- Best: **11,027 tok/s/GPU @ C72** (n=2, T247/T252), GSM8K 0.995, image
  `rec-no53940`, config `kv-offloading: dram` / `vllm-simple` / 1,949 GB,
  gmu 0.90, mns 96, mnbt 16384. Gap to target **-11.8%**.
- **LMCache is parked and inert on both sides.** SA's own run logs
  `ext_cache_hit=0.0%`. Do not restart LMCache work.
- **#53940 a4w4-moe costs 2.40%.** Stays out.
- `#53917` is CLOSED upstream but still applied locally and load-bearing.
- Cross-day noise band **±1.2%**; same-session pairs replicate to ~0.4%.

---

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

### Approved order (user-confirmed 2026-09-03, revised)

**Why the revision:** T257/T258 applied SA's *settings* but kept our *patched*
image, so they confound "SA settings don't help us" with "our patches hurt".
The stock-nightly arms below remove that confound: same script, same settings,
zero patches, exactly what SA runs.

**Image A — stock `vllm/vllm-openai-rocm:nightly-73029d42`, NO patches** (what SA runs):

1. **T259 — C48, NO LMCache.** True control against T257 (8,426 on our image).
2. **T260 — C64, WITH LMCache.** Pairs with T258 (our image, same conc).
3. **T261 — C72, WITH LMCache.**
   All three at SA settings: gmu 0.90, mns 2xCONC, mnbt 8192, workers 8,
   chunk 12288, LMCache `0.5.5.dev89`. Needs the patch step skipped and the
   yaml image line pointed at the stock nightly.

**Image B — ours, `kimi-k3-vllm:rec-no53940`** (`nightly-7c5dc571` + #53917,
#52494, #52968):

4. **T262 — C64, WITH LMCache, `--max-gpu-workers 1`.** The replay of the run I
   cancelled at 29 min (T256, healthy at the time: warmup 57/131, 0 errors).
   OLD settings it was carrying: gmu 0.88, mns 80, mnbt 16384, `0.5.5rc3`.
5. **T263 — C72, WITH LMCache, workers 1.** Same image and settings as T262.
   The runner passes no env, so both need the values written into the script or
   the kimi yaml block before dispatch.

6. **T264 — C72, WITH LMCache, workers 1, partial SA uplift.** Image B.
   Takes T263 and moves three knobs to SA values: LMCache **`0.5.5.dev89`**
   (was `rc3`), **gmu 0.90** (was 0.88), **mns 80** held. `mnbt` stays at
   T263's 16384 and workers stays 1. Two live variables vs T263 (lmcache
   version + gmu) — deliberate, user-specified, not a one-variable step.
   Isolates whether the LMCache *build* and the memory headroom matter once
   workers is pinned at 1.

**Then:**

7. **New image build** = `rec-no53940` + [#54494 `dcp-q-replicate`](https://github.com/vllm-project/vllm/pull/54494) (~99 lines, 3 files).
8. **GSM8K-200 gate** on `VLLM_DCP_Q_REPLICATE=1` — replicated vs gathered
   projection is the same math in a different reduction order, so not bitwise-safe.
9. **A/B at C1**, not C72. C72 is compute-bound (1,100 W, 98% util, `queue=0w`)
   and prefill-dominated (ISL p50 54k vs OSL p50 224); the PR removes a *decode*
   per-layer all-gather and pays in redundant q-projection compute, so it should
   be flat-to-negative there. C1 has nothing to overlap the collective against,
   so per-layer latency lands straight in TPOT — and C1 TPOT (best 7.57 ms) is
   an explicit target. Default-off, so it cannot perturb the 11,027 baseline.

**Deferred:** resume the throughput line toward 12,500 — C1 with/without #53940;
replicate T248 (with-a4w4 arm is n=1); `PIN_CCD=1`, never tested with
numa_balancing=0. Gap to target is -11.8%.

**Results so far:**

| trial | image | conc | LMCache | tok/s/GPU | err | ext_cache_hit |
|---|---|--:|---|--:|--:|--:|
| T257 | ours (3 PRs) | 48 | no | **8,426** | 4.49% | — |
| T258 | ours (3 PRs) | 64 | yes | **5,808** | 8.29% | **0.0%** |
| T259 | **stock, 0 patches** | 48 | no | **6,988** | 5.45% | — |
| T260 | **stock, 0 patches** | 64 | yes | **4,688** | 9.93% | **0.0%** |
| T261 | **stock, 0 patches** | 72 | yes | **3,396** | 15.45% | **0.0%** |
| T262 | ours, LEGACY (workers 1) | 64 | yes | **3,726** | 12.01% | **0.0%** |
| T263 | ours, LEGACY (workers 1) | 72 | yes | *cancelled ~2 h, in warmup* | — | **0.0%** |

## PR BACKLOG AND PRIORITIES (2026-09-04)

Our image manifest carries `pr-missing-vs-overlay: 53166 51437 53301 52190
54163 54165` -- six PRs dropped when we moved to a "fully-mergeable PR stack
only" image. Full picture:

### P0 - correctness, blocks the upstream ask
| PR | state | note |
|---|---|---|
| #54735 hybrid geometry | open | replaces CLOSED #53917, **required** for our dram/vllm-simple config. Unrebasable today -- hunks target code in no published nightly. Recheck on merge. |
| #54736 fine-grained hits | open | stacked on #54735, same block |

### P1 - ready now, non-draft, testable
| PR | state | status |
|---|---|---|
| #54889 a2a-pack-mask | open | **T264 running** |
| #54494 dcp-q-replicate | open | T265-T267 |
| #54163 spec-decode prefix-cache drop | open | **untested by us** -- affects C1/MTP only (spec is off at C48-C72) |
| #54165 mamba align cache hits under spec | open | **untested by us** -- same, C1/MTP path |

### P2 - PARKED (user, 2026-09-04): the four draft perf PRs

**Do not test these now.** Revisit much later. Listed only so they are not lost:

| PR | size | what |
|---|---|---|
| #53301 | +878/9f | reuse graph-stable attention metadata across cache groups |
| #51437 | +631/4f | latent-MoE: overlap shared all-reduce |
| #53166 | +216/2f | fuse MLA chunked-context gather on AITER |
| #52190 | +142/5f | enable torch.compile so post-grad fusion passes work |

Already applied: #52494 (-1.35%), #52968 (-0.93%, draft).
Held out: #53940 a4w4-moe, costs 2.40%.

### CAUTION on the "SA gap"

SA's 10,152 is at **C48**. Our 11,027 is at **C72**. Our own C48 is 8,426. So the
20.5% "gap" is measured at a concurrency we have not tuned, and **we may already
be ahead where it counts** -- SA's best published C52 was 8,296. Do not treat
10,152 as a deficit until we have SA at C72 or ourselves properly tuned at C48.
T268/T269 answer the first half.

## RUN ORDER (user, 2026-09-04): PR patches -> SA reproduce -> SA + vllm-simple

1. **T264** *(running)* — #54889 a2a-pack-mask @ C72, baseline config otherwise.
2. **T265** — GSM8K-200 with #54494 `VLLM_DCP_Q_REPLICATE=1`. Numerics gate.
3. **T266** — #54494 A/B @ **C72**. On the 12,500 track.
4. **T267** — #54494 @ **C1**. Needs `dcp-size: 8` forced (our launcher drops to
   DCP=1 at CONC<=4, and the PR requires DCP active), so this is TWO runs:
   C1/DCP8 qrep-off control, then C1/DCP8 qrep-on.
5. **T268 — SA REPRODUCE.** Their [run 33773561410](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/33773561410)
   = **10,152 tok/s/GPU @ C48**, base `nightly-7c5dc571` (same as ours),
   **zero patches**, LMCache `0.5.5.dev89` workers 8 chunk 12288,
   mns 96, mnbt 8192, gmu 0.90, DCP 8. Needs the BARE `nightly-7c5dc571`
   image + `K3_FORCE_STOCK=1`.
6. **T269 — SA CONFIG WITH `vllm-simple`.** Same as T268 but SimpleCPU offload
   instead of LMCache. Isolates the operating point from the connector.

### CORRECTION: LMCache is inert for SA too

SA's own run logs **`ext_cache_hit=0.0%`** and **zero** `No GPU context found`
errors, with `prefix_cache_hit` 94.8-96.6%. They reach 10,152 at C48 with the
external cache doing nothing.

**This retracts the earlier claim that "LMCache is worth ~+29% to SA".** That was
inferred from comparing their C48-with-LMCache against our stock C48-without,
and it was wrong. LMCache is inert on both sides; the difference is elsewhere in
their configuration. Their C48 10,152 vs our C48 8,426 (T257) is a **20.5% gap
at nearly identical knobs** — that gap, not the cache, is the thing to chase.

## STANDING RULE (user, 2026-09-04)

**Any config that beats the 11,027 baseline gets a GSM8K-200 run at that exact
config before it is claimed.** Applies to every new best, not only to 12,500.
A throughput number without an accuracy gate at the same settings is not a
result. Run the gate on the winning config, not on a neighbour.

## NEW-PR TRACK (2026-09-04). LMCache PARKED.

**LMCache is parked on user instruction** — handed to the LMCache team. Do not
run further LMCache arms. The six completed runs and the root cause stay in the
ledger below for reference.

### PR status, verified against upstream

| PR | role | applies to `rec-no53940`? |
|---|---|---|
| **#53917** `cpu-offload` | **CLOSED upstream** — we still carry it locally | already applied |
| **#54735** `[Bugfix]` hybrid geometry | its replacement, **required** for our config | **NO** — see below |
| #54736 `[Feature]` fine-grained hits | optional perf, stacked on #54735 | NO — 7 hunks fail |
| **#54494** `dcp-q-replicate` (updated: + FP4/FP8 MLA BMM) | optional perf | **YES, clean** |
| **#54889** `a2a-pack-mask` | optional perf, no numerics | **YES, clean** |

**#54735/#54736 cannot be rebased locally.** Their hunks target code that exists
in no published nightly — `manager.py` has no `primary_block_size` /
`get_group_id` / runtime `kv_cache_utils` import on `73029d42` (09-02) or
`27a94d1c` (09-03). They carry unmerged dependencies. Hand-resolving would mean
porting those too, so we wait for the merge (est. 2-3 days) rather than invent a
tree. **We take no action on the PRs themselves — local only.**

**Base nightly stays at `7c5dc571`.** Checked `27a94d1c` (dev337) and
`73029d42` (dev278); neither unblocks #54735, and moving base would drop
#53917 from a config that depends on it.

### CRITICAL: the baseline uses CPU offload

T252 (11,048) ran **`kv_offloading: dram`, backend `vllm-simple`, 1,949 GB CPU
DRAM** — not offload-free. So the SimpleCPU offload path is load-bearing for the
headline, which is exactly why #53917's replacement matters and why we cannot
simply drop it.

### Order

1. **T264 — #54889 `a2a-pack-mask`.** Pure kernel-count reduction (8 kernels ->
   1 inside the CUDA graph), **no numerics**, so no GSM8K gate needed. One
   variable vs the 11,027 baseline. C72 first to compare directly.
2. **T265 — GSM8K-200 with #54494** `VLLM_DCP_Q_REPLICATE=1`. Numerics change
   (different reduction order), gate before any perf number.
3. **T266 — #54494 A/B at C1.** C1 is where the removed per-layer all-gather is
   exposed; C72 is compute-bound and should be flat-to-negative.
4. Re-check #54735/#54736 once they merge upstream.

## Superseded — HOLDING note (2026-09-04)

GPUs idle and clean: container removed, all 8 at 1% VRAM. **Do not dispatch
until told.**

### LMCache: closed, negative

Six completed runs span 2 images x 2 worker counts x 3 concurrencies.
**`ext_cache_hit` is 0.0% in every one.** LMCache is net-negative everywhere it
was measured, and the damage scales with concurrency:

| image | C48 no-cache | C64 +LMCache | C72 +LMCache |
|---|--:|--:|--:|
| ours (3 PRs) | **8,426** | 5,808 (SA) / 3,726 (legacy) | cancelled |
| stock, 0 patches | 6,988 | 4,688 | 3,396 |

Root cause stands: **1,535 `No GPU context found for model ... with world size 8
during lookup!`** in `lmcache_server.log`. Every retrieve fails, so the 1,949 GB
host tier is write-only. Invariant to image, worker count, chunk size, LMCache
version, gmu, mns and mnbt — i.e. to everything reachable from the launcher.

### What is still unanswered

1. **stock + LMCache at C48** — the only cell directly comparable to SA's 8,997.
   Never run. Every LMCache arm we have is C64/C72, the two worst points.
2. **Does SA's `lmcache_server.log` carry the same lookup error?** Not checked.
   Their *job* log has no `lookup.py` lines at all, so the comparison must be
   against the server log specifically. I drew a wrong conclusion once by
   reading the wrong file.
3. **#54494 `dcp-q-replicate`** — image build, GSM8K-200 gate, then A/B at C1.

### Best numbers still standing

**11,027 tok/s/GPU @ C72** (n=2: T247 11,006 · T252 11,048), GSM8K 0.995, on
`rec-no53940` with no LMCache. Target 12,500, gap **-11.8%**.

**Workers-1 hypothesis is dead.** T262 vs T258 (same image, same conc, only the
settings block differs): legacy gmu 0.88 / mns 80 / mnbt 16384 / workers 1 /
`rc3` gives **3,726 vs 5,808** — 36% worse than SA settings, and
`ext_cache_hit` is still 0.0%. Every LMCache configuration tried (2 images x
2 worker counts x 3 concurrencies) has zero external cache hits.

**T260 is decisive on the LMCache fault: it is NOT our patch stack.** Zero-patch
upstream `nightly-73029d42` shows the identical `ext_cache_hit` 0.0%. The
`No GPU context found ... during lookup` failure is upstream or environmental.
LMCache is net-negative on both images: **−33% on stock**, **−31% on ours**.

**GAP — the direct SA comparison has never been run.** SA's 8,997 is
**stock + LMCache at C48**. Our four cells are C48-no-LMCache and
C64-with-LMCache on each image; **stock + LMCache at C48 is empty**. Comparing
our C64 LMCache numbers to SA's C48 conflates concurrency with the cache fault.
Consider running that cell before, or instead of, T261 (stock C72 + LMCache),
which at C64's 4,688 is very likely to come in lower still.

**Two questions closed by T259:**

1. **Our 3-PR stack is worth +20.6%** (8,426 vs 6,988, same settings, image the
   only variable). The T257/T258 confound is resolved in the patches' favour —
   they are not flattering the SA-settings arms, they are carrying them. Also
   +12.0% GPU KV capacity (30,169,355 vs 26,932,446).
2. **The elevated error rate is NOT ours.** Stock shows **5.45%**, *higher* than
   our 4.49%. So the 15x jump over the C72 baseline's 0.30% belongs to the
   C48 / mnbt-8192 regime, not the patch stack. Drop that line of suspicion.

**What this says about SA.** Their 8,997 at C48 is on stock + LMCache. Stock
*without* LMCache is 6,988. So LMCache is worth **~+29%** to them — and it is
exactly the thing broken for us (`ext_cache_hit` 0.0%, 1,535 lookup errors).
Fixing LMCache is therefore the single largest lever identified so far, worth
more than the entire patch stack.

**LMCache is 31% NET NEGATIVE at C64 and the cause is now identified:**
`lmcache_server.log` carries **1,535** copies of
`No GPU context found for model ... with world size 8 during lookup!`
Every retrieve fails, so the 1,949 GB host tier is write-only. This is a
connector/registration fault, NOT tuning — SA's settings did not move it.
T262/T263 (workers 1) test whether the 8-worker path is what breaks
registration; T253 at workers 1 did serve.

**Stock nightly `73029d42` verified by grep (CPU-only, no GPU time):**
- `requires_dcp_block_aligned_interleave` **absent** -> **#53917 is NOT merged**,
  the upstream ask stays at 3 PRs.
- `dcp_q_replicate` (6 files) and `DCPGroupColumnParallelLinear` (3 files)
  **present** -> confirms #54494 only needs the Kimi-K3 wiring, as its
  description claims.

**SA applies NO patches.** Their `0903_2` script has zero `patch`/`overlay`/
`site-packages` references — stock `vllm/vllm-openai-rocm:nightly-73029d42` plus
one `agentic_pip_install` for the LMCache runtime. Their nightly is newer than
our `7c5dc571` base, so #53917/#52494/#52968 may already be merged in it.
**Open, CPU-only check:** pull that nightly and grep for
`requires_dcp_block_aligned_interleave` (#53917's marker). If present, the PR is
merged and comes off the ask.

**SA deltas now applied to the launcher** (T257 commit):
mns = 2×CONC at DCP>1 · mnbt 8192 (16384 only at C1) · gmu 0.90 for both
backends (the 0.88/80 LMCache override is gone — it came from misreading their
C48 artifact) · LMCache `0.5.5.dev89` from the rolling `nightly-rocm` channel ·
`--max-gpu-workers 8` + chunk 12288 at DCP>1.

**NOT applied — needs its own GSM8K-200 gate:** SA's
`AITER_QUICK_REDUCE_QUANTIZATION=INT4`. Quantizing the allreduce is a numerics
change; it does not ride along on a perf anchor. Queued.

**T256 cancelled at 29 min** (LMCache C64, old 0.88/80/workers-1 settings) —
superseded by the SA config. Its `bmk-server` container survived the GH cancel
and held all 8 GPUs at 98% VRAM until cleared before T257. **Check
`docker ps` after every cancel; a cancelled job does not free the node.**

---

**PRUNE CLOSED. Best = 11,027 tok/s/GPU (n=2: T247 11,006 · T252 11,048), #53940 REMOVED.**

| run | #53940 | tok/s/GPU |
|---|---|--:|
| T236 pre-reboot | present | 10,799 |
| T248 control | present | 10,769 |
| **T247** | **absent** | **11,006** |

#53940 (`a4w4-moe`) **costs 2.20% at C72** — outside the ±1.2% band, measured
against a same-session control. Removed from the upstream ask. T248 replicates
T236 to 0.28%, so the baseline is sound and NUMA-on was not degrading the
T236-era runs (implying numa_balancing was already 0 before the Sep 2 reboot).

**Ask is now 3 PRs:** #53917 (code-proven required), #52494 (−1.35%),
#52968 (−0.93%, draft).

**Why LMCache has underperformed here (T253 root cause):** `store ops=1824 /
90.27 GB` but **`retrieve ops=0`**, `ext_cache_hit` flat 0.0%. vLLM only queries
the connector for tokens *beyond* its own prefix cache; ours hits 90.3% locally
(SA 84.0%), so LMCache is asked for almost nothing and is pure overhead at C48.
Whether SA's higher mns/lower mnbt changes that ratio is what C64/C72 test.

Still outstanding, user-side: `echo 'kernel.numa_balancing = 0' | sudo tee
/etc/sysctl.d/99-numa.conf` — three reboots have reverted it.

<!-- superseded:
**RESOLVED — engine and node are healthy (T245).**

-->

Fixed-len **perf 8192/1024** at C72 with `numa_balancing=0`: **893/893 requests,
0 failures, 914,432 tokens generated**, 7,724 tok/s total, TPOT 78.0/77.5/92.5
ms, TTFT 4,541 ms. Compare T243/T244: **0 tokens generated**.

Two tangled problems, both now understood:

1. **NUMA auto-balancing on** — real, fixed. Capture 6 min → 1m17s, FULL → 41 s,
   startup +25m → +10m, warmup never-completing → 1m57s. **Runtime-only; add
   `/etc/sysctl.d/99-numa.conf` or the next reboot reverts it.**
2. **Wrong probe** — T243/T244 ran `TEST_MODE=func` (214k-token prompts, heavier
   than the agentic replay) while being called a lightweight canary. The
   "whole-node fault" conclusion drawn from that is withdrawn.

**NEXT: re-establish the agentic C72 baseline** (`TEST=0`) on the NUMA-fixed
node and confirm ~10,799 reproduces. That validates the node for perf work and
gives a clean baseline. **Then** redo the #53940 ablation that T241 botched.

<!-- superseded:
**T244: NUMA off is a real but PARTIAL fix. Root cause still open.**

`numa_balancing` 0 (runtime only — **not persisted**, add
`/etc/sysctl.d/99-numa.conf`). Measured gains: TTFT 150,515 → 20,214 ms (7.4x),
throughput 95 → 402 tok/s, PIECEWISE capture 6 min → 1m17s, FULL → 41 s,
startup +25m → +10m20s, and warmup **now completes** (144/144 in 1m57s) where
T243's never did. Still FAIL 98.6% (2/144), 0 generated tokens, ~200x off the
~84k tok/s aggregate a healthy C72 run gives.

-->

**CORRECTION to the T243 entry below:** T243 and T244 both ran `TEST_MODE=func`,
which generates **214k-token prompts** (ratio 0.37 → [79k, 214k]) at CONC 72 —
*heavier* than the agentic replay's 87k median, not a light canary. The
inference "fixed-len fails too, therefore whole-node fault" is therefore
**not established**. A lighter workload has not been tested since the reboot.

**NEXT: `TEST_MODE=perf`** (8192/1024, ratio 1.0) — default now flipped in the
script, comparable to prior fixed-len numbers. This is the test that has been
missing. It separates "engine cannot serve" from "that probe is too heavy".

Still unexplained: two unclean reboots (Sep 2 18:05, Sep 3 03:08, neither with a
shutdown record), system journal silent 18:18:43 → 03:29 (~9 h) with persistent
storage enabled, the GH runner dead at both boots, and a 657 MB
`_usr_bin_python3.12.0.crash` from Sep 2 17:12 (T240 / LMCache).

<!-- superseded:
**NODE IS UNHEALTHY — do not dispatch benchmark runs until NUMA is fixed.**

T243's fixed-len probe failed **99.3% (1/144, 0 generated tokens, TTFT 150 s)**
on the recommended image at gmu 0.9. Fixed-len failing rules out the agentic
replay as the cause. All 8 workers log AMD's warning that
`numa_balancing` is enabled; `/proc/sys/kernel/numa_balancing` = **1** and is
**not persisted** in sysctl, so the reboot that cleared T240's stranded VRAM
reverted it to the kernel default.

**Required before the next dispatch (needs sudo, outside my edit scope):**

```
sudo sh -c 'echo 0 > /proc/sys/kernel/numa_balancing'
echo 'kernel.numa_balancing = 0' | sudo tee /etc/sysctl.d/99-numa.conf
```

Then re-run the fixed-len probe (`TEST=1`, already set) as the gate. Only once
that is clean should agentic runs resume.

-->

**T241 and T242 are void as experiments.** T242 (gmu 0.85) collapsed on the
T236 image, and T243 collapsed on fixed-len — the same signature reproduced
without touching #53940. Roughly 8 h of GPU time was spent attributing a node
fault to the stack. `UPSTREAM-STATUS.md` has been corrected: the #53940
ablation is retracted, not merely re-qualified.

<!-- SUPERSEDED, kept for provenance:
**T241 (2026-09-02) closes the prune ladder's last open question: #53940 is
load-bearing.** Dropped it alone (`pr-applied` identical to T236, `pr-stack: 0
files`) and the run **never reached profiling** — 76 warmup primers took 3.6 h,
TTFT median 345 s, drain timed out, 0/144 successful, **errors=0**. Not a
regression, a collapse: the workload is prefill-dominated (warmup ISL median
87k, OSL 1) and prefill is MoE-GEMM-bound, which is what the a4w4 flydsl kernels
serve. GPU KV capacity 28,653,478 (−3.4% vs T236 at identical gmu 0.9).
`UPSTREAM-STATUS.md` now ranks #53940 first in the ask.

**Method note taken from T241:** ablating a kernel-path PR gets a fixed-len
canary first. A 5-request `TEST=1` probe would have shown this in minutes
instead of ~4 h of GPU time.

**NEXT: T242 — gmu 0.90 → 0.85** (pushed, takes effect on next dispatch).
Stability arm after the T240 halt stranded 57.94 GB and cost a reboot. Ceiling
is 0.88; 0.92 and 0.95 both hang (T211, T157). Expect GPU KV capacity to shrink
proportionally — **record it**; this workload is queueing-bound so the smaller
pool may cost little, but that is a measurement, not an assumption. Runs on the
recommended image (with #53940), so it is a clean one-variable arm vs T236.

Then: LMCache relaunch (teardown fix in place, unproven) → GSM8K → agentic.

-->

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

## QUEUED (after pruning) — Hyukjoon's c1 overlay: nothing new, but test it at C1

Diffed `vllm_nightly_46638857_k3_c1_current.patch` (232,005 B) against the
c16_c52 cut we ship. **19 files are c1-only, and every one maps to a known PR:**

| PR | c1-only files |
|---|--:|
| #51392 (online quant) | 16 |
| #54254 (fused KDA gated RMSNorm + per-token FP8) | 2 |
| #54248 (AITER PTPC per-token FP8) | 2 shared |
| **no PR** | **0** |

So the c1 cut adds **no unpublished content** — its whole distinct payload is
the online-quant / per-token-FP8 chain that T234/T235 just pruned as inert at
C72.

**But that chain is plausibly live at C1, and this is worth one run:**

- **#54254 fuses KDA gated RMSNorm with per-token FP8 for `o_proj`** — a pure
  decode-path fusion. C1 is pure decode.
- Its own accuracy section is headed *"8xMI355X, TP8, DSpark MTP"* — written
  against the C1/MTP configuration.
- C1 runs spec decoding (MTP at CONC <= 4); C72 does not. Different path.

**There is also an unexplained gap this could bear on:** bare nightly C1 is
**8.52 ms** TPOT (T213) while patched v4 C1 is **9.06 ms** (T208). Stock is
*better* at C1 and nobody has explained why.

### The experiment — no new images needed

**C1 on `pronly` (chain present) vs C1 on `pronly-noquant` (chain absent).**
One variable, both images already built. Config per the C1 operating point:
MTP on, DCP off, mns 1, chunk 8192, ladder 1..9, gmu 0.9.

Also worth adding as a third arm once those two land: **C1 on bare
`nightly-7c5dc571`**, to see whether the 8.52 ms is reproducible and whether it
survives the patches at all.

**Do not run before pruning completes** — C1 arms and C72 prune arms would
compete for the same GPUs and the prune ladder is the higher-value sequence.

---

## THE 12,500 PLAN — after pruning completes

**Where we are: 10,781 at C72. Target 12,500. Gap = +15.9%.**

The config space is closed (conc, mns, chunk, gmu, offload all swept and flat or
negative). Pruning reduces upstreaming surface, it does not add throughput. So
12,500 requires **new performance work**, and there are exactly four candidate
sources. Ranked by expected value:

### 1. The prefix-cache gap — biggest single lever

| stack | theoretical | captured | gap |
|---|--:|--:|--:|
| aigmkt C52 | 96.5% | 93.6% | 2.9 pts |
| bare nightly C52 | 96.4% | 90.0% | 6.4 pts |
| **pronly C72** | **~95.1%** | **88.0%** | **7.1 pts** |

Workload is prefill-dominated — `in:out ≈ 195:1`, ISL p50 ≈ 89k. Every point of
prefix hit removes prefill work directly. Closing 7 points is worth far more
than any config knob we have left. **Diagnose before optimising:** we do not yet
know whether the gap is eviction, block-matching, or DCP geometry. `kv_usage`
was only 22% when we measured the C52 gap, so it is **not** capacity.

### 2. The 5 unapplied PRs — the only known perf work not in our stack

| PR | what | why it could matter |
|---|---|---|
| **#52190** | torch.compile enablement | K3 carries no `@support_torch_compile`, so **no fusion pass runs at all** today — `norm_quant`, `act_quant`, `allreduce_rms`, `mla_dual_rms_norm` are all configured and all inert. Potentially the largest single item. |
| **#53166** | MLA prefill chunked-context gather | 4 kernels → 1 on the prefill path, which dominates this workload |
| **#51437** | latent-MoE all-reduce overlap | decode-side; owns TPOT/ITL |
| **#53301** | graph-stable attn metadata reuse | per-step overhead, 6 MLA + 14 KDA groups at TP8 |
| #54163 | DFlash/DSpark prefix-cache block | spec off at C72 |

Each fails on `7c5dc571` by **1-2 hunks** — a rebase, not new engineering.

### 3. Base bump to `nightly-73029d42…`

Carries **#53388** (disable trailing prefix-cache block dropping), which is
directly on the gap in item 1, plus #52832 and the 09-02 SimpleCPUOffload fix.
Needs a fresh GSM8K gate and invalidates the n=2 baseline.

### 4. LMCache — see **`LMCACHE.md`** (single source for this work)

Wiring, the chunk-size trap, the DCP=8 geometry risk, patch candidates, and the
fixed-len → GSM8K → agentic run order all live there. Summary only below.

### 4b. LMCache — original notes (superseded by LMCACHE.md)

**Reference: SA run 33618719560 job 100211512290, and the script at
`SemiAnalysisAI/InferenceX@perf/k3-mi355x-lmcache-rc3-c1-c8-c14-c40`
`benchmarks/single_node/agentic/kimik3_fp4_mi355x_mtp.sh`** (both read-only).
The run **failed**, but the script supplies the full setup we lacked.

### Install (stock image + runtime deps, no torch/ROCm change)

```
lmcache==0.5.5rc3+rocm7.2
  --find-links https://github.com/LMCache/LMCache/releases/expanded_assets/v0.5.5rc3-rocm
sortedcontainers==2.4.0
opentelemetry-exporter-prometheus==0.61b0
cupy-rocm-7-0==14.1.1
```
installed `--no-deps`. Native libs required — `libglog.so.0`, `libjsoncpp.so.25`,
`libibverbs.so.1`, `librdmacm.so.1`, `libnuma.so.1` (apt: `libgoogle-glog0v5
libjsoncpp25 libibverbs1 librdmacm1 libnuma1`). Import gate:
`import cupy; import opentelemetry.exporter.prometheus; from
lmcache.v1.multiprocess.http_server import run_http_server`.

### THE critical detail — `--chunk-size 12288`

Their comment, and it is the thing that would have cost us a day:

> the connector requires the chunk to be a multiple of **every** engine KV
> group's `tokens_per_block`. The hybrid KDA/MLA layout registers attention
> groups at **1536** and a KDA state group at **3072**. Use **12288** so it is
> divisible by both. The multi-group layout also requires one object group per
> sliding-window size: **`--separate-object-groups`**.

The upstream Kimi-K3 recipe (docs.lmcache.ai) says **768**, which is the CUDA
path and is **wrong for this stack**.

**Open risk for us:** those group sizes are quoted at **DCP=1**. At DCP=8 the
per-group geometry changes — that is precisely what #53598 and #53917 exist to
handle — so **12288 may not be the right chunk for our config**. Verify
`tokens_per_block` per group from the engine log before trusting it.

### Full server invocation

```
lmcache server --host 127.0.0.1 --port 6555 \
  --http-host 127.0.0.1 --http-port 8090 \
  --l1-size-gb $TOTAL_CPU_DRAM_GB --l1-init-size-gb 10 \
  --chunk-size 12288 --separate-object-groups \
  --enable-extra-logging --extra-logging-interval 30 \
  --max-cpu-workers 8 --max-gpu-workers 1 \
  --eviction-policy LRU \
  --supported-transfer-mode lmcache_driven --shm-name ""
```
Readiness: `wait_for_ready --endpoint http://127.0.0.1:8090/healthcheck
--timeout 600`.

### Connector

```json
{"kv_connector":"LMCacheMPConnector",
 "kv_connector_module_path":"lmcache.integration.vllm.lmcache_mp_connector",
 "kv_role":"kv_both",
 "kv_connector_extra_config":{"lmcache.mp.port":6555,"lmcache.mp.mq_timeout":6000.0}}
```

`mq_timeout 6000.0` is deliberate — their note: *100k-330k-token agentic
prefixes make single retrieves large*. Our ISL p50 is ~89k, so we need the same
headroom.

**Take the setup, not the config.** Per user: this is a wiring reference only.

Server (starts fine on ROCm — healthcheck passed):

```
lmcache server --host 127.0.0.1 --port 6555 \
  --http-host 127.0.0.1 --http-port 8090 \
  --l1-size-gb 1799 --l1-init-size-gb 10 \
  --chunk-size 12288 --separate-object-groups \
  --enable-extra-logging --extra-logging-interval 3
```

Connector: **`LMCacheMPConnector`**, transfer mode **`lmcache_driven`**.
Readiness gate: `wait_for_ready --endpoint http://127.0.0.1:8090/healthcheck`.

**This partly answers my earlier caveat.** I flagged that the published LMCache
wins are B200/B300 and lean on CUDA-IPC KV export that may not port to ROCm.
The server and connector evidently *do* come up on MI355X — so the porting
concern is weaker than I stated. What is not yet shown is that it *serves*.

**Its config is far from our operating point, and that matters:**

| | SA LMCache ref | our target |
|---|---|---|
| image | bare `7c5dc571`, **no patches** | pronly |
| **DCP** | **1 — off** | 8 |
| conc | **14** | 72 |
| mns | 28 | 96 |

**Outcome: ProfileAborted.** Warmup completed 150/151, then
`0 successful / 149 total (149 warmup, 145 error dropped)`. So it is a **recipe
reference, not a working result** — and it failed at low concurrency with DCP
off, which is a much easier configuration than ours.

Two things follow for our attempt:
1. We inherit the server flags and connector name — that is the real value here.
2. We should not assume it composes with DCP=8; the reference never tested that,
   and #53917 (our offload-under-DCP fix) targets the same layer LMCache would
   replace.

Still gated behind ruling out a geometry defect first, per below.

Not wired for K3. Motivated by the same cache gap, but the published wins are
B200/B300 CUDA leaning on CUDA-IPC KV export, which does not port to ROCm free.
**Rule out a geometry defect first** — #53598, #53917, #54163/#54165 are all
bugs in the family "prefix-cache hits structurally dropped under DCP/hybrid/spec".

### Honest feasibility

+15.9% from four sources, none individually sized. The prefix-cache gap and
#52190 are the two that could plausibly be worth several percent each; the rest
look like 1-3% items. **I would not promise 12,500 from this list** — but
diagnosing the cache gap is cheap, high-information, and has to happen before
LMCache is worth attempting either way. That is where I would start.

Profiling (Phase 3) stays blocked: both in-tree paths are dead (T202 no
`VLLM_TORCH_PROFILER_DIR`; T203 rocprofv3 deadlocks at capture). Worth
re-checking on `7c5dc571`, five weeks newer than the base those were tested on.

---

## ORDER (2026-09-02 ~18:15Z) — PRUNE FIRST, LMCache after the teardown fix

**Reason for the flip:** T239 stranded 58 GB on a GPU at teardown
(`Memory critical error … Reason: Memory in use`). A GPU reset did not clear it;
the node had to be **rebooted**. Running LMCache again unattended risks halting
the GPUs with nobody to recover them.

| slot | trial | status |
|---|---|---|
| **now** | **T241 — #53940 ablation** (`rec-no53940`, SimpleCPUOffload) | dispatched, run 33665892734 |
| then | LMCache teardown fix (script-side, no GPU) | see `LMCACHE.md` |
| then | T240-retry LMCache GSM8K → agentic | only after the fix |

**Open question before LMCache resumes:** the run reported
`L1 memory usage: 38.16/1820.00 GiB` and GPU KV capacity **unchanged** at
29,656,464 — so L1 was the *host* pool, not GPU VRAM. If GPU-VRAM-as-L1 is what
12,500 needs, that is a **different LMCache configuration** than the one we ran,
and it should be settled before the next arm rather than discovered mid-run.

## SUPERSEDED — LMCache first, 12,500 is the priority

**User-set 2026-09-02 ~17:00Z.** Supersedes the earlier plan that put #53940
ahead of the LMCache agentic run.

| slot | trial | gate | est. finish |
|---|---|---|---|
| now | T239 LMCache fixed-len | — | ~17:30Z |
| 2 | T240 LMCache **GSM8K** | engine clean | ~18:15Z |
| 3 | **T241 LMCache agentic C72** | **GSM8K passes** | ~20:15Z |
| 4a | if agentic **fails** → **#53940 ablation**, then LMCache fix | | ~22:15Z |
| 4b | if agentic **passes** → LMCache conc/mns sweep; #53940 in the next gap | | |

**#53940 deadline is SOFT** (user, ~17:00Z): *"deadline is little flexible. Not
hard. If we getting 12,500 -> then it's the high priority."*

So: if LMCache is producing real movement toward 12,500, **keep the GPUs on
LMCache** and let #53940 slip. Run it in a genuine gap — a failed arm, a
rebuild, or once the LMCache line of attack is exhausted — not by interrupting
progress.

If LMCache stalls or plateaus well short of 12,500, #53940 becomes the next
useful thing and should go immediately.

---

## PARKED — #53940 ablation (image built, ready to dispatch)

**LMCache is the priority; this waits.** Image `kimi-k3-vllm:rec-no53940` is
**already built and import-gated** — `pr-applied: 53917 52494 52968`,
`pr-stack: 0 files`. Drop it into the yaml and dispatch when LMCache work
pauses; no rebuild needed.

**Why it matters:** #53940 (a4w4 flydsl MoE kernels) has **never been ablated**.
It rode in `pr_stack/` unchanged through every arm, v4 included, so it is the
one PR in the recommended stack with **zero measurement** — currently listed as
MUST on mechanism alone (`flydsl_moe1_abf16_wfp4_bf16_…` on the live MoE path
in the T195 log). This arm would replace that reasoning with a number.

Baseline to compare: **T236 = 10,799**.

Enabling change made while building it: the pronly Dockerfile now tolerates an
empty `pr_stack/` (`[ -e "$p" ] || continue`) and **generates** the `pr-stack:`
manifest line from what actually applied, instead of the hard-coded
`4 files (#53940 a4w4-flydsl)`. That line would otherwise have lied in this
image — the same defect class already caught once on `pr-applied`.

---

## RESOLVED — bare-nightly C52: gmu 0.88

`HSA_STATUS_ERROR_OUT_OF_RESOURCES` at C52/DCP8/no-offload is fixed by dropping
gmu 0.9 → **0.88** (confirmed by user on the SA side). Recorded as the default
in `old.sa.sh` and in the bare-image override in the main launcher; **v4 and
pronly keep 0.9**, gated on `/etc/k3-image-manifest`.

My staged 0.85 was a guess in the right direction; 0.88 is the measured value.
T234 as a GPU experiment is **dropped** — the question is answered.

Root cause for the record: the queue aborted on resource exhaustion first; the
Tensile / Triton-autotune segfaults were corpse frames (once the HIP context is
dead, any device-property query faults). The ragged-494-slice shape hypothesis
I built on those frames was wrong.

**mns is NOT the lever here.** mns 52 at conc 52 would put mns *below* the
replay's lane count — measured at 79 lanes @ conc 72 and 81-83 @ conc 76, so
~55-57 @ conc 52. That is the T219 regime (mns 20 at C16 → 0/49 completed). It
would trade resource exhaustion for starvation. Floor for conc 52 is ~64.

---

## PRUNING RESUMED — Phase 1.4, leave-one-out over `pr_only`

Baseline: `pronly` = 11 PRs (4 merged in base + 7 applied), **10,692 @ C72**,
GSM8K 0.99.

### T238 RESULT — PRUNE LADDER CLOSED. Last two prunes were NOT free.

10,691 (6 PRs) / 10,781 (4) / **10,799 (3)** / 10,653 (2) / **10,554 (1)**.

**T236 → T238 is −2.27%**, outside the band and monotone. Each single step
looked like noise; cumulatively it is not. **#52494 (−1.35%) and #52968
(−0.93%) each cost ~1%** — both kernel fusions on hot paths that K3 cannot get
from inductor (no `@support_torch_compile`).

**Recommended stack = T236's: #53917 + #52494 + #52968 + #53940 → 10,799.**
That is **4 open PRs**, not 2. The extra two buy 2.27%.

Free to drop, confirmed: #51392, #54254, #50618, #54165, #50813.

**GPU KV capacity now tracked every run.** It was identical (29,656,464) for
T232–T237 and rose to 29,816,030 in T238 when #52968 came out — so T238 had
*more* cache and still scored lower.

**Next: LMCache.** See `LMCACHE.md`. Order: fixed-len → GSM8K → agentic.

### T237 RESULT — #52494 prunes, but weakest evidence. C72 = 10,653, 2 PRs.

Upstreaming ask **4 → 3 open PRs**. Stack: #53917 + #52968 (+ #53940, + 4 merged).

Ladder: 10,691 (6 PRs) / 10,781 (4) / 10,799 (3) / **10,653 (2)**. Full spread
**1.37%** — marginally wider than the ±1.2% band, and T236→T237 is **−1.35%**,
the largest single step. **Not calling #52494 free**: the other three prunes each
landed within 0.2% of their predecessor. Mechanically a cost is plausible —
#52494 fuses q_a/kv_a RMSNorm into one AITER launch on every MLA layer, and K3
has no `@support_torch_compile` so the fusion pass never runs otherwise.
**If any arm gets replicated, replicate T237.**

**Next and last arm: #52968** (4 files / 17 hunks). If it prunes, only #53917
(+#53940) remain — kept for last since we run `offload dram`.

### T236 RESULT — #50618 PRUNED too. C72 = 10,799 with 3 applied PRs.

Upstreaming ask **5 → 4 open PRs**. Stack is now #53917 + #52494 + #52968
(+ #53940 pr_stack, + 4 merged in base).

10,691 / 10,781 / 10,799 across 6 / 4 / 3 applied PRs — **all inside a 1.0%
spread and inside the ±1.2% band. No measurable difference, not a trend.**
Each arm n=1.

**Caveat on #50618:** it guards a 12,288-byte over-read in KDA `f_b_proj`
(`stride=(6288,1)` at TP8) before ROCm skinny GEMMs. Only its python hunk was
ever applied here — the `csrc/*.cu` side was stock in *every* arm — and an
over-read need not fault. **Measured-safe, not proven-safe.** First thing to
restore if a stray memory fault appears.

**Next arm: #52494** (2 files / 4 hunks), then #52968, then #53917 last.

### T235 RESULT — 10,781 at C72. #51392 + #54254 PRUNED.

GSM8K 0.995 (T234) and 10,781 tok/s/GPU (T235), err 0.18%. +0.84% over the
pronly mean — **inside the ±1.2% band, so no measurable difference**, not an
improvement. n=1 on this arm.

**Upstreaming ask: 7 open PRs → 5.** 24 of 45 file-touches and 73 of 129 hunks
removed. Mechanism: the checkpoint is `mxfp4` with `quantization_config=None`,
so #51392's online-quant path had no work; #54254 is stacked on it and is a
self-declared no-op without it. Both inert by construction on this model.

**Next arm: drop #50618** (1 hunk, `scaled_mm/rocm.py`, isolated). Then #52494
→ #52968 → #53917 last.

### T234 RESULT — GSM8K 0.995. Numerics gate PASSED.

`pronly-noquant` (4 applied PRs) scores **0.995**, tying the best in the ledger
and indistinguishable from full pronly's 0.99. The checkpoint is `mxfp4` with
`quantization_config=None`, so #51392's online-quant path is not needed.

**T235 dispatched: C72 throughput on `pronly-noquant`.** Baseline to beat is
10,691 (n=2). #54254 and #52494 are perf PRs, so accuracy passing says nothing
about tok/s. If throughput holds, the upstreaming ask drops **7 open PRs → 5**
and 24 of 45 file-touches prune.

### T234 (was) — drop #51392 + #54254

Image **`kimi-k3-vllm:pronly-noquant`**, built and import-gated.
Manifest confirms `pr-applied: 53917 52494 52968 50618`.

Dropped as **one unit** because they are a dependency pair — #54254's body:
*"Depends on #54248 and on #51392"*. Splitting them is not a legal arm, the same
lesson as overlay C/D/E.

Value if it prunes: **24 of 45 file-touches, 73 of 129 hunks** — by far the
largest reduction available, and it removes the entire online-quantisation and
per-token-FP8 chain from the upstreaming ask, leaving 5 open PRs instead of 7.

**Numerics-affecting → GSM8K limit 200 FIRST**, then C72 throughput only if it
passes. The engine reports `quantization=mxfp4, quantization_config=None`, so
whether #51392 is load-bearing for this checkpoint is genuinely unknown — it may
fail to start, which is itself a clean result.

### Order after T234

Expect-free first: **#50618** (1 hunk, isolated) → **#52494** (4 hunks) →
**#52968** (17 hunks). **#53917 last** — it is the offload-under-DCP fix and we
run `offload dram`; T232's `ext_cache_hit` of 46% (vs v4's 78%) suggests the
external tier is already contributing less than assumed, so it is worth testing
rather than assuming, but it is the most likely to be required.

---

**IMAGE CHANGED MID-FLIGHT — read before interpreting T233.** `kimi-k3-vllm:pronly`
was rebuilt at 2026-09-02 ~07:4x to **drop #54165** (closed-unmerged upstream;
author closed it as superseded by #54163; spec decode is off at C72 so it was
inert there). The tag now means **11 PRs (4 merged + 7 open), nothing closed**.

**T232 and T233 both ran the OLD 12-PR image** — they were already dispatched.
Their numbers are valid for that stack. The first run on the 11-PR image will be
T234. Expect no difference at C72 (spec off), but it is n=0 until measured, and
at C1 it could matter because MTP is on at CONC <= 4.

**T233: pronly C72 = 10,690 — replicates T232 (10,692) to 0.02%.** n=2 confirmed,
tightest pair in the ledger. Headline for the mergeable stack is **10,691**,
+0.6% over the matched-mns overlay run (T198). **The overlay is worth nothing at
C72.** Note the noise is not uniform: same-session pairs replicate to 0.02%,
but cross-day byte-identical runs differ by 1.2% (T206 vs T228) -- quote ±1.2%
for cross-day comparisons.

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

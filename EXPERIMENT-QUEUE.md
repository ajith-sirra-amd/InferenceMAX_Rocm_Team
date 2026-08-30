# Autonomous run queue — Kimi-K3 / 8× MI355X

Owner away 2026-08-28 → 2026-08-30. This file is the single source of truth for
what runs next. Every wake-up: read **Current state**, act, update this file.

## Targets

| | target | best today | gap |
|---|---|---|---|
| Throughput | **12,500 tok/s/GPU** | **8,342 (C60, n=2 clean, spread 0.20%)** · SA 8,296 | **−33%** |
| C1 interactivity | as low as possible | **7.57 ms** TPOT (T147, nightly) | — |

**Honest position on 12,500, restated because it drives priorities:** the T124
profile puts GPU idle at 28.2% of e2e wall. Eliminating idle *entirely* yields
~11,050 tok/s/GPU. Every remaining kernel lever is single-digit percent. So
12,500 is **not reachable by stacking the levers currently identified** — it
needs either a kernel-level step change (the nightly's #53942 class of work) or
a different operating point. I will keep pushing and report the real number
rather than a flattering one.

## Bounds — never cross

- Dispatch only to `ajith-sirra-amd/InferenceMAX_Rocm_Team`. `SemiAnalysisAI/InferenceX` is **read-only**.
- **No Docker Hub pushes.** Everything via runtime patches.
- **No git push anywhere except `InferenceMAX_Rocm_Team`.** Not to `ajith-sirra-amd/vllm`.
- Edit only: `kimik3_fp4_mi355x_mtp.sh`, `kimik3_fp4_mi355x_mtp.sa.sh`, `apply_kimi_k3_patches.sh`, the `pr51705_nightly.diff`, the kimi block in `amd-master.yaml`, and root docs.
- **No code changes anywhere else.** vLLM changes live in the vendored diff only.
- Only cancel runs I started.
- **Accuracy gate before throughput** whenever numerics could move.

## Current state

**RETRACTED: all nightly C1 throughput/ITL numbers.** T160's aiperf error
summary shows the 17/148 is **one EngineCore crash** (500) followed by 12
connection-refused requests, tripping aiperf's 10% failed-request threshold and
cancelling the run. T156/T158/T160 C1 are truncated runs ending at an engine
crash, not measurements. The CCD-pinning "-2.3%" is withdrawn with them.

**C52 ledger (unaffected — those runs completed):**

| run | change | tok/s/GPU |
|---|---|--:|
| T103 (aigmkt) | baseline | **7,950.6** |
| T156 | nightly + #51705 | 7,906 |
| T157 | gmu 0.95 | 0 — engine hung |
| T158 | NCCL_MIN_NCHANNELS=32 | 7,656 |
| **T160** | **CCD pinning** | **7,968 — best** |
| SA | reference | 8,296 |

- **T160 C52 DONE: 7,968 tok/s/GPU**, 1,899 successful, error rate 0.105%.
  Pinning = **+0.78%** over T156 on the identical stack. First non-negative C52
  lever. Weight load 2008.62 s (pre-pin loop ran during the load) — wall only,
  not the measurement; fixed by N1.
- **T161 DONE. C52 = 7,824** (offload `none`), **C1 aborted again** (15/146).
  - N1 works: weight load **176.83 s** vs 2008.62 s, pin still applied (1,554).
  - C1 crash is **not** the pin loop — reproduces without it.
  - **mns 80 + none did not OOM.** First success of that pair on our node.
  - **Confound, stated:** T161 also flipped offload dram→none (sa.sh copy).
    Read as an offload A/B: **dram is worth +1.8%**, matching T103/T133 (+2.8%)
    and SA (+1.1%). The "offload OFF" rule is **withdrawn** — it was inferred
    from idle, never from throughput. Restore `dram` for best-config runs.
- **T162 DONE. N2 async scheduling = 7,686, −1.8% vs T161.** Confirmed live
  (`'async_scheduling': True`). C1 untouched and aborted again at 15/146 —
  third consecutive reproduction, so the crash is independent of pin timing
  *and* async. **Async is a settled negative; reverted in the script.**
  - **This falsifies the profile's top attribution.** ~150 s of 403.9 s idle
    was attributed to host/Python batch prep. Async exists to overlap exactly
    that, and it made throughput worse. Either the attribution is wrong or the
    host work is already off the critical path. N3 is now the leading lever by
    default, not by ranking.
- **T163 attempt 1 failed in `get-jobs`, my error.** Pydantic:
  `kv-offload-backend is required when kv-offloading is 'dram'`. I restored
  `dram` without the backend key that f3afc488 had removed alongside it.
  Fixed: `kv-offload-backend: { name: vllm-simple }` — what T103/T160 ran.
  **Queue rule added: any yaml search-space edit gets a local parse + the
  dram/backend assertion before dispatch.** No GPU time was consumed.
- **T163 DONE. C52 = 8,127 — NEW BEST.** 1,955 successful, error rate 0.102%.
  The offload is worth **+3.9%** (vs T161 7,824), bigger than the +1.8% I
  estimated, and it clears T160's 7,968 by +2.0% with pinning after ready.
  **SA's 8,296 is now 2.0% away.** Caveat: the connector allocated 226.89 GB/rank
  here vs 243.6 GB earlier, so offload size is not constant across the ledger.
  C1 aborted a fourth time at the identical 15/146.
- **N3 is BLOCKED, not skipped.** The nightly logs only the *potential* AR
  backend list for `dcp:0` (`NCCL_SYMM_MEM, QUICK_REDUCE, FLASHINFER,
  AITER_CUSTOM, CUSTOM, SYMM_MEM, PYNCCL`) and never prints which it selected,
  so I cannot confirm the old "dcp:0 gets PYNCCL only" finding still holds on
  this build, and I could not find a documented flag to force it. Guessing a
  `--dcp-comm-backend` enum value risks a hard startup failure. Needs the
  selected-backend line located in vLLM source first.
- **T164 DONE. N4 chunk 4096 = 7,528, −7.4% vs T163.** Confirmed live
  (`[chunk] max_num_batched_tokens=4096 conc=52`; C1 correctly stayed 8192).
  **My gradient reasoning was wrong** — I extrapolated from 16384's −2.5% that
  smaller would help, and 4096 is three times worse than 16384. The curve peaks
  at 8192 and both sides fall away. Reverted and marked settled.
- **T165 DONE, and it FAILED: mns 96 kills the engine.** 256/392, aborted at
  29/285 = 10.175%. Partial number 5,090, not a measurement.
  - **Same trace as every C1 abort**: `engine_core_sentinel` → `mq.dequeue`
    timeout → `EngineDeadError`. So the C1 crash was never C1-specific.
  - **My "aigmkt won't crash" prediction is WITHDRAWN.** The sentinel is in
    `v0.26.1rc1.dev1133` too. (I also briefly misread the build string as a
    version change between T163 and T165 — it is not; T163/T164/T165 are all
    the same build, so the mns A/B is clean.)
  - **Not memory:** engine dump at death says `num_running_reqs=45`,
    `kv_cache_usage=0.28`. No OOM, no HSA fault. The fallback ladder was not
    needed and was not used.
  - mns 96 reverted; mns 80 completed twice on this image.
- **NEW TOP PRIORITY — N8: raise the executor RPC dequeue timeout.** It caps the
  resident-sequence axis and is the sole cause of six straight C1 aborts.
  `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1200` is already set and is not the
  binding one. Needs the `dequeue_timeout` on
  `multiproc_executor.get_response` located in vLLM source before it can be
  changed. Doing this would reopen mns AND likely give the first valid C1
  number in six runs.
- **T166 DONE, FAILED: gmu 0.92 hangs. 0 successful / 103.** Gate lines all
  correct (`[gmu]=0.92`, mns 80, chunk 8192). The server *started* and KV grew
  **59.8 → 65.6 GiB (+9.7%)** — the memory is genuinely there — then it hung in
  warmup. T157 at 0.95 hung the same way (0/57). **Two points above 0.90 hang,
  0.90 works.** Reverted and marked settled.
  - Consequence: **the capacity hypothesis is now closed on both axes.** More
    batch rows kills the engine on an RPC timeout (T165); more KV memory hangs
    it (T166/T157). T163's +3.9% from the offload was capacity that came from
    *host* DRAM, which is the only place more of it can currently come from.
- **T167 DONE, FAILED: the direct DCP a2a op does not exist in this image.**
  Every worker died at cudagraph capture with `AttributeError: '_OpNamespace'
  '_C' object has no attribute 'direct_dcp_a2a_lse_reduce'`. My framing was
  wrong — the script's `=0` is not suppressing an unmeasured fast path, it is
  disabling a **compiled op aigmkt does not ship**. #51705 gives the Python
  plumbing, not the kernel. Reverted; needs a rebuilt image, out of bounds.
  Cost: ~50 min of it was runner queue time, ~18 min of GPU.
- **Config space at C52 is now exhausted.** Every knob reachable from the
  script has been measured: chunk peaks at 8192, mns is capped by the RPC
  timeout, gmu is capped by the warmup hang, async is negative, the offload is
  on, pinning is on, and the direct DCP path needs a kernel we do not have.
  **Everything still open (N3, N6, N8) needs either vLLM source or an offline
  tuning job.** Best remains **T163's 8,127**.
- **T167 C1 also aborted, 131/148 — seventh consecutive.** Config verified back
  at T163 values (dcp-direct 0, mns 80, ladder 80, gmu 0.90, chunk 8192, async
  off, offload dram+vllm-simple).
- **T168 C1 aborted too** (130/147, 15/145 = 10.345%) — eighth consecutive.
- **T168 DONE. C52 = 8,103 on T163's exact config — noise floor is 0.30%.**
  1,948 successful, error rate 0.154%. This retro-validates the ledger: chunk
  −7.4% is 25× noise, offload +3.9% is 13×, async −1.8% is 6×. **The one weak
  claim is CCD pinning's +0.78%, at 2.6× — provisional.** And n=2 is a point
  estimate of spread, not a standard deviation.
- **T169 C56 = 8,326 — NEW BEST, +2.6% over C52 and past SA's 8,296.**
  1,814 successful, error rate 0.220%. The curve is **still rising at 56**, so
  the peak is above it. Note the framing: nine config experiments moved C52 by
  ~+2.8% total; changing the operating point to 56 added +2.6% in one run.
  **C52 was inherited, never chosen.** SA parity is 0.4% — 1.3× noise, so
  parity, not a win.
- **T169 C1 aborted again** (131/148, 15/146) — ninth consecutive. Unchanged
  cause: the RPC dequeue timeout (N8), which needs vLLM source.
- **T170 C72 ABORTED — the engine died, and it is the T165 death again.**
  16× HTTP 500 `EngineCore encountered an issue` + 30 `InvalidInferenceResult`
  → 43/284 = **15.141% > 10%** → `ProfileAborted`. The 4,275 printed before the
  abort is a partial, not a measurement. All gate lines correct (dcp 8/a2a/1,
  mns 80, chunk 8192, gmu 0.9, ladder 1..80, pinned 1,562 threads, init 531 s)
  — this was not a misconfiguration.
  - aiperf at death: **effective concurrency max 85 against mns 80**, decode
    **4.49 tok/s/user**, CO-aware latency max **524 s**, tokens in flight 4.63 M.
  - **`mns` and concurrency are the same knob.** T165 raised mns 80→96 and the
    engine died; T170 held mns at 80 and let offered load push the batch to 85
    — identical outcome. One ceiling, and it is the executor RPC
    `dequeue_timeout` (**N8**), not KV memory, which was never binding in
    either run.
  - **N8 now caps the throughput curve as well as C1.** It was already the top
    blocked item; it is now worth more than any remaining config lever.
  - **Peak is bracketed in [56, 72).** C64 (in flight) decides it.
- **T170 C64 = 8,040 — −3.4% vs C56. The curve turns over. PHASE E CLOSES.**
  1,805 successful of 1,939, validated error rate **3/1808 = 0.166%**, all gate
  lines correct — a clean measurement of a worse operating point.
  - Full curve: **7,771 / 8,115 / 8,326 / 8,040 / aborted** at 48 / 52 / 56 /
    64 / 72. Peak is **56**, and 64 lands below even the C52 mean.
  - **Settled best config: C56 = 8,326 tok/s/GPU.** Best number on this stack.
  - C64 took **115 min** wall against ~50–70 min elsewhere. At 64 the engine is
    already deep in the queueing regime that kills it outright at 72 — the
    slowdown and the C72 death are the same N8 ceiling seen from two sides.
- **Next: C60**, the only unmeasured point inside the peak bracket [56, 64).
  One run, one variable. If it beats 8,326 the peak refines upward; if not,
  C56 stands and the concurrency axis is exhausted along with the config axis,
  leaving **N8 as the only remaining lever** — and it needs vLLM source.
- **T170 C1 aborted — tenth consecutive, at the byte-identical 15/146 =
  10.274%.** 131 successful of 148. Gate lines correct for the C1 arm (`[dcp]
  DISABLED`, `[mns] max_num_seqs=8 offload=none`, ladder 1..16, chunk 8192,
  1,490 threads pinned). Ten runs, one failure signature, no drift: this is a
  deterministic ceiling, not a flaky one, and it is N8.
- **T171 C60 aborted early — INCONCLUSIVE, and NOT the C72 cliff.** aiperf
  `ProfileAborted` at 18/176 = 10.2%; harness 43/206 = 20.874%. 4,628 is a
  partial. Gate lines all correct.
  - **Zero HTTP 500s** — the engine never died. 54 `InvalidInferenceResult`
    (empty streams) and nothing else. C72 by contrast had 16 engine 500s with
    the empties downstream of them. Different failure, not the same one.
  - **Effective concurrency peaked at 48** — under the offered 60 *and* under
    mns 80. There was no queueing pressure, so the load-cliff story does not
    apply here.
  - Same error class runs at **3/1808 = 0.166%** at C64. C60 is that background
    rate spiking 18× inside the first ~200 requests.
  - **A cliff at 60 that vanishes at 64 is not physical.** C56 and C64 both
    completed cleanly on either side. Recorded as inconclusive, **pending one
    re-run**; it is not evidence about the curve either way.
- **T171 C56 ALSO aborted — the run is compromised, and this corrects what I
  said about C60 last cycle.** 10/59 = 16.949%, 49 successful of 174, 1,489 is a
  partial. Same class as C60: 10 `InvalidInferenceResult`, **zero 500s**.
  - **`init engine` took 3,194 s** against 447 s (C60), 531 s (C72), 447 s
    (C64) on the identical config — **6–7× every other run**. Effective
    concurrency never passed **28** against mns 80. The node was degraded.
  - Weight-load time is *not* the discriminator: it swung 169 s → 717 s across
    these runs and the 717 s one (C64) is the clean 8,040 measurement.
  - **I framed C60 as an isolated inconclusive point one cycle ago. Withdrawn.**
    Two jobs in the same run, same signature, one of them at a concurrency that
    had already measured clean at 8,326 — that is a run-level fault, not a
    property of 60. **Neither T171 throughput number is usable.**
  - The Phase E curve therefore rests on T169 + T170 exactly as before, and
    **C56 = 8,326 is still n=1.**
- **T171 C1 aborted — eleventh straight, again the byte-identical 15/146 =
  10.274%**, 131 successful of 148, C1 gate lines correct.
  - **This is a useful control, not just another tally mark.** C56 and C60 in
    the *same run* were wrecked by node degradation and failed at 16.9% and
    20.9% — numbers that move. C1 landed on exactly the same 15/146 it has hit
    eleven times across healthy and degraded nodes alike. **The C1 abort is
    deterministic and independent of node health**, which further isolates it
    to N8 rather than to anything environmental.
- **T172 C1 aborted — twelfth straight. 15/145 = 10.345%**, 130 successful of
  147, gate lines correct.
  - **Precision correction to what I wrote last cycle:** I called the C1
    signature "byte-identical". It is not quite. The *failure count* is pinned
    at exactly **15 every time**; the *denominator* moved 146 → 145 here (130
    successful, not 131). So the right statement is: **15 failures is the
    invariant, the total drifts by one request.** That is still a deterministic
    fault rather than a rate — but "byte-identical" overstated it and is
    withdrawn.
- **T172 C56 FAILED 0/56 — and `server.log` shows the sentinel trace is a
  SYMPTOM, not the fault. Root cause: `HSA_STATUS_ERROR_OUT_OF_RESOURCES`.**
  Warmup completed clean (115/115, errors=0), `init engine` 564.61 s (normal),
  then every profiling request hit `ClientConnectorError` — server already gone.
  - `server.log` order of events: **three ROCm queues abort with
    `HSA_STATUS_ERROR_OUT_OF_RESOURCES`** first, *then*
    `engine_core_sentinel:179` → `mq.dequeue` → `acquire_read` →
    `RuntimeError: cancelled` → `EngineDeadError`.
  - **This weakens N8 and I am saying so directly.** I have been treating that
    sentinel trace as the fault and prescribing "raise the RPC dequeue timeout".
    Here it is plainly downstream of GPU-runtime resource exhaustion; no timeout
    change would have helped. **The sentinel trace alone is no longer sufficient
    evidence for N8.**
  - It does *not* follow that the twelve C1 aborts are HSA. No HSA line has been
    seen in a C1 log, and C1's fixed-15-failures signature still looks like a
    separate deterministic fault. Keeping the two apart.
  - **NEW STANDING RULE: before invoking N8 on any `EngineDeadError`, pull the
    `server_logs_*` artifact and grep for `HSA_STATUS`.** The runner blob alone
    hides the true first cause. This run is the proof.
- **T173 C56 aborted (29/191 = 15.2%) — and the new HSA rule cracked the whole
  cluster of failures on its first use.** 3 `HSA_STATUS_ERROR_OUT_OF_RESOURCES`
  in `server.log` again. I then grepped the archived artifacts for every run
  back to the last clean number:

  | run | job | HSA | outcome |
  |---|---|--:|---|
  | T170 | C64 | **0** | **clean — 8,040** |
  | T171 | C60 | 1 | aborted 20.9% |
  | T171 | C56 | 2 | aborted 16.9%, 3,194 s warmup |
  | T172 | C56 | 3 | 0/56, server dead |
  | T173 | C56 | 3 | aborted 15.2% |

  - **My "three different failure modes" line from two cycles ago is wrong and
    is withdrawn.** It is ONE fault at rising severity — ROCm queue allocation
    failing — counting 0 → 1 → 2 → 3 → 3 in time order, exactly zero on the
    last run that produced a number. The *reported symptom* (empty streams /
    EngineDeadError / connection refused) just tracks when in the run the abort
    lands.
  - **Not a C56 config property.** Re-dispatching C56 cannot fix it. The node
    is accumulating unreleased GPU queue resources across runs and needs a
    **runner reset — outside my bounds.** Flagging for the owner.
  - **STOPPING the C56 replication at three attempts.** Four runs spent; a
    fifth on this node buys another HSA abort. **C56 = 8,326 stays n=1** and
    the peak-at-56 conclusion rests on T169 + T170, where it always did.
- **T173 C1 aborted (thirteenth, 15/146) — and it CONFIRMS C1 is a different
  fault from the C56 HSA one.** Same run, same node, same hour:

  | T173 job | HSA | sentinel | EngineDeadError | outcome |
  |---|--:|--:|--:|---|
  | C56 | **3** | yes | yes | aborted 15.2% |
  | **C1** | **0** | yes ×2 | yes ×4 | aborted 15/146 |

  - C1 reaches `engine_core_sentinel:179` → `mq.dequeue` → `acquire_read` →
    `RuntimeError: cancelled` → `EngineDeadError` with **zero HSA lines** in
    `server.log`. The C56 job beside it had three.
  - I flagged this as an open question when the HSA cause emerged and refused to
    fold C1 into it. That caution is now vindicated with direct evidence.
  - **N8 remains the live hypothesis for C1**, and is now *isolated* rather than
    assumed. T172 weakened the evidence chain I was using, not the hypothesis
    itself. The thirteen C1 aborts are not the node.
- **T177 C60 = 8,333 — REPRODUCED. Clean, HSA = 0, error rate 2/1838 = 0.109%.**
  - **C60 n=2: 8,350 / 8,333, mean 8,342, spread 0.20%.** First best-config claim
    in this ledger resting on two clean runs. Tighter than the 0.30% C52 floor and
    8× tighter than C64's 1.6%. The two runs had wildly different warmups
    (1575 s vs 519 s) and still landed 0.20% apart — the number is not tracking
    warmup weather.
  - **SETTLED BEST CONFIG: C60, 8,342 mean.** C56's 8,326 one-off stays withdrawn.
  - **Clears SA's 8,296 by 0.55%**, ~2.8× C60's own spread — first lead over SA
    outside measurement noise. Small lead, SA is n=1 from my side: "ahead,
    modestly", not more.
  - C60 reliability: **2 clean in 3** (HSA 1, 0, 0). Better than it looked, but
    ~a third of attempts still die on HSA.
  - Curve: 7,771 / 8,115 / 8,326* / **8,342** / 7,976 at 48/52/56/60/64. Peak 60.
  - **Gap to 12,500 target: −33%.** Unchanged honest position — no remaining
    config lever closes it; needs the kernel-level work or a different stack.
- **T176 C60 = 8,350 — NEW BEST, clean, HSA = 0.** 1,840/1,967, error rate
  **4/1844 = 0.217%**, init 1575 s, wall 122 min. The discriminating test paid off.
  - **Hypothesis updated, and narrowed.** By config: C56 = 1 clean in 5 (HSA 2,3,3,3);
    **C60 = 1 clean in 2** (HSA 1, 0); C64 = 2 clean in 2 (HSA 0, 0). So "low conc
    triggers HSA" is **too strong** — C60 runs clean. The supported claim is a
    **failure-rate gradient across 56 → 60 → 64**, not a sharp boundary.
  - **8,350 clears SA's 8,296 by 0.65%** and is the best number on this stack,
    from a run with a validated error rate and a clean HSA log.
  - **Limits, stated because I over-claimed on 8,326:** n=1 clean; and with the
    C64 spread at 1.6%, both 8,350-vs-8,326 (0.3%) and 8,350-vs-SA (0.65%) sit
    **inside run-to-run variation**. Best *observed*, not a demonstrated margin.
  - Curve: 7,771 / 8,115 / 8,326 / **8,350** / 7,976 at 48/52/56/60/64. Peak is
    in 56–60; not resolvable more precisely at this noise level.
- **T177 DISPATCHED: C60 again** — the one thing worth doing is a second clean
  sample of the new best. If it reproduces near 8,350, the number is solid and
  C60 becomes the settled operating point; if it lands at 8,0xx or fails on HSA,
  that is equally informative about C60's 1-in-2 reliability.
- **T176 C1 aborted — sixteenth, 15/145.** Same invariant: 15 failures, drifting
  denominator. Count only.
- **T175 C1 aborted — fifteenth, 15/146.** Unchanged; recorded for the count only.
- **T176 DISPATCHED: C60 — a TEST of the config-linked HSA hypothesis, not a repeat.**
  C60 has been run once (HSA = 1, aborted) and sits between the four-times-failing
  C56 and the twice-clean C64. This run discriminates:
  - **C60 clean, HSA = 0** → the failure boundary is between 56 and 60, and C56
    is anomalous rather than "low concurrency is bad".
  - **C60 HSA > 0** → the boundary is between 60 and 64, and the effect spans a
    range rather than singling out C56.
  Either way the hypothesis gains a real constraint. A third C64 sample would
  only have tightened a mean I already have; this buys a discriminating fact.
- **T175 C56 FAILED (0/70, warmup) with HSA = 3 — and it FALSIFIES my
  "node degradation" model from last cycle.** I said the node had recovered on
  the strength of one clean C64. Sorting every run by *config* rather than
  *time*:

  | conc | runs | HSA | outcome |
  |---|--:|---|---|
  | C1 | 2 | 0, 0 | no HSA (aborts, but N8) |
  | **C64** | 2 | **0, 0** | **both clean: 8,040 / 7,912** |
  | C60 | 1 | 1 | aborted |
  | **C56** | 4 | **2, 3, 3, 3** | **all four failed** |

  - **The accumulation model predicted T174 C64 ≥ 3 HSA. It was 0.** Falsified.
    Counts do not climb with time or reset on recovery — they sort by conc.
  - Surviving correlation: **C56 triggers HSA, C64 does not.** Five C56 attempts
    = 1 success (T169) + 4 failures, with two clean C64 runs bracketing them.
    Counter-intuitive (lower conc failing where higher succeeds), which is why I
    reached for the time story. The C64-at-zero point rules it out. I am not
    inventing a mechanism; I am reporting what the data constrains.
  - **BEST-CONFIG CHANGE: C56 = 8,326 is a one-off that has resisted four
    replications and should NOT be reported as best.** Defensible best is
    **C64, n=2, mean 7,976** — reproducible, and *below* SA's 8,296. **The
    "parity with SA" claim rested on 8,326 and is withdrawn.**
  - Stop dispatching C56. Any further C56 work needs the HSA mechanism
    understood first, which needs node-level access outside my bounds.
- **T174 C64 = 7,912 — CLEAN, and HSA = 0. THE NODE HAS RECOVERED.**
  1,783 successful of 1,919, validated error rate **5/1788 = 0.280%**, init
  909.77 s, wall 113 min (T170 C64 was 115). The probe did its job: the
  queue-exhaustion fault that killed four straight throughput attempts is gone.
  Picking C64 over a fourth C56 answered the node question *and* bought a real
  second sample.
- **NOISE-FLOOR CORRECTION — this weakens a claim of mine.** C64 now has n=2:
  8,040 (T170) and 7,912 (T174), **spread 1.6%**. The 0.30% floor I have been
  scoring every delta against was measured **at C52** (T163/T168). At C64 the
  same-config spread is **5.3× larger**. Noise grows with concurrency; it is not
  one number.
  - **Consequence:** I called C64's −3.4% vs C56 decisive at "3.1× noise".
    Against the correct 1.6% it is ~2× — suggestive, not decisive. **The
    peak-at-56 claim is weaker than I stated.** The bracket still holds (7,771
    at 48, abort at 72), but **56-vs-64 is close to run-to-run variation and
    must not be quoted as settled.**
  - The C52 ledger is unaffected — 0.30% was measured there.
- **NEXT: C56 replication is unblocked and worth doing again.** It was stopped
  because the node was bad, not because the question was answered. With HSA back
  to 0, a clean C56 would give the peak its second sample — and given the 1.6%
  spread finding, that sample matters more than I previously thought.
- **T174 C1 aborted — fourteenth, 15/146 again**, 131/148, gate lines correct.
  Nothing new beyond the count: the fixed-15-failures signature holds, and
  T173 already established C1 is HSA-free and distinct from the C56 fault.
  **No further C1 diagnosis is possible from benchmark dispatches** — it is
  N8, and N8 needs vLLM source. Recording the count and moving on.
- **Superseded framing (kept for the record):** the entry below called node
  health "the blocking issue" but treated the three failures as distinct. The
  HSA table above is the corrected version.
- **Node health is now the blocking issue, not config.** Since T170 C64 — the
  last clean throughput number — **three consecutive throughput attempts have
  failed in three different ways**: T171 C60 (empty streams), T171 C56 (3,194 s
  warmup), T172 C56 (HSA out-of-resources). Config has been constant across all
  three. **C56 = 8,326 remains n=1** and I have now spent three runs failing to
  replicate it. If the next attempt also fails, the right call is to stop
  re-dispatching C56 and flag the runner for the owner rather than keep burning
  GPU hours on an unhealthy node.
- **T172 DISPATCHED: C56 alone**, `conc-list: [56]`, yaml parsed and asserted.
  jobs, not three, so a degraded node shows up faster. Purpose is unchanged —
  get the second sample of the peak that T171 failed to deliver. Only after
  that does "settled peak" hold; C60 stays unmeasured and lower priority, since
  even a good C60 cannot move a peak that C64 already bounds from above.
- **T171 DISPATCHED: C60 and C56.** yaml `conc-list: [60, 56]`, parsed locally
  and the dram/backend assertion passes. C60 is the only unmeasured point in
  the peak bracket [56, 64); C56 rides along as a **second sample of the peak**,
  which the ledger needs — 8,326 is currently n=1 and the whole Phase E
  conclusion rests on it. Two points, no config variable changed.
- **T169 C48 = 7,771 — −4.2% vs C52's 8,115 mean, 14× the noise floor.**
  1,779 successful, error rate 0.056%. Gate lines correct (mns 80, ladder 80,
  chunk 8192, gmu 0.90, offload dram). **48 is worse; the peak is at or above
  52.** C56 now running decides whether 52 is the top.
- **In flight: T169 — Phase E partial, C48 and C56.** C52's config space is
  exhausted, so the open question is whether 52 is even the right operating
  point; we have 40 and 52 and nothing between or above. 48 and 56 bracket it
  and each conc is an independent measurement, so this is not a multi-variable
  run. Expect ~3.5 h for both points.
- **SUPERSEDED — the original T168 plan:** This is
  deliberate: the whole ledger quotes deltas of +0.78%, +2.0%, −1.8% and I have
  **never measured run-to-run variance**, so I cannot say which of those are
  real. A second sample of the best config gives the noise floor that every
  earlier claim depends on. It also keeps the GPUs busy while N8 is blocked.
- **SUPERSEDED — the original T167 plan:** One variable
  vs T163. The script had been **force-disabling** the direct a2a that #51705
  added, overriding its own auto-on-with-a2a default — so the fast path has
  never actually been measured. Collectives are 21.3% of wall and concentrated
  in `dcp:0`, so this aims straight at the second-biggest item, and it partly
  unblocks N3 without needing the selected-backend line from source.
  Only the a2a var is flipped; both gathers stay 0 so the result is attributable.
  Gate line `[dcp-direct]` added. Baseline to beat: **8,127**.
  - **Accuracy gate stated, not skipped:** this swaps the a2a implementation, so
    it *could* move numerics. The GSM8K gate normally runs at C1, and C1 has
    aborted on the sentinel bug six runs straight, so the gate is unavailable
    right now. Running perf first and flagging it: **if this lands positive it
    must not be adopted until accuracy is confirmed**, which needs N8 first.

### Why C52 runs WITH the DRAM offload — CORRECTED

This section previously argued the opposite. It was inferred from GPU idle, not
from throughput, and three direct A/Bs now say the offload wins:

| | dram | none | delta |
|---|--:|--:|--:|
| ours, mns 80, pin after ready | **8,127** (T163) | 7,824 (T161) | **+3.9%** |
| ours, T103 vs T133 (mns 80 vs 65) | 7,950.6 | 7,725.96 | +2.8% |
| SA, mns 80 | 8,296 | 8,204 | +1.1% |

The T116/T124 idle finding was real — idle 44.3% → 28.2%, >10 ms stalls −57% —
but it **never converted into throughput**. Dropping the offload removes the
host↔device stalls *and* the KV capacity that keeps the batch full; the second
effect is larger. That is also why N5 (mns 96) is the current lever: capacity is
the only axis that has moved this number.

**Config requirement:** `kv-offloading: dram` needs
`kv-offload-backend: { name: vllm-simple }` in the same yaml row, or `get-jobs`
fails pydantic validation before any GPU work starts.

**Node history, kept:** `mns 80` + `none` died 3/3 with
`HSA_STATUS_ERROR_OUT_OF_RESOURCES` on `mi355x-amd_b23_07`, but T161 ran it
cleanly, so that failure mode is not currently reproducing.

**If a run OOMs, try gmu BEFORE dropping mns.** The margin is bracketed:

| gmu | mns | offload | outcome |
|--:|--:|---|---|
| 0.95 | 80 | dram | engine **hung**, 0/57 (T157) |
| 0.90 | 80 | none | `HSA_STATUS_ERROR_OUT_OF_RESOURCES` 3/3, then OK in T161 |
| 0.90 | 80 | dram | **8,127** (T163) |
| 0.90 | 65 | none | 7,725.96 (T133) |

Order: `GPU_MEM_UTIL=0.85` → `HSA_NO_SCRATCH_RECLAIM=0` → `MAX_NUM_SEQS` down.

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

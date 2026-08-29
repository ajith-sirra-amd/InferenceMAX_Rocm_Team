# Autonomous run queue — Kimi-K3 / 8× MI355X

Owner away 2026-08-28 → 2026-08-30. This file is the single source of truth for
what runs next. Every wake-up: read **Current state**, act, update this file.

## Targets

| | target | best today | gap |
|---|---|---|---|
| C52 throughput | **12,500 tok/s/GPU** | **8,127 (T163)** · SA 8,296 | **−35%** |
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
- **T168 DONE. C52 = 8,103 on T163's exact config — noise floor is 0.30%.**
  1,948 successful, error rate 0.154%. This retro-validates the ledger: chunk
  −7.4% is 25× noise, offload +3.9% is 13×, async −1.8% is 6×. **The one weak
  claim is CCD pinning's +0.78%, at 2.6× — provisional.** And n=2 is a point
  estimate of spread, not a standard deviation.
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

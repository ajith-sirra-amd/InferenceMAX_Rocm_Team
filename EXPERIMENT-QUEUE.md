# Autonomous run queue — Kimi-K3 / 8× MI355X

Owner away 2026-08-28 → 2026-08-30. This file is the single source of truth for
what runs next. Every wake-up: read **Current state**, act, update this file.

## Targets

| | target | best today | gap |
|---|---|---|---|
| C52 throughput | **12,500 tok/s/GPU** | **7,968 (T160)** · SA 8,296 | **−36%** |
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
- **In flight: T161 C52 = N1**, pinning moved one-shot to after
  `wait_for_server_ready`, background pre-pin loop deleted. One variable vs T160.
  Expect throughput ≈ 7,968 and ~23 min less wall. If throughput drops, the gain
  came from pinning the loader threads too and N1 must be reverted.
- **NEXT (user-requested): run `sa.sh` at C1 + C52.** Already staged — sa.sh
  copied over `kimik3_fp4_mi355x_mtp.sh`, image reverted to
  `aigmkt/kimi-k3-vllm:latest`. **This doubles as the crash A/B**: sa.sh has no
  runtime patch, no CCD pinning, and runs the aigmkt image on which T123
  completed 190/193. If C1 still aborts at ~10% the crash is not caused by the
  nightly or the rebase; if it completes, it is.
  Config: C1 dcp=1 k=8 mns=8 ladder 1..16; C52 dcp=8 k=0 mns=80 ladder 1..80,
  **kv-offloading: none**.

### Why C52 runs WITHOUT the DRAM offload

I had staged it with `dram` because sa.sh pairs mns 80 with the offload. That
ignored the measured reason the offload was dropped (T116 vs T124, same point,
both traced so rocprof overhead cancels):

| | offload ON | offload OFF |
|---|--:|--:|
| GPU idle | **44.3%** | **28.2%** |
| >10 ms stalls | 265.7 s, n=4,104 | 114.6 s, n=877 (**-57%**) |
| collectives, % busy | 34.31% | 29.44% |

The multi-millisecond stalls *were* the offload's host<->device traffic; the
sub-200 us launch gaps did not move, which is the expected signature.

**Known risk, stated:** `mns 80` + `kv-offloading: none` is the combination that
died 3/3 with `HSA_STATUS_ERROR_OUT_OF_RESOURCES` on `mi355x-amd_b23_07`. It is
**not** a config limit -- SA ran exactly that on `mi355x-amds_01` for 8,204
tok/s/GPU -- but it is a limit on OUR node. If it OOMs, that is the node, and
the fallback order is below.

**If it OOMs, try gmu BEFORE dropping mns.** The margin is bracketed:

| gmu | mns | offload | outcome |
|--:|--:|---|---|
| 0.95 | 80 | dram | engine **hung**, 0/57 (T157) |
| 0.90 | 80 | none | `HSA_STATUS_ERROR_OUT_OF_RESOURCES` 3/3 (our node) |
| 0.90 | 80 | dram | 7,950.6 (T103) |
| 0.90 | 65 | none | 7,725.96 (T133) |

Headroom is already marginal at 0.90 -- 0.95 hung it -- so mns 80 failing is
plausibly the same resource. Order:

1. **`GPU_MEM_UTIL=0.85`**, mns 80, none. Cheap, no numerics change.
2. **`HSA_NO_SCRATCH_RECLAIM=0`**. The script sets `1`, keeping scratch
   allocated rather than returning it. `HSA_STATUS_ERROR_OUT_OF_RESOURCES` is
   HSA runtime exhaustion -- queues, signals, scratch -- not a plain HBM OOM, so
   this targets it more directly than gmu.
3. `MAX_NUM_SEQS=65` (7,725.96) last, since it concedes the mns-80 point SA gets
   8,204 from -- the limit is our node, not the config.
- **Then:** root-cause the crash from `results/server.log` (the runner console
  log stops at `Application startup complete`; the engine trace is in that
  artifact).

### OPERATIONAL: DRAM offload and slow model loads — PARTLY RETRACTED

**Correction.** I previously wrote that the DRAM offload poisons the *next*
run's model load by ~30x (46.2 s/shard). That conclusion was built on a
`Loading safetensors checkpoint shards` line that is from an **SA run, not
ours**. I misattributed it. The 46.2 / 51.6 s/shard figures and the "~30x,
~70 min of idle on the next run" claim are **withdrawn** — I have no shard-rate
data for our own runs, because the log blob for T154 C1 has never flushed.

**What IS measured on our runner:**

| | |
|---|---|
| C52 offload allocation | `SimpleCPUOffloadConnector: 243,625,000,000 B/rank` x 8 = **1.949 TB** host DRAM (`TOTAL_CPU_DRAM_GB: 1949`) |
| C52 weight load (that same job) | **576.66 s** |
| C1 weight load (T145/T147/T151) | **149-155 s** |
| serve -> ready | C52 **19.4 min**; C1 **6.5-7.9 min** |

So our C52 arm loads ~3.8x slower than our C1 arms. That is the job that
*allocates* the offload, not one following it, so its own offload does not
explain it. Cache state, the larger `mns 80 / ladder 1..80` capture, and DCP
init are all unseparated here. **Untested hypothesis, do not act on it as fact.**

**Queue rule, kept but on weaker grounds:** screen C52 with
`kv-offloading: none` + `mns 65` (T133 = 7,725.96) and use `dram` + `mns 80`
(T103 = 7,950.6) only for a final confirmation. Justification is now simply that
the offload is worth ~2.9% and adds ~12 min of startup per run, not the
retracted next-run penalty.

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

### N1 — CCD pinning, applied AFTER `wait_for_server_ready` *(next script edit)*

T160 measured **2008.62 s** to load weights (vs 576–681 s unpinned): the pre-pin
loop confines the ~190 loader threads per worker to one CCD's 8 physical cores
during weight load and cudagraph capture. Pinning must be **one-shot, after the
server reports ready** — steady-state locality is the thing we want; load and
capture are one-time and must run across all cores. This also makes C3
measurable for the first time (T160's number is confounded by the load penalty).
Remove the background pre-pin loop entirely.

### N2 — async scheduling, retested on the nightly *(largest addressable item)*

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

### N4 — AITER tuned GEMM configs

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

# Autonomous run queue — Kimi-K3 / 8× MI355X

Owner away 2026-08-28 → 2026-08-30. This file is the single source of truth for
what runs next. Every wake-up: read **Current state**, act, update this file.

## Targets

| | target | best today | gap |
|---|---|---|---|
| C52 throughput | **12,500 tok/s/GPU** | 7,950.6 (T103) · SA 8,296 | **−57%** |
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

- **T157 gmu 0.95 = HARD FAIL.** 0/57 requests at C52; KV pool grew +26% then
  every warmup request hung for 1200 s with zero errors. Not an OOM. **Reverted
  to 0.9 — settled, do not retry.** C1 arm cancelled since the config is known
  bad.
- **T156 C52 = 7,906** (nightly flat vs T103 7,950.6). **T156 C1 = 1,509
  tok/s/GPU / ITL p50 7.89 ms but DIRTY** (11.5% drops) — still needs a clean
  re-run.
- **In flight: T158** — C52 with `NCCL_MIN_NCHANNELS=32`. Collectives are 21.3%
  of e2e wall and RCCL has never been tuned. No numerics change.
- **Next:** #52190 torch.compile (numerics -> GSM8K first), then CCD pinning,
  then a clean C1 re-run.

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

## Queue — run in order, one variable per run

### Phase B — establish the both-points baseline on nightly + #51705
- **B1** `FORCE_EVAL=0`, agentic, C1 + C52. Headline numbers for the new stack.
  Compare C52 vs T103 7,950.6 / SA 8,296; C1 vs T123 6.70 agentic.

### Phase C — C52 throughput, highest evidence first
- **C1** *(strongest)* **AITER/hipBLASLt tuned GEMM configs.** One C52 run logs
  **45,250** `not found tuned config in /tmp/aiter_configs/bf16_tuned_gemm.csv`.
  Dense GEMM is **11.9% of e2e wall**. Top shapes: `M:935 N:6288 K:7168`,
  `M:935 N:3584 K:7168`, `M:7928 N:3072 K:512`, `M:640 N:6288 K:7168`, and at
  C1 `M:7 N:20480 K:7168`, `M:7 N:7168 K:35840`, `M:7 N:2880 K:7168`.
  No numerics change → no GSM8K needed.
- **C2** **#52190 — torch.compile is silently disabled.** Log still says
  `torch.compile is turned on, but the model does not support it`, so we run
  with **zero post-grad fusion** despite `fuse_allreduce_rms`, `fuse_norm_quant`,
  `fuse_mla_dual_rms_norm` all configured true. Numerics change → **GSM8K first**.
- **C3** **CCD / NUMA pinning.** Written, archived, never measured. Workers run
  unpinned (`0-255`) with GPU0-3 threads seen on node1 cores. One 32 MiB L3 per GPU.
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

# Kimi-K3 / vLLM — Learnings

A working reference for the knobs, metrics and failure modes we have actually hit
on **Kimi-K3 FP4, 8× MI355X, TP8 / DCP8, spec-decode MTP, SimpleCPU dram offload**.

Everything marked **[measured]** is a number from our own runs. Everything else is
mechanism. Where we do not know, it says so.

---

## 1. CUDA graphs

### What they are and why

Each decode step launches thousands of tiny GPU kernels. At batch sizes typical of
decode, the CPU cannot enqueue kernels faster than the GPU drains them — the GPU
goes idle waiting for launches. A **CUDA graph** records one forward pass as a
frozen DAG of kernel launches and replays it with a single call, removing per-kernel
launch overhead.

The cost: a graph is **static**. Shapes, addresses and control flow are baked in at
capture. So vLLM must capture one graph **per batch size** it wants to serve, and
every buffer the graph touches must live at a fixed address for the process
lifetime.

### The four modes

| mode | what is captured | needs torch.compile? | memory | when it fits |
|---|---|---|---|---|
| `NONE` | nothing, all eager | no | none | debugging; very large batches where launch overhead is amortised anyway |
| `PIECEWISE` | graphable *sub-regions*; the rest eager | **yes** (or breakable) | moderate | models with dynamic control flow — attention with variable-length KV |
| `FULL` | one graph for the **whole** forward | no | graph pool only | model is fully static end-to-end |
| `FULL_AND_PIECEWISE` | both: full graphs for decode, piecewise for prefill/mixed | **yes** (or breakable) | highest | the usual production default |

`has_piecewise_cudagraphs()` is true for `PIECEWISE` and `FULL_AND_PIECEWISE`. That
predicate is the whole story behind the trap in §1.4.

### Breakable CUDA graphs

`VLLM_USE_BREAKABLE_CUDAGRAPH=1` is an **alternative way to get piecewise graphs
without torch.compile**. Instead of the compiler marking graph-safe regions, the
runtime records the forward and *breaks* the recording at ops it cannot capture,
stitching the graphable spans together.

It works — but it instantiates a persistent runner (`init_breakable_cg_runner`)
holding its own set of static buffers, and those buffers are **large**.

### The trap we hit (T278 → T281)

The guard, in `vllm/v1/worker/gpu/cudagraph_utils.py:543`:

```python
if self.cudagraph_mode.has_piecewise_cudagraphs() and not (
    self.use_breakable_cg or has_compiled_submodule(model)
):
    raise RuntimeError("... piecewise CUDA graphs unavailable, model is not "
                       "torch-compiled and breakable CUDA graph is off. "
                       "Set VLLM_USE_BREAKABLE_CUDAGRAPH=1 or cudagraph_mode=NONE/FULL.")
```

Read it as: *you asked for piecewise graphs, but you gave me no way to make them.*

For Kimi-K3, **`has_compiled_submodule(model)` is false** — the model is not
torch-compiled despite `--compilation-config '{"mode":3,...}'`. Older nightlies did
not enforce this; `nightly-1970f3ed` (09-06) onward do. So the same launcher that
booted for months suddenly died at engine init.

Setting `VLLM_USE_BREAKABLE_CUDAGRAPH=1` fixed the boot **and silently cost 35% of
the KV cache**:

| config | KV/GPU | KV tokens | note |
|---|---|---|---|
| `FULL_AND_PIECEWISE`, breakable=0, base `7c5dc571` | **49.79 GiB** | **28,653,478** | **[measured]** T274 → 11,095 tok/s/GPU |
| `FULL_AND_PIECEWISE`, breakable=1, base `1970f3ed` | 32.14 GiB | 18,475,453 | **[measured]** T278 v2 |
| `FULL_AND_PIECEWISE`, breakable=1, base `d9105ea8` | 32.14 GiB | 18,475,453 | **[measured]** T280 — *identical*, so the base is innocent |

**Cost of the flag: 17.65 GiB per GPU, −35.5% of the KV pool.**

Two lessons worth more than the numbers:

1. **An identical fingerprint across two different bases isolates the cause.** We
   suspected 21 upstream commits; the matching KV number ruled them out in one
   comparison, with no extra run.
2. **Attribute the memory before turning knobs.** I first blamed the breakable
   runner's buffers. Wrong — see the breakdown below: the cost is the **piecewise
   graph pool**, and breakable graphs and torch.compile are two routes to the same
   capture. Raising `gpu-memory-utilization` was also wrong (it claws back ~11.6%
   against a ~40% need, and spends OOM headroom). Only *not capturing piecewise*
   returns the memory.

### The full arc: how one guard cost us 35% of the KV pool

This is the most instructive sequence in the campaign, so it is worth in full.

| stage | mode requested | breakable | model compiled? | outcome | KV/GPU |
|---|---|---|---|---|---|
| **Before** — T274, base `7c5dc571` | `FULL_AND_PIECEWISE` | **0** | no | worked; old engine did not enforce the guard | **49.79 GiB** → 11,095 tok/s/GPU |
| **Guard appears** — T278 v1, base `1970f3ed` | `FULL_AND_PIECEWISE` | **0** | no | **engine init dies** | — |
| **Breakable workaround** — T278 v2, T280 | `FULL_AND_PIECEWISE` | **1** | no | boots, breakable runner allocates buffers | **32.14 GiB** (−35.5%) |
| **Try to dodge piecewise** — T281 | `FULL` | 0 | no | **fails** — engine resolves FULL → FULL_AND_PIECEWISE, guard fires | — |
| **Root-cause fix attempt** — T282 | `FULL_AND_PIECEWISE` | **0** | **yes**, via #52190 | boots, but **recovers nothing** — 32.28 GiB | 18,555,236 |
| **Diagnosis** — T284 | `NONE` | 0 | n/a | proves the pool is the cost | **54.21 GiB / 31,217,931** |
| **Resolution** — T285 | **`FULL_DECODE_ONLY`** | 0 | not needed | guard never fires; decode graphs kept | expect ~31 M |

Three things this teaches:

1. **We were relying on a check not existing.** `FULL_AND_PIECEWISE` at breakable=0
   with an uncompiled model was never *valid* — older nightlies simply did not
   verify it. When the guard landed, a config that had run for months died at init.
   An upgrade did not break us; it revealed that we were already broken.

2. **`FULL` is not reachable for this model.** We passed `"cudagraph_mode":"FULL"`,
   the engine logged `CUDAGraphMode.FULL`, and then **resolved it back** to
   `FULL_AND_PIECEWISE` — 26 occurrences in the log — because the model has
   non-graphable regions that pure FULL cannot express. *A config value you set is
   not necessarily the config value that runs.* Always grep the log for what the
   engine actually resolved.

3. **Pick the branch that removes the cause, not the one that silences the error.**
   The guard has three escapes; only the third is free:

   | escape | works? | cost |
   |---|---|---|
   | `VLLM_USE_BREAKABLE_CUDAGRAPH=1` | yes | **−17.65 GiB KV/GPU** |
   | `cudagraph_mode=NONE` | yes | loses all graph speedup, incl. decode |
   | `cudagraph_mode=FULL` | **no** — silently upgraded back | — |
   | torch.compile the model (#52190) | boots, but **recovers no memory** | same 17.6 GiB — it enables the same piecewise capture |
   | **`cudagraph_mode=FULL_DECODE_ONLY`** | **yes** | **none** — guard never fires, decode graphs kept |

**Correction to my own reasoning above:** I predicted torch.compile would recover
the memory. It did not — 18,475,453 → 18,555,236 tokens, +0.4%. Both it and the
breakable flag *enable piecewise capture*, and the capture is the cost. The real
answer was a mode that never asks for piecewise at all.

**Still worth noting:** because K3 was never torch-compiled, its **post-grad fusion
passes were silently inert** — `aiter::fused_qk_rmsnorm_kernel` and
`aiter::allreduce_fusion_kernel_1stage` never ran. So enabling torch.compile is not
only a memory fix; it may also switch on optimisations we believed were already
active. A useful reminder that *configured* and *effective* are different things.

### Where the memory actually is — and the mode that fixes it

I got this wrong twice before the numbers settled it. The breakdown vLLM prints at
startup (`gpu_worker.py`, the `non-torch` line) is the ground truth:

| config | weights+non-torch | peak activation | **CUDAGraph mem** | KV/GPU | KV tokens |
|---|---|---|---|---|---|
| T274 — `FULL_AND_PIECEWISE`, piecewise **silently absent** | 199.79 | 9.47 | **2.66 GiB** | 49.79 | 28,653,478 |
| T280 — breakable=1 | 199.08 | 26.77 | **20.30 GiB** | 32.14 | 18,475,453 |
| T282 — torch.compile, breakable=0 | 199.84 | 26.94 | **20.30 GiB** | 32.28 | 18,555,236 |
| T284 — `cudagraph_mode=NONE` | — | — | **~0** | **54.21** | **31,217,931** |

**The cost is the piecewise graph pool, not the breakable runner.** Breakable graphs
and torch.compile are two different *routes to the same thing* — piecewise capture —
so swapping one for the other recovered nothing (18,475,453 → 18,555,236, +0.4%).
Peak activation also triples, because capture pins intermediate buffers.

**`FULL_DECODE_ONLY` is the mode you want on a prefill-bound workload:**

```python
NONE               = 0
PIECEWISE          = 1
FULL               = 2
FULL_DECODE_ONLY   = (FULL, NONE)        # decode_mode=FULL, mixed_mode=NONE
FULL_AND_PIECEWISE = (FULL, PIECEWISE)
```

| mode | decode | mixed/prefill | piecewise pool | KV tokens |
|---|---|---|---|---|
| `FULL_AND_PIECEWISE` | full graph | piecewise | 20.3 GiB | 18.5 M |
| `NONE` | eager | eager | 0 | 31.2 M |
| **`FULL_DECODE_ONLY`** | **full graph** | eager | **0** | expect ~31 M |

`has_piecewise_cudagraphs()` is false for it, so the guard never fires — **no
breakable flag, no torch.compile, no patch.** It keeps graphs where they pay (pure
decode, small kernels, CPU-starved GPU) and drops them where they don't (mixed
batches carrying 16 k-token prefill chunks, where a 5–10 µs launch is noise against
millisecond GEMMs).

**Can you just revert the nightly's guard?** No. `_init_candidates()` builds capture
descriptors from `cudagraph_mode` alone — `decode_mode()` / `mixed_mode()` — with no
reference to whether a piecewise mechanism exists. `FULL_AND_PIECEWISE` therefore
*always* creates PIECEWISE descriptors and the capture loop iterates
`[PIECEWISE, FULL]`. Delete the guard and it attempts piecewise capture with nothing
to produce it, failing deeper and less legibly. The guard is a fail-fast for a real
incompatibility, not the change itself.

### Reasoning about graphs from the workload, not from defaults

The single most useful frame: **CUDA graphs only buy back kernel-launch overhead,
and launch overhead only matters when the GPU is starved.**

| | our numbers | graphs worth it? |
|---|---|---|
| prefill | 16,384-token chunks, GEMMs run for ms, `tput_in` ~91,000/s | **no** — µs of launch against ms of work |
| decode | batch 1–96, thousands of small kernels, `tput_out` ~580/s | **yes** |
| share of total work | prefill ≈ **99%** | — |

So paying **10.2 M KV tokens** to accelerate ~1% of the work is not a two-sided
trade — it is a near-certain loss, and it can be reasoned out *before* spending an
hour measuring it. Check which phase dominates (`tput_in` vs `tput_out`) before
touching any graph knob.

**Does eager prefill hurt more without `--async-scheduling`?** In principle yes —
without async the host already serialises between steps, and eager adds host work
inside the step too. In practice, no:

1. T274 ran exactly this combination — eager mixed batches, `--no-async-scheduling` —
   and scored **11,095**, our best number.
2. `--async-scheduling` measured **−1.8%**. If the host were the bottleneck,
   overlapping host prep with GPU execution would have helped. It hurt — so the host
   is not limiting, and launch overhead will not make it so.

The exception is **C1**, where there is no prefill to hide behind and decode
dominates. That is precisely why `FULL_DECODE_ONLY` beats `NONE`: it keeps decode
graphs.

### The capture ladder

```
graphs: dense ladder 1..96 (mns=96 x 1 rows), DCP=8
```

- One graph per batch size, **1 through 96** — a *dense* ladder. Stock vLLM uses a
  sparse ladder (1, 2, 4, 8, 16, …).
- `rows = num_speculative_tokens + 1` when spec-decode is on: each speculative depth
  is a distinct token count and needs its own graph.
- Rule in our launcher: the ladder **always** covers `mns × SPEC_ROWS`, never clamped
  below it. A missing size silently falls back to eager for that batch — a
  perf cliff that shows up as latency variance, not as an error.

---

## 2. KV cache and how the pool is sized

```
Available KV cache memory: 49.79 GiB      # per GPU, after everything else
GPU KV cache size: 28,653,478 tokens      # aggregate across all 8 ranks
```

The arithmetic:

```
HBM per GPU (288 GB) × gpu-memory-utilization
  − model weights (sharded by TP)
  − activation peak
  − cudagraph pool
  − breakable-CG runner buffers   ← the 17.65 GiB
= Available KV cache memory
```

**[measured]** per-token KV for our config ≈ **14.9 KB** (427.7 GB total pool ÷
28.65 M tokens).

### KV fingerprints as config signatures

The token count is a deterministic function of the whole config, which makes it the
fastest way to confirm what actually ran. Our catalogue:

| KV tokens | config |
|---|---|
| 28,653,478 | gmu 0.90, mnbt 16384, breakable=0 — **the baseline** |
| 30,169,355 | gmu 0.90, mnbt 8192 |
| 31,981,568 | gmu 0.92 |
| 27,319,963 | gmu 0.90, mnbt 32768 |
| 28,220,371 | q-replicate active |
| 26,932,446 | gmu 0.88 stock |
| 18,475,453 | breakable=1 |

**Habit worth keeping: read the fingerprint before trusting any result.** It caught
T272 running with a stale `mnbt`, and it isolated the breakable-CG cost.

### gpu-memory-utilization is a threshold, not a slope

**[measured]** 0.88 → 0.90 was **+16.9%** throughput at C48. 0.90 → 0.92 was
**+0.20%** at C72 — nothing. In both cases `kv_usage` sat at 40–55%, so neither was
capacity-bound; 0.90 is simply above the knee. A knob that is dead in one regime can
come alive in another, and vice versa — always check whether the resource is
actually binding before turning it.

*Node caveat:* `gmu > 0.90` has hung this node before (T166 at 0.92 returned 0/103).
T272 later ran 0.92 cleanly at C72, so it is config-dependent, not universally fatal.

---

## 3. Parallelism

| flag | meaning | ours |
|---|---|---|
| `--tensor-parallel-size` (TP) | split each weight matrix across GPUs; all-reduce every layer | 8 |
| `--enable-expert-parallel` (EP) | shard MoE experts instead of replicating | 1 (off) |
| `--decode-context-parallel-size` (DCP) | shard the **KV cache along sequence length** across ranks during decode | 8 |
| `--dcp-comm-backend` | how DCP ranks exchange partial attention | `a2a` |
| `--cp-kv-cache-interleave-size` | granularity of the sequence-dim split | 1 |

**DCP is the one that surprises people.** With TP alone, every rank holds the whole
KV for its head slice. With DCP, the *sequence* is split too, so each rank holds
1/8 of the tokens and they exchange partial attention outputs plus LSE
(log-sum-exp) terms to reconstruct the true softmax. That is why:

- long contexts become affordable (our ISL median is ~90 k tokens),
- **only full attention is DCP-sharded** — Mamba/recurrent, sliding-window and
  chunked-local specs stay at `dcp_world_size=1`. Assuming otherwise is the bug
  #54735 fixes,
- the a2a and LSE-mask paths become hot enough that fusing them is a real PR
  (#54889).

**Gotcha [measured]: DCP > 1 and MTP are mutually exclusive.** The MTP draft model
uses `TRITON_MLA`, which rejects non-causal MLA under DCP:
`Selected backend AttentionBackendEnum.TRITON_MLA is not valid ... ['non-causal MLA
attention with DCP not supported']`. Any PR that only shows at C1 with DCP is
untestable while MTP is on.

---

## 4. Speculative decoding (MTP / DSpark)

```
--speculative-config '{"model":"Inferact/Kimi-K3-DSpark","num_speculative_tokens":k,
                       "method":"dspark","attention_backend":"TRITON_MLA","kv_cache_dtype":"fp8"}'
```

- A small **draft** model proposes `k` tokens; the target verifies them in one
  forward and accepts the longest correct prefix.
- **Acceptance length (AL)** is the payoff metric: AL 4.0 at k=8 means ~4 tokens per
  target forward instead of 1.
- Costs: draft compute, a second KV cache (`DRAFT_KV_DTYPE=fp8`), and
  `SPEC_ROWS = k + 1` extra cudagraph rows.
- Spec-decode interacts badly with prefix caching in ways that are still being
  fixed upstream (#54163, #54165) — the family is "prefix-cache hits structurally
  dropped under DCP / hybrid / spec".

---

## 5. Prefix caching and the KV offload tiers

Three tiers, and the log reports each separately:

| tier | where | metric | our value |
|---|---|---|---|
| GPU prefix cache | HBM | `prefix_cache_hit` | **[measured]** 73–90%, falls as unique traffic grows |
| external / offload | CPU dram (or disk) | `ext_cache_hit` | **[measured]** 0% → 82.6% across one hour, still climbing |
| theoretical ceiling | what the trace offers | `theoretical_prefix_cache_hit` | **[measured]** ~92–95% |

The gap between **theoretical 92.6%** and **captured 73.4%** is the headroom
everything in the PR backlog is chasing.

### `kv-offloading: dram` with `vllm-simple`

```
SimpleCPUOffloadConnector: 226.89 GB/rank × 8 ranks, 11473 blocks, mode=eager
```

**[measured] It is worth 2.57×.** T277 disabled it and throughput fell from 11,095
to 4,325 tok/s/GPU (−61%), with TTFT p50 1,560 → 3,128 ms. It is not optional. Every
stall we have seen involves this path, but the stalls are a **price**, not a bug to
route around.

Capacity is *not* the limit: 1,815 GB total ÷ 14.9 KB per token ≈ **121 M tokens**,
against **87 M** unique tokens generated in an hour. The tier never saturates inside
the measurement window because the trace keeps presenting new content, not because
it is evicting.

### LMCache — parked

`ext_cache_hit` was **0.0%** on every LMCache run, with 1,535 instances of
`No GPU context found for model ... with world size 8 during lookup!`. Every
retrieve failed; the host tier was write-only. **The same 0.0% appears in
SemiAnalysis's own logs**, so it contributes nothing to their number either.

---

## 6. Batching and scheduling knobs

| knob | what it controls | ours | evidence |
|---|---|---|---|
| `--max-num-batched-tokens` (mnbt) | chunked-prefill chunk size | **16384** | 32768 **deterministically fails** on the dram-offload path — T275 and T276 both died in warmup on the *same trace* |
| `--max-num-seqs` (mns) | max concurrent sequences | 96 | drives the cudagraph ladder height |
| `--enable-prefix-caching` | reuse KV across requests | on | the whole game for agentic traces |
| `--async-scheduling` | overlap host scheduling with GPU execution | **off** | **[measured] settled negative: −1.8%** (T162 7,686 vs T161 7,824) |
| `--kv-cache-dtype` | KV precision | `fp8` | halves KV bytes/token vs bf16 |
| `--attention-backend` | main attention kernel | `ROCM_AITER_MLA` | |
| `--attention-config mla_prefill_backend` | prefill-specific kernel | `ROCM_AITER_FA` | prefill and decode want different kernels |
| `--eviction-policy` | offload-tier eviction | `LRU` | |

**Why mnbt matters here:** our workload is **prefill-dominated** — ISL median ~90 k
vs OSL mean ~823, `tput_in` ~91,000/s vs `tput_out` ~580/s. Nearly all GPU time is
prefill, chunked at `mnbt`. That makes it look like the obvious lever, which is why
we tried 32768 — and why the deterministic failure was worth knowing.

---

## 7. Reading the runtime metrics line

```
srv  prefix_cache_hit=82.5% unique_in_srv=87,245,854 ext_cache_hit=82.6%
     kv_usage=48.7% queue=69r/0w tput_in_srv=91,368/s tput_out_srv=580/s
```

| field | meaning | what to watch for |
|---|---|---|
| `prefix_cache_hit` | GPU-tier hit rate | falls as unique content accumulates |
| `ext_cache_hit` | offload-tier hit rate | 0% means the tier is write-only — broken |
| `kv_usage` | fraction of KV pool in use | >80% risks eviction; ~50% means headroom |
| `queue=Nr/Mw` | running / waiting | `w > 0` sustained = admission-limited |
| `tput_in` vs `tput_out` | prefill vs decode token rate | ratio tells you which phase you are optimising |
| `intvty p50/p95/p99` | inverse TPOT (tok/s per user) | p99 ≫ p50 means stragglers |

Latency terms:

- **TTFT** — time to first token: prefill cost.
- **TPOT / ITL** — inter-token latency: decode cost. What users feel as "speed".
- **E2E** — whole request.

---

## 8. Failure modes we have actually seen

| symptom | cause | response |
|---|---|---|
| Engine init: *"piecewise CUDA graphs unavailable"* | model not torch-compiled, breakable off | `cudagraph_mode=FULL`, or breakable=1 (costs 17.65 GiB KV) |
| Warmup frozen, **GPUs at 0% util**, requests in flight, 0 errors | host-side spin in the SimpleCPU offload path | intermittent at mnbt 16384; **deterministic at 32768** |
| Same trace ID aborts twice | deterministic, not flaky | trust it — two identical failures is a verdict |
| VRAM stays at 99% after a run, PIDs gone | zombie KFD entries, driver-level leak | **wait — it self-clears, measured at ~21 min on 2026-09-10 (not the ~1 h inferred from T273).** ALL THREE alternatives are tested and useless: `docker stop -t 60` takes 73 s and ends in SIGKILL anyway because the workers are wedged in the RCCL busy-wait and never process SIGTERM; letting the RCCL watchdog kill the engine leaves the node just as dirty; `rocm-smi --gpureset` is a verified no-op. Waiting is not the fallback, it is the only thing that works. `kill -9` says "No such process". **`sudo rocm-smi --gpureset -d N` does NOT help** — verified 2026-09-09: it reports *"Successfully reset GPU 0"* and VRAM stays at 99%. There is no live process to detach, so the reset has nothing to release. Only waiting, or a node reboot, clears it |
| VRAM **rising** after you cancelled a job | **`gh run cancel` does NOT stop the container.** The job goes green-cancelled while `bmk-server` keeps running and holding memory | `docker ps` after **every** cancel, then `docker rm -f bmk-server`. The `rm` may report *"did not receive an exit event"* and still succeed — check `rocm-smi`, not the exit code |
| GH job green but `Requests: 0 successful` | result-writer swallows the non-zero exit | **never trust the green check** — grep `Requests: N successful` |
| `InvalidInferenceResultError` at 4–15% | environmental, not ours | it cleared on its own after T274 (0.17%) |

---

## 9. Method notes that cost us runs

- **One variable per run.** T272 v1 changed `mnbt` *and* the thing under test,
  because a previous arm had left `mnbt` at 8192. Check **every** gate line against
  the intended config, not just the knob you meant to move.
- **Gate numerics before perf.** Any change to dtypes, kernels, caching or cache
  reuse gets a GSM8K-200 first. This campaign once produced a 9,482 number that had
  to be thrown away because it was never validated.
- **Fetch patches fresh.** A build reused a 3-day-old cached diff because it guarded
  the fetch with `[ -f file ]`. The PR had been rebased twice; results would have
  been attributed to code we were not running. **Record the head SHA in the image
  manifest.**
- **Cancelling a run is not the same as stopping it.** Three cancels in one window;
  on the third the node read 93% and *climbing* while GitHub showed the run
  cancelled. A pre-dispatch check does not catch this — the orphan appears *after*
  the cancel. Always `docker ps` in the cancel path itself.
- **Dispatch resolves the branch, not your worktree.** Uncommitted or unpushed edits
  silently run the *old* config.
- **n=1 is not a result.** Cross-day noise is **±1.2%**; same-session pairs replicate
  to ~0.4%. Anything under ~1.2% needs n=2 before it is claimed.
- **Drop a fix once it stops being a fix.** #52190 (torch.compile) was applied to
  satisfy the cudagraph guard. Switching to `FULL_DECODE_ONLY` made the guard
  irrelevant — and #52190 silently became an *uncontrolled third variable* in what
  was supposed to be a clean baseline. A workaround that outlives its cause becomes
  a confound. Re-audit the config after every root-cause fix.
- **A patch that applies is not a patch that is current.** #54736 applied to one
  nightly and failed on another purely because the author rebased; #54165 did the
  reverse. Always dry-run against the exact base you will build.

---

## 10. Settled negatives — do not re-litigate

| thing | verdict | evidence |
|---|---|---|
| `--async-scheduling` | **−1.8%** | T162 vs T161 |
| `kv-offloading: none` | **−61%** | T277 |
| `mnbt 32768` | deterministic warmup failure | T275, T276 — same trace both times |
| LMCache | `ext_cache_hit` 0.0% for us *and* SA | six runs |
| gmu 0.92 | +0.20%, neutral | T272 |
| `VLLM_DCP_Q_REPLICATE=1` (#54494) | +0.09%, neutral at C72 | T267 |
| Trimming the cudagraph ladder to recover the 17.65 GiB | wrong target — graph pool is a few hundred MB | reasoning, not a run |

---

## 11. Environment variables in play

| var | effect |
|---|---|
| `VLLM_USE_BREAKABLE_CUDAGRAPH` | piecewise graphs without torch.compile — **costs 17.65 GiB KV/GPU** |
| `VLLM_ROCM_USE_AITER` / `_MLA` / `_MOE` | route attention/MoE through AMD's AITER kernels |
| `VLLM_ROCM_AITER_MLA_ASM_PADDING` | assembly-kernel padding for MLA |
| `VLLM_DCP_Q_REPLICATE` | replicate Q across DCP ranks instead of gathering (#54494) |
| `VLLM_USE_DIRECT_DCP_A2A` / `_Q_GATHER` / `_KV_GATHER` | direct DCP ops; need a compiled C++ op that not every image ships |
| `AITER_QUICK_REDUCE_QUANTIZATION` | quantised all-reduce; `NONE` for us |
| `HSA_NO_SCRATCH_RECLAIM` | stops ROCm reclaiming scratch between kernels |
| `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS` | include graph memory in the profiler estimate |
| `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS` | raise it for very long prefills |

---

## 12. Open questions

1. ~~**Why is Kimi-K3 not torch-compiled**~~ — **being answered now.** #52190
   ("Enable torch.compile so post-grad fusion passes work") is applied in
   `kimi-k3-vllm:rec-d9105-tc` and under GSM8K gate as T282. If it holds, it
   restores `FULL_AND_PIECEWISE` at breakable=0 with the full 49.79 GiB pool — the
   exact configuration that produced 11,095 — *and* switches on fusion passes that
   have been inert all along.
2. **The ~20% gap to SemiAnalysis** at C48 (our gmu-matched 8,426 vs their 10,152)
   is still unexplained. Base nightly accounts for +2.0%, our patches +1.2%,
   gmu 0.88→0.90 +16.9%.
3. **Both in-tree profiling paths are dead** — no `VLLM_TORCH_PROFILER_DIR` on the
   old base (T202), and `rocprofv3` deadlocks the engine (T203). Until one works we
   are optimising without a profile.

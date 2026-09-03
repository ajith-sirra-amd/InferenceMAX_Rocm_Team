# LMCACHE — status, wiring, and the path to 12,500

**Status: LMCache has NEVER served successfully here. Root cause UNKNOWN.**

Corrected 2026-09-03. This file previously recorded T239 as a pass; it was not.
Both attempts failed with **zero generated tokens**:

| trial | `--max-gpu-workers` | generated tokens | result |
|---|--:|--:|---|
| T239 | **1** (= SA reference) | **0** | FAIL 96.5% (5/144) |
| T250 | **8** | **0** | FAIL 100% (0/144 and 0/893) |

So `--max-gpu-workers` is **not** the differentiator, and the hypothesis built on
"T239 passed with 1" is withdrawn. The 1 -> 1 revert is still correct for
matching the SA reference, but it is **not** a fix and must not be presented as
one.

The real question is why LMCache generates zero tokens in every configuration
while `vllm-simple` on the identical image/node/concurrency serves 893/893
(T245). That has not been answered, and the next step is reading the LMCache
server log itself, not another benchmark run.

<!-- superseded:
**Status: ROOT-CAUSED and FIXED (fix not yet exercised).** T239 passed its benchmark then stranded 58 GB on one
GPU at teardown. `Memory critical error … Reason: Memory in use` — LMCache still
held DMA registrations on vLLM's KV buffers when vLLM freed them. Required a
**node reboot**; a GPU reset did not clear it. Fix not yet applied — see below.
This file is the single source for LMCache work; updated after every trial.

---

## Why LMCache is the 12,500 lever

Current: **10,653–10,799 tok/s/GPU** at C72 on `pronly`. Target **12,500** =
**+15.9%**. Config space is closed (conc, mns, chunk, gmu, offload all swept),
and pruning removes upstreaming surface without adding throughput. So the gain
has to come from somewhere new.

**The metric points at the cache.** The headline appears to be **prompt-token
throughput ÷ 8**:

| evidence | |
|---|---|
| T236 headline × 8 | 10,799 × 8 = **86,392** |
| live `tput_in_srv` | **85,452 – 90,092** |
| output tok/s × 8 | 515 × 8 = 4,120 — nowhere near |

**Stated as a strong inference, NOT verified in code.** Verify before building
on it. If it holds, a cached prompt token still counts as processed but costs
almost nothing to serve — so **prefix-cache hit rate is the mechanism, not
merely a lever**.

| stack | theoretical | captured | gap |
|---|--:|--:|--:|
| pronly C72 | ~95–97% | **88.0%** | **7–9 pts** |
| aigmkt C52 | 96.5% | 93.6% | 2.9 pts |
| bare nightly C52 | 96.4% | 90.0% | 6.4 pts |

Workload is prefill-dominated: `in:out ≈ 195:1`, ISL p50 ≈ 89k.

---

## Wiring (reference only — from SA, read-only)

Source: `SemiAnalysisAI/InferenceX@perf/k3-mi355x-lmcache-rc3-c1-c8-c14-c40`,
`benchmarks/single_node/agentic/kimik3_fp4_mi355x_mtp.sh`, and run
33618719560 / job 100211512290. **Take the wiring, not their config.**

### Install (stock image untouched)

```
lmcache==0.5.5rc3+rocm7.2
  --find-links https://github.com/LMCache/LMCache/releases/expanded_assets/v0.5.5rc3-rocm
sortedcontainers==2.4.0
opentelemetry-exporter-prometheus==0.61b0
cupy-rocm-7-0==14.1.1
```
`--no-deps`. Native libs: `libgoogle-glog0v5 libjsoncpp25 libibverbs1
librdmacm1 libnuma1`. Import gate: `import cupy;
import opentelemetry.exporter.prometheus;
from lmcache.v1.multiprocess.http_server import run_http_server`.

### Server

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
Readiness: `http://127.0.0.1:8090/healthcheck`, timeout 600.

### Connector

```json
{"kv_connector":"LMCacheMPConnector",
 "kv_connector_module_path":"lmcache.integration.vllm.lmcache_mp_connector",
 "kv_role":"kv_both",
 "kv_connector_extra_config":{"lmcache.mp.port":6555,"lmcache.mp.mq_timeout":6000.0}}
```

`mq_timeout 6000.0` is deliberate — 100k–330k-token prefixes make single
retrieves large. Our ISL p50 ~89k needs the same headroom.

---

## THE trap: `--chunk-size`

The connector requires the chunk to be a multiple of **every** KV group's
`tokens_per_block`. The hybrid KDA/MLA layout registers **1536** (attention) and
**3072** (KDA state), so 12288 divides both. `--separate-object-groups` is
required because the multi-group layout needs one object group per
sliding-window size.

**The upstream Kimi-K3 recipe says 768. That is the CUDA path and is wrong
here.**

**Open risk:** those group sizes are quoted at **DCP=1**. At **DCP=8** the
per-group geometry changes — exactly what #53598 / #53917 / #54457 handle — so
**12288 may not be right for us.** Read `tokens_per_block` per group from the
engine log before trusting it.

---

## Their reference is a recipe, not a result

| | SA reference | our target |
|---|---|---|
| image | bare `7c5dc571`, **no patches** | `pronly` |
| **DCP** | **1 — off** | **8** |
| conc | 14 | 72 (may move — see below) |
| mns | 28 | 96 |
| outcome | **ProfileAborted**, 0/149 successful | — |

It failed at low concurrency with DCP off — an easier configuration than ours.
So the server and connector **do** come up on ROCm (healthcheck passed), which
weakens the earlier "CUDA-IPC may not port" concern. It does **not** show that
LMCache serves, and it says nothing about LMCache × DCP=8.

---

## LMCache has NO GPU-VRAM tier — checked, not assumed

Enumerated `lmcache server --help` (0.5.5rc3+rocm7.2):

| flag | what it actually is |
|---|---|
| `--l1-size-gb` | under **"L1 Memory Manager"** — **host** memory |
| `--l1-devdax-path` | `/dev/dax` or mmap-able file as the L1 arena |
| `--gds-l1-path` | GPU-Direct **Storage** (disk), not VRAM |
| `--max-gpu-workers` | **"Worker threads for the GPU affinity pool"** — threads, not memory |

**There is no VRAM tier option.** So the hierarchy already exists, just named
differently:

| tier | what it is | sized by |
|---|---|---|
| GPU | **vLLM's own KV cache** — 29,656,464 tokens | `gpu-memory-utilization` |
| L1 | LMCache host DRAM | `--l1-size-gb` |
| L2 | optional remote/disk adapter | `--l2-adapter` |

**Consequences:** LMCache cannot be configured to put L1 in VRAM; the GPU tier
is grown with **gmu**, not with LMCache. And `--max-gpu-workers` 1 → 8 was
*not* a fix for the stranded VRAM — it changed a thread-pool size. That
hypothesis is withdrawn.

What LMCache actually offers over `SimpleCPUOffloadConnector` is a **larger and
smarter host tier** (1,820 GB vs 226 GB/rank, LRU eviction, prefetch policies),
so any gain must show up as a higher **external** hit rate — 16.0–16.4% is the
number to beat at C72.

## Concurrency is NOT fixed at 72 for this work

C72 was the peak *given SimpleCPUOffload's KV economics* — 226 GB/rank. LMCache
L1 is `$TOTAL_CPU_DRAM_GB` (the reference used 1,799 GB). Different cache
economics move the concurrency optimum, and a higher hit rate frees GPU time
per request, which can support more in flight.

**So conc and mns are in scope for LMCache runs**, and they move together
(T230's rule: mns must exceed the replay's lane count, which runs above CONC).
This is the one place the strict one-variable discipline is the wrong tool —
testing the old optimum in a new regime would leave gains on the table.

Nobody has explored **LMCache × DCP=8 × high conc**. SA is sweeping the other
end (their branch: `c1-c8-c14-c40`, DCP off).

---

## Patches we may need

Permission granted to apply PR patches as required (our repo / local images
only).

### Is #53917 dead under LMCache? Only a third of it.

Hunk split of `pr_only/01_53917.patch`:

| file | hunks | under LMCache |
|---|--:|---|
| `simple_kv_offload/manager.py` | 8 | **dead** |
| `simple_cpu_offload_connector.py` | 1 | **dead** |
| `single_type_kv_cache_manager.py` | 6 | live — per-group KV geometry |
| `kv_cache_coordinator.py` | 4 | live |
| `config/vllm.py` | 3 | live |
| `sched/scheduler.py` | 1 | live |
| `kv_cache_manager.py` | 1 | live |
| `kv_connector/v1/base.py` | 1 | **live — connector base class** |
| `kv_connector/factory.py` | 1 | **live — LMCacheMPConnector loads through this** |
| `offloading_connector.py` | 1 | generic, not SimpleCPU |

**9 of 27 hunks are SimpleCPU-specific and genuinely dead. 17 are not.**
`LMCacheMPConnector` is instantiated through that factory and inherits that base
class, and the geometry hunks are what set `tokens_per_block` per group at
DCP=8 — the number `--chunk-size` must divide.

**Plan: keep #53917 for the first LMCache run**, so a failure is attributable to
LMCache rather than to a simultaneous prune. Then drop it as a clean
one-variable arm. It is the one PR never tested for removal, so the arm is worth
having either way.

### Other patch candidates

| PR | why it may be needed |
|---|---|
| **#54457** | *"Do not adjust `dcp_kv_cache_interleave_size` for CP"* — #53917's declared dependency, open. Sits directly on the DCP interleave geometry that sets `tokens_per_block`, i.e. the same thing `--chunk-size` must divide. |
| #53598 | already merged in base; per-group DCP cache geometry. |

**The chunk-size question and the patch question are the same question:** what
`tokens_per_block` each group registers at DCP=8.

---

## Run order (user-set)

1. **Fixed-len** (`TEST=1`) — does the engine/server come up with LMCache at all
2. **GSM8K limit 200** — new KV connector is a major change; accuracy gates
3. **Agentic C72** — then sweep conc/mns if the cache economics have moved

Do not skip 1 → 2. Do not burn a 2 h agentic run to discover the server did not
start.

---

-->

## Trial log

| trial | config | result |
|---|---|---|
| **T239** | fixed-len (`TEST=1`, func 214k), `pronly-nq-no50618`, DCP 8, conc 72, chunk 12288, l1=1949 GB, **gpu-workers 1** | **FAIL — 5/144, 96.5% failure rate, 0 generated tokens.** Previously mis-recorded here as "PASS 5/5"; corrected 2026-09-03 from the run log. The 920.34 tok/s is prefill-only. |
| **T240** | GSM8K, same config | **BLOCKED — never started.** Died in the pre-run GPU-reclaim wait: `waiting for prior-job GPU memory reclaim: vram%max=18` for 90×10 s, then gave up. The eval never ran; there is no accuracy result. |

### T240 — leaked VRAM, and the likely cause is LMCache

The run failed *before* the server started. The runner's pre-flight loop waits
for max VRAM across GPUs to fall to ≤10%; it sat at **18%** for the full 15 min
window.

State after the failure, with **nothing of ours running**:

| | |
|---|---|
| GPU 0,1,2,4,5,6,7 | **0%** VRAM |
| **GPU 3** | **18%** VRAM, **0% util** |
| containers | none running |
| lmcache processes | none |
| other tenants (SA/GLM) | no active runs |

**~52 GB held on exactly one GPU, by no visible process.**

### ROOT CAUSE (from the T239 artifacts, post-reboot)

The last line of `server.log`:

```
Memory critical error by agent node-0 (Agent handle: 0x5eac2930)
on address 0x7fbb51601000. Reason: Memory in use.
```

Timeline:

| time | event |
|---|---|
| 17:12:5x | fixed-len benchmark **completed** — 5/5, result written |
| 17:12:58 | `EngineDeadError` — in-flight completions start returning 500 |
| 17:12:59 | vLLM `MPClient` shutdown; `15.0s ROCm cleanup grace`; SIGTERM to EngineCore |
| 17:12:59 | LMCache begins its own shutdown |
| — | **`Memory critical error … Reason: Memory in use`** — ROCr refuses to release |
| **17:13:49** | **LMCache still logging** `L1 memory usage: 38.16/1820.00 GiB` — **50 s after shutdown began, still alive** |

**vLLM freed GPU memory that LMCache still had registered for DMA.** The
`LMCacheMPConnector` registers vLLM's KV buffers with the MP server so it can
DMA into them; those registrations outlived the vLLM teardown, so the ROCr free
failed with *Memory in use* and ~58 GB was orphaned with no owning process.

Corroborating: the server log is full of
`pin_memory failed for chunk at ptr=… size=67108864; DMA performance may be
degraded` — host-pinning was already failing throughout the run, so the DMA
registration path was in a degraded state before teardown.

**Not recoverable in software.** `sudo rocm-smi --gpureset -d 3` reported
`Successfully reset GPU 3` and the 57.94 GB **did not move**. Only a node reboot
cleared it.

### The ordering bug is ours

`cleanup_agentic_services` already stops LMCache before the vLLM server — but
that trap runs at script EXIT, and **EngineCore had already died a second
earlier**. The teardown order that matters is inside vLLM's own shutdown, which
we do not control from the trap. Worse, LMCache needs **>50 s** to wind down and
we allow 60 s, so it is borderline even when the ordering is right.

### FIX APPLIED (2026-09-02 ~18:20Z)

New `lmcache_shutdown()` in the launcher:

1. **Called explicitly right after the workload**, while the vLLM server is
   still up — *not* left to the EXIT trap, which raced EngineCore in T239.
2. SIGTERM, then **poll until the process is actually gone, up to 300 s**
   (observed 50 s+ under a 5-request load; an agentic run will be slower). The
   old allowance was 60 s.
3. SIGKILL only as a last resort, and it **prints a loud warning** that VRAM may
   be stranded so the next run is not started blind.
4. Idempotent — safe to call again from the trap.

**Also fixed a bug I introduced in the original cleanup:** `local exit_code=$?`
sat *after* the LMCache stop, so it captured that command's status instead of
the script's real exit code. Every LMCache run would have reported a misleading
exit status. Now captured first.

**Still unexercised.** The fix is reasoned from the T239 log, not yet proven by
a run.

**Superseded hypothesis (kept — it was wrong):** the LMCache MP server leaks a GPU allocation on teardown.
We launch it with **`--max-gpu-workers 1`**, and **exactly one GPU** is holding
memory. That correlation is the strongest signal available. The server is a
separate process from vLLM; our `cleanup_agentic_services` reaps `LMCACHE_PID`,
but if its GPU-side registration is not released before the container dies, the
allocation is orphaned with no host process to attribute it to.

**Precedent:** T220 hit the same *class* of failure (orphaned KFD allocations,
no host processes) and drained on its own after ~1 h. So this may clear
passively — but if the cause is LMCache, **it will recur on every LMCache run**,
which makes back-to-back LMCache arms impossible without a fix.

**Do not re-dispatch until VRAM clears.** Re-dispatching into 18% just burns
another 15-minute wait and fails identically.

**FIX APPLIED (user, ~17:45Z): `--max-gpu-workers` = `$TP` = 8**, was 1.

One GPU worker in an 8-rank deployment is asymmetric, and T240 stranded memory
on **exactly one** GPU. Matching workers to TP addresses the asymmetry directly
and is also simply the right shape for TP8 — the SA reference used 1, but it is
the only value they tried. Overridable via `LMCACHE_GPU_WORKERS`. Gate line now
prints `gpu_workers=`.

**This is a hypothesis under test, not a confirmed fix.** If T240-retry still
strands VRAM, the next candidates are:
1. `--max-gpu-workers 0` — the L1 pool is host DRAM; a GPU worker may not be
   needed for our path at all.
2. Explicit server shutdown (HTTP or SIGTERM) *before* container teardown,
   rather than relying on process-tree reaping.
3. Longer drain in the launcher's own cleanup, so the allocation is released
   while the container still exists.

### Notes on the T239 wiring

- **Inlined the readiness poll.** Our `benchmark_lib.sh` has no `wait_for_ready`
  or `append_command` — those exist only in the newer SA lib. A straight port
  would have died at the first call. The inline poll fails fast if the server
  exits during startup and dumps the log tail.
- `LMCACHE_PID` wired into `cleanup_agentic_services` so a failed run does not
  leak the server process.
- Overridable: `LMCACHE_CHUNK_SIZE`, `LMCACHE_PORT`, `LMCACHE_HTTP_PORT`,
  `LMCACHE_VERSION`.
- **Image is the recommended stack** (`pronly-nq-no50618`, T236 = 10,799), not
  the minimal one — LMCache is being added to the best-measured base.

### T239 RESULT — all three checks cleared

| check | result |
|---|---|
| server up | `[lmcache] chunk=12288 l1=1949GB port=6555` → `server READY on :8090 after 26s` |
| ROCm path | `Auto-selected backend [rocm] for accelerator 'cuda'` on all 8 workers + EngineCore; `multi_layer_block_kv_transfer mode: ptr` |
| **chunk geometry at DCP=8** | **no rejection.** The connector registered every KV group without complaint, so 12288 (derived from DCP=1 sizes) holds at DCP=8 |
| fixed-len | **5/5 successful**, 920.34 tok/s total, median ITL 1254 ms, no engine fault |
| GPU KV capacity | **29,656,464** — identical to the SimpleCPUOffload arms |

**Two risks I had flagged are now closed:**

1. *"CUDA-IPC may not port to ROCm"* — it does; LMCache auto-selects the rocm
   backend. My original caveat was overstated, and the SA reference already
   hinted at it.
2. *"12288 may not divide every group at DCP=8"* — it does. No patch beyond
   #53917 was needed.

**Do not over-read the fixed-len numbers.** 5 requests, TTFT 40 s, ITL 1254 ms
are a cold cache with no reuse — this arm proves the engine *works*, nothing
about performance. The agentic replay is where LMCache either earns its keep or
does not.

**KV capacity unchanged (29,656,464)** means LMCache is not buying GPU-side
cache — its L1 is the 1,949 GB host pool. So any gain must come from a higher
*external* hit rate, not more GPU cache. That is the number to watch in T241:
SimpleCPUOffload gave ext_cache_hit 16.0–16.4% at C72.

---

## T250 (2026-09-03) — 100% failure both passes. Root cause candidate found.

Run [33736043897](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/33736043897)
· `rec-no53940` · C72 · numa_balancing=0 at launch.

| stage | result |
|---|---|
| Function (isl 214k, 144 prompts) | **0/144 — 100% fail** |
| Fixed Perf (isl 8192/1024, 893 prompts) | **0/893 — 100% fail** |

Zero tokens generated in either. **The fixed-perf config passed 893/893 with
914,432 tokens in T245 on `vllm-simple`** — same image, node and concurrency,
so the KV backend is the only variable.

### The failure is instant, not slow

```
warmup:    1/144 [11:29<27:24:17, 689.91s/it]   <- first request 690 s
           144/144 [11:30]  Warmup completed.
benchmark: 144/144 [00:00<00:00, 340.95it/s]    <- ALL failed in 0 s
UserWarning: All requests failed. This is likely due to a misconfiguration
```

Warmup limped through at 690 s to first token; every *benchmark* request then
failed immediately. Instant rejection, not a stall — so this is an error path,
not a throughput problem. `kv_load_failure_policy='fail'` in the vLLM config
turns any LMCache KV-load miss into a hard request failure, which fits.

### Diff vs the SA reference recipe (read-only)

`SemiAnalysisAI/InferenceX` @ `perf/k3-mi355x-lmcache-rc3-c1-c8-c14-c40`,
`benchmarks/single_node/agentic/kimik3_fp4_mi355x_mtp.sh`:

| arg | SA reference | T250 |
|---|---|---|
| **`--max-gpu-workers`** | **1** | **8** |
| `--l1-size-gb` | `$TOTAL_CPU_DRAM_GB` | same |
| `--l1-init-size-gb` | 10 | same |
| `--chunk-size` | 12288 | same |
| `--max-cpu-workers` | 8 | same |
| `--separate-object-groups`, `--eviction-policy LRU`, `--supported-transfer-mode lmcache_driven`, `--shm-name ""` | ✓ | same |
| connector JSON (`lmcache.mp.port`, `mq_timeout 6000.0`) | ✓ | identical |

**`--max-gpu-workers` is the only difference.** Our default is
`LMCACHE_GPU_WORKERS="${LMCACHE_GPU_WORKERS:-$TP}"` = 8; the reference pins 1.
That value was changed 1 -> 8 on 2026-09-02 after I withdrew a wrong hypothesis
about it controlling VRAM (`--help` calls it "Worker threads for the GPU
affinity pool").

**Next LMCache attempt must set `--max-gpu-workers 1`** to match the reference,
and change nothing else.

### What worked

- LMCache server: `READY on :8090 after 18s`
- **Teardown fix validated**: `server exited cleanly after 44s`, no SIGKILL, no
  stranded-VRAM warning. T240's mechanism did not recur.
- GPU KV pool unchanged at 28,653,478 — LMCache does not shrink the GPU tier.

### Node rebooted at 09:22

Third unclean reboot (18:05, 03:08, 09:22), none with a shutdown record, around
T250's completion. **Causation not established.** After it: `amdgpu` was not
loaded and `numa_balancing` had reverted to 1; both recovered manually.
`/etc/sysctl.d/99-numa.conf` is **still absent**.

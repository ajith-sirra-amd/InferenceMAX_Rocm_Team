# LMCACHE — status, wiring, and the path to 12,500

**Status: T239 PASSED.** LMCache runs on ROCm at DCP=8. T240 (GSM8K) dispatched.
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

## Trial log

| trial | config | result |
|---|---|---|
| **T239** | fixed-len (`TEST=1`), `pronly-nq-no50618`, DCP 8, conc 72, chunk 12288, l1=1949 GB | **PASS** — 5/5 requests, 920.34 tok/s total, no engine fault |

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

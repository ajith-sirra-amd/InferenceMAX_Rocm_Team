# Kimi-K3 DCP Experiments

Newest first. Kimi-K3 (2.8T MoE, 24 of 93 layers are full-attn MLA, 96 heads, 1M ctx,
MXFP4) on 8× MI355X, TP8, agentic replay. **Target: 12,500 tok/s/GPU.**

---

## 1. Summary — what was tried, and what it showed

| # | Tried | Trial | Result | Finding |
|---|---|---|---|---|
| 1 | DCP at all (block-table fix) | T4 | ✅ **0x1016 fixed** | Tables were sized `max_model_len/dcp` but indexed with the undivided length. Boundary tracked the ratio exactly. Made DCP usable. |
| 2 | DRAM KV offload | T18 | ✅ **781 → 1,991** (2.55×) | Largest single DCP win. But no effect on the decode straggler. |
| 3 | GSM8K correctness gate | T23 | ✅ **0.9659 / 0.9644** vs 0.9651 | DCP is numerically correct. The case against it is performance only. |
| 4 | World size 8 → 4 | T24/T25 | ❌ +2.2% | **Decisive.** Halving collective traffic moved TPOT 4% → cost is *not* world-size-scaled. Rules out every scaling fix. DCP=2 is illegal (24 heads). |
| 5 | Concurrency sweep | T19/T26 | ❌ optimum c20 | c8 969 · c20 2,034 · c64 1,041. Axis closed. |
| 6 | Combine algo a2a vs ag_rs | T28 | ❌ 2,034 vs 1,978 | Both ROCm options within 3%. |
| 7 | Shard granularity 1 → 16 | T29 | ❌ 1,977 | Locality before the combine wasn't the cost. |
| 8 | Attention backend | T21/T45 | ❌ within noise | Not backend-bound. |
| 9 | CUDA graphs | T7/T45 | ❌ no TPOT effect | Not launch-count-bound at the graph level. |
| 10 | Async scheduling on/off | T41/T42 | ❌ no effect | 0 `ref_cnt` asserts at c20; the workaround was unnecessary. |
| 11 | NUMA pinning, node-level | T45 | ❌ no effect | — |
| 12 | NUMA pinning, per-rank slices | T46 | ❌ **worse** | 188→218 ms rising vs unpinned converging to 208. Not an OS-scheduling problem. |
| 13 | Ported direct P2P collective to ROCm | T31b | ⚠️ works, **−0.9%** | Hand-ported the a2a combine HIP kernel. Functional, not faster. |
| 14 | MTP under DCP | T20 | ❌ killed | Draft KV is replicated → costs W× per rank. |
| 15 | Profiling DCP vs non-DCP | T35e/T37 | 🔍 **bottleneck located** | 92.65% collectives vs 56%. DCP shards no work; the TP all-reduce inflates 8.8×. |
| 16 | Non-DCP best config to completion | T47 | 📊 **~2,800 sustained** | Also revealed 94.1% theoretical prefix hit vs 30.3% achieved — quantifies what KV capacity is worth. |
| 17 | **All-reduce implementation** | **T48** | **staged** | The one lever identified and never dispatched. |

### Performance, all measured runs

| Config | tok/s/GPU | TPOT | TTFT | Trial |
|---|---:|---:|---:|---|
| SA reference | **5,388** | 0.038 | — | — |
| non-DCP + DRAM offload (older image) | **3,341** | 0.043 | — | T22 |
| non-DCP, current stack, sustained | ~2,800 | — | — | T47 |
| **best DCP** — DCP=4, c20, DRAM | **2,034** | **0.167** | 3.80 s | **T25** |
| DCP=8 + ported direct a2a | 2,015 | 0.172 | 4.69 s | T31b |
| DCP=8, c20, DRAM | 1,991 | 0.174 | 4.64 s | T18 |
| DCP=4, ag_rs combine | 1,978 | 0.184 | — | T28 |
| DCP=4, interleave 16 | 1,977 | 0.177 | 4.81 s | T29 |
| DCP=8 + TRITON_MLA, spec off | 1,948 | 0.186 | 5.4 s | T21 |
| non-DCP, c1 (latency point) | 1,225 | **0.0042** | — | T22 |
| DCP=8, c64 | 1,041 | 0.683 | 501 s | T19 |
| non-DCP c32, no cudagraph patch | 1,015 | 0.170 | — | T11 |
| DCP=4, c8 | 969 | 0.164 | — | T26 |
| DCP=8 + PIECEWISE cudagraph | 781 | 0.687 | — | T7 |
| DCP=8, no offload | 742 | 0.663 | — | T5 |
| DCP=8 + pinning | — | 0.218 | — | T46 |
| DCP + everything | — | 0.242 | — | T45 |

**DCP's ceiling is ~2,000 tok/s/GPU — 39% below best non-DCP.**

---

## 2. The bottleneck

Trace, rank 0, 15 s decode window ([T35e DCP=8](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/32333672290) vs [T37 non-DCP](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/32340149740)):

| | DCP=8 | non-DCP |
|---|---:|---:|
| GPU kernel time | 14.801 s | 5.033 s |
| `cross_device_reduce_2stage` (TP all-reduce) | **78.37%** | 55.48% |
| MoE GEMM | 1.91% | 13.33% |
| **collectives / compute** | **92.65% / 7%** | 56% / 44% |

Per decode step (~118 all-reduces/step in **both**):

| | all-reduce | compute | TPOT |
|---|---:|---:|---:|
| non-DCP | 14.6 ms | 11.5 ms | 43 ms |
| DCP=8 | **129 ms** | 11.6 ms | 167 ms |

**Findings:**
1. **DCP shards no decode work** — compute/step identical (11.5 vs 11.6 ms).
2. **The penalty isn't DCP's own collective** — the a2a combine is ~8 µs.
3. **It's the TP all-reduce inflating 8.8×**, at the same call count. A collective
   **DCP does not own**.
4. **Mechanism is host-side** — one rank starved by ~124k launch gaps of ~62 µs;
   seven others block inside the all-reduce.
5. **The straggler moves** rank 7 → 0 → 1 → 5 across runs — not a bad GPU or link.
6. **GPU load is balanced** to within 5%.
7. **Cost is not world-size-scaled** (T25) — which rules out every scaling fix.

*Caveat: kernel duration includes time blocked on peers, so this localises where time
is attributed, not by itself the root cause.*

---

## 3. Patches

| # | Name | What it fixes |
|---|---|---|
| [1] | aiter pybind11 | Standalone pybind11 (internals v12) outranks torch's bundled (v11) via `-I` vs `-isystem`. Separate type registry per internals id → JIT module can't see `aiter_tensor_t` → `TypeError` at warmup. Unblocks `ROCM_AITER_FA` prefill. |
| [2] | TritonMLA cudagraph | `_cudagraph_support = UNIFORM_SINGLE_TOKEN_DECODE` caps `min_cg_support`, downgrading FULL→PIECEWISE under spec-decode, leaving the DSpark drafter fully eager. TRITON_MLA can't be swapped — it's the only ROCm MLA backend with `supports_non_causal_multi_token_decode=True`. **Measured 14.05 → 77.65 tok/s, ITL 71.16 → 12.88 ms (5.52×).** |
| [3] | KV block-pool clamp | `allocate_external_computed_blocks()` passes a **negative** count to `get_new_blocks`, which is silently destructive: `num_free_blocks -= n` *increases* it, `range(n)` iterates zero times, free list untouched. Later pop walks past the tail → `assert block.ref_cnt == 0` mid-run. Load-dependent: c10 died 3612 s, c12 487 s, c16 354 s. |
| [4] | our DCP-LSE | Plumbs LSE + round-robin CP through AITER MLA decode. aiter already had it (`return_lse`/`cp_world_size`/`cp_rank`); vLLM never wired it. aiter's LSE is natural-log, sm_scale folded, `[B,H]` fp32 — already DCP's layout. **Superseded by [5], now OFF.** |
| [5] | vLLM PR #51705 | Upstream DCP for Kimi-K3 DSpark. Open, in no nightly; fetched at runtime, **pinned by sha256**; diff filtered to `vllm/` only. **Predicted before running, and confirmed: does NOT fix 0x1016** — it exempts groups only when `spec.non_causal_multi_token_decode`, and with `speculative_config=None` that set is empty. |
| [6] | DCP block-table sizing | **THE 0x1016 FIX.** Tables sized `cdiv(max_model_len, block_size × dcp_size)` but indexed with the *undivided* `max_model_len` → at 1M/DCP8 only 131,072 tokens addressable → OOB in chunked prefill. |
| [7] | direct DCP a2a (ROCm port) | vLLM compiles the kernel only under `VLLM_GPU_LANG == CUDA`. Ported the a2a combine to HIP (`st/ld.global.{release,acquire}.sys.u32` → `__hip_atomic_*` at `SYSTEM` scope). `q_gather`/`kv_gather` use `multimem.st.*` PTX (NVLink multicast) — no AMD equivalent, not ported. **Works, −0.9%.** Env-gated off. |
| [8] | DCP gathered-head sizing | Supplies PR #51705's failing hunk 7 (`self._decode_num_heads = num_heads × dcp_world_size`). Guard must match the **assignment** — the PR adds 14 *uses* of the name while the defining hunk fails. |

Patch [6] boundary, measured — tracks `max_model_len/dcp` exactly:

| | budget | last chunk fit | faulted at |
|---|---:|---:|---:|
| DCP8 bf16 | 131,072 | 119,040 | 134,400 |
| DCP4 bf16 | 262,144 | 254,976 | 262,656 |
| DCP8 fp8 | 131,072 | 119,808 | 135,168 |

Halving DCP doubled both → block **count**, not bytes. All pure prefill, which is why
swapping decode backends never helped.

### Patches per trial

| Trial | [1] | [2] | [3] | [4] | [5] | [6] | [7] | [8] |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| T1–T2 | – | – | – | ✅ | – | – | – | – |
| T3 | – | – | – | – | ✅ | – | – | – |
| T4–T7 | – | – | – | – | ✅ | ✅ | – | – |
| T8 | ❌ | ❌ | ❌ | – | ✅ | ✅ | – | – |
| T9 | ✅ | ✅ | ✅ | – | ✅ | ✅ | – | – |
| T10 | ✅ | ✅ | ✅ | – | – | – | – | – |
| T11–T13 | ✅ | ❌ | ✅ | – | – | – | – | – |
| T17–T29 | ✅ | – | ✅ | – | – | ✅ | – | – |
| T31b | ✅ | – | ✅ | – | – | ✅ | ✅ | – |
| T35e–T48 | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | off | ✅ |

Images: T1 `ac7509e2b` · T2–T13,T16 `311b3513` · T14–T15 `yukiozzz/kimi-k3-pr51705` ·
T17–T31b `ac7509e2b` · T22 unified `3fa1b88a` · T35e→ `nightly-5a4c8d99`.

---

## 4. Key trials

### T48 — `--disable-custom-all-reduce` (staged)
DCP=8 · conc 20 · DRAM offload · fp8 KV · spec MTP · pinning.

`cross_device_reduce_2stage` **is** the AITER custom all-reduce — 78% of DCP kernel time.
Engine runs `disable_custom_all_reduce=False`, dispatch order
`['QUICK_REDUCE','AITER_CUSTOM','PYNCCL']`, so AITER_CUSTOM is what every trial has
measured. This flag forces PyNCCL/RCCL — different host-side launch and sync behaviour,
which is what the identified mechanism is made of.

*Flagged in the log as "worth one run" three times and never dispatched. A real gap,
caught by review rather than by me.*

### T47 — best config, run to completion
[Run 32372517517](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/32372517517) ·
DCP **off** · TRITON_MLA · pinning · DRAM offload · fp8 KV · conc 20 · spec MTP

The investigation had no completed aggregate for its own best config — T44 showed the
lowest ITL ever measured (32 ms cold / 130 ms steady) but was cancelled, so everything
downstream was extrapolation from a cancelled run.

```
tput_in_srv 20,994–22,669/s  ->  2,624–2,834 tok/s/GPU
0 preemptions · 0 errors · full 20/20 concurrency
```

**Finding — what the KV capacity is worth:**
```
theoretical_prefix_cache_hit  94.1%   <- unbounded cache
prefix_cache_hit              30.3%   <- actual, 4.4M-token pool
ext_cache_hit                 90.8%   <- DRAM catching the misses
unique_in_srv                 36.2M tokens
```
The workload has 94.1% reusable prefix. Non-DCP holds 30% in HBM; DRAM catches the rest,
so little is recomputed — but at PCIe latency. **DCP=8's 32.7M pool is the only measured
config that fits this working set in HBM.** That benefit is real and **prefill-side**
(T25 TTFT 3.80 s vs T18 4.64 s).

**Correction:** the ~2,000 tok/s/GPU DCP ceiling is *not* evidence the KV capacity is
useless — it's evidence the decode collective eats a benefit that genuinely exists.

*Prediction stated before the result: prefix hit would decay ~91% → ~30% with DRAM
absorbing overflow. Observed 48.0% → 30.3%, ext_cache 21% → 90.8%.*

### T46 — NUMA pinning isolated: no effect
[Run 32368684074](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/32368684074) ·
DCP=8 · per-rank dedicated CPU slices, the only delta vs T36.
```
T46 pinned:    188 -> 175 -> 200 -> 216 -> 218 ms  (RISING)
T36 unpinned:  221 -> 219 -> 217 -> 213 -> 208 ms  (converging DOWN)
```
Prefix cache was back to 52.3% at 218 ms, so not a cold-cache artifact. Cancelled.
*I earlier called an early 188 ms sample "worth ~15%" — wrong, light-load artifact.*

### T25 — best DCP
[Run 32143877066](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/32143877066) ·
DCP=4 · conc 20 · a2a · interleave 1 · DRAM offload
```
2,033.7 tok/s/GPU · TPOT 0.1674 · TTFT 3.80 s · 449/490 · KV 17.0M (16.25x)
```
| | DCP=8 (T18) | DCP=4 (T25) |
|---|---:|---:|
| tok/s/GPU | 1,990.8 | **2,033.7** (+2.2%) |
| TPOT | 0.1742 | 0.1674 (−4%) |
| TTFT | 4.64 s | 3.80 s (−18%) |

**Decisive:** halving world size halves collective traffic and moves TPOT 4% → the cost
is **not world-size-scaled**, it's a fixed per-step overhead of doing the merge at all.

### T22 — best overall
[Run 32123047671](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/32123047671) ·
DCP off · DRAM offload → **c20: 3,340.5 tok/s/GPU, TPOT 0.0429** (c1: 1,225.1 / 0.0042)

### T18 — DCP breakthrough
[Run 32096487055](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/32096487055) ·
DCP=8 + DRAM offload → **1,990.8 tok/s/GPU. 781 → 1,991 = 2.55×.** Largest single DCP win.

### T23 — correctness gate
[Run 32138970163](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/32138970163) ·
GSM8K **0.9659 strict / 0.9644 flexible** vs 0.9651 baseline.

### T4 — 0x1016 fixed
[Run 32043813560](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/32043813560) ·
PR #51705 + patch [6] → errors=0, no fault. The run that made DCP usable at all.

---

## 5. Full ledger

| # | run | config | result |
|---|---|---|---|
| 1 | 32025696861 | [4], DCP8, bf16 | FAIL 0x1016 @134,400 |
| 2 | 32039650984 | [4], DCP4, bf16 | FAIL 0x1016 @262,656 |
| 3 | 32042030173 | #51705 only, DCP8 | FAIL 0x1016 @135,168 |
| 4 | 32043813560 | #51705 + **[6]** | **0x1016 FIXED**, errors=0 |
| 5 | 32049134216 | + `AIPERF_FAST=1` | 742.4, TPOT 663 ms; 113/183 dropped, validity suspect |
| 6 | 32055440757 | GSM8K gate | timeout not fault — 3.5–6.8 tok/s, ≈15 h needed |
| 7 | 32060082326 | + PIECEWISE cudagraph | 780.6, TPOT 0.687 |
| 8 | 32066978737 | DCP off, [1]–[3] off | FAIL @15 min — **my error**, `AITER_DISABLE_FMHA_OPUS` only set in the DCP branch |
| 9 | 32068474469 | DCP off, [1]–[3] on | FAIL @25 min — RCCL watchdog |
| 10 | 32070778181 | DCP off, [1][2][3] | FAIL @21 min — same signature, so [5] wasn't the cause |
| 11 | 32073039787 | DCP off, [2] off | 1,014.9, TPOT 0.170 |
| 12 | 32077536567 | + full profile | 999.4, TPOT 0.1698 |
| 13 | 32084553677 | chunk 8192→32768 | LOST — cancelled by workflow concurrency rule |
| 14 | 32089974051 | yukiozzz, c64, MTP | FAIL — `mla_gluon requires batch_size=1, got 64` |
| 15 | 32090860356 | yukiozzz, c64 | FAIL — same, got 3 |
| 16 | 32093774227 | c64, chunk 32768 | cancelled @20 min |
| 17 | 32094907936 | DCP8 c20 DRAM | FAIL — **my error**, leftover `MAX_NUM_BATCHED_TOKENS=32768` |
| 18 | 32096487055 | DCP8 c20 **DRAM** | **1,990.8**, TPOT 0.1742 |
| 19 | 32101946357 | conc 64 | 1,041.1, TTFT 501 s — worse |
| 20 | 32110276088 | TRITON_MLA + MTP | cancelled @55 min, degrading |
| 21 | 32114847961 | TRITON_MLA, spec off | 1,948.4 — noise vs T18 |
| 22 | 32123047671 | DCP off, c1+c20 | **3,340.5 — BEST OVERALL** |
| 23 | 32138970163 | GSM8K gate | **PASSED 0.9659/0.9644** |
| 24 | 32143146154 | DCP=2 | FAIL @init — `is_valid_num_heads(24)` illegal at TP8 |
| 25 | 32143877066 | DCP=4 c20 | **2,033.7 — BEST DCP** |
| 26 | 32154159649 | DCP=4 **c8** | 969.4 — concurrency axis closed |
| 27 | 32162233477 | direct symm-mem | FAIL @9 min — op absent, CUDA-only |
| 28 | 32163743641 | **ag_rs** | 1,978.4 — a2a wins |
| 29 | 32172937044 | interleave 1→16 | 1,977.2 — worse |
| 31a | 32269094879 | local image tag | FAIL @6 min, no GPU — harness `--pull always` can't resolve a local tag |
| 31b | 32270805303 | **ported direct a2a** | 2,014.7 — works, **−0.9%** |
| 35e | 32333672290 | DCP=8 profiling | 92.65% collectives |
| 37 | 32340149740 | non-DCP profiling | 56.05% collectives |
| 45 | — | DCP + everything | TPOT 0.242 |
| 46 | 32368684074 | DCP=8 + pinning | 0.218 rising — no help |
| 47 | 32372517517 | non-DCP, full 3600 s | ~2,800 sustained |

---

## 6. Levers vs the DCP decode penalty

| Lever | Trial | Result |
|---|---|---|
| DRAM KV offload | T18/T39 | ✅ 2.55× as a feature · ❌ no effect on straggler |
| Concurrency | T19/T26 | ❌ optimum at c20 (c8 969 · c20 2,034 · c64 1,041) |
| World size 8→4 | T24/T25 | ❌ +2%; DCP=2 illegal |
| a2a vs ag_rs | T28 | ❌ 2,034 vs 1,978 |
| Interleave 1→16 | T29 | ❌ 1,977 |
| Attention backend | T21/T45 | ❌ noise |
| CUDA graphs | T7/T45 | ❌ no TPOT effect |
| Async scheduling | T41/T42 | ❌ no effect |
| NUMA pinning (node) | T45 | ❌ no effect |
| NUMA pinning (per-rank) | T46 | ❌ **worse** |
| Ported direct P2P collective | T31b | ⚠️ works, −0.9% |
| MTP under DCP | T20 | ❌ draft KV replicated, W× per rank |
| **All-reduce implementation** | **T48** | **the one untested lever** |

---

## 7. Wrong turns (kept)

- **"The DCP decode cost is irreducible"** — wrong in an interesting way. The cost is real
  but mostly **not DCP's**; it's the TP all-reduce any TP=8 decode pays, which DCP worsens
  by adding sync points.
- **T31a** — dispatched a local-only image tag; the harness does `--pull always`.
- **`-p2` dry-run false positive** — claimed a patch applied to both nightlies; correct
  `-p1` showed only `5a4c8d99` works (23/24 hunks).
- **Patch [8] guard matched uses, not the assignment** — would have thrown `AttributeError`
  at first decode. Caught in local validation.
- **`VLLM_TORCH_PROFILER_DIR` obsolete** — removed from `envs.py`; use `--profiler-config`.
- **`PROFILER_ARGS` used before defined** — line 786 vs 792, so it never reached the server.
- **Shadowed `DCP_SIZE`** — an assignment in the auto-concurrency block ran first, so runs
  showed DCP=4 / 17× KV when 8 / 32× was intended.
- **Acceptance length misreported as 2.65–2.69** — that was `sort -u | head`. True mean
  **2.706** (n=58, p50 2.70, max 2.78) under `standard` rejection — better than the 2.51 the
  perf model assumes. The 0.6467 collapse came solely from `synthetic` rejection.
- **"The causal hack is unnecessary, just use TRITON_MLA"** — wrong here: Kimi-K3 fp8 KV
  decode requires a backend accepting an fp8 query. The causal rewrite is a *consequence*
  of the fp8 KV choice.
- **Patch [2] "target not found"** — the check container had no GPU, which changes vLLM's
  import behaviour. The real run log shows `[triton-mla-cudagraph] patched`.

---

## 8. Conclusion

**DCP works, is correct, and is structurally unprofitable on this workload.**

The decode penalty is a fixed per-step cost of doing the gather/merge at all — independent
of ranks, batch, algorithm, granularity, backend, and CPU placement. Twelve levers measured,
all negative. The cost lands on a collective DCP doesn't own, inflated 8.8× because DCP's
host-side work starves one rank while seven wait.

One lever remains: the **all-reduce implementation** (T48). If PyNCCL/RCCL doesn't move it,
the next step isn't another config trial — it's instrumenting vLLM's DCP host path to find
what serialises one rank per step.

Against 12,500 tok/s/GPU: this stack reaches ~2,800, the SA reference 5,388. **The gap is
not DCP-specific — it's structural to this configuration.**

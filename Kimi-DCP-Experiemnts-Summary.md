# Kimi-K3 on 8x MI355X — Performance

Kimi-K3 (2.8T MoE, 1M context, MXFP4) · vLLM ROCm · agentic replay · TP8.
**Target: 12,500 tok/s/GPU.**

---

# Where we are

## C1 interactivity sweep, 2026-08-28 (T138–T151) — CURRENT

Fixed-length screen: ISL 122k (effective ~63.8k after the BPE round-trip),
OSL 500, 100 requests, DCP off, k=8 @ AL 4.00, draft KV fp8.
Chosen to match the agentic replay's 122,657-token mean input, because DCP's
value is context-length dependent and an 8k screen would mis-rank it.

| run | image | DCP | mns / ladder | TPOT mean | p50 | p90 | ITL |
|---|---|--:|---|--:|--:|--:|--:|
| [T147](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/33171360827) | **nightly 6f7df92a8e** | 1 | 8 / 1..72 | **7.57** | 7.47 | **7.91** | 29.47 |
| [T148](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/33183801155) | nightly | 1 | 8 / 1..72 | 7.57 | 7.49 | 7.93 | — |
| [T150](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/33185946417) | nightly | 1 | **1 / 1..9** | 7.58 | 7.48 | 7.93 | — |
| [T151](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/33187965065) | aigmkt | 1 | 8 / 1..16 | 7.93 | 7.77 | 8.20 | — |
| [T145](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/33168461313) | aigmkt | 1 | 8 / 1..72 | 8.77 | 8.63 | 9.06 | 34.16 |
| [T146](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/33169840000) | aigmkt | 1 + comm flags | 8 / 1..72 | 8.79 | 8.64 | 9.07 | 34.29 |
| [T144](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/33166811023) | aigmkt | **8** | 8 / 1..72 | 11.97 | 11.91 | 13.71 | 46.72 |
| [T143](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/33165123199) | aigmkt | **8** | 8 / 1..72 (draft bf16) | 11.93 | 11.84 | 13.70 | 46.54 |

**Four results, in order of size.**

1. **DCP=8 costs +36.5% TPOT at C1** (T144 vs T145: 11.97 → 8.77) and widens the
   tail (13.71 → 9.06 p90). It buys a 6.4× KV pool — 20,580,438 vs 3,218,642
   tokens — which is the one resource that does not bind with a single resident
   request. DCP adds a per-layer a2a + KV gather + LSE merge to a step that is
   already barrier-bound, to parallelise attention work that is negligible at
   batch 9. **SA reached the same conclusion independently: their C1 runs
   `decode_context_parallel_size=1`.** The doc's older "SA has DCP on" claim is
   about their **C52** arm.
2. **The nightly is worth −13.7%** (T145 → T147, 8.77 → 7.57, sole variable =
   175 vLLM commits). Most likely #53942 (`eh_proj` low-latency GEMM, explicitly
   "enabling m=1 and m=2", which is exactly the C1 batch regime, 12.9–25.2%
   kernel improvement) and #53818 (ROCm graphs were being captured on a stream
   that never ran warmup).
3. **Neither ladder size nor scheduler headroom matters at C1** on the nightly:
   mns 8/ladder 72, mns 8/ladder 16 and mns 1/ladder 9 all land 7.57–7.58 mean,
   7.91–7.93 p90. But ladder 9 cuts graph capture **44 s → 7 s** and graph
   memory 1.46 → 0.83 GiB/GPU, and grows the KV pool 2.3%. Free; take it.
4. **`--dcp-comm-backend a2a` and `--cp-kv-cache-interleave-size 1` are inert at
   DCP size 1** (T145 vs T146: identical TPOT, byte-identical KV pool
   3,218,642). They only do work when there is a CP group.

**Unexplained, stated rather than buried:** on the *aigmkt* image, ladder 72 → 16
plus fastsafetensors → auto gave −9.6% (8.77 → 7.93), yet both deltas measured
**inert** on the nightly. The image digest was identical across both runs
(`sha256:7f0dfe6304c9`, build `dev1133+gf94666b60`), so the tag did not move.
Working hypothesis, not established: #53818 fixes graph capture on a stream that
never ran warmup, so a 72-graph ladder had more exposure to that bug on the old
engine. The same pair also leaves the old T123 (6.70) vs T133 (7.18) gap
unexplained — both of its candidate causes are now measured inert.

## Draft model: constraints found while optimising it

- **The DSpark draft cannot leave `TRITON_MLA`.** It is the only ROCm MLA backend
  declaring `supports_non_causal_multi_token_decode`, which the non-causal draft
  requires; `flashinfer_mla` and `tokenspeed_mla` also declare it but are NVIDIA.
  `ROCM_AITER_MLA` inherits the base default `False` and `mla_attention.py`
  raises `"Non-causal multi-token MLA requires an explicitly supported attention
  group"`.
- **Flipping that ClassVar to `True` would be unsafe**, not merely ineffective:
  the flag is a declaration, not an implementation, the aiter ASM path has no
  gqa=64 kernel past qseqlen 1, and because `rejection_sample_method` is
  `synthetic` the accept length is **imposed**, so wrong drafts would still
  report AL 4.00. That whole class of change is untestable on this harness and
  needs real rejection sampling + GSM8K.
- **`TritonMLAMetadataBuilder._cudagraph_support` must stay `UNIFORM_BATCH`.**
  Cudagraph capability is the **minimum across attention groups**, so with
  upstream's `UNIFORM_SINGLE_TOKEN_DECODE` the draft demotes the whole engine to
  PIECEWISE and runs eager: **14.05 → 77.65 tok/s, ITL 71.16 → 12.88 ms**.
  Acceptance test: the server log must show `Capturing model for DSpark
  speculator...`.
- **Draft KV `auto` → `fp8` is TPOT-neutral but grows the KV pool 36.5%**
  (15,077,972 → 20,580,438 tokens in the same 53.84 GiB). The draft was holding
  KV at 2 bytes/element while the target held 1. Keep fp8.

## SA comparison, corrected twice

Reading SA's logs directly (read-only) retired two claims in this document.

| SA run | conc | offload | mns | result |
|---|--:|---|--:|---|
| [32968517728](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/32968517728) | 52 | **dram** | 80 | **8,296** tok/s/GPU |
| [33062469329](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/33062469329) j98484457387 | 52 | **none** | 80 | **8,204** tok/s/GPU |
| 33062469329 j98484457440 | 1 | none | 8 | ITL p50 **8.64** ms |
| [33083417848](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/33083417848) | 1 | none | 8 | ITL p50 **10.21** ms |

- **"SA C52 runs no offload" was wrong** — the 8,296 reference runs `dram`. Both
  their C52 configs are the same shape as our T103 (7,950.6), so we are **3–4%
  behind on an identical recipe**, not blocked on something we cannot run.
- **"`mns` 80 without the offload died 3/3 including both SA C16/C52" was wrong.**
  SA ran exactly that and got 8,204 on `mi355x-amds_01`. The OOM
  (`HSA_STATUS_ERROR_OUT_OF_RESOURCES`) is specific to `mi355x-amd_b23_07`.
  **It is a node limit, not a config limit.**
- **The 1.56× "SA is faster at C1" gap is a client metrics artifact.** Their two
  runs of the *same* config differ only in `RUNNER_NAME`, and one records
  `Decode Duration` min = **0.00 ms**, which makes `1/tpot` explode: `intvty`
  p95 3,465 and p99 22,987 tok/s/user (0.043 ms/token — not physical), mean
  output 1,280.95 tok/s/user with max 65,065, against the clean node's 100.24 /
  207.66. Aggregate throughput is identical (1,236 vs 1,239). **Treat any SA C1
  percentile above p50 as suspect unless `Decode Duration` min is non-zero.**
  Like-for-like we are at parity: ours T133 = 1,237.2 tok/s/GPU, TPOT mean 7.18,
  p50 8.69, p90 11.04.


> **Where the time goes — now MEASURED,** by rocprofv3 on
> [T116](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/32964875218)
> (2.3 GB trace, 6.36M dispatches). See
> [Kimi-K3-Where-The-Time-Goes.md](Kimi-K3-Where-The-Time-Goes.md).
> **The GPU is idle 44.1% of the serving window.** Of the 55.9% it is busy:
> collectives **34.3%**, MLA attention **21.8%**, dense BF16 GEMM **15.3%**,
> MoE (already FP8×MXFP4) 12.5%, KDA 5.2%.
> This retires two earlier claims from this document: attention is **21.8%, not
> 5.6%**, and dense BF16 GEMM is **15.3%, not 94%**.
> **Biggest lever is the idle, then collectives — not any GEMM.**

## FP8 attention weights: tested and rejected (T117–T119)

`amd/Kimi-K3-Quark-MXFP4-AttnFP8` converts exactly the 15.3% dense-BF16 block
(`q/k/v/o_proj`, MLA `q_a/q_b/kv_a/kv_b`) to F8_E4M3, keeping routed experts
MXFP4. Two real blockers were found and **both fixed**:

1. **Load OOM at shard 11/12, 272 MB free.** Not size — it needs 188.2 GB/GPU
   against moonshotai's 195.1. Cause: `fastsafetensors` stages a whole shard
   batch on device (8 × ~15.7 GB ≈ 117 GiB). Fix: `LOAD_FORMAT=auto`.
2. **Wrong MoE kernel.** AMD's config declares MXFP4 *activations*, so
   `OCP_MX_Scheme` resolved to `w_mxfp4_a_mxfp4` → `AITER_MXFP4_MXFP4`
   ("W4A4: CK kernel"), not the `AITER_MXFP4_BF16` + situv2 a8w4 path every
   working run uses. Fix: `global_quant_config.input_tensors = null` in the
   downloaded checkpoint (weights bit-identical; MXFP4 activation quant is
   `is_dynamic`, so nothing stored is invalidated; `*self_attn*` keeps its own
   fp8 spec so FP8 attention survives).

**It still loses, on memory:**

| | moonshotai | AttnFP8 |
|---|---:|---:|
| Model loading took | 192.56 GiB | **240.75 GiB** |
| Available KV cache | 54.41 GiB | **9.38 GiB** |
| GPU KV cache | ~32.7M tok | **5,391,048 tok** |
| Prefix cache hit | 93.8% | **0.0%** |

**+48 GiB/GPU more resident while 7 GB/GPU smaller on disk** — unexplained, and
*not* MoE tile padding (K3's AITER A16W4 SiTU kernel takes native intermediate
size). At conc 52: `Running: 1 reqs, Waiting: 18`. Cancelled rather than measure
KV starvation. **Verdict: trading a 93.8% prefix cache for a block worth +3–5%
is a bad trade.** Parked until the +48 GiB is explained.

## Interactivity: TPOT 22.43 → 10.30 ms at concurrency 1 (2.18×)

Goal was 3×. We got **2.18×**, and the shortfall is structural, not a missing knob.

| Metric | T106 base | T121 DCP off + no offload | **T122 + MTP** | SA c1 |
|---|---:|---:|---:|---:|
| **TPOT mean** | 0.02243 | 0.02165 | **0.01030** | 0.02156 |
| TPOT p90 | 0.02851 | 0.02732 | **0.01357** | 0.02666 |
| **Interactivity** tok/s/user | 44.58 | 46.20 | **97.08** | 46.37 |
| **tok/s/GPU** | 940.0 | 973.3 | **1,210.4** | 980.6 |
| e2e mean | 19.32 s | 18.56 s | **16.48 s** | 18.45 s |
| TTFT mean | 1.429 | 1.367 | **1.964** | 1.241 |
| TTFT p95 | 5.043 | 4.821 | **6.277** | 3.394 |

**Decomposition — this is the useful part:**

- **DCP off + offload off = −3.5% only** (22.43 → 21.65 ms). It closed the gap to
  SA c1 exactly (973.3 vs 980.6), so the ~4% deficit *was* the DRAM offload. But
  it proves TPOT at conc 1 is **barrier-latency-bound**: ~118 global 8-rank syncs
  per decode step, which no config change removes.
- **MTP = the whole 2.18×** (21.65 → 10.30 ms). It doesn't make a step cheaper —
  it amortises the step over ~2.5 accepted tokens.

**Measured acceptance was real, not assumed:** mean acceptance length **2.48–2.55**,
draft acceptance rate **77%**, which validates the 2.51 synthetic constant.

**Why 2.18× and not 2.54×:** MTP costs draft forward passes plus verification, and
its weights and draft KV shrank the pool 4.57M → 2.94M tokens, dropping prefix hit
94.7% → 89.5%.

**The cost, not hidden: TTFT got worse** — mean +37%, p95 +24%. The draft model runs
on prefill steps too and at conc 1 there is nothing to hide it behind. MTP is a
TPOT-for-TTFT trade, not a free win.

**To go further:** `num_speculative_tokens` 2→3 is cheap to try but acceptance decays
with depth. The real ceiling is the ~118 barriers/step — the remaining 10.3 ms lives
there, which is the collectives problem (34.3% of GPU busy), not a knob.
Raising `synthetic_acceptance_length` would hit 3× on paper, but measured acceptance
is 2.48–2.55 and inflating it would be manufacturing the number.

## SA is ahead of us by ~4–5%, and the cause is the DRAM KV offload

SA run [32968517728](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/32968517728)
uses **our own image** (`aigmkt/kimi-k3-vllm:latest`) and beats us at both ends:

| | SA | ours | Δ |
|---|---:|---:|---:|
| **c52** tok/s/GPU | **8,295.9** | 7,914.8 (T120) | **−4.8%** |
| c52 TPOT | 0.0828 | 0.0892 | +7.7% worse |
| **c1** tok/s/GPU | **980.6** | 940.0 (T106) | **−4.1%** |
| c1 TPOT | 0.02156 | 0.02243 | +4.0% worse |
| c1 TTFT p95 | 3.394 s | 5.043 s | **+48.6% worse** |

Their server log confirms **DCP is on** (`decode_context_parallel_size=8`) — the
`dcp1` in their artifact names is just the matrix label, exactly as our own
artifacts say `ep1` while the script hardcodes DCP=8. Diffing the two logs, every
material setting matches: dcp_comm_backend a2a, ROCM_AITER_MLA, max_num_seqs 80,
max_num_batched_tokens 8192, FULL_AND_PIECEWISE, speculative_config None,
`Model loading took 192.56 GiB`, GPU KV 32,756,602 tokens.

**One difference remains: `kv-offloading` — they run `none`, we run `dram`.**

T116's idle attribution independently points at the same thing. `copyBuffer`
(the DRAM offload's host↔device transfer) is the **single largest idle cause**:
**114.2 s of idle immediately before it (17.6% of all idle) and 67.3 s after**,
while contributing only 0.84% of actual GPU work across 697,367 calls.

The offload earned its place when the GPU pool was small (T92 vs T94, non-DCP,
3.3×). DCP now gives us 32.7M tokens on-device, so it has become pure host
traffic. **Next run: conc 52 with `kv-offloading: none`.**

## SOLVED: GPU idle 44.3% -> 28.2% by removing the DRAM KV offload

T116 profiled C52 **with** the offload; T124 re-profiled the same point **without**
it. Both traced identically, so rocprof's overhead cancels.

| | T116 offload ON | **T124 offload OFF** |
|---|---:|---:|
| GPU busy | 55.7% | **71.8%** |
| **Idle** | **44.3%** | **28.2%** |
| Collectives (% busy) | 34.31% | **29.44%** |
| dispatches | 6.36M | 8.47M |
| tok/s/GPU *(traced, not quotable)* | 3,146.0 | **6,821.5** |
| requests | 375/524 | **1123/1232** |

**Idle fell 16.1 points, and precisely the predicted stalls vanished:**

| gap size | T116 | T124 | change |
|---|---:|---:|---|
| 10-200 us | 115.2 s (7.9%) | 124.0 s (8.7%) | unchanged - pure launch overhead |
| 0.2-1 ms | 147.5 s (10.1%) | 104.8 s (7.3%) | -29% |
| 1-10 ms | 118.7 s (8.1%) | 57.6 s (4.0%) | -51% |
| **>10 ms** | **265.7 s (18.1%), n=4,104** | **114.6 s (8.0%), n=877** | **-57%, 4.7x fewer** |

The multi-millisecond stalls were the offload's host<->device traffic. The
sub-200 us launch gaps did not move at all, which is the expected signature:
those are dispatch overhead, not memory traffic.

**What remains, in order:** 10-200 us launch gaps 8.7% of serving (2.06M events,
a host-bound floor); >10 ms stalls 8.0% (877 events at ~130 ms each); 0.2-1 ms
per-step scheduler work 7.3%; 1-10 ms 4.0%. `copyBuffer` is still the top
idle-follower at 71.9 s (down from 114.2 s) since it is a generic HIP copy used
beyond the offload. `merge_attn_states` at 28.0 s is DCP's partial-attention
merge and is intrinsic to DCP.

## Collectives: the DCP group never gets the fast all-reduce

From the server log:

    group 'tp:0'  -> ['AITER_CUSTOM', 'PYNCCL']
    group 'dcp:0' -> ['PYNCCL']
    group 'ep:0'  -> ['PYNCCL']

Generic `ncclDevKernel_Generic_1` is **22.55%** of GPU busy against **3.63%** for
`cross_device_reduce_2stage`, the tuned AITER path TP uses. The cost is
concentrated in the group that did not get the fast backend. T116 detail, per
decode step (~290 collective launches per step on one GPU):

| %busy | calls | us/call | per step | grid | family |
|---:|---:|---:|---:|---:|---|
| 16.93 | 415,346 | 334.5 | 152.3 | 28672 | nccl |
| 3.63 | 80,682 | 369.0 | 29.6 | 40960 | cross_device_reduce (AITER custom) |
| 3.43 | 247,288 | 113.8 | 90.7 | 1792 | nccl |
| 2.19 | 47,312 | 379.0 | 17.4 | 28416 | nccl |
| ~5.9 | few | 5-9 ms | <1 | 4096-16384 | nccl (prefill) |

## Superseded: GPU idle is 44.3% (measured with the offload ON)

From T116 (1467 s serving window, 649.5 s idle across 2,724,405 gaps):

| kernel **after** the gap | idle_s | % serving | % of idle |
|---|---:|---:|---:|
| **`copyBuffer`** (DRAM KV offload) | **114.2** | **7.78%** | **17.6%** |
| `Cijk_` (dense GEMM) | 69.6 | 4.75% | 10.7% |
| `elementwise_manual_unroll` | 52.7 | 3.59% | 8.1% |
| `ncclDevKernel` | 43.5 | 2.97% | 6.7% |
| `merge_attn_states` (DCP partial-attn merge) | 41.9 | 2.86% | 6.5% |
| `fillBufferAligned` | 35.2 | 2.40% | 5.4% |

`fillBufferAligned` is only 6,069 gaps but averages **5.7 ms** each — an
allocator/memset path worth investigating separately.

---

**Best: 7,950.6 tok/s/GPU — DCP=8 + full graphs at concurrency 52 (T103).**
**148% of the SA reference (5,388)** · **64% of the 12,500 target**.
Clean full window, 0 faults, 0 aborts.
**Accuracy verified: GSM8K 98.5%** on the identical config
([T97](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/32819265466)).

**The DCP crash is fixed.** Cause was our **sparse capture ladder**: decode
batches were padded up to the next captured size, and the padded rows read out
of bounds. A dense ladder (every size 1…40, as the SA reference uses) runs
clean. DCP now beats non-DCP *and* keeps 31× the KV pool at ~94% prefix hit.

**Concurrency is where DCP pays.** Doubling to 40 gave **+57%** (4,585 → 7,206)
while KV usage stayed at **11.3%** and prefix hit held at 93.7%. Non-DCP cannot
follow — it is already at 54.3% hit on a 4.4M pool with a 243 GB/rank DRAM
crutch. **But concurrency 72 is past the knee.** T98 aborted at 1,210 s — 29/287
requests failed on timeout, TTFT 20.22 s. Not a capacity problem: **zero
preemptions, KV plateaued at 57.7%**. The cause is our `max_num_seqs = CONC × 2`
convention, which let **91 sequences decode concurrently**. Each step then
attends over ~5× the KV, decode eats the step budget, prefill starves (input
58,000 → 32,000/s), and latency blows out.

**Conc 40 is the sweet spot so far.** The ceiling is decode time per step, not
KV capacity — so the lever is `max_num_seqs`, not more offered load.

| Config | DCP | MTP | Offload | Graphs | Conc | Tok/s/GPU | TPOT | TTFT | Status | Run |
|---|---|---|---|---|---|---:|---:|---:|---|---|
| **SA c52 — ahead of us** | **8** | No | **No** | Full | **52** | **8,295.9** | **0.0828** | **4.37 s** | OK 3,629 s | [SA](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/32968517728) |
| SA reference | No | Yes | Yes | — | 20 | 5,388 | 0.0382 | 12.2 s | OK | [SA](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/31993981851/job/95282381888) |
| **T103 best** | **8** | No | **Yes** | Full | **52** | **7,950.6** | 0.0913 | 4.76 s | **OK 3,628 s** | [T103](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/32855763638) |
| T120 parity, new image | **8** | No | **Yes** | Full | 52 | 7,914.8 | 0.0892 | 4.85 s | OK 3,629 s | [T120](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/32976534358) |
| T105 | **8** | No | **Yes** | Full | 56 | 7,844.0 | 0.1041 | 7.03 s | OK 3,630 s | [T105](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/32876995670) |
| T104 | **8** | No | **Yes** | Full | 64 | 7,650.7 | 0.1264 | 8.99 s | OK 3,627 s | [T104](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/32866152287) |
| T96 | **8** | No | **Yes** | Full | **40** | **7,206.4** | 0.0722 | 3.73 s | **OK 3,627 s** | [T96](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/32812667740) |
| T102 | **8** | **Yes** | **Yes** | Full | 40 | 1,075.4 | 0.1900 | 262.6 s | OK 3,612 s | [T102](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/32846271434) |
| T98 | **8** | No | **Yes** | Full | **72** | 2,039.9 | 0.1542 | 20.22 s | **Aborted 1,210 s** | [T98](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/32821348618) |
| T95 | **8** | No | **Yes** | Full | 20 | **4,585.3** | **0.0445** | **2.24 s** | **OK 3,630 s** | [T95](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/32806967290) |
| T94 non-DCP | No | No | **Yes** | Full | 20 | 4,537.0 | 0.0454 | 2.82 s | OK 3,627 s | [T94](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/32801140212) |
| T64 | No | No | **Yes** | Full | 20 | **4,622.8** | 0.0461 | 2.14 s | **OK 3,612 s** | [T64](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/32436403856) |
| T86 | **8** | No | No | Full | 40 | 4,621.5 | 0.0736 | 4.42 s | Fault 420 s | [T86](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/32754837021) |
| T87 | **8** | No | No | Full | 40 | 4,611.6 | 0.0734 | 5.75 s | Fault 421 s | [T87](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/32758545737) |
| T74 | **8** | No | No | Full | 20 | 4,551.0 | 0.0499 | — | Fault 2,771 s | [T74](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/32699967765) |
| T81 | **8** | No | No | Full | 20 | 4,489.3 | 0.0501 | 2.18 s | Fault 2,997 s | [T81](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/32729114956) |
| T83 | **8** | No | No | Full | 40 | 4,457.1 | 0.0672 | 5.16 s | Fault 425 s | [T83](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/32745214998) |
| T77 | **8** | No | No | Full | 20 | 4,451.9 | 0.0503 | — | Fault 3,013 s | [T77](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/32708108429) |
| T79 | **8** | No | No | Full | 20 | 4,421.6 | 0.0488 | 2.25 s | Fault 2,992 s | [T79](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/32716127572) |
| T73 | **8** | No | **Yes** | Full | 20 | 4,421.2 | 0.0497 | — | Fault 3,033 s | [T73](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/32694320223) |
| T80 | **8** | No | No | Full | 20 | 4,298.4 | 0.0444 | — | Fault 2,987 s | [T80](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/32722526689) |
| T67b DP2/TP4 | No | No | Yes | Full | 20 | 2,998.6 | 0.1140 | 8.57 s | OK | [T67b](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/32458502570) |
| T82 Triton MLA | **8** | No | No | Full | 20 | 2,210.0 | 0.1508 | — | OK 3,608 s | [T82](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/32736571541) |
| T65 | No | **Yes** | Yes | Piecewise | 20 | 2,045.4 | 0.1003 | 67.7 s | OK | [T65](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/32444354043) |
| T66c | **8** | No | **Yes** | **Piecewise** | 20 | 1,574.5 | 0.2174 | 12.4 s | **OK 3,628 s** | [T66c](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/32451498395) |
| T92 | No | No | **No** | Full | 20 | 1,397.0 | 0.2332 | 35.0 s | OK 3,629 s | [T92](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/32789572478) |
| T93c | No | No | No* | Full | 20 | 1,343.0 | 0.2292 | 40.3 s | OK 3,613 s | [T93c](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/32795749613) |
| T88 | No | **Yes** | No | Full | 20 | 982.4 | 0.1659 | 212.7 s | OK 3,605 s | [T88](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/32763413965) |
| T91 Async sched | No | No | No | Full | 20 | 963.5 | 0.2498 | 137.5 s | OK 3,467 s | [T91](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/32782978418) |
| T90 | No | **Yes** | No | Full | 20 | 724.2 | 0.1374 | 374.8 s | OK 3,629 s | [T90](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/32774606161) |

\* T93c asked for offload; the script silently ignored it. See below.

---

# The five things that matter

- **DCP + full graphs crashes.** GPU page fault, 10 runs out of 10. Nothing
  else does. Full graphs alone are fine (T64). DCP alone is fine (T66c).
- **That crash is where the headroom is.** DCP holds 32.8M KV tokens vs 4.4M,
  used only 10%, 94% cache hit. T64 is 14x oversubscribed at 53% hit and needs
  a 243 GB/rank DRAM crutch. Only DCP can take more concurrency.
- **MTP loses 5x.** Cache hit falls 53% -> 10%. Not fixable by pool size or
  scheduling; both tried.
- **`--async-scheduling` loses 4.8x.** It defers block frees, pool fills to
  96%, cache evicts. The reference uses it — do not copy that flag.
- **KV offload is worth 3.3x** on non-DCP. It was silently missing from T74
  onward, so DCP-vs-non-DCP is **not yet a fair comparison**.

---

# The DCP + full graphs crash — SOLVED

**Cause:** our capture ladder was sparse (`1,2,4,8,16,24,32,40`). Every decode
batch was padded up to the next captured size, and the padded rows read out of
bounds under DCP. Fix: capture **every** size, `1…max_num_seqs`.

- Crashed 10 of 10 runs with the sparse ladder. Clean on the first dense-ladder
  run (T95, 3,630 s, 7.35M unique tokens — past every previous crash point).
- Only DCP + full graphs was affected. Full graphs alone (T64) and DCP with
  piecewise (T66c) were always fine.
- Timing scaled with concurrency, not time or tokens: conc 20 crashed ~2,900 s,
  conc 40 ~420 s. More concurrency means more distinct batch sizes, so more
  padding events. `max_num_seqs` made no difference (40 and 80 both ~421 s).
- Signature fit padding: all 8 ranks faulted at the **same low offset** over
  unrelated bases — one buffer overrun by a fixed amount on every rank.

**What confirmed it:** a SemiAnalysis run
([32746060058](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/32746060058))
ran DCP8 + full graphs for 2 hours with **zero faults** using a dense ladder
(every integer 2…60). That run used TRITON_MLA, which first suggested the
backend was the difference — the ladder turned out to be the real variable, and
AITER MLA works fine with a dense ladder while being ~2× faster
(4,585.3 vs ~2,210–2,476).

**Cost of finding it:** six runs were spent varying DCP knobs — offload, opus
rows, comm backend, query replication, `max_num_seqs`, images — while the ladder
sat unexamined. The SA reference script was available throughout and its ladder
was never diffed against ours.

---

**Images** (`vllm/vllm-openai-rocm:nightly-<sha>`)

| Tag | Date | Used by | PR #51705 |
|---|---|---|---|
| `ac7509e2` | 08-13 | T64 | — |
| `5a4c8d99` | 08-19 | T66c | — |
| `d626108b` | 08-20 | T73–T85 | 4 failed hunks |
| `f94666b6` | 08-24 | T86–T94 | 0 failed hunks |

- `ac7509e2` is where our best result was measured — three images back.
- `f94666b6` is current. Still no PR #51705 upstream, gate still present.
  Kept only because our patch applies cleanly on it.
- Image is **not** the cause of the DCP crash: T86 reproduced it on
  `f94666b6` exactly as T85 did on `d626108b`.
- T65 and T67b images were not verified from logs.

**Accuracy**

| Check | Config | Result |
|---|---|---|
| GSM8K 5-shot | T96 config (DCP=8, full graphs, AITER MLA, dense ladder 80, offload) | **98.5%** exact_match ±0.0086 |

Flexible-extract and strict-match agree exactly, so the score is not an
extraction artifact. Run: [T97](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/32819265466), `EVAL_LIMIT=200`.
Set `EVAL_ONLY=true` in the `.sh` to re-run it; `EVAL_LIMIT=full` for the whole
dataset.

**Applied upstream PRs**

| PR | What it gives us |
|---|---|
| [#51705](https://github.com/vllm-project/vllm/pull/51705) | DCP for Kimi-K3; `VLLM_ALLOW_DCP_FULL_CUDAGRAPH` hatch; softmax-LSE for AITER MLA decode. **Vendored** as `pr51705_vllm.diff` — still unmerged upstream, and DCP cannot run without it |
| [#52188](https://github.com/vllm-project/vllm/pull/52188) | Kimi-K3 DCP with DSpark (`prepare_dcp_local_seq_lens`, `cp_local_slot`). **Merged upstream, already in the image** — nothing to apply |

Everything else considered (#51040, #51171, #50619, #50791, #50883, #52269,
#48392, #51203, ROCm/aiter#4915) was either inert, conflicting or inapplicable —
see `archive/README-patches.md`.

**Applied patches** (`apply_kimi_k3_patches.sh`, 3 of them)

| Patch | What it does |
|---|---|
| `pr51705` | Vendored vLLM PR — DCP support and the full-graph hatch |
| `pr51705-rejects` | Adds `enable_dcp_q_replicate` to `MultiHeadLatentAttention.__init__`; the PR's own hunk rejects on this image |
| `kv-blockpool` | Clamps a negative block count |

- `pr51705` is **vendored in-repo** since T76. Before that it was fetched live
  and pinned by SHA — the PR was force-pushed twice in one day and killed two
  runs, hence the vendoring.
- `pr51705-rejects` fixes `enable_dcp_q_replicate` missing from
  `MultiHeadLatentAttention.__init__`. Silent with speculation off; kills every
  rank at init with MTP on. Needed on `d626108b`, not on `f94666b6`.
- All three are **pre-applied** in `aigmkt/kimi-k3-vllm:latest`, so the runtime
  patch step is commented out in the launcher.
- Archived patches and the reasoning for each removal:
  `archive/README-patches.md`.

---

# What to do next

1. ~~GSM8K accuracy gate~~ — **done, passed.** 98.5% exact_match (±0.0086),
   5-shot, flexible-extract and strict-match identical, run on the exact T96
   config (DCP=8, dense ladder 80, AITER MLA, offload). This mattered because
   the bug the dense ladder fixed was an out-of-bounds *read*, which can return
   garbage instead of crashing. It does not. T96's 7,206.4 is real.
2. **Decouple `max_num_seqs` from concurrency.** Conc 72 with `max_num_seqs`
   144 aborted (T98); the in-flight decode batch, not the pool, is the limit.
   Retry high concurrency with `max_num_seqs` held near 80.
3. ~~Re-test MTP~~ — **done, MTP is closed.** T102 ran DCP+MTP cleanly and lost
   ~85%: 1,075.4 vs 7,206.4. Dense ladder, clean draft, DCP working — no
   confounds left. Cause is memory: draft + graphs halve the KV pool
   (32.8M → 14.2M), prefix hit falls 93.7% → 28.6%, workload thrashes. DCP's
   advantage *is* KV capacity; speculation spends it.

4. **Concurrency 1 and 4 — interactivity runs.** Different axis from everything
   above: these measure **TPOT/TTFT, not throughput**, and will look poor on
   tok/s/GPU by design (~340 and ~1,300 predicted). Expect TPOT ~19–24 ms.
   This is where DCP should look best — one request's KV is sharded over all 8
   GPUs, so attention is 8-way parallel at 1/8 the KV read per rank.

Reminder: the capture ladder must be dense and must track `max_num_seqs`
exactly. Sparse ladders caused the crash.

**Concurrency sweep — complete.** Throughput peaks at 52; latency is best at 4.

| conc | n | Tok/s/GPU | TPOT | TTFT p50 | Prefix hit | Run |
|---:|---:|---:|---:|---:|---:|---|
| **1 + MTP** | 8 | **1,210.4** | **0.0103** | 1.09 s | 89.5% | [T122](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/32992548212) |
| 1 (DCP off, no offload) | 8 | 973.3 | 0.0217 | 0.93 s | 94.7% | [T121](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/32986150759) |
| 1 | 4 | 940.0 | 0.0224 | 0.96 s | 95.5% | [T106](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/32887252465) |
| **4** | 6 | 1,525.1 | 0.0256 | **0.87 s** | — | [T107](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/32895236167) |
| 40 | 43 | 7,206.4 | 0.0722 | — | 93.7% | [T96](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/32812667740) |
| **52** | 54 | **7,950.6** | 0.0913 | — | 93.7% | [T103](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/32855763638) |
| 56 | 66 | 7,844.0 | 0.1041 | — | 92.7% | [T105](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/32876995670) |
| 64 | 80 | 7,650.7 | 0.1264 | — | 90.0% | [T104](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/32866152287) |

- **Best throughput: 7,950.6 tok/s/GPU @ conc 52** — 148% of the SA reference,
  64% of target, GSM8K 98.5%.
- **Best interactivity: conc 4** — TPOT 25.6 ms, TTFT p50 0.87 s. Beats conc 1 on
  TTFT *and* throughput; conc 1's `max_num_seqs 4` was too tight and queued the
  replay's branched sub-requests.
- `max_num_seqs` needs **branching headroom** (~1.25× conc). Too tight raises
  TTFT (T106); too loose lets residents run away and abort (T98, `mns` 144 → 91
  residents → TTFT 20.2 s).

**Throughput peaks at concurrency 52 — it does not keep climbing.**

| conc | n | tok/s/GPU | TPOT | TTFT | prefix hit |
|---:|---:|---:|---:|---:|---:|
| 40 | 43 | 7,206.4 | 0.0722 | 3.73 s | 93.7% |
| **52** | 54 | **7,950.6** | 0.0913 | 4.76 s | 93.7% |
| 56 | 66 | 7,844.0 | 0.1041 | 7.03 s | 92.7% |
| 64 | 80 | 7,650.7 | 0.1264 | 8.99 s | 90.0% |

Flat-topped between 52 and 56 (1.3% apart), then falls. Cause is prefix-hit
erosion (93.7% → 90.0%) as concurrent contexts multiply, plus TPOT rising faster
than resident count.

**The model below is superseded and kept only for its TPOT/TTFT fits.** Its
throughput term `T(n)=16,823·n/(n+48)` is monotonic and predicted conc 64 would
beat 52 by ~20%; it fell 3.8% instead. The earlier "~10,000–10,500 ceiling"
followed from that form and is **wrong** — measured peak is ~7,950.

**Scaling model** (fitted on T95 n=18, T96 n=36, T98 n=91; `n` = resident requests)

```
TPOT(n) = 0.01745 + 0.001503*n     s     (n=36 predicted 0.0716 vs 0.0722 measured)
TTFT(n) = -7.06   + 0.300*n        s
T(n)    = 16,823 * n/(n + 48)      tok/s/GPU
```

- Best realistic landing **~10,000–10,500 tok/s/GPU** at conc ~64–80 — 80–84% of target.
- **Latency binds first, not cache.** Throughput asymptote is 16,823, above
  target, but 12,500 needs n≈145 → TTFT ~36 s. T98 aborted at 20.2 s.
- **KV offload adds capacity DCP cannot use**: 11.3% pool usage at conc 40 and
  `ext_cache_hit` 0.0–0.6%. It was worth 3.3× for *non-DCP* only.
- Passing 12,500 needs the **0.0015 s/request decode slope** cut — a faster MLA
  decode kernel or shorter effective context. Not more concurrency, not more cache.

# Known traps

| Trap | Cost |
|---|---|
| **Sparse capture ladder under DCP** | **GPU page fault** — use every size 1…N |
| `--async-scheduling` on | 4.8x |
| KV offload block missing from the `.sh` | 3.3x |
| Capture ladder below `max_num_seqs x (1+spec)` | Decode silently drops to piecewise |
| `spec-decoding` matrix field | Also picks the script filename |
| `per_gpu.total_tput_tps` | Divides by `tp`, not GPU count — 2x wrong on DP |
| `agg_bmk.json` `spec_decoding` | Reports the matrix label, not engine state |
| Workflow "completed success" | Says nothing about whether the benchmark aborted |

---

# Appendix — detail, evidence and history

Everything below is the working record: mechanisms, profiling, falsified
hypotheses, corrections and the full trial ledger.

## 2. Findings, in priority order

### 2.0 DCP's penalty was a ROCm-only cudagraph downgrade, not DCP itself

For most of this investigation DCP looked structurally slow — roughly a third of
plain TP8. It was not. `vllm/platforms/rocm.py` silently downgrades the
cudagraph mode whenever DCP is on:

```python
if compilation_config.cudagraph_mode.has_full_cudagraphs():
    # decode context parallel does not support full cudagraphs
    if parallel_config.decode_context_parallel_size > 1:
        compilation_config.cudagraph_mode = CUDAGraphMode.PIECEWISE
```

This gate exists **only on the ROCm path** — `cuda.py` has zero occurrences, so
NVIDIA DCP keeps full graphs. Piecewise splits the graph at every attention op
and hands each decode step back to the host to relaunch. Host-launch latency was
already the profiled bottleneck (3.4), so this downgrade lands directly on the
limiter, and every DCP number recorded before this point was measuring the
downgrade rather than DCP.

Lifting it (vLLM PR #51705, which gates the same code on
`VLLM_ALLOW_DCP_FULL_CUDAGRAPH`):

| | piecewise | full graphs | |
|---|---:|---:|---|
| tok/s/GPU | 1,574.5 | 4,551.0 | +189% |
| TPOT | 0.2174 | 0.0499 | −77% |
| prefix cache hit | 71.0% | 94.3% | |

DCP now sits within 2% of the best non-DCP run while carrying **31× the KV
capacity** (32.8M vs 4.4M tokens). That combination — reference-class TPOT at
32× cache — is the only configuration on this page with a credible path past the
reference, because it is the only one where concurrency can rise without
evicting. KV usage at concurrency 20 was 10.5% with zero preemptions.

**Blocker.** Both full-graph runs die on a GPU page fault before the window
closes (2,771 s and 3,033 s). Until that closes these are not results. See 2.0b.

### 2.0b The full-graph page fault — evidence and open question

All eight ranks fault simultaneously with `VM_L2_PROTECTION_FAULT_STATUS`,
`PERMISSION_FAULTS: 0x3`, IH client `0x1b (UTCL2)`, TCP client — a
vector-memory out-of-bounds **read**.

What the addresses say: the eight faulting addresses share the same low offset
(six at `...ce000`, two at `...cc000`) across completely unrelated bases. That is
one logical per-rank buffer overrun by a *fixed* amount on every rank. A KV or
block-table walk running off the end would scatter across ranks instead.

**The trigger is DCP, not full graphs.** T64 ran non-DCP with the *same*
`FULL_AND_PIECEWISE` mode and completed 3,612 s cleanly. So full graphs alone
are safe; DCP + full graphs is the failing combination.

**Every DCP configuration knob has now been falsified.** Six runs, one variable
each, all landing in the same band:

| Run | variable moved | duration | unique tokens | tok/s/GPU |
|---|---|---:|---:|---:|
| T73 | KV offload on (no opus patch) | 3,033 s | 6.36M | 4,421.2 |
| T74 | offload off | 2,771 s | 6.06M | 4,551.0 |
| T77 | opus rows present | 3,013 s | 6.51M | 4,451.9 |
| T79 | capture ladder 40 | 2,992 s | 6.34M | 4,421.6 |
| T80 | `ag_rs` comm backend | 2,987 s | 6.42M | 4,298.4 |
| T81 | `VLLM_DCP_Q_REPLICATE=0` | 2,997 s | 6.35M | 4,489.3 |

Falsified: opus rows, KV offload, capture ladder, comm backend, query
replication. The fault is invariant to configuration, so it lives in the DCP +
full-graph decode path itself, not in how we drive it.

Two corrections to earlier reasoning on this page, both mine:

- The capture-ladder theory was wrong *on its own terms*. `FULL_AND_PIECEWISE`
  already routes mixed prefill+decode batches to piecewise and reserves full
  graphs for uniform decode batches, so the 48/64 slots were never the
  mixed-batch path claimed.
- T80 was not the clean single-variable test it was described as. PR #51705
  redefines `VLLM_DCP_Q_REPLICATE` as *auto — on when `dcp_comm_backend=="a2a"`*,
  and T80 ran `ag_rs` while still pinning it on. `ag_rs` was tested in an
  off-design pairing and is not cleanly refuted (it was also ~3% slower).

**The trigger scales with concurrency, not with elapsed time or accumulated
work.** Two results pin this down, and the second corrects the first:

- T82 ran **3,607.6 s** clean — longer than every faulting run — while reaching
  only 4.12M unique tokens. So it is not elapsed time, and a run can look clean
  purely by not doing enough work.
- T83 raised concurrency 20 → 40 and faulted at **424.5 s** with only
  **1,629,728** unique tokens. So it is not accumulated work either.

| | concurrency 20 (6 runs) | concurrency 40 (T83) |
|---|---|---|
| fault at | ~2,900 s | **424.5 s** |
| unique tokens | ~6.3M | **1.63M** |
| kv_usage | ~11% | 25.9% |
| requests completed | ~924–958 | 243 |

Concurrency brings the fault **7× sooner** while the request rate rose only
~1.8×. The tight ~6.3M clustering across the six earlier runs was an artifact of
holding concurrency fixed at 20, **not** a property of the bug — anything built
on that correlation should be discarded. The evidence now points at something
scaling with the number of *concurrent* sequences.

Practical consequence: the fault reproduces in **~7 minutes** at concurrency 40
instead of ~50, which makes previously unaffordable diagnostics (kernel
serialization, component bisects) cheap.

**Piecewise safety is real, not an artifact.** T66c reached **14,317,525**
unique tokens clean, well past the ~6.3M trigger, so it was not merely too slow
to get there.

**Conclusion: this is an upstream bug, not a misconfiguration.** The fault is
invariant to everything available to us:

| Dimension varied | Result |
|---|---|
| KV offload on/off | faults |
| aiter opus rows present/removed | faults |
| capture ladder 64 / 40 / 80 | faults |
| comm backend `a2a` / `ag_rs` | faults |
| `VLLM_DCP_Q_REPLICATE` 1 / 0 | faults |
| decode backend AITER / TRITON | inconclusive (T82/T84) |
| concurrency 20 / 40 | faults, 7× sooner at 40 |
| `max_num_seqs` 40 / 80 | faults, **no effect on timing** |
| image `d626108b` / `f94666b6` | faults identically |
| `AMD_SERIALIZE_KERNEL=3` | faults, no attribution |

**It scales with in-flight decode batch size, not with buffer sizing.** T87 held
`max_num_seqs=40` while running concurrency 40 and faulted at 420.6 s — the same
as `max_num_seqs=80`. That eliminates the buffer-sizing hypotheses outright,
since `paged_kv_indices` (`max_num_seqs × max_model_len`) and the aiter MLA
metadata workspace (`max_num_reqs = max_num_seqs`) are both sized by the
quantity held constant.

Corroborating: `VLLM_ALLOW_DCP_FULL_CUDAGRAPH` defaults to **`False`** upstream —
the PR author opted not to enable this combination by default.

**Reproduction for an upstream report:** concurrency 40, DCP=8,
`FULL_AND_PIECEWISE`, `VLLM_ALLOW_DCP_FULL_CUDAGRAPH=1` → GPU page fault on all
8 ranks in **~420 s**, 10 of 10 runs. All ranks fault at one shared low offset
over unrelated bases. No kernel attribution available: `Reason: Unknown`, and
the GPU coredump handler fails with `execvp` ENOENT.

**The image is the one variable never changed.** Every run T64–T83 used
`nightly-d626108b` (2026-08-20). Checked against the newest nightly
`f94666b6` (2026-08-24, 139 commits later): PR #51705 is **still not merged**
upstream and the ROCm DCP→PIECEWISE gate is **still present**, so a newer image
does not fix this by itself. It is a better base for one concrete reason —
the vendored diff applies with **0 failed hunks** there versus **4** on
`d626108b`, which removes the silent `enable_dcp_q_replicate` reject that killed
T78. **Now confirmed end-to-end (T86): the newer image faults identically**
(419.9 s), and it genuinely ran the new build (`dev1133+gf94666b60` vs
`dev994+gd626108b1`), so this is not a cache artifact. We stay on it for the
cleaner patch application, not for any fix.

**The aiter opus-rows patch is exonerated**, on both fault and performance:

| Run | opus rows | offload | duration | tok/s/GPU | outcome |
|---|---|---|---:|---:|---|
| T73 | present | yes | 3,033 s | 4,421.2 | fault |
| T74 | removed | no | 2,771 s | 4,551.0 | fault |
| T77 | **present** | no | 3,013 s | 4,451.9 | fault |

With opus rows left fully intact the run still faults, at the same point, with
the same throughput. An earlier reading here — that untuned-GEMM fallbacks rose
~480× (88 → 42,320) because of the opus patch — was **wrong**. T77 logged
**46,056** of those messages *with* opus rows present. That gap was piecewise vs
full graphs changing batch shapes, not opus. Adjacency of untuned-GEMM lines to
the fault was never evidence either: they run 42,320 deep from line 202 onward.

What the fault correlates with:

| Run | unique tokens at fault | kv_usage | prefix hit |
|---|---:|---:|---:|
| T74 | 6,063,765 | 11.8% | 94.2% |
| T77 | 6,508,095 | 11.8% | 94.1% |
| T73 | 6,359,477 | 10.5% | 94.3% |
| **T64 (non-DCP, clean)** | **62,697,249** | 41.1% | 53.4% |

T64 ran ten times past the DCP fault point without incident, so the unique-token
count is not itself the trigger. Note that at fault time the DCP runs have
**never evicted anything** — 94% prefix hit against a 31× pool, so sequences
grow monotonically. That fits a per-request length limit frozen into a captured
graph, which only DCP's pool is large enough to let sequences reach.

Ruled out by inspection: `paged_kv_indices`, the persistent page-index buffer,
is sized `max_num_seqs × max_model_len` (167 MB) against at most ~8.4M entries
in use.

Demoted: `min_kv_seq_len` frozen into the captured graph at
`rocm_aiter_mla.py:1268`, where Gluon derives `NUM_KV_SPLITS` from it as launch
geometry. A stale split count is hazardous when replay sequences are *shorter*
than capture, but these faults arrive only after context *growth*.

Closest upstream analogue: vLLM **#50791**, which sizes the FlashInfer sparse
MLA workspace for DCP. Not applicable to us (B200/FlashInfer), but it documents
this exact bug class — DCP forces LSE during decode, the LSE slab is carved from
a DCP-unaware workspace, and every DCP worker dies at once. The ROCm-side
equivalent is the place to look next.

**DCP is parked meanwhile.** At concurrency 20 it does not win anyway
(4,451–4,551 vs non-DCP 4,622.8); its value is capacity headroom for higher
concurrency, which is worth nothing until a run survives the window.

### 2.1 The decode attention backend is worth 2.4×
Forcing `--attention-backend TRITON_MLA` on the target model costs **72% of
throughput and 60% of TPOT** versus leaving it unset and letting ROCm select
`ROCM_AITER_MLA`.

| | forced TRITON_MLA | AITER MLA | Δ |
|---|---:|---:|---|
| tok/s/GPU | 2,685.0 | **4,622.8** | **+72.2%** |
| TPOT mean | 0.1161 | **0.0461** | **−60.3%** |
| TTFT mean | 3.450 s | **2.143 s** | −37.9% |
| requests OK | 569/611 | 1,046/1,090 | +84% |

Single variable: same image, patches, KV pool (byte-identical 4,376,929 tokens),
concurrency and spec setting. The reference sets **no** target backend; the
`TRITON_MLA` string in its config belongs to `--speculative-config`, i.e. the
**draft** model. AMD documents AITER MLA at 1.2–1.6× TRITON_MLA; measured here
it is 2.4× on this workload.

### 2.2 All three parallelism/speculation features lose against plain TP8

| Feature | Δ vs TP8 | Mechanism |
|---|---:|---|
| **DCP=8** | **−66%** | Collective-bound. Delivered its full capacity promise — 31.26× KV, 71.0% prefix hit, ≤15% utilised, `ext_cache_hit` 0.0% — and still lost 3×. |
| **MTP** | **−56%** | Accelerates decode, which is **0.6% of scored tokens**, while consuming 36% of the KV pool serving the other 99.4%. TTFT 2.14 s → 67.7 s. |
| **DP attention** | **−35%** | Best of the three. TPOT p50 0.0652 vs mean 0.1140 — a spread no other configuration shows. |

### 2.3 KV capacity is not the constraint
DCP=8 supplies 7.5× the KV pool (32,779,397 vs 4,376,929 tokens), raises prefix
hit from ~52% to 71.0%, never exceeds 15% utilisation and never spills to the
DRAM tier — and is still 3× slower. **99.4% of scored tokens are prefill**
(input 21,360 tok/s vs output 119), and prefill throughput is set by compute and
synchronisation, not by spare decode KV.

### 2.4 The workload is prefill-dominated
Any optimisation targeting decode addresses 0.6% of the scored tokens. This
single fact explains why MTP loses, why DCP's decode-side capacity is worthless
here, and why TTFT regressions matter more than TPOT regressions.

### 2.5 DCP + MTP — no longer blocked upstream (finding below is superseded)

**Superseded as of the current image.** vLLM **#52188 "[Spec decode] Support
Kimi-K3 DCP with DSpark"** merged 2026-08-17 and is present in
`nightly-d626108b`. It landed `prepare_dcp_local_seq_lens` and `cp_local_slot`
in `cp_utils.py` — including at `spec_decode/dflash/cudagraph.py:45`, which is
precisely the "local sequence lengths are not advanced between draft steps"
gap described below. Together with vendored PR #51705, which supplies
`supports_non_causal_multi_token_dcp` on Triton MLA and `supports_dcp_with_varlen`
on AITER MLA, both capability flags the validator looks for are now satisfied.

Related, still open upstream: **#52269** (Kimi-K3 DSpark under DCP, draft — not
worth vendoring yet) and **#48392** (dense GQA/MHA drafts; Kimi-K3 is MLA, so
#52188 already covers it).

Untested here so far, because DCP + full graphs still faults (2.0b). The
original analysis is kept below for the record:

`mla_attention.py::_validate_dspark_dcp_support` requires one of two capability
flags. Across the entire image:
```
supports_non_causal_multi_token_dcp = True  ->  tokenspeed_mla.py  (only)
supports_dcp_with_varlen            = True  ->  flashinfer_mla, flashattn_mla,
                                                flashinfer_mla_sparse, flashmla_sparse
```
Neither `triton_mla.py` nor `rocm_aiter_mla.py` declares either. `TOKENSPEED_MLA`
is absent from `platforms/rocm.py` and not installed. `triton_mla.py` also
disables its draft path under DCP in code:
```python
# DCP local sequence lengths are not advanced between draft steps.
self.supports_draft_decode_metadata_update = self.dcp_world_size == 1
def update_draft_decode_metadata(self, _metadata): pass   # no-op
```
Setting the flag alone would let drafting proceed on stale per-rank KV ranges —
wrong output, no crash. This is an implementation gap, not a configuration gap.

### 2.6 The reference's advantage is speculation, not parallelism
The MI355X reference runs **no DCP and no EP** — plain TP8 + MTP. Its command
carries no `--decode-context-parallel-size`, no `--max-num-batched-tokens` and no
`--attention-backend`. Its 5,388 is measured with `rejection_sample_method:
synthetic` at `synthetic_acceptance_length: 2.51`.

### 2.7 Chunked-prefill sizing controls the KV pool
`--max-num-batched-tokens` feeds
`mla_attention.py::determine_chunked_prefill_workspace_size`. Pinning it reserves
memory the KV pool cannot then have. With MTP enabled, omitting the flag moved
the pool from 1,385,293 → 2,646,059 tokens (1.32× → 2.52×), exactly matching the
reference. It is a memory knob, not only a scheduling knob.

### 2.8 DP attention is memory-feasible at intermediate layouts
Measured from the checkpoint: experts 1,446.46 GB (92.67%), non-expert 114.40 GB
(7.33%). With EP sharding experts across all ranks, non-expert weights shard
within the TP group and replicate only across DP:

| Layout | per-GPU | fits (259 GB budget) |
|---|---:|---|
| DP1/TP8 | 195.1 GB | yes — current |
| **DP2/TP4/EP8** | **209.4 GB** | yes |
| DP4/TP2/EP8 | 238.0 GB | yes |
| DP8/TP1/EP8 | 295.2 GB | **no** — exceeds the 288 GB card |

Only the TP1 extreme fails. Any DP layout requires `ep > 1` or the experts do not
shard.

---

## 3. Limiting kernels and where the time goes

Torch profiler, decode-only windows, 8 ranks. **Caveat: all existing traces were
captured with the forced TRITON_MLA backend** — the 2.4× handicap. Traces of the
corrected stack are in progress.

### 3.1 Whole-trace breakdown (rank 0, 15 s decode window)

| | DCP=8 | non-DCP |
|---|---:|---:|
| Total GPU kernel time | 14.801 s | 5.033 s |
| Kernels launched | 121,030 | 235,224 |
| `aiter::cross_device_reduce_2stage<bf16,8,false>` | **78.37%** / 10,602 calls | 55.48% / 22,599 |
| `ncclDevKernel_Generic_1` | 8.22% / 950 | — |
| `mscclKernel_Sum_hip_bfloat16_Simple` | 6.06% / 912 | — |
| `mfma_moe1_silu_mul_afp8_wfp4` | 1.20% / 3,496 | 8.37% / 7,452 |
| `mfma_moe2_afp8_wfp4_bf16` | 0.71% / 3,496 | 4.96% / 7,452 |
| `_gemm_a16_w16` | — | 3.24% / 7,533 |
| `_attn_res_kernel` | — | 2.34% / 15,147 |
| **collectives / compute** | **92.65% / ~7%** | 56.05% / ~44% |

### 3.2 The decisive measurement

```
aiter::cross_device_reduce_2stage   SAME kernel, SAME 287 KB payload
  non-DCP :     8 us per call  (p50, every rank)
  DCP=8   : 1,094 us per call  (11.599 s / 10,602 calls)
  ratio   : 137x
```

Per-step kernel density is unchanged (DCP 1,345 vs non-DCP 1,232 kernels/step)
and compute per step is identical (11.6 vs 11.5 ms). **The kernel did not become
slower; it was made to wait.**

Per decode step, ~118 all-reduces in **both** arms:

| | all-reduce | compute | steps/15 s | TPOT |
|---|---:|---:|---:|---:|
| non-DCP | 14.6 ms | 11.5 ms | ~191 | 43 ms |
| DCP=8 | **129 ms** | 11.6 ms | ~90 | 167 ms |

### 3.3 Root cause on the profiled stack: host-launch starvation

An all-reduce returns when the **last** rank arrives, so the rank spending the
least time inside it arrives late.

```
DCP=8 all-reduce seconds per rank
  r0 11.72  r1 11.71  r2 10.13  r3 9.32  r4 10.59  r5 11.69  r6 8.47  r7 2.07  <- straggler
  p50 ~1000 us on 7 of 8 ranks -> waiting on EVERY call, not a tail
  non-DCP for contrast: p50 8-9 us on every rank
```
```
Host timeline over the same 15.20 s window
  straggler  gpu_busy  5.15 s  idle 10.05 s (66%)  123,900 launch gaps
             gap p50 62 us  p90 190 us  max 1.2 ms
  normal     gpu_busy 14.86 s  idle  0.35 s ( 2%)  116,678 gaps
             gap p50 0 us  p90 0 us
  host op totals IDENTICAL: 96,452 ms vs 97,114 ms
```
The straggler's GPU waits ~62 µs after each kernel for the host to launch the
next, ~124k times ≈ 10 s idle. Host *work* is identical between ranks, so this is
launch **delay**, not extra work. The other seven block inside the all-reduce,
which is why the cost is *attributed* to collective time.

### 3.4 Limiters, ranked

| # | Limiter | Evidence |
|---|---|---|
| 1 | **~118 global barriers per decode step** | Identical count in both arms. Each all-reduce is a hard sync across 8 ranks, so per-rank jitter is amplified 118× per step — this converts 62 µs of host delay into 129 ms/step. |
| 2 | **Host launch latency on the critical path** | Straggler idle 66%, ~124k gaps, identical host work, `max_concurrent_batches=1` leaves nothing to overlap. |
| 3 | **DCP adds synchronisation while sharding no work** | compute/step 11.5 → 11.6 ms. Only 24 of 93 layers are full-attn MLA, so there is little to shard. |
| 4 | **Workload shape** | 99.4% prefill; decode-side capacity is not the constraint. |

### 3.5 Which kernels would need acceleration

**No kernel needs to be made faster.** The dominant 78.37% is *blocking time*
charged to a kernel that completes in 8 µs when not waiting. Substituting the
implementation (PYNCCL for AITER custom) produced **−23% throughput, +31% TPOT** —
replacing the kernel makes it worse.

Addressable work, by measured share and tractability:

| Target | Share / status | Notes |
|---|---|---|
| **Fusion to cut barrier and launch count** | attacks limiters #1 and #2 | NVIDIA ships `VLLM_ENABLE_K3_LATENT_MOE_TAIL_FUSION`; **absent from the ROCm build**. The only lever that hits both limiters. |
| **DCP's own collectives** | `ncclDevKernel_Generic_1` 8.22% + `mscclKernel_Sum` 6.06% ≈ **14.3%** | Genuine work (gather + LSE merge), but a minority — perfect execution caps the win near 14%. |
| `cp_lse_ag_out_rs` direct path | ported, **−0.9%** | `direct_dcp_a2a_lse_reduce` hand-ported to HIP. Functional, no gain. |
| `q_gather` / `kv_gather` direct path | **not portable** | `multimem.st.*` PTX (NVLink hardware multicast, `__CUDA_ARCH__ >= 900`). No AMD equivalent. |
| Query all-gather elimination | `VLLM_DCP_Q_REPLICATE` | Requires `DCPGroupColumnParallelLinear`; Kimi-K3 builds plain `ColumnParallelLinear` (`models/kimi_k3/amd/linear.py:376,384`). Addresses part of the 14.3%. |
| Remove host from the launch path | untested | Full-step graph capture. Async scheduling is the cheap version and did not help. |

**Summary:** the cost is *serialisation*, not arithmetic. Kernel-level work can
address at most the ~14.3% that DCP's own collectives occupy. The other 78% is
seven GPUs idling in a barrier waiting on one host thread, fixable only by
removing barriers or removing the host from the launch path.

---

## 4. Recommended direction

1. **Keep the target attention backend unset.** Largest single effect measured.
2. **Do not enable DCP or MTP** on this workload at c20 with this request mix.
3. **Pursue fusion** — `VLLM_ENABLE_K3_LATENT_MOE_TAIL_FUSION` has no ROCm
   equivalent, and barrier/launch count is the binding constraint.
4. **Investigate DP attention's latency tail** — TPOT p50 0.0652 against mean
   0.1140 suggests the all-reduce removal works and DP-group imbalance costs the
   difference. Load balancing across DP groups is untested.
5. **Upstream request, precisely scoped:** implement
   `update_draft_decode_metadata` for a ROCm MLA backend so DCP local sequence
   lengths advance between draft steps, then declare
   `supports_draft_decode_metadata_update` and
   `supports_dcp_with_varlen` / `supports_non_causal_multi_token_dcp`.

---

## 5. Measurement hazards in this harness

Two reported fields produce incorrect conclusions if taken at face value.

| Field | Behaviour | Impact |
|---|---|---|
| `agg_bmk.json` → `spec_decoding` | Reports the **matrix label**, not engine state. Read `'none'` while the engine logged `speculative_config=SpeculativeConfig` and 364 SpecDecoding samples. | Caused a run to be recorded as speculative when it was not, and a non-existent "2.7× decode regression" (a spec-on vs spec-off comparison). |
| `agg_bmk.json` → `per_gpu.total_tput_tps`, and the job footer `Throughput per GPU` | Divides TOTAL by **`tp`**, not the GPU count. Correct whenever `tp` == GPUs; wrong by `8/tp` for any DP layout. | DP2/TP4 reported **5,997.2 tok/s/GPU** — apparently beating the reference — against an actual **2,998.6**. Confirmed from the same run's `world_size=8`, `rank=0..7`, `local_rank=0..7`, `EngineCore_DP0/DP1`. |

**Any DP or DP+EP result published from this harness is overstated 2× unless the
divisor is corrected.** Aggregates should be reconciled against the engine log.

A third hazard is reproducibility rather than measurement. vLLM PR #51705 was
fetched at runtime and pinned by sha256; the PR was then force-pushed **twice in
one day**, and each time the pin correctly refused to apply — which silently
removed the DCP patches and killed the run at engine init with `Decode Context
Parallelism (DCP) requires attention implementations to return the softmax LSE
during decode`. The pin behaved exactly as designed, but a hard pin against an
actively developed PR guarantees repeated breakage. The diff is now **vendored
in-repo** (`pr51705_vllm.diff`, `vllm/`-only), so the exact bytes that produced a
result sit next to the script that applies them, with no network fetch. Any
comparison spanning that change must confirm which revision was actually in the
image.

---

## 6. Configuration reference

### 6.1 In-container patches

| # | Name | Purpose |
|---|---|---|
| [1] | aiter pybind11 | Standalone pybind11 (internals v12) outranks torch's bundled copy (v11) via `-I` vs `-isystem`. Separate type registry per internals id → JIT module cannot see `aiter_tensor_t` → `TypeError` at warmup. Unblocks `ROCM_AITER_FA` prefill. |
| [2] | TritonMLA cudagraph | Raises `_cudagraph_support` to `UNIFORM_BATCH` so the DSpark drafter keeps FULL cudagraphs. Measured 14.05 → 77.65 tok/s single-stream (5.52×). **Incompatible with speculation on current nightlies** — it permits full capture, which then trips `assert m.max_query_len <= reorder_batch_threshold`. Skipped when spec is on. |
| [3] | KV block-pool clamp | `allocate_external_computed_blocks()` can pass a negative count to `get_new_blocks`, which is silently destructive (`num_free_blocks -= n` increases it; `range(n)` iterates zero times). A later pop walks past the tail. Load-dependent: c10 died at 3612 s, c12 at 487 s, c16 at 354 s. Requires `--kv-transfer-config`. |
| [4] | DCP-LSE plumbing | Plumbs softmax LSE + round-robin CP through AITER MLA decode. aiter's LSE is natural-log with sm_scale folded, `[B,H]` fp32. Superseded by [5]. |
| [5] | vLLM PR #51705 | Upstream DCP for Kimi-K3 DSpark, **and** the `VLLM_ALLOW_DCP_FULL_CUDAGRAPH` gate that unlocks 2.0. Now **vendored** in-repo as `pr51705_vllm.diff` and applied from disk. Does **not** fix the 0x1016 fault. |
| [6] | DCP block-table sizing | **The 0x1016 fix.** Tables sized `cdiv(max_model_len, block_size × dcp_size)` but indexed with the undivided `max_model_len`. Boundary tracks the ratio exactly: DCP8/bf16 faulted at 134,400; DCP4/bf16 at 262,656; DCP8/fp8 at 135,168 — block **count**, not bytes. |
| [7] | direct DCP a2a (ROCm port) | vLLM compiles the kernel only under `VLLM_GPU_LANG == CUDA`. The a2a combine ported to HIP (`st/ld.global.{release,acquire}.sys.u32` → `__hip_atomic_*` at SYSTEM scope). Gather kernels use `multimem.st.*`, not ported. **−0.9%.** |
| [8] | DCP gathered-head sizing | Supplies PR #51705's failing hunk (`_decode_num_heads = num_heads × dcp_world_size`). Requires image `5a4c8d99`; on `ac7509e2b` the builder predates the plumbing and lacks `dcp_world_size`. |

Patches [5], [6] and [8] are DCP-only and are skipped on non-DCP arms.

### 6.2 Layout constraints

```
TP × DP must equal 8 (the GPU count)
DP > 1 requires EP > 1, or the experts do not shard
tp_size % dcp_size == 0        -> at TP=4, legal DCP is 1, 2, 4
weights/GPU = non_expert/TP + experts/(EP ? ranks : TP)   must be < 288 GB
DCP patches require image 5a4c8d99; ac7509e2b lacks PR #51705 plumbing
```

---

## 7. Trial ledger

| # | Run | Configuration | Result |
|---|---|---|---|
| 138 | 33158510878 | fixed-len 8k, C1, k=6, ladder 56 | ITL 32.70, TPOT 8.70 — 10 requests, percentiles unusable |
| 139 | 33159693747 | k=8, ladder 72 | ITL 33.63, TPOT 8.42 |
| 140 | 33160495127 | k=8, ladder 9 | TPOT 8.37; capture 65s→26s, 1.46→0.83 GiB |
| 141 | — | (superseded) | |
| 142 | 33164549184 | 1000 prompts @ 8k | CANCELLED — wrong ISL for a DCP screen |
| 143 | 33165123199 | **122k**, 100 req, DCP=8, draft bf16 | TPOT 11.93 / p90 13.70 |
| 144 | 33166811023 | DCP=8, draft **fp8** | TPOT 11.97 / p90 13.71; KV pool +36.5% |
| 145 | 33168461313 | **DCP off**, draft fp8 | **TPOT 8.77 / p90 9.06** — DCP costs +36.5% |
| 146 | 33169840000 | DCP=1 + a2a + interleave | 8.79 / 9.07 — flags **inert** at size 1 |
| 147 | 33171360827 | **nightly 6f7df92a8e** + 1-line cg patch | **TPOT 7.57 / p90 7.91** — nightly worth −13.7% |
| 148 | 33183801155 | nightly + load-format auto | 7.57 / 7.93 — load-format **inert** |
| 149 | 33185690716 | (ladder cap 16) | CANCELLED to run mns=CONC first |
| 150 | 33185946417 | nightly, **mns=1 ladder 1..9** | 7.58 / 7.93; capture 44s→**7s**, KV +2.3% |
| 151 | 33187965065 | aigmkt, mns 8, ladder 1..16 | 7.93 / 8.20 — this is `sa.sh`'s C1 config |
| 152 | 33190157834 | nightly + #51705 (bad rebase) | FAIL — `MultiHeadLatentAttention.__init__() got an unexpected keyword argument 'enable_dcp_q_replicate'`; I wrongly discarded `kimi_k3/nvidia/mla.py` as "NVIDIA-only" |
| 153 | 33191059734 | nightly + #51705 (fixed rebase) | cancelled for the accuracy gate |
| 154 | 33191753746 | **GSM8K** C1 + C52, limit 200, nightly + #51705 | in flight |
| 1 | 32025696861 | patch [4], DCP8, bf16 | FAIL 0x1016 @134,400 |
| 2 | 32039650984 | [4], DCP4, bf16 | FAIL 0x1016 @262,656 |
| 3 | 32042030173 | PR#51705 only, DCP8 | FAIL 0x1016 @135,168 |
| 4 | 32043813560 | #51705 + **[6]** | **0x1016 fixed**, errors=0 |
| 5 | 32049134216 | + AIPERF_FAST | 742.4, TPOT 663 ms; 113/183 dropped |
| 6 | 32055440757 | GSM8K gate | timeout — 3.5–6.8 tok/s, ≈15 h needed |
| 7 | 32060082326 | + PIECEWISE cudagraph | 780.6, TPOT 0.687 |
| 8 | 32066978737 | DCP off, [1]–[3] off | FAIL — `AITER_DISABLE_FMHA_OPUS` only set in the DCP branch |
| 9 | 32068474469 | DCP off, [1]–[3] on | FAIL @25 min — RCCL watchdog |
| 10 | 32070778181 | DCP off, [1][2][3] | FAIL @21 min — same signature, [5] not the cause |
| 11 | 32073039787 | DCP off, [2] off | 1,014.9, TPOT 0.170 |
| 12 | 32077536567 | + full profile | 999.4, TPOT 0.1698 |
| 13 | 32084553677 | chunk 8192→32768 | lost to the workflow concurrency rule |
| 14 | 32089974051 | yukiozzz image, c64, MTP | FAIL — `mla_gluon requires batch_size=1, got 64` |
| 15 | 32090860356 | yukiozzz, c64 | FAIL — same, got 3 |
| 16 | 32093774227 | c64, chunk 32768 | cancelled @20 min |
| 17 | 32094907936 | DCP8 c20 DRAM | FAIL — leftover `MAX_NUM_BATCHED_TOKENS=32768` |
| 18 | 32096487055 | DCP8 c20 **DRAM offload** | **1,990.8**, TPOT 0.1742 — 781 → 1,991 (2.55×) |
| 19 | 32101946357 | conc 64 | 1,041.1, TTFT 501 s |
| 20 | 32110276088 | TRITON_MLA + MTP | cancelled @55 min |
| 21 | 32114847961 | TRITON_MLA, spec off | 1,948.4 |
| 22 | 32123047671 | DCP off, c1 + c20, MTP | c20 **3,340.5**, TPOT 0.0429 |
| 23 | 32138970163 | GSM8K gate | **passed** 0.9659 / 0.9644 vs 0.9651 baseline |
| 24 | 32143146154 | DCP=2 | FAIL @init — `is_valid_num_heads(24)` |
| 25 | 32143877066 | DCP=4 c20 | **2,033.7**, TPOT 0.1674 — best DCP pre-correction |
| 26 | 32154159649 | DCP=4 **c8** | 969.4 |
| 27 | 32162233477 | direct symm-mem | FAIL @9 min — op absent, CUDA-only |
| 28 | 32163743641 | **ag_rs** combine | 1,978.4 |
| 29 | 32172937044 | interleave 1→16 | 1,977.2 |
| 31a | 32269094879 | local image tag | FAIL @6 min — harness uses `--pull always` |
| 31b | 32270805303 | **ported direct a2a** | 2,014.7 — **−0.9%** |
| 35e | 32333672290 | DCP=8 profiling | 92.65% collectives |
| 37 | 32340149740 | non-DCP profiling | 56.05% collectives |
| 38–41 | — | per-rank straggler analysis | straggler moves 7→0→1→5 |
| 45/46 | 32368684074 | NUMA pinning, node + per-rank | no effect / worse |
| 47 | 32372517517 | non-DCP, full window | 2,655.7, TPOT 0.1176 |
| 48 | 32381243949 | DCP=8 + **PYNCCL** | 1,531.6, TPOT 0.2271 — **−23%** |
| 50 | 32390477829 | non-DCP + MTP, FULL cudagraphs | FAIL @init — MLA full-capture assert |
| 51 | 32392005995 | non-DCP + MTP, c20 | cancelled — KV-starved at 1.32× |
| 54 | 32396466979 | non-DCP + MTP, **c8** | 541.0, TPOT 0.2921 |
| 55 | 32404418920 | EP=8 attempt | FAIL — `spec-decoding` also selects the script filename |
| 57 | 32406159156 | **EP=8** verified | 2,619.2, TPOT 0.1177 — neutral |
| 58 | 32414712217 | image bisect on ac7509e2b | 2,685.0, TPOT 0.1161 |
| 59 | 32424123796 | DCP + MTP on ac7509e2b | FAIL — config gate rejects the pair |
| 60 | 32425822552 | DCP + MTP on 5a4c8d99 | FAIL — no ROCm backend declares the capability |
| **64** | **32436403856** | **TP8, AITER MLA, spec off** | **4,622.8, TPOT 0.0461** |
| 65 | 32444354043 | + MTP | 2,045.4, TPOT 0.1003 |
| 66c | 32451498395 | + DCP=8 | 1,574.5, TPOT 0.2174 |
| 67b | 32458502570 | **DP2/TP4/EP8** | 2,998.6 corrected (5,997.2 reported), TPOT 0.1140 |
| 68 | 32466225062 | DP2/TP4 + MTP | cancelled |

---

## 8. Investigation history

### 8.1 Levers tested against the DCP decode penalty — all negative

| Lever | Result |
|---|---|
| DRAM KV offload | 2.55× as a feature; no effect on the straggler |
| Concurrency (c8 / c20 / c64) | optimum at c20: 969 / 2,034 / 1,041 |
| World size 8 → 4 | +2%; DCP=2 illegal (24 heads) |
| Combine algorithm a2a vs ag_rs | 2,034 vs 1,978 |
| Shard granularity interleave 1 → 16 | 1,977 |
| Attention backend (within DCP) | within noise |
| CUDA graphs | no TPOT effect |
| Async scheduling on/off | no effect; straggler persists |
| NUMA pinning, node-level | no effect |
| NUMA pinning, per-rank slices | worse (188 → 218 ms, rising) |
| Hand-ported direct P2P collective | works, −0.9% |
| All-reduce implementation (PYNCCL) | −23% throughput, +31% TPOT |
| MTP under DCP | refused at engine init |

### 8.2 Hypotheses the profiling eliminated

| Hypothesis | Evidence against |
|---|---|
| KV offload causes the straggler | Removing offload entirely; straggler persisted |
| A specific bad GPU or link | Straggler identity moves: rank 7 → 0 → 1 → 5 |
| GPU load imbalance | Compute per rank 1.05–1.11 s, ≤5% spread |
| DCP shards decode work | 11.5 vs 11.6 ms/step — no reduction |
| Async scheduling | Enabling it left the straggler intact (rank 5, 89% idle) |
| The collective implementation | PYNCCL substitution measurably worse |
| NUMA placement | No effect at node level; worse per-rank |
| A clean 1-vs-7 split | Intermediate ranks appear (265/272 µs) — a gradient |

### 8.3 Corrections made during the investigation

| Earlier claim | Correction |
|---|---|
| "The reference decodes on TRITON_MLA" | It sets no target backend; TRITON_MLA is the **draft's**. Forcing it cost 2.4× and was present in every non-DCP trial. |
| "The reference runs DCP=8 + MTP" | It runs **no DCP**. That configuration came from the B300 (NVIDIA) script — a different platform and file. |
| "MTP starves this workload" | The KV shortfall came from a pinned `--max-num-batched-tokens`. With it omitted the pool matched the reference exactly. MTP still loses, for a different reason. |
| "There is a 2.7× decode regression" | An artefact of comparing spec-on TPOT (per accepted token) against spec-off: `0.1161 / 2.509 = 0.0463` vs 0.0429. |
| "DCP's ceiling is ~2,000 tok/s/GPU" | Measured against a handicapped control. Re-baselined, DCP is −66%, not −24%. |
| "DP attention is infeasible on 8 GPUs" | Only the DP8/TP1 extreme fails. DP2/TP4 fits with ~50 GB spare and runs. |
| "The DCP decode cost is irreducible" | The cost is real but mostly **not DCP's** — it is the TP all-reduce any TP=8 decode pays, which DCP worsens by adding sync points. |

### 8.4 Script defect class — five instances

A value that appears set at one site is decided or discarded at another.

| Defect | Effect |
|---|---|
| Shadowed `DCP_SIZE` | An assignment inside the auto-concurrency block won; runs used DCP=4 / 17× KV when 8 / 32× was intended. |
| `DISABLE_SPEC` in a concurrency-gated block labelled DCP | Non-DCP arms silently ran without speculation while labelled `spec-mtp`. |
| Dead `CUDAGRAPH_CAPTURE_SIZES` | A sparse ladder overwritten by a later assignment in the same branch. |
| **Orphaned `EP_ARGS`** | Computed but never referenced in `VLLM_CMD` — **expert parallelism was unreachable for 55 trials**. |
| Split `DCP_SIZE` default | The patch gate read `1` while the run set `8`, so DCP ran without its patches. |

Structural remedies applied: `DCP_SIZE` resolves once before any consumer;
layout guards compute from measured weights rather than remembered rules; an
orphan check parses `VLLM_CMD` and diffs it against defined `*_ARGS` arrays.

### 8.5 Other notable failures

- A locally built image tag cannot resolve — the harness uses `docker pull --pull always`.
- `spec-decoding` in the matrix also selects the **script filename**; setting it to `none` pointed the harness at a non-existent file.
- `VLLM_TORCH_PROFILER_DIR` is obsolete; the knob is `--profiler-config`.
- A `-p2` patch dry-run reported success where a correct `-p1` showed only one nightly applies (23/24 hunks).
- Patch [8]'s guard matched *uses* of `_decode_num_heads` rather than its assignment, reporting "already patched" against a partial application.
- Acceptance length was misread from `sort -u | head` (the bottom five values) rather than the distribution; true mean 2.706 under `standard` rejection.
- Kimi-K3 fp8 KV decode requires a backend accepting an fp8 query input, so the draft-causal rewrite is a consequence of the fp8 KV choice, not an independent option.

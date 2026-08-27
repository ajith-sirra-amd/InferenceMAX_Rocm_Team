# Kimi-K3 — where the time actually goes

Companion to [Kimi-DCP-Experiemnts-Summary.md](Kimi-DCP-Experiemnts-Summary.md).

Measured on 8× MI355X, DCP=8, concurrency 52. Current data is
[T124](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/33033974874)
(rocprofv3, **no KV offload**); [T116](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/32964875218)
(offload ON) is kept below for the before/after. Raw profiler output is committed
verbatim under [kimi-k3-profiles/](kimi-k3-profiles/).

---

## MEASURED — rocprofv3, T124 (current, offload OFF)

Source:
[T124](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/33033974874),
`rocprofv3 --kernel-trace --stats`, 3.1 GB trace, **8,474,873 dispatches**,
DCP=8, conc 52, `max_num_seqs` 65, **no KV offload**, no MTP.
Same method as T116 below: negative/>1 s durations discarded, busy is the
**union of merged intervals**, dead periods >2 s excluded. Serving window 1430.7 s.

**T124 supersedes T116.** T116 profiled the same point *with* the DRAM offload,
which we have since removed on its own evidence.

### Removing the KV offload cut idle by 16 points

| | T116 offload ON | **T124 offload OFF** |
|---|---:|---:|
| GPU busy | 55.7% | **71.8%** |
| **Idle** | **44.3%** | **28.2%** |
| collectives (% busy) | 34.31% | 29.68% |
| dispatches | 6.36M | 8.47M |
| tok/s/GPU *(traced both sides, not quotable)* | 3,146.0 | **6,821.5** |
| requests | 375/524 | **1123/1232** |

| gap size | T116 | T124 | change |
|---|---:|---:|---|
| 10–200 µs | 115.2 s (7.9%) | 124.0 s (8.7%) | unchanged — pure launch overhead |
| 0.2–1 ms | 147.5 s (10.1%) | 104.8 s (7.3%) | −29% |
| 1–10 ms | 118.7 s (8.1%) | 57.6 s (4.0%) | −51% |
| **>10 ms** | **265.7 s, n=4,104** | **114.6 s, n=877** | **−57%, 4.7× fewer** |

The multi-millisecond stalls were the offload's host↔device traffic. The
sub-200 µs launch gaps did not move — the signature of dispatch overhead rather
than memory traffic. Predicted before the run; it held.

### Prefill vs decode — two different problems

Method caveat first: chunked prefill **interleaves both in one forward pass**, so
this classifies 1 ms windows by whether prefill attention (`fmha`) was running.
A mixed pass runs `fmha` in some milliseconds and its MoE/GEMM/collectives in
others, so prefill's non-attention work lands in the decode bucket. The tell is
MoE showing 0.1% in prefill windows. **Prefill's true share is higher than 33%
and decode's collective share is inflated.** Treat the split as indicative.

| category | prefill-active | | decode-only | |
|---|---:|---:|---:|---:|
| **collectives** | 34.0 s | 9.0% | **304.0 s** | **39.9%** |
| **MLA attn (fmha)** | **268.7 s** | **71.3%** | 0 | — |
| dense GEMM | 30.6 s | 8.1% | 158.3 s | 20.8% |
| MoE | 0.3 s | 0.1% | 154.8 s | 20.3% |
| KDA | 0 | — | 62.1 s | 8.2% |
| attn residual | 3.1 s | 0.8% | 39.7 s | 5.2% |
| MLA attn (decode/DCP) | 22.6 s | 6.0% | 6.4 s | 0.8% |
| **total** | **377.1 s** | **33.1%** | **761.7 s** | **66.9%** |

**Prefill is attention-bound (71%); decode is collective-bound (40%).**

What survives the caveat cleanly: `fmha` = **268.7 s = 23.6% of all busy,
exclusively prefill**; `gather`+`merge` = **29.1 s = 2.6%, exclusively decode**.
Everything else is genuinely shared and cannot be split by kernel name.

Workload context: **152–175 input tokens per output token**, but prefix cache hit
is ~94%, so only ~6% of input tokens are actually computed. That is how a 175:1
token ratio reconciles with prefill being only ~1/3 of GPU time.

### The idle is almost entirely in decode

| | idle | share |
|---|---:|---:|
| prefill-active windows | 29.8 s | **7.4%** |
| **decode-only windows** | **374.1 s** | **92.6%** |

Prefill keeps the GPU nearly saturated — it submits 8192-token chunks that occupy
it for tens of ms. Decode submits ~52-row batches across 93 layers where every
kernel is short and the host cannot stay ahead.

**What the GPU is waiting for, in decode:**

| idle | share | n | avg gap | next kernel |
|---:|---:|---:|---:|---|
| 121.2 s | 32.4% | 592,738 | 205 µs | elementwise/other |
| **102.8 s** | **27.5%** | 537,111 | 191 µs | **copy/fill** |
| 34.8 s | 9.3% | 325,151 | 107 µs | KDA |
| **29.0 s** | **7.8%** | 20,717 | **1,400 µs** | **MLA attn decode (DCP)** |
| 26.3 s | 7.0% | 294,448 | 89 µs | MoE |
| 24.0 s | 6.4% | 228,603 | 105 µs | dense GEMM |

Two distinct causes:

- **~60% of decode idle is 100–200 µs gaps before small ops** across ~1.75M
  events. The host is not issuing fast enough — a **launch-rate** problem, not a
  memory one.
- **`MLA attn decode (DCP)` is the outlier**: only 20,717 gaps but **1,400 µs
  each**. That is the DCP partial-attention path waiting on a cross-rank sync —
  structural to DCP, not host overhead.

### What the CPU is actually doing

Naming the kernel the GPU was **waiting to start** identifies which host stage
stalled. From T124's 403.9 s of intra-serving idle:

| idle | % idle | n | avg gap | waiting to start | host work |
|---:|---:|---:|---:|---|---|
| **71.9 s** | **17.8%** | 512,608 | 140 µs | `__amd_rocclr_copyBuffer` | **H2D upload of batch metadata** — `input_ids`, `positions`, `slot_mapping`, `block_tables`, `seq_lens`, `query_start_loc`. **~127 copies per step** |
| 28.3 s | 7.0% | **6,050** | **4,683 µs** | `__amd_rocclr_fillBufferAligned` | `hipMemset` — allocator zeroing |
| 28.0 s | 6.9% | 22,651 | **1,237 µs** | `merge_attn_states_kernel` | **not CPU** — DCP cross-rank sync |
| 26.2 s | 6.5% | 569,384 | 46 µs | `ncclDevKernel_Generic_1` | **not CPU** — collective rank skew |
| **25.2 s** | **6.2%** | **79** | **319,062 µs** | `arange_cuda` | 79 events at 319 ms each — **unexplained** |
| 22.1 s | 5.5% | 164,868 | 134 µs | `elementwise_manual_unroll` | sampling / penalties / metadata math |
| 15.6 s | 3.9% | 91,331 | 171 µs | `elementwise_manual_unroll` | as above |
| 12.0 s | 3.0% | 101,467 | 118 µs | `CatArrayBatchedCopy` | `torch.cat` in attention-metadata assembly |

**Rough attribution of the 403.9 s:**

| | | |
|---|---:|---|
| host / Python | **~150 s (37%)** | batch-tensor build + upload (84 s), sampling/elementwise (38 s), allocator memsets (28 s) |
| **not** Python | ~54 s (13%) | DCP merge sync (28 s), collective rank skew (26 s) |
| unexplained | ~25 s (6%) | `arange_cuda`, 79 events |
| remainder | | per-dispatch overhead, partly rocprof's own |

The per-step host sequence is: scheduler decisions (which requests run, KV block
alloc/free, prefix-cache block hashing) → `_prepare_inputs()` building ~127 CPU
tensors and uploading them → attention-metadata construction (DCP adds per-rank
`seq_lens`/`cu_seqlens` work) → sampling → incremental detokenization and IPC to
the API server. **The ~127 `copyBuffer`/step is the fingerprint of that second
stage**, and it is the single largest host item.

### The tokenizer is slow — but it is not this

Kimi-K3 ships **no `tokenizer.json`**, so there is no HF Rust
`PreTrainedTokenizerFast` path:

    tokenizer_class : TikTokenTokenizer   (auto_map -> tokenization_kimi.py)
    tokenizer.json  : ABSENT
    vLLM log        : "slow tokenizer" x10,  tokenizer_mode=kimi

**This does not explain the measured idle.** The stalls above are tensor uploads
in EngineCore; tokenization is once per *request* and detokenization is
`IncrementalDetokenizer` inside `OutputProcessor` — both live in the **API server
process**, not the GPU step loop. Accelerating the tokenizer would not move the
`copyBuffer` gaps.

Where it *could* matter is TTFT: at ~66,000 input tok/s every prompt is tokenized
in full, **including the ~94% that hit the prefix cache and are never computed**.
So tokenizer cost scales with raw input, not with the 6% actually prefilled.

Levers, if that is ever shown to bind:

- **`--api-server-count N`** — scales the front-end processes doing
  tokenize/detokenize/HTTP. The natural fix, since the work is there and not in
  EngineCore.
- `--skip-tokenizer-init` — only if the client sends token IDs; our replay sends
  text, so not applicable without changing the harness.
- A real `tokenizer.json` for Kimi-K3 would unlock the Rust fast path, but that is
  a checkpoint asset we do not control.

**Untested.** No measurement yet shows tokenization binding — it is a hypothesis
about TTFT, not a finding, and it is listed here so it is not mistaken for one.

### BF16 vs FP8, weighted into end-to-end wall

GEMM only — **27.41% of GPU busy (312.1 s)**:

| | time share | compute share | sec |
|---|---:|---:|---:|
| **BF16 GEMM** | **60.5%** | 42.9% | 189.0 |
| **FP8 × MXFP4** | **39.4%** | 57.1% | 123.1 |

Efficiency **2.50 vs 5.12** G MACs per 1% of busy = **2.04×**, essentially
unchanged from T116's 2.02×. Removing the offload changed the idle, not kernel
efficiency. All MoE runs `afp8_wfp4`; the `a16wfp4` BF16-activation variant is
0.01%, i.e. absent.

**Weighted into overall e2e wall** (1431 s serving, 71.8% busy):

| category | % busy | **% e2e wall** |
|---|---:|---:|
| **IDLE** | — | **28.23%** |
| collectives (BF16 payload) | 29.68% | **21.30%** |
| MLA attn prefill (BF16 QK/PV) | 23.60% | **16.94%** |
| **BF16 dense GEMM** | 16.59% | **11.91%** |
| FP8 × MXFP4 MoE GEMM | 10.81% | 7.76% |
| elementwise/copy/other | 7.55% | 5.42% |
| KDA (BF16 io) | 5.46% | 3.92% |
| attn residual (BF16) | 3.76% | 2.70% |
| MLA attn decode (FP8 KV→BF16) | 2.55% | 1.83% |

**BF16 dense GEMM is 11.91% of wall.** Converting all of it to FP8 at the
observed 2.04× saves ~6.1% of wall ≈ **+6.5% throughput** — up from +4.4% when
idle was 44%, because there is less idle to dilute it.

### FP16 is not a faster GEMM path — measured

Prompted by QuickReduce's note that "bfloat16 kernels are slower than fp16" on
ROCm. That comment is about QuickReduce's **codec** (bit-level pack/unpack for
INT4/INT8 quantization), not MFMA. Benchmarked directly on the real TP8-sharded
shapes, one MI355X, hipBLASLt via torch:

| shape | M | N | K | BF16 TFLOP/s | FP16 TFLOP/s | FP16/BF16 |
|---|---:|---:|---:|---:|---:|---:|
| KDA q/k/v decode | 64 | 1536 | 7168 | 86.5 | 87.7 | 1.014× |
| **KDA q/k/v prefill** | 8192 | 1536 | 7168 | **1212.5** | 1095.0 | **0.903×** |
| KDA o_proj decode | 64 | 7168 | 1536 | 112.4 | 111.2 | 0.990× |
| KDA o_proj prefill | 8192 | 7168 | 1536 | 1020.9 | 1023.4 | 1.003× |
| MLA q_b decode | 64 | 2304 | 1536 | 36.9 | 36.4 | 0.989× |
| MLA q_b prefill | 8192 | 2304 | 1536 | 891.4 | 854.6 | 0.959× |
| MLA kv_b prefill | 8192 | 3072 | 512 | 753.0 | 707.2 | 0.939× |
| **shared_expert prefill** | 8192 | 768 | 7168 | **1023.4** | 906.8 | **0.886×** |

**FP16 loses on 6 of 8 shapes, by up to 11.4%.** There is no hidden hipBLASLt
tuning advantage. Casting GEMM to FP16 would cost throughput *and* add FP16's
range risk (max 65,504 vs BF16's ~3.4e38, which is why BF16 exists). Closed.

This also corroborates the profile from a second direction: BF16 GEMM measured
2.50 G MACs per 1% of busy against FP8×MXFP4's 5.12, a 2.04× ratio that matches
the hardware's BF16→FP8 ratio exactly. BF16 is already running at full
matrix-core rate.

### Decode GEMMs run at 7% of prefill efficiency

The more important result from the same benchmark:

| | BF16 TFLOP/s |
|---|---:|
| prefill, M=8192 | **1,212** |
| decode, M=64 | **86.5** |

**14× worse, purely because M=64.** At that shape the matrix cores idle waiting
on memory — it is not a precision problem. Decode cannot be fixed by making the
math cheaper, because the math is not the cost. That agrees with the profile
reaching the same conclusion from the other side (decode is collective- and
launch-bound).

It also sizes the BF16→FP8 lever honestly: FP8 gives ~2× on **prefill** GEMMs,
which already run at 1,212 TFLOP/s. On decode GEMMs at 86.5 TFLOP/s it will buy
far less, because those are latency-bound rather than compute-bound.

### Collectives: the DCP group never gets the fast all-reduce

    group 'tp:0'  -> ['AITER_CUSTOM', 'PYNCCL']
    group 'dcp:0' -> ['PYNCCL']
    group 'ep:0'  -> ['PYNCCL']

Generic `ncclDevKernel_Generic_1` is **22.55%** of GPU busy against **3.63%** for
`cross_device_reduce_2stage`, the tuned AITER path TP uses. The cost sits in the
group that did not get the fast backend — a concrete target, not just a fact.

### Priority order, by e2e wall

| target | % e2e wall | handle |
|---|---:|---|
| **Idle (92.6% of it in decode)** | **28.2%** | host launch rate in decode |
| **Collectives** | **21.3%** | `dcp:0` on generic PYNCCL |
| MLA attn prefill | 16.9% | needs an FP8 FMHA kernel |
| BF16 dense GEMM | 11.9% | weight quantization, ~+6.5% |

---

## Superseded — rocprofv3, T116 (offload ON)

Everything below this section is superseded by real data. Source:
[T116](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/32964875218),
`rocprofv3 --kernel-trace --stats`, 2.3 GB trace, **6,356,992 kernel dispatches**
on one GPU (Agent 2), DCP=8, conc 52, dense ladder 1…80.

Method, so the numbers can be checked: 31 dispatches with negative or >1 s
durations were discarded (rocprof mis-times the spin-wait custom all-reduce; its
`k_kernel_stats.csv` is unusable — one row reports 2⁶⁴ ns and a bogus
`Percentage=100.00`, which is why these figures come from the raw trace).
Busy time is the **union of merged intervals**, not a naive sum. 32 dead periods
longer than 2 s (largest **207 s**, benchmark phase boundaries) are excluded to
give a serving window of 1467 s.

### The headline is idle, not any kernel

| | |
|---|---:|
| Serving window | 1467.1 s |
| GPU busy (union) | 820.5 s |
| **Occupancy** | **55.9%** |
| **Idle** | **44.1%** |

Sum-of-kernels (820.5 s) ≈ union-busy, so there is **essentially no kernel
overlap** — it is one serialized stream. The idle is **diffuse, not one stall**:
2.7M separate gaps — 0.2–1 ms 10.1%, 1–10 ms 8.1%, >10 ms 18.1%, 50–200 µs 5.8%.
Tracing overhead cannot explain it (3.5k dispatches/s × ~1–2 µs ≈ 0.5% of wall).

**No kernel optimisation can address more than 55.9% of the clock.**

### Where the 55.9% goes

| Group | % busy | % wall | Precision |
|---|---:|---:|---|
| **Collectives** | **34.31** | 19.19 | BF16 payload (nccl 732k calls @336 µs, `cross_device_reduce`, msccl) |
| **MLA attention** | **21.79** | 12.19 | BF16 QK/PV, FP32 softmax (`fmha_fwd_hd192_hd128_bf16`, 65k @ **2.65 ms**) |
| **Dense GEMM** | **15.26** | 8.53 | BF16→BF16 14.22, BF16→FP32 1.03 (hipBLASLt `Cijk_*`) |
| MoE GEMM | 12.49 | 6.99 | **act FP8 × wt MXFP4** 9.88, BF16 reduce 1.69, FP32 topk 0.68 |
| KDA linear attn | 5.16 | 2.88 | BF16 io, FP32 accum (Triton) |
| Attn residual | 3.82 | 2.14 | BF16 io (`_attn_res_kernel`) |
| Elementwise / copy | 3.70 | 2.07 | incl. 0.84 `copyBuffer` = DRAM KV offload |
| KV cache | 2.42 | 1.35 | **FP8 KV → BF16** dequant |
| Quant / norm | 0.88 | 0.49 | BF16 → FP8/FP4 group-32 |

### Reading BF16 correctly

~74% of GPU time *touches* BF16, but only **15.26%** is a BF16 **weight matrix in
a linear layer** — the only thing a checkpoint's weight precision can change. The
rest is BF16 being **shipped over the network** (34.31%), **fed through attention
softmax** (21.79%), or **added/normalised elementwise** (17.67%).

| Role of the BF16 | % busy | Fix | Weight quant helps? |
|---|---:|---|:--:|
| Network payload | 34.31 | FP8 all-reduce | ✗ |
| Attention math | 21.79 | an FP8 FMHA kernel | ✗ |
| Linear-layer weights | **15.26** | **weight quantization** | **✓** |
| State / elementwise | 17.67 | nothing — memory-bound | ✗ |
| Already FP8×MXFP4 | 10.05 | already done | n/a |
| FP32 (topk/sort) | 0.75 | negligible | ✗ |

### Pure GEMM only — BF16 vs FP8, time share vs compute share

Restricting to **GEMM kernels alone** (hipBLASLt `Cijk_*` + AITER `mfma_moe1/2`),
which are **24.68% of GPU busy** — 202.5 s of 820.5 s. Compute is MACs/token from
the checkpoint shapes; time is measured.

| | **Time share** | **Compute share** | sec | G MACs/token |
|---|---:|---:|---:|---:|
| **BF16 GEMM** (KDA proj + MLA proj + latent/gate) | **60.3%** | **42.9%** | 122.1 | 41.56 |
| **FP8 × MXFP4 GEMM** (MoE experts) | **39.7%** | **57.1%** | 80.4 | 55.29 |

**The shares are inverted.** BF16 does **43% of the math but consumes 60% of the
time**; FP8×MXFP4 does **57% of the math in 40% of the time**.

**Efficiency — G MACs per 1% of GPU-busy time:**

| | |
|---|---:|
| BF16 | 2.79 |
| FP8 × MXFP4 | **5.64** |
| **Ratio** | **2.02×** |

That 2.02× is essentially the hardware's BF16→FP8 ratio, which means **neither
path is underperforming**: the AITER `afp8_wfp4` kernels and the hipBLASLt BF16
kernels are both running about as well as the silicon allows. The difference is
precision, not a tuning defect.

**Ceiling if BF16 GEMM ran at the FP8 path's efficiency:**
14.88% → 7.37% of busy, saving 61.6 s = **4.20% of the 1467 s wall ≈ +4.4%
throughput.** Derived independently of the earlier estimate and agreeing with it.

**As a share of total GPU busy:** BF16 GEMM **14.88%**, FP8 GEMM **9.80%**,
non-GEMM **75.32%**. With 44.1% idle on top, GEMM precision is a ~4% lever on a
problem that needs 1.57×. This closes the GEMM line of inquiry.

### Three earlier claims this retires

1. **"Attention is 5.6%"** — it is **21.79%**. The MAC model treated attention as
   compute-bound; it is memory/softmax-bound.
2. **"~94% dense BF16 GEMMs"** — **15.26%**.
3. **"Dense→FP8 buys ~1.2×"** — halving 15.26% of a 55.9%-busy GPU saves ~4.3% of
   wall, so **+3–5%**, not +20%. Confirmed by T117–T119 (see summary): the AMD
   AttnFP8 checkpoint converts exactly this block.

### What to attack, in order

1. **44.1% idle** — host-bound. Largest single lever, bigger than all kernels.
2. **34.31% collectives** — TP=8 all-reduce + DCP=8 all-to-all, ~270 nccl calls
   per forward pass. FP8 payload would be the direct attack.
3. **21.79% MLA attention** — needs an FP8 FMHA kernel; KV cache is *already* FP8
   but `fmha_fwd_..._bf16` dequantises to BF16 for the math.

---

## Superseded — the first-principles budget (kept for history)

An earlier version of this document claimed **"~94% of time is BF16 dense
GEMMs."** That was derived by subtraction (attention measured at 5.6%, therefore
"everything else is GEMM") and it is **wrong**. It ignored the 69 KDA layers and
TP collectives entirely.

A first-principles budget built from the actual checkpoint shapes gives a very
different split — and then fails its own sanity check. The measured section above
now settles it: the budget's 49.1% collectives / 35.0% dense split was directionally
right on collectives and **2.3× too high on dense**.

### Per-token MACs (whole model, from safetensors shapes)

| Component | Weights | Compute | MACs/token | Share of GEMM |
|---|---|---|---:|---:|
| MoE experts (93L, top-16 + 2 shared) | **MXFP4** | **a8w4** | 55.29 G | **57.1%** |
| KDA projections (69L) | BF16 | **BF16** | 30.61 G | **31.6%** |
| MLA projections (24L) | BF16 | **BF16** | 5.57 G | 5.8% |
| latent + gate (93L) | BF16 | **BF16** | 5.38 G | 5.6% |
| **Total** | | | **96.85 G** | |

Only the experts are quantised. Everything else is BF16 **in the checkpoint** —
not a config choice, and unaffected by `--kv-cache-dtype fp8`, which governs KV
storage only.

**Why KDA is the largest BF16 term:** MLA uses low-rank compression (q → 1536,
kv → 576 latent), while KDA uses four *full-rank* `[12288 × 7168]` projections
(12288 = 96 heads × 128) plus `o_proj`. So a KDA layer is **1.9×** an MLA layer,
and there are **2.9×** as many → **5.5×** the total.

### Converted to time (per token, per GPU, TP8)

| Component | Precision | ns/token | Share |
|---|---|---:|---:|
| **TP collectives** (theory) | BF16 payload | 11,666 | **49.1%** |
| Dense GEMM (KDA + MLA + latent) | **BF16** | 8,311 | 35.0% |
| MoE GEMM | **a8w4** | 2,765 | 11.6% |
| MLA attention (measured) | **BF16** | 858 | 3.6% |
| KDA state update (upper bound) | BF16 | 177 | 0.7% |

MoE carries **57% of the MACs but only 11.6% of the time** — a8w4 runs ~4×
BF16's FLOP rate. Conversely the BF16 dense paths hold ~43% of MACs and ~35% of
time. That ratio is the whole argument for quantising dense weights.

### If dense BF16 became FP8, what would it buy?

Upper bound, using the theory table: dense BF16 is **35.0%** of device time. FP8
runs ~2× BF16 on this hardware, so halving it gives:

| | Share | Best case |
|---|---:|---:|
| Dense BF16 → FP8 | 35.0% → 17.5% | **~1.21× overall** |
| Same, if collectives are overstated and dense is really ~60% | 60% → 30% | ~1.43× |

So **~1.2×, maybe ~1.4× if the collectives estimate is too pessimistic** — real,
but short of the 1.57× needed for 12,500 on its own. And that is a ceiling: it
assumes perfect 2× on every dense GEMM with no quantise/dequantise overhead.

Caveats that matter before anyone starts:

- **It changes numerics.** Dense layers are BF16 in the checkpoint because
  that is where accuracy is most sensitive. Needs GSM8K (98.5% today) to validate.
- **It is a checkpoint/quantisation change**, not a flag.
- **The 35% is unverified** — the budget it comes from overpredicts throughput
  by 5.2×. If collectives dominate as the theory suggests, dense quantisation
  buys proportionally less.
- **KDA is where it would pay**, not "dense" generically — 30.61 G of the
  41.56 G dense MACs. Quantising only MLA would be near-pointless.

### This model does not reconcile with reality

It implies **42.1 k tok/s/GPU**; we measure **~8.1 k**. A **5.2× gap**.

So something large is unaccounted for. Candidates, in rough order of suspicion:

1. **Host-side per-request cost** — separately measured at ~1.5 ms/request,
   ~82% of TPOT at n=54. The budget above models only device work.
2. **Collectives estimate is crude** — assumes ~400 GB/s effective and perfect
   ring all-reduce. Real latency at these small message sizes (7168×2 B) is
   likely far worse, which would *increase* their share.
3. **In-situ GEMM efficiency below microbenchmark** — the 1250 TFLOP/s figure
   comes from isolated back-to-back calls, not interleaved with everything else.

**What survives from all this:**

- Attention is small — measured 5.6% empirically, 3.6% by theory. Both agree it
  is not the target. **fp8 attention remains not worth it.**
- **KDA projections are 31.6% of GEMM MACs and were entirely missing from the
  earlier analysis.** 69 of 93 layers; ignoring them was the main error.
- MoE dominates raw MACs (57.1%) but is a8w4, so its *time* share is ~12%.
- Dense BF16 GEMM is significant (~35%) but nowhere near the claimed 94%.
- **Collectives may be the largest single device-side term** and have never been
  measured here.

**A real profiler run is required** before optimising further. Everything in
this section is arithmetic on shapes, not observation.

## BF16 dense GEMM profile

**Scope, which matters:** every row here is **BF16 → BF16**. All 111,064 logged
dispatches in the run are `dtype='torch.bfloat16' otype='torch.bfloat16'` —
there is no dtype variation between rows. **MoE GEMMs do not appear at all**
(0 matches): they run the Situv2 / a8w4 path, which does not emit these
messages. So the shares below are shares *of BF16 dense GEMM time*, **not of
total time**.

Shapes and frequencies from the real run; timings measured at M=7729 (observed
prefill chunk). Shapes are post-TP8 sharding.

| N | K | Precision | Dispatches | ms | TFLOP/s | Share of BF16 GEMM | Likely source |
|---:|---:|---|---:|---:|---:|---:|---|
| 6288 | 7168 | BF16 | 14,456 | 0.577 | 1208 | **33.0%** | KDA (fused q/k/v/g) |
| 8448 | 7168 | BF16 | 8,800 | 0.675 | 1387 | **23.5%** | unmapped |
| 3584 | 7168 | BF16 | 14,456 | 0.330 | 1202 | **18.9%** | `routed_expert_down_proj` |
| 7168 | 4224 | BF16 | 8,800 | 0.374 | 1250 | 13.1% | dense MLP (33792/8) |
| 7168 | 1536 | BF16 | 8,800 | 0.151 | 1127 | 5.3% | `o_proj` (12288/8) |
| 7168 | 768 | BF16 | 8,800 | 0.093 | 912 | 3.3% | unmapped |
| 2304 | 1536 | BF16 | 8,800 | 0.063 | 871 | 2.2% | MLA `q_b_proj` (18432/8) |
| 1536 | 128 | BF16 | 14,456 | 0.014 | 222 | 0.8% | KDA `f_b_proj` (12288/8) |

**~1200–1400 TFLOP/s against ~2600 BF16 peak — about 50%.**

Source mappings are inferred from checkpoint shapes divided by TP=8; two are
unresolved and marked as such rather than guessed.

### The "not found tuned config" messages are not a slow path

111,064 of them per run, and they look alarming. They are not:

```python
if not default_config:
    default_config["libtype"] = "torch"     # aiter/tuned_gemm.py
```

Untuned shapes fall back to **torch/hipBLASLt**, which is what the table above
measures. The warning is informational.

---

## Attention is not the lever

MLA prefill attention at K3's real dims (12 heads/rank, qk=192, v=128):

| chunk | BF16 flash-attn | fp8 cast alone |
|---:|---:|---:|
| 2048 | 0.132 ms | 0.016 ms |
| 4096 | 0.105 ms | 0.025 ms |
| 8192 | **0.293 ms** | 0.037 ms |

24 MLA layers × 0.293 ms = **7.0 ms** against ~126 ms per 8192-token chunk =
**5.6%**. A perfect fp8 attention kernel — 2× on that slice, minus a cast
costing 12–24% of the saving — is worth **~2.5–2.8% end-to-end**.

### FP8 prefill is also unreachable on ROCm

Investigated and closed:

- The kernels exist and report supported: `aiter.mla_prefill_ps_asm_fwd`,
  `aiter.mla_reduce_v1`, `_fp8_mla_prefill_supported() == True`.
- vLLM wires them into `rocm_aiter_mla.forward_mha` — but **Kimi-K3 never calls
  it**. Its only backend entry points are `forward_mqa` and
  `forward_mqa_with_dcp_verify_window`; prefill goes through its own
  `_forward_prefill_fused`. Source comment: *"there is no dense-MHA
  (forward_mha) fallback."*
- The backend K3 does use, `aiter_flash_attn.py`, has **zero** fp8 references,
  and `aiter.flash_attn_varlen_func` exposes no fp8 scale parameters.
- The only fp8-capable MLA prefill backend, `tokenspeed_mla`, is absent from
  `platforms/rocm.py` and its package is not installed.

So **vLLM PR #51040 is inert for Kimi-K3** — it patches a function this model
never calls.

---

## What could still close the gap to 12,500 (1.57× needed)

**Ranking deferred until a real profile exists.** The theoretical budget and the
earlier subtraction-based one disagree sharply, and the budget misses reality by
5.2×, so any ranking now would be guesswork dressed as analysis.

Candidates, with what we actually know about each:

| Lever | Attacks | Evidence | Status |
|---|---|---|---|
| **Real profiler run** | everything | — | **Do this first** |
| Host per-request cost | ~82% of TPOT | measured: 1.5 ms/request | Strong, upstream vLLM work |
| TP collectives | maybe ~49% device time | theory only, never measured | Unknown — profile it |
| Tune aiter GEMM kernels | ~35% (dense) | measured ~50% of BF16 peak | Zero numerics risk |
| Quantise dense weights to fp8 | ~35% (dense) | up to ~2× on that slice | Changes numerics; needs GSM8K |
| fp8 attention | 3.6–5.6% | measured both ways | **Closed — not worth it** |

Everything reachable by configuration is already measured and closed:
concurrency peaks at 52, chunk 16384 costs 2.5%, async scheduling costs 9.2%,
MTP costs 85%, extra cache tiers do nothing (`ext_cache_hit` ≈ 0, pool 23.5%
used).

### What the profile run needs to capture

- Per-kernel device time (torch profiler or rocprof), not sampled throughput
- **Collective time separately** — the largest unknown
- KDA vs MLA layer split — 69 vs 24 layers, KDA never measured
- Host gaps, to confirm or refute the 1.5 ms/request term on the timeline

---

## Method note

The GEMM timings use `a@b` (torch/hipBLASLt), which is the path aiter actually
falls back to for these shapes, so they represent production behaviour. Shares
are frequency-weighted from the run log rather than assumed. The attention
benchmark initially used the 576-wide *latent* head dim and failed with "CK only
supports head dimension at most 256"; the numbers above use the correct
post-decompression dims (qk=192, v=128).

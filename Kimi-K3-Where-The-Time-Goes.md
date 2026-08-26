# Kimi-K3 — where the time actually goes

Companion to [Kimi-DCP-Experiemnts-Summary.md](Kimi-DCP-Experiemnts-Summary.md).

Measured on 8× MI355X, DCP=8, concurrency 52, from run
[T103](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/32855763638)
(7,950.6 tok/s/GPU) plus kernel microbenchmarks on the same host.

---

## Headline — and a correction

An earlier version of this document claimed **"~94% of time is BF16 dense
GEMMs."** That was derived by subtraction (attention measured at 5.6%, therefore
"everything else is GEMM") and it is **wrong**. It ignored the 69 KDA layers and
TP collectives entirely.

A first-principles budget built from the actual checkpoint shapes gives a very
different split — and then fails its own sanity check, so treat both as
provisional until a real profiler run exists.

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

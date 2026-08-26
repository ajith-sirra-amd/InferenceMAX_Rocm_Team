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

| Component | MACs/token | Share of GEMM |
|---|---:|---:|
| MoE experts (93L, top-16 + 2 shared) | 55.29 G | **57.1%** |
| KDA projections (69L) | 30.61 G | **31.6%** |
| MLA projections (24L) | 5.57 G | 5.8% |
| latent + gate (93L) | 5.38 G | 5.6% |
| **Total** | **96.85 G** | |

### Converted to time (per token, per GPU, TP8)

| Component | ns/token | Share |
|---|---:|---:|
| **TP collectives** (theory) | 11,666 | **49.1%** |
| Dense GEMM, BF16 | 8,311 | 35.0% |
| MoE GEMM, a8w4 | 2,765 | 11.6% |
| MLA attention (measured) | 858 | 3.6% |
| KDA state update (upper bound) | 177 | 0.7% |

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

## GEMM profile

Shapes and frequencies taken from the real run; timings measured at M=7729
(observed prefill chunk).

| N | K | Dispatches | ms | TFLOP/s | Share |
|---:|---:|---:|---:|---:|---:|
| 6288 | 7168 | 14,456 | 0.577 | 1208 | **33.0%** |
| 8448 | 7168 | 8,800 | 0.675 | 1387 | **23.5%** |
| 3584 | 7168 | 14,456 | 0.330 | 1202 | **18.9%** |
| 7168 | 4224 | 8,800 | 0.374 | 1250 | 13.1% |
| 7168 | 1536 | 8,800 | 0.151 | 1127 | 5.3% |
| 7168 | 768 | 8,800 | 0.093 | 912 | 3.3% |
| 2304 | 1536 | 8,800 | 0.063 | 871 | 2.2% |
| 1536 | 128 | 14,456 | 0.014 | 222 | 0.8% |

**~1200–1400 TFLOP/s against ~2600 BF16 peak — about 50%.**

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

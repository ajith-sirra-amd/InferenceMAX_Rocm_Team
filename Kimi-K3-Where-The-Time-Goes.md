# Kimi-K3 — where the time actually goes

Companion to [Kimi-DCP-Experiemnts-Summary.md](Kimi-DCP-Experiemnts-Summary.md).

Measured on 8× MI355X, DCP=8, concurrency 52, from run
[T103](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/32855763638)
(7,950.6 tok/s/GPU) plus kernel microbenchmarks on the same host.

---

## Headline

**~94% of time is BF16 dense GEMMs. Attention is 5.6%. MoE is already quantised
and cheap.**

| Component | Share of time | Precision | State |
|---|---:|---|---|
| **Dense GEMMs** | **~94%** | BF16 | hipBLASLt, ~50% of peak |
| MLA attention | 5.6% | BF16 | fp8 unavailable on ROCm |
| MoE experts | small | a8w4 | already quantised |

---

## The paradox: 7.3% of weights, 94% of time

| | Params | Share | Active per token |
|---|---:|---:|---:|
| Experts | 2,595 B | 92.67% | ~23–46 B (top-k of **896**) |
| **Dense** | **205 B** | **7.33%** | **205 B — all of it** |

Two effects compound:

1. **Sparsity.** Every token traverses every dense layer; it touches only a
   handful of 896 experts. Dense does **4–9× the FLOPs** of the active experts.
2. **Precision.** Dense is BF16; MoE is a8w4, ~4× faster per FLOP. That
   multiplies the gap again.

Predicted dense time share: **94.7% (top_k=16) … 97.3% (top_k=8)**.
Measured: **~94%**. Model and measurement agree.

**Parameter count tracks memory, not compute.** MoE made parameters cheap — so
effectively the *unquantised* 7.3% now owns the runtime.

---

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

| Lever | Attacks | Est. | Risk |
|---|---|---|---|
| **Tune aiter GEMM kernels** for the 8 shapes | 94% | 50% → 80% of peak ≈ **1.5×** | None — same precision |
| **Quantise dense weights to fp8** | 94% | up to ~2× on that slice | Changes numerics; needs GSM8K |
| Reduce host per-request cost (1.5 ms/req) | 82% of TPOT | raises the concurrency peak | Upstream vLLM work |
| fp8 attention | 5.6% | ~2.8% | Needs a new ROCm prefill backend |

Everything reachable by configuration is already measured and closed:
concurrency peaks at 52, chunk 16384 costs 2.5%, async scheduling costs 9.2%,
MTP costs 85%, extra cache tiers do nothing (`ext_cache_hit` ≈ 0, pool 23.5%
used).

---

## Method note

The GEMM timings use `a@b` (torch/hipBLASLt), which is the path aiter actually
falls back to for these shapes, so they represent production behaviour. Shares
are frequency-weighted from the run log rather than assumed. The attention
benchmark initially used the 576-wide *latent* head dim and failed with "CK only
supports head dimension at most 256"; the numbers above use the correct
post-decompression dims (qk=192, v=128).

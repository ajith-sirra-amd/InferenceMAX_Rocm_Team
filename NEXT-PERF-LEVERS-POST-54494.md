# Next perf levers after #54494 — ranked, with gain estimates

**Written:** 2026-10-09. Context: `#54494` (DCP MLA query replication, decode-side)
is dispatched and running at conc=96 isolated against the 14,077 tok/s/GPU
clean baseline (see `HANDOFF-FMHA-PR54494.md` for its full story — patch
crash, root cause, fix, dry-run validation). This doc answers "what else do we
touch after that" — the remaining levers, ranked by expected payoff, with the
reasoning behind each estimate made explicit so the numbers can be checked
rather than taken on faith.

All percentages are share-of-total-wall-clock from the rocprofv3 + aiperf
breakdown in `Kimi-K3-Where-The-Time-Goes.md` (2026-10-07 section), measured
at conc=72/96 on this same nightly image. That breakdown has a known caveat:
chunked prefill interleaves prefill and decode work within a single step, so
the prefill/decode split is indicative, not exact.

---

## Ranking table

**Two corrections from the first cut of this doc, both caught during this
same write-up pass, kept visible rather than silently fixed:**

1. "Attention FP8 via `use_prefill_query_quantization`" was originally
   ranked #1 as a free flag flip. Wrong — direct code inspection traced
   `backend_supports_prefill_query_quantization()` and found it hardcodes
   support to GB200 (device capability 100) only; FlashAttention-family
   backends (what we run) are explicitly excluded. Deliberate hardware
   gate, not a cheap unlock. **Demoted, see appendix.**
2. "CK `is_v3_atomic_fp32` unlock" was then promoted to #1 as the easy
   patch. Also wrong — verified against two independent sources (ROCm
   TransformerEngine's `NVTE_CK_IS_V3_ATOMIC_FP32` docs, and a
   facebookexperimental/triton PR benchmarking the same AITER flag) that
   this controls **dQ gradient accumulation in the backward pass only**.
   `FlashAttnVarlenFunc.forward` only carries it in its signature because
   that class is a `torch.autograd.Function` staticmethod stashing the
   value into `ctx` for a paired `backward()`. vLLM inference never calls
   `.backward()` — this flag is **inert for serving**, not a lever at any
   size. **Also demoted, see appendix.** No patch was built for it.

Net result: there is no remaining "easy, free" lever on this list. The two
candidates that looked cheapest both turned out to not actually affect
inference. What's left is one real-but-costlier patch (dense GEMM) and one
large-but-unquantified investigation (idle).

| Rank | Lever | Share of wall | Mechanism | Patch effort | Numerics risk | Estimated ceiling |
|---|---|---:|---|---|---|---:|
| 1 | Dense GEMM BF16→FP8 (`#55811`/`#56036`) | 11.9% | Quantize dense (non-MoE) GEMMs to FP8 | Medium — apply + wire quant path | High (changes GEMM precision broadly) | **~6.5%** |
| 2 | Idle / decode launch-rate | 28.2% | Scheduler-level: reduce per-step host-side launch overhead | Unknown — no patch identified | N/A | **Unquantified, likely largest if solved** |
| — | ~~Attention FP8 (`use_prefill_query_quantization`)~~ | 16.9% | Hard GB200-only gate in vLLM | High | Unknown | **Dead end; see appendix** |
| — | ~~CK `is_v3_atomic_fp32` unlock~~ | — | Backward-pass-only parameter | — | — | **Dead end — inert for inference; see appendix** |

**Read on stacking:** these don't add linearly — they share the same GPU-busy
budget (Amdahl's law applies directly: speeding up a share `s` of wall-clock
by factor `k` yields at most `s - s/k` of total wall-clock recovered, and two
levers that touch overlapping phases of the same step recover less than the
sum of their individual numbers when combined). Treat the ceilings below as
upper bounds on each lever in isolation, not a budget to sum.

---

## Lever 1 — Dense GEMM BF16→FP8 (`#55811` / `#56036`)

### What it is
The non-MoE ("dense") GEMMs in the model currently run BF16. These two PRs
quantize them to FP8.

### Gain estimate — how the ~6.5% was derived
Previously-measured number from earlier in the campaign — a direct
BF16→FP8 dense-GEMM A/B comparison at the kernel level gave ~+6.5%
end-to-end. Carried forward unchanged since the underlying GEMM shapes and
hardware haven't changed since that measurement.

### Why this is now the top real candidate
Both of the two levers that looked cheaper (attention FP8 via a flag, CK
atomic-fp32 via a one-line patch) turned out to be dead ends on closer
inspection — see the appendix. This is the only remaining candidate with
both a real patch path and a previously-measured, non-guessed number behind
it. The tradeoff is real too: higher effort (apply + wire the quantization
path, not a flag or single-parameter change) and higher numerics risk
(touches GEMMs broadly across the model). Known tension: AMD's
`Kimi-K3-Quark-MXFP4-AttnFP8` checkpoint was already tried once and dropped
specifically because it shrank native KV cache capacity (9.38 GiB vs 59.81
GiB) — this lever is only clearly worth it if KV-capacity headroom from
other work (offload tuning, etc.) has grown enough to absorb that again, or
if this quantization path doesn't carry the same KV-shrinking side effect
(needs re-verification before staging).

### Status
Identified, not staged. Mandatory GSM8K gate before any numbers are
trusted — this is the most invasive numerics change on this list.

---

## Lever 2 — Idle / decode launch-rate

### What it is
28.2% of total wall-clock — the single largest slice in the breakdown — is
GPU idle time during decode, i.e. host-side step overhead (scheduling,
kernel-launch rate, Python-loop overhead between CUDA-graph-captured
regions) rather than GPU compute itself.

### Why it's ranked #2 despite being the biggest slice
No concrete patch has been identified for this yet. It's flagged here
deliberately as the honest biggest opportunity, not omitted, but ranked
below Lever 1 because Lever 1 has a real, previously-measured patch path
while this one requires actual investigation — likely scheduler-loop
profiling to find exactly what's stalling the GPU between steps
(step-to-step Python overhead, collective sync stalls, or admission-queue
gaps are all plausible candidates based on patterns already seen in this
campaign's KV-saturation diagnosis work).

### Gain estimate
Deliberately left unquantified. Unlike Lever 1, there's no existing
mechanism (flag, PR, or kernel parameter) to anchor a ratio-based estimate
against — any number here would be a guess dressed up as analysis. What can
be said: if even a third of this 28.2% converts to useful work, that alone
would exceed every other lever on this list combined. This is why it's
worth dedicated investigation time even without a number yet.

### Suggested next step
Profile a decode-only window (no prefill interleaved) with rocprofv3 at
step granularity to separate "GPU truly idle waiting on host" from "GPU
idle waiting on a collective" from "GPU idle waiting on KV/admission" —
these have different fixes (reduce launch overhead vs. overlap collectives
vs. the capacity-cliff class of fix already found for the C72+MTP case).

---

## Appendix — two demoted dead ends

### Demoted #1 — Attention FP8 (`use_prefill_query_quantization`)

Kept here for the record since it was the original #1 pick and the mistake
is worth being able to check later.

### What looked promising
vLLM's base `MLACommonImpl.forward_mha` threads FP8 query quantization
through both the new-tokens prefill path and the cached-context-merge path,
gated by a single attention-config flag: `use_prefill_query_quantization`.
Our real executing prefill kernel (`AiterFlashAttnPrefillBackend` →
`aiter.flash_attn_varlen_func` → CK's `FlashAttnVarlenFunc.forward`) has no
FP8 special-casing of its own, so turning the flag on would exercise
vLLM's generic FP8-query path, not an AMD-specific path we'd need to build
— and `flydsl_flash_attn_fp8_supported(...)` confirmed `SUPPORTED = True`
for our exact MLA shapes (nq=nkv=12, qk_hdim=192, v_hdim=128).

### Why it's actually dead
`backend_supports_prefill_query_quantization()` — the real gate vLLM checks
before honoring the flag — hardcodes support to GB200 (device capability
100) only. FlashInfer and TRT-LLM Ragged backends are explicitly listed as
supported; FlashAttention-family backends (`AiterFlashAttnPrefillBackend`,
what we run) are explicitly not. Confirmed by reading the server log's
fallback-rejection message directly, not just the source. This is a
deliberate hardware gate, not a missing-registration gap — unlocking it
for real would mean patching vLLM's own gate function and accepting
whatever correctness assumption that gate was protecting, which hasn't
been characterized. Not pursued further without a much better understanding
of *why* the gate excludes FlashAttention-family backends specifically.

### Demoted #2 — CK `is_v3_atomic_fp32` unlock

The CK kernel that executes our attention (`FlashAttnVarlenFunc.forward`)
has a parameter `is_v3_atomic_fp32: bool | None = True` in its signature,
hardcoded `True` by the public `aiter.flash_attn_varlen_func` wrapper and
never exposed to callers. This looked like a trivial, low-risk unlock
(expose it as a passthrough parameter, default unchanged).

**Why it's actually dead:** verified against two independent sources —
ROCm TransformerEngine's docs for the same flag (exposed there as
`NVTE_CK_IS_V3_ATOMIC_FP32`) and a facebookexperimental/triton PR
benchmarking against AITER — that this controls whether AITER's CK v3
kernels accumulate the **dQ gradient in the backward pass** using FP32
atomics vs. cheaper fp16/bf16 atomics. It has no effect on the forward
computation. `FlashAttnVarlenFunc.forward` only carries this parameter in
its signature because that class is a `torch.autograd.Function`
staticmethod — `forward()` stashes the value into `ctx` purely for a
paired `backward()` to consume. vLLM inference serving never calls
`.backward()` (pure forward generation, no training), so this flag is
inert for our workload: flipping it changes zero measured throughput,
latency, or numerics in production serving. No patch was built for this;
building one would have been pure wasted GPU-test time.

Sources: [ROCm/TransformerEngine](https://github.com/ROCm/TransformerEngine)
(`NVTE_CK_IS_V3_ATOMIC_FP32` — backward-pass dQ atomic accumulation),
[facebookexperimental/triton PR #3775](https://github.com/facebookexperimental/triton/pull/3775)
("Add FP32 dQ accumulation and optimize varlen attention **backward**",
benchmarked directly against AITER's `is_v3_atomic_fp32=True` baseline).

---

## Bottom line

No single lever here reaches 16,000 tok/s/GPU on its own from the 14,077
baseline (+13.7% needed). Of the four candidates considered, two turned out
to be dead ends on closer inspection (both demoted above) — a reminder that
"looks like a cheap flag/parameter flip" needs verification against the
actual mechanism before it's trusted. What's left: Lever 1 (dense GEMM) is
a known quantity (previously measured ~6.5%) but higher effort and
numerics risk. Lever 2 (idle) is the biggest number on paper but needs real
profiling investigation before it has a number at all. Realistic path to
16,000 likely requires both landing together, not one silver bullet —
consistent with the honest framing already in `HANDOFF-FMHA-PR54494.md`.

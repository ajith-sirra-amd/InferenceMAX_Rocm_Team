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

| Rank | Lever | Share of wall | Mechanism | Patch effort | Numerics risk | Estimated ceiling |
|---|---|---:|---|---|---|---:|
| 1 | Attention FP8 (`use_prefill_query_quantization`) | 16.9% | Quantize prefill query to FP8, reuse native vLLM FP8 prefill path | None — existing flag | Medium (changes attention precision) | **~4-9%** |
| 2 | Idle / decode launch-rate | 28.2% | Scheduler-level: reduce per-step host-side launch overhead | Unknown — no patch identified | N/A | **Unquantified, likely largest if solved** |
| 3 | Dense GEMM BF16→FP8 (`#55811`/`#56036`) | 11.9% | Quantize dense (non-MoE) GEMMs to FP8 | Medium — apply + wire quant path | High (changes GEMM precision broadly) | **~6.5%** |
| 4 | CK `is_v3_atomic_fp32` unlock | (subset of 16.9%) | Expose hardcoded `True` softmax-accumulation flag as a passthrough param | Trivial — one-line signature change | Low (default stays `True`, opt-in only) | **~3-4%** |

**Read on stacking:** these don't add linearly — they share the same GPU-busy
budget (Amdahl's law applies directly: speeding up a share `s` of wall-clock
by factor `k` yields at most `s - s/k` of total wall-clock recovered, and two
levers that touch overlapping phases of the same step recover less than the
sum of their individual numbers when combined). Treat the ceilings below as
upper bounds on each lever in isolation, not a budget to sum.

---

## Lever 1 — Attention FP8 (`use_prefill_query_quantization`)

### What it is
vLLM's base `MLACommonImpl.forward_mha` already threads FP8 query
quantization through both the new-tokens prefill path and the
cached-context-merge path. It's gated by a single attention-config flag:
`use_prefill_query_quantization`. No patch needed — this is a native,
already-shipped capability; we've just never turned it on.

### Why it should work here
- Our real executing prefill kernel (`AiterFlashAttnPrefillBackend` →
  `aiter.flash_attn_varlen_func` → CK's `FlashAttnVarlenFunc.forward`) has
  no FP8 special-casing of its own — it forwards whatever dtype it's given.
  So turning the flag on exercises vLLM's generic FP8-query path end-to-end,
  not an AMD-specific code path we'd need to build.
- Directly confirmed via live call: `flydsl_flash_attn_fp8_supported(device,
  nq=12, nkv=12, qk_hdim=192, v_hdim=128, dtype=torch.float8_e4m3fn)` →
  `SUPPORTED = True` for our exact MLA shapes.
- vLLM's own source recommends it explicitly for our regime: *"For
  long-context workloads (ISL >= 4K), enabling FP8 prefill attention can
  significantly optimize prefill latency."* Our median ISL is ~100K.

### Gain estimate — how the ~4-9% was derived
Attention is 16.9% of total wall-clock. FP8 vs BF16 compute throughput on
this hardware measured elsewhere in this campaign (dense-GEMM A/B tests) runs
roughly 1.3-2x. Applying that same ratio to attention's share:
`16.9% × (1 - 1/1.3)` to `16.9% × (1 - 1/2.0)` = **~3.9% to ~8.5%**, rounded
to ~4-9%. This assumes attention's *compute* is what's FP8-accelerated and
that memory-bandwidth-bound portions of the kernel don't get the full ratio
— a conservative-leaning estimate, not an optimistic one.

### One known interaction to re-measure
`#59070` ("keep DCP prefill context FP8 through AllGather") requires
`q.dtype == torch.bfloat16` to engage its own optimization. Turning on
`use_prefill_query_quantization` makes q FP8, so `#59070`'s specific path
steps aside in favor of the base class's own FP8-capable context path — not
a correctness issue, just something that changes which code path runs.

### Status — not yet dispatched
Next in queue once `#54494`'s C96 result lands. GSM8K-200 gate (0.995
threshold) mandatory before trusting any throughput number, same as every
other numerics-affecting change this campaign.

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
below Lever 1 because Lever 1 is immediately actionable (a flag flip,
already measured as feasible) while this one requires actual
investigation — likely scheduler-loop profiling to find exactly what's
stalling the GPU between steps (step-to-step Python overhead, collective
sync stalls, or admission-queue gaps are all plausible candidates based on
patterns already seen in this campaign's KV-saturation diagnosis work).

### Gain estimate
Deliberately left unquantified. Unlike Levers 1, 3, and 4, there's no
existing mechanism (flag, PR, or kernel parameter) to anchor a ratio-based
estimate against — any number here would be a guess dressed up as analysis.
What can be said: if even a third of this 28.2% converts to useful work,
that alone would exceed every other lever on this list combined. This is
why it's worth dedicated investigation time even without a number yet.

### Suggested next step
Profile a decode-only window (no prefill interleaved) with rocprofv3 at
step granularity to separate "GPU truly idle waiting on host" from "GPU
idle waiting on a collective" from "GPU idle waiting on KV/admission" —
these have different fixes (reduce launch overhead vs. overlap collectives
vs. the capacity-cliff class of fix already found for the C72+MTP case).

---

## Lever 3 — Dense GEMM BF16→FP8 (`#55811` / `#56036`)

### What it is
The non-MoE ("dense") GEMMs in the model currently run BF16. These two PRs
quantize them to FP8, same precision-swap idea as Lever 1 but applied to
GEMM instead of attention.

### Gain estimate — how the ~6.5% was derived
This is a previously-measured number from earlier in the campaign (not a
fresh ratio-based estimate like Lever 1) — a direct BF16→FP8 dense-GEMM A/B
comparison at the kernel level gave ~+6.5% end-to-end. It's carried forward
here unchanged because the underlying GEMM shapes and hardware haven't
changed since that measurement.

### Why it's ranked below Lever 1 despite a comparable/larger number
Higher effort (requires applying + wiring the quantization path, not a
single flag) and higher numerics risk (touches GEMMs broadly across the
model, not a single well-scoped attention path with a documented support
matrix). Also has a known tension: AMD's
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

## Lever 4 — CK `is_v3_atomic_fp32` unlock

### What it is
The CK kernel that actually executes our attention (`FlashAttnVarlenFunc.
forward`) has a real, named precision-control parameter:
`is_v3_atomic_fp32: bool | None = True`. It controls whether softmax
accumulation happens in atomic FP32 vs. a cheaper path. The public wrapper
(`aiter.flash_attn_varlen_func`, what vLLM actually calls) hardcodes this to
`True` and never exposes it to callers.

### Patch
Trivial: change the hardcoded `True` in the wrapper to a passthrough
parameter, defaulting to `True` so behavior is unchanged unless a caller
explicitly flips it. Lowest-risk patch on this list — opt-in, no default
behavior change, single call site.

### Gain estimate — how the ~3-4% was derived
Same ratio logic as Lever 1, but scoped down: this only affects the
*softmax-accumulation* portion of attention's cost, not the whole attention
kernel, so it's a fraction of attention's 16.9% share rather than all of it
— hence the smaller ~3-4% ceiling versus Lever 1's ~4-9% for the full
attention-dtype change. Not yet tested; this is a candidate ceiling, not a
measured one.

### Status
Saved as a candidate (also recorded in project memory). Not yet
implemented or tested. Explicitly kept on the list per standing instruction:
*even a ~3% benefit is worth taking* — small, low-risk levers aren't
dropped just because their ceiling is modest.

---

## Bottom line

No single lever here reaches 16,000 tok/s/GPU on its own from the 14,077
baseline (+13.7% needed). Lever 1 (attention FP8) is the best next move —
cheapest to test, no patch required, support already confirmed live on our
hardware. Lever 2 (idle) is the biggest number on paper but needs real
investigation before it has a number at all. Levers 3 and 4 are known
quantities (previously measured / structurally bounded) but either higher
risk (3) or lower ceiling (4). Realistic path to 16,000 likely requires at
least two of these landing together, not one silver bullet — consistent
with the honest framing already in `HANDOFF-FMHA-PR54494.md`.

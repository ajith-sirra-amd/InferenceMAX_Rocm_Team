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

**Status as of the latest pass: every specific candidate lever proposed in
this doc has failed verification.** Three corrections, all caught by
checking the actual mechanism/source instead of trusting the original
framing:

1. "Attention FP8 via `use_prefill_query_quantization`" — direct code
   inspection traced `backend_supports_prefill_query_quantization()` and
   found it hardcodes support to GB200 (device capability 100) only;
   FlashAttention-family backends (what we run) are explicitly excluded.
   Deliberate hardware gate, not a cheap unlock. **Dead end, see appendix.**
2. "CK `is_v3_atomic_fp32` unlock" — verified against two independent
   sources (ROCm TransformerEngine's `NVTE_CK_IS_V3_ATOMIC_FP32` docs, and
   a facebookexperimental/triton PR benchmarking the same AITER flag) that
   this controls **dQ gradient accumulation in the backward pass only**,
   inert for inference (vLLM never calls `.backward()`). **Dead end, see
   appendix.**
3. "Dense GEMM BF16→FP8 (`#55811`/`#56036`)" — pulled the real PR bodies
   from `vllm-project/vllm` and neither is what the name claimed. `#55811`
   is "Add merged MoE front" — fuses three separate GEMMs into one BF16
   GEMM (kernel-count reduction), not an FP8 dtype change. `#56036` is
   "Integrate AITER KDA prefill" — an alternative prefill backend for the
   KDA kernel, and its own benchmark states plainly: *"On gfx950, native
   fused remains the fastest backend ... this PR does not claim an
   end-to-end performance improvement over current main on gfx950."*
   Neither PR is about dense-GEMM FP8 quantization, and the second is an
   explicit non-win on our exact hardware by its author's own numbers.
   Separately, `#59069` ("Fuse AttnRes output with per-token FP8 input
   quantization," already merged, already in our enabled bundle) turns out
   to already cover part of what "dense GEMM FP8" would have meant — so
   even the premise of this being untapped work was stale. **Dead end,
   see appendix.**

Net result: there is no verified, actionable lever left on this list beyond
the one candidate that was never a specific mechanism to begin with — idle
investigation. This is worth stating plainly: today's research process
repeatedly took old/remembered PR numbers and mechanism descriptions at
face value instead of re-checking them against the real source, and every
single one broke on inspection. Any future candidate on this list should be
verified against real source (`gh pr view`, direct code read) *before*
being written down as a ranked lever, not after.

| Rank | Lever | Share of wall | Mechanism | Patch effort | Numerics risk | Estimated ceiling |
|---|---|---:|---|---|---|---:|
| 1 | Idle / decode launch-rate | 28.2% | Scheduler-level: reduce per-step host-side launch overhead | Unknown — no patch identified | N/A | **Unquantified, likely largest if solved** |
| — | ~~Attention FP8 (`use_prefill_query_quantization`)~~ | 16.9% | Hard GB200-only gate in vLLM | High | Unknown | **Dead end; see appendix** |
| — | ~~CK `is_v3_atomic_fp32` unlock~~ | — | Backward-pass-only parameter | — | — | **Dead end — inert for inference; see appendix** |
| — | ~~Dense GEMM BF16→FP8 (`#55811`/`#56036`)~~ | — | Neither PR is what it was described as | — | — | **Dead end — mischaracterized; see appendix** |

**Read on stacking:** these don't add linearly — they share the same GPU-busy
budget (Amdahl's law applies directly: speeding up a share `s` of wall-clock
by factor `k` yields at most `s - s/k` of total wall-clock recovered, and two
levers that touch overlapping phases of the same step recover less than the
sum of their individual numbers when combined). Treat the ceilings below as
upper bounds on each lever in isolation, not a budget to sum.

---

## Lever 1 — Idle / decode launch-rate

### What it is
28.2% of total wall-clock — the single largest slice in the breakdown — is
GPU idle time during decode, i.e. host-side step overhead (scheduling,
kernel-launch rate, Python-loop overhead between CUDA-graph-captured
regions) rather than GPU compute itself.

### Why it's #1 by default, not by merit
Every other candidate considered this round (attention FP8, CK
atomic-fp32, dense GEMM) turned out to be a dead end on verification — see
the appendix. This is the only item left standing, and it's here because
it's an honestly-scoped *area* (a measured share of wall-clock with no
specific mechanism attached yet), not because a patch path has been found
for it. Treat it as the next thing to investigate, not the next thing to
patch.

### Gain estimate
Deliberately left unquantified. There's no existing mechanism (flag, PR, or
kernel parameter) to anchor a ratio-based estimate against — any number
here would be a guess dressed up as analysis. What can be said: if even a
third of this 28.2% converts to useful work, that alone would exceed every
other lever considered this round combined. This is why it's worth
dedicated investigation time even without a number yet.

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

### Demoted #3 — Dense GEMM BF16→FP8 (`#55811` / `#56036`)

Described as "quantize the non-MoE dense GEMMs to FP8, ~6.5% ceiling from a
previously-measured A/B." The ~6.5% number itself was never re-derived in
this doc — it was carried forward from an earlier reference without
re-checking what PRs actually implement it, and the two PR numbers attached
to it turned out to be wrong:

- [`#55811`](https://github.com/vllm-project/vllm/pull/55811) — "Add merged
  MoE front." Replaces three separate projections (shared gate/up, router
  logits, routed latent) with one BF16 GEMM into an FP32 accumulator,
  followed by AITER's SiTU/split epilogue. This is a GEMM-fusion
  (kernel-count reduction) change, not a BF16→FP8 dtype change. It's
  specific to the MoE front, not dense/non-MoE GEMMs.
- [`#56036`](https://github.com/vllm-project/vllm/pull/56036) — "Integrate
  AITER KDA prefill." Adds AITER's FlashKDA as an alternative prefill
  backend for the KDA kernel, selectable via `--kda-prefill-backend`. Its
  own benchmark table shows native fused is faster at both 4K and 16K ISL
  on gfx950, and the PR body says directly: *"This PR therefore does not
  claim an end-to-end performance improvement over current main on
  gfx950."* Not a win for us by the author's own numbers, and still not
  about FP8 quantization.
- Separately, [`#59069`](https://github.com/vllm-project/vllm/pull/59069)
  ("Fuse AttnRes output with per-token FP8 input quantization") — already
  merged, already enabled in our bundle — fuses per-token FP8 quantization
  into the AttnRes output for linears that accept `kFp8DynamicTokenSym`.
  This is actually closer to what "dense GEMM FP8" would mean, and it's
  already shipped and running, not untapped work.

No real open PR matching "quantize dense GEMMs to FP8 for ~6.5%" was found.
If this is still worth pursuing, it needs to start from scratch: profile
which dense (non-MoE) linears in the current deployment still run BF16
end-to-end (not just the ones `#59069` already covers), and check whether
a real quantization path exists for them — not from a remembered PR number.

---

## Bottom line

No single lever here reaches 16,000 tok/s/GPU on its own from the 14,077
baseline (+13.7% needed) — and as of this pass, no *verified* lever reaches
it either. All three specific candidates considered this round (attention
FP8, CK atomic-fp32, dense GEMM) turned out to be dead ends, mischaracterized,
or already-shipped on direct verification against source. What's left is
one honestly-unquantified area (idle, 28.2% of wall-clock, no patch
identified) that needs real profiling investigation before it becomes an
actual lever with a number attached. The more important takeaway for next
steps: verify every candidate against real source (`gh pr view`, direct
code read, actual benchmark data) *before* ranking or estimating it, not
after — that would have caught all three dead ends here before they were
written down.

# Handoff — FMHA exploration, #54494 (collectives), and the path to 16,000 tok/s/GPU

**Written:** 2026-10-09. Target: 16,000 tok/s/GPU. Current clean baseline:
**14,077 tok/s/GPU** (conc=96, zero patches, new image
`nightly-rocm100-81198e97ba7eee2a22540caaa756b7fdddcb4d93`, runner
`cluster:mi355x-amds-2`). Gap: **+13.7%**.

Everything below is measured or directly confirmed in installed source unless
marked **[inference]** or **[pending]**.

---

## Part 1 — FMHA (attention) exploration

### Why attention is worth attacking
Prefill attention (`fmha`) is **71.3% of prefill-active GPU time** and **16.9%
of total end-to-end wall-clock** — the #3 lever behind idle (28.2%) and
collectives (21.3%). See `Kimi-K3-Where-The-Time-Goes.md`'s 2026-10-07 section
for the full breakdown.

**Important correction baked into that doc:** our workload's ~94-95%
prefix-cache hit rate does **not** mean attention is cheap. Cache hits only
save re-doing the K/V *projection* for old tokens — every new query still has
to *attend across* the full cached context (ISL ~100K median) regardless of
how much of it was freshly projected. That's why attention dominates despite
"only 5-6%" of tokens being freshly computed.

### The chain of findings, in order

1. **The FP8 MLA prefill ASM kernel (`mla_prefill_ps_asm_fwd` +
   `mla_reduce_v1`) exists and is wired into `rocm_aiter_mla.py`'s
   `forward_mha` override.** Confirmed by direct inspection of the installed
   image: `_fp8_mla_prefill_supported()` → True, `is_quantized_kv_cache` →
   True (we run `--kv-cache-dtype fp8`), `AiterMLAHelper.is_valid_num_heads(12)`
   → True (12 heads/rank at TP8 is explicitly handled via replicate-padding).

2. **But it only fires when `attn_metadata.prefill.chunked_context is
   None`** — i.e. no cached prior context to merge. `ChunkedContextMetadata`
   is built from `context_lens` (the cached-context length), and it also
   carries a `dcp_manager` field for cross-rank gathering. Two independent
   reasons point to the same conclusion: for our config (ISL ~100K + DCP=8),
   `chunked_context` is essentially *always* populated, so this fast path is
   **structurally almost never reached by real traffic**.

3. **Went looking for "is BF16 softmax a flag" — traced the actual kernel
   dispatch, found it's not FlyDSL.** `aiter.flash_attn_varlen_func` tries, in
   order: a gfx1250-only ASM path (skipped, wrong arch) → FlyDSL → Triton/CK.
   Read FlyDSL's actual gating code directly: its BF16 path requires
   `get_gfx() == "gfx1250"` literally in the boolean expression — **always
   False on our gfx950 hardware**, confirmed in code, not inferred. Its FP8
   path only fires for FP8 inputs. So **FlyDSL never runs for us at all**,
   BF16 or FP8-blocked-by-context.

4. **The real executing kernel is CK's `FlashAttnVarlenFunc.forward`**
   (confirmed `aiter.jit.core.ENABLE_CK = True` in the installed build). Its
   signature has a real, named precision-control argument:
   `is_v3_atomic_fp32: bool | None = True`. The public wrapper
   (`aiter.flash_attn_varlen_func`, what vLLM calls) **hardcodes this to
   `True`** when invoking `FlashAttnVarlenFunc.apply(...)` — it is never
   exposed to callers. Saved to memory as a candidate: unlocking it is a tiny,
   low-risk patch (change the hardcoded `True` to a passthrough parameter,
   default `True` so behavior is unchanged unless flipped). **[pending]** —
   not yet tested; ceiling estimated ~3-4%, same reasoning as item 6 below.

5. **The actual unlock turned out to be much bigger and much cheaper than
   expected.** Our real prefill backend (`AiterFlashAttnPrefillBackend`,
   `ROCM_AITER_FA`) has **no FP8 special-casing of its own** — both
   `run_prefill_new_tokens` (new tokens) and `run_prefill_context_chunk`
   (cached context) just forward whatever dtype arrives to the same
   `flash_attn_varlen_func` dispatcher from item 3. The *base* (non-ROCm-
   specific) `MLACommonImpl.forward_mha` and `_compute_prefill_context` /
   `_context_parallel_compute_prefill_context` already thread
   `use_fp8_prefill` generically through **both** the new-tokens path and the
   context-merge path — this is a **native vLLM capability, already fully
   implemented**, gated by the attention-config flag
   `use_prefill_query_quantization`. vLLM's own source literally recommends
   it: *"For long-context workloads (ISL >= 4K), enabling FP8 prefill
   attention can significantly optimize prefill latency."* Our ISL is ~100K.

6. **Verified the one remaining hard gate directly** — called
   `flydsl_flash_attn_fp8_supported(device, nq=12, nkv=12, qk_hdim=192,
   v_hdim=128, dtype=torch.float8_e4m3fn)` live on a GPU. Result:
   **`SUPPORTED = True`** for our exact MLA shapes.

### What this means for the 16,000 target
If `use_prefill_query_quantization=True` works end-to-end, the expected
ceiling is **the same ~4-9% I estimated earlier for "full FP8 kernel"** — that
estimate already assumed attention's entire cost gets the ~1.3-2x FP8/BF16
ratio measured elsewhere in this campaign. What changed is *reachability*:
this used to look like a multi-week kernel-merge rewrite; it turned out to be
one existing, already-coded flag.

### Known trade-off when combined with the patch stack later
`#59070` ("keep DCP prefill context FP8 through AllGather") has an eligibility
check that requires `q.dtype == torch.bfloat16`. Turning on
`use_prefill_query_quantization` makes q FP8, so `#59070`'s specific
optimization would step aside in favor of the base class's own (also
FP8-capable) context path — not a correctness problem, just something to
re-measure once both are combined.

### Current test status — [pending]
Dispatched as `ENABLE_FP8_PREFILL_QUERY_QUANT=1` (adds
`"use_prefill_query_quantization":true` to `--attention-config`, merged with
the existing `mla_prefill_backend` key). **Isolated**: all `APPLY_PR*` flags
off, including `#54494`, for a clean read against the 14,077 baseline. Run
`37909323153` was still in progress as of this writing — no crash in the
first several minutes (mildly positive, not conclusive). **Needs: GSM8K gate
before trusting any throughput number, same as every other numerics-affecting
change this campaign.**

---

## Part 2 — #54494: DCP MLA query replication (collectives, 21% of wall)

### What it does
Under DCP=8, MLA decode normally all-gathers the query every layer because
the KV cache is sharded across the group. This PR replicates the query
projection across the DCP group instead — each rank computes the full head
set locally (a small amount of redundant compute) and skips the per-layer
all-gather entirely. Targets collectives, our #2 bottleneck (21.3% of wall),
with a mechanism (removing a collective, not just speeding it up) that should
be a clean win on hardware where network sync is the expensive part.

Upstream PR reports **no throughput number** — only GSM8K validation
(0.9629 vs 0.9598 off, both passing).

### The modification I made
The patch didn't apply cleanly — not because it conflicts with our other
patches (none of them touch `mla_attention.py`), but because **the code it
targets was refactored upstream since the PR was authored**. The quantization
section in `mla_attention.py` used to do direct `self.W_K = ...` attribute
assignment; it now stages into local variables and applies them via a
`replace_parameter(self, name, tensor, prefer_copy=True)` loop, specifically
to keep captured CUDA graphs valid across weight reloads.

Two hunks failed as a result:
1. `linear.py`, one import line — trivial, patch's context-matcher just
   couldn't align it after `#59693`-style import-block drift. Not a real
   conflict.
2. `mla_attention.py`, the FP4-BMM branch's `dcp_q_replicate` insertion — a
   real adaptation. I located where `self.W_K`/`self.W_K_scale` actually
   become real attributes post-refactor (after the `replace_parameter` loop,
   not immediately after the quantize call), and moved the PR's own
   already-written FP4-branch logic (`get_dcp_group().all_gather(...)` on
   both `W_K` and `W_K_scale`) to that correct, post-refactor insertion point.
   **Nothing was invented** — this is the PR author's own code, relocated to
   where the current codebase's structure requires it. Cross-checked against
   the FP8-BMM branch's equivalent hunk, which *did* apply cleanly with fuzz,
   confirming the landing spot and pattern were correct.

Verified via dry-run → real apply → `py_compile` → import checks against the
**raw, unpatched** pinned image (not just on top of our stack), so the patch
is portable and testable standalone. Saved as
`patches/pr54494-dcp-query-replication.diff`.

### Current test status — [pending, interrupted]
Was staged for an isolated test (all other `APPLY_PR*` off) against the
14,077 baseline, runner switched to `cluster:mi355x-amds-2` since
`cluster:mi355x-amds` was occupied by the C72+MTP run (see Part 3). That
dispatch got consumed by a kernel-inspect check instead of the real benchmark
(early-exit hook was still defaulted on from the prior investigation step) —
**`#54494` itself has not yet produced a real throughput number.** Needs
re-dispatch with `APPLY_PR54494=1`, everything else off, `KERNEL_INSPECT=0`.

**Note:** `#58861` (AttnRes runtime strides/launch tuning) was also found to
have one real, non-mechanical conflict with our staged `#59693` in the same
MTP aux-hidden-state block of `linear.py`'s `forward()` — flagged but not yet
resolved; held per "don't engineer third-party PRs without asking" discipline.

---

## Part 3 — New data point: DCP8 + MTP(k=3), first real agentic measurement

Built a new case arm in `kimik3_fp4_mi355x_mtp.sh` combining `DCP_SIZE=8` with
MTP speculative decoding (previously two mutually-exclusive code paths — the
low-concurrency branch forced `DCP_SIZE=1` to get MTP). The underlying vLLM
blocker (no backend declaring non-causal-DCP support) is already fixed
natively in the current image (`supports_non_causal_multi_token_dcp = True`,
confirmed). HANDOFF.md's 2026-09-23 note flagged "agentic MTP+DCP numbers
still unmeasured" (only validated on the fixed-length harness at the time) —
this closes that gap.

**Result** (conc=72, k=3, run `37889759055`, confirmed genuinely active via
live container logs — not the `spec_decoding: "none"` reporting artifact):

| Metric | DCP8+MTP(k=3), conc=72 | Baseline, conc=96, no MTP |
|---|---:|---:|
| Per-GPU throughput | 10,481 tok/s | 14,077 tok/s |
| TPOT (mean) | **50.5 ms** | 103.6 ms |
| TTFT (mean) | 20.2 s | 3.1 s |
| Success rate | 2526/3394 (74.4%) | — |

**Read:** MTP under DCP8 roughly **halves TPOT** (draft+verify producing
multiple tokens per step) but comes with a large TTFT cost and lower raw
per-GPU throughput at this (lower) concurrency. Not a clean win or loss —
different concurrency levels confound a direct comparison, and this is
genuinely new ground. Worth a same-concurrency re-run before drawing a firm
conclusion.

---

## Part 4 — Honest take on reaching 16,000 tok/s/GPU

**Where we actually are:** 14,077 measured baseline. One bundle of small,
individually-reasonable patches (`#59069` + `#59070` + `#59693` + `#59966` +
`#54625`) measured **+1.2%** — right at the campaign's noise floor, not a
real signal. The small-patch-stacking approach is not going to reach target
on its own.

**The levers, ranked, with honest ceilings:**

| Lever | Share of wall | Status | Honest ceiling |
|---|---:|---|---|
| Idle (decode launch-rate) | 28.2% | Untouched | Biggest, no concrete patch identified yet |
| Collectives (`#54494`) | 21.3% | Patch staged, hand-adapted, **not yet measured** | Mechanism removes a collective outright — plausibly real, magnitude unknown (upstream reports no number) |
| Attention (`use_prefill_query_quantization`) | 16.9% | **Live test in flight, no patch needed** | ~4-9%, same math as before, now cheap to reach |
| Dense GEMM (`#55811`/`#56036`) | 11.9% | Identified, not staged | ~+6.5% ceiling from earlier BF16→FP8 dense-GEMM analysis |

**The math, done honestly:** even optimistic, non-overlapping stacking of the
top three unmeasured/in-flight levers doesn't cleanly add up past the
required +13.7% — they share the same GPU-busy budget, so realized gains
when combined are reliably less than the sum of their individual ceilings.
**16,000 very likely requires at least two of these actually landing
together, not one silver bullet.**

**What would change this assessment:** a positive, GSM8K-clean result from
the `use_prefill_query_quantization` test, combined with a real (not
noise-level) number from `#54494` once properly isolated-tested. Both are
pending. Until then, 16,000 is a plausible-but-unproven target, not a
confirmed one.

**Immediate next steps, in order:**
1. Pull the `use_prefill_query_quantization` result; GSM8K-gate if it ran clean.
2. Re-dispatch `#54494` alone (patches off, `KERNEL_INSPECT=0`) for its first
   real throughput number.
3. If both land positive and are GSM8K-clean, stack them together and
   re-measure — do not assume additivity.
4. Revisit idle (28%, still the single biggest untouched lever) — no
   concrete patch identified for it yet in this campaign.

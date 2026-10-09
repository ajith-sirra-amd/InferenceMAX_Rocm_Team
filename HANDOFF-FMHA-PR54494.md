# Handoff — FMHA exploration, #54494 (collectives), and the path to 16,000 tok/s/GPU

**Written:** 2026-10-09. **Updated:** 2026-10-09 (same day, after a full day
of GPU-verified debugging). Target: 16,000 tok/s/GPU. Current clean baseline:
**14,077 tok/s/GPU** (conc=96, zero patches, image
`nightly-rocm100-81198e97ba7eee2a22540caaa756b7fdddcb4d93`, runner
`cluster:mi355x-amds-2`). Gap: **+13.7%**.

**Read this first:** this doc's original version (Part 1, items 4-6, and
Part 4's lever table) turned out to contain real mistakes, caught later the
same day by actually verifying claims against real source/PR bodies instead
of trusting earlier framing. Corrected inline below, with the mistake kept
visible rather than silently fixed. See `NEXT-PERF-LEVERS-POST-54494.md` for
the full corrected lever analysis, and `PR54494-ROOT-CAUSE-AND-FIX.md` for
the full `#54494` debugging saga (crash → real fix → still broken → smoke
test). This doc is the overview; those two are the detailed record.

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
   None`** — i.e. no cached prior context to merge. For our config (ISL
   ~100K + DCP=8), `chunked_context` is essentially *always* populated, so
   this fast path is **structurally almost never reached by real traffic**.

3. **FlyDSL never runs for us.** `aiter.flash_attn_varlen_func` tries, in
   order: a gfx1250-only ASM path (skipped, wrong arch) → FlyDSL → Triton/CK.
   FlyDSL's BF16 path requires `get_gfx() == "gfx1250"` literally — always
   False on gfx950. Its FP8 path only fires for FP8 inputs. Confirmed in
   code, not inferred.

4. **~~CK's `is_v3_atomic_fp32` unlock~~ — CORRECTED, this is a dead end.**
   The original version of this doc described `is_v3_atomic_fp32` (a
   parameter on CK's `FlashAttnVarlenFunc.forward`, hardcoded `True` by the
   public wrapper) as a cheap softmax-precision unlock worth a small patch.
   **This was wrong.** Verified later the same day against two independent
   sources (ROCm TransformerEngine's `NVTE_CK_IS_V3_ATOMIC_FP32` docs, and a
   facebookexperimental/triton PR benchmarking the same AITER flag): this
   parameter controls **dQ gradient accumulation in the backward pass
   only**. `FlashAttnVarlenFunc.forward` carries it in its signature purely
   because that class is a `torch.autograd.Function` staticmethod stashing
   the value into `ctx` for a paired `backward()`. vLLM inference never
   calls `.backward()` — the flag is **inert for serving**, not a lever at
   any size. No patch was built; building one would have been pure wasted
   GPU time. Full writeup in `NEXT-PERF-LEVERS-POST-54494.md`'s appendix.

5. **~~Attention FP8 via `use_prefill_query_quantization`~~ — CORRECTED,
   also a dead end.** The original version of this doc (items 5-6 below,
   and the "Current test status" section) described this as "the actual
   unlock" — a native, already-implemented vLLM feature gated by a single
   flag, with support confirmed live on our hardware. **This was also
   wrong**, caught later the same day: `backend_supports_prefill_query_
   quantization()` — the real gate vLLM checks before honoring the flag —
   hardcodes support to GB200 (device capability 100) only.
   FlashAttention-family backends (`AiterFlashAttnPrefillBackend`, what we
   run) are explicitly excluded. This was confirmed two ways: first by
   reading the gate function's source, then independently re-confirmed by
   reading the server log's actual fallback-rejection message from a live
   test. Deliberate hardware gate, not a missing-registration gap — not a
   cheap unlock. Full writeup in `NEXT-PERF-LEVERS-POST-54494.md`'s
   appendix.

6. **What's left standing from this investigation:** nothing with a real
   patch path. See Part 4 below and `NEXT-PERF-LEVERS-POST-54494.md` for
   the current, honest state — only "idle" (28.2% of wall, unquantified, no
   mechanism identified) remains as a real area to investigate.

---

## Part 2 — #54494: DCP MLA query replication (collectives, 21% of wall)

**This section is a summary. Full details, including the real bug, the
real fix, and the second failure, are in `PR54494-ROOT-CAUSE-AND-FIX.md`
— read that file for the complete story.**

### What it does
Under DCP=8, MLA decode normally all-gathers the query every layer because
the KV cache is sharded across the group. This PR replicates the query
projection across the DCP group instead — each rank computes the full head
set locally (a small amount of redundant compute) and skips the per-layer
all-gather entirely. Targets collectives, our #2 bottleneck (21.3% of wall).

### The saga, in order

1. **First hand-adaptation (round 1):** patch didn't apply cleanly against
   our pinned image because the targeted code had been refactored upstream.
   Hand-adapted to the new `replace_parameter`-based structure. Looked
   correct, tested positive via `bash -n`/import checks.

2. **First real GPU test crashed immediately:** `TypeError: cannot assign
   'torch.cuda.ByteTensor' as parameter 'W_K'`. Root cause: the hand-adapted
   patch did plain `self.W_K = <raw tensor>` reassignment on an attribute
   that's registered as an `nn.Parameter`, which `nn.Module` rejects. Fixed
   by routing through `replace_parameter(...)` like the surrounding code
   already does.

3. **Hunk-header bugs (round 2):** the fix above was written with
   miscounted unified-diff hunk headers (new-side line counts didn't match
   actual content) — `patch` rejected the file outright with "malformed
   patch." Fixed by writing a small arithmetic validator script and
   re-checking every hunk in the file, not just the ones that were edited.
   This became a standing practice for every patch edited afterward.

4. **First successful-apply real GPU test hit a genuine NCCL deadlock:**
   ran cleanly for ~40 minutes, then the engine's request-processing loop
   silently stopped (HTTP server stayed up, scheduler stopped). Exactly 600
   seconds later (the NCCL default collective timeout), all 8 ranks'
   watchdogs fired on an `_ALLGATHER_BASE` collective, and the engine
   crashed (`c10::DistBackendError`).

5. **Root-caused via direct source reading (not guessing):** fetched the
   *exact pinned* vLLM commit from GitHub (not main HEAD — main moves
   ~85 commits/day, a real risk caught mid-investigation) and traced the
   bug precisely: the hand-adapted patch overwrote `self.W_K`/`W_K_scale`
   in place with the all-gathered (bigger) tensor, but `self.W_K` is also
   read unconditionally by every decode call regardless of replication
   state. Upstream's own working pattern for a different (non-BMM) code
   path never does this — it creates a *separate* new parameter
   (`W_UK_T_dcp_qrep`) and branches on a `qrep_decode` flag at the call
   site. Our case (FP4/FP8 BMM) was the one upstream explicitly marked
   `NotImplementedError` for — our hand-adaptation had *deleted* that guard
   instead of implementing the missing feature.

6. **Built the real fix:** mirrored the proven working pattern exactly —
   new `W_K_dcp_qrep`/`W_K_scale_dcp_qrep` parameters, originals untouched,
   `qrep_decode` branch at the decode call site. Verified by applying for
   real (`patch -p1`) against the exact pinned source and `py_compile`-ing
   the result — not just checking hunk arithmetic.

7. **Second real GPU test hit the IDENTICAL deadlock** — same
   `_ALLGATHER_BASE` op, same exact tensor shapes (`38043648/4755456 = 8.0`,
   an 8-way gather), despite the W_K fix being genuinely correct. Traced the
   shape to `MLADCPManager._gather_query` in `dcp.py` — the *old*,
   non-replicated query-gather collective that `qrep_decode=True` is
   supposed to skip entirely. Conclusion: `qrep_decode` isn't staying
   `True` for every decode step/rank while the feature is enabled, and when
   that diverges across the 8 ranks, the group hangs.

8. **Built a fast reproduction instead of guessing a third fix:** added
   temporary debug logging (`qrep_decode` per rank per decode step, gated
   behind `VLLM_DEBUG_QREP=1`) and a `SMOKE_TEST=1` launcher mode that
   skips the ~1hr agentic replay in favor of a bounded, direct curl-based
   load test against the live server.

9. **Smoke test ran clean — no divergence, no crash.** Across 93,312
   logged decode calls (8 ranks × full 8-minute window, 32 concurrent
   requests), `qrep_decode` was `True` 100% of the time, zero collective
   errors. This is useful negative evidence: the trigger isn't "decode
   under load" generically — both real failures happened **~40-50 minutes**
   in, specifically as requests backed up on **external KV-cache
   fetches** (`Deferred: 26 reqs, KV fetch ... in progress`). The smoke
   test's short, light load never built up that kind of queue/offload
   pressure. (Also found, unrelated to `#54494`: the smoke test's own
   cleanup code had a bug — a bare `wait` caught the backgrounded vLLM
   server process too, hanging the script for 42 minutes after the actual
   smoke window had already finished cleanly. Not yet fixed.)

### Current status
`#54494` is **disabled by default** (`APPLY_PR54494=0`). The `W_K` fix is
real and should stay (commit `aa652bbb`), but it is not sufficient — a
second, still-unidentified bug causes `qrep_decode` to diverge across
ranks under sustained load/KV-offload pressure. Do not re-enable without a
new finding. See `PR54494-ROOT-CAUSE-AND-FIX.md` for the complete,
up-to-date record and next-step options.

---

## Part 3 — DCP8 + MTP(k=3): status

Built a case arm in `kimik3_fp4_mi355x_mtp.sh` combining `DCP_SIZE=8` with
MTP speculative decoding. First real measurement (conc=72, k=3, run
`37889759055`):

| Metric | DCP8+MTP(k=3), conc=72 | Baseline, conc=96, no MTP |
|---|---:|---:|
| Per-GPU throughput | 10,481 tok/s | 14,077 tok/s |
| TPOT (mean) | **50.5 ms** | 103.6 ms |
| TTFT (mean) | 20.2 s | 3.1 s |
| Success rate | 2526/3394 (74.4%) | — |

A follow-up C64+MTP test (`MAX_NUM_SEQS=80`, fixing a KV-capacity-cliff
issue found in a C72+MTP run — KV usage pinned ~97-99.7%, admission queue
never draining) failed with `RuntimeError: cancelled` inside EngineCore —
traced to an external cancellation (the user cancelling the GPU
reservation mid-run), not a code bug. Deprioritized, not yet re-run.

---

## Part 4 — Honest take on reaching 16,000 tok/s/GPU

**Where we actually are:** 14,077 measured baseline. One bundle of small,
individually-reasonable patches (`#59069` + `#59070` + `#59693` + `#59966`
+ `#54625`) measured **+1.2%** — right at the campaign's noise floor, not
a real signal.

**The levers — this table superseded `NEXT-PERF-LEVERS-POST-54494.md`'s
own first draft, which itself needed two rounds of correction. Read that
doc for the full story; here is the current, final state:**

| Lever | Share of wall | Status | Honest ceiling |
|---|---:|---|---|
| Idle (decode launch-rate) | 28.2% | Untouched, no mechanism identified | Biggest, unquantified |
| Collectives (`#54494`) | 21.3% | Real bug fixed, second bug found, not yet root-caused | Unknown until the divergence bug is understood |
| Attention FP8 | 16.9% | **Dead end** — hardcoded GB200-only gate | N/A |
| CK atomic-fp32 | (subset of 16.9%) | **Dead end** — backward-pass-only, inert for inference | N/A |
| Dense GEMM FP8 (`#55811`/`#56036`) | — | **Dead end** — neither PR is what it claimed to be; `#59069` (already shipped) covers the real version of this idea | N/A |

**The honest math:** of the five candidates examined in detail today, three
turned out to be dead ends on verification, and the fourth (`#54494`) has a
real, partially-fixed bug that still isn't safe to ship. The only lever
left standing with no verified blocker is idle/decode-launch-rate — and it
has no concrete patch attached to it yet, just a measured share of
wall-clock time.

**What would change this assessment:** root-causing `#54494`'s remaining
divergence bug (real GPU-hours needed, likely a longer/heavier smoke test
or closer inspection of the Kimi-K3 model's own per-layer forward code),
or a real profiling pass into what's actually idle during decode (step
granularity, separating host-launch overhead from collective-wait from
KV/admission-wait).

**Immediate next steps, in order:**
1. Decide whether to keep pushing on `#54494`'s divergence bug (needs
   either a heavier smoke test that reproduces sustained KV-offload
   pressure, or direct tracing into the Kimi-K3 model's forward code to
   find where `q_dcp_replicated` gets decided per-step).
2. Fix the smoke-test harness's own `wait` bug (bare `wait` catches the
   backgrounded server process) before relying on it again.
3. Start real profiling on decode idle time (28.2% of wall, the single
   biggest untouched lever) — no concrete patch identified for it yet in
   this campaign.
4. Re-measure `#59591` (now merged unconditionally into today's nightly,
   `8cbd5d0300...`) in isolation — our script has it disabled for a
   measured -1.3% regression, but the PR's own body claims +0.9-30% under
   KV pressure. Worth resolving the discrepancy before any image upgrade.

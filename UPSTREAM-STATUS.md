# UPSTREAM STATUS — what we carry, and what needs upstreaming

**Newest first.** Anything below "HISTORICAL" is kept for provenance only and
was true when written; do not act on it without checking the current section.

---

## CURRENT (T232, 2026-09-02) — RESOLVED: the overlay is not needed

**`kimi-k3-vllm:pronly` — 12 upstream PRs on `nightly-7c5dc571`, zero vendor
patch — delivers 10,692 tok/s/GPU at C72 (err 0.18%, GSM8K 0.99).** That is
+0.6% over the matched-mns overlay run and inside the +/-1.2% replication band.

**The delivery risk recorded below is therefore largely void.** Nothing in the
unpublished 264 KB is worth measurable throughput at C72, so it does not have to
be upstreamed for the number to be reproducible. The five PRs that fail to apply
to `7c5dc571` (#53166, #51437, #53301, #52190, #54163) are optional rather than
blocking.

Caveats: n=1 against a +/-1.2% band, replication in flight; C72 agentic only.
C1 TPOT on pronly is untested and #51437 owns the decode all-reduce overlap, so
C1 is where a gap would appear if one exists.

---

## What we ship now

| image | stack | C72 | GSM8K |
|---|---|--:|--:|
| **`kimi-k3-vllm:pronly`** *(preferred)* | `nightly-7c5dc571` + 12 upstream PRs, **no vendor patch** | **10,692** | 0.99 |
| `kimi-k3-vllm:v4` | `nightly-46638857` + 264 KB overlay + #53940 | 10,607 ±1.2% (n=4) | 0.995 |

`pronly` composition — every piece traceable to a public PR:

| source | PRs |
|---|---|
| merged, already in base | #51705, #53598, #52707, #52033 |
| applied from `pr_only/` | #53917, #51392, #54254, #52494, #52968, #50618, #54165 |
| applied from `pr_stack/` | #53940 (a4w4 flydsl) |

**Optional, not blocking** — five PRs that fail to apply to `7c5dc571` by 1–2
hunks each (context drift, not design conflict): #53166, #51437, #53301, #52190,
#54163. T232 shows they cost nothing at C72. #51437 owns the decode all-reduce
overlap, so C1 is the place to look if a gap exists.

**Live risks:**
1. **Base drift.** `nightly` moves daily (`73029d42…` landed 2026-09-02 05:31).
   Any "nightly + patches" recipe needs a pinned digest and a rebase cadence.
2. **#54165 is closed-unmerged** upstream; its open successor #54163 does not
   apply. One of the twelve is therefore a closed PR.
3. **n=1.** T232 stands against a ±1.2% replication band; T233 is in flight.

---

# THE ACTUAL PR LIST — from Hyukjoon (supersedes my reverse-engineered guesses)

Hyukjoon supplied the manifest. It is **not** in the patch file (I grepped: zero
`#NNNNN` / `pull/` / `Signed-off-by` hits), which is why my earlier
symbol-matching table was partly wrong. That table is superseded by this.

## Shared stack — all arms

| PR | purpose | merged? | in 46638857 (v4 base) | in 7c5dc571 |
|---|---|---|---|---|
| **#51705** | Kimi-K3 DSpark/DCP attention and verification | **yes** 08-31 | **no** | **yes** |
| **#53598** | Per-group hybrid-cache geometry and DCP prefix lookup | **yes** 08-31 | **no** | **yes** |
| **#52707** | Prevent negative external-block allocation | **yes** 08-28 | **no** | **yes** |
| #53917 | Hybrid-cache geometry, mamba replay boundaries, failed-load recovery | open | — | — |
| #52494 | Fuse MLA q/kv RMSNorm | open | — | — |
| #52968 | attn res + sigmoid_mul + conv fusions | open | — | — |
| #53166 | AITER MLA chunked-context gather and KV-index construction | open | — | — |
| #54165 / #54163 | Preserve hybrid-mamba cache hits with DSpark/DFlash + KV connector | open | — | — |
| #50618 | Densify strided activations before ROCm wvSplitKQ (python hunk only) | open | — | — |

## C1-specific

| PR | purpose | merged? |
|---|---|---|
| #51392 | Online quantization on top of a pre-quantized checkpoint | open |
| #54248 | Per-token FP8 input for AITER PTPC linears | open |
| #54254 | Fused KDA gated RMSNorm + per-token-FP8 o_proj | open |

**This confirms the C1/C52 split empirically.** I found by diffing that the `c1`
cut uniquely carries `layers/quantization/online/*`, `config/quantization.py`
and `rmsnorm_gated_fp8_per_token.py` — those are exactly #51392, #54248, #54254.
It also confirms the `ATOM#1752` port note found in the c1 patch belongs to
#54254.

## C16 / C52-specific

| PR | purpose | merged? |
|---|---|---|
| **#52033** | ROCm dual-stream shared-expert (multi-stream forced OFF in the selected runs) | **yes** 08-30 |
| #51437 | Overlap shared all-reduce with routed up-projection, local 880-token guard | open |

## C16 compile / CPRR candidates (not release-clean)

| ref | purpose |
|---|---|
| #52190 | Kimi-K3 torch.compile enablement, safe custom-op boundaries |
| AITER #4521 | Newer MLA dispatcher, ABI, FP8 CPRR runtime base |
| AITER #4964 | GQA/QH32 CPRR kernel dispatch, replayed on #4521 |
| #53301 | Cross-group attention metadata reuse — later align3 candidate |

## What this changes

**Four of the seventeen are merged, and all four are in `7c5dc571` but NOT in
our v4 base `46638857`** — ancestry-checked against each merge commit:

```
#51705  46638857=behind   7c5dc571=ahead
#53598  46638857=behind   7c5dc571=ahead
#52707  46638857=behind   7c5dc571=ahead
#52033  46638857=behind   7c5dc571=ahead
```

So moving the base to `7c5dc571` absorbs four carried patches for free,
including the two DCP-critical ones (#51705, #53598). The remaining thirteen are
still open and would have to be carried either way.

This also revises the upstreaming picture in section 3: the overlay is not
mostly-unpublished work. **Most of it has PRs already open.** The blocker is
review throughput, not authorship — which is a much better position than assumed.

---

# HISTORICAL — superseded, kept for provenance

Everything below predates T232. The reasoning was correct given what was known
at the time; the conclusions about the overlay being load-bearing are not.

---

## 1. PR stack — upstream PRs we apply (`k3_patches/pr_stack/`)

| PR | title | files | status here |
|---|---|--:|---|
| **#53940** | a4w4 flydsl kernels for Kimi-K3 (AMD MoE path) | 4 | **KEPT** — in use; the T195 log shows `flydsl_moe1_abf16_wfp4_bf16_…` on the live MoE path |
| ~~#50813~~ | opt-in K3 SiTUv2 A8W4 routed MoE (`quark_moe.py`) | 1 | **PRUNED** — dead code for this model |

**Why #50813 was pruned.** `quark_moe.py` is only reachable when the model
declares Quark quantization. `moonshotai/Kimi-K3`'s `config.json` has **no**
`quantization_config`, and the MoE path in use is aiter flydsl. `Situv2` in the
kernel name comes from `AITER_SITUV2_A8W4=1` being read by **aiter**, not by
`quark_moe.py`. Retained at `k3_patches/pr_stack_disabled/` for provenance.

### Excluded PRs

| PR | why |
|---|---|
| #54095 | `cudagraph_utils.py` hunk 2 fails at line 362 on this base |
| #53154, #37682 | edit files the overlay rewrites — will not apply on top |
| #50647, #54255 | NVIDIA path |

### Already merged upstream — no longer carried

| PR | evidence it is in the base |
|---|---|
| #51705 DCP for K3 DSpark | `merged=true`; `dcp.py`, `dcp_comm_backend`, `spec_decode/dspark/speculator.py` all present unpatched |
| #52707 kv-blockpool | superseded; `apply_kimi_k3_patches.sh` is skipped via `SKIP_KIMI_PATCHES=1` |

---

---

## 2. Overlay — Hyukjoon's 264 KB patch, **this is what needs upstreaming**

No PR number of its own. Split losslessly into five independently-appliable
groups at `k3_patches/overlay_split/` (recombined = 264,116 B, 199 hunks, 34
files — byte-identical to the monolith).

| grp | scope | files | bytes | closest upstream PR | that PR's state |
|---|---|--:|--:|---|---|
| **A** | DCP a2a buffer pool (`v1/attention/ops/dcp.py`) | 1 | 3,555 | **none** — it is a fix *on top of* merged #51705 | n/a |
| **B** | spec-decode cudagraph (`dflash/cudagraph`, `speculator`, `config/speculative`) | 3 | 4,434 | **none found** | n/a |
| **C** | KV-offload + cache manager (`v1/core/*`, `simple_kv_offload`, `kv_cache_*`) | 9 | 76,944 | #53917, #54457 | **open, unmerged** |
| **D** | ROCm AITER MLA (`backends/mla/*`, `rocm_aiter_mla_reduce`, `mla_attention`) | 5 | 76,806 | #54546 (partial), #54639 | open / **closed unmerged** |
| **E** | Kimi-K3 model path (`models/kimi_k3/**`, MoE runner, `envs`, `platforms/rocm`) | 16 | 102,377 | #54038 | **open, unmerged** |

### Not matched to any public PR

`models/kimi_k3/amd/latent_moe_runner.py`, `v1/attention/ops/rocm_aiter_mla_reduce.py`
(new file), and the `v1/core/sched/scheduler.py` hunks (+116/−65).

### None of the candidates are merged

Checked directly: #53917, #54038, #54457, #54546 are **open**; #54639 and
#53154, #37682 are **closed/open but unmerged**. Consequence: **no newer nightly
contains them**, so moving the base image gains nothing for overlay content and
would only cost us the overlay (it is cut against 46638857 and will not apply
elsewhere).

Dry-run against pristine `nightly-46638857`, all six candidates:

| PR | applies to our base? |
|---|---|
| #54546 | **yes** (1 file, +17/−1) |
| #51705 | no — already merged into the base |
| #53917, #54038, #54457, #54639 | no — cut against much newer bases |

### #54546 is a partial substitute only

| | overlay | #54546 | nightly today |
|---|---|---|---|
| `supports_non_causal_multi_token_dcp` | yes | yes (ROCm-gated) | absent |
| `supports_dcp_with_varlen` | yes | yes | absent |
| **`_cudagraph_support = UNIFORM_BATCH`** | **yes** | **no** | old `…multi_token_decode` name |

The third row is load-bearing: without it the DSpark draft demotes the engine
`FULL_AND_PIECEWISE → PIECEWISE`, measured at **14.05 → 77.65 tok/s**
(ITL 71.16 → 12.88 ms, 5.52×). So #54546 is a supplement, not a replacement.

---

## 3. Suggested upstreaming order

Easiest and most generally useful first.

| # | group | Δlines | note |
|---|---|--:|---|
| 1 | **A** — DCP a2a buffer pool | +48/−10 | self-contained ROCm correctness fix; RCCL bakes buffer addresses into a FULL graph and function-local `torch.empty` can be freed post-capture → aperture violation. Best first PR. |
| 2 | **B** — spec-decode cudagraph | +34/−11 | small, no K3 coupling |
| 3 | **C** — KV-offload + cache manager | +692/−170 | highest value (generic, not K3-specific); overlaps open #53917/#54457, so coordinate rather than duplicate |
| 4 | **D** — ROCm AITER MLA | +1241/−178 | overlaps #54546/#54639 |
| 5 | **E** — Kimi-K3 model path | +1550/−230 | overlaps #54038; depends on D |

**Caveats.** Authorship is Hyukjoon's — nothing should be filed without their
sign-off. The overlay is cut against a July-era nightly, so every new-file hunk
needs re-checking against current `main` before filing.

---

---

## Known limitation (SUPERSEDED by T232) — read this before relying on any number here

**The overlay may never land upstream, and that is a delivery risk.**

Everything in this ledger, including the 10,632 tok/s/GPU peak, depends on a
264 KB patch that:

- has **no PR of its own** and no upstream provenance,
- is authored outside this team (Hyukjoon's), so we cannot unilaterally file it,
- is cut against `nightly-46638857` and applies to **no other base**,
- overlaps several PRs that are all still **open or closed-unmerged**.

If it does not upstream, the consequence is concrete: these numbers are only
reproducible on an image we build and carry ourselves. A stock vLLM release will
not reach them, and "run it with our custom docker" is not a durable answer for
anyone outside this team.

**Current decision: pursue peak performance first, solve distribution second.**
This is deliberate, not an oversight. The ordering is defensible because the
ablation (A/B/C/D/E) reduces the surface that would need upstreaming, and the
measurement work is a prerequisite either way -- you cannot argue for upstreaming
a group without knowing what it is worth.

**What reduces the risk, in order of cost:**

1. Prune. Every group the ablation shows is neutral is a group nobody has to
   upstream. #50813 already fell this way at zero GPU cost.
2. File group A (+48/-10, self-contained ROCm correctness fix) as the first PR,
   with Hyukjoon's sign-off. Small, reviewable, and independent of K3.
3. For groups C and D, coordinate with the existing open PRs (#53917, #54457,
   #54546, #54639) rather than filing competing patches.
4. Accept that group E (the Kimi-K3 AMD model path) is the hardest and may
   remain vendor-carried for some time.

---


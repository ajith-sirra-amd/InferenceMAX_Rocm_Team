# UPSTREAM STATUS — what we carry, and what needs upstreaming

**Newest first.** Everything below "HISTORICAL" is provenance only.

*Tested and produced, not shipped* — `pronly` is a local image validated on
b23_07, not published to a registry.

---

# WHAT WE TESTED & PRODUCED — `kimi-k3-vllm:pronly`

`nightly-7c5dc571` + **11 upstream PRs** (4 merged, 7 open). No vendor patch.
**10,692 tok/s/GPU @ C72**, err 0.18%, GSM8K 0.99 (T232).

## PRs applied — 11 total

### Already merged upstream (in the base image)

| PR | what it does |
|---|---|
| **[#51705](https://github.com/vllm-project/vllm/pull/51705)** | causal multi-token verification under DCP |
| **[#53598](https://github.com/vllm-project/vllm/pull/53598)** | per-group DCP cache geometry, prefix-cache hits |
| **[#52707](https://github.com/vllm-project/vllm/pull/52707)** | clamp negative external block allocation |
| **[#52033](https://github.com/vllm-project/vllm/pull/52033)** | ROCm dual-stream shared-expert |

### Still open — we apply these, they must merge for this to be stock

| PR | what it does | applied as | prune status |
|---|---|---|---|
| **[#53917](https://github.com/vllm-project/vllm/pull/53917)** | `SimpleCPUOffloadConnector` under DCP | `pr_only/01` | keep — test last |
| **[#51392](https://github.com/vllm-project/vllm/pull/51392)** | online quant on pre-quantized checkpoint | `pr_only/02` | **GSM8K 0.995 without it (T234)** — throughput pending |
| **[#54254](https://github.com/vllm-project/vllm/pull/54254)** | fused KDA gated RMSNorm + per-token FP8 | `pr_only/03` | **same pair as #51392** |
| **[#52494](https://github.com/vllm-project/vllm/pull/52494)** | fuse MLA q/kv RMSNorm | `pr_only/04` | untested |
| **[#52968](https://github.com/vllm-project/vllm/pull/52968)** | attn_res + sigmoid_mul + conv fusions | `pr_only/05` | untested |
| **[#50618](https://github.com/vllm-project/vllm/pull/50618)** | densify strided activations for wvSplitKQ | `pr_only/06` | untested |
| **[#53940](https://github.com/vllm-project/vllm/pull/53940)** | a4w4 flydsl MoE kernels | `pr_stack/` (4 files) | held constant |

**7 open PRs are the entire upstreaming ask.** Nothing closed, nothing vendor.
**#51392 + #54254 may drop to 5** — T234 passed GSM8K at 0.995 without them;
T235 is measuring throughput.

Apply gotchas: **#54254 is stacked on #54248** — its diff already contains
#54248, so applying both double-applies and fails; #54254 alone gets the whole
per-token-FP8 chain. **#50618 is python-hunk only**; its `csrc/*.cu` hunks cannot
apply to a prebuilt image.

## Measured results

| trial | image | what | result |
|---|---|---|--:|
| T231 | pronly | GSM8K | 0.99 |
| T232 | pronly | C72 throughput | **10,692** |
| T233 | pronly | C72 replication | **10,690** (0.02% spread) |
| T234 | pronly-noquant | GSM8K | **0.995** |
| T235 | pronly-noquant | C72 throughput | *running* |

**Headline: 10,691 tok/s/GPU at C72 (n=2)** — statistically identical to the
264 KB overlay it replaces (v4: 10,607 ±1.2%, n=4; matched-mns T198: 10,630).

Noise is **not uniform**: same-session pairs replicate to 0.02% (T232/T233,
T195/T198), but cross-day byte-identical runs differ by 1.2% (T206 vs T228).
Quote **±1.2%** for cross-day comparisons; 0.02% is not the general error bar.

## Base-bump candidate — `nightly-73029d42…`

Three commits landed after our base `7c5dc571` and are in
`nightly-73029d42441321b631779db3475031f5ec26dd6c`
(`sha256:cef549da00e0efaeadd9338ac8f351f2b96ff71a5ab8651a99bf989458bf1684`,
cut 2026-09-02 05:31), ancestry-verified:

| PR | what | touches |
|---|---|---|
| **[#53388](https://github.com/vllm-project/vllm/pull/53388)** | disable trailing prefix-cache block dropping | `sched/scheduler.py`, `single_type_kv_cache_manager.py`, `kv_cache_utils.py`, `simple_kv_offload/manager.py` |
| **[#52832](https://github.com/vllm-project/vllm/pull/52832)** | Mooncake: offload producer partial tails on finish | `sched/scheduler.py`, `kv_cache_manager.py` |
| — | fix eager SimpleCPUOffload cache registration | `simple_kv_offload/manager.py` (09-02) |

#53388 is on the prefix-cache-retention path, which is where we measured a
93.6% → 90.0% hit-rate drop between the aigmkt stack and bare nightly.

**Not taken yet** — it would invalidate the n=2 pronly baseline and needs a
fresh GSM8K gate. Finish the prune first; do not change two things at once.

**Still orphaned regardless of base:** `v1/core/block_pool.py` and
`v1/core/kv_cache_interface.py` — no PR touches them, and no upstream commits
since 08-31.

## Risks

1. **Base drift.** `nightly` moves daily (`73029d42…` landed 2026-09-02 05:31).
   Pin a digest and set a rebase cadence.
3. **n=1** against a ±1.2% band. T233 replication in flight.
4. **C72 agentic only.** C1 TPOT on pronly untested.

---

# OPTIONAL — not applied, not needed for 10,692

Five PRs fail on `7c5dc571` by 1–2 hunks each. Context drift, not design
conflict — each is a rebase, not new engineering. **T232 shows they cost nothing
at C72.**

| PR | what it does | fails | worth rebasing? |
|---|---|---|---|
| **[#51437](https://github.com/vllm-project/vllm/pull/51437)** | overlap shared all-reduce with routed up-proj, decode batches | 1 of 6 hunks | **yes, for C1** — owns decode |
| **[#53166](https://github.com/vllm-project/vllm/pull/53166)** | fuse MLA prefill chunked-context gather, 4 kernels → 1 | 1 of 8 | maybe — prefill path |
| **[#53301](https://github.com/vllm-project/vllm/pull/53301)** | reuse attn metadata across 6 MLA + 14 KDA groups | 2 | maybe — per-step overhead |
| **[#52190](https://github.com/vllm-project/vllm/pull/52190)** | torch.compile enablement so fusion passes run | 1 of 1 | low — not release-clean |
| **[#54163](https://github.com/vllm-project/vllm/pull/54163)** | DFlash/DSpark last prefix-cache block | 1 of 1 | low — spec off at C72 |
| **[#54165](https://github.com/vllm-project/vllm/pull/54165)** | mamba align cache under spec decode | *dropped* | **no** — closed-unmerged, superseded by [#54163](https://github.com/vllm-project/vllm/pull/54163) |

If C1 TPOT on pronly comes in worse than v4's 9.06 ms, **[#51437](https://github.com/vllm-project/vllm/pull/51437) is the first
one to rebase.**

## Dropped for cause

| PR | why |
|---|---|
| **[#50813](https://github.com/vllm-project/vllm/pull/50813)** | dead code — `moonshotai/Kimi-K3` declares no `quantization_config`, so `quark_moe.py` is unreachable. `Situv2` in the kernel name comes from `AITER_SITUV2_A8W4=1` read by **aiter**, not by `quark_moe.py`. |
| **[#54248](https://github.com/vllm-project/vllm/pull/54248)** | superseded — [#54254](https://github.com/vllm-project/vllm/pull/54254) is stacked on it and contains it. Applying both fails. |
| **[#54546](https://github.com/vllm-project/vllm/pull/54546)** | partial substitute for overlay group D; missing `_cudagraph_support = UNIFORM_BATCH`, without which the DSpark draft demotes the engine FULL_AND_PIECEWISE → PIECEWISE (14.05 → 77.65 tok/s). Moot now — pronly does not need group D. |
| [#54095](https://github.com/vllm-project/vllm/pull/54095), [#53154](https://github.com/vllm-project/vllm/pull/53154), [#37682](https://github.com/vllm-project/vllm/pull/37682), [#50647](https://github.com/vllm-project/vllm/pull/50647), [#54255](https://github.com/vllm-project/vllm/pull/54255), [#54038](https://github.com/vllm-project/vllm/pull/54038), [#54457](https://github.com/vllm-project/vllm/pull/54457), [#54639](https://github.com/vllm-project/vllm/pull/54639) | do not apply, NVIDIA-only, or superseded by the pronly result |

---

## The overlay is no longer needed

Hyukjoon's 264 KB patch bought **nothing measurable** at C72: pronly is +0.6%
over the matched-mns overlay run (T198, 10,630), inside the band. The
upstreaming risk that dominated this file is void.

---

# THE ACTUAL PR LIST — from Hyukjoon (supersedes my reverse-engineered guesses)

Hyukjoon supplied the manifest. It is **not** in the patch file (I grepped: zero
`#NNNNN` / `pull/` / `Signed-off-by` hits), which is why my earlier
symbol-matching table was partly wrong. That table is superseded by this.

## Shared stack — all arms

| PR | purpose | merged? | in 46638857 (v4 base) | in 7c5dc571 |
|---|---|---|---|---|
| **[#51705](https://github.com/vllm-project/vllm/pull/51705)** | Kimi-K3 DSpark/DCP attention and verification | **yes** 08-31 | **no** | **yes** |
| **[#53598](https://github.com/vllm-project/vllm/pull/53598)** | Per-group hybrid-cache geometry and DCP prefix lookup | **yes** 08-31 | **no** | **yes** |
| **[#52707](https://github.com/vllm-project/vllm/pull/52707)** | Prevent negative external-block allocation | **yes** 08-28 | **no** | **yes** |
| [#53917](https://github.com/vllm-project/vllm/pull/53917) | Hybrid-cache geometry, mamba replay boundaries, failed-load recovery | open | — | — |
| [#52494](https://github.com/vllm-project/vllm/pull/52494) | Fuse MLA q/kv RMSNorm | open | — | — |
| [#52968](https://github.com/vllm-project/vllm/pull/52968) | attn res + sigmoid_mul + conv fusions | open | — | — |
| [#53166](https://github.com/vllm-project/vllm/pull/53166) | AITER MLA chunked-context gather and KV-index construction | open | — | — |
| [#54165](https://github.com/vllm-project/vllm/pull/54165) / [#54163](https://github.com/vllm-project/vllm/pull/54163) | Preserve hybrid-mamba cache hits with DSpark/DFlash + KV connector | open | — | — |
| [#50618](https://github.com/vllm-project/vllm/pull/50618) | Densify strided activations before ROCm wvSplitKQ (python hunk only) | open | — | — |

## C1-specific

| PR | purpose | merged? |
|---|---|---|
| [#51392](https://github.com/vllm-project/vllm/pull/51392) | Online quantization on top of a pre-quantized checkpoint | open |
| [#54248](https://github.com/vllm-project/vllm/pull/54248) | Per-token FP8 input for AITER PTPC linears | open |
| [#54254](https://github.com/vllm-project/vllm/pull/54254) | Fused KDA gated RMSNorm + per-token-FP8 o_proj | open |

**This confirms the C1/C52 split empirically.** I found by diffing that the `c1`
cut uniquely carries `layers/quantization/online/*`, `config/quantization.py`
and `rmsnorm_gated_fp8_per_token.py` — those are exactly [#51392](https://github.com/vllm-project/vllm/pull/51392), [#54248](https://github.com/vllm-project/vllm/pull/54248), [#54254](https://github.com/vllm-project/vllm/pull/54254).
It also confirms the `ATOM#1752` port note found in the c1 patch belongs to
[#54254](https://github.com/vllm-project/vllm/pull/54254).

## C16 / C52-specific

| PR | purpose | merged? |
|---|---|---|
| **[#52033](https://github.com/vllm-project/vllm/pull/52033)** | ROCm dual-stream shared-expert (multi-stream forced OFF in the selected runs) | **yes** 08-30 |
| [#51437](https://github.com/vllm-project/vllm/pull/51437) | Overlap shared all-reduce with routed up-projection, local 880-token guard | open |

## C16 compile / CPRR candidates (not release-clean)

| ref | purpose |
|---|---|
| [#52190](https://github.com/vllm-project/vllm/pull/52190) | Kimi-K3 torch.compile enablement, safe custom-op boundaries |
| AITER #4521 | Newer MLA dispatcher, ABI, FP8 CPRR runtime base |
| AITER #4964 | GQA/QH32 CPRR kernel dispatch, replayed on #4521 |
| [#53301](https://github.com/vllm-project/vllm/pull/53301) | Cross-group attention metadata reuse — later align3 candidate |

## What this changes

**Four of the seventeen are merged, and all four are in `7c5dc571` but NOT in
our v4 base `46638857`** — ancestry-checked against each merge commit:

```
[#51705](https://github.com/vllm-project/vllm/pull/51705)  46638857=behind   7c5dc571=ahead
[#53598](https://github.com/vllm-project/vllm/pull/53598)  46638857=behind   7c5dc571=ahead
[#52707](https://github.com/vllm-project/vllm/pull/52707)  46638857=behind   7c5dc571=ahead
[#52033](https://github.com/vllm-project/vllm/pull/52033)  46638857=behind   7c5dc571=ahead
```

So moving the base to `7c5dc571` absorbs four carried patches for free,
including the two DCP-critical ones ([#51705](https://github.com/vllm-project/vllm/pull/51705), [#53598](https://github.com/vllm-project/vllm/pull/53598)). The remaining thirteen are
still open and would have to be carried either way.

This also revises the upstreaming picture in section 3: the overlay is not
mostly-unpublished work. **Most of it has PRs already open.** The blocker is
review throughput, not authorship — which is a much better position than assumed.

---

# HISTORICAL — superseded, kept for provenance

## `kimi-k3-vllm:v4` — the overlay image, superseded by pronly

`nightly-46638857` + Hyukjoon's 264 KB overlay + [#53940](https://github.com/vllm-project/vllm/pull/53940). **10,607 ±1.2% (n=4)**
at C72, GSM8K 0.995. Pushed as `aigmkt/kimi-k3-vllm:v4`
(`sha256:88c8438f5aa0fc2fa01ee1736eb0f8a88e478b26a93a733f535b4f964bb197f2`).

Still the reference for C1 (TPOT 9.06 ms) until pronly is measured there.
Otherwise prefer pronly: same throughput, no vendor patch, reproducible outside
this team.

---

Everything below predates T232. The reasoning was correct given what was known
at the time; the conclusions about the overlay being load-bearing are not.

---

## 1. PR stack — upstream PRs we apply (`k3_patches/pr_stack/`)

| PR | title | files | status here |
|---|---|--:|---|
| **[#53940](https://github.com/vllm-project/vllm/pull/53940)** | a4w4 flydsl kernels for Kimi-K3 (AMD MoE path) | 4 | **KEPT** — in use; the T195 log shows `flydsl_moe1_abf16_wfp4_bf16_…` on the live MoE path |
| ~~[#50813](https://github.com/vllm-project/vllm/pull/50813)~~ | opt-in K3 SiTUv2 A8W4 routed MoE (`quark_moe.py`) | 1 | **PRUNED** — dead code for this model |

**Why [#50813](https://github.com/vllm-project/vllm/pull/50813) was pruned.** `quark_moe.py` is only reachable when the model
declares Quark quantization. `moonshotai/Kimi-K3`'s `config.json` has **no**
`quantization_config`, and the MoE path in use is aiter flydsl. `Situv2` in the
kernel name comes from `AITER_SITUV2_A8W4=1` being read by **aiter**, not by
`quark_moe.py`. Retained at `k3_patches/pr_stack_disabled/` for provenance.

### Excluded PRs

| PR | why |
|---|---|
| [#54095](https://github.com/vllm-project/vllm/pull/54095) | `cudagraph_utils.py` hunk 2 fails at line 362 on this base |
| [#53154](https://github.com/vllm-project/vllm/pull/53154), [#37682](https://github.com/vllm-project/vllm/pull/37682) | edit files the overlay rewrites — will not apply on top |
| [#50647](https://github.com/vllm-project/vllm/pull/50647), [#54255](https://github.com/vllm-project/vllm/pull/54255) | NVIDIA path |

### Already merged upstream — no longer carried

| PR | evidence it is in the base |
|---|---|
| [#51705](https://github.com/vllm-project/vllm/pull/51705) DCP for K3 DSpark | `merged=true`; `dcp.py`, `dcp_comm_backend`, `spec_decode/dspark/speculator.py` all present unpatched |
| [#52707](https://github.com/vllm-project/vllm/pull/52707) kv-blockpool | superseded; `apply_kimi_k3_patches.sh` is skipped via `SKIP_KIMI_PATCHES=1` |

---

---

## 2. Overlay — Hyukjoon's 264 KB patch, **this is what needs upstreaming**

No PR number of its own. Split losslessly into five independently-appliable
groups at `k3_patches/overlay_split/` (recombined = 264,116 B, 199 hunks, 34
files — byte-identical to the monolith).

| grp | scope | files | bytes | closest upstream PR | that PR's state |
|---|---|--:|--:|---|---|
| **A** | DCP a2a buffer pool (`v1/attention/ops/dcp.py`) | 1 | 3,555 | **none** — it is a fix *on top of* merged [#51705](https://github.com/vllm-project/vllm/pull/51705) | n/a |
| **B** | spec-decode cudagraph (`dflash/cudagraph`, `speculator`, `config/speculative`) | 3 | 4,434 | **none found** | n/a |
| **C** | KV-offload + cache manager (`v1/core/*`, `simple_kv_offload`, `kv_cache_*`) | 9 | 76,944 | [#53917](https://github.com/vllm-project/vllm/pull/53917), [#54457](https://github.com/vllm-project/vllm/pull/54457) | **open, unmerged** |
| **D** | ROCm AITER MLA (`backends/mla/*`, `rocm_aiter_mla_reduce`, `mla_attention`) | 5 | 76,806 | [#54546](https://github.com/vllm-project/vllm/pull/54546) (partial), [#54639](https://github.com/vllm-project/vllm/pull/54639) | open / **closed unmerged** |
| **E** | Kimi-K3 model path (`models/kimi_k3/**`, MoE runner, `envs`, `platforms/rocm`) | 16 | 102,377 | [#54038](https://github.com/vllm-project/vllm/pull/54038) | **open, unmerged** |

### Not matched to any public PR

`models/kimi_k3/amd/latent_moe_runner.py`, `v1/attention/ops/rocm_aiter_mla_reduce.py`
(new file), and the `v1/core/sched/scheduler.py` hunks (+116/−65).

### None of the candidates are merged

Checked directly: [#53917](https://github.com/vllm-project/vllm/pull/53917), [#54038](https://github.com/vllm-project/vllm/pull/54038), [#54457](https://github.com/vllm-project/vllm/pull/54457), [#54546](https://github.com/vllm-project/vllm/pull/54546) are **open**; [#54639](https://github.com/vllm-project/vllm/pull/54639) and
[#53154](https://github.com/vllm-project/vllm/pull/53154), [#37682](https://github.com/vllm-project/vllm/pull/37682) are **closed/open but unmerged**. Consequence: **no newer nightly
contains them**, so moving the base image gains nothing for overlay content and
would only cost us the overlay (it is cut against 46638857 and will not apply
elsewhere).

Dry-run against pristine `nightly-46638857`, all six candidates:

| PR | applies to our base? |
|---|---|
| [#54546](https://github.com/vllm-project/vllm/pull/54546) | **yes** (1 file, +17/−1) |
| [#51705](https://github.com/vllm-project/vllm/pull/51705) | no — already merged into the base |
| [#53917](https://github.com/vllm-project/vllm/pull/53917), [#54038](https://github.com/vllm-project/vllm/pull/54038), [#54457](https://github.com/vllm-project/vllm/pull/54457), [#54639](https://github.com/vllm-project/vllm/pull/54639) | no — cut against much newer bases |

### [#54546](https://github.com/vllm-project/vllm/pull/54546) is a partial substitute only

| | overlay | [#54546](https://github.com/vllm-project/vllm/pull/54546) | nightly today |
|---|---|---|---|
| `supports_non_causal_multi_token_dcp` | yes | yes (ROCm-gated) | absent |
| `supports_dcp_with_varlen` | yes | yes | absent |
| **`_cudagraph_support = UNIFORM_BATCH`** | **yes** | **no** | old `…multi_token_decode` name |

The third row is load-bearing: without it the DSpark draft demotes the engine
`FULL_AND_PIECEWISE → PIECEWISE`, measured at **14.05 → 77.65 tok/s**
(ITL 71.16 → 12.88 ms, 5.52×). So [#54546](https://github.com/vllm-project/vllm/pull/54546) is a supplement, not a replacement.

---

## 3. Suggested upstreaming order

Easiest and most generally useful first.

| # | group | Δlines | note |
|---|---|--:|---|
| 1 | **A** — DCP a2a buffer pool | +48/−10 | self-contained ROCm correctness fix; RCCL bakes buffer addresses into a FULL graph and function-local `torch.empty` can be freed post-capture → aperture violation. Best first PR. |
| 2 | **B** — spec-decode cudagraph | +34/−11 | small, no K3 coupling |
| 3 | **C** — KV-offload + cache manager | +692/−170 | highest value (generic, not K3-specific); overlaps open [#53917](https://github.com/vllm-project/vllm/pull/53917)/[#54457](https://github.com/vllm-project/vllm/pull/54457), so coordinate rather than duplicate |
| 4 | **D** — ROCm AITER MLA | +1241/−178 | overlaps [#54546](https://github.com/vllm-project/vllm/pull/54546)/[#54639](https://github.com/vllm-project/vllm/pull/54639) |
| 5 | **E** — Kimi-K3 model path | +1550/−230 | overlaps [#54038](https://github.com/vllm-project/vllm/pull/54038); depends on D |

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
   upstream. [#50813](https://github.com/vllm-project/vllm/pull/50813) already fell this way at zero GPU cost.
2. File group A (+48/-10, self-contained ROCm correctness fix) as the first PR,
   with Hyukjoon's sign-off. Small, reviewable, and independent of K3.
3. For groups C and D, coordinate with the existing open PRs ([#53917](https://github.com/vllm-project/vllm/pull/53917), [#54457](https://github.com/vllm-project/vllm/pull/54457),
   [#54546](https://github.com/vllm-project/vllm/pull/54546), [#54639](https://github.com/vllm-project/vllm/pull/54639)) rather than filing competing patches.
4. Accept that group E (the Kimi-K3 AMD model path) is the hardest and may
   remain vendor-carried for some time.

---


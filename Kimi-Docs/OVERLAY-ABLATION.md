# OVERLAY ABLATION — what in Hyukjoon's 264 KB patch is actually droppable

Two independent decompositions of the same overlay
(`vllm_nightly_46638857_k3_c16_c52_current.patch`, 264,116 B, 199 hunks,
34 files), both verified lossless before any GPU time was spent.

| axis | grouping | status |
|---|---|---|
| **1. File-group (A–E)** | by subsystem | **measured, closed** |
| **2. PR bucket** | by Hyukjoon's PR manifest | **split built + verified; runs pending** |

Everything below is measured on `kimi-k3-vllm:v4` (all layers baked, no runtime
patching) at C72 unless stated.

---

## Axis 1 — file-group ablation A/B/C/D/E (CLOSED)

Leave-one-out. Each arm applies four of five groups; the control applies all
five through the same split machinery, so the machinery itself is validated.

| trial | arm | tok/s/GPU | verdict |
|---|---|--:|---|
| T221 | ABCDE (control) | 10,756 | split path == monolith |
| T222 | **A-out** (BCDE) | 10,719 | **A detachable, neutral** |
| T223 | **B-out** (ACDE) | 10,747 | **B detachable, inert** |
| T224 | C-out (ABDE) | **fails to start** | C coupled to E |
| T225 | D-out (ABCE) | **fails** | D required under DCP |
| T226 | E-out (ABCD) | **fails** | E required |

| grp | scope | files | bytes |
|---|---|--:|--:|
| A | DCP a2a buffer pool (`v1/attention/ops/dcp.py`) | 1 | 3,555 |
| B | spec-decode cudagraph (`dflash/cudagraph`, `speculator`, `config/speculative`) | 3 | 4,434 |
| C | KV-offload + cache manager (`v1/core/*`, `simple_kv_offload`) | 9 | 76,944 |
| D | ROCm AITER MLA (`backends/mla/*`, `mla_attention`) | 5 | 76,806 |
| E | Kimi-K3 model path (`models/kimi_k3/**`, MoE runner, `envs`, `platforms/rocm`) | 16 | 102,377 |

**Result: the overlay is one coupled 256 KB unit (C+D+E) plus two small
detachable fixes (A+B, 8 KB combined).** Nothing meaningful to prune.

Two caveats that cost runs to learn:

1. **Patch-independence is not import-independence.** All five groups apply
   cleanly in isolation — that is what "lossless split" means. It does not make
   every subset a legal *experiment*. C/D/E-out fail on imports, not on hunks.
   T224/T225/T226 are three runs spent proving that.
2. **A and B measured neutral at C72 only.** A is a correctness fix (RCCL bakes
   buffer addresses into a FULL graph; a function-local `torch.empty` can be
   freed post-capture → aperture violation). Neutral-for-throughput is not
   safe-to-drop.

---

## Axis 2 — PR-bucket split (BUILT, VERIFIED, RUNS PENDING)

Hyukjoon's manifest lists 17 PRs mixed into the overlay (18 with [#53940](https://github.com/vllm-project/vllm/pull/53940), which rides in the separate pr_stack). PR boundaries are what
upstream reviews, so they are the more useful axis — but they do **not**
partition the patch.

### Coverage of the overlay by the PR list

| | hunks | % | meaning |
|---|--:|--:|---|
| Uniquely owned — file touched by exactly 1 PR | 49 | **25%** | cleanly detachable |
| Contested — file touched by 2–4 PRs | 124 | **62%** | needs hunk-level attribution |
| No PR at all | 26 | **13%** | nothing to detach *to* |

Contested files are the big ones: `rocm_aiter_mla.py` (38 hunks / 3 PRs),
`single_type_kv_cache_manager.py` (25 / 4), `kda.py` (14 / 4),
`kv_cache_coordinator.py` (11 / 2), `linear.py` (10 / 3), `sched/scheduler.py`
(9 / 3).

Unmatched by any PR — 9 files plus 4 new-file hunks: `dcp.py`, `block_pool.py`,
`kv_cache_interface.py`, `triton_mla.py`, `speculator.py`, `dflash/cudagraph.py`,
`kimi_gdn_linear_attn.py`, `kda/chunk.py`, `chunk_delta_h.py`.

### The split — `k3_patches/pr_split/`

Recombination verified **byte-identical**: 34 sections, sorted-section SHA256
`d8b4cdced51b1354…` on both sides, 264,116 B total.

| bucket | PR | files | hunks | bytes |
|---|---|--:|--:|--:|
| REST | contested + unmatched | 22 | 149 | 212,078 |
| P53917 | KV-offload / cache mgr | 2 | 25 | 23,701 |
| P53301 | `kda_metadata.py` | 1 | 3 | 6,679 |
| P52033 | `shared_experts.py` | 1 | 6 | 6,199 |
| P52494 | `kimi_k3/amd/mla.py` (new file) | 1 | 1 | 4,584 |
| P51392 | `nvidia/mla.py` | 1 | 5 | 3,867 |
| P52968 | `envs.py`, `layers/mla.py` | 2 | 5 | 2,569 |
| P53598 | `kv_cache_utils.py` | 1 | 2 | 1,719 |
| P51705 | `platforms/rocm.py` | 1 | 1 | 1,183 |
| P50618 | `scaled_mm/rocm.py` | 1 | 1 | 777 |
| P54165 | `offloading/scheduler.py` | 1 | 1 | 760 |

Ten detachable buckets, 52,038 B — **20% of the overlay**. The other 80% is REST.

### Why we slice the overlay and not rebuild from PRs

Rebuilding the stack by applying the 17 PRs to a pristine base **will not
build**, and this was checked rather than assumed: dry-run against pristine
`nightly-46638857` applied only [#54546](https://github.com/vllm-project/vllm/pull/54546). [#53917](https://github.com/vllm-project/vllm/pull/53917), [#54038](https://github.com/vllm-project/vllm/pull/54038), [#54457](https://github.com/vllm-project/vllm/pull/54457) and [#54639](https://github.com/vllm-project/vllm/pull/54639) all
failed — they are cut against much newer bases. The PRs also collide with each
other (four touch `kda.py`, four touch `single_type_kv_cache_manager.py`).

Slicing the overlay sidesteps this: every arm is overlay-derived, so it applies
by construction.

---

## Correction: it is 18 PRs, not 17

Hyukjoon's manifest lists 17. The stack also carries **[#53940](https://github.com/vllm-project/vllm/pull/53940) (a4w4 flydsl
kernels for Kimi-K3)**, which is not in the overlay at all — it is the 4-file
`k3_patches/pr_stack/`, applied as a separate layer. It belongs in every list
below. Evidence it is live: the T195 server log shows
`flydsl_moe1_abf16_wfp4_bf16_…` on the MoE path.

A nineteenth, **[#50813](https://github.com/vllm-project/vllm/pull/50813)** (opt-in K3 SiTUv2 A8W4 routed MoE), was **already
pruned** at zero GPU cost: `quark_moe.py` is only reachable when the model
declares Quark quantization, and `moonshotai/Kimi-K3` has no
`quantization_config`. The `Situv2` in the kernel name comes from
`AITER_SITUV2_A8W4=1` being read by **aiter**, not by `quark_moe.py`. v4 ships a
4-file stack instead of 5; T206 and T228 confirm no regression. Retained at
`k3_patches/pr_stack_disabled/` for provenance.

---

## How much of each PR is actually present in this overlay

Measured, not assumed — each PR's `vllm/` source files intersected against the
overlay's 34 target paths.

| PR | src files | in overlay | coverage |
|---|--:|--:|---|
| [#53598](https://github.com/vllm-project/vllm/pull/53598), [#52707](https://github.com/vllm-project/vllm/pull/52707), [#52494](https://github.com/vllm-project/vllm/pull/52494), [#52968](https://github.com/vllm-project/vllm/pull/52968), [#53166](https://github.com/vllm-project/vllm/pull/53166), [#54165](https://github.com/vllm-project/vllm/pull/54165), [#54163](https://github.com/vllm-project/vllm/pull/54163), [#50618](https://github.com/vllm-project/vllm/pull/50618), [#52033](https://github.com/vllm-project/vllm/pull/52033), [#51437](https://github.com/vllm-project/vllm/pull/51437) | — | — | **FULL** |
| [#51705](https://github.com/vllm-project/vllm/pull/51705) | 3 | 2 | 66% partial |
| [#52190](https://github.com/vllm-project/vllm/pull/52190) | 5 | 3 | 60% partial |
| [#53917](https://github.com/vllm-project/vllm/pull/53917) | 10 | 5 | 50% partial |
| [#53301](https://github.com/vllm-project/vllm/pull/53301) | 6 | 3 | 50% partial |
| [#54254](https://github.com/vllm-project/vllm/pull/54254) | 4 | 1 | 25% partial |
| [#51392](https://github.com/vllm-project/vllm/pull/51392) | 20 | 2 | **10% partial** |
| [#54248](https://github.com/vllm-project/vllm/pull/54248) | 3 | 0 | **absent** |

This is file-level, so a "FULL" PR may still share files with others — but the
partials and the absence are solid, and two of them change the weighting:

- **[#52190](https://github.com/vllm-project/vllm/pull/52190) does not include `vllm/config/compilation.py`.** That file is the
  actual torch.compile enablement. Without it K3 still carries no
  `@support_torch_compile`, so **no fusion pass runs** and the three model files
  that did land are inert. [#52190](https://github.com/vllm-project/vllm/pull/52190)'s headline benefit is not in this image.
- **[#54248](https://github.com/vllm-project/vllm/pull/54248) is entirely absent and [#54254](https://github.com/vllm-project/vllm/pull/54254) is 25% present.** [#54254](https://github.com/vllm-project/vllm/pull/54254)'s own body
  says it is a no-op until [#54248](https://github.com/vllm-project/vllm/pull/54248) *and* [#51392](https://github.com/vllm-project/vllm/pull/51392) land. [#51392](https://github.com/vllm-project/vllm/pull/51392) is 10% present.
  The whole C1 per-token-FP8 chain is dead here.

---

## Weighing the 18 for goal A (reproduce 10,607 at C72)

Our C72 operating point: DCP 8, **spec decoding OFF**, offload dram,
**multi-stream OFF**, prefix caching on, eager (no torch.compile).

### Tier 0 — required to run. Cannot prune.

| PR | why |
|---|---|
| **[#53940](https://github.com/vllm-project/vllm/pull/53940)** | a4w4 flydsl on the live MoE path (separate pr_stack, not overlay) |
| [#53598](https://github.com/vllm-project/vllm/pull/53598) | per-group DCP geometry → prefix-cache hits under DCP; we measure 73.5% hit |
| [#52707](https://github.com/vllm-project/vllm/pull/52707) | clamps negative external block allocation — crash guard |
| [#53917](https://github.com/vllm-project/vllm/pull/53917) | `SimpleCPUOffloadConnector` under DCP; we run offload dram |
| [#51705](https://github.com/vllm-project/vllm/pull/51705) | DCP attention support / backend registration |

### Tier 1 — perf-bearing at C72. Keep.

| PR | weight | why |
|---|---|---|
| [#53166](https://github.com/vllm-project/vllm/pull/53166) | **highest** | 4 kernels → 1 per MLA prefill context chunk; agentic ISL p50 ≈ 87k tokens, so this path dominates |
| [#51437](https://github.com/vllm-project/vllm/pull/51437) | high | overlaps shared all-reduce with routed up-projection at decode batch sizes |
| [#52968](https://github.com/vllm-project/vllm/pull/52968) | medium | attn_res / sigmoid_mul / conv1d fusions |
| [#52494](https://github.com/vllm-project/vllm/pull/52494) | low–med | fuses MLA q/kv RMSNorm into one AITER launch |
| [#53301](https://github.com/vllm-project/vllm/pull/53301) | reduced | only 50% present, so at most partial benefit |

### Tier 2 — prunable for goal A. Inert at this operating point.

| PR | why it is inert here |
|---|---|
| [#54165](https://github.com/vllm-project/vllm/pull/54165), [#54163](https://github.com/vllm-project/vllm/pull/54163) | spec-decode paths; **spec is off at C72** |
| [#52033](https://github.com/vllm-project/vllm/pull/52033) | dual-stream decode; **multi-stream forced OFF** in our config |
| [#51392](https://github.com/vllm-project/vllm/pull/51392) | 10% present — only `nvidia/mla.py`, dead on AMD |
| [#54254](https://github.com/vllm-project/vllm/pull/54254) | 25% present, and a self-declared no-op without [#54248](https://github.com/vllm-project/vllm/pull/54248) (absent) |
| [#52190](https://github.com/vllm-project/vllm/pull/52190) | the enablement file is missing; remaining hunks do nothing |
| [#50618](https://github.com/vllm-project/vllm/pull/50618) | correctness guard (KDA `f_b_proj` over-reads 12 KB at TP8) — **zero perf weight, but keep**; dropping it risks memory faults |

**Six of eighteen are inert at C72**, leaving twelve that plausibly matter.

**These are reasoned, not measured.** Only [#50813](https://github.com/vllm-project/vllm/pull/50813) has been confirmed prunable by
experiment. `k3_patches/pr_split/` can test three of the six directly
([#51392](https://github.com/vllm-project/vllm/pull/51392), [#54165](https://github.com/vllm-project/vllm/pull/54165), [#52033](https://github.com/vllm-project/vllm/pull/52033) have their own buckets); [#54163](https://github.com/vllm-project/vllm/pull/54163), [#52190](https://github.com/vllm-project/vllm/pull/52190) and [#54254](https://github.com/vllm-project/vllm/pull/54254) sit
in contested files inside REST and cannot be detached without hunk-level work.

---

## Reading the 18 PRs — perf-bearing vs. required-to-run

Sorted by whether dropping it could plausibly move a number.

| PR | what it does | perf impact |
|---|---|---|
| [#53166](https://github.com/vllm-project/vllm/pull/53166) | 4 kernels → 1 per context chunk in MLA prefill (`gather_kv_b_proj`) | **high** — agentic ISL p50 is ~87k tokens, prefill-dominated |
| [#51437](https://github.com/vllm-project/vllm/pull/51437) | overlap shared all-reduce with routed up-projection at decode batch sizes | **high** |
| [#53301](https://github.com/vllm-project/vllm/pull/53301) | build attention metadata once, not per group (TP8 K3 = 6 MLA + 14 KDA groups) | **high** — per-step overhead |
| [#52190](https://github.com/vllm-project/vllm/pull/52190) | K3 carries no `@support_torch_compile`, so **no fusion pass ever runs** | **high**, not release-clean |
| [#52968](https://github.com/vllm-project/vllm/pull/52968) | attn_res, sigmoid_mul, conv1d fusions (ATOM ports) | medium |
| [#52494](https://github.com/vllm-project/vllm/pull/52494) | fuse MLA q/kv RMSNorm into one AITER launch | low–medium |
| [#54248](https://github.com/vllm-project/vllm/pull/54248) / [#54254](https://github.com/vllm-project/vllm/pull/54254) | per-token FP8 fusions — **no-op until [#51392](https://github.com/vllm-project/vllm/pull/51392) lands** | C1 only |
| [#52033](https://github.com/vllm-project/vllm/pull/52033) | dual-stream decode with hipgraphs — **multi-stream forced OFF in our config** | inert for us |
| [#54165](https://github.com/vllm-project/vllm/pull/54165) / [#54163](https://github.com/vllm-project/vllm/pull/54163) | mamba align cache under spec decode — **spec is off at C72** | inert at C72 |
| [#51392](https://github.com/vllm-project/vllm/pull/51392) | online quantization; only `nvidia/mla.py` reaches this overlay | dead on AMD |
| [#51705](https://github.com/vllm-project/vllm/pull/51705) | causal multi-token verification under DCP | **required**, not perf |
| [#53598](https://github.com/vllm-project/vllm/pull/53598) | per-group DCP geometry → prefix-cache hits under DCP | required; indirectly large (we measure 73.5% hit) |
| [#53917](https://github.com/vllm-project/vllm/pull/53917) | `SimpleCPUOffloadConnector` under DCP — **we run offload dram** | required |
| [#52707](https://github.com/vllm-project/vllm/pull/52707) | clamp negative external block allocation | required (crash guard) |
| [#50618](https://github.com/vllm-project/vllm/pull/50618) | densify strided activations; KDA `f_b_proj` over-reads 12 KB at TP8 | correctness guard |

### Planned detach order

Functional check first (does it import and serve), perf later. Fixed-len
(`TEST=1`, ~15 min) rather than agentic (~2 h).

1. **Expect-free:** P51392 (NVIDIA path) → P54165 (spec off at C72) →
   P52033 (multi-stream off) → P50618 (isolated guard)
2. **Expect-coupled, informative when they fail:** P52494 (new file — if REST's
   `linear.py` imports it, instant fail), P52968 (`envs.py` flags REST reads),
   P53301 (only `kda_metadata.py` is separable)
3. **Not worth dropping:** P51705, P53598, P53917 — all required-to-run

### Known limit of this ablation

The three highest-perf PRs — **[#53166](https://github.com/vllm-project/vllm/pull/53166), [#51437](https://github.com/vllm-project/vllm/pull/51437), [#52190](https://github.com/vllm-project/vllm/pull/52190)** — are **not separable at
all**. They live entirely inside contested files in REST. So this experiment can
tell us what is *droppable*; it cannot measure what is *valuable*.

A second limit: fixed-len barely exercises KV offload or the scheduler, so a
P53917 detach will likely look neutral at fixed-len and still break agentic —
the same trap as T219 (mns 20 at C16 looked reasonable and caused total
starvation). Anything surviving fixed-len needs one agentic confirmation before
being called prunable.

---

## Bottom line

- **File-group axis is closed:** 8 KB detachable, 256 KB is one coupled unit.
- **PR axis adds 20% of the patch as individually-detachable buckets**, which is
  better than A/B/C/D/E gave us, but still leaves 80% inseparable.
- **Neither axis has found meaningful perf to prune.** The value of the work is
  upstreaming surface reduction, not throughput.
- Four of the seventeen PRs are already merged ([#51705](https://github.com/vllm-project/vllm/pull/51705), [#53598](https://github.com/vllm-project/vllm/pull/53598), [#52707](https://github.com/vllm-project/vllm/pull/52707), [#52033](https://github.com/vllm-project/vllm/pull/52033))
  and sit in `nightly-7c5dc571` but **not** in our v4 base `46638857`
  (ancestry-verified). Moving the base would absorb them for free — at the cost
  of the overlay, which is cut against 46638857 and applies to no other base.

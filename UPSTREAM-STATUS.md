# UPSTREAM STATUS — what we carry, and what needs upstreaming

Two separate stacks sit on top of `vllm/vllm-openai-rocm:nightly-46638857…`.
This file lists both and their upstream state.

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

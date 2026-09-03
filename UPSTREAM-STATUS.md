# UPSTREAM STATUS — what we carry, and what needs upstreaming

**Newest first.** Everything below "HISTORICAL" is provenance only.

*Tested and produced, not shipped* — `pronly` is a local image validated on
b23_07, not published to a registry.

---

# WHAT WE TESTED & PRODUCED — `kimi-k3-vllm:pronly`

`nightly-7c5dc571` + **8 upstream PRs** (4 merged in base, 4 applied). No vendor
patch. **10,799 tok/s/GPU @ C72**, err 0.09%, GSM8K 0.995.
Image: `kimi-k3-vllm:pronly-nq-no50618`.

Started at 11 PRs and 10,692 (T232). Pruning removed 3 for free and found that
2 more were **not** free (2.27% combined), so the recommended stack keeps them.
**Upstreaming ask: 4 open PRs, down from 7.**

## PRs applied — 8 total. **Recommended stack = [T236](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/33622043517)'s.**

Prune ladder is **closed** (T232–[T238](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/33643182788)). The recommended set is **not** the
smallest that runs — the last two prunes cost 2.27% and are worth keeping.

### THE ASK — 4 open PRs

`status` = has the PR been **removed and re-measured**?
✅ verified · ⏳ yet to verify. **⏳ does not mean "not needed"** — both must-haves
are believed load-bearing, the drop arm simply has not been run. PRs we measured
as genuinely droppable are in **DO NOT NEED** further down.

#### MUST HAVE — 2 PRs

| # | PR | tag | what it does | PR state | Δ if dropped | status |
|--:|---|---|---|---|--:|---|
| 1 | [#53917](https://github.com/vllm-project/vllm/pull/53917) | `cpu-offload` | fix per-group KV geometry for CPU offload under DCP | **open** | perf unmeasured | ✅ **required** — code-proven |
| 2 | [#53940](https://github.com/vllm-project/vllm/pull/53940) | `a4w4-moe` | a4w4 FP4 MoE kernels | **open** | unmeasured | ⏳ yet to verify |

#### GOOD TO HAVE — 2 PRs, 2.27% together — in priority order

| # | PR | tag | what it does | PR state | Δ if dropped | status |
|--:|---|---|---|---|--:|---|
| 3 | [#52494](https://github.com/vllm-project/vllm/pull/52494) | `mla-rmsnorm-fusion` | fuse MLA q/kv layernorm | **open** | **−1.35%** | ✅ verified ([T237](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/33631955363)) |
| 4 | [#52968](https://github.com/vllm-project/vllm/pull/52968) | `attn-conv-fusion` | fuse attn_res, conv1d, sigmoid+mul | ⚠️ **DRAFT** | **−0.93%** | ✅ verified ([T238](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/33643182788)) |

**#53917 is code-proven required (CPU-only check, 2026-09-03).** No GPU run
needed — the mechanism is decidable from source:

1. #53917 adds `requires_dcp_block_aligned_interleave` (default `True` in
   `KVConnectorBase_V1`) and sets it **`False`** on `SimpleCPUOffloadConnector`.
   Verified absent from stock `nightly-7c5dc571` — `grep` finds no occurrence.
2. Stock `config/vllm.py:2772` fires for **any** connector and overwrites
   `cp_kv_cache_interleave_size = local_block_size`, logging a *"PD
   disaggregation"* message. We are not doing PD; we are doing CPU offload.
3. We run `--cp-kv-cache-interleave-size 1` with block sizes 1536 / 3072, so
   **stock silently rewrites our 1 → 1536.** The PR is what makes the flag stick.

So the *functional* requirement is settled. What remains unmeasured is the
**throughput cost** of that rewrite — that still needs a drop arm.

**Priority order rationale.** #52494 first: larger delta (−1.35%), outside the
±1.2% band, and the PR is open and review-ready. #52968 second: −0.93% sits
*inside* the band at n=1, so the per-PR figure is indicative rather than proven,
and the PR is a draft so it cannot merge yet either way. The pair's combined
**−2.27%** is the number that holds up — outside the band and monotone across
T236 → T237 → T238.

**#52968 is still a DRAFT** — it cannot merge until the author marks it ready
for review. That is a blocker independent of our measurements, and the cheapest
one to clear. PR state checked 2026-09-03; the other three are open and
review-ready.

**Two of four have a measured delta.** The must-haves rest on mechanism and are
**still needed** — they are pending measurement, not rejected.

**#53917 note (verified by inspection, 2026-09-03):** the CPU-offload connector
is **already in stock `nightly-7c5dc571`** — `vllm/v1/simple_kv_offload/` and
`simple_cpu_offload_connector.py` both exist unpatched. #53917 does not add it;
it **fixes** it, across 10 files / 622 lines, most of them the KV cache manager
layer (`kv_cache_manager`, `kv_cache_coordinator`, `single_type_kv_cache_manager`,
`sched/scheduler`). That is what K3 needs under DCP, where KV groups are
heterogeneous (1536 attention / 3072 KDA) and per-group geometry must be right.
T216's C52 starvation on bare nightly is the symptom.

**Therefore the drop arm is runnable** — a `no-53917` image will still boot and
serve, just with stock geometry. #53917 can get a real number instead of ⏳.

**RETRACTED (2026-09-03): the #53940 ablation is confounded — do not cite it.**

[T241](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/33665892734) dropped #53940 and the run never reached profiling, and this file
previously recorded that as proof the PR is load-bearing. That conclusion does
not hold:

- **[T242](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/33689675072)** ran gmu 0.85 on `pronly-nq-no50618` — the image that scored 10,799 in
  [T236](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/33622043517), **with #53940 present** — and collapsed with the identical signature.
- **[T243](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/33706235658)** ran a fixed-len probe on that same image at the ledger's gmu 0.9 and
  failed 99.3% (1/144, 0 generated tokens, TTFT 150 s).

So the collapse reproduces without removing #53940, on two other configurations,
including a non-agentic workload. The common factor is the **node**, not the
patch: `numa_balancing` was reset to 1 by the reboot and is not persisted in
sysctl. Details and fix in `Kimi-DCP-Experiemnts-Summary.md`.

**#53940 stays in the ask on mechanism** — it is live on the MoE path
(`flydsl_moe1/moe2` in the T195 and [T243](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/33706235658) logs) and every good number in the
ledger was measured with it applied. But its removal cost is **unmeasured**, and
the ablation must be rerun on a healthy node before any number is quoted.

Baseline **10,799 tok/s/GPU @ C72**, err 0.09%, GSM8K 0.995, image
`kimi-k3-vllm:pronly-nq-no50618`. Dropping both "good" PRs costs **2.27%**
(10,799 → 10,554) — outside the ±1.2% band and monotone.

**File #53917 first.** Only open must-have, largest block (27 hunks), and
everything else composes on top of it. Detail and rationale below.

### Already merged upstream — nothing to ask for, listed for provenance

| PR | what it does |
|---|---|
| **[#51705](https://github.com/vllm-project/vllm/pull/51705)** | causal multi-token verification under DCP |
| **[#53598](https://github.com/vllm-project/vllm/pull/53598)** | per-group DCP cache geometry, prefix-cache hits |
| **[#52707](https://github.com/vllm-project/vllm/pull/52707)** | clamp negative external block allocation |
| **[#52033](https://github.com/vllm-project/vllm/pull/52033)** | ROCm dual-stream shared-expert |

## UPSTREAMING PRIORITY — must-have vs good-to-have

Ordered by measured cost of dropping it. Baseline for every delta is
**[T236](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/33622043517) = 10,799 tok/s/GPU @ C72** on `kimi-k3-vllm:pronly-nq-no50618`.

### MUST HAVE — the stack does not work without these

| # | PR | tag | why | evidence | status |
|--:|---|---|---|---|---|
| 1 | **[#53917](https://github.com/vllm-project/vllm/pull/53917)** | `cpu-offload` | `SimpleCPUOffloadConnector` + per-group KV geometry under DCP. We run `offload dram`; bare nightly **starves entirely** at C52 without this class of fix (T216). 17 of its 27 hunks are generic cache geometry + connector base/factory, not SimpleCPU-specific. | never removed | ⏳ yet to verify |
| 2 | **[#53940](https://github.com/vllm-project/vllm/pull/53940)** | `a4w4-moe` | a4w4 flydsl MoE kernels. Live on the MoE path (`flydsl_moe1_abf16_wfp4_bf16_…` in the T195/[T243](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/33706235658) logs). Rides in the separate `pr_stack/`. | **CONFOUNDED — [T241](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/33665892734)'s ablation is void.** The same collapse reproduced in [T242](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/33689675072)/[T243](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/33706235658) *with* #53940 applied, so the run measured the node, not the patch. Removal cost unmeasured; rerun required. | ⏳ yet to verify |

**Neither has a measured number, and that is the honest state.** #53917 is
load-bearing by mechanism and by the T216 starvation result; #53940 has never
been removed. Both are must-have on reasoning, not on a delta.

### GOOD TO HAVE — measured, worth 2.27% together

| # | PR | tag | Δ if dropped | throughput | why it costs | status |
|--:|---|---|--:|--:|---|---|
| 3 | **[#52494](https://github.com/vllm-project/vllm/pull/52494)** | `mla-rmsnorm-fusion` | **−1.35%** | 10,799 → **10,653** ([T237](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/33631955363)) | fuses `q_a_layernorm`/`kv_a_layernorm` into one AITER launch. K3 carries no `@support_torch_compile`, so `MLADualRMSNormFusionPass` never runs — without the PR those stay two launches on **every MLA layer, every forward**. | ✅ verified |
| 4 | **[#52968](https://github.com/vllm-project/vllm/pull/52968)** | `attn-conv-fusion` | **−0.93%** | 10,653 → **10,554** ([T238](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/33643182788)) | attn_res + add_rmsnorm_quant fused, causal_conv1d fused for qkv, native sigmoid+mul → fused kernel. Same mechanism: inductor never fuses these for K3. | ✅ verified |
| | **both** | | **−2.27%** | 10,799 → **10,554** | outside the ±1.2% band, monotone | ✅ verified |

### DO NOT NEED — measured free

| PR | Δ if dropped | throughput |
|---|--:|--:|
| [#51392](https://github.com/vllm-project/vllm/pull/51392) + [#54254](https://github.com/vllm-project/vllm/pull/54254) | **+0.83%** (noise) | 10,691 → 10,781 (T235) |
| [#50618](https://github.com/vllm-project/vllm/pull/50618) | **+0.17%** (noise) | 10,781 → 10,799 ([T236](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/33622043517)) |
| [#54165](https://github.com/vllm-project/vllm/pull/54165) | inert — spec off at C72 | — |
| [#50813](https://github.com/vllm-project/vllm/pull/50813) | dead code | — |

**Caveat on the free ones:** #50618 guards a 12,288-byte over-read in KDA
`f_b_proj`. Measured-safe over 1h47m and 2,900+ requests, **not proven-safe** —
an over-read need not fault. First to restore on any stray memory fault.

### Filing order if upstreaming

**#53917 first** — it is the only must-have that is still open, it is the
largest single block (27 hunks), and everything else composes on top of it.
Then #52494 and #52968, which are small, self-contained ROCm kernel fusions
with a measured number attached — the easiest kind of PR to argue for.

---

Image: **`kimi-k3-vllm:pronly-nq-no50618`** — **10,799 tok/s/GPU @ C72**,
err 0.09%, GSM8K 0.995.

## PRUNED — measured

| PR | evidence | confidence |
|---|---|---|
| **[#51392](https://github.com/vllm-project/vllm/pull/51392)** + **[#54254](https://github.com/vllm-project/vllm/pull/54254)** | GSM8K 0.995 (T234), 10,781 (T235) | **strong** — dependency pair, inert by construction: checkpoint is `mxfp4` with `quantization_config=None`, so the online-quant path had no work |
| **[#50618](https://github.com/vllm-project/vllm/pull/50618)** | 10,799, err 0.09% ([T236](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/33622043517)) | **measured-safe, NOT proven-safe** — guards a 12,288-byte over-read in KDA `f_b_proj` (`stride=(6288,1)` at TP8); only the python hunk ever applied. **First to restore on any stray memory fault.** |
| **[#54165](https://github.com/vllm-project/vllm/pull/54165)** | closed-unmerged; spec off at C72 | author closed it as superseded by [#54163](https://github.com/vllm-project/vllm/pull/54163) |
| **[#50813](https://github.com/vllm-project/vllm/pull/50813)** | dead code, zero GPU cost | `quark_moe.py` unreachable |

## The full ladder — with GPU KV capacity

| applied PRs | trial | tok/s/GPU | err | **GPU KV capacity** |
|--:|---|--:|--:|--:|
| 6 | T232/T233 | 10,691 (n=2) | 0.18–0.22% | 29,656,464 |
| 4 | T235 | 10,781 | 0.18% | 29,656,464 |
| **3** | **[T236](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/33622043517)** | **10,799** | **0.09%** | 29,656,464 |
| 2 | [T237](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/33631955363) | 10,653 | 0.22% | 29,656,464 |
| 1 | [T238](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/33643182788) | 10,554 | 0.18% | **29,816,030** |

### A reading I had to correct

Each single step sat inside the ±1.2% band, so I called them all "no measurable
difference." **Cumulatively that was wrong.** [T236](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/33622043517) → [T238](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/33643182788) is **−2.27%**,
outside the band and **monotone** across three arms. The two kernel-fusion PRs
each cost ~1%, which is exactly what the mechanism predicts: K3 carries no
`@support_torch_compile`, so inductor never fuses those launches — without the
PRs they stay unfused on every layer, every forward.

**Lesson for the method:** single-step-inside-band does not license
"free." Check the cumulative walk.

### GPU KV capacity is now tracked every run

Identical (29,656,464) across T232–[T237](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/33631955363), rose to **29,816,030** (+0.54%) in [T238](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/33643182788)
when #52968 came out — its fused kernels free device memory the KV pool absorbs.
So [T238](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/33643182788) had **more** cache and still scored **lower**, which strengthens the
conclusion but means that arm was not perfectly one-variable. Every earlier arm
was clean on this axis.

## Next: LMCache — see **`LMCACHE.md`**

Target 12,500 (+15.8% from 10,799). Headline metric appears to be **prompt-token
throughput ÷ 8** (10,799 × 8 = 86,392 vs live `tput_in_srv` 85–90k; output was
only 515 tok/s) — **strong inference, not verified in code**. If it holds,
prefix-cache hit rate is the mechanism: we capture **~75%** GPU + 16% external
against a theoretical **96.9%**.

Base bump to `nightly-73029d42…` remains available but **not taken** — it would
invalidate the baseline and needs a fresh GSM8K gate. Nothing has landed on the
cache/scheduler paths since it was cut.

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


# Archived patches

Removed from `apply_kimi_k3_patches.sh`; kept here for the record.

| Patch | Why archived |
|---|---|
| `aiter-opus-rows` (ROCm/aiter#4915) | **Exonerated.** Changed neither the GPU page fault nor throughput: T74 with opus rows removed faulted at 2,771 s / 4,551.0; T77 with them present faulted at 3,013 s / 4,451.9. Untuned-GEMM count was ~the same either way (42,320 vs 46,056), so the "480×" effect once attributed to it was really piecewise-vs-full graphs changing batch shapes. |
| `pr51171_vllm.diff` (vLLM #51171) | **Unusable alongside #51705.** Both rewrite `rocm_aiter_mla.py`; stacking them leaves 5 of 8 hunks rejected, losing the verify-path persistent buffers that are the point of the PR. #51705 is required for DCP, so #51171 cannot be applied with it. Its `triton_mla.py` hunk also would not have fixed the DSpark DCP gate — it uses the same `non_causal_multi_token_decode` predicate. |
| `dcp-lse`, `dcp-blocktable`, `dcp-direct-a2a`, `dcp-gathered-heads` | Superseded by #51705 (removed earlier). |

## Still active

| Patch | Effect on `nightly-f94666b6` |
|---|---|
| `pr51705` (vendored `e72380a5`) | **Essential.** DCP support, the `VLLM_ALLOW_DCP_FULL_CUDAGRAPH` hatch, softmax-LSE for AITER MLA decode. Without it DCP dies at init: *"requires attention implementations to return the softmax LSE during decode"*. |
| `pr51705-rejects` | Adds `enable_dcp_q_replicate` to `MultiHeadLatentAttention.__init__` (PR hunk rejects on this image due to a `run_gemm_rs_ar` → `run_gemm_rs` rename). Silent with spec off; fatal at init with MTP on. |
| `kv-blockpool` | Applies. Clamps a negative block count. |
| `aiter-pybind11` | No-op on this image ("internals already agree") but harmless and arch-dependent; kept as a guard. |
| `triton-mla-cudagraph` | Auto-skipped when `pr51705_vllm.diff` is present, since #51705 supersedes it. Retained as a fallback for when the diff is absent. |

---

# Upstream PRs considered and NOT applied

Moved here from the DCP summary so the main page carries only what is live.
Each was investigated on this stack; none is applied.

| PR | Verdict |
|---|---|
| [#51040](https://github.com/vllm-project/vllm/pull/51040) | **Inert for Kimi-K3.** Enables the FP8 ASM MLA prefill by padding 12 → 16 heads. Applies cleanly (0 failed hunks) but patches `rocm_aiter_mla.forward_mha`, which K3 **never calls** — it prefills through its own `_forward_prefill_fused`. Source comment: *"there is no dense-MHA (forward_mha) fallback."* |
| [#51171](https://github.com/vllm-project/vllm/pull/51171) | **Unusable alongside #51705.** Both rewrite `rocm_aiter_mla.py`; stacking rejects 5 of 8 hunks, losing the verify-path persistent buffers that are the PR's purpose. #51705 is mandatory for DCP. Its `triton_mla.py` hunk also uses the same `non_causal_multi_token_decode` predicate, so it would not have fixed the DSpark DCP gate. |
| [#50619](https://github.com/vllm-project/vllm/pull/50619) | Not applied. Overlaps #51171 on the draft-downgrades-target defect. |
| [#50791](https://github.com/vllm-project/vllm/pull/50791) | Not applicable — B200 / FlashInfer. Cited during the crash hunt only as the *bug class* (DCP forces LSE, slab carved from a DCP-unaware workspace). |
| [#50883](https://github.com/vllm-project/vllm/pull/50883) | Not applied. Scales `UniformTypeKVCacheSpecs` by DCP; would matter for hybrid + DCP + CPU offload. |
| [#52269](https://github.com/vllm-project/vllm/pull/52269) | Draft. Kimi-K3 DSpark under DCP, follow-on to #52188. Not vendored. |
| [#48392](https://github.com/vllm-project/vllm/pull/48392) | Not applicable — dense GQA/MHA drafts. K3 is MLA, covered by #52188. |
| [#51203](https://github.com/vllm-project/vllm/pull/51203) | Not applicable — MiniMax-M3 specific, fails on an assert rather than a page fault. Matched our crash *shape* during the hunt, nothing more. |
| [ROCm/aiter#4915](https://github.com/ROCm/aiter/pull/4915) | **Exonerated** — see `aiter-opus-rows` above. |

## Why FP8 prefill is unreachable on this stack

Relevant to #51040, and worth not re-deriving:

- The kernels exist and report supported: `aiter.mla_prefill_ps_asm_fwd`,
  `aiter.mla_reduce_v1`, `_fp8_mla_prefill_supported() == True`.
- vLLM wires them into `forward_mha`, which K3 never calls.
- The backend K3 does use, `aiter_flash_attn.py`, has **zero** fp8 references,
  and `aiter.flash_attn_varlen_func` exposes no fp8 scale parameters.
- The only fp8-capable MLA prefill backend, `tokenspeed_mla`, is absent from
  `platforms/rocm.py` and its package is not installed.

Measured upside anyway: MLA attention is ~5.6% of time, so a perfect fp8
attention kernel is worth **~2.8% end-to-end**. See
`Kimi-K3-Where-The-Time-Goes.md`.

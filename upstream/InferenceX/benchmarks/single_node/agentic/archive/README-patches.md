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

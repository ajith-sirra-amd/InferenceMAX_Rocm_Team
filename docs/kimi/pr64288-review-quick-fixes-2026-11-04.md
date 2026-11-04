# PR #64288: first review fixes — 2026-11-04

User requested the least time-consuming reviewer fixes first.

## Published change

- PR: https://github.com/vllm-project/vllm/pull/64288
- Current source: 8a55b8eacf9020a0b421a3c50a927c4a40e61604
- Parent: 2c01e1ce21a262f6108861a831eae5a7d2a2ca8e
- Three source files changed: 14 insertions, 117 deletions (103 net lines removed).
- Removed unused k_rope/quantized_q interfaces and ROPE_DIM/SKIP_Q kernel paths.
- Removed signature introspection for APIs present in pinned AITER v0.1.27.
- Reused self._use_flydsl_fp8_prefill in the cached-context call.
- Removed the code comment's reference to an out-of-tree prototype.

The five Triton helper ASTs were compared against the old functions specialized to the removed options' default values. Their active operations matched. This is a source-structure check, not GPU execution evidence. The existing quantization arithmetic and scale rule were retained.

## Completed validation

- Fork run: https://github.com/ajith-sirra-amd/vllm/actions/runs/41772698165
- Result: success, 1m36s.
- Workflow SHA: 0b6706c590cd1fc14eaed54ecc3bb42ecfca3139
- Validated native source: 8a55b8eacf9020a0b421a3c50a927c4a40e61604
- Artifact: 13296080161, native-fp8-review-static.
- Ruff, formatting, applicable pre-commit/mypy and exact source comparisons passed.
- All three quantizer call sites use the retained q/k/v interface.
- The original kernel tests were retained; no GPU run was dispatched for this cleanup.
- Upstream DCO passed. The current contributor-authorization gate still fails before pre-commit; no label was requested or added.
- Gate: https://github.com/vllm-project/vllm/actions/runs/41772958875/job/126386667744

The PR source was advanced only after the fork checks passed. The validation workflow/helper remain on a separate fork branch and were not added to the PR.

## Review replies

- Unused options: https://github.com/vllm-project/vllm/pull/64288#discussion_r4463561093
- Signature check: https://github.com/vllm-project/vllm/pull/64288#discussion_r4463561307
- Prototype wording: https://github.com/vllm-project/vllm/pull/64288#discussion_r4463561510
- Stored enablement flag: https://github.com/vllm-project/vllm/pull/64288#discussion_r4463561669

Replies are posted; review threads were not forcibly resolved.

## Remaining work

- Isolate the C56 ALLGATHER stall with controlled runs of the failing source. This cleanup does not establish a fix for that failure.
- Separate the two asynchronous-copy fixes and validate their default-path behavior.
- Preserve existing ASM attention for uncached prefills rather than disabling it globally.
- Address cached-value saturation with numerical coverage and a deliberate scale-policy decision.
- Run long-context accuracy through normal routing, proving actual cache reuse and FP8 execution.
- Provide matched current-head/main/flag-off bench-serve results at several concurrencies and below the routing cutoff.
- Validate the final source on GPU, then complete C56/C96 runs.
- Keep the extra environment-variable discussion open.
- Do not modify or rerun #59124; do not mix its source with our direct projection.
- New-token FP8 remains deferred.

Earlier GPU/GSM8K/performance evidence applies to 492080779, not this source. The PR description now records the current revision, independent static result and C56 failure; it does not claim current-head GPU validation.

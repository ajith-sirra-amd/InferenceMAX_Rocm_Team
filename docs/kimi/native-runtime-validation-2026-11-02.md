# Native FP8 runtime validation — 2026-11-02

This report supersedes the pending-runtime status in earlier review notes.
PR [#64288](https://github.com/vllm-project/vllm/pull/64288) remains at
`492080779e9feea19308092b0aba0840e87c6115` and remains a draft.

## Completed full replay

[Run 41413372088](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/41413372088)
completed successfully. GPU job: 125171486390.
Execution: `b56b28bb1fc7302f8f35095341165ddea89de4fa`.
Workflow: `3307b42d65b9d848ff9eb7425b0f62f1429a160e`.

C48, TP8/DCP8, normal routing/graphs, 3600-second agentic replay, synthetic
MTP acceptance. Direct FP8 projection and cached-context FMHA; new-token
attention remains BF16. This is an optimized current-image reference,
not a BF16 baseline or an isolated comparison against the historical prototype.

- Throughput: **18,244.04135 total tokens/s/GPU** over an aggregate interval
  of **3627.88643 seconds**.
- TTFT and inter-token coverage: **100.0% / 100.0%**; unchanged requirement 98%.
- **3823 successful + 11 error records / 3834 profiled records (0.29%).**
  Ten responses contained no actual content; one request-body write failed
  because its connection was closing. Their underlying causes remain open.
- Eight further requests were cancelled at the configured duration/grace-period
  cutoff. The server drained to zero running requests.
- **Zero BLAS warnings, server runtime errors, tracebacks or quota errors.**
  Startup shared-memory waits were logged; startup subsequently succeeded.
- All eight distinct workers executed FP8 projection and cached FMHA during
  replay, with normal routing. Each reported `_BlasBackend.Cublas`.
- All eight checked native source hashes match the existing static/accuracy
  evidence. No periodic stack files were created.

[Machine-readable evidence](../../results/atom-attention/run41413372088/full-replay-evidence.json)
records artifact IDs, exact revisions, settings and limitations.

## Runtime correction and scope

The recipe sets `TORCH_BLAS_PREFER_HIPBLASLT=0` and
`TORCH_BLAS_PREFER_CUBLASLT=0`. Earlier isolated BF16 tests reproduced
hipBLASLt errors without FP8 imports on both images. Explicit BLAS selection
passed 112/112 cases on each image and current-head native GPU tests passed
8/8 on the newer image.

The added periodic faulthandler diagnostic is disabled. In the preceding
startup failure, one worker's first timed stack dump was truncated, followed
immediately by EngineCore shutdown. Successful startup with the timer disabled
supports that correction. It does not establish a C-level crash cause.

The original late serving stall did not recur in this full replay. One pass
does not prove that the earlier hipBLASLt warnings caused that stall.
No global BLAS override was added to the native attention implementation.
Keep this runtime workaround documented in reproduction settings, and use
the same settings in every matched performance comparison.

## Newer-image accuracy validation

[Run 41426504427](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/41426504427)
has been dispatched and is being followed through completion.
GPU job: 125214526544.
Workflow SHA: `c047b417fe471c7a3e913544ebd289343f82bd5c`.
Execution/PR SHAs and BLAS settings are unchanged; timer remains disabled.

This is full GSM8K with C48/TP8/DCP8, real MTP verification, eager execution
and 512-token accuracy-stress routing. The fixed score threshold remains 0.90.
The newer pinned image is
`vllm/vllm-openai-rocm:nightly-rocm100-e0d00e4b8eb0c89ec30921d0695cfdd409e9c211@sha256:b1374c3814b6e7706c49968ba17ca3b35484382f7b9fc7115dd3e66068d53f2f`.

After accuracy validation, continue the matched newer-image performance run.
The historical BF16-reference absolute-LSE <0.20 criterion remains unpassed.
Reviewer comparisons and upstream CI/human review remain outstanding.

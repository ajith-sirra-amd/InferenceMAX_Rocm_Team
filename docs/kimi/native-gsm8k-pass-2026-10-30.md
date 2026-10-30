# Native FP8 validation passed - 2026-10-30

PR revision: `492080779e9feea19308092b0aba0840e87c6115`.

[Full GSM8K run 41064573708](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/41064573708) completed successfully.

- Strict match: **1278/1319 (96.8916%)**.
- Flexible extract: **1277/1319 (96.8158%)**.
- Full 1319 questions, five-shot, temperature zero, C48, TP8/DCP8, real MTP block verification.
- Native FP8 projection and cached-context FMHA executed on all eight distinct workers during evaluation.
- `VLLM_GPU_SYNC_CHECK=error` remained enabled. Server log: zero runtime errors, tracebacks, ERROR lines or quota errors.
- Exact installed source hashes match [passing static checks 41065468482](https://github.com/ajith-sirra-amd/vllm/actions/runs/41065468482).

[Archived evidence](../../results/atom-attention/run41064573708/gsm8k-evidence.json) identifies source revisions, artifacts, metrics and executed paths. This supersedes the pending native-GSM8K status in earlier task notes.

The two metadata transfers now use `async_tensor_h2d` to stage integer metadata in pinned CPU memory. This does not change FP8 quantization or arithmetic.

Validation used eager mode and 512-token accuracy stress. New-token attention remained BF16. Normal-graph native performance, matched native comparisons, and the historical absolute-LSE <0.20 criterion remain outstanding. PR #64288 stays draft. No matched native BF16 control has been run, so historical prototype score differences cannot be attributed to FP8.

An optional post-evaluation lm-eval Git metadata lookup reported dubious ownership of `/outputs`. It did not fail evaluation; exact native source provenance was independently verified. That warning remains a separate harness issue.

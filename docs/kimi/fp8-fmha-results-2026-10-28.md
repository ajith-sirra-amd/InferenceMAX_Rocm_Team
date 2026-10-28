# Kimi hybrid FP8 prefill: full performance result

[中文](fp8-fmha-results-2026-10-28_zh.md) · [Plugin and flag](atom-flydsl-attention.md)

[Run 40646139769](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/40646139769) completed successfully. Compared with baseline [39239112271](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/39239112271), the complete candidate improved total throughput by **9.30%** and output throughput by **14.35%**.

## Measured results

| Metric | Baseline | Candidate | Change |
|---|---:|---:|---:|
| Total tokens/s/GPU | 15476.46881 | 16915.72154 | +9.30% |
| Output tokens/s/GPU | 103.05251 | 117.84290 | +14.35% |
| TTFT p90, seconds | 4.11628 | 3.18346 | -22.66% |
| ITL p90, seconds | 0.04778 | 0.04205 | -11.99% |
| Successful measured requests | 3357 | 3598 | |
| Measured errors | 6 empty responses | 6 empty responses + 1 ClientOSError | |
| GPU KV pool, tokens | 20951355 | 22665373 | |
| GPU cache hit | 81.165% | 85.205% | |
| CPU/external cache hit | 11.884% | 8.192% | |
| Overall cache hit | 93.050% | 93.396% | |

Warmup is separate: baseline had 101 successes and 430 empty-response errors; candidate had 105 successes and 426 empty-response errors. Successful workflow completion does not erase these failures.

## Configuration and observed paths

The serving checkout was **`e4cf47a7650321e4bd40c83746088a2679ec0509`**, verified in the setup-job log. The workflow head contains later documentation and is not the serving-code identity.

- Model: `moonshotai/Kimi-K3`; TP8/DCP8, concurrency 48, max sequences 96, max batched tokens 8192.
- ROCm image: `vllm/vllm-openai-rocm:nightly-1b2a666391f2a24228a1436b64d547c34cc6b6ce`.
- Native DSpark k4; synthetic acceptance 3.36 for the performance replay.
- Warmup 10 requests/lane; requested replay 3600 seconds; measured reporting duration 3627.79603 seconds.
- All eight worker PIDs logged BF16 causal attention, FP8 cached FMHA, native compact-FP8 gather with BF16 `kv_b_proj`, same-forward Q reuse, and disabled main ASM prefill metadata.
- Final graph capture used 6.67 GiB/GPU. Model loading used 184.08 GiB/GPU and took about 765 seconds outside the benchmark window. Earlier sizing capture is a separate phase and must not be added to final capture.
- Server log had no matches for traceback, HIP error, EngineDeadError, hipBLASLt or NaN. This is not accuracy evidence.

The raw aggregate has incomplete identity fields, including `spec_decoding: none`. The captured launch command establishes actual DSpark usage; the raw aggregate is preserved unchanged.

## Interpretation and next comparison

This is **not an isolated FMHA speedup**. The candidate also changes online weight quantization, metadata allocation and Q preparation. Cache residency and request mix differ. Total throughput includes cached input tokens; it is not a measure of fresh-prefill compute alone. Input-length bins show roughly 11–13% lower ITL p90; TTFT p90 improved except in the below-8192 bin (+1.06%). These bins are descriptive, not paired identical request schedules.

The plugin-only opt-in flag was added afterward in [2a656fd](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/commit/2a656fdc330761bb076b5c9115307dff4cbadc3a). Five CPU routing tests passed; this revision has not received a new GPU serving test. No native vLLM patch is required to use the plugin.

After accuracy review, compare `ATOM_USE_FLYDSL_FP8_PREFILL_ATTN=0` versus `1` on that same revision, holding the model, image, weight-quantization targets, trace, speculation, topology and warmup fixed. Verify cached FP8 engagement in the enabled arm. This isolates the attention switch more closely; residual replay/cache differences still need reporting.

## Accuracy follow-up and evidence

[GSM8K run 40658068830](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/40658068830) was dispatched immediately after the GPU performance job finished and remains pinned to `e4cf47a`. It uses full GSM8K, five-shot, temperature 0, real block rejection and no evaluation limit. **The run completed successfully; strict-match was 97.1948% (1282/1319), and flexible-extract was 97.1190% (1281/1319).** Both validator checks passed. The repository's Kimi validator requires at least 0.90 on both exact-match extraction metrics.

**There were zero FP8-FMHA and zero Q-reuse markers.** All eight workers logged BF16 causal and cached attention. The sample artifact contains 2638 rows: two extraction filters for the same 1319 questions. This establishes the tested serving configuration's GSM8K pass; it does not establish model accuracy of the cached FP8 kernel or validate the later flag/native-port revisions. See [accuracy proof](../../results/atom-attention/run40658068830/accuracy_proof.json) and [lm-eval results](../../results/atom-attention/run40658068830/lm_eval_results.json).

Evidence: [raw benchmark aggregate](../../results/atom-attention/run40646139769/benchmark_summary.json), [runtime and command provenance](../../results/atom-attention/run40646139769/runtime_evidence.json), [baseline comparison](../../results/atom-attention/run40646139769/baseline_comparison.json), [workload/error details](../../results/atom-attention/run40646139769/workload_evidence.json), and [GSM8K dispatch](../../results/atom-attention/run40658068830/dispatch.json). Source artifacts: raw `12802705904`, benchmark `12802277146`, server `12802601010`.

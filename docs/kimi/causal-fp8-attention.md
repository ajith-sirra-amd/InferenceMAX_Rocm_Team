# Causal FP8 attention experiment

**English** | [中文](./causal-fp8-attention_zh.md)

The baseline is completed plugin run 40695099743: 17406.08912 total
input + output tokens/s/GPU. This experiment is separate from native vLLM PR #64288.

Reference: [vLLM #59124](https://github.com/vllm-project/vllm/pull/59124),
inspected head `132f6daa6b17cb4000ef58355943bb39339104f2`.
Use FlyDSL `causal=True`, equal query/key sequence offsets, `cross_seqlen=True`,
and independent original Q/K/V descales. The existing quantizer is unchanged.

`ATOM_USE_FLYDSL_FP8_CAUSAL_ATTN=1` enables causal FP8 only when this forward
already prepares direct-FP8 cached KV. The causal call reuses those Q/K/V tensors;
it does not introduce another quantization pass. The original causal K/V descales
remain separate from the 2x-headroom cached-projection output descales.
Forwards without direct-KV preparation retain BF16 causal attention.
Cached-context thresholds, Q-slice quantization, weight conversion, DCP,
NUMA/CCD pinning, speculative settings and the model remain unchanged.

Fifteen CPU routing/evidence tests passed. GPU diagnostics cover strided inputs,
ragged causal sequences, independent FP64 references with BF16 and identical
FP8 operands, fixed-scale future-token masking, preallocated output, LSE,
single preparation, and graph replay after inputs change.
The existing RMS <0.10 and absolute LSE <0.20 diagnostic gates are unchanged;
passing controlled inputs does not erase the earlier captured-BF16 failure.

Full GSM8K must retain the fixed task and scoring and prove both cached FP8
projection and causal FP8 calls on the same eight workers during evaluation.
The verifier uses a separate causal marker and rejects missing or pre-evaluation
evidence. Full 3600-second performance follows successful diagnostics and GSM8K.
No serving smoke or new performance/accuracy claim is made at preparation time.

## Result and native candidate: 2026-10-30

[Attempt 2](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/40804432321/job/123956986964) completed successfully with no quota annotations.
Full C48 + MTP new-token FP8 measured **15,264.82341 total tokens/s/GPU**,
a **12.30182% regression** against **17,406.08912**.

| Metric | Established cached FP8 | With new-token FP8 |
| --- | ---: | ---: |
| Total tokens/s/GPU | 17,406.08912 | 15,264.82341 |
| Output tokens/s/GPU | 120.16691 | 100.53104 |
| Mean TTFT, seconds | 1.76809 | 2.01387 |
| p95 TTFT, seconds | 4.36319 | 5.43159 |
| Mean TPOT, milliseconds | 23.15 | 28.61 |
| Successful measured requests | 3,644 | 3,319 |
| Measured errors | 5 / 3,649 | 8 / 3,327 |

Server logs verify FP8 projection, cached FMHA and causal FMHA on all eight
workers. Causal attention reuses prepared FP8 Q/K/V; it adds no second quantization
pass. Actual server configuration confirms DSpark MTP, despite the summary's
inconsistent `spec_decoding=none` field. The current host is `pit2-p03-g25`;
the established host was `pit2-p03-g15`. The user confirmed identical servers
and declined an extra server/control run. Request populations, output lengths
and cache hits differ, so this is not an isolated causal-kernel measurement.

**Decision:** do not promote new-token FP8. Keep cached-only native PR #64288
at `08e5d81a0d4fc76c20a97f9a517195db5a5dc5be`.
[Native kernel validation 41036211097](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/41036211097) passed recipe/launcher preflight
but failed during wheel resolution, before numerical tests. HIP component
version `7.15.26333` was incorrectly used to request `rocm715`; the ROCm 10
Torch package identifies the correct `rocm100` release. The exact-base
`a1cfd2c1b584c4ac197db9cc9ac84a6d973d39df` rocm100 metadata exists.
Harness fix `e7b3309859056b3bd39d7c54bd347f0df4e77244` derives the release
from the Torch package and checks exact installed distribution versions.
The old resolver reproduced HTTP 404; the corrected resolver passed that case.
Full CPU checks precede the next GPU attempt; native kernel/model results
remain pending. The original captured-BF16 LSE gate remains unresolved.

[Performance evidence](../../results/atom-attention/run40804432321/attempt2-performance-evidence.json).

## GPU diagnostics passed; full runs queued

Source: `bb8bed732f514fd697856c5abd1d2bdd9d56a1e5`.
[GPU diagnostics](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/40712781470) passed four causal cases,
fixed-scale future masking, preallocated outputs, single quantization,
and changed-input graph replay.
Maximum controlled BF16-reference RMS was 0.04442 and LSE error 0.10964;
same-FP8-reference RMS was at most 0.01858 and LSE error 0.00005395.
These passes do not resolve the earlier real-model captured-BF16 failure.

[Full GSM8K](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/40712931283) requires the successful diagnostic.
[Full performance](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/40713116775) requires successful GSM8K,
including causal invocation proof. Both execute the pinned source above.
The checker and evidence collector poll every 60 seconds.
[Dispatch and diagnostic evidence](../../results/atom-attention/causal-fp8-20261028.json).

## Full performance cancelled at user request

The user instructed: **Do not dispatch full run**.
Previously queued run 40713116775 was cancelled before its GPU benchmark job;
the agentic benchmark job was skipped. Only GSM8K run 40712931283 continues.
Do not dispatch or restart a full performance run without a new user instruction.
The accuracy checker and artifact collector remain active.

## GSM8K completed; full performance authorized again

The later instruction to dispatch the full run supersedes the cancellation above.
The cancelled run remains cancelled. A new [full performance run 40719654583](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/40719654583)
uses execution source `bb8bed732f514fd697856c5abd1d2bdd9d56a1e5`, identical to the tested candidate.
Its workflow revision is `a6ac3e9550b5df48ddd52238d91165952a5a8580`; only documentation/evidence changed since the tested source.
The standard 3,600-second AgentX replay uses concurrency 48 and TP8/DCP8 on eight MI355X GPUs.
Compare against 17,406.08912 total input + output tokens/s/GPU, with request errors and actual FP8 execution checked.
The checker and artifact collector poll every 60 seconds.

GSM8K run 40712931283 completed successfully on all 1,319 questions, five-shot, temperature 0:

| Scoring | Correct | Accuracy | Change from cached-only FP8 prototype |
| --- | ---: | ---: | ---: |
| Strict | 1,279 / 1,319 | 96.96740% | -2 answers |
| Flexible | 1,278 / 1,319 | 96.89158% | -3 answers |

Artifacts `12828229398` (results/samples) and `12828099867` (server log) were checked directly.
The unchanged validator with `require_causal=True` verified cached FP8 projection/FMHA and causal FP8 calls on the same eight workers during evaluation.
This is plugin evidence under eager/512-token accuracy routing, not native-PR GPU validation.
It does not close the captured-BF16 LSE gate.

After the performance result, include new-token FP8 in native PR #64288 only if the measured uplift supports it.
Then validate the final native revision on GPU: kernel numerics, graph replay, optimized-path GSM8K, synchronization checks, serving performance, and required CI.
The reviewer also requested each reference PR (#59124 and #62447) alone versus the same PR plus direct FP8 up-projection, with identical weights and runtime settings within each pair.
Those ablations remain pending and are separate from run 40719654583.
Update the PR with actual native evidence and mark it ready for review only after the required checks pass.
The additional-environment-variable discussion remains open for a later decision.

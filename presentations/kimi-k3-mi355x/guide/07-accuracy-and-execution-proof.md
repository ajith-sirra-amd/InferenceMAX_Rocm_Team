# Accuracy and execution proof

[Handbook contents](../HANDBOOK.md) · [Worked examples](11-worked-examples.md)

<a id="s-60"></a>
## Slide 60: Quantization error enters at several points

A wider accumulator cannot restore information lost before multiplication.

- **Cache quantization:** Latent representation error
- **Weight quantization:** Projection parameter error
- **Q/K/V rounding:** Attention operand error
- **Softmax + merge:** Errors can change relative attention mass



Dequantizing a cache row to BF16 does not recreate its original unquantized value. Both the BF16-projection and direct-FP8-projection routes start from approximate cached information. Direct FP8 projection additionally uses quantized weights and an output-scale policy, while avoiding an expanded BF16 storage round-trip. It is therefore incorrect to assume that eliminating a quantize/dequantize pair must improve accuracy. FP32 accumulation reduces accumulation error but does not eliminate input, weight or output rounding. Q and K errors perturb scores before softmax; V errors affect the weighted content. Scale saturation and clipping can be more damaging than ordinary rounding for outlier values.

### Sources

- [Projection implementation and numerical limits](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/blob/b786b8cae9992c190c3d3323197c937d8ed8f8b8/docs/kimi/native-fp8-up-projection-audit.md)
- [Campaign data-flow diagrams and measured stages](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/blob/b786b8cae9992c190c3d3323197c937d8ed8f8b8/docs/kimi/fp8-fmha-dataflow.md)

---

<a id="s-61"></a>
## Slide 61: The cached output scale is a policy to validate

Twice the new-token range is headroom—not a bound on all cached values.

- **New-token amax:** Estimate current K/V ranges
- **Cached output scale:** 2 × max(new-token descale, 10⁻⁶)
- **Old cached outlier:** May exceed that range
- **Validation:** Measure clipping, output error and LSE error



The inspected ATOM-inspired direct-output policy derives K/V output descales from the newly prepared token tensors and multiplies them by two, after applying a small floor. This avoids a separate amax pass over the reconstructed cached chunk, helping performance. But earlier context can contain values outside the new-token range. Zero, very small or unrepresentative new-token activations are especially important test cases. The policy does not prove that all cached values fit in FP8. Alternatives such as a cached-chunk range pass, conservative static scales or more granular scaling have different performance and accuracy costs. The presentation records the current policy and its unresolved numerical limitation rather than declaring it universally safe.

### Sources

- [ATOM MLA reference implementation](https://github.com/ROCm/ATOM/blob/661ef7661eef0d804d2b52d81ed41d80aeaf9fdb/atom/model_ops/attention_mla.py)
- [Campaign data-flow diagrams and measured stages](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/blob/b786b8cae9992c190c3d3323197c937d8ed8f8b8/docs/kimi/fp8-fmha-dataflow.md)
- [Projection implementation and numerical limits](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/blob/b786b8cae9992c190c3d3323197c937d8ed8f8b8/docs/kimi/native-fp8-up-projection-audit.md)

---

<a id="s-62"></a>
## Slide 62: Validate three different questions separately

Arithmetic, quantization fidelity and model quality require different evidence.

- **Same FP8 operands:** Does the kernel compute the intended quantized operation?
- **Original BF16 reference:** How much error does the full quantized path introduce?
- **End-to-end model task:** Does the executed implementation preserve task quality?



A high-precision reference built from exactly the same FP8 operands and scales isolates arithmetic and implementation errors. Comparing with the original BF16 inputs instead measures the combined effect of quantization and kernel arithmetic. Full-model evaluation then tests whether errors affect actual task outcomes. These tests answer different questions and none automatically replaces the others. Include masks, nonunit independent scales, ragged sequences, strided views, empty/short cases where supported, extreme ranges and graph replay with changed inputs. A cached output tensor from a previous forward can make a superficial numerical test pass while serving the wrong data, so pointer reuse and graph replay need explicit coverage.

### Sources

- [Projection implementation and numerical limits](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/blob/b786b8cae9992c190c3d3323197c937d8ed8f8b8/docs/kimi/native-fp8-up-projection-audit.md)
- [Causal FP8 implementation and accuracy evidence](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/blob/b786b8cae9992c190c3d3323197c937d8ed8f8b8/docs/kimi/causal-fp8-attention.md)

---

<a id="s-63"></a>
## Slide 63: Accuracy evidence is strong—but has a defined scope

The prototype passed full GSM8K; native-PR validation remains separate.

- **1,319:** GSM8K questions
- **96.9674%:** Strict: 1,279 correct
- **96.8916%:** Flexible: 1,278 correct
- **8 workers:** Optimized-path invocation verified



The new-token FP8 prototype's full evaluation used all 1,319 GSM8K questions, five-shot prompting, temperature zero and real speculative verification. It recorded successful cached FP8 projection, cached FP8 FMHA and causal FP8 calls on the same eight workers. Accuracy routing used eager execution with 512-token chunks to exercise the intended kernels; the normal performance run uses different routing and graphs. Controlled causal diagnostics passed, but the historical captured-BF16 absolute-LSE gate remains unpassed, with an observed discrepancy around 0.935 versus the unchanged 0.20 limit. The separately ported native vLLM PR has not inherited GPU validation merely because the prototype passed.

### Sources

- [New-token FP8 full GSM8K evaluation](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/40712931283)
- [Causal FP8 implementation and accuracy evidence](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/blob/b786b8cae9992c190c3d3323197c937d8ed8f8b8/docs/kimi/causal-fp8-attention.md)
- [Projection implementation and numerical limits](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/blob/b786b8cae9992c190c3d3323197c937d8ed8f8b8/docs/kimi/native-fp8-up-projection-audit.md)
- [Native vLLM draft PR](https://github.com/vllm-project/vllm/pull/64288)

---

<a id="s-64"></a>
## Slide 64: Execution proof is part of correctness

A passing test is useful only if it exercised the optimized path.

- **Pin the source:** Exact serving code and image
- **Verify operands:** Dtypes, layouts and scales
- **Observe calls:** Successful execution on all workers
- **Separate phases:** Warmup, capture, evaluation and replay



A feature flag can be enabled while an eligibility guard routes the actual workload to BF16. Short GSM8K prompts can fail the normal large-cached-context threshold, so a test might report good accuracy without exercising the optimized path. Execution proof should identify the exact code revision, loaded implementation, worker identity, relevant call, input/output dtypes and successful completion during the evaluation phase. Graph capture alone is not proof of steady-state replay. A prototype plugin and a native backend can have different routing or integration bugs even when they call related kernels. Preserve the fixed task and thresholds, and disclose any accuracy-only routing override.

### Sources

- [Causal FP8 implementation and accuracy evidence](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/blob/b786b8cae9992c190c3d3323197c937d8ed8f8b8/docs/kimi/causal-fp8-attention.md)
- [Projection implementation and numerical limits](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/blob/b786b8cae9992c190c3d3323197c937d8ed8f8b8/docs/kimi/native-fp8-up-projection-audit.md)
- [Native vLLM draft PR](https://github.com/vllm-project/vllm/pull/64288)

---

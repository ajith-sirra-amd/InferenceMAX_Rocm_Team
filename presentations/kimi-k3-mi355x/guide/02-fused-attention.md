# Fused attention

[Handbook contents](../HANDBOOK.md) · [Worked examples](11-worked-examples.md)

<a id="s-17"></a>
## Slide 17: FlashAttention changes data movement

Tile the computation and keep temporary scores close to the compute units.

- **Load a Q tile:** Keep queries in fast local storage
- **Stream K/V tiles:** Compute scores and update online softmax
- **Write the final output:** Avoid full score/probability matrices in HBM



FlashAttention is an exact attention algorithm in real arithmetic that reorganizes computation around the memory hierarchy. It processes tiles that fit in registers and shared memory, called LDS on AMD GPUs, while streaming K/V from HBM. It updates the running softmax statistics and output accumulator for each tile. This avoids writing and rereading the entire score and probability matrices in HBM. Floating-point results can differ from a naive reference because operation order changes. “Fused” means operations share a kernel and reuse local intermediates; it does not specify the input dtype. A BF16 FMHA and an FP8 FMHA can both be fused implementations.

### Sources

- [FlashAttention: exact attention with IO awareness](https://arxiv.org/abs/2205.14135)
- [AMD ROCm GPU architecture specifications](https://rocm.docs.amd.com/en/latest/reference/gpu-arch-specs.html)

---

<a id="s-18"></a>
## Slide 18: Online softmax carries three pieces of state

A running maximum, a normalization sum, and a weighted accumulator.

- **m:** Largest score seen so far
- **l:** Rescaled sum of exponentials
- **a:** Rescaled weighted value sum
- **Finish:** O = a/l;  LSE = m + ln(l)

m′ = max(m, max(s));  l′ = e^(m−m′)l + Σe^(s−m′)

For each new score tile s, the maximum may increase. Rescale the old normalization sum by exp(m-m') and add the new tile's exponentials relative to m'. Rescale the old weighted accumulator by the same factor, then add the new probability-weighted values. At the end, divide the accumulated weighted values by the normalization sum. This produces the same normalized attention operation without storing every score. LSE is the natural logarithm of the unnormalized attention mass, not another attention output vector. The exact intermediate dtypes are kernel-specific; higher precision for reductions is crucial, and the campaign API returns BF16 output plus FP32 LSE.

### Sources

- [FlashAttention: exact attention with IO awareness](https://arxiv.org/abs/2205.14135)
- [Campaign data-flow diagrams and measured stages](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/blob/b786b8cae9992c190c3d3323197c937d8ed8f8b8/docs/kimi/fp8-fmha-dataflow.md)

---

<a id="s-19"></a>
## Slide 19: A fused kernel can still lose end to end

Kernel speed must repay quantization, memory traffic and dispatch costs.

- **Kernel-only view:** Time just the new FMHA call.
- **Execution-path view:** Include preparation, quantization, layout changes and launches.
- **Serving view:** Include scheduling, communication, cache behavior and errors.

Net saving = time removed − time introduced

An FP8 matrix product can be faster than a BF16 matrix product while the complete serving path becomes slower. Preparing FP8 operands may require a maximum reduction, extra kernels and large temporary buffers. A layout conversion can add HBM traffic; a specialized kernel can have padding overhead or weaker occupancy on a particular shape. Synchronization may expose previously hidden work. For this reason, first measure the component, then the complete attention path, and finally the unchanged serving workload. This also explains why a prior FP8 ASM attempt could regress without disproving the potential of an FP8 fused implementation with lower preparation overhead.

### Sources

- [Campaign data-flow diagrams and measured stages](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/blob/b786b8cae9992c190c3d3323197c937d8ed8f8b8/docs/kimi/fp8-fmha-dataflow.md)
- [Projection implementation and numerical limits](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/blob/b786b8cae9992c190c3d3323197c937d8ed8f8b8/docs/kimi/native-fp8-up-projection-audit.md)

---

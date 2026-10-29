# MI355X execution

[Handbook contents](../HANDBOOK.md) · [Worked examples](11-worked-examples.md)

<a id="s-36"></a>
## Slide 36: MI355X supplies compute, capacity and bandwidth

The GPU must move tensors and execute them efficiently.

- **Architecture:** AMD CDNA 4 / gfx950
- **Compute structure:** 256 CUs across 8 XCDs
- **Memory:** 288 GB advertised HBM3E capacity
- **Execution group:** 64-lane wavefront



MI355X is an AMD Instinct accelerator built on CDNA 4. The ROCm specification table lists gfx950, 256 compute units, 32 CUs per XCD, and 64-lane wavefronts. HBM3E stores weights, persistent caches and large activations. XCD is a compute chiplet inside one GPU; it is not another rank or a separate accelerator. A GPU is a hierarchy of compute units and memories rather than a single undifferentiated matrix multiplier. Peak arithmetic rates are dtype-dependent, but peak specifications alone cannot predict a serving workload. This deck uses the vendor's advertised memory capacity convention and does not infer a measured bandwidth from it.

### Sources

- [AMD ROCm GPU architecture specifications](https://rocm.docs.amd.com/en/latest/reference/gpu-arch-specs.html)

---

<a id="s-37"></a>
## Slide 37: The memory hierarchy determines reuse

Registers and LDS are small and close; HBM is large and farther away.

- **Registers:** Per-thread / wave computation state
- **LDS:** On-chip workgroup-shared tiles
- **Caches:** Hardware-managed reuse
- **HBM:** Weights, KV, activations and workspaces



A fast matrix or attention kernel loads a tile from HBM, reuses it many times close to the compute units, and writes only the required result. Registers hold accumulators and local values; LDS supports data sharing among threads in a workgroup. Cache hits can reduce HBM traffic, so logical tensor sizes are not the same as measured HBM bytes. Excessive register or LDS use can reduce occupancy, limiting the number of active workgroups and the ability to hide latency. A fused kernel can remove intermediate HBM traffic but increase local resource pressure. That is why fusion is a tradeoff to measure rather than an automatic performance guarantee.

### Sources

- [AMD ROCm GPU architecture specifications](https://rocm.docs.amd.com/en/latest/reference/gpu-arch-specs.html)
- [FlashAttention: exact attention with IO awareness](https://arxiv.org/abs/2205.14135)

---

<a id="s-38"></a>
## Slide 38: A GEMM becomes tiles, waves and matrix instructions

Large matrix products reuse data across many multiply-accumulates.

- **Output tiles:** Workgroups own regions of C
- **Reduction tiles:** Stream matching A and B blocks
- **Matrix instructions:** MFMA operates on small fragments
- **Accumulators:** Higher precision before output conversion

C[M,N] = A[M,K] × B[K,N];  work ≈ 2MNK FLOPs

GEMM means general matrix multiplication. A GPU kernel divides the output into tiles, assigns them to workgroups and streams tiles of the reduction dimension K. Threads organized into wavefronts cooperate on loads and matrix instructions. MFMA is AMD's matrix fused multiply-accumulate instruction family. Input precision, tile shape, alignment, packing and accumulator precision affect which instructions are available. A large prefill M gives more reuse of weight tiles across tokens. A small decode M can leave work dominated by weight movement, launch overhead or inefficient tile occupancy. Transposition in a formula is a logical operation; the kernel may consume a packed layout without physically transposing a full tensor.

### Sources

- [AMD ROCm GPU architecture specifications](https://rocm.docs.amd.com/en/latest/reference/gpu-arch-specs.html)

---

<a id="s-39"></a>
## Slide 39: The roofline connects FLOPs to bytes

A kernel is limited by whichever resource it needs more urgently.

- **Arithmetic intensity:** Useful FLOPs / bytes moved
- **Bandwidth region:** Too little reuse to reach compute peak
- **Compute region:** Enough reuse; arithmetic throughput dominates

Attainable FLOP/s ≤ min(compute ceiling, bandwidth × intensity)

Arithmetic intensity is work per byte moved at a chosen memory level. With low intensity, a kernel cannot feed the compute units quickly enough, so attainable performance is bounded by bandwidth times intensity. With enough reuse, the compute ceiling dominates. The roofline is an upper-bound model, not a complete latency model: launch cost, synchronization, cache behavior, occupancy and dependencies can hold a kernel below either ceiling. FP8 may improve both arithmetic throughput and operand byte count, but only where the kernel can exploit the corresponding instructions and layouts. A small or communication-bound operation can gain little from faster arithmetic.

### Sources

- [AMD ROCm GPU architecture specifications](https://rocm.docs.amd.com/en/latest/reference/gpu-arch-specs.html)

---

<a id="s-40"></a>
## Slide 40: Why eight GPUs are needed

Sparse activation reduces work; the full weight set still consumes memory.

- **2.8T parameters:** Model-card total; not all activated per token
- **4-bit lower bound:** ≈ 1.4 TB before scales and higher-precision weights
- **8 × MI355X:** 2,304 GB advertised aggregate HBM
- **Remaining memory:** Cache, activations, graph pools and workspaces

Ideal weight bytes = parameter count × bits per parameter / 8

The simplest capacity estimate multiplies total parameters by bytes per parameter. At four bits, 2.8 trillion parameters would occupy about 1.4 trillion bytes. That is only a lower-bound illustration: the checkpoint leaves important modules in higher precision, and quantized weights require scales, packing metadata and sometimes runtime repacking. Eight MI355X accelerators provide 2,304 GB of advertised aggregate memory, but the memory is distributed across devices rather than a single automatically shared pool. Each rank needs its own local working memory and some replicated tensors. Activated parameter count estimates per-token model work; it must not replace total parameter count when estimating model residency.

### Sources

- [Kimi-K3 model card and checkpoint configuration](https://huggingface.co/moonshotai/Kimi-K3/tree/f831ab66814297da540d832a5235f8e904f29d06)
- [AMD ROCm GPU architecture specifications](https://rocm.docs.amd.com/en/latest/reference/gpu-arch-specs.html)

---

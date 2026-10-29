# Performance reasoning

[Handbook contents](../HANDBOOK.md) · [Worked examples](11-worked-examples.md)

<a id="s-65"></a>
## Slide 65: The three established performance stages

Total input + output tokens/s/GPU under the recorded full configurations.

- **Baseline:** 15,476.46881
- **Cached FP8 FMHA:** 16,915.72154
- **FP8 projection + FMHA:** 17,406.08912



The first established full result is 15,476.46881 total input-plus-output tokens per second per GPU. The cached FP8 FMHA stage measured 16,915.72154, about 9.30 percent higher. The direct FP8 projection plus cached FP8 FMHA stage measured 17,406.08912, about 2.90 percent above the preceding stage and 12.47 percent above the baseline. These are full benchmark configurations, including other tuning, so the deltas are not isolated kernel attribution. All values are preserved exactly in the companion evidence table. New-token causal FP8 has accuracy evidence but no completed accepted performance result; the last attempt failed because the runner exhausted its disk quota.

### Sources

- [Baseline benchmark](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/39239112271/job/118015427785)
- [Cached FP8 FMHA benchmark](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/40646139769)
- [FP8 projection plus cached FP8 FMHA benchmark](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/40695099743)
- [New-token performance attempt; failed on runner quota](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/40804432321)
- [Campaign data-flow diagrams and measured stages](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/blob/b786b8cae9992c190c3d3323197c937d8ed8f8b8/docs/kimi/fp8-fmha-dataflow.md)

---

<a id="s-66"></a>
## Slide 66: Read throughput and latency together

A higher token rate can coexist with worse user-visible responsiveness.

- **TTFT:** Time to first output token
- **ITL / TPOT:** Spacing or average time per output token
- **Output tok/s:** Actual generation rate
- **Total tok/s:** Input + output, including workload reuse semantics



TTFT includes waiting, prompt processing and production of the first token. ITL measures gaps between successive streamed tokens; TPOT is an average derived from output generation time and token count, and exact definitions should come from the benchmark. Tail percentiles reveal stalls hidden by averages. Total token throughput can be dominated by long input prefixes and cache reuse, while output throughput reflects generated tokens. Normalize by the actual GPU count, keep request errors visible, and compare equivalent durations and workload definitions. Concurrency sweeps should identify a throughput-latency operating point rather than maximizing a single scalar at any cost.

### Sources

- [Baseline benchmark](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/39239112271/job/118015427785)
- [FP8 projection plus cached FP8 FMHA benchmark](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/40695099743)

---

<a id="s-67"></a>
## Slide 67: Profile steady state before choosing the next kernel

Locate the critical path, then distinguish compute, memory and waiting.

- **Warmup excluded:** Model load, compilation and graph capture
- **Stable replay window:** Representative cache and request mix
- **All workers:** Kernels, copies, collectives and idle gaps
- **Unprofiled confirmation:** Measure final performance without tracing



A useful serving profile captures a representative stable window after startup and warmup. Record CPU scheduling, GPU kernels, memory copies, collectives and rank timing together. Kernel-duration sums can exceed wall time when kernels overlap; calculate interval unions for busy time and avoid double-counting nested annotations. Separate prefill, decode and mixed steps where possible, and do not infer a whole forward's cost from one kernel name. A long gap before a collective might reflect rank skew rather than slow arithmetic. Tracing perturbs execution, so instrumented throughput is diagnostic evidence and should not replace an unprofiled benchmark. Historical profiles can guide hypotheses, but final conclusions need the current optimized source.

### Sources

- [vLLM Kimi-K3 AMD implementation](https://github.com/ajith-sirra-amd/vllm/tree/08e5d81a0d4fc76c20a97f9a517195db5a5dc5be/vllm/models/kimi_k3/amd)
- [Campaign data-flow diagrams and measured stages](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/blob/b786b8cae9992c190c3d3323197c937d8ed8f8b8/docs/kimi/fp8-fmha-dataflow.md)

---

<a id="s-68"></a>
## Slide 68: Amdahl's law puts a ceiling on local speedups

The fastest kernel matters only for the fraction of time it can remove.

- **Fraction f:** Share of end-to-end time affected
- **Local speedup s:** Improvement to that portion
- **Remaining work:** Unaffected time still has to execute

Overall speedup = 1 / [(1 − f) + f/s]

If a kernel accounts for 20 percent of elapsed time and becomes twice as fast, total time becomes 80 percent plus 10 percent, giving a speedup of 1/0.9 or about 11.1 percent. It is not a 100 percent end-to-end improvement. If the optimization adds quantization, communication or synchronization, those costs must be included. The affected fraction must be measured on the relevant current workload; a percentage of GPU busy time is not automatically the same as a percentage of wall time. As one bottleneck improves, another can become dominant. This is why steady-state profiling should follow major execution-path changes.

### Sources

- [FlashAttention: exact attention with IO awareness](https://arxiv.org/abs/2205.14135)
- [Campaign data-flow diagrams and measured stages](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/blob/b786b8cae9992c190c3d3323197c937d8ed8f8b8/docs/kimi/fp8-fmha-dataflow.md)

---

<a id="s-69"></a>
## Slide 69: Choose the next experiment from the limiting resource

Change one hypothesis at a time and preserve comparable evidence.

- **Memory-bound:** Remove an intermediate; improve reuse or dtype.
- **Launch / sync-bound:** Fuse small work; inspect graph coverage and rank skew.
- **Batch / cache-bound:** Sweep C56 + MTP, C64 + MTP, C96 without MTP.



An experiment should state what resource is limiting, what change addresses it and what outcome would support the hypothesis. For the campaign, the planned concurrency points are C56 with MTP, C64 with MTP and C96 without MTP, preserving the existing CCD binding. Native NUMA changes are deferred. A valid comparison keeps the model, code, image, TP/DCP, workload, duration and relevant runtime settings fixed unless one is the deliberate independent variable. Read accuracy, errors and tail latency alongside throughput. Enabling another FP8 flag without establishing its eligibility, cost and numerical behavior is not a complete optimization experiment.

### Sources

- [Campaign data-flow diagrams and measured stages](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/blob/b786b8cae9992c190c3d3323197c937d8ed8f8b8/docs/kimi/fp8-fmha-dataflow.md)
- [Causal FP8 implementation and accuracy evidence](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/blob/b786b8cae9992c190c3d3323197c937d8ed8f8b8/docs/kimi/causal-fp8-attention.md)
- [Native vLLM draft PR](https://github.com/vllm-project/vllm/pull/64288)

---

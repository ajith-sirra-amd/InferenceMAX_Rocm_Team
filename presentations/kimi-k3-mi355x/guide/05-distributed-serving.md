# Distributed serving

[Handbook contents](../HANDBOOK.md) · [Worked examples](11-worked-examples.md)

<a id="s-41"></a>
## Slide 41: The serving stack turns requests into kernels

Every layer of software has a different responsibility.

- **Client + API:** Text, sessions and tool-call turns
- **vLLM scheduler:** Batching, cache blocks and speculation
- **PyTorch + backends:** Tensor operations and dispatch
- **AITER / FlyDSL / ROCm:** GPU kernels, runtime and collectives



The API layer handles request formatting, tokenization and output streaming. The scheduler chooses which request tokens run next and manages cache allocation and prefix reuse. The model implementation maps those tokens through KDA, MLA, MoE and residual operations. Backend dispatch selects kernels such as AITER attention, matrix multiplication, routing and collective operations. HIP is the ROCm runtime interface for launches, memory and streams. RCCL supplies collective communication, although inherited interfaces and kernel names can still say NCCL. FlyDSL is used to implement specialized kernels; it is not itself an attention mathematical definition. Enabling a flag only requests a path—the runtime's eligibility checks determine whether it executes.

### Sources

- [vLLM Kimi-K3 AMD implementation](https://github.com/ajith-sirra-amd/vllm/tree/08e5d81a0d4fc76c20a97f9a517195db5a5dc5be/vllm/models/kimi_k3/amd)
- [Campaign data-flow diagrams and measured stages](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/blob/b786b8cae9992c190c3d3323197c937d8ed8f8b8/docs/kimi/fp8-fmha-dataflow.md)
- [AMD ROCm GPU architecture specifications](https://rocm.docs.amd.com/en/latest/reference/gpu-arch-specs.html)

---

<a id="s-42"></a>
## Slide 42: Tensor parallelism splits model calculations

TP8 divides selected weight dimensions across eight ranks.

- **Column-parallel projection:** Each rank computes a subset of output features
- **Local heads / expert slices:** Work proceeds on each GPU
- **Row-parallel projection:** Each rank contributes a partial output
- **Collective reduction:** Combine partial results where required

96 heads / 8 TP ranks = 12 local prefill heads per rank

A rank is one participant in a distributed process group, typically associated with a GPU worker. Tensor parallelism shards dimensions of model weights and their calculations. For a column-parallel linear layer, ranks own different output features. A matching row-parallel layer consumes those partitions and produces partial contributions that must be combined. The exact collectives depend on the chosen implementation and possible fusion. TP8 does not imply that every tensor is divided evenly or that every expert is assigned wholly to one GPU. Some tensors are replicated, and MoE weights can be tensor-sharded even when expert parallelism is disabled.

### Sources

- [vLLM Kimi-K3 AMD implementation](https://github.com/ajith-sirra-amd/vllm/tree/08e5d81a0d4fc76c20a97f9a517195db5a5dc5be/vllm/models/kimi_k3/amd)
- [Kimi-K3 model card and checkpoint configuration](https://huggingface.co/moonshotai/Kimi-K3/tree/f831ab66814297da540d832a5235f8e904f29d06)

---

<a id="s-43"></a>
## Slide 43: DCP partitions the context across the same ranks

TP8 and DCP8 use eight GPUs here—not sixty-four.

- **Cache partition:** Distribute compressed token history across ranks
- **Attention work:** Read local context or gather chunks as required
- **Combine results:** Communicate and merge attention contributions
- **Same eight devices:** DCP operates within the existing TP group



Decode context parallelism distributes sequence-context storage and work among ranks in a tensor-parallel group. TP concerns model-feature dimensions; DCP concerns sequence-context ownership. Their sizes do not multiply the GPU count in this recipe. Under DCP8, different context rows belong to different ranks, subject to the configured interleave and block mapping. A decode implementation may exchange queries and merge local partial attention outputs. The campaign's cached-prefill route gathers compressed FP8 cache chunks before local up-projection and FMHA. Thus it is too simplistic to say that DCP always performs exactly one particular all-gather at every stage. Communication needs follow the chosen forward path.

### Sources

- [vLLM Kimi-K3 AMD implementation](https://github.com/ajith-sirra-amd/vllm/tree/08e5d81a0d4fc76c20a97f9a517195db5a5dc5be/vllm/models/kimi_k3/amd)
- [Campaign data-flow diagrams and measured stages](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/blob/b786b8cae9992c190c3d3323197c937d8ed8f8b8/docs/kimi/fp8-fmha-dataflow.md)

---

<a id="s-44"></a>
## Slide 44: Collectives move or combine distributed tensors

All-gather, all-reduce and all-to-all solve different problems.

- **All-gather:** Everyone receives all participants' pieces.
- **All-reduce:** Everyone receives a reduction, such as a sum.
- **All-to-all:** Each participant sends a different piece to each peer.



All-gather assembles distributed pieces, such as partitions needed by a subsequent computation. All-reduce combines contributions, often summing row-parallel outputs. All-to-all redistributes pieces according to destination, useful in some expert or context-parallel schemes. Communication performance depends on message size, topology, synchronization and rank skew. Small collectives are often latency-sensitive; large ones can be bandwidth-sensitive. A fast GPU may wait for a slower rank, so communication time cannot be diagnosed by bytes alone. Interconnect bandwidth is not the same as HBM bandwidth. Our diagrams show logical communication and intentionally avoid claiming an unverified physical link topology for the replacement server.

### Sources

- [vLLM Kimi-K3 AMD implementation](https://github.com/ajith-sirra-amd/vllm/tree/08e5d81a0d4fc76c20a97f9a517195db5a5dc5be/vllm/models/kimi_k3/amd)
- [AMD ROCm GPU architecture specifications](https://rocm.docs.amd.com/en/latest/reference/gpu-arch-specs.html)

---

<a id="s-45"></a>
## Slide 45: Merge attention chunks with their normalization mass

Average by exp(LSE), not by chunk length or a fixed 50/50 split.

- **Chunk A:** Output O_A and LSE_A
- **Chunk B:** Output O_B and LSE_B
- **Stable mass weights:** Normalize exponentials of the two LSEs
- **Merged output:** w_A O_A + w_B O_B

L = logaddexp(L_A,L_B);  O = e^(L_A−L)O_A + e^(L_B−L)O_B

Each separately computed attention chunk has normalized its own probabilities to sum to one. Those outputs cannot be simply added or averaged because the chunks may have very different total unnormalized attention mass. LSE tells us that mass in logarithmic form. Compute the log of the combined mass stably, then use exp(LSE_chunk-LSE_total) as each chunk's weight. The operation applies independently to each query and head. It supports merging cached-prefix attention with new-token causal attention, or local context partitions. An LSE error can therefore change the final output even when each chunk's normalized output appears reasonable. Log base and tensor layout must match the merge API.

### Sources

- [FlashAttention: exact attention with IO awareness](https://arxiv.org/abs/2205.14135)
- [Campaign data-flow diagrams and measured stages](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/blob/b786b8cae9992c190c3d3323197c937d8ed8f8b8/docs/kimi/fp8-fmha-dataflow.md)
- [Causal FP8 implementation and accuracy evidence](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/blob/b786b8cae9992c190c3d3323197c937d8ed8f8b8/docs/kimi/causal-fp8-attention.md)

---

<a id="s-46"></a>
## Slide 46: Continuous batching keeps requests moving

The scheduler admits new work as other requests finish.

- **Request A:** Decode next output token
- **Request B:** Process a new prompt chunk
- **Request C:** Reuse a prefix, then decode
- **Next step:** Rebuild the active token batch



Continuous batching lets the server change batch membership between model steps. Requests with different prompt lengths, output lengths and completion times share GPU work. Chunked prefill limits how many prompt tokens a single step processes, allowing decode progress between chunks. The max-num-batched-tokens setting bounds the token work scheduled in a step; max-num-seqs bounds sequence slots; concurrency is the offered client workload and is not automatically equal to either limit. Ragged batches use packed token tensors plus sequence-offset metadata. More concurrency can improve hardware utilization while worsening latency or cache pressure. A mixed step can delay decoding when a large prefill chunk dominates its duration.

### Sources

- [vLLM Kimi-K3 AMD implementation](https://github.com/ajith-sirra-amd/vllm/tree/08e5d81a0d4fc76c20a97f9a517195db5a5dc5be/vllm/models/kimi_k3/amd)

---

<a id="s-47"></a>
## Slide 47: Prefix reuse and offload trade compute for movement

Agentic turns often repeat a long prefix and append a small suffix.

- **HBM hit:** Reuse resident prefix blocks
- **Host-cache hit:** Load cached state from CPU DRAM
- **Miss:** Recompute the needed prefix
- **Hybrid state:** MLA cache and KDA state must agree



A multi-turn agent may repeatedly send a conversation or code context with a small appended tool result. Prefix caching recognizes reusable token blocks and avoids recomputing them. CPU-DRAM offload can retain more state than HBM alone, but moving it back costs bandwidth, synchronization and CPU work. For a hybrid KDA/MLA model, cache reuse also requires a consistent recurrent state at the reuse boundary; caching only an MLA row is not a complete hybrid-state solution. Cache misses, eviction, fragmentation and request dependency timing affect throughput. Raw input tokens counted by a benchmark can include reused prefix tokens, so input-plus-output throughput does not equal newly computed token throughput.

### Sources

- [vLLM Kimi-K3 AMD implementation](https://github.com/ajith-sirra-amd/vllm/tree/08e5d81a0d4fc76c20a97f9a517195db5a5dc5be/vllm/models/kimi_k3/amd)
- [Campaign data-flow diagrams and measured stages](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/blob/b786b8cae9992c190c3d3323197c937d8ed8f8b8/docs/kimi/fp8-fmha-dataflow.md)

---

<a id="s-48"></a>
## Slide 48: MTP proposes several tokens; the target verifies them

Speed depends on accepted progress per complete draft-and-verify cycle.

- **Draft:** Propose a block of candidate tokens
- **Verify:** Target evaluates the candidates together
- **Accept / correct:** Keep valid progress; handle rejection
- **Commit state:** Advance KV and recurrent state consistently

Useful speedup ≈ accepted progress per cycle / relative cycle cost

Multi-token prediction and speculative decoding try to amortize expensive target-model execution across several output tokens. In this campaign, the serving recipe uses an external Kimi-K3 DSpark draft model. The target checkpoint has num_nextn_predict_layers=0, so do not imply it contains the active draft module used by the benchmark. A correct speculative algorithm must use the configured acceptance/rejection scheme and commit only the appropriate token and recurrent-state progress. High acceptance can help, but drafting, verification, communication and state recovery also cost time. The performance harness uses prescribed synthetic acceptance behavior, while GSM8K accuracy uses real speculative verification; those are different evidence types.

### Sources

- [Kimi-K3 model card and checkpoint configuration](https://huggingface.co/moonshotai/Kimi-K3/tree/f831ab66814297da540d832a5235f8e904f29d06)
- [Causal FP8 implementation and accuracy evidence](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/blob/b786b8cae9992c190c3d3323197c937d8ed8f8b8/docs/kimi/causal-fp8-attention.md)
- [vLLM Kimi-K3 AMD implementation](https://github.com/ajith-sirra-amd/vllm/tree/08e5d81a0d4fc76c20a97f9a517195db5a5dc5be/vllm/models/kimi_k3/amd)

---

<a id="s-49"></a>
## Slide 49: CPU placement and GPU graphs address dispatch overhead

Locality and launch reuse matter when individual GPU operations are short.

- **NUMA / CCD binding:** Place worker CPU threads near relevant memory and devices.
- **Graph replay:** Reuse a captured GPU launch dependency graph.
- **Tradeoffs:** Affinity needs topology; graph pools consume persistent memory.



NUMA describes different CPU-memory locality domains. A CCD is an AMD CPU compute die, distinct from a GPU XCD. Binding worker threads to suitable cores can reduce migration, contention and remote-memory access, but core choices must come from the actual server topology. This campaign preserves its existing CCD pinning; native NUMA experiments are deferred. GPU graph replay reduces repeated host-side launch overhead by replaying a captured set of operations. ROCm uses HIP graph facilities, even when vLLM configuration or PyTorch APIs retain CUDA naming. Graphs require stable buffers and supported shapes/control flow; capture pools and intermediate storage can consume memory otherwise available for caches.

### Sources

- [AMD ROCm GPU architecture specifications](https://rocm.docs.amd.com/en/latest/reference/gpu-arch-specs.html)
- [vLLM Kimi-K3 AMD implementation](https://github.com/ajith-sirra-amd/vllm/tree/08e5d81a0d4fc76c20a97f9a517195db5a5dc5be/vllm/models/kimi_k3/amd)

---

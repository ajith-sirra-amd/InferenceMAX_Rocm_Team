# Kimi-K3 architecture

[Handbook contents](../HANDBOOK.md) · [Worked examples](11-worked-examples.md)

<a id="s-20"></a>
## Slide 20: Kimi-K3 is a hybrid model

Different mechanisms share the work of understanding long context.

- **KDA:** 69 layers: recurrent sequence mixing
- **Gated MLA:** 24 layers: compressed-cache attention
- **Latent MoE:** Sparse expert computation
- **Attention Residuals:** Learned mixing across model depth



The checkpoint combines Kimi Delta Attention with gated Multi-head Latent Attention. KDA handles most sequence-mixing layers using a recurrent state, while MLA periodically performs attention over cached token representations. The feature transformation side uses one dense layer followed by sparse mixture-of-experts layers, with a lower-dimensional routed-expert space. Attention Residuals govern information flow across depth. A vision encoder and projector can supply visual tokens, but the agentic text benchmark does not measure the full vision pipeline. The model card reports about 2.8 trillion total parameters and 104 billion activated parameters; those figures do not mean all weights are read for every token.

### Sources

- [Kimi-K3 model card and checkpoint configuration](https://huggingface.co/moonshotai/Kimi-K3/tree/f831ab66814297da540d832a5235f8e904f29d06)
- [vLLM Kimi-K3 AMD implementation](https://github.com/ajith-sirra-amd/vllm/tree/08e5d81a0d4fc76c20a97f9a517195db5a5dc5be/vllm/models/kimi_k3/amd)

---

<a id="s-21"></a>
## Slide 21: The actual 93-layer pattern

Mostly three KDA layers followed by one MLA layer, then a final MLA.

- **Layers 1–92:** 23 groups of KDA → KDA → KDA → MLA
- **Layer 93:** One additional MLA layer
- **Feed-forward side:** Layer 1 dense; later layers use MoE

69 KDA + 24 MLA = 93 sequence-mixing layers

The checkpoint lists full-attention layers using one-based indices: 4,8,12 and so on through 92, plus layer 93. The remaining 69 layers use KDA. Therefore a drawing that shows every layer as ordinary softmax attention would misrepresent this architecture. The first_k_dense_replace setting is 1, so the first decoder layer uses the dense feed-forward path and subsequent layers use the configured MoE path. Attention type and feed-forward type are separate choices within a decoder layer. The final MLA layer is an exception to the repeating four-layer pattern. Source arrays are included in the checkpoint snapshot shipped with the handbook.

### Sources

- [Kimi-K3 model card and checkpoint configuration](https://huggingface.co/moonshotai/Kimi-K3/tree/f831ab66814297da540d832a5235f8e904f29d06)
- [vLLM Kimi-K3 AMD implementation](https://github.com/ajith-sirra-amd/vllm/tree/08e5d81a0d4fc76c20a97f9a517195db5a5dc5be/vllm/models/kimi_k3/amd)

---

<a id="s-22"></a>
## Slide 22: Checkpoint dimensions worth remembering

These numbers describe the referenced Kimi-K3 checkpoint.

- **7,168:** Hidden-state width
- **96:** Attention heads
- **512:** Compressed KV latent width
- **1,536:** Query latent width
- **896 / 16:** Routed experts / selected per token
- **1,048,576:** Configured maximum context



Hidden-state width is the size of each token's representation between sublayers. The query projection factors through a 1,536-dimensional latent, and MLA KV factors through a 512-dimensional latent plus a 64-column shared key component. There are 96 logical heads. The routed expert branch narrows the 7,168-dimensional hidden state to 3,584 features before expert computation; each routed expert has intermediate width 3,072. Two shared experts operate in addition to the selected routed experts. The configured context limit is an architectural/configuration limit, not proof that a particular GPU-memory budget, concurrency and offload setting can serve every request at that length.

### Sources

- [Kimi-K3 model card and checkpoint configuration](https://huggingface.co/moonshotai/Kimi-K3/tree/f831ab66814297da540d832a5235f8e904f29d06)

---

<a id="s-23"></a>
## Slide 23: KDA keeps a learned summary of the past

Update a state matrix instead of storing a growing key/value list.

- **Previous state:** Associative memory from earlier tokens
- **Current token:** Query, key, value and learned gates
- **Updated state:** Forget selectively; correct remembered content
- **Output:** Read the updated state with the query



Kimi Delta Attention belongs to the family of recurrent linear-attention mechanisms. Instead of reading an ever-growing list of past key/value rows for every step, it maintains a matrix state for each head. The current key addresses that state, the current value supplies content, and learned gates control decay and update strength. A delta update corrects the mismatch between the content currently retrieved by the key and the desired value. This bounded state is efficient for long-sequence processing, but it is a compressed memory: it does not preserve an independently addressable row for every past token. Periodic MLA layers provide complementary explicit attention over token history.

### Sources

- [Kimi-K3 model card and checkpoint configuration](https://huggingface.co/moonshotai/Kimi-K3/tree/f831ab66814297da540d832a5235f8e904f29d06)
- [Kimi Linear: Kimi Delta Attention](https://arxiv.org/abs/2510.26692)
- [vLLM Kimi-K3 AMD implementation](https://github.com/ajith-sirra-amd/vllm/tree/08e5d81a0d4fc76c20a97f9a517195db5a5dc5be/vllm/models/kimi_k3/amd)

---

<a id="s-24"></a>
## Slide 24: KDA's update, one step at a time

Decay → predict → correct → read.

- **Decay:** S̄ = diag(α) S_previous
- **Correct:** r = v − S̄ᵀk;   S = S̄ + βkrᵀ
- **Read:** o = Sᵀq

S ∈ R^(d_k × d_v);  α controls forgetting;  β controls update strength

This is an explanatory column-vector convention for the gated delta rule, not a literal transcription of a fused kernel. The state S maps a key-space vector to a value-space vector. First apply a learned decay to obtain Sbar. Query that decayed memory with the current key k, compare the retrieved value with the desired current value v, and form the residual r. The rank-one update beta times k times r transpose writes the correction. Finally the query q reads the updated state. Actual KDA includes normalization, learned gate parameterizations, short convolutions, output gating and chunked algorithms; those details preserve the conceptual sequence while changing efficient implementation.

### Sources

- [Kimi Linear: Kimi Delta Attention](https://arxiv.org/abs/2510.26692)
- [vLLM Kimi-K3 AMD implementation](https://github.com/ajith-sirra-amd/vllm/tree/08e5d81a0d4fc76c20a97f9a517195db5a5dc5be/vllm/models/kimi_k3/amd)

---

<a id="s-25"></a>
## Slide 25: KDA prefill and decode use different algorithms

The state is recurrent; processing a known chunk can still be parallelized.

- **Prefill:** Chunked matrix operations process known token blocks.
- **Decode:** Update a small persistent state for each arriving token.
- **Speculative verification:** Preserve state checkpoints and commit only accepted progress.



A recurrence does not require a GPU implementation to process every prompt token with an isolated tiny kernel. Chunked formulations express portions of the computation as matrix operations and carry a state between chunks. Decode benefits from a compact update because only a small number of tokens arrive. Kimi-K3 also uses a short causal convolution of width four before the recurrent mixing, so its persistent cache includes convolution history as well as the main state. Speculative generation creates an additional correctness issue: rejected candidate tokens must not permanently modify either state. Restore or checkpoint strategies must keep accepted and rejected histories consistent.

### Sources

- [Kimi-K3 model card and checkpoint configuration](https://huggingface.co/moonshotai/Kimi-K3/tree/f831ab66814297da540d832a5235f8e904f29d06)
- [vLLM Kimi-K3 AMD implementation](https://github.com/ajith-sirra-amd/vllm/tree/08e5d81a0d4fc76c20a97f9a517195db5a5dc5be/vllm/models/kimi_k3/amd)
- [Kimi Linear: Kimi Delta Attention](https://arxiv.org/abs/2510.26692)

---

<a id="s-26"></a>
## Slide 26: MLA compresses the persistent token cache

Store a shared latent per token, then reconstruct what attention needs.

- **Hidden state:** 7,168 features
- **KV compression:** 512 latent + 64 shared K features
- **Persistent cache:** 576 values per token / MLA layer
- **Attention representation:** Per-head K and V when needed



MLA means Multi-head Latent Attention. A learned down-projection transforms each token's hidden state into a compact shared KV latent. The checkpoint also retains a 64-feature shared key component in the layout field historically named RoPE. In this NoPE checkpoint, do not interpret that name as an active positional rotation. The persistent cache stores the compressed representation rather than separate expanded K/V for every head. Prefill can reconstruct per-head K/V to use an efficient FMHA kernel. Some decode algorithms instead absorb projection matrices into queries and outputs, retaining computation in latent space. Compression is a learned model parameterization, distinct from FP8 numeric quantization.

### Sources

- [Kimi-K3 model card and checkpoint configuration](https://huggingface.co/moonshotai/Kimi-K3/tree/f831ab66814297da540d832a5235f8e904f29d06)
- [Campaign data-flow diagrams and measured stages](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/blob/b786b8cae9992c190c3d3323197c937d8ed8f8b8/docs/kimi/fp8-fmha-dataflow.md)
- [vLLM Kimi-K3 AMD implementation](https://github.com/ajith-sirra-amd/vllm/tree/08e5d81a0d4fc76c20a97f9a517195db5a5dc5be/vllm/models/kimi_k3/amd)

---

<a id="s-27"></a>
## Slide 27: Query preparation also uses a latent projection

The current token's query is expanded into per-head features.

- **Hidden X:** Width 7,168
- **q_a + normalization:** Query latent width 1,536
- **q_b projection:** 96 × 192 query features
- **TP8 local Q:** [Tq, 12, 192]



The query branch projects the hidden state down to a 1,536-dimensional latent, normalizes it, then applies q_b_proj to obtain per-head query features. In the local TP8 prefill representation, Q has 12 heads and width 192 per head. That width consists of 128 non-shared key-matching features plus the 64-column shared-key layout component. The implementation can fuse the initial Q/KV projections or gate preparation, so these boxes describe logical transformations rather than guaranteed individual kernels. Query preparation is not the same operation as kv_b_proj: q_b_proj expands the current query latent, while kv_b_proj reconstructs key/value features from compressed cached KV.

### Sources

- [Kimi-K3 model card and checkpoint configuration](https://huggingface.co/moonshotai/Kimi-K3/tree/f831ab66814297da540d832a5235f8e904f29d06)
- [vLLM Kimi-K3 AMD implementation](https://github.com/ajith-sirra-amd/vllm/tree/08e5d81a0d4fc76c20a97f9a517195db5a5dc5be/vllm/models/kimi_k3/amd)
- [Campaign data-flow diagrams and measured stages](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/blob/b786b8cae9992c190c3d3323197c937d8ed8f8b8/docs/kimi/fp8-fmha-dataflow.md)

---

<a id="s-28"></a>
## Slide 28: kv_b_proj reconstructs per-head keys and values

The weight matrix is neither the KV cache nor an FMHA input.

- **Compressed C:** [Tk, 512]
- **Local W_kv:** [3,072, 512]
- **Projected Z:** [Tk, 12, 256] → K_base + V
- **Assembled tensors:** K [Tk,12,192]; V [Tk,12,128]

Z = C W_kvᵀ;  3,072 = 12 × (128 + 128)

Under TP8, the projection weight has 3,072 output rows because each of 12 local heads needs 128 key features and 128 value features. Multiplication produces 256 features per head, which are split into the non-shared key part and the value. The 64-feature cached shared-key component is appended to every head's key, giving final key width 192. Thus C, W_kv, expanded K/V and attention output are four distinct objects. FMHA receives Q, K and V plus scales and metadata; it does not receive the kv_b_proj weight. Kernel fusion can perform row gathering, projection, splitting and assembly without materializing all of these logical intermediates.

### Sources

- [Campaign data-flow diagrams and measured stages](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/blob/b786b8cae9992c190c3d3323197c937d8ed8f8b8/docs/kimi/fp8-fmha-dataflow.md)
- [vLLM Kimi-K3 AMD implementation](https://github.com/ajith-sirra-amd/vllm/tree/08e5d81a0d4fc76c20a97f9a517195db5a5dc5be/vllm/models/kimi_k3/amd)

---

<a id="s-29"></a>
## Slide 29: MLA decode can avoid expanded K/V

Associativity lets projections move across the attention calculation.

- **Explicit prefill path:** Build K/V per head, then call FMHA.
- **Latent decode path:** Transform Q into latent space; attend to compressed cache.
- **After the latent read:** Apply the value projection to obtain each head's output.

q(CW_Kᵀ)ᵀ = (qW_K)Cᵀ

For the non-shared key part, projecting cached latent rows C to K and then taking q dot K is algebraically equivalent to first multiplying q by the key-projection weight and then taking its dot product with C. Similarly, multiplying attention probabilities by C and then by the value-projection weight is equivalent to using expanded values. These identities motivate MLA decode implementations that avoid expanding the entire history. Shared key components and masks still need their own correct handling. The performance tradeoff depends on query count and matrix shapes, so prefill and decode can legitimately choose different realizations of the same model. The direct FP8 up-projection experiment in this deck targets the expanded cached-prefill route.

### Sources

- [vLLM Kimi-K3 AMD implementation](https://github.com/ajith-sirra-amd/vllm/tree/08e5d81a0d4fc76c20a97f9a517195db5a5dc5be/vllm/models/kimi_k3/amd)
- [Campaign data-flow diagrams and measured stages](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/blob/b786b8cae9992c190c3d3323197c937d8ed8f8b8/docs/kimi/fp8-fmha-dataflow.md)

---

<a id="s-30"></a>
## Slide 30: Gated MLA controls the attention output

A learned gate modulates retrieved features before the next projection.

- **FMHA output:** Weighted information from context
- **Learned output gate:** Feature-wise modulation
- **Output projection:** Return to hidden width
- **Residual mixing:** Carry information into the next sublayer



The checkpoint enables mla_use_output_gate. An output gate is different from a causal mask and different from an MoE router. The causal mask decides which token pairs are allowed. The attention softmax determines relative weights over allowed keys. The output gate modulates the retrieved features, typically through a learned bounded nonlinearity, and the output projection mixes head outputs into the model hidden space. Implementations may fuse gate preparation with an earlier projection and fuse gate application with output processing. Gate shape, dtype and operation order must still follow the reference path when optimizing a kernel.

### Sources

- [Kimi-K3 model card and checkpoint configuration](https://huggingface.co/moonshotai/Kimi-K3/tree/f831ab66814297da540d832a5235f8e904f29d06)
- [vLLM Kimi-K3 AMD implementation](https://github.com/ajith-sirra-amd/vllm/tree/08e5d81a0d4fc76c20a97f9a517195db5a5dc5be/vllm/models/kimi_k3/amd)

---

<a id="s-31"></a>
## Slide 31: MoE selects a small subset of a large expert pool

Each token activates 16 routed experts, plus the shared branch.

- **Router:** Compute scores for 896 routed experts
- **Select + weight:** Top 16 per token; normalize selected weights
- **Experts:** Different learned feature transformations
- **Combine:** Weighted routed output + shared output



A mixture-of-experts layer contains many learned feed-forward networks. A router scores experts for each token, selects a configured subset and combines their outputs. Kimi-K3 uses sigmoid router scores, a correction-bias mechanism for selection, and renormalized selected weights in the inspected implementation. The routing logits and correction bias are handled in FP32. Sixteen out of 896 routed experts is about 1.79 percent of the pool per token, but a batch of many tokens can collectively touch many more experts. Sparse activation reduces compute relative to running every expert; it does not remove the need to store the full expert pool somewhere accessible.

### Sources

- [Kimi-K3 model card and checkpoint configuration](https://huggingface.co/moonshotai/Kimi-K3/tree/f831ab66814297da540d832a5235f8e904f29d06)
- [vLLM Kimi-K3 AMD implementation](https://github.com/ajith-sirra-amd/vllm/tree/08e5d81a0d4fc76c20a97f9a517195db5a5dc5be/vllm/models/kimi_k3/amd)

---

<a id="s-32"></a>
## Slide 32: Latent MoE reduces the routed branch's width

Experts operate in a smaller feature space, then return to model width.

- **Hidden state:** 7,168
- **Routed down-projection:** 3,584 latent features
- **Selected experts:** Intermediate width 3,072 each
- **Normalize + up-project:** Back to 7,168

routed(x) = W_up · norm(Σ_e w_e · Expert_e(W_down x))

The routed branch uses a learned down-projection from hidden width 7,168 to routed-expert width 3,584. Selected experts transform that lower-dimensional representation. Their weighted combination is normalized when latent_moe_use_norm is enabled and projected back to hidden width. The router itself scores from the original hidden representation in the inspected code. Shared experts operate on the original hidden space and form a parallel branch. Do not confuse this routed-expert up-projection with attention kv_b_proj: they both expand features but have different weights, inputs, mathematical roles and performance costs. Runtime implementations can fuse parts of the routed tail with communication or residual processing.

### Sources

- [Kimi-K3 model card and checkpoint configuration](https://huggingface.co/moonshotai/Kimi-K3/tree/f831ab66814297da540d832a5235f8e904f29d06)
- [vLLM Kimi-K3 AMD implementation](https://github.com/ajith-sirra-amd/vllm/tree/08e5d81a0d4fc76c20a97f9a517195db5a5dc5be/vllm/models/kimi_k3/amd)

---

<a id="s-33"></a>
## Slide 33: SiTU-GLU is the expert's gated nonlinearity

Two projection branches interact, with soft clipping to control range.

- **Gate branch:** g = 4 tanh(a/4) · sigmoid(a)
- **Value branch:** u = 25 tanh(b/25)
- **Multiply + project:** z = g ⊙ u; then apply the down-projection

Expert(x) = W_down [SiTU(W_gate x) ⊙ softclip(W_up x)]

Kimi-K3's hidden_act is situ, with activation_situ_beta=4 and activation_situ_linear_beta=25. The inspected SiTU implementation computes beta*tanh(gate/beta)*sigmoid(gate), and softly clips the up branch with linear_beta*tanh(up/linear_beta). It then multiplies the two branches elementwise. This is a gated linear unit variant, not simply an ordinary ReLU or a plain SiLU. The clipping is smooth, which shapes the activation range without a hard discontinuity. The two configured shared experts are implemented as a combined shared feed-forward branch with intermediate width 2*3,072=6,144. The first dense layer has intermediate width 33,792.

### Sources

- [Kimi-K3 model card and checkpoint configuration](https://huggingface.co/moonshotai/Kimi-K3/tree/f831ab66814297da540d832a5235f8e904f29d06)
- [SiTU-GLU reference implementation](https://github.com/ajith-sirra-amd/vllm/blob/08e5d81a0d4fc76c20a97f9a517195db5a5dc5be/vllm/model_executor/layers/activation.py)
- [vLLM Kimi-K3 AMD implementation](https://github.com/ajith-sirra-amd/vllm/tree/08e5d81a0d4fc76c20a97f9a517195db5a5dc5be/vllm/models/kimi_k3/amd)

---

<a id="s-34"></a>
## Slide 34: Attention Residuals attend across depth

Learn which earlier layer-block representations should contribute.

- **Earlier blocks:** Stored depth representations
- **Learned weights:** Content-dependent mixing over available blocks
- **Current sublayer:** Consumes the mixed representation
- **Block size:** Configured as 12 layers



Ordinary residual addition sums a previous representation and a new transformation with a fixed structure. Attention Residuals add learned mixing over available representations across depth. In the inspected Kimi-K3 implementation, block-state storage and learned projection/norm operations produce the residual input for attention and feed-forward sublayers. This is a different axis from token attention: token attention retrieves across sequence positions, while Attention Residuals mix across model depth. The checkpoint uses attn_res_block_size=12. Block storage limits the overhead compared with keeping every sublayer output independently. GPU execution introduces extra norm, scoring, weighted-sum and state-management work that can be targets for fusion.

### Sources

- [Kimi-K3 model card and checkpoint configuration](https://huggingface.co/moonshotai/Kimi-K3/tree/f831ab66814297da540d832a5235f8e904f29d06)
- [vLLM Kimi-K3 AMD implementation](https://github.com/ajith-sirra-amd/vllm/tree/08e5d81a0d4fc76c20a97f9a517195db5a5dc5be/vllm/models/kimi_k3/amd)

---

<a id="s-35"></a>
## Slide 35: Vision joins the model through visual tokens

A vision encoder and projector connect images/video to the text backbone.

- **Image / video:** Patch and frame preparation
- **MoonViT-V2:** 27 layers; hidden width 1,024
- **Patch merging + projector:** Map visual features to width 7,168
- **Hybrid backbone:** Reason over visual and text representations



The model is natively multimodal, but visual input has an additional processing path. The checkpoint vision configuration uses 14-pixel patches, 27 encoder layers, width 1,024 and 12 vision attention heads, followed by patch merging and a projector that matches the text hidden width of 7,168. The model card names this encoder MoonViT-V2 and reports about 401 million vision parameters. Dynamic image preprocessing, spatial/temporal attention and patch merging affect the number and cost of visual tokens. The campaign's agentic text throughput figures should not be read as image or video throughput. Exact multimodal packing details belong to the processor implementation.

### Sources

- [Kimi-K3 model card and checkpoint configuration](https://huggingface.co/moonshotai/Kimi-K3/tree/f831ab66814297da540d832a5235f8e904f29d06)
- [vLLM Kimi-K3 AMD implementation](https://github.com/ajith-sirra-amd/vllm/tree/08e5d81a0d4fc76c20a97f9a517195db5a5dc5be/vllm/models/kimi_k3/amd)

---

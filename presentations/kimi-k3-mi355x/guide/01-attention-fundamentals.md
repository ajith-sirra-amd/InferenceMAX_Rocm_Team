# Attention fundamentals

[Handbook contents](../HANDBOOK.md) · [Worked examples](11-worked-examples.md)

<a id="s-03"></a>
## Slide 03: Attention asks: which earlier information matters?

A token builds its next representation by reading relevant context.

- **Current token:** “it” needs context
- **Earlier tokens:** “The robot moved the box because it was heavy”
- **Weighted read:** Different context tokens contribute different amounts



Attention is a learned retrieval-and-mixing operation. At each layer a token has a vector describing its current representation. It compares a query derived from that vector with keys from available context tokens. Those comparisons become weights, and the weights mix the corresponding values. In the example, resolving the meaning of “it” can require information about the robot, the box and the surrounding words. The illustration is conceptual, not an actual measured attention map. A head's weights are not a complete explanation of the model's reasoning, and different layers may use attention for different purposes.

### Sources

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)

---

<a id="s-04"></a>
## Slide 04: Tokens become vectors; vectors become matrices

A token is an ID. Its hidden state is a learned list of numbers.

- **Text:** Subword tokenization
- **Token IDs:** Indices into an embedding table
- **Hidden states X:** T rows × d_model columns
- **Kimi-K3 example:** d_model = 7,168

X ∈ R^(T × d_model)

A tokenizer turns text into discrete token IDs. A token is not always a word: common words, word fragments, punctuation and whitespace patterns may each have IDs. The embedding table maps each ID to a vector. After the first layer, the hidden state is contextual rather than a fixed lookup. Pack T token vectors as rows of X so a GPU can process them with matrix operations. A matrix's shape describes its logical dimensions; dtype describes how each number is represented; layout and strides describe where the numbers live in memory. The Kimi-K3 vocabulary has 163,840 entries and hidden width 7,168.

### Sources

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [Kimi-K3 model card and checkpoint configuration](https://huggingface.co/moonshotai/Kimi-K3/tree/f831ab66814297da540d832a5235f8e904f29d06)

---

<a id="s-05"></a>
## Slide 05: Q, K and V are three learned projections

Query = what to look for. Key = how to match. Value = what to retrieve.

- **Q = XW_Q:** One query per query token
- **K = XW_K:** One key per context token
- **V = XW_V:** Content to mix into the output



A projection is a matrix multiplication with learned weights. For row-vector hidden states, Q=XW_Q, K=XW_K and V=XW_V. The same token can play all three roles in self-attention. Q and K need matching feature widths so their dot product is defined. V may have a different width. Query count Tq and key/value count Tk need not be equal: during decode a request may have one new query but thousands of cached keys. Projection weights are model parameters; Q, K and V are runtime activations. In MLA, the implementation factors some of these projections through smaller latent vectors, which we will introduce later.

### Sources

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)

---

<a id="s-06"></a>
## Slide 06: The dot product produces a matching score

Aligned features reinforce each other; unrelated features contribute less.

- **Multiply features:** q = [1, 2],  k = [3, 1]
- **Add the products:** q · k = 1×3 + 2×1 = 5
- **Control the scale:** Usually divide by √d_k

score(q, k) = (q · k) / √d_k

A dot product multiplies matching coordinates and sums the results. The raw score is not a probability and can be negative. If feature components have similar independent variance, summing more dimensions makes scores spread out more. The usual inverse-square-root factor compensates for that tendency and avoids overly saturated softmax. The general formula uses 1/sqrt(d_k), but an implementation should use the model's actual configured softmax scale, called alpha in later slides. Do not hard-code a new scale because a tensor was quantized. Quantization descales and the model softmax scale serve different purposes.

### Sources

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)

---

<a id="s-07"></a>
## Slide 07: Softmax turns scores into relative weights

Subtract the maximum, exponentiate, and normalize each query row.

- **Scores:** [2, 1, 0]
- **Stable exponentials:** [1, 0.368, 0.135]
- **Weights:** [0.665, 0.245, 0.090]

p_j = exp(s_j − m) / Σ_i exp(s_i − m),  m = max_i s_i

Softmax converts a row of scores into positive weights that sum to one. Subtracting the largest score leaves the probabilities unchanged because the shared exponential factor cancels. This subtraction prevents large positive exponentials from overflowing. The example subtracts 2 from [2,1,0] before exponentiation. Softmax is relative: increasing one score reduces other tokens' share even if their scores stay fixed. Masked entries are assigned negative infinity so their exponential contribution is zero. The log of the normalization sum, with the maximum added back, is called log-sum-exp or LSE; it is essential for exact merging of separately processed attention chunks.

### Sources

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [FlashAttention: exact attention with IO awareness](https://arxiv.org/abs/2205.14135)

---

<a id="s-08"></a>
## Slide 08: The output is a weighted sum of values

Attention returns a vector—not the index of the best-matching token.

- **V₁ = [10, 0]:** Weight 0.665
- **V₂ = [0, 10]:** Weight 0.245
- **V₃ = [5, 5]:** Weight 0.090

O = 0.665V₁ + 0.245V₂ + 0.090V₃ ≈ [7.10, 2.90]

Each attention weight multiplies the complete value vector, and the weighted vectors are summed. This is a soft read rather than an argmax lookup. A value need not resemble the word embedding of its token: it is a learned transformation of a contextual hidden state. In this two-dimensional example, the dominant value contributes mostly to the first output coordinate. With many tokens and heads, the same principle becomes the matrix product P times V. The output is then combined across heads and projected back to the model hidden width. Errors in Q and K can change the weights; errors in V change the content mixed by those weights.

### Sources

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)

---

<a id="s-09"></a>
## Slide 09: Causal masking prevents future-token reads

A token can see the existing prefix and its own allowed position.

- **Rows are queries:** One row for each new token
- **Columns are keys:** One column for each available token
- **Future entries:** Masked before softmax

M[i,j] = 0 for allowed keys; otherwise −∞

In autoregressive generation, a token's prediction must not use information from future target tokens. For a sequence without a cached prefix, the allowed region is lower triangular, including the diagonal. With an existing prefix of length P and a new chunk of length N, new query i can attend to all prefix tokens and new tokens up to i. The cached-prefix attention portion therefore needs no causal triangle within that prefix block, while the new-token portion does. A backend's causal alignment when Tq differs from Tk must be checked carefully; setting causal=True is not sufficient unless the sequence offsets and alignment match the intended mask.

### Sources

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [Causal FP8 implementation and accuracy evidence](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/blob/b786b8cae9992c190c3d3323197c937d8ed8f8b8/docs/kimi/causal-fp8-attention.md)

---

<a id="s-10"></a>
## Slide 10: One equation joins the complete operation

Two matrix products surround the softmax.

- **QKᵀ:** Tq × Tk score matrix
- **Scale + mask:** Apply model scale and visibility
- **Softmax:** Normalize each query row
- **PV:** Tq × d_v output

O = softmax(αQKᵀ + M)V

For one head, Q has shape [Tq,d_k], K has shape [Tk,d_k], and V has shape [Tk,d_v]. QK transpose has shape [Tq,Tk], one score for every query-key pair. Softmax works across the key dimension independently for each query. Multiplying the resulting probabilities by V produces [Tq,d_v]. A transpose changes how dimensions are interpreted; it need not imply a materialized copy. We often write the full score and probability matrices to understand the math, but an efficient FlashAttention implementation does not store them in global memory. Batch dimensions, ragged sequence offsets and head dimensions are omitted from this equation for clarity.

### Sources

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [FlashAttention: exact attention with IO awareness](https://arxiv.org/abs/2205.14135)

---

<a id="s-11"></a>
## Slide 11: Multiple heads learn multiple kinds of relationships

Each head produces its own weighted read; an output projection mixes them.

- **Independent heads:** Distinct learned Q/K/V projections
- **Parallel reads:** Different patterns can coexist
- **Output projection:** Concatenate head outputs → model width

Y = concat(O₁, …, O_H) W_O

Multi-head attention gives the model several learned subspaces in which to retrieve information. Heads do not have guaranteed human-readable roles, but they can learn different relationships. In the Kimi-K3 MLA prefill path, there are 96 logical heads, giving 12 local heads per rank under TP8. Head count is distinct from the number of requests, GPU threads or attention layers. Multi-query attention and grouped-query attention reduce the number of separately stored key/value heads; MLA instead stores a low-dimensional shared latent cache and reconstructs or algebraically absorbs the expanded projections. These methods address related cache costs through different parameterizations.

### Sources

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [Kimi-K3 model card and checkpoint configuration](https://huggingface.co/moonshotai/Kimi-K3/tree/f831ab66814297da540d832a5235f8e904f29d06)
- [Campaign data-flow diagrams and measured stages](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/blob/b786b8cae9992c190c3d3323197c937d8ed8f8b8/docs/kimi/fp8-fmha-dataflow.md)

---

<a id="s-12"></a>
## Slide 12: Residuals, normalization and MLPs complete the block

Attention is one part of a model layer.

- **Normalize:** Control activation magnitude
- **Attention:** Mix information across tokens
- **Residual path:** Preserve and combine earlier features
- **MLP / MoE:** Transform features within each token

RMSNorm(x) = g ⊙ x / √(mean(x²) + ε)

A conventional pre-normalized transformer alternates attention and a feed-forward network, with residual additions around the sublayers. RMSNorm rescales a vector using its root-mean-square magnitude and learned gain g; unlike LayerNorm, it does not subtract the mean. An MLP applies learned projections and a nonlinear activation independently to each token. Residual paths preserve information and support effective training of deep stacks. This slide is a prerequisite template, not the exact Kimi-K3 residual algorithm. Kimi-K3 uses Attention Residuals, which learn how to mix representations across depth; that change is introduced explicitly later.

### Sources

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [Kimi-K3 model card and checkpoint configuration](https://huggingface.co/moonshotai/Kimi-K3/tree/f831ab66814297da540d832a5235f8e904f29d06)
- [vLLM Kimi-K3 AMD implementation](https://github.com/ajith-sirra-amd/vllm/tree/08e5d81a0d4fc76c20a97f9a517195db5a5dc5be/vllm/models/kimi_k3/amd)

---

<a id="s-13"></a>
## Slide 13: Position information: separate the general idea from K3

RoPE is common in attention; this checkpoint's MLA uses NoPE.

- **General RoPE:** Rotate query/key feature pairs according to position.
- **Kimi-K3 MLA:** mla_use_nope = true; inspected wrapper passes rotary_emb=None.
- **Naming trap:** A 64-column “RoPE” field still exists in the MLA layout.



A plain attention operation without position information is equivariant to reordering its inputs, so models need mechanisms that represent sequence order. RoPE is a common solution that rotates paired query/key features by position-dependent angles. However, this Kimi-K3 checkpoint sets mla_use_nope=true, and the inspected AMD MLA implementation passes no rotary embedding object. Its MLA tensor layout retains a 64-column field named qk_rope_head_dim, but that field name does not prove a rotary transform runs. KDA recurrence and short causal convolutions carry order information in the hybrid model. The deck labels those 64 columns as the shared K component or legacy RoPE slot.

### Sources

- [Kimi-K3 model card and checkpoint configuration](https://huggingface.co/moonshotai/Kimi-K3/tree/f831ab66814297da540d832a5235f8e904f29d06)
- [vLLM Kimi-K3 AMD implementation](https://github.com/ajith-sirra-amd/vllm/tree/08e5d81a0d4fc76c20a97f9a517195db5a5dc5be/vllm/models/kimi_k3/amd)

---

<a id="s-14"></a>
## Slide 14: Prefill and decode are different execution regimes

Prefill processes known tokens; decode produces the next token.

- **Prefill:** Many query rows; large matrix products; new-token causal attention.
- **Decode:** Few query rows per request; read cached history; repeat every output step.
- **Mixed serving:** New prompt chunks can share a scheduler step with decoding requests.



During prefill, the server processes prompt tokens whose identities are already known. It can compute many query rows together while enforcing causal visibility. During ordinary decode, it computes the next token for each active request, typically one new query per request per step. Speculative verification increases that query count. Prefill often offers better weight reuse and larger GEMMs; decode often faces smaller operations, cache reads and synchronization overhead. These are tendencies, not universal bottleneck declarations. Long-context decode can have substantial attention traffic, and an agentic workload can alternate cached prefixes, new prompt chunks and output generation. Always profile the actual batch regime.

### Sources

- [FlashAttention: exact attention with IO awareness](https://arxiv.org/abs/2205.14135)
- [vLLM Kimi-K3 AMD implementation](https://github.com/ajith-sirra-amd/vllm/tree/08e5d81a0d4fc76c20a97f9a517195db5a5dc5be/vllm/models/kimi_k3/amd)

---

<a id="s-15"></a>
## Slide 15: The KV cache avoids recomputing the past

Cache persistent context; compute the new query for the current step.

- **Already processed:** Reuse stored K/V or MLA latent rows
- **New tokens:** Append their cache entries
- **Current query:** Read the allowed history

Standard KV bytes ≈ 2 × layers × tokens × KV_heads × head_dim × bytes/value

Without a cache, every generated token would force the model to recompute key/value representations for the entire earlier sequence. A KV cache retains those representations. Queries are generally not cached because a future output step uses a new query. The simple storage formula assumes equal key/value widths and ordinary attention; it does not describe MLA or recurrent KDA exactly. MLA stores a compressed per-token latent representation, while KDA keeps recurrent state plus short-convolution state. Cache allocation uses blocks and metadata rather than a single contiguous buffer per request. Prefix reuse, precision, sharding, alignment, scale storage and speculative state all affect real memory usage.

### Sources

- [Kimi-K3 model card and checkpoint configuration](https://huggingface.co/moonshotai/Kimi-K3/tree/f831ab66814297da540d832a5235f8e904f29d06)
- [vLLM Kimi-K3 AMD implementation](https://github.com/ajith-sirra-amd/vllm/tree/08e5d81a0d4fc76c20a97f9a517195db5a5dc5be/vllm/models/kimi_k3/amd)

---

<a id="s-16"></a>
## Slide 16: Why attention becomes expensive at long context

Every query can compare against many keys.

- **Full prefill:** Attention pair count grows roughly with T²
- **One-token decode:** Attention pair count grows roughly with cached T
- **Naive intermediates:** Scores and probabilities can dominate temporary memory

Attention FLOPs ≈ 2TqTk(d_k + d_v) per head

The approximate multiply-add count includes QK transpose and P times V and counts one multiply-add as two floating-point operations. It ignores masking savings, softmax and other elementwise work. During a full unchunked prefill, both Tq and Tk grow with sequence length, yielding quadratic pair count. With a single decode query, only Tk grows. A naive 32,768 by 32,768 score matrix contains over one billion elements: two GiB per head in a two-byte dtype before also storing probabilities. Practical fused kernels avoid materializing that matrix, and causal kernels skip some masked work. Reducing IO does not eliminate the underlying exact-attention comparisons.

### Sources

- [FlashAttention: exact attention with IO awareness](https://arxiv.org/abs/2205.14135)

---

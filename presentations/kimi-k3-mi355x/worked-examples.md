# Worked examples and a deeper reference

This companion expands the slide explanations. Every main chapter includes the slide's concepts, equations, speaker notes and source links. Read the chapters in numeric order.

## 1. Attention with actual numbers

Let one query see three keys whose already-scaled scores are [2, 1, 0]. Subtract the maximum 2: [0, -1, -2]. Exponentials are approximately [1, 0.367879, 0.135335]. Their sum is 1.503214, giving probabilities [0.665241, 0.244728, 0.090031]. Choose values [10,0], [0,10], [5,5]. The output is approximately [7.10256, 2.89744]. The output is a blend, not a selected token ID.

The natural-log normalization statistic is LSE = 2 + ln(1.503214) = 2.407606. If all scores increase by the same constant 10, probabilities stay the same while LSE increases by 10. This distinction explains why a normalized output check alone cannot establish that an LSE value is correct.

The full attention equation for one head is:

$$S=\alpha QK^T+M,\qquad P_{ij}=\frac{e^{S_{ij}-m_i}}{\sum_k e^{S_{ik}-m_i}},\qquad O=PV.$$

Here m_i is the maximum score in row i. Tq counts query tokens, Tk counts key/value tokens, and d_k and d_v count features. Q is [Tq,d_k], K is [Tk,d_k], V is [Tk,d_v], and O is [Tq,d_v]. Batch and head dimensions are omitted.

A mask is applied before normalization. For a prohibited key, its score is negative infinity and its exponential contribution is zero. Padding tokens, future tokens and request boundaries require correct metadata as well as correct arithmetic. If a row has no legal keys, the backend must implement its defined empty-row behavior; never infer that behavior from the ordinary softmax equation.

## 2. Merging two chunks

Suppose chunk A has LSE ln(3), output [2,0], and chunk B has LSE ln(1), output [0,4]. Their masses are 3 and 1. The merged output is 0.75[2,0] + 0.25[0,4] = [1.5,1]. An equal average would give [1,2], which is wrong.

Use stable logaddexp:

$$m=\max(L_A,L_B),\quad L=m+\ln(e^{L_A-m}+e^{L_B-m}),$$
$$O=e^{L_A-L}O_A+e^{L_B-L}O_B.$$

This is the same normalization principle used by online softmax. It applies per query and per head, not once for an entire batch. The outputs must refer to disjoint allowed key subsets of the same logical attention problem. Overlapping keys would double-count their mass. Consistent softmax scale, position treatment, dtype descales and logarithm base are prerequisites.

## 3. A scalar gated-delta example

A scalar example makes the KDA update tangible, but does not replace its vector/matrix implementation. Start with state S_previous=2. Let decay alpha=0.5, key k=1, current value v=4, update strength beta=0.25 and query q=1.

1. Decay: Sbar=0.5*2=1.
2. Retrieve by the current key: Sbar*k=1.
3. Compute correction: r=4-1=3.
4. Update: S=1+0.25*1*3=1.75.
5. Read with the query: o=1*1.75=1.75.

A full write with beta=1 would move this toy state to 4. A small beta preserves more of the previous memory. The real KDA uses a state matrix, per-channel learned decay, normalized inputs, learned gates, short convolutions and efficient chunked formulations. State orientation differs across libraries, so transpose conventions must be checked when translating equations.

For column-vector keys and values, one explanatory convention is:

$$\bar S_t=\mathrm{diag}(\alpha_t)S_{t-1},\quad r_t=v_t-\bar S_t^T k_t,$$
$$S_t=\bar S_t+\beta_t k_t r_t^T,\qquad o_t=S_t^T q_t.$$

KDA's bounded state gives a different memory/computation tradeoff from token-addressable softmax attention. It is not simply FlashAttention with an implementation trick that removes all key rows.

## 4. Quantization units and an outlier example

For E4M3FN and X=[0,1,2], an amax-based descale is s=2/448=1/224. Stored values are approximately [0,224,448], which are exactly representable in this particular example. Multiplying by s reconstructs [0,1,2]. This convenient example does not mean arbitrary values are exactly representable: FP8 spacing is nonuniform and grows with magnitude.

If a supplied scale was chosen for magnitude 2 but a cached value is 5, the scaled value would be 1120. Clipping to 448 reconstructs only 2. Wider accumulation does not fix that clipping error. If the cached-output policy doubles the scale, the represented range extends to 4; the value 5 can still clip. Thus a 2x headroom factor is a heuristic, not a mathematical guarantee.

For quantized projection:

$$C\approx s_C C_8,\quad W_j\approx s_{W,j}W_{8,j},$$
$$Z_j\approx s_Cs_{W,j}\sum_i C_{8,i}W_{8,j,i}.$$

To store output directly in FP8, divide K and V by their separate output descales before rounding. The final FMHA call interprets those stored operands using the same descales:

$$S\approx \alpha s_Qs_K Q_8K_8^T+M,\quad P=\mathrm{softmax}(S),\quad O\approx s_VPV_8.$$

These equations describe logical units. The inspected FP8 attention implementation also quantizes probability tiles internally for matrix multiplication; it does not necessarily materialize P as the real-valued matrix implied by the mathematical notation. The API returns BF16 O and FP32 natural-log LSE.

A product error can be separated conceptually as (C+delta_C)(W+delta_W)-CW = delta_C W + C delta_W + delta_C delta_W. Output rounding adds another error. Which term dominates depends on distributions, scales, clipping, dimension and the model.

## 5. Shape and storage arithmetic

The local TP8 cached-prefill head count is 96/8=12. Projection output width is 12*(128 key features + 128 value features)=3072. The shared 64-column key component is appended to each projected key head.

Expanded K plus V contain 12*(192+128)=3840 values per cached token per rank. At two bytes each this is 7680 bytes, and at one byte each it is 3840 bytes. At a hypothetical Tk=8192, those logical buffers are 60 MiB and 30 MiB respectively.

The compressed cache has 512+64=576 values per token per MLA layer before scale metadata and alignment. Across 24 MLA layers, this is 13,824 bytes per token if all those values use one-byte storage, before sharding, metadata, padding or backend-specific allocations. This is not a complete model-cache budget: KDA has recurrent and convolution state, and speculative execution can require additional state copies/checkpoints.

A single [128,128] head state contains 16,384 values: 32 KiB in BF16 or 64 KiB in FP32. Real KDA state storage multiplies by relevant heads, layers, request slots and state-management overhead. The actual state dtype comes from the runtime configuration.

## 6. Matrix layout, packing and kernel eligibility

A tensor's shape tells you its logical dimensions. Its strides specify how many storage elements to skip when moving along each dimension. A view can have the same shape as another tensor but a different memory layout. Packed weights may reorder or pad dimensions to match matrix instructions. Transpose in an equation does not automatically mean a separate transpose kernel or allocation.

Kernel eligibility may depend on GPU architecture, dtype, query count, context length, head count, feature width, contiguity, alignment, scale layout and output-buffer availability. A fallback is part of the implementation contract. A profiler or execution observer must establish which branch actually executed. Reusing Q8 across two calls is valid only if both calls refer to the same immutable query data and matching layout within the intended lifetime.

Graph replay retains launch structure and buffer addresses. It does not make input values immutable. Tests must change input contents between replays and verify that outputs change correctly. Allocating new output storage or reading stale scale buffers during capture/replay can create integration bugs absent in an eager kernel microbenchmark.

## 7. Logits, sampling and speculative correctness

The final hidden state is normalized and projected to vocabulary logits. Logits are unnormalized scores over token IDs. Temperature rescales logits before a sampling distribution is constructed. Top-k and top-p restrict candidate sets under configured rules. Greedy decoding selects a maximum-scoring token, subject to the implementation's tie behavior.

Small numerical changes can move a near-tie across an argmax boundary, changing one output token and subsequently the entire generated continuation. Therefore identical aggregate task accuracy does not imply identical answers or bitwise equivalence. Conversely, a difference in one token does not by itself establish a task-quality regression.

Speculative decoding uses a draft proposal and target verification. Exact distribution preservation requires a correct acceptance/rejection and correction algorithm for the selected sampling scheme; the word “speculative” alone does not guarantee that property. The campaign's GSM8K evaluation uses real verification, while performance uses the benchmark's prescribed synthetic acceptance behavior. Keep those claims separate.

The Kimi-K3 target configuration lists zero internal next-token prediction layers. The campaign uses an external DSpark draft checkpoint. The labels “MTP” and “speculation” in the benchmark configuration must be interpreted through the actual resolved serving command.

## 8. Training, inference and architectural compression

Training learns embedding tables, projection weights, gates, expert parameters and routing behavior using an objective and parameter updates. Inference reads those learned parameters to produce predictions; it does not normally update them. Quantization-aware training incorporates representation effects during model training, while online weight conversion in the serving recipe happens after checkpoint loading. One does not validate the other automatically.

Low-rank latent projections and sparse experts are part of the learned architecture. FP8 or MXFP4 are numerical representations. A lower-dimensional latent is not equivalent to storing a full vector with fewer bits. MLA applies both ideas when a compressed latent cache is stored in FP8.

The model card describes a 2.8T model with 104B activated parameters and MXFP4 weights/MXFP8 activations under quantization-aware training. The checkpoint quantization configuration excludes several modules from the packed weight scheme. Actual serving precision is determined by that checkpoint plus online quantization targets and backend kernels. Avoid a single “the model is FP4” label when discussing a particular operation.

## 9. Throughput arithmetic and interpretation

For the recorded complete configurations:

| Stage | Total tokens/s/GPU | Relative to previous | Relative to baseline |
|---|---:|---:|---:|
| Baseline | 15476.46881 | — | — |
| Cached FP8 FMHA | 16915.72154 | +9.2996% | +9.2996% |
| Direct FP8 projection + cached FP8 FMHA | 17406.08912 | +2.8989% | +12.4681% |

These are configuration comparisons, not controlled attribution to a single kernel. Total throughput counts input plus output under the benchmark's semantics, including prefix reuse. Report generated-output throughput and latency separately.

Amdahl example: if 20% of wall time speeds up 2x, new time is 0.8+0.2/2=0.9 of the original; speedup is 1.1111. Add any new quantization and communication costs before claiming a net benefit.

## 10. Accuracy checks and fixed thresholds

Use the same-FP8-operand reference to test implementation arithmetic. Use captured original BF16 inputs to measure the full representation change. Use real-model evaluation to test task quality. Require proof of the optimized path in the relevant phase and on every intended worker.

The controlled new-token diagnostic recorded a maximum BF16-reference RMS of 0.04442 and absolute LSE error of 0.10964. Same-FP8-reference RMS was at most 0.01858 and LSE error 0.00005395. These controlled cases passed the existing gates. They do not override the separate historical captured-BF16 LSE failure of about 0.935 against a 0.20 limit.

Full new-token prototype GSM8K passed with strict 1279/1319 and flexible 1278/1319. The complete sample set, task/scoring and real speculative verification were retained. Exact native-PR GPU, full-model and performance validation remain separate. Do not loosen thresholds merely to convert a failing diagnostic into a pass.

## 11. Common misunderstandings

- **“FP8 KV means FMHA is FP8.”** It can describe only cache storage; inspect attention operands.
- **“BF16 weights require BF16 output storage.”** Accumulation and output storage are separate choices.
- **“Pure FP8 eliminates numerical error.”** FP8 inputs and output rounding remain lossy even with FP32 accumulation.
- **“Fused means one numerical precision.”** A fused kernel can use several dtypes internally.
- **“DCP8 times TP8 means 64 GPUs.”** Here DCP8 operates inside the eight-rank TP group.
- **“KDA has no cache.”** It keeps recurrent and short-convolution state rather than a growing per-token K/V list.
- **“RoPE in a tensor field means rotation runs.”** The inspected K3 MLA uses NoPE.
- **“All experts are absent except the active sixteen.”** The full weight pool still needs residency.
- **“A green CI check proves FP8 execution.”** It proves only the checks actually run.
- **“A newer source inherits an older accuracy result.”** Changes to code, scales, routing or integration require appropriate validation.

# Kimi-K3 DCP Experiments

Newest first. Kimi-K3 (2.8T MoE, 24 of 93 layers are full-attn MLA, 96 heads, 1M ctx,
MXFP4) on 8× MI355X, TP8, agentic replay. **Target: 12,500 tok/s/GPU.**

---

## 0. RETRACTED -- I had never read the reference's actual command

Everything below headed "THE ANSWER" was built on an inference, not a reading. I
took the MI355X reference's configuration from the **B300 script** -- a different
platform, a different file -- and only read the reference's own
`vllm_command.txt` (run 31993981851) late in the session. It says:

```
--tensor-parallel-size 8 --max-num-seqs 40 --gpu-memory-utilization 0.9
--kv-cache-dtype fp8 --no-async-scheduling --enable-prefix-caching
--attention-config {"mla_prefill_backend":"ROCM_AITER_FA"}
--compilation-config {FULL_AND_PIECEWISE, max_capture 120, sizes [1..120]}
--speculative-config {dspark, TRITON_MLA, synthetic, accept 2.51}
NO --decode-context-parallel-size   NO --max-num-batched-tokens   NO --attention-backend
```

**Three of my own errors, all load-bearing:**

| # | I claimed | Actually |
|---|---|---|
| 1 | The reference runs **DCP=8 + MTP** together | It runs **no DCP**. Non-DCP + MTP. T60's "DCP+MTP is refused on ROCm" is still true, but it is **not** what separates us from the reference. |
| 2 | **MTP starves** this workload (T51 cancelled, T54 = 541) | **We starved it.** Our forced `--max-num-batched-tokens 8192` sizes the MLA chunked-prefill workspace. Omitting it ([T61](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/32428889181)) reproduced the reference's KV pool **byte for byte**: 2,646,059 tokens (2.52x) vs our 1,385,293 (1.32x). |
| 3 | The reference **decodes on TRITON_MLA** | It sets **no** target backend. TRITON_MLA appears only inside `--speculative-config`, i.e. for the **draft**. I mistook the draft's backend for the target's and forced the slower one on the main model in every non-DCP trial -- **including [T58](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/32414712217), our best result**. |

### Trials from the correction

| # | Change | Outcome |
|---|---|---|
| [T61](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/32428889181) | omit `--max-num-batched-tokens` | **KV fixed exactly**; then hit the T50 cudagraph assert -- so that assert is *not* version-specific, as I'd assumed |
| [T62](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/32430343818) | + drop patch [2] | **assert cleared**, MTP live, KV 2.52x -- but tput decayed 9,713 -> 4,353/s, 17 queued. KV was not the whole story. |
| [T63](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/32434647582) | + release the forced backend | `ROCM_AITER_MLA` selected (verified), reference backend split reproduced -- then `HSA_STATUS_ERROR_OUT_OF_RESOURCES`. Only delta vs T62 was the backend, so **AITER MLA needs more workspace**. |
| [T64](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/32436403856) | **T58 + AITER MLA only** | running -- clean test of the 1.2-1.6x backend claim against our best result |

**Patch [2] causes the assert.** It raises TritonMLA's `_cudagraph_support` to
`UNIFORM_BATCH`, permitting full capture, which then builds metadata with
`max_query_len = 1 + num_speculative_tokens`. The reference carries **no patch [2]**
and lets vLLM downgrade the drafter to PIECEWISE by itself.

---

## 0b. (Superseded) DCP + MTP is unreachable on ROCm

The reference reaches 5,388 tok/s/GPU by running **DCP=8 and MTP together**. We can
run each alone but never both, and the reason is not tuning:

* **MTP alone starves** -- the drafter cuts the KV pool 4.17x -> 1.31x;
  [T51](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/32392005995) collapsed, [T54](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/32396466979) completed at 541 tok/s/GPU.
* **DCP alone caps** at 2,034 ([T25](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/32143877066)); twelve levers, all negative.
* **DCP + MTP fixes both** -- [T60](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/32425822552) reached **18.43x KV with
  speculation on** (19,326,694 tokens vs 1,385,293 without DCP), confirming DCP pays
  for the drafter's memory -- and then the engine **refused to start**.

```
TritonMLAMetadataBuilder does not support causal multi-token MLA attention for
DSpark with decode context parallelism. Select a backend with explicit DSpark DCP
support or set decode_context_parallel_size=1.
```

`mla_attention.py::_validate_dspark_dcp_support` requires one of two capability
flags. Every declaration of either, across the entire image:

```
supports_non_causal_multi_token_dcp = True
    tokenspeed_mla.py                                  <- the ONLY one

supports_dcp_with_varlen = True
    flashinfer_mla · flashattn_mla
    flashinfer_mla_sparse · flashmla_sparse            <- all NVIDIA
```

**Neither `triton_mla.py` nor `rocm_aiter_mla.py` declares either flag.** And
`TOKENSPEED_MLA` -- the backend the B300 reference uses -- is **not in
`vllm/platforms/rocm.py`'s backend list** and **not installed**
(`import tokenspeed` -> `ModuleNotFoundError`).

**So the reference's configuration is unavailable on this hardware because the only
backends implementing the required capability are NVIDIA-only.** Not a tuning gap,
not a kernel-speed gap, not reachable from configuration.

**Best achievable on this stack: [T58](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/32414712217) -- 2,685.0 tok/s/GPU,
TPOT 0.1161** (non-DCP, spec off, real tokens).

### I reopened this and it holds -- with stronger evidence

A missing *declaration* would be a weak reason to stop, since custom patches are
authorised. And the flag looked patchable: its only effect is one branch
(`if DCP>1 and not supports_dcp_with_varlen: reorder_batch_threshold = 1`), and two
backends declare it **conditionally on KV interleave size == 1** -- a data-layout
precondition our DCP runs already satisfy (T25 uses interleave 1).

But `triton_mla.py` disables its draft-decode path under DCP **in code, with the
reason stated**:

```python
# DCP local sequence lengths are not advanced between draft steps.
self.supports_draft_decode_metadata_update = self.dcp_world_size == 1

def update_draft_decode_metadata(self, _metadata) -> None:
    pass                                     # no-op
```

`speculator.py:112` consumes that flag to decide whether multi-step drafting is
legal. Flipping `supports_dcp_with_varlen` would therefore not enable anything --
it would let drafting run while each draft step reads **stale per-rank sequence
lengths**, i.e. wrong KV ranges: **wrong output, no crash**. The gate that would
catch it (GSM8K) needs ~15 h on this stack, so an un-gated run would produce a
throughput number that looks like progress and isn't.

**The upstream ask, precisely:** implement `update_draft_decode_metadata` for
TritonMLA (or an AITER MLA backend) so DCP local sequence lengths advance between
draft steps, then set `supports_draft_decode_metadata_update` and
`supports_dcp_with_varlen` / `supports_non_causal_multi_token_dcp` accordingly.
Exact file, function, and reason -- a filable vLLM/AITER issue.

### Two of my own conclusions this overturned
1. **"MTP under DCP is blocked by `mla_gluon[bh16bn128] requires batch_size=1`"**
   (T14/T15). I repeated this for many trials as the structural reason. It was
   carried across image changes without rechecking; the real blocker is the backend
   capability flag, and the mla_gluon path was never the binding one.
2. **The config-level ban is version-dependent.** `config/speculative.py`'s
   "MLA DSpark does not currently support decode context parallelism" is **present**
   on ac7509e2b ([T59](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/32424123796) died on it) and **removed** on 5a4c8d99.
   Upstream lifted the config ban; the backend requirement is what actually binds.

---

## 0. WARNING -- Correction: the "2.7x decode regression" does not exist

I reported a 2.7x decode regression (T22 TPOT 0.043 -> T47 0.118, "the largest
unexplained result in this investigation") and queued a GPU bisect against it.
**It was a measurement artefact.**

[T22's server log](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/32123047671) shows `speculative_config=SpeculativeConfig`,
364 SpecDecoding samples, acceptance mean **2.509** -- T22 ran **with speculation**.
Its 0.0429 is per *accepted* token. T47/T57/T58 all ran spec **off**. Normalising:

```
0.1161 / 2.509 = 0.0463   vs   T22's 0.0429
```

**How it happened:** `agg_bmk.json`'s `spec_decoding` field read `'none'` for T22.
That is the same field I flagged as unreliable after T54 -- and I never went back
to re-audit the conclusions already built on it. Discovering a measurement is
untrustworthy obliges re-checking past conclusions, not just future ones.

The bisect ([T58](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/32414712217)) was still worth running: it cleared the image
conclusively (1.3 percent TPOT difference), and that negative result is what
forced me to find the real explanation.

**Consequence:** the reference's 5,388 / 0.038 is spec-ON with *synthetic*
acceptance; our 2,685 / 0.1161 is spec-OFF with real tokens. **These were never
comparable.** Normalised TPOT is 0.046 vs 0.038. The naive throughput scaling
(2,685 x 2.51) is *not* valid either -- [T54](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/32396466979) measured what
actually happens when speculation is enabled here: 541 tok/s/GPU, because the
drafter cuts the KV pool 4.17x -> 1.31x.

---

## 1. Summary -- what was tried, and what it showed

| # | Tried | Trial | Result | Finding |
|---|---|---|---|---|
| 1 | DCP at all (block-table fix) | [T4](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/32043813560) | ✅ **0x1016 fixed** | Tables were sized `max_model_len/dcp` but indexed with the undivided length. Boundary tracked the ratio exactly. Made DCP usable. |
| 2 | DRAM KV offload | [T18](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/32096487055) | ✅ **781 → 1,991** (2.55×) | Largest single DCP win. But no effect on the decode straggler. |
| 3 | GSM8K correctness gate | [T23](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/32138970163) | ✅ **0.9659 / 0.9644** vs 0.9651 | DCP is numerically correct. The case against it is performance only. |
| 4 | World size 8 → 4 | [T24](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/32143146154) / [T25](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/32143877066) | ❌ +2.2% | **Decisive.** Halving collective traffic moved TPOT 4% → cost is *not* world-size-scaled. Rules out every scaling fix. DCP=2 is illegal (24 heads). |
| 5 | Concurrency sweep | [T19](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/32101946357) / [T26](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/32154159649) | ❌ optimum c20 | c8 969 · c20 2,034 · c64 1,041. Axis closed. |
| 6 | Combine algo a2a vs ag_rs | [T28](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/32163743641) | ❌ 2,034 vs 1,978 | Both ROCm options within 3%. |
| 7 | Shard granularity 1 → 16 | [T29](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/32172937044) | ❌ 1,977 | Locality before the combine wasn't the cost. |
| 8 | Attention backend | [T21](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/32114847961) / T45 | ❌ within noise | Not backend-bound. |
| 9 | CUDA graphs | [T7](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/32060082326) / T45 | ❌ no TPOT effect | Not launch-count-bound at the graph level. |
| 10 | Async scheduling on/off | T41 / T42 | ❌ no effect | 0 `ref_cnt` asserts at c20; the workaround was unnecessary. |
| 11 | NUMA pinning, node-level | T45 | ❌ no effect | — |
| 12 | NUMA pinning, per-rank slices | [T46](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/32368684074) | ❌ **worse** | 188→218 ms rising vs unpinned converging to 208. Not an OS-scheduling problem. |
| 13 | Ported direct P2P collective to ROCm | [T31b](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/32270805303) | ⚠️ works, **−0.9%** | Hand-ported the a2a combine HIP kernel. Functional, not faster. |
| 14 | MTP under DCP | [T20](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/32110276088) | ❌ killed | Draft KV is replicated → costs W× per rank. |
| 15 | Profiling DCP vs non-DCP | [T35e](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/32333672290) / [T37](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/32340149740) | 🔍 **bottleneck located** | 92.65% collectives vs 56%. DCP shards no work; the TP all-reduce inflates 8.8×. |
| 16 | Non-DCP best config to completion | [T47](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/32372517517) | 📊 **2,656 / TPOT 0.118** | Non-DCP beats DCP by 30.6% tput and 30% TPOT on an identical stack — completed-run comparison. **Ran with MTP OFF** (label said on). Also: 96.7% theoretical cache hit vs ~30% achieved. |
| 21 | **DP attention** | — | ❌ **infeasible on 8 GPUs** | Non-expert weights are 114.4 GB and DP replicates them per rank → 295.2 GB/GPU vs a 288 GB card. Needs 16 GPUs (204.8 GB). Verified from checkpoint headers; not dispatched. |
| 20 | **Complete patch [2]** (`query_len_support = UNIFORM`) | **T52** | **queued** | Self-consistent completion; recovers FULL cudagraphs under MTP. **Gated on GSM8K first** — if the Triton decode kernel can't take `query_len > 1` this fails *silently wrong*, not loudly. |
| 19 | **Bisect the 2.7× decode regression** | **T49** | **queued** | T22 0.043 → T47 0.1176 TPOT with speculation off on BOTH sides. Re-run T47's config on `ac7509e2b`. Largest unexplained result in the investigation, and not DCP's. |
| 18 | **MTP actually on, non-DCP** | [T50](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/32390477829) → [T51](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/32392005995) | ❌ **MTP closed: 541 tok/s/GPU, 4.9× worse than no-MTP** | MTP genuinely active (`method='dspark'`), but full cudagraph capture asserted. **Exposed that patch [2] is incomplete**: it raises `_cudagraph_support` to `UNIFORM_BATCH` without raising `query_len_support` to match, so the reorder threshold stays 1 and `max_query_len = 1+num_spec_tokens` trips `mla_attention.py:2288`. Hidden until now because every patch-[2] trial ran spec **off** — one bug concealed the other. T51 uses PIECEWISE to sidestep it. |
| 17 | **All-reduce implementation** (PYNCCL vs AITER_CUSTOM) | [T48](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/32381243949) | ❌ **−23% tput, +31% TPOT** | Hypothesis was well-founded — 78.37% of kernel time is in `cross_device_reduce_2stage`, and NVIDIA also swaps their all-reduce. Answer: **AITER's implementation was already the better one.** The 78% attribution stands; it just isn't addressable by substitution. Also measured DCP's capacity win: prefix hit **70.6%** vs T47's ~30%, `ext_cache_hit` 0%. | The one lever identified and never dispatched. |

*T41, T42 and T45 predate run-URL capture in the ledger; all others link to their GH run.*

### Performance, all measured runs

| Config | tok/s/GPU | TPOT | TTFT | Trial |
|---|---:|---:|---:|---|
| SA reference | **5,388** | 0.038 | — | — |
| non-DCP + DRAM offload (older image) | **3,341** | 0.043 | — | [T22](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/32123047671) |
| **non-DCP, best completed** | **2,685** | **0.116** | 3.45 s | **[T58](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/32414712217)** |
| non-DCP, 5a4c8d99 | 2,656 | 0.118 | 4.43 s | [T47](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/32372517517) |
| non-DCP + EP=8 | 2,619 | 0.118 | 5.09 s | [T57](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/32406159156) |
| non-DCP + **MTP**, c8 | 541 | 0.292 | 77.8 s | [T54](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/32396466979) |
| **best DCP** — DCP=4, c20, DRAM | **2,034** | **0.167** | 3.80 s | **[T25](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/32143877066)** |
| DCP=8, PYNCCL all-reduce | 1,532 | 0.227 | 11.31 s | [T48](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/32381243949) |
| DCP=8 + ported direct a2a | 2,015 | 0.172 | 4.69 s | [T31b](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/32270805303) |
| DCP=8, c20, DRAM | 1,991 | 0.174 | 4.64 s | [T18](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/32096487055) |
| DCP=4, ag_rs combine | 1,978 | 0.184 | — | [T28](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/32163743641) |
| DCP=4, interleave 16 | 1,977 | 0.177 | 4.81 s | [T29](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/32172937044) |
| DCP=8 + TRITON_MLA, spec off | 1,948 | 0.186 | 5.4 s | [T21](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/32114847961) |
| non-DCP, c1 (latency point) | 1,225 | **0.0042** | — | [T22](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/32123047671) |
| DCP=8, c64 | 1,041 | 0.683 | 501 s | [T19](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/32101946357) |
| non-DCP c32, no cudagraph patch | 1,015 | 0.170 | — | [T11](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/32073039787) |
| DCP=4, c8 | 969 | 0.164 | — | [T26](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/32154159649) |
| DCP=8 + PIECEWISE cudagraph | 781 | 0.687 | — | [T7](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/32060082326) |
| DCP=8, no offload | 742 | 0.663 | — | [T5](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/32049134216) |
| DCP=8 + pinning | — | 0.218 | — | [T46](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/32368684074) |
| DCP + everything | — | 0.242 | — | T45 |

**DCP's ceiling is ~2,000 tok/s/GPU — 39% below best non-DCP.**

---

## 2. Reference-design findings (B300 script, read-only)

Read `SemiAnalysisAI/InferenceX` `kimik3_fp4_b300_vllm_mtp.sh`. Three results.

**a) The reference uses NO DP attention and NO expert parallelism.** Pure
TP8 + DCP=8 + MTP (`grep -c "enable-expert-parallel|data-parallel"` = 0). On
NVIDIA, **DCP is the intended path and it performs.** Our DCP decode penalty is a
ROCm collective problem, not a flaw in the DCP approach.

**b) NVIDIA also refuses the default all-reduce** — `VLLM_ALLREDUCE_USE_FLASHINFER=1`
substitutes a different implementation. FlashInfer is NV-only, but T48's
`--disable-custom-all-reduce` is the same move. Independent corroboration that the
all-reduce implementation is a real lever: reached from our profile, confirmed by
the reference.

**c) The 5,388 reference number is measured with SYNTHETIC acceptance.** B300 uses
`rejection_sample_method: "synthetic"` with `synthetic_acceptance_length` on the
*benchmark* path, and `"block"` only under `EVAL_ONLY`. Synthetic acceptance commits
drafts regardless of target logits, inflating the token count. Our completed runs
(T22 3,341 · T47 2,656 · T25 2,034) ran spec fully **off** — 100% real tokens.
**The comparison has never been like-for-like, and the asymmetry favours the
reference.** To be quantified by T50, not asserted.

### DP attention: NOT RUNNABLE on 8 GPUs (measured from the checkpoint)

Read the actual safetensors headers of the cached checkpoint and split every
tensor by expert vs non-expert:

```
expert       1,446.46 GB   (92.67%)
non-expert     114.40 GB   ( 7.33%)
total        1,560.86 GB
```

Per-GPU weight footprint on 8 × 288 GB (budget 259 GB at util 0.9):

| Layout | per-GPU | fits |
|---|---:|---|
| **TP8 pure** (all sharded 8×) | **195.1 GB** | ✅ |
| DP8 / TP1 / EP8 (experts/8, **attention replicated**) | **295.2 GB** | ❌ over by 36 GB |
| TP4 | 390.2 GB | ❌ |

DP attention replicates the **114.4 GB** of non-expert weights on every rank:
114.40 + 1446.46/8 = 295.2 GB, which exceeds the 288 GB card **even at
utilisation 1.0** — no tuning rescues it, and headroom for KV is −36 GB. It
would OOM during weight load.

This reproduces upstream's `strategy_min_gpus` threshold exactly: at DP16,
114.40 + 1446.46/16 = **204.8 GB**, which fits. Hence "DEP 16+".

**Not dispatched** — it would have burned 8-GPU time to produce an OOM.

**EP=8 survives and is still untested**: `--enable-expert-parallel` shards experts
without replicating attention, staying at the 195 GB TP8 footprint. Every kimi
matrix row sets `ep: 1`, so the flag has never been passed in 54 trials.

### Upstream levers checked and rejected (no GPU time spent)

| Lever | Verdict | Why |
|---|---|---|
| `VLLM_USE_V2_MODEL_RUNNER=1` | **no-op** | Auto-enables for default-V2 architectures; T47 and T48 both log `Using V2 Model Runner`. |
| `VLLM_DCP_Q_REPLICATE=1` (vLLM #45964) | **no-op for Kimi-K3** | Needs `DCPGroupColumnParallelLinear`; Kimi-K3 builds plain `ColumnParallelLinear` (`models/kimi_k3/amd/linear.py:376,384`). Env is read only by `deepseek_v2.py` + an NV-only model. |

Q-replication would also be **low value even if patched in**: it removes the query
all-gather, but the DCP-specific collectives are only ~14% of kernel time
(`ncclDevKernel` 8.22% + `msccl` 6.06%). `cross_device_reduce_2stage` — the **TP**
all-reduce, untouched by Q-replication — is **78.37%**. T48 attacks that one.

### Knob transfer

| B300 knob | our stack | usable |
|---|---|---|
| `VLLM_USE_V2_MODEL_RUNNER=1` | **already active** (auto-on for Kimi-K3) | no-op |
| `VLLM_ENABLE_K3_LATENT_MOE_TAIL_FUSION=1` | absent from ROCm build | NV-only |
| `VLLM_ALLREDUCE_USE_FLASHINFER=1` | FlashInfer is NV-only | not usable |
| `VLLM_USE_DIRECT_DCP_A2A=1` | ported (T31b) | −0.9% |
| `VLLM_USE_DIRECT_DCP_{Q,KV}_GATHER=1` | `multimem.st.*` PTX | no AMD equivalent |
| `--max-num-batched-tokens 16384` | we run 8192 | ✅ **untested** |
| `VLLM_PREFIX_CACHE_RETENTION_INTERVAL=0` | present, unset | ✅ **untested** |
| `PYTHONHASHSEED=42` | unset | ✅ **untested** |

> **Wrong turn caught before spending GPU time.** `VLLM_USE_V2_MODEL_RUNNER` reads as
> unset and `cp_utils.py:50` gates a DCP path on it. I called it the most significant
> find of the session. It is a **no-op** — `config/vllm.py::use_v2_model_runner`
> auto-enables for default-V2 architectures, and both T47 and T48 log
> `gpu_worker.py:396 Using V2 Model Runner`.

---

## 3. The bottleneck

Trace, rank 0, 15 s decode window ([T35e DCP=8](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/32333672290) vs [T37 non-DCP](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/32340149740)):

| | DCP=8 | non-DCP |
|---|---:|---:|
| GPU kernel time | 14.801 s | 5.033 s |
| `cross_device_reduce_2stage` (TP all-reduce) | **78.37%** | 55.48% |
| MoE GEMM | 1.91% | 13.33% |
| **collectives / compute** | **92.65% / 7%** | 56% / 44% |

Per decode step (~118 all-reduces/step in **both**):

| | all-reduce | compute | TPOT |
|---|---:|---:|---:|
| non-DCP | 14.6 ms | 11.5 ms | 43 ms |
| DCP=8 | **129 ms** | 11.6 ms | 167 ms |

**Findings:**
1. **DCP shards no decode work** — compute/step identical (11.5 vs 11.6 ms).
2. **The penalty isn't DCP's own collective** — the a2a combine is ~8 µs.
3. **It's the TP all-reduce inflating 8.8×**, at the same call count. A collective
   **DCP does not own**.
4. **Mechanism is host-side** — one rank starved by ~124k launch gaps of ~62 µs;
   seven others block inside the all-reduce.
5. **The straggler moves** rank 7 → 0 → 1 → 5 across runs — not a bad GPU or link.
6. **GPU load is balanced** to within 5%.
7. **Cost is not world-size-scaled** (T25) — which rules out every scaling fix.

*Caveat: kernel duration includes time blocked on peers, so this localises where time
is attributed, not by itself the root cause.*

---

## 4. Patches

| # | Name | What it fixes |
|---|---|---|
| [1] | aiter pybind11 | Standalone pybind11 (internals v12) outranks torch's bundled (v11) via `-I` vs `-isystem`. Separate type registry per internals id → JIT module can't see `aiter_tensor_t` → `TypeError` at warmup. Unblocks `ROCM_AITER_FA` prefill. |
| [2] | TritonMLA cudagraph | `_cudagraph_support = UNIFORM_SINGLE_TOKEN_DECODE` caps `min_cg_support`, downgrading FULL→PIECEWISE under spec-decode, leaving the DSpark drafter fully eager. TRITON_MLA can't be swapped — it's the only ROCm MLA backend with `supports_non_causal_multi_token_decode=True`. **Measured 14.05 → 77.65 tok/s, ITL 71.16 → 12.88 ms (5.52×).** |
| [3] | KV block-pool clamp | `allocate_external_computed_blocks()` passes a **negative** count to `get_new_blocks`, which is silently destructive: `num_free_blocks -= n` *increases* it, `range(n)` iterates zero times, free list untouched. Later pop walks past the tail → `assert block.ref_cnt == 0` mid-run. Load-dependent: c10 died 3612 s, c12 487 s, c16 354 s. |
| [4] | our DCP-LSE | Plumbs LSE + round-robin CP through AITER MLA decode. aiter already had it (`return_lse`/`cp_world_size`/`cp_rank`); vLLM never wired it. aiter's LSE is natural-log, sm_scale folded, `[B,H]` fp32 — already DCP's layout. **Superseded by [5], now OFF.** |
| [5] | vLLM PR #51705 | Upstream DCP for Kimi-K3 DSpark. Open, in no nightly; fetched at runtime, **pinned by sha256**; diff filtered to `vllm/` only. **Predicted before running, and confirmed: does NOT fix 0x1016** — it exempts groups only when `spec.non_causal_multi_token_decode`, and with `speculative_config=None` that set is empty. |
| [6] | DCP block-table sizing | **THE 0x1016 FIX.** Tables sized `cdiv(max_model_len, block_size × dcp_size)` but indexed with the *undivided* `max_model_len` → at 1M/DCP8 only 131,072 tokens addressable → OOB in chunked prefill. |
| [7] | direct DCP a2a (ROCm port) | vLLM compiles the kernel only under `VLLM_GPU_LANG == CUDA`. Ported the a2a combine to HIP (`st/ld.global.{release,acquire}.sys.u32` → `__hip_atomic_*` at `SYSTEM` scope). `q_gather`/`kv_gather` use `multimem.st.*` PTX (NVLink multicast) — no AMD equivalent, not ported. **Works, −0.9%.** Env-gated off. |
| [8] | DCP gathered-head sizing | Supplies PR #51705's failing hunk 7 (`self._decode_num_heads = num_heads × dcp_world_size`). Guard must match the **assignment** — the PR adds 14 *uses* of the name while the defining hunk fails. |

Patch [6] boundary, measured — tracks `max_model_len/dcp` exactly:

| | budget | last chunk fit | faulted at |
|---|---:|---:|---:|
| DCP8 bf16 | 131,072 | 119,040 | 134,400 |
| DCP4 bf16 | 262,144 | 254,976 | 262,656 |
| DCP8 fp8 | 131,072 | 119,808 | 135,168 |

Halving DCP doubled both → block **count**, not bytes. All pure prefill, which is why
swapping decode backends never helped.

### Patches per trial

| Trial | [1] | [2] | [3] | [4] | [5] | [6] | [7] | [8] |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| T1–T2 | – | – | – | ✅ | – | – | – | – |
| T3 | – | – | – | – | ✅ | – | – | – |
| T4–T7 | – | – | – | – | ✅ | ✅ | – | – |
| T8 | ❌ | ❌ | ❌ | – | ✅ | ✅ | – | – |
| T9 | ✅ | ✅ | ✅ | – | ✅ | ✅ | – | – |
| T10 | ✅ | ✅ | ✅ | – | – | – | – | – |
| T11–T13 | ✅ | ❌ | ✅ | – | – | – | – | – |
| T17–T29 | ✅ | – | ✅ | – | – | ✅ | – | – |
| T31b | ✅ | – | ✅ | – | – | ✅ | ✅ | – |
| T35e–T48 | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | off | ✅ |

Images: T1 `ac7509e2b` · T2–T13,T16 `311b3513` · T14–T15 `yukiozzz/kimi-k3-pr51705` ·
T17–T31b `ac7509e2b` · T22 unified `3fa1b88a` · T35e→ `nightly-5a4c8d99`.

---

## 5. Key trials

### T48 — `--disable-custom-all-reduce` (RUNNING)
[Run 32381243949](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/32381243949) · dispatched 2026-08-20 14:45 UTC
DCP=8 · conc 20 · DRAM offload · fp8 KV · spec MTP · pinning.

`cross_device_reduce_2stage` **is** the AITER custom all-reduce — 78% of DCP kernel time.
Engine runs `disable_custom_all_reduce=False`, dispatch order
`['QUICK_REDUCE','AITER_CUSTOM','PYNCCL']`, so AITER_CUSTOM is what every trial has
measured. This flag forces PyNCCL/RCCL — different host-side launch and sync behaviour,
which is what the identified mechanism is made of.

*Flagged in the log as "worth one run" three times and never dispatched. A real gap,
caught by review rather than by me.*

### T47 — best config, run to completion ✅ COMPLETED
[Run 32372517517](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/32372517517) ·
DCP **off** · TRITON_MLA · pinning · DRAM offload · fp8 KV · conc 20 · **spec OFF** (see correction)

The investigation had no completed aggregate for its own best config — T44 showed the
lowest ITL ever measured (32 ms cold / 130 ms steady) but was cancelled, so everything
downstream was extrapolation from a cancelled run.

```
2,655.7 tok/s/GPU   (input 2,640.9 · output 14.8)
TOTAL 21,245.9 tok/s · TPOT/ITL mean 0.1176 p50 0.11835 p90 0.14884
TTFT mean 4.427 p50 1.735 · e2e mean 78.05 p50 33.91
558/600 successful · 38 InvalidInferenceResultError · 3,629.6 s window
```

**Best steady-state TPOT measured at c20 on any config here** — 0.118 vs T25's
0.167 and T45/T46's 0.242/0.218. **Non-DCP beats DCP by 30.6% throughput and 30%
TPOT on an identical stack.**

*Correction: while it ran I quoted "~2,800 sustained" from `tput_in_srv`. That was
the **input** rate / 8. The official total per-GPU aggregate is 2,655.7.*

**Still 20.5% below T22** (2,656 vs 3,341) — the newer nightly plus full patch
stack has not recovered the older unified image's number. That gap is its own open
question, and it is not DCP's fault.

> **⚠️ CORRECTION — T47 ran with MTP OFF.** `speculative_config=None`. I reported it
> as "spec MTP" from the matrix label instead of the engine's own output. Cause:
> `DISABLE_SPEC="${DISABLE_SPEC:-1}"` lives inside a block gated on **concurrency**
> (`CONC >= DCP_AUTO_CONC_THRESHOLD`) but commented and echoed as the *DCP* config
> block. At c20 it fires whether DCP is on or off, so setting `DCP_SIZE=1` left
> speculation silently disabled. **T44 is affected the same way.**
>
> I then briefly concluded the T22→T47 regression was "largely MTP" and withdrew the
> bisect. Checking T22's aggregate showed `spec: 'none'` there too — **both arms are
> non-speculative, so MTP explains none of that gap.** The regression stands and the
> bisect is back on. All "DCP ceiling" numbers in this document should be read as
> **non-speculative** throughout.

**Finding — what the KV capacity is worth:**
```
theoretical_cache_hit_rate    96.7%   <- unbounded cache (final aggregate)
prefix_cache_hit              30.3%   <- actual, 4.4M-token pool
ext_cache_hit                 90.8%   <- DRAM catching the misses
unique_in_srv                 36.2M tokens
```
The workload has 94.1% reusable prefix. Non-DCP holds 30% in HBM; DRAM catches the rest,
so little is recomputed — but at PCIe latency. **DCP=8's 32.7M pool is the only measured
config that fits this working set in HBM.** That benefit is real and **prefill-side**
(T25 TTFT 3.80 s vs T18 4.64 s).

**Correction:** the ~2,000 tok/s/GPU DCP ceiling is *not* evidence the KV capacity is
useless — it's evidence the decode collective eats a benefit that genuinely exists.

*Prediction stated before the result: prefix hit would decay ~91% → ~30% with DRAM
absorbing overflow. Observed 48.0% → 30.3%, ext_cache 21% → 90.8%.*

### T46 — NUMA pinning isolated: no effect
[Run 32368684074](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/32368684074) ·
DCP=8 · per-rank dedicated CPU slices, the only delta vs T36.
```
T46 pinned:    188 -> 175 -> 200 -> 216 -> 218 ms  (RISING)
T36 unpinned:  221 -> 219 -> 217 -> 213 -> 208 ms  (converging DOWN)
```
Prefix cache was back to 52.3% at 218 ms, so not a cold-cache artifact. Cancelled.
*I earlier called an early 188 ms sample "worth ~15%" — wrong, light-load artifact.*

### T25 — best DCP
[Run 32143877066](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/32143877066) ·
DCP=4 · conc 20 · a2a · interleave 1 · DRAM offload
```
2,033.7 tok/s/GPU · TPOT 0.1674 · TTFT 3.80 s · 449/490 · KV 17.0M (16.25x)
```
| | DCP=8 (T18) | DCP=4 (T25) |
|---|---:|---:|
| tok/s/GPU | 1,990.8 | **2,033.7** (+2.2%) |
| TPOT | 0.1742 | 0.1674 (−4%) |
| TTFT | 4.64 s | 3.80 s (−18%) |

**Decisive:** halving world size halves collective traffic and moves TPOT 4% → the cost
is **not world-size-scaled**, it's a fixed per-step overhead of doing the merge at all.

### T22 — best overall
[Run 32123047671](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/32123047671) ·
DCP off · DRAM offload → **c20: 3,340.5 tok/s/GPU, TPOT 0.0429** (c1: 1,225.1 / 0.0042)

### T18 — DCP breakthrough
[Run 32096487055](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/32096487055) ·
DCP=8 + DRAM offload → **1,990.8 tok/s/GPU. 781 → 1,991 = 2.55×.** Largest single DCP win.

### T23 — correctness gate
[Run 32138970163](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/32138970163) ·
GSM8K **0.9659 strict / 0.9644 flexible** vs 0.9651 baseline.

### T4 — 0x1016 fixed
[Run 32043813560](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/32043813560) ·
PR #51705 + patch [6] → errors=0, no fault. The run that made DCP usable at all.

---

## 6. Full ledger

| # | run | config | result |
|---|---|---|---|
| 1 | [32025696861](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/32025696861) | [4], DCP8, bf16 | FAIL 0x1016 @134,400 |
| 2 | [32039650984](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/32039650984) | [4], DCP4, bf16 | FAIL 0x1016 @262,656 |
| 3 | [32042030173](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/32042030173) | #51705 only, DCP8 | FAIL 0x1016 @135,168 |
| 4 | [32043813560](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/32043813560) | #51705 + **[6]** | **0x1016 FIXED**, errors=0 |
| 5 | [32049134216](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/32049134216) | + `AIPERF_FAST=1` | 742.4, TPOT 663 ms; 113/183 dropped, validity suspect |
| 6 | [32055440757](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/32055440757) | GSM8K gate | timeout not fault — 3.5–6.8 tok/s, ≈15 h needed |
| 7 | [32060082326](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/32060082326) | + PIECEWISE cudagraph | 780.6, TPOT 0.687 |
| 8 | [32066978737](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/32066978737) | DCP off, [1]–[3] off | FAIL @15 min — **my error**, `AITER_DISABLE_FMHA_OPUS` only set in the DCP branch |
| 9 | [32068474469](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/32068474469) | DCP off, [1]–[3] on | FAIL @25 min — RCCL watchdog |
| 10 | [32070778181](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/32070778181) | DCP off, [1][2][3] | FAIL @21 min — same signature, so [5] wasn't the cause |
| 11 | [32073039787](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/32073039787) | DCP off, [2] off | 1,014.9, TPOT 0.170 |
| 12 | [32077536567](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/32077536567) | + full profile | 999.4, TPOT 0.1698 |
| 13 | [32084553677](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/32084553677) | chunk 8192→32768 | LOST — cancelled by workflow concurrency rule |
| 14 | [32089974051](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/32089974051) | yukiozzz, c64, MTP | FAIL — `mla_gluon requires batch_size=1, got 64` |
| 15 | [32090860356](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/32090860356) | yukiozzz, c64 | FAIL — same, got 3 |
| 16 | [32093774227](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/32093774227) | c64, chunk 32768 | cancelled @20 min |
| 17 | [32094907936](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/32094907936) | DCP8 c20 DRAM | FAIL — **my error**, leftover `MAX_NUM_BATCHED_TOKENS=32768` |
| 18 | [32096487055](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/32096487055) | DCP8 c20 **DRAM** | **1,990.8**, TPOT 0.1742 |
| 19 | [32101946357](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/32101946357) | conc 64 | 1,041.1, TTFT 501 s — worse |
| 20 | [32110276088](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/32110276088) | TRITON_MLA + MTP | cancelled @55 min, degrading |
| 21 | [32114847961](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/32114847961) | TRITON_MLA, spec off | 1,948.4 — noise vs T18 |
| 22 | [32123047671](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/32123047671) | DCP off, c1+c20 | **3,340.5 — BEST OVERALL** |
| 23 | [32138970163](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/32138970163) | GSM8K gate | **PASSED 0.9659/0.9644** |
| 24 | [32143146154](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/32143146154) | DCP=2 | FAIL @init — `is_valid_num_heads(24)` illegal at TP8 |
| 25 | [32143877066](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/32143877066) | DCP=4 c20 | **2,033.7 — BEST DCP** |
| 26 | [32154159649](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/32154159649) | DCP=4 **c8** | 969.4 — concurrency axis closed |
| 27 | [32162233477](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/32162233477) | direct symm-mem | FAIL @9 min — op absent, CUDA-only |
| 28 | [32163743641](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/32163743641) | **ag_rs** | 1,978.4 — a2a wins |
| 29 | [32172937044](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/32172937044) | interleave 1→16 | 1,977.2 — worse |
| 31a | [32269094879](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/32269094879) | local image tag | FAIL @6 min, no GPU — harness `--pull always` can't resolve a local tag |
| 31b | [32270805303](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/32270805303) | **ported direct a2a** | 2,014.7 — works, **−0.9%** |
| 35e | 32333672290 | DCP=8 profiling | 92.65% collectives |
| 37 | [32340149740](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/32340149740) | non-DCP profiling | 56.05% collectives |
| 45 | — | DCP + everything | TPOT 0.242 |
| 46 | [32368684074](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/32368684074) | DCP=8 + pinning | 0.218 rising — no help |
| 48 | [32381243949](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/32381243949) | DCP=8 + **PYNCCL all-reduce** | **1,531.6**, TPOT 0.2271 — worse than AITER_CUSTOM |
| 50 | [32390477829](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/32390477829) | non-DCP + MTP, FULL cudagraphs | **FAIL @init** — MLA full-capture assert; exposed patch [2] incomplete |
| 51 | [32392005995](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/32392005995) | non-DCP + MTP, c20 | **cancelled** — KV-starved, 9,538→3,833/s, 16 queued |
| 54 | [32396466979](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/32396466979) | non-DCP + MTP, **c8** | **541.0**, TPOT 0.2921, TTFT 77.8 s — MTP closed |
| 55 | [32404418920](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/32404418920) | EP=8 attempt | **FAIL** -- spec-decoding also selects the script filename |
| 57 | [32406159156](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/32406159156) | **EP=8** (flag verified) | **2,619.2**, TPOT 0.1177 -- neutral |
| 58 | [32414712217](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/32414712217) | **image bisect** on ac7509e2b | **2,685.0**, TPOT 0.1161 -- image is NOT the cause |
| 47 | [32372517517](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/32372517517) | non-DCP, full 3600 s | **2,655.7**, TPOT 0.1176 — completed |

---

## 7. Levers vs the DCP decode penalty

| Lever | Trial | Result |
|---|---|---|
| DRAM KV offload | T18/T39 | ✅ 2.55× as a feature · ❌ no effect on straggler |
| Concurrency | T19/T26 | ❌ optimum at c20 (c8 969 · c20 2,034 · c64 1,041) |
| World size 8→4 | T24/T25 | ❌ +2%; DCP=2 illegal |
| a2a vs ag_rs | T28 | ❌ 2,034 vs 1,978 |
| Interleave 1→16 | T29 | ❌ 1,977 |
| Attention backend | T21/T45 | ❌ noise |
| CUDA graphs | T7/T45 | ❌ no TPOT effect |
| Async scheduling | T41/T42 | ❌ no effect |
| NUMA pinning (node) | T45 | ❌ no effect |
| NUMA pinning (per-rank) | T46 | ❌ **worse** |
| Ported direct P2P collective | T31b | ⚠️ works, −0.9% |
| MTP under DCP | T20 | ❌ draft KV replicated, W× per rank |
| **All-reduce implementation** | [T48](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/32381243949) | ❌ **−23% / +31% TPOT** — AITER_CUSTOM already better |

---

## 8. Wrong turns (kept)

- **"The DCP decode cost is irreducible"** — wrong in an interesting way. The cost is real
  but mostly **not DCP's**; it's the TP all-reduce any TP=8 decode pays, which DCP worsens
  by adding sync points.
- **T31a** — dispatched a local-only image tag; the harness does `--pull always`.
- **`-p2` dry-run false positive** — claimed a patch applied to both nightlies; correct
  `-p1` showed only `5a4c8d99` works (23/24 hunks).
- **Patch [8] guard matched uses, not the assignment** — would have thrown `AttributeError`
  at first decode. Caught in local validation.
- **`VLLM_TORCH_PROFILER_DIR` obsolete** — removed from `envs.py`; use `--profiler-config`.
- **`PROFILER_ARGS` used before defined** — line 786 vs 792, so it never reached the server.
- **Shadowed `DCP_SIZE`** — an assignment in the auto-concurrency block ran first, so runs
  showed DCP=4 / 17× KV when 8 / 32× was intended.
- **Acceptance length misreported as 2.65–2.69** — that was `sort -u | head`. True mean
  **2.706** (n=58, p50 2.70, max 2.78) under `standard` rejection — better than the 2.51 the
  perf model assumes. The 0.6467 collapse came solely from `synthetic` rejection.
- **"The causal hack is unnecessary, just use TRITON_MLA"** — wrong here: Kimi-K3 fp8 KV
  decode requires a backend accepting an fp8 query. The causal rewrite is a *consequence*
  of the fp8 KV choice.
- **Patch [2] "target not found"** — the check container had no GPU, which changes vLLM's
  import behaviour. The real run log shows `[triton-mla-cudagraph] patched`.

---

## 9. Conclusion

**DCP works, is correct, and is structurally unprofitable on this workload.**

The decode penalty is a fixed per-step cost of doing the gather/merge at all — independent
of ranks, batch, algorithm, granularity, backend, and CPU placement. Twelve levers measured,
all negative. The cost lands on a collective DCP doesn't own, inflated 8.8× because DCP's
host-side work starves one rank while seven wait.

One lever remains: the **all-reduce implementation** (T48). If PyNCCL/RCCL doesn't move it,
the next step isn't another config trial — it's instrumenting vLLM's DCP host path to find
what serialises one rank per step.

Against 12,500 tok/s/GPU: this stack reaches ~2,800, the SA reference 5,388. **The gap is
not DCP-specific — it's structural to this configuration.**

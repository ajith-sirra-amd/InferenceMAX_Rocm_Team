# Kimi-K3 on 8× MI355X — Performance Investigation

Kimi-K3 (2.8T MoE, 93 layers of which 24 are full-attention MLA, 96 heads, 1M
context, MXFP4) · vLLM ROCm · agentic-replay workload · concurrency 20.
**Target: 12,500 tok/s/GPU.**

---

# 1. Current state

| Configuration | tok/s/GPU | TPOT | TTFT | Speculation | Complete |
|---|---:|---:|---:|---|---|
| SA reference (MI355X) | 5,388 | 0.0382 | 12.2 s | MTP, **synthetic** acceptance | yes |
| **Best complete — TP8, AITER MLA** | **4,622.8** | **0.0461** | **2.14 s** | none, real tokens | yes, 3,612 s |
| DCP=8 + full graphs, no offload | 4,551.0 | 0.0499 | — | none | **no — GPU fault @ 2,771 s** |
| DCP=8 + full graphs + offload | 4,421.2 | 0.0497 | — | none | **no — GPU fault @ 3,033 s** |
| DP2/TP4/EP8 attention | 2,998.6 | 0.1140 (p50 0.0652) | 8.57 s | none | yes |
| TP8 + MTP | 2,045.4 | 0.1003 | 67.7 s | MTP, synthetic | yes |
| TP8 + DCP=8, piecewise graphs | 1,574.5 | 0.2174 | 12.4 s | none | yes, 3,628 s |

The best complete run reaches **86% of the reference's throughput without
speculation**, where the reference runs MTP with synthetic acceptance (drafts
committed regardless of target logits, which inflates its token count).
Against the 12,500 target: **37%**.

**DCP is no longer the loser it appears to be further down this page.** Enabling
full CUDA graphs moved DCP from 1,574.5 to 4,551.0 tok/s/GPU (+189%) and TPOT
from 0.2174 to 0.0499 — see 2.0. Those runs are listed as incomplete because
they die on a GPU page fault before the benchmark window closes, so they are
**not** claimable results yet. Closing that fault is the current blocker.

**Best-known configuration**
```
TP8 · no DCP · no DP · no speculation
attention: target backend UNSET  -> ROCm selects ROCM_AITER_MLA
           prefill ROCM_AITER_FA via --attention-config
--max-num-batched-tokens 8192 · --kv-cache-dtype fp8 · --max-num-seqs 40
--gpu-memory-utilization 0.9 · DRAM KV offload (SimpleCPUOffloadConnector)
image: vllm/vllm-openai-rocm:nightly-ac7509e2b1db40fec2f03dde1ed4e9dfdc2338c9
patches: [1] aiter pybind11 · [2] TritonMLA cudagraph · [3] KV block-pool clamp
GPU KV cache 4,376,929 tokens (4.17×)
```

---

# 2. Findings, in priority order

### 2.0 DCP's penalty was a ROCm-only cudagraph downgrade, not DCP itself

For most of this investigation DCP looked structurally slow — roughly a third of
plain TP8. It was not. `vllm/platforms/rocm.py` silently downgrades the
cudagraph mode whenever DCP is on:

```python
if compilation_config.cudagraph_mode.has_full_cudagraphs():
    # decode context parallel does not support full cudagraphs
    if parallel_config.decode_context_parallel_size > 1:
        compilation_config.cudagraph_mode = CUDAGraphMode.PIECEWISE
```

This gate exists **only on the ROCm path** — `cuda.py` has zero occurrences, so
NVIDIA DCP keeps full graphs. Piecewise splits the graph at every attention op
and hands each decode step back to the host to relaunch. Host-launch latency was
already the profiled bottleneck (3.4), so this downgrade lands directly on the
limiter, and every DCP number recorded before this point was measuring the
downgrade rather than DCP.

Lifting it (vLLM PR #51705, which gates the same code on
`VLLM_ALLOW_DCP_FULL_CUDAGRAPH`):

| | piecewise | full graphs | |
|---|---:|---:|---|
| tok/s/GPU | 1,574.5 | 4,551.0 | +189% |
| TPOT | 0.2174 | 0.0499 | −77% |
| prefix cache hit | 71.0% | 94.3% | |

DCP now sits within 2% of the best non-DCP run while carrying **31× the KV
capacity** (32.8M vs 4.4M tokens). That combination — reference-class TPOT at
32× cache — is the only configuration on this page with a credible path past the
reference, because it is the only one where concurrency can rise without
evicting. KV usage at concurrency 20 was 10.5% with zero preemptions.

**Blocker.** Both full-graph runs die on a GPU page fault before the window
closes (2,771 s and 3,033 s). Until that closes these are not results. See 2.0b.

### 2.0b The full-graph page fault — evidence and open question

All eight ranks fault simultaneously with `VM_L2_PROTECTION_FAULT_STATUS`,
`PERMISSION_FAULTS: 0x3`, IH client `0x1b (UTCL2)`, TCP client — a
vector-memory out-of-bounds **read**.

What the addresses say: the eight faulting addresses share the same low offset
(six at `...ce000`, two at `...cc000`) across completely unrelated bases. That is
one logical per-rank buffer overrun by a *fixed* amount on every rank. A KV or
block-table walk running off the end would scatter across ranks instead.

Two candidate causes, currently confounded — the last clean run had **neither**
full graphs **nor** the aiter opus-rows patch, while both faulting runs had
**both**:

| Run | opus rows removed | full graphs | outcome |
|---|---|---|---|
| T66c | no | no | clean, 3,628 s |
| T73 | yes | yes | fault @ 3,033 s |
| T74 | yes | yes | fault @ 2,771 s |

The opus patch is not a small change. Untuned-GEMM fallbacks per run went from
**88** (T66c) to **42,320** (T74) — ~480×. Stripping opus rows from the tuned
GEMM CSVs does not merely drop a few bad configs; it pushes thousands of shapes
onto a fallback dispatch path, which is a credible source of a mis-sized
workspace. Note also that this row-removal is a blunter instrument than what
ROCm/aiter#4915 does upstream, so any fault here may belong to the local
approximation rather than the PR.

Explicitly **not** evidence: untuned-GEMM log lines appearing just above the
fault. They run 42,320 deep from line 202 onward, so anything that faults will
have them adjacent.

A secondary hypothesis — `min_kv_seq_len` frozen into the captured graph at
`rocm_aiter_mla.py:1268`, since Gluon derives `NUM_KV_SPLITS` from it as launch
geometry — reads the wrong way round on inspection: a stale split count is
hazardous when replay sequences are *shorter* than capture, whereas these faults
arrive only after ~46 min of context *growth*. Not eliminated, but demoted.

Open: T77 re-runs the faulting configuration with the opus patch disabled and
full graphs kept, moving exactly one variable.

### 2.1 The decode attention backend is worth 2.4×
Forcing `--attention-backend TRITON_MLA` on the target model costs **72% of
throughput and 60% of TPOT** versus leaving it unset and letting ROCm select
`ROCM_AITER_MLA`.

| | forced TRITON_MLA | AITER MLA | Δ |
|---|---:|---:|---|
| tok/s/GPU | 2,685.0 | **4,622.8** | **+72.2%** |
| TPOT mean | 0.1161 | **0.0461** | **−60.3%** |
| TTFT mean | 3.450 s | **2.143 s** | −37.9% |
| requests OK | 569/611 | 1,046/1,090 | +84% |

Single variable: same image, patches, KV pool (byte-identical 4,376,929 tokens),
concurrency and spec setting. The reference sets **no** target backend; the
`TRITON_MLA` string in its config belongs to `--speculative-config`, i.e. the
**draft** model. AMD documents AITER MLA at 1.2–1.6× TRITON_MLA; measured here
it is 2.4× on this workload.

### 2.2 All three parallelism/speculation features lose against plain TP8

| Feature | Δ vs TP8 | Mechanism |
|---|---:|---|
| **DCP=8** | **−66%** | Collective-bound. Delivered its full capacity promise — 31.26× KV, 71.0% prefix hit, ≤15% utilised, `ext_cache_hit` 0.0% — and still lost 3×. |
| **MTP** | **−56%** | Accelerates decode, which is **0.6% of scored tokens**, while consuming 36% of the KV pool serving the other 99.4%. TTFT 2.14 s → 67.7 s. |
| **DP attention** | **−35%** | Best of the three. TPOT p50 0.0652 vs mean 0.1140 — a spread no other configuration shows. |

### 2.3 KV capacity is not the constraint
DCP=8 supplies 7.5× the KV pool (32,779,397 vs 4,376,929 tokens), raises prefix
hit from ~52% to 71.0%, never exceeds 15% utilisation and never spills to the
DRAM tier — and is still 3× slower. **99.4% of scored tokens are prefill**
(input 21,360 tok/s vs output 119), and prefill throughput is set by compute and
synchronisation, not by spare decode KV.

### 2.4 The workload is prefill-dominated
Any optimisation targeting decode addresses 0.6% of the scored tokens. This
single fact explains why MTP loses, why DCP's decode-side capacity is worthless
here, and why TTFT regressions matter more than TPOT regressions.

### 2.5 DCP + MTP is unavailable on ROCm
`mla_attention.py::_validate_dspark_dcp_support` requires one of two capability
flags. Across the entire image:
```
supports_non_causal_multi_token_dcp = True  ->  tokenspeed_mla.py  (only)
supports_dcp_with_varlen            = True  ->  flashinfer_mla, flashattn_mla,
                                                flashinfer_mla_sparse, flashmla_sparse
```
Neither `triton_mla.py` nor `rocm_aiter_mla.py` declares either. `TOKENSPEED_MLA`
is absent from `platforms/rocm.py` and not installed. `triton_mla.py` also
disables its draft path under DCP in code:
```python
# DCP local sequence lengths are not advanced between draft steps.
self.supports_draft_decode_metadata_update = self.dcp_world_size == 1
def update_draft_decode_metadata(self, _metadata): pass   # no-op
```
Setting the flag alone would let drafting proceed on stale per-rank KV ranges —
wrong output, no crash. This is an implementation gap, not a configuration gap.

### 2.6 The reference's advantage is speculation, not parallelism
The MI355X reference runs **no DCP and no EP** — plain TP8 + MTP. Its command
carries no `--decode-context-parallel-size`, no `--max-num-batched-tokens` and no
`--attention-backend`. Its 5,388 is measured with `rejection_sample_method:
synthetic` at `synthetic_acceptance_length: 2.51`.

### 2.7 Chunked-prefill sizing controls the KV pool
`--max-num-batched-tokens` feeds
`mla_attention.py::determine_chunked_prefill_workspace_size`. Pinning it reserves
memory the KV pool cannot then have. With MTP enabled, omitting the flag moved
the pool from 1,385,293 → 2,646,059 tokens (1.32× → 2.52×), exactly matching the
reference. It is a memory knob, not only a scheduling knob.

### 2.8 DP attention is memory-feasible at intermediate layouts
Measured from the checkpoint: experts 1,446.46 GB (92.67%), non-expert 114.40 GB
(7.33%). With EP sharding experts across all ranks, non-expert weights shard
within the TP group and replicate only across DP:

| Layout | per-GPU | fits (259 GB budget) |
|---|---:|---|
| DP1/TP8 | 195.1 GB | yes — current |
| **DP2/TP4/EP8** | **209.4 GB** | yes |
| DP4/TP2/EP8 | 238.0 GB | yes |
| DP8/TP1/EP8 | 295.2 GB | **no** — exceeds the 288 GB card |

Only the TP1 extreme fails. Any DP layout requires `ep > 1` or the experts do not
shard.

---

# 3. Limiting kernels and where the time goes

Torch profiler, decode-only windows, 8 ranks. **Caveat: all existing traces were
captured with the forced TRITON_MLA backend** — the 2.4× handicap. Traces of the
corrected stack are in progress.

### 3.1 Whole-trace breakdown (rank 0, 15 s decode window)

| | DCP=8 | non-DCP |
|---|---:|---:|
| Total GPU kernel time | 14.801 s | 5.033 s |
| Kernels launched | 121,030 | 235,224 |
| `aiter::cross_device_reduce_2stage<bf16,8,false>` | **78.37%** / 10,602 calls | 55.48% / 22,599 |
| `ncclDevKernel_Generic_1` | 8.22% / 950 | — |
| `mscclKernel_Sum_hip_bfloat16_Simple` | 6.06% / 912 | — |
| `mfma_moe1_silu_mul_afp8_wfp4` | 1.20% / 3,496 | 8.37% / 7,452 |
| `mfma_moe2_afp8_wfp4_bf16` | 0.71% / 3,496 | 4.96% / 7,452 |
| `_gemm_a16_w16` | — | 3.24% / 7,533 |
| `_attn_res_kernel` | — | 2.34% / 15,147 |
| **collectives / compute** | **92.65% / ~7%** | 56.05% / ~44% |

### 3.2 The decisive measurement

```
aiter::cross_device_reduce_2stage   SAME kernel, SAME 287 KB payload
  non-DCP :     8 us per call  (p50, every rank)
  DCP=8   : 1,094 us per call  (11.599 s / 10,602 calls)
  ratio   : 137x
```

Per-step kernel density is unchanged (DCP 1,345 vs non-DCP 1,232 kernels/step)
and compute per step is identical (11.6 vs 11.5 ms). **The kernel did not become
slower; it was made to wait.**

Per decode step, ~118 all-reduces in **both** arms:

| | all-reduce | compute | steps/15 s | TPOT |
|---|---:|---:|---:|---:|
| non-DCP | 14.6 ms | 11.5 ms | ~191 | 43 ms |
| DCP=8 | **129 ms** | 11.6 ms | ~90 | 167 ms |

### 3.3 Root cause on the profiled stack: host-launch starvation

An all-reduce returns when the **last** rank arrives, so the rank spending the
least time inside it arrives late.

```
DCP=8 all-reduce seconds per rank
  r0 11.72  r1 11.71  r2 10.13  r3 9.32  r4 10.59  r5 11.69  r6 8.47  r7 2.07  <- straggler
  p50 ~1000 us on 7 of 8 ranks -> waiting on EVERY call, not a tail
  non-DCP for contrast: p50 8-9 us on every rank
```
```
Host timeline over the same 15.20 s window
  straggler  gpu_busy  5.15 s  idle 10.05 s (66%)  123,900 launch gaps
             gap p50 62 us  p90 190 us  max 1.2 ms
  normal     gpu_busy 14.86 s  idle  0.35 s ( 2%)  116,678 gaps
             gap p50 0 us  p90 0 us
  host op totals IDENTICAL: 96,452 ms vs 97,114 ms
```
The straggler's GPU waits ~62 µs after each kernel for the host to launch the
next, ~124k times ≈ 10 s idle. Host *work* is identical between ranks, so this is
launch **delay**, not extra work. The other seven block inside the all-reduce,
which is why the cost is *attributed* to collective time.

### 3.4 Limiters, ranked

| # | Limiter | Evidence |
|---|---|---|
| 1 | **~118 global barriers per decode step** | Identical count in both arms. Each all-reduce is a hard sync across 8 ranks, so per-rank jitter is amplified 118× per step — this converts 62 µs of host delay into 129 ms/step. |
| 2 | **Host launch latency on the critical path** | Straggler idle 66%, ~124k gaps, identical host work, `max_concurrent_batches=1` leaves nothing to overlap. |
| 3 | **DCP adds synchronisation while sharding no work** | compute/step 11.5 → 11.6 ms. Only 24 of 93 layers are full-attn MLA, so there is little to shard. |
| 4 | **Workload shape** | 99.4% prefill; decode-side capacity is not the constraint. |

### 3.5 Which kernels would need acceleration

**No kernel needs to be made faster.** The dominant 78.37% is *blocking time*
charged to a kernel that completes in 8 µs when not waiting. Substituting the
implementation (PYNCCL for AITER custom) produced **−23% throughput, +31% TPOT** —
replacing the kernel makes it worse.

Addressable work, by measured share and tractability:

| Target | Share / status | Notes |
|---|---|---|
| **Fusion to cut barrier and launch count** | attacks limiters #1 and #2 | NVIDIA ships `VLLM_ENABLE_K3_LATENT_MOE_TAIL_FUSION`; **absent from the ROCm build**. The only lever that hits both limiters. |
| **DCP's own collectives** | `ncclDevKernel_Generic_1` 8.22% + `mscclKernel_Sum` 6.06% ≈ **14.3%** | Genuine work (gather + LSE merge), but a minority — perfect execution caps the win near 14%. |
| `cp_lse_ag_out_rs` direct path | ported, **−0.9%** | `direct_dcp_a2a_lse_reduce` hand-ported to HIP. Functional, no gain. |
| `q_gather` / `kv_gather` direct path | **not portable** | `multimem.st.*` PTX (NVLink hardware multicast, `__CUDA_ARCH__ >= 900`). No AMD equivalent. |
| Query all-gather elimination | `VLLM_DCP_Q_REPLICATE` | Requires `DCPGroupColumnParallelLinear`; Kimi-K3 builds plain `ColumnParallelLinear` (`models/kimi_k3/amd/linear.py:376,384`). Addresses part of the 14.3%. |
| Remove host from the launch path | untested | Full-step graph capture. Async scheduling is the cheap version and did not help. |

**Summary:** the cost is *serialisation*, not arithmetic. Kernel-level work can
address at most the ~14.3% that DCP's own collectives occupy. The other 78% is
seven GPUs idling in a barrier waiting on one host thread, fixable only by
removing barriers or removing the host from the launch path.

---

# 4. Recommended direction

1. **Keep the target attention backend unset.** Largest single effect measured.
2. **Do not enable DCP or MTP** on this workload at c20 with this request mix.
3. **Pursue fusion** — `VLLM_ENABLE_K3_LATENT_MOE_TAIL_FUSION` has no ROCm
   equivalent, and barrier/launch count is the binding constraint.
4. **Investigate DP attention's latency tail** — TPOT p50 0.0652 against mean
   0.1140 suggests the all-reduce removal works and DP-group imbalance costs the
   difference. Load balancing across DP groups is untested.
5. **Upstream request, precisely scoped:** implement
   `update_draft_decode_metadata` for a ROCm MLA backend so DCP local sequence
   lengths advance between draft steps, then declare
   `supports_draft_decode_metadata_update` and
   `supports_dcp_with_varlen` / `supports_non_causal_multi_token_dcp`.

---

# 5. Measurement hazards in this harness

Two reported fields produce incorrect conclusions if taken at face value.

| Field | Behaviour | Impact |
|---|---|---|
| `agg_bmk.json` → `spec_decoding` | Reports the **matrix label**, not engine state. Read `'none'` while the engine logged `speculative_config=SpeculativeConfig` and 364 SpecDecoding samples. | Caused a run to be recorded as speculative when it was not, and a non-existent "2.7× decode regression" (a spec-on vs spec-off comparison). |
| `agg_bmk.json` → `per_gpu.total_tput_tps`, and the job footer `Throughput per GPU` | Divides TOTAL by **`tp`**, not the GPU count. Correct whenever `tp` == GPUs; wrong by `8/tp` for any DP layout. | DP2/TP4 reported **5,997.2 tok/s/GPU** — apparently beating the reference — against an actual **2,998.6**. Confirmed from the same run's `world_size=8`, `rank=0..7`, `local_rank=0..7`, `EngineCore_DP0/DP1`. |

**Any DP or DP+EP result published from this harness is overstated 2× unless the
divisor is corrected.** Aggregates should be reconciled against the engine log.

A third hazard is reproducibility rather than measurement. vLLM PR #51705 was
fetched at runtime and pinned by sha256; the PR was then force-pushed **twice in
one day**, and each time the pin correctly refused to apply — which silently
removed the DCP patches and killed the run at engine init with `Decode Context
Parallelism (DCP) requires attention implementations to return the softmax LSE
during decode`. The pin behaved exactly as designed, but a hard pin against an
actively developed PR guarantees repeated breakage. The diff is now **vendored
in-repo** (`pr51705_vllm.diff`, `vllm/`-only), so the exact bytes that produced a
result sit next to the script that applies them, with no network fetch. Any
comparison spanning that change must confirm which revision was actually in the
image.

---

# 6. Configuration reference

### 6.1 In-container patches

| # | Name | Purpose |
|---|---|---|
| [1] | aiter pybind11 | Standalone pybind11 (internals v12) outranks torch's bundled copy (v11) via `-I` vs `-isystem`. Separate type registry per internals id → JIT module cannot see `aiter_tensor_t` → `TypeError` at warmup. Unblocks `ROCM_AITER_FA` prefill. |
| [2] | TritonMLA cudagraph | Raises `_cudagraph_support` to `UNIFORM_BATCH` so the DSpark drafter keeps FULL cudagraphs. Measured 14.05 → 77.65 tok/s single-stream (5.52×). **Incompatible with speculation on current nightlies** — it permits full capture, which then trips `assert m.max_query_len <= reorder_batch_threshold`. Skipped when spec is on. |
| [3] | KV block-pool clamp | `allocate_external_computed_blocks()` can pass a negative count to `get_new_blocks`, which is silently destructive (`num_free_blocks -= n` increases it; `range(n)` iterates zero times). A later pop walks past the tail. Load-dependent: c10 died at 3612 s, c12 at 487 s, c16 at 354 s. Requires `--kv-transfer-config`. |
| [4] | DCP-LSE plumbing | Plumbs softmax LSE + round-robin CP through AITER MLA decode. aiter's LSE is natural-log with sm_scale folded, `[B,H]` fp32. Superseded by [5]. |
| [5] | vLLM PR #51705 | Upstream DCP for Kimi-K3 DSpark, **and** the `VLLM_ALLOW_DCP_FULL_CUDAGRAPH` gate that unlocks 2.0. Now **vendored** in-repo as `pr51705_vllm.diff` and applied from disk. Does **not** fix the 0x1016 fault. |
| [6] | DCP block-table sizing | **The 0x1016 fix.** Tables sized `cdiv(max_model_len, block_size × dcp_size)` but indexed with the undivided `max_model_len`. Boundary tracks the ratio exactly: DCP8/bf16 faulted at 134,400; DCP4/bf16 at 262,656; DCP8/fp8 at 135,168 — block **count**, not bytes. |
| [7] | direct DCP a2a (ROCm port) | vLLM compiles the kernel only under `VLLM_GPU_LANG == CUDA`. The a2a combine ported to HIP (`st/ld.global.{release,acquire}.sys.u32` → `__hip_atomic_*` at SYSTEM scope). Gather kernels use `multimem.st.*`, not ported. **−0.9%.** |
| [8] | DCP gathered-head sizing | Supplies PR #51705's failing hunk (`_decode_num_heads = num_heads × dcp_world_size`). Requires image `5a4c8d99`; on `ac7509e2b` the builder predates the plumbing and lacks `dcp_world_size`. |

Patches [5], [6] and [8] are DCP-only and are skipped on non-DCP arms.

### 6.2 Layout constraints

```
TP × DP must equal 8 (the GPU count)
DP > 1 requires EP > 1, or the experts do not shard
tp_size % dcp_size == 0        -> at TP=4, legal DCP is 1, 2, 4
weights/GPU = non_expert/TP + experts/(EP ? ranks : TP)   must be < 288 GB
DCP patches require image 5a4c8d99; ac7509e2b lacks PR #51705 plumbing
```

---

# 7. Trial ledger

| # | Run | Configuration | Result |
|---|---|---|---|
| 1 | 32025696861 | patch [4], DCP8, bf16 | FAIL 0x1016 @134,400 |
| 2 | 32039650984 | [4], DCP4, bf16 | FAIL 0x1016 @262,656 |
| 3 | 32042030173 | PR#51705 only, DCP8 | FAIL 0x1016 @135,168 |
| 4 | 32043813560 | #51705 + **[6]** | **0x1016 fixed**, errors=0 |
| 5 | 32049134216 | + AIPERF_FAST | 742.4, TPOT 663 ms; 113/183 dropped |
| 6 | 32055440757 | GSM8K gate | timeout — 3.5–6.8 tok/s, ≈15 h needed |
| 7 | 32060082326 | + PIECEWISE cudagraph | 780.6, TPOT 0.687 |
| 8 | 32066978737 | DCP off, [1]–[3] off | FAIL — `AITER_DISABLE_FMHA_OPUS` only set in the DCP branch |
| 9 | 32068474469 | DCP off, [1]–[3] on | FAIL @25 min — RCCL watchdog |
| 10 | 32070778181 | DCP off, [1][2][3] | FAIL @21 min — same signature, [5] not the cause |
| 11 | 32073039787 | DCP off, [2] off | 1,014.9, TPOT 0.170 |
| 12 | 32077536567 | + full profile | 999.4, TPOT 0.1698 |
| 13 | 32084553677 | chunk 8192→32768 | lost to the workflow concurrency rule |
| 14 | 32089974051 | yukiozzz image, c64, MTP | FAIL — `mla_gluon requires batch_size=1, got 64` |
| 15 | 32090860356 | yukiozzz, c64 | FAIL — same, got 3 |
| 16 | 32093774227 | c64, chunk 32768 | cancelled @20 min |
| 17 | 32094907936 | DCP8 c20 DRAM | FAIL — leftover `MAX_NUM_BATCHED_TOKENS=32768` |
| 18 | 32096487055 | DCP8 c20 **DRAM offload** | **1,990.8**, TPOT 0.1742 — 781 → 1,991 (2.55×) |
| 19 | 32101946357 | conc 64 | 1,041.1, TTFT 501 s |
| 20 | 32110276088 | TRITON_MLA + MTP | cancelled @55 min |
| 21 | 32114847961 | TRITON_MLA, spec off | 1,948.4 |
| 22 | 32123047671 | DCP off, c1 + c20, MTP | c20 **3,340.5**, TPOT 0.0429 |
| 23 | 32138970163 | GSM8K gate | **passed** 0.9659 / 0.9644 vs 0.9651 baseline |
| 24 | 32143146154 | DCP=2 | FAIL @init — `is_valid_num_heads(24)` |
| 25 | 32143877066 | DCP=4 c20 | **2,033.7**, TPOT 0.1674 — best DCP pre-correction |
| 26 | 32154159649 | DCP=4 **c8** | 969.4 |
| 27 | 32162233477 | direct symm-mem | FAIL @9 min — op absent, CUDA-only |
| 28 | 32163743641 | **ag_rs** combine | 1,978.4 |
| 29 | 32172937044 | interleave 1→16 | 1,977.2 |
| 31a | 32269094879 | local image tag | FAIL @6 min — harness uses `--pull always` |
| 31b | 32270805303 | **ported direct a2a** | 2,014.7 — **−0.9%** |
| 35e | 32333672290 | DCP=8 profiling | 92.65% collectives |
| 37 | 32340149740 | non-DCP profiling | 56.05% collectives |
| 38–41 | — | per-rank straggler analysis | straggler moves 7→0→1→5 |
| 45/46 | 32368684074 | NUMA pinning, node + per-rank | no effect / worse |
| 47 | 32372517517 | non-DCP, full window | 2,655.7, TPOT 0.1176 |
| 48 | 32381243949 | DCP=8 + **PYNCCL** | 1,531.6, TPOT 0.2271 — **−23%** |
| 50 | 32390477829 | non-DCP + MTP, FULL cudagraphs | FAIL @init — MLA full-capture assert |
| 51 | 32392005995 | non-DCP + MTP, c20 | cancelled — KV-starved at 1.32× |
| 54 | 32396466979 | non-DCP + MTP, **c8** | 541.0, TPOT 0.2921 |
| 55 | 32404418920 | EP=8 attempt | FAIL — `spec-decoding` also selects the script filename |
| 57 | 32406159156 | **EP=8** verified | 2,619.2, TPOT 0.1177 — neutral |
| 58 | 32414712217 | image bisect on ac7509e2b | 2,685.0, TPOT 0.1161 |
| 59 | 32424123796 | DCP + MTP on ac7509e2b | FAIL — config gate rejects the pair |
| 60 | 32425822552 | DCP + MTP on 5a4c8d99 | FAIL — no ROCm backend declares the capability |
| **64** | **32436403856** | **TP8, AITER MLA, spec off** | **4,622.8, TPOT 0.0461** |
| 65 | 32444354043 | + MTP | 2,045.4, TPOT 0.1003 |
| 66c | 32451498395 | + DCP=8 | 1,574.5, TPOT 0.2174 |
| 67b | 32458502570 | **DP2/TP4/EP8** | 2,998.6 corrected (5,997.2 reported), TPOT 0.1140 |
| 68 | 32466225062 | DP2/TP4 + MTP | cancelled |

---

# 8. Investigation history

### 8.1 Levers tested against the DCP decode penalty — all negative

| Lever | Result |
|---|---|
| DRAM KV offload | 2.55× as a feature; no effect on the straggler |
| Concurrency (c8 / c20 / c64) | optimum at c20: 969 / 2,034 / 1,041 |
| World size 8 → 4 | +2%; DCP=2 illegal (24 heads) |
| Combine algorithm a2a vs ag_rs | 2,034 vs 1,978 |
| Shard granularity interleave 1 → 16 | 1,977 |
| Attention backend (within DCP) | within noise |
| CUDA graphs | no TPOT effect |
| Async scheduling on/off | no effect; straggler persists |
| NUMA pinning, node-level | no effect |
| NUMA pinning, per-rank slices | worse (188 → 218 ms, rising) |
| Hand-ported direct P2P collective | works, −0.9% |
| All-reduce implementation (PYNCCL) | −23% throughput, +31% TPOT |
| MTP under DCP | refused at engine init |

### 8.2 Hypotheses the profiling eliminated

| Hypothesis | Evidence against |
|---|---|
| KV offload causes the straggler | Removing offload entirely; straggler persisted |
| A specific bad GPU or link | Straggler identity moves: rank 7 → 0 → 1 → 5 |
| GPU load imbalance | Compute per rank 1.05–1.11 s, ≤5% spread |
| DCP shards decode work | 11.5 vs 11.6 ms/step — no reduction |
| Async scheduling | Enabling it left the straggler intact (rank 5, 89% idle) |
| The collective implementation | PYNCCL substitution measurably worse |
| NUMA placement | No effect at node level; worse per-rank |
| A clean 1-vs-7 split | Intermediate ranks appear (265/272 µs) — a gradient |

### 8.3 Corrections made during the investigation

| Earlier claim | Correction |
|---|---|
| "The reference decodes on TRITON_MLA" | It sets no target backend; TRITON_MLA is the **draft's**. Forcing it cost 2.4× and was present in every non-DCP trial. |
| "The reference runs DCP=8 + MTP" | It runs **no DCP**. That configuration came from the B300 (NVIDIA) script — a different platform and file. |
| "MTP starves this workload" | The KV shortfall came from a pinned `--max-num-batched-tokens`. With it omitted the pool matched the reference exactly. MTP still loses, for a different reason. |
| "There is a 2.7× decode regression" | An artefact of comparing spec-on TPOT (per accepted token) against spec-off: `0.1161 / 2.509 = 0.0463` vs 0.0429. |
| "DCP's ceiling is ~2,000 tok/s/GPU" | Measured against a handicapped control. Re-baselined, DCP is −66%, not −24%. |
| "DP attention is infeasible on 8 GPUs" | Only the DP8/TP1 extreme fails. DP2/TP4 fits with ~50 GB spare and runs. |
| "The DCP decode cost is irreducible" | The cost is real but mostly **not DCP's** — it is the TP all-reduce any TP=8 decode pays, which DCP worsens by adding sync points. |

### 8.4 Script defect class — five instances

A value that appears set at one site is decided or discarded at another.

| Defect | Effect |
|---|---|
| Shadowed `DCP_SIZE` | An assignment inside the auto-concurrency block won; runs used DCP=4 / 17× KV when 8 / 32× was intended. |
| `DISABLE_SPEC` in a concurrency-gated block labelled DCP | Non-DCP arms silently ran without speculation while labelled `spec-mtp`. |
| Dead `CUDAGRAPH_CAPTURE_SIZES` | A sparse ladder overwritten by a later assignment in the same branch. |
| **Orphaned `EP_ARGS`** | Computed but never referenced in `VLLM_CMD` — **expert parallelism was unreachable for 55 trials**. |
| Split `DCP_SIZE` default | The patch gate read `1` while the run set `8`, so DCP ran without its patches. |

Structural remedies applied: `DCP_SIZE` resolves once before any consumer;
layout guards compute from measured weights rather than remembered rules; an
orphan check parses `VLLM_CMD` and diffs it against defined `*_ARGS` arrays.

### 8.5 Other notable failures

- A locally built image tag cannot resolve — the harness uses `docker pull --pull always`.
- `spec-decoding` in the matrix also selects the **script filename**; setting it to `none` pointed the harness at a non-existent file.
- `VLLM_TORCH_PROFILER_DIR` is obsolete; the knob is `--profiler-config`.
- A `-p2` patch dry-run reported success where a correct `-p1` showed only one nightly applies (23/24 hunks).
- Patch [8]'s guard matched *uses* of `_decode_num_heads` rather than its assignment, reporting "already patched" against a partial application.
- Acceptance length was misread from `sort -u | head` (the bottom five values) rather than the distribution; true mean 2.706 under `standard` rejection.
- Kimi-K3 fp8 KV decode requires a backend accepting an fp8 query input, so the draft-causal rewrite is a consequence of the fp8 KV choice, not an independent option.

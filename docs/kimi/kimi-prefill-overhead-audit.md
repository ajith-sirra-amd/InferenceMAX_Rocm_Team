# Kimi prefill: code path, overheads and next-run gate

[中文](kimi-prefill-overhead-audit_zh.md) · [Baselines](atom-attention-BASELINES.md) · [Current adapter](atom-flydsl-attention.md)

## Problem first

- Goal: beat the verified Moonshot Kimi-K3 baseline of **15,476.47 total tokens/sec/GPU**, with vLLM on 8×MI355X.
- A faster FP8 FMHA kernel can still produce a slower complete attention call: expansion, quantization, communication and merging also consume time.
- Previous all-FP8 agentic smoke completed at **11,789.99 total tokens/sec/GPU**; its short duration and request errors prevent a matched baseline comparison.
- Current candidate keeps BF16 where measured conversion overhead outweighed FP8 compute savings and uses FP8 for larger cached-context chunks.
- Current smoke: [40643563020](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/40643563020), source `e4cf47a7650321e4bd40c83746088a2679ec0509`. Artifact verification passed; see the verified smoke and full-replay status below.

## References: checkpoint names matter

| Reference | Checkpoint | GPU KV capacity | Total tokens/sec/GPU | Output tokens/sec/GPU |
|---|---|---:|---:|---:|
| Primary baseline, run39239112271 | Moonshot Kimi-K3 | 20,951,355 | 15,476.47 | 103.05 |
| Capacity reference, run39601846940 | AMD Kimi-K3, Quark quantized | 23,098,919 | 15,258.78 | See raw baseline JSON |
| Old all-FP8 smoke, run40634885348 | Moonshot Kimi-K3 | 22,695,620 | 11,789.99 | 61.13 |

- Historical notes sometimes call the primary reference “AMD baseline”; the actual checkpoint is **Moonshot**.
- AMD Quark and Moonshot have different checkpoint quantization. The current experiment uses **Moonshot only** and online FP8 targets.
- KV capacity is measured after runtime allocations; smaller weights alone do not establish a 23M pool.
- Total throughput includes input tokens, including cached input. Report output throughput and cache-hit rate separately.
- Primary baseline reporting window: 3629.50 seconds; old smoke: 620.30 seconds. These are not matched experiments.

## Exact precision and matrix flow: TP8 / DCP8

- Model: 96 attention heads; **12 heads per TP rank**.
- Use `T` for tokens handled by the current operation; collective buffers include their own packing and padding.

| Step | Tensor / matrix dimensions per rank | Precision | Work and implication |
|---|---|---|---|
| Stored compressed KV | Latent `C:[T,512]`, RoPE `K_pe:[T,64]` | FP8 cache, with scales | Stores compressed representations, not expanded per-head K/V |
| Native fused DCP route | Compact `[T,576]` representation | FP8 communication | FP8 payload is 576 bytes/token before padding and collective scheduling |
| Latent up-projection | `C[T,512] × W_kv_bᵀ[512,3072]` | Current candidate has BF16 `W_kv_b[3072,512]` | Native fused route expands latent KV using weights/scales |
| Split and RoPE assembly | `K:[T,12,192]`, `V:[T,12,128]` | BF16 | K includes 128 non-RoPE dimensions plus 64 RoPE dimensions |
| Query entering prefill | `Q:[Tq,12,192]` | BF16 | FP8 projection weights do not imply FP8 output activations |
| Causal / small cached FMHA | Same expanded head dimensions | BF16 inputs/output | Avoids extra FP8 preparation on cases with measured regressions |
| Large cached FMHA preparation | Expanded Q/K/V plus per-tensor scales | BF16 → E4M3 FP8 | Added quantization work; Q can be reused within the same forward |
| Large cached FMHA | `Q8,K8,V8`, 192/128 head dimensions | FP8 operands; BF16 output; FP32 LSE | AITER FlyDSL API used by ATOM |
| Chunk combination | Partial attention output and LSE | BF16 output / FP32 LSE | Native normalization/merge remains necessary |
| Decode and draft | Native vLLM AITER path | Native configuration | Main-model prefill changes do not replace draft or decode kernels |

**Current cached flow:**
`compressed FP8 cache → native fused FP8 DCP gather + latent expansion → BF16 expanded K/V → FP8 Q/K/V preparation → FlyDSL FP8 FMHA → BF16 output + FP32 LSE → native merge`.

- This is **FP8 communication plus BF16 expansion**, followed by FP8 attention preparation.
- Direct production of expanded FP8 K/V is **not implemented**.
- BF16 `kv_b_proj` weights are an eligibility requirement of the pinned native fused route used here; this is not a mathematical requirement of FP8 all-gather.
- The pinned fallback gathers compact BF16 latent KV first, reorganizes it, then invokes the `kv_b_proj` module to expand K/V. Its collective payload is 1,152 bytes/token before padding, versus 576 bytes/token for compact FP8. The candidate temporarily observes that module call to distinguish the actual route.
- Quantizing these weights to FP8 may save memory but can select the fallback when the fused implementation cannot consume that weight format.

## Measured complete-call costs

GPU diagnostics use 12 heads on a shared host. These are CUDA-event timings, not full-model throughput or hardware-counter roofline data.

| Noncausal case | BF16 complete | FP8 preparation + FMHA complete | BF16 time / FP8 time |
|---|---:|---:|---:|
| Q256 / KV4096 | 88.47 µs | 107.03 µs | 0.827×: regression |
| Q1024 / KV8192 | 177.39 µs | 107.13 µs | 1.656×: improvement |
| Two chunks, Q1024 / KV8192 | 330.84 µs | 201.20 µs | 1.644×: improvement |

| Component | Measured or estimated cost | Status |
|---|---|---|
| QKV quantization | Approximately 55.3–55.7 µs | Measured diagnostic |
| K/V quantization when Q is reused | Approximately 51.7 µs | Measured diagnostic; most conversion cost remains |
| FP8 FMHA only | Approximately 36.3 / 50.5 µs for the two single-chunk cases | Measured diagnostic; excludes preparation |
| Expanded BF16 K/V storage | 7,680 bytes/token/rank | Logical tensor size, not measured HBM traffic |
| Expanded FP8 K/V storage | 3,840 bytes/token/rank | Logical tensor size |
| K/V conversion minimum read + write | 11,520 bytes/token/rank | Estimate; excludes scales, reductions, allocations and rereads |
| BF16 kv_b weight tradeoff | Approximately 36 MiB/GPU over 24 MLA layers versus FP8 | Raw weight estimate; runtime KV pool still must be measured |
| hipBLASLt recovery in old smoke | 96 warnings across 12 GEMM shapes, m2880/k7168 | Observed recovery to BLAS fallback; duration and invocation count unmeasured |
| DCP collective, expansion GEMM, offload and merge time | Not isolated in current evidence | Do not substitute GPU peak bandwidth estimates for measurements |
| Prior model loading | Approximately 23.6 minutes, 184.12 GiB/GPU loaded | Startup cost in old smoke; excluded from its reporting window |

- Old-smoke request JSONL confirms warmup14success/85empty-response errors and measured476success/2empty-response errors. Watch the new server logs for the GEMM recovery warnings; their duration and contribution to throughput are unmeasured.
- Numerical FP8 checks show roughly 4–5% RMS drift versus an unquantized FP64 reference.
- Numerical checks are not model accuracy evidence. GSM8K run40658068830 passed: strict-match 1282/1319 (97.1948%) and flexible-extract 1281/1319 (97.1190%), both above 90%. No FP8 FMHA engagement markers occurred; this validates the serving configuration on BF16-routed GSM8K prompts, not the cached FP8 kernel.
- Crossover policy is provisional: BF16 if `max_query_len < 512` **or** `max_seq_len < 8192`; FP8 otherwise. It is supported by the measured examples, not an exhaustive shape sweep.

## Avoidable overhead and implemented changes

| Cost | Change | Remaining limit |
|---|---|---|
| Causal FP8 conversion regressions | Retain native BF16 causal FMHA | No causal FP8 claim |
| Small-context conversion dominates | Retain BF16 below the shape thresholds | More shapes may need future tuning |
| Repeated Q quantization across context chunks | Reuse identical Q views within one forward; clear in `finally` | K/V conversion remains; no cross-forward reuse |
| Unused main-model native ASM PS planning/scratch | Disable before allocation in custom metadata builder | Native draft PS buffers may still be allocated |
| FP8 kv_b weight override selects incompatible fused route | Remove `*.self_attn.kv_b_proj` from online FP8 targets | Costs approximately 36 MiB/GPU raw weight storage |
| Broad diagnostic printing | One-time path/config markers and one-call route hook | No per-request tensor dumps |
| Model accuracy evaluation | Separate full GSM8K run40658068830, real verification | Both metrics pass; no FP8 FMHA engagement |
| NCCL log flooding | `NCCL_DEBUG=VERSION`; explicit single-node loopback bootstrap | Reducing logging does not by itself fix a collective hang |

- FP8 → FP4 → FP8 **weight formats across layers do not inherently cause activation casts**.
- Check actual GEMM input/output dtypes and scale handling before attributing quant/dequant overhead to adjacent weight formats.
- `--load-format` is absent: vLLM auto loader is used. `SAFETENSORS_FAST_GPU=1` is set; it does not force `fastsafetensors`.
- `--language-model-only` is enabled.
- No installed vLLM/AITER source patches; the repository extension is loaded with `PYTHONPATH`.

## Smoke gate and next run

| Gate | Required evidence |
|---|---|
| Preload GPU check | Numerical preflight PASS and QUERY_REUSE_PASS before model loading |
| Requests | 20/20 successful controlled prefix requests; zero failed |
| Cached work | Prefix-cache-hit counter increases |
| All eight workers | BF16 causal, FP8 cached, disabled main ASM PS, Q reuse and native fused FP8 gather markers |
| Memory | Actual `GPU KV cache capacity` line |
| Revision | Tested source SHA matches the pinned full-run checkout |

- Prefix stages: 32K prefix / 1024 suffix / OSL16 / concurrency8; 128K prefix / 1024 suffix / OSL8 / concurrency1.
- Context workspace cap is 64K; the long-prefix stage exercises multiple context chunks.
- Watcher polls every 60 seconds; after the gate passes, it queues **full agentic replay, warmup10 per lane, duration3600 seconds**, with no GSM8K.
- The controller prevents a duplicate full dispatch and pins the tested source SHA.
- A failed or incomplete gate requires logs and a correction before another full run.
- Full-run comparison must include KV pool, total/output throughput, request errors, reporting duration and cache-hit rates.

## Pinned native implementation cross-check

- [Native AITER MLA](https://github.com/vllm-project/vllm/blob/1b2a666391f2a24228a1436b64d547c34cc6b6ce/vllm/v1/attention/backends/mla/rocm_aiter_mla.py): fused eligibility explicitly checks BF16 contiguous kv_b weights, absent weight_scale/bias, BF16 Q and row-index metadata.
- [Native common MLA](https://github.com/vllm-project/vllm/blob/1b2a666391f2a24228a1436b64d547c34cc6b6ce/vllm/model_executor/layers/attention/mla_attention.py): fallback communicates the compact latent representation in BF16, then expands; it does not communicate expanded per-head K/V.
- The64K chunk cap applies to global context rows. Under DCP8 the local row budget is64K/8, so a cached128K prefix requires multiple chunks.

## Code locations

Paths below are repository-relative.

| File | Responsibility |
|---|---|
| `upstream/InferenceX/benchmarks/single_node/agentic/atom_attention/inferencex_atom_attention/backend.py` | Hybrid FMHA selection, Q lifetime, metadata allocation and route marker |
| `.../inferencex_atom_attention/quantize.py` | Fused QKV preparation and KV-only reuse specialization |
| `.../inferencex_atom_attention/register.py` | vLLM extension registration |
| `.../atom_attention/path_probe.py` | Controlled prefix benchmark and invocation gate |
| `upstream/InferenceX/benchmarks/single_node/agentic/kimik3_fp4_mi355x_mtp.sh` | Server settings, quantization targets, preload check and replay |
| `utils/atom_attention/preflight.py` | Numerical, strided-input and changed-input graph checks |
| `utils/atom_attention/continue_benchmark.py` | Artifact gate, pinned dispatch and run monitoring |

- Working branch: `perf/kimi-attnfp8-kv-allgather-20261021`.
- Preserve `chore/sa-agentx-v1.0`; code writes stay in this repository.

## Verified prefix smoke and resumed full replay  2026-10-28

- Smoke [40643563020](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/40643563020) passed: **20/20 successful requests, zero failures**, and **620,544** additional prefix-hit tokens.
- Runtime capacity: **22,665,373 GPU KV tokens**. All eight workers confirmed BF16 causal, FP8 cached FMHA, disabled main ASM PS metadata, same-forward Q reuse, and native fused FP8 gather.
- Both numerical preflight and QUERY_REUSE_PASS succeeded. The latest Q1024/KV8192 complete-call diagnostic measured **172.28 µs BF16 versus 111.99 µs hybrid (1.538)**; two cached chunks measured **337.45 versus 204.27 µs (1.652)**. These remain single-GPU diagnostics excluding reconstruction and collectives.
- This smoke loaded **489 online-quantized layers per worker**, retaining BF16 kv_b weights. Model loading was **439.4440.3 seconds** and **184.08 GiB/GPU**, shorter than the older 23.6-minute observation.
- Requested FULL_AND_PIECEWISE was **resolved to FULL_DECODE_ONLY** because the model had no compiled submodule and breakable graphs were disabled. Earlier graph-sizing capture used **9.209.64 GiB/GPU**; final capture reported **6.67 GiB/GPU**. These are separate stages, not additive retained allocations.
- Runtime aligned attention blocks to **1536 tokens**, padded Mamba pages by **3.97%**, and warned that 18 added padding layers could waste at most **26.09%** of the affected cache grouping. The latter is a planner bound, not a measured global memory loss.
- No hipBLASLt recovery warning appeared in this controlled prefix smoke. That does not establish its absence in full AgentX shapes.
- Full performance run40646139769 succeeded: 16,915.72154 total and 117.84290 output tokens/s/GPU (+9.30% and +14.35% versus baseline39239112271). This is the complete candidate, not isolated FMHA attribution. [Results and evidence](fp8-fmha-results-2026-10-28.md). GSM8K run40658068830 passed: strict-match 1282/1319 (97.1948%) and flexible-extract 1281/1319 (97.1190%), both above 90%. No FP8 FMHA engagement markers occurred; this validates the serving configuration on BF16-routed GSM8K prompts, not the cached FP8 kernel.
- Full settings: Moonshot, TP8/DCP8, C48, DSpark k4, warmup10/lane, duration3600 seconds, no GSM8K. The completed setup-job log verifies checkout **e4cf47a7650321e4bd40c83746088a2679ec0509**; the workflow head is a later documentation commit.
- Full throughput, measured/warmup errors and cache-hit rates are now recorded in the linked results. GSM8K run40658068830 passed: strict-match 1282/1319 (97.1948%) and flexible-extract 1281/1319 (97.1190%), both above 90%. No FP8 FMHA engagement markers occurred; this validates the serving configuration on BF16-routed GSM8K prompts, not the cached FP8 kernel.

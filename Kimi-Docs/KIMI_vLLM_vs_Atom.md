# Kimi-K3 FP4 / MI355X — ATOM vs vLLM at C4 and C14

Same hardware (`cluster:mi355x-amds`, 8×MI355X, TP8, fp4, MTP). Values read from server commands
and engine kwargs, not job titles.

ATOM: [run 35487769456](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/35487769456/attempts/2) ·
vLLM: [run 35157825655](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/35157825655)

## Measured

| conc | engine | tok/s/GPU | ITL p90 | 1/ITL p90 |
|---|---|---|---|---|
| C4 | ATOM | **2,837.4** | **8.98** | **111.4** |
| C4 | vLLM | 2,798.0 | 9.42 | 106.2 |
| C14 | ATOM | **6,981.1** | 20.33 | 49.2 |
| C14 | vLLM | 6,764.8 | **19.84** | **50.4** |

vLLM loses both axes at C4. At C14 it loses throughput but wins interactivity — not dominated.

## Differences

**C4** — ATOM: k=7, mns 32, block 128, cudagraph FULL · vLLM: k=5, mns 8, block 1536 (flag absent,
engine-resolved), cudagraph FULL_AND_PIECEWISE.
Same: TP8, DCP1, no offload, gmu 0.90, mnbt 8192, fp8 KV.

**C14** — ATOM: DCP-8 + LMCache (1028 GB), mns 32, block 128 · vLLM: DCP-1 + vllm-simple (1799 GB),
mns 28, block 1536, ladder 112. Both cudagraph FULL.
MTP is **on in both** at k=3 — `SpeculativeConfig(method='dspark', num_spec_tokens=3)` each side.
vLLM's DCP-1 is by absence of the flag; ATOM passes `8` explicitly (confirmed in engine kwargs).

## k=3 misses the fused non-causal draft kernel

Both vLLM runs request `"attention_backend":"ROCM_AITER_MLA"`. Only one gets the fused non-causal
path from [vllm#55966](https://github.com/vllm-project/vllm/pull/55966). `msk0` = mask-off = non-causal.

| run | k | `msk0` loaded |
|---|---|---|
| SA vllm **c14** | **3** | **no** |
| SA vllm c4 | 5 | yes |
| our C12 (×2) | 4 | yes |
| probe C1 | 6 | yes |
| probe C4 | 5 | yes |

No error — the draft block is flattened into single causal rows instead. Not the offload connector:
our C12 has `offload=dram` *and* k=4 *and* loads `msk0`, so k is the only discriminator across six
runs. SA ships k=3 at C12, C14, C16 — all three are affected.

## DCP + MTP

ATOM runs MTP under DCP-8, logging `enable_query_replication disabled: speculative decode (qlen>1
cprr path) not supported in the first cut`. vLLM cannot run MTP under DCP at all —
`supports_non_causal_multi_token_dcp` is set only by `flashinfer_mla` and `tokenspeed_mla`, both
gated on CUDA `capability.major == 10`. Fix: [vllm#57085](https://github.com/vllm-project/vllm/pull/57085) (open, +11/−2).

## Caveats

- ATOM does not log kernel names, so whether **its** draft block runs fused is unknown.
- **aiter-causal vs fused-non-causal has never been measured in isolation.** The only number (our
  C12: 18.79 → 17.41 ms, 5,746 → 6,203 tok/s) changed both image and drafter backend, and its
  baseline was Triton, not aiter-causal. Clean test is C14 k=3 vs k=4 on one image — not yet run.
- C4 and C14 each have several uncontrolled variables; no single difference can be assigned a share
  of the gap.

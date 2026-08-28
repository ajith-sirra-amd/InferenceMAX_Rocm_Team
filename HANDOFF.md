# Kimi-K3 on 8× MI355X — handoff

Last updated 2026-08-28. Target **12,500 tok/s/GPU**.

## Best results

| point | metric | run |
|---|---|---|
| **C52 throughput** | **7,950.6 tok/s/GPU** | [T103](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/32855763638) — DCP=8, mns 80, DRAM offload, full graphs, no MTP |
| **C1 interactivity** | **8.37 ms TPOT · 33.46 ms ITL · 119.5 tok/s/user** | [T140](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/33160495127) — fixed-length 8000/2000, DCP off, k=8 @ AL 4.00, ladder 1…9 |
| C1, agentic harness | 6.70 ms TPOT · 149.31 tok/s/user | [T123](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/33029724147) — **different harness, not comparable to T140** |
| reference | 8,296.0 tok/s/GPU | [SA C52](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/32968517728) — DCP=8, mns 80, **DRAM offload**, `fastsafetensors`, KV 32,756,602 tokens |

Accuracy gate: **GSM8K 96.82% / 96.89%** over the full 1,319 problems.

## Read these first

| file | what |
|---|---|
| `Kimi-DCP-Experiemnts-Summary.md` | trials, results table, config history |
| `Kimi-K3-Where-The-Time-Goes.md` | rocprofv3 profile analysis |
| `IMAGE-RECIPE.md` | how to rebuild the image |
| `kimi-k3-profiles/` | raw rocprofv3 CSVs, committed verbatim |

## Boundaries — do not cross

- **Dispatch only to `ajith-sirra-amd/InferenceMAX_Rocm_Team`.** `SemiAnalysisAI/InferenceX` is **read-only** — fetch via API, never push.
- **Edit only**: `benchmarks/single_node/agentic/kimik3_fp4_mi355x_mtp.sh`, `apply_kimi_k3_patches.sh`, and the `kimik3-fp4-mi355x-vllm-agentic-mtp` block in `configs/amd-master.yaml`. Docs at repo root are fine when asked.
- **No Docker Hub pushes** without explicit permission.
- **Only cancel runs you started.** Other jobs on this box may be the user's.
- `ajith-sirra-amd/vllm` is the user's vLLM fork — push allowed there, but **sign as `ajith-sirra-amd`, never Claude**.

## Where the time goes (T124 profile, C52, no offload)

GPU busy 71.8%, idle 28.2%. Of **e2e wall**:

| | % wall |
|---|---:|
| **idle** (92.6% of it in decode) | **28.2** |
| collectives | 21.3 |
| MLA attention (prefill) | 16.9 |
| BF16 dense GEMM | 11.9 |
| FP8×MXFP4 MoE GEMM | 7.8 |

**Even eliminating idle entirely gives ~11,050.** Every remaining kernel lever is single-digit percent. 12,500 is not reachable with the levers identified so far — say so rather than implying otherwise.

## C1 ITL sweep (T138/T139/T140)

Fixed-length `run_benchmark_serving`, ISL 8000 / OSL 2000 / range-ratio 0,
non-agentic, DCP off, 10 prompts + 2 warmups. ~10 min per cell against the
agentic replay's 1h12m.

| | k=6 · ladder 56 | k=8 · ladder 72 | **k=8 · ladder 9** |
|---|--:|--:|--:|
| run | [T138](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/33158510878) | [T139](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/33159693747) | [**T140**](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/33160495127) |
| golden AL | 3.75 | 4.00 | 4.00 |
| Mean ITL (ms) | 32.70 | 33.63 | **33.46** |
| Mean TPOT (ms) | 8.70 | 8.42 | **8.37** |
| P99 TPOT (ms) | 8.83 | 8.47 | **8.43** |
| tok/s/user | 114.9 | 118.8 | **119.5** |
| Mean TTFT (ms) | 1055.5 | 1058.8 | 1048.9 |
| graph capture | 56 s | 65 s | **26 s** |
| graph memory | 1.21 GiB | 1.46 GiB | **0.83 GiB** |

Measured ITL/TPOT is **3.76** and **3.994** against golden AL 3.75 and 4.00 —
the synthetic-AL wiring lands to three digits.

**The C1 step is 87.5% fixed overhead.** Two points with only k varying solve
`step = a + b·(k+1)`: **b = 0.465 ms** per verify row, **a = 29.44 ms** fixed.
That model predicts both measured points to within 0.02 ms. Extrapolated over
the golden AL curve, TPOT is *still falling* at k=8 (16.42 / 12.29 / 10.44 /
9.46 / 8.91 / 8.72 / 8.64 / **8.41** for k=1…8) — **k=8 is the best only
because the golden AL table ends there**, not because the curve turned. A
marginal row costs 0.465 ms against a 33.6 ms step, so deeper speculation stays
profitable as long as AL rises at all. Getting AL past k=8 needs a new entry in
`golden_al_distribution/`.

**Ladder: C1 and C4 do not need a heavy one.** `mns` floored at 8, so C1
captured 8×9 = 72 graphs to serve a decode batch that is always exactly 9 rows.
Dropping the floor for CONC ≤ 4 cut capture 65 s → 26 s and graph memory 1.46 →
0.83 GiB/GPU. TPOT moved −0.6%, same direction at every percentile but too
small to quote from one run — call it a tie on latency, a win on cost. The
ladder stays **dense**, only shorter, so the sparse-ladder padding fault cannot
recur. C52 unchanged at `mns` 65.

Two client-harness facts worth carrying:

- **The runner exports `ISL=0`, `OSL=0`, `RANDOM_RANGE_RATIO=0.8`** into the
  container for the agentic scenario. `${ISL:-8000}` never fires — the variable
  is set, just to zero. T137 died with `ValueError: low >= high`. Use
  `ITL_ISL` / `ITL_OSL`, which the runner does not touch.
- **The runner checks `[ -f "$RESULT_FILENAME.json" ]` in the workspace root**,
  not in `$RESULT_DIR`. Pass `--result-dir /workspace/` as the `fixed_seq_len`
  scripts do, or the job goes red on a benchmark that actually succeeded (T138).
- **Effective ISL is 4,320, not 8,000.** The random dataset decodes token IDs to
  text and re-encodes; Kimi's BPE re-merges and the 10 convergence retries never
  close. Still one 8192 prefill chunk. Affects TTFT, not a C1 decode step.

## Settled — do not re-litigate

| thing | result |
|---|---|
| DRAM KV offload | Removing it cut idle **44.3% → 28.2%**. `mns` 80 without it dies with `HSA_STATUS_ERROR_OUT_OF_RESOURCES` — **on `mi355x-amd_b23_07` only**. It is **not** a config limit: SA ran exactly that ([33062469329](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/33062469329) job 98484457387 — C52, `KV_OFFLOADING: none`, `mns` 80, DCP=8) on `mi355x-amds_01` and got **8,204 tok/s/GPU**, 1984/2097 requests. **This is a node difference.** On our box, cap `mns` at 65 without the offload; on theirs, 80 is fine. |
| **SA's numbers are our own recipe** | Both SA C52 configs are the shape of our T103 (7,950.6): DCP=8, mns 80, `fastsafetensors` — **8,296 with `dram`**, **8,204 with `none`**. So the offload is worth ~1.1% to them, and we are **3–4% behind on an identical config**, not blocked. Two earlier claims here were wrong: that the 8,296 ran "no offload", and that `mns` 80 without offload died on SA too. |
| `mns` 65 + no offload at C52 | **Answered: no.** [T133](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/33130489071) = **7,725.96 tok/s/GPU**, 1817/1926 requests, TTFT p95 18.8 s. Survives the crash but lands **below** T103's 7,950.6, so it is not the route to 8,296. |
| **async scheduling** | **T108: 7,222.3 vs 7,950.6 = −9.2%.** Already tested on the **no-offload** path (`kv=none`, mns 80). Not stale. |
| QuickReduce FP | **−8.39%**. Accuracy fine. It *preempts* `AITER_CUSTOM` in dispatch order rather than supplementing it. |
| FP16 GEMM | Loses on **6 of 8** real shapes, up to −11.4%. BF16 already runs at full matrix-core rate (2.04× vs FP8 = the hardware ratio). |
| AMD `Kimi-K3-Quark-MXFP4-AttnFP8` | Loads and serves, but needs **240.75 GiB vs 192.63** → KV 9.38 GiB vs 59.81. Dropped. |
| MTP at high concurrency | −85% at C40. **Only** enable at CONC ≤ 4. |
| draft KV dtype `auto` → `fp8` | **TPOT no-op, KV win.** T143 11.93 vs T144 11.97 ms (122k ctx, C1) — identical. But the KV pool grew **15,077,972 → 20,580,438 tokens (+36.5%)** in the same 53.84 GiB: the draft was holding KV at 2 B/element while the target held 1. Keep fp8. Validate draft quality separately — `synthetic` rejection **imposes** AL, so bad drafts still report 4.00. |
| draft on `ROCM_AITER_MLA` | **Impossible.** `TRITON_MLA` is the only ROCm MLA backend declaring `supports_non_causal_multi_token_decode`, which the DSpark draft requires; the other two (`flashinfer_mla`, `tokenspeed_mla`) are NVIDIA. Flipping the ClassVar True is unsafe — the aiter ASM path has no gqa=64 kernel past qseqlen 1, and synthetic AL would hide wrong drafts. |
| chunk size | prefill knob; 16384 measured −2.5%. Moves TTFT, not TPOT. |
| EP=8 | −4.7% |
| C96 | 4,667.9, TTFT **122 s** (p95 414 s). Past the knee; offload doesn't rescue it. |

## Open — worth running

1. **#52190** — `torch.compile` is **silently disabled** for Kimi-K3. Log says `torch.compile is turned on, but the model does not support it`, so we run with **zero post-grad fusion**, including `aiter::allreduce_fusion_kernel_1stage`. Hits launch-bound idle *and* collectives. 2/3 hunks apply; the failing one is a single line in `compilation.py`.
2. **#51437** — overlaps the shared all-reduce with the routed up-projection. Works at **any** message size, so unlike QuickReduce it reaches decode. 5/6 hunks apply.
3. **CCD pinning** — written, never measured, archived. Workers run **unpinned** (affinity `0-255`) with GPU0-3 threads observed on node1 cores. One L3 domain (32 MiB) per GPU. See `archive/kimik3-rocprof-and-cpu-pinning.sh.txt`.
4. **The 8 MiB custom-AR ceiling** — prefill all-reduces are ~117 MB so they *always* fall back to RCCL. That's where 16.93% of GPU busy lives.

## Next steps, ranked

Ordered by expected value. Items 1–5 are untested and each has direct evidence behind it.

**1. AITER / hipBLASLt tuned GEMM configs.** A single C52 run logs **45,250**
`[aiter] not found tuned config in /tmp/aiter_config` messages — every dense GEMM
falls back to hipBLASLt heuristic selection. Dense GEMM is **11.9% of e2e wall**.
Top missing shapes (M varies with batch, N/K fixed):
`M:935 N:6288 K:7168`, `M:935 N:3584 K:7168`, `M:7928 N:3072 K:512`, `M:640 N:6288 K:7168`.
Run offline tuning for these and ship the config with the image. No code risk, no
numerics change. *I dismissed this early on as "not a slow path" without testing it.*

**2. #52190 — torch.compile is disabled.** Log: `torch.compile is turned on, but
the model does not support it`. We run with **zero post-grad fusion**, including
`aiter::allreduce_fusion_kernel_1stage` and `fused_qk_rmsnorm`. Fewer kernels
attacks the launch-bound idle; the fused all-reduce attacks collectives. 2/3 hunks
apply, failing one is a single line in `compilation.py`.

**3. CCD pinning.** Written, archived, **never measured**. Workers run unpinned
(`0-255`) with GPU0-3 threads observed on node1 cores — cross-socket for every
host-side op. Plausibly upstream of the 71.9 s idle before `copyBuffer` and the
26.2 s of collective rank skew. One 32 MiB L3 domain per GPU.

**4. `CUDAGRAPH_MODE=FULL`** instead of `FULL_AND_PIECEWISE`. Untested. With
chunked prefill most steps are mixed and take the piecewise path; forcing FULL
would show whether graph coverage is the launch-rate problem. Risk: batches
outside the ladder fall back.

**5. `--api-server-count N`.** Kimi-K3 has **no `tokenizer.json`** — vLLM logs
"slow tokenizer" ×10 and uses `tokenizer_mode=kimi`. At ~66k input tok/s *every*
prompt is tokenized in full, including the ~94% that hit the prefix cache and are
never computed. Frontend work scales with raw input, not with what is computed.

**6. #51437** — overlaps the shared all-reduce with the routed up-projection.
Works at any message size, so unlike QuickReduce it reaches decode collectives
(304 s vs prefill's 34 s). 5/6 hunks apply.

**7. RCCL env tuning** — `NCCL_ALGO`, `NCCL_MIN_NCHANNELS`, `NCCL_PROTO`. Never
touched. ~90% of collective volume is on RCCL.

**8. `gpu-memory-utilization 0.95`** — more KV headroom. Weak prior: KV usage is
only ~28% at C52, so the pool is not binding.

**9. Concurrency fill-in at 44/48/50.** We have 40 and 52 and nothing between,
and the peak sits somewhere in that gap.

**10. Raise the custom all-reduce `max_size`** (currently `8192*1024` = 8 MiB).
Prefill all-reduces are ~117 MB and always fall back to RCCL.

## Traps — I hit each of these

- **Orphaned variables.** `EP_ARGS` ignored for 55 trials; the KV-offload block deleted in a cleanup (cost 3.3×); MTP wired but never fired. The orphan-check catches *unused* arrays, not arrays that are used but always empty.
- **`SPEC_DECODING` is not forwarded** into the container on this runner. It arrives only inside `RESULT_FILENAME` (`..._spec-mtp_...`). **`DCP_SIZE` is also not forwarded** — confirmed by `[dcp] size=1 source=conc-fallback`. Do **not** make the matrix authoritative for DCP or C52 silently drops to DCP=1.
- **`spec-decoding: mtp` must stay on every matrix row** — the runner uses it to pick the script filename (`_mtp` suffix). Only `kimik3_fp4_mi355x_mtp.sh` exists. Whether MTP *runs* is decided by the `CONC ≤ 4` gate.
- **Capture ladder must be dense** `1…mns×(k+1)`. Sparse ladders pad decode batches and the padded rows read out of bounds — that was ~20 trials of GPU page faults.
- **rocprofv3 inflates idle.** It intercepts every dispatch, costs ~17–18% throughput. Traced numbers are not quotable; only ratios transfer.
- **Check the launch region after editing the script.** A block once wrapped `wait_for_server_ready` and deleting it took that line too. `bash -n` passes on a script that never starts the server.

## Corrections I made — don't repeat them

- **`pr51705-rejects` was a no-op.** Zero `.rej` files on this base; the parameter lands from the PR itself. Removed and archived. My `run_gemm_rs` rename claim was carried from an older base without rechecking.
- **The DCP all-reduce PR was wrong.** `get_dcp_group()` only calls `all_gather`, never `all_reduce` — widening an all-reduce backend gate does nothing. Branch deleted before opening. The log showing `dcp:0 -> ['PYNCCL']` tells you which backends are *available*, not which are *used*.
- **`mns` 65 vs 80.** I replaced the validated `2×CONC` with `1.25×` on a theory; it bound at C52 (T128 hit maxRun=65). Currently back at `1.25×` deliberately, to test the no-offload survivor.
- **SA vs us at C1 — RESOLVED, and it was a client metrics artifact.** I chased k=2, then different machines, then slower step time. All wrong. Diffing two SA C1 runs of the *same* config ([33062469329](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/33062469329) vs [33083417848](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/33083417848), read-only) shows the only env difference is `RUNNER_NAME: mi355x-amds_02` vs `_04` — and one of them records **`Decode Duration` min = 0.00 ms** and **ITL min = 0.00 ms**. That makes `1/tpot` explode: `intvty` p95 **3,465** and p99 **22,987** tok/s/user (= 0.043 ms/token, not physical), and mean output tok/s/user 1,280.95 with max 65,065. The clean node reports 100.24 / max 207.66. Aggregate throughput is identical across the two (1,236 vs 1,239 tok/s/GPU; decode 74.47 vs 74.57). **Any SA C1 latency percentile above p50 should be treated as suspect unless `Decode Duration` min is non-zero.**
- **"SA is ahead of us at C1 by 4–5%" is stale.** It compared against T106, before the DRAM offload came out. Like-for-like today (agentic, k=8 @ AL 4.00, DCP off, mns 8): **ours T133 = 1,237.2 tok/s/GPU, TPOT mean 7.18 ms, p50 8.69, p90 11.04** vs SA **1,236 / 1,239 tok/s/GPU, ITL p50 8.64 / 10.21**. We are at parity, marginally ahead.
- **SA does NOT run DCP at C1.** `Kimi-DCP-Experiemnts-Summary.md` says "SA has DCP on" — that is their **C52** arm. Their C1 runs `decode_context_parallel_size=1`. So DCP-at-C1 is not something SA is doing that we are missing.

## In flight

Nothing. T140 was the last dispatch.

Script gates, all default-off except MTP's k:
`EVAL_ONLY=0`, `QUICK_REDUCE_QUANTIZATION=NONE`, `ASYNC_SCHED=0`,
`MAX_BATCHED_TOKENS=8192`, `SPEC_NUM_TOKENS=8` (CONC ≤ 4) / `0` otherwise,
`mns = clamp(1.25×CONC, CONC if CONC≤4 else 8, 80)`, `--result-dir /workspace/`.

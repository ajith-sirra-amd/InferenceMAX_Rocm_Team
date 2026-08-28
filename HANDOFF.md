# Kimi-K3 on 8× MI355X — handoff

Last updated 2026-08-28. Target **12,500 tok/s/GPU**.

## Best results

| point | metric | run |
|---|---|---|
| **C52 throughput** | **7,950.6 tok/s/GPU** | [T103](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/32855763638) — DCP=8, mns 80, DRAM offload, full graphs, no MTP |
| **C1 interactivity** | **6.70 ms TPOT · 149.31 tok/s/user** | [T123](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/33029724147) — DCP off, MTP k=8 @ golden AL 4.00, mns 8 |
| reference | 8,296.0 tok/s/GPU | SA C52, no offload — **fastest on record, but that config crashes on our runner** |

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

## Settled — do not re-litigate

| thing | result |
|---|---|
| DRAM KV offload | Removing it cut idle **44.3% → 28.2%**. But `mns` 80 **without** it died 3/3 (`HSA_STATUS_ERROR_OUT_OF_RESOURCES`) — T129 + both SA C16/C52. Every completed `mns` 80 run had the offload. |
| **async scheduling** | **T108: 7,222.3 vs 7,950.6 = −9.2%.** Already tested on the **no-offload** path (`kv=none`, mns 80). Not stale. |
| QuickReduce FP | **−8.39%**. Accuracy fine. It *preempts* `AITER_CUSTOM` in dispatch order rather than supplementing it. |
| FP16 GEMM | Loses on **6 of 8** real shapes, up to −11.4%. BF16 already runs at full matrix-core rate (2.04× vs FP8 = the hardware ratio). |
| AMD `Kimi-K3-Quark-MXFP4-AttnFP8` | Loads and serves, but needs **240.75 GiB vs 192.63** → KV 9.38 GiB vs 59.81. Dropped. |
| MTP at high concurrency | −85% at C40. **Only** enable at CONC ≤ 4. |
| chunk size | prefill knob; 16384 measured −2.5%. Moves TTFT, not TPOT. |
| EP=8 | −4.7% |
| C96 | 4,667.9, TTFT **122 s** (p95 414 s). Past the knee; offload doesn't rescue it. |

## Open — worth running

1. **#52190** — `torch.compile` is **silently disabled** for Kimi-K3. Log says `torch.compile is turned on, but the model does not support it`, so we run with **zero post-grad fusion**, including `aiter::allreduce_fusion_kernel_1stage`. Hits launch-bound idle *and* collectives. 2/3 hunks apply; the failing one is a single line in `compilation.py`.
2. **#51437** — overlaps the shared all-reduce with the routed up-projection. Works at **any** message size, so unlike QuickReduce it reaches decode. 5/6 hunks apply.
3. **CCD pinning** — written, never measured, archived. Workers run **unpinned** (affinity `0-255`) with GPU0-3 threads observed on node1 cores. One L3 domain (32 MiB) per GPU. See `archive/kimik3-rocprof-and-cpu-pinning.sh.txt`.
4. **The 8 MiB custom-AR ceiling** — prefill all-reduces are ~117 MB so they *always* fall back to RCCL. That's where 16.93% of GPU busy lives.

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
- **SA vs us at C1**: same k, same AL 4.00, same acceptance 37.5%, same backends, same machines — engine generation rate differs only ~4% (87.7 vs 90.9 tok/s) yet client TPOT differs 1.56×. **Unresolved; likely client-side.** I wrongly guessed k=2, then different machines, then slower step time.

## In flight

Run [33130489071](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/33130489071) — C1 + C52. C52 is **mns 65 / no offload**, the one untested combination that might reach 8,296 without the crash. Matrix currently points at `kimi-k3-vllm:no-rejects` (2-patch image) for C52.

Script gates, all default-off except MTP's k:
`EVAL_ONLY=0`, `QUICK_REDUCE_QUANTIZATION=NONE`, `ASYNC_SCHED=0`, `MAX_BATCHED_TOKENS=8192`, `SPEC_NUM_TOKENS=8`, `mns = clamp(1.25×CONC, 8, 80)`.

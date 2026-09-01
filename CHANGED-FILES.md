# CHANGED FILES — Kimi-K3 MI355X perf work

Every file created, modified or deleted on branch `chore/sa-agentx-v1.0` during
the Kimi-K3 FP4 / MI355X performance effort (2026-08-24 → 2026-09-01).
Nothing outside this list was touched.

Reproduce this list:

```bash
git log --name-only --format='' --since='2026-08-24' | sort -u | grep -v '^$'
```

## Live files — these affect what a run actually does

| file | status | bytes | what it is |
|---|---|--:|---|
| `upstream/InferenceX/benchmarks/single_node/agentic/kimik3_fp4_mi355x_mtp.sh` | **modified** | 30,586 | **The launcher.** All benchmark behaviour lives here: K3 overlay apply, PR-stack apply, baked-image short-circuit, DCP env gating, mns/ladder invariant, chunk, gmu, `TEST=` fixed-len branch, `EVAL_ONLY=` GSM8K branch, `PROFILE=` branch. |
| `upstream/InferenceX/configs/amd-master.yaml` | **modified** | 75,046 | Kimi block only: `image:` tag and the `conc-list` / `dcp-size` / `kv-offloading` search-space row. |
| `runners/launch_mi355x-amd.sh` | **modified** | 3,687 | Local-image support: tolerate a failed `docker pull` when the tag is present locally, fall back from `.RepoDigests` to `.Id`, and switch `--pull always` → `--pull never` in that case. Registry images behave exactly as before. Needed because a locally built tag has no `RepoDigests` and the old `{{index .RepoDigests 0}}` aborted with *"index out of range"*. |
| `upstream/InferenceX/benchmarks/single_node/agentic/kimik3_fp4_mi355x_mtp.sa.sh` | created | 5,266 | Clean-room reference config (156 lines). **Not invoked by the runner** — the harness resolves `kimik3_fp4_mi355x_mtp.sh`. |
| `upstream/InferenceX/benchmarks/single_node/agentic/apply_kimi_k3_patches.sh` | modified | 9,135 | Legacy aigmkt-era in-container patcher. **Dead on the current stack** — the launcher exports `SKIP_KIMI_PATCHES=1` whenever the overlay lands. |

## Patch payloads — `k3_patches/`

Under `upstream/InferenceX/benchmarks/single_node/agentic/k3_patches/`.

| file | status | bytes | |
|---|---|--:|---|
| `vllm_nightly_46638857_k3_c16_c52_current.patch` | created | 264,116 | Hyukjoon's K3 overlay. sha256 `90f975fa…f64dcc0`. **The one behind every number in the ledger**, now used at all concurrencies. |
| `vllm_nightly_46638857_k3_c1_current.patch` | created | 232,005 | Older C1 cut. sha256 `554ec638…749a4e6d`. **Retired** — kept for reference only. |
| `pr_stack/vllm___aiter_ops.py.patch` | created | 2,077 | #53940 a4w4 flydsl |
| `pr_stack/vllm__envs.py.patch` | created | 1,058 | #53940 |
| `pr_stack/vllm__model_executor__layers__fused_moe__experts__rocm_aiter_moe.py.patch` | created | 1,221 | #53940 |
| `pr_stack/vllm__model_executor__layers__fused_moe__oracle__mxfp4.py.patch` | created | 880 | #53940 |
| `pr_stack/vllm__model_executor__layers__quantization__quark__quark_moe.py.patch` | created | 5,437 | #50813 SiTUv2 A8W4 MoE |
| `pr_stack/vllm__v1__worker__gpu__cudagraph_utils.py.patch` | **deleted** | — | #54095. Hunk #2 never applied on 46638857 (fails at line 362). Removed as dead weight. |
| `vllm_pr_53940_54095_on_46638857.patch` | **deleted** | — | The monolithic version. Replaced by the per-file `pr_stack/` because `patch(1)` is all-or-nothing per invocation, so one stale hunk vetoed all five files. |

## Dockerfiles — `k3_patches/`

| file | status | bytes | |
|---|---|--:|---|
| `Dockerfile.kimi-k3-vllm.v4` | created | 2,992 | **Current.** base nightly + overlay + full pr-stack, all baked; writes `/etc/k3-image-manifest`; one overlay for all concurrencies; no runtime patching. |
| `Dockerfile.kimi-k3-vllm.v3` | created | 1,952 | overlay + #53940 + #50813 |
| `Dockerfile.kimi-k3-vllm.v2` | created | 1,592 | overlay + #53940 |
| `Dockerfile.kimi-k3-current` | created | 553 | overlay only |
| `Dockerfile.kimi-k3-vllm-v2` | **deleted** | — | renamed to the dotted form |

Images are built **locally only** — nothing was pushed to Docker Hub.

## Vendored diffs (agentic/) — superseded

| file | status | bytes | |
|---|---|--:|---|
| `pr51705_nightly.diff` | created | 137,591 | #51705 DCP-for-DSpark, rebased to nightly |
| `pr51705_vllm.diff` | created | 170,216 | #51705 @ `e72380a5` for the aigmkt base |
| `pr51040_vllm.diff` | created | 9,222 | fp8 MLA prefill |
| `pr51171_vllm.diff` | **deleted** | — | moved to `archive/` |

**All obsolete.** #51705 and #52707 are merged into `nightly-46638857` — verified
in the image (`dcp.py`, `dcp_comm_backend`, `spec_decode/dspark/speculator.py`
present unpatched).

## Documentation (repo root)

| file | status | bytes | |
|---|---|--:|---|
| `Kimi-DCP-Experiemnts-Summary.md` | **modified** | 155,475 | The trial ledger. Every run T1xx–T2xx with config, result, and the reasoning — including retractions. |
| `EXPERIMENT-QUEUE.md` | created | 25,360 | Live state: what is running, what is queued, which levers are closed. |
| `IMAGE-RECIPE.md` | created | 12,942 | How to reproduce 10,632 — base image, both patch layers, gate lines, peak config, v4. |
| `CHANGED-FILES.md` | created | — | This file. |
| `Kimi-K3-Where-The-Time-Goes.md` | created | 28,136 | Time-breakdown analysis |
| `HANDOFF.md` | created | 15,928 | Context handoff |
| `DCP-TRIALS-LOG.md` | created | 1,563 | Early DCP log |
| `kimi-k3-profiles/README.md` | created | 4,133 | Profile notes |
| `kimi-k3-profiles/T116_rocprofv3_k_kernel_stats.csv` | created | 166,397 | aigmkt-era rocprof kernel stats |
| `kimi-k3-profiles/T124_rocprofv3_k_kernel_stats_no-offload.csv` | created | 170,572 | same, offload off |

## Archive (agentic/archive/) — reference only, nothing reads these

`apply_kimi_k3_patches.configurable.sh` (44,636) ·
`kimik3_fp4_mi355x_mtp.configurable.sh` (69,744) ·
`apply_kimi_k3_patches.with-aiter-pybind11-and-triton-cudagraph.sh` (19,214) ·
`pr51171_vllm.diff` (13,589) · `README.md` (3,485) · `README-patches.md` (4,736) ·
`kimik3-rocprof-and-cpu-pinning.sh.txt` (5,078) ·
`patch_dcp_aiter_allreduce.sh.txt` (3,382) ·
`patch_pr51705_rejects.sh.txt` (3,880) ·
`patch_pr51040-fp8-mla-prefill.sh.txt` (1,445)

Plus `apply_k3_container_patches.sh` (14,897, created, in `agentic/`).

## What was NOT touched

- `SemiAnalysisAI/InferenceX` — read-only throughout
- Docker Hub — no push, ever
- Any repo other than `ajith-sirra-amd/InferenceMAX_Rocm_Team`
- Any benchmark, config or runner outside the kimi path, with the single
  exception of `runners/launch_mi355x-amd.sh` above, which is shared and was
  changed only after explicit approval

## If you want to review just the substance

Two files hold essentially all the behaviour:

```bash
git log -p --since='2026-08-24' -- \
  upstream/InferenceX/benchmarks/single_node/agentic/kimik3_fp4_mi355x_mtp.sh \
  runners/launch_mi355x-amd.sh
```

The rest is patch payloads, Dockerfiles, and documentation.

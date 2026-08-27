# Reproducing `aigmkt/kimi-k3-vllm:latest`

```
kimi.base    : nightly-f94666b6
kimi.patches : kv-blockpool, pr51705(e72380a5), pr51705-rejects
```

## Recipe

| # | piece | upstream | needed? |
|---|---|---|---|
| 1 | base image | `vllm/vllm-openai-rocm:nightly-f94666b60d4c58ec0807d22c837cfae322a1dde9` | yes |
| 2 | `kv-blockpool` | [#52707](https://github.com/vllm-project/vllm/pull/52707) | yes |
| 3 | `pr51705` | [#51705](https://github.com/vllm-project/vllm/pull/51705) @ `e72380a5` | yes |
| 4 | **`pr51705-rejects`** | none — fixes #51705's rejected hunk | **yes** |

Base + #51705 + #52707 alone is **not** enough. See below.

## Where things are

Both under `upstream/InferenceX/benchmarks/single_node/agentic/`:

| file | |
|---|---|
| patch script | [`apply_kimi_k3_patches.sh`](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/blob/chore/sa-agentx-v1.0/upstream/InferenceX/benchmarks/single_node/agentic/apply_kimi_k3_patches.sh) |
| vendored #51705 diff (170 KB) | [`pr51705_vllm.diff`](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/blob/chore/sa-agentx-v1.0/upstream/InferenceX/benchmarks/single_node/agentic/pr51705_vllm.diff) |

Inside the patch script:

| function | defined | invoked |
|---|---:|---:|
| `patch_kv_blockpool` | [L60](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/blob/chore/sa-agentx-v1.0/upstream/InferenceX/benchmarks/single_node/agentic/apply_kimi_k3_patches.sh#L60) | [L246](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/blob/chore/sa-agentx-v1.0/upstream/InferenceX/benchmarks/single_node/agentic/apply_kimi_k3_patches.sh#L246) |
| `patch_pr51705` | [L156](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/blob/chore/sa-agentx-v1.0/upstream/InferenceX/benchmarks/single_node/agentic/apply_kimi_k3_patches.sh#L156) | [L291](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/blob/chore/sa-agentx-v1.0/upstream/InferenceX/benchmarks/single_node/agentic/apply_kimi_k3_patches.sh#L291) |
| **`patch_pr51705_rejects`** | [**L196**](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/blob/chore/sa-agentx-v1.0/upstream/InferenceX/benchmarks/single_node/agentic/apply_kimi_k3_patches.sh#L196) | [**L297**](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/blob/chore/sa-agentx-v1.0/upstream/InferenceX/benchmarks/single_node/agentic/apply_kimi_k3_patches.sh#L297) |
| `patch_dcp_aiter_allreduce` * | [L252](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/blob/chore/sa-agentx-v1.0/upstream/InferenceX/benchmarks/single_node/agentic/apply_kimi_k3_patches.sh#L252) | [L303](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/blob/chore/sa-agentx-v1.0/upstream/InferenceX/benchmarks/single_node/agentic/apply_kimi_k3_patches.sh#L303) |

\* not part of the image — untested local patch for the `dcp:0` -> PYNCCL gate.

## Why `-rejects` is required

#51705's diff does not apply cleanly to `f94666b6`. Hunk 4 of
`vllm/models/kimi_k3/nvidia/mla.py` rejects: the PR's context line is
`run_gemm_rs_ar`, this nightly has `run_gemm_rs` (renamed upstream).

The failure is asymmetric, which is what makes it dangerous:

- the hunk that **uses** `enable_dcp_q_replicate` applies
- the hunk that **adds it to the signature** rejects

So `MultiHeadLatentAttention` references a parameter its `__init__` never gained.
Harmless with spec decoding off — nothing instantiates it — but `dspark_mla.py`
passes the kwarg, so **DCP + MTP dies at init**:

```
TypeError: MultiHeadLatentAttention.__init__() got an unexpected
           keyword argument 'enable_dcp_q_replicate'
```

That killed **T78**.

`patch_pr51705_rejects` inserts `enable_dcp_q_replicate: bool = True` into the
signature, matching the PR's placement.

## Usage

```bash
bash upstream/InferenceX/benchmarks/single_node/agentic/apply_kimi_k3_patches.sh
```

Marker-guarded and idempotent — safe to re-run. Not called from
`kimik3_fp4_mi355x_mtp.sh`, because the image ships the patches pre-applied.

## Caveats

- **Pin the revision.** We vendored `e72380a5`. #51705 is open and mutating — a
  reviewer asked to retire `VLLM_ALLOW_DCP_FULL_CUDAGRAPH` and the author agreed.
  Today's PR head is not the same as this image.
- **`-rejects` is base-specific.** It exists only because #51705 is unmerged and
  its diff is stale against this nightly. Once #51705 merges, it is unnecessary.
- **Once #51705 and #52707 both merge**, a stock nightly containing them is
  equivalent and this image can be retired. Budget one validation run at C52:
  a later nightly is hundreds of unrelated commits ahead, and DCP is the path
  that produced ~20 trials of GPU page faults when its base last shifted.

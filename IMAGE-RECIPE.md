# Reproducing `aigmkt/kimi-k3-vllm:latest`

```
kimi.base    : nightly-f94666b6
kimi.patches : kv-blockpool, pr51705(e72380a5), pr51705-rejects
```

## Recipe

| # | piece | source |
|---|---|---|
| 1 | base image | `vllm/vllm-openai-rocm:nightly-f94666b60d4c58ec0807d22c837cfae322a1dde9` |
| 2 | `kv-blockpool` | [#52707](https://github.com/vllm-project/vllm/pull/52707) |
| 3 | `pr51705` | [#51705](https://github.com/vllm-project/vllm/pull/51705) @ `e72380a5` |
| 4 | **`pr51705-rejects`** | no upstream PR — fixes #51705's rejected hunk |

**Base + #51705 + #52707 is not sufficient. Item 4 is required.**

## Build

```bash
docker run --rm -v "$PWD/upstream/InferenceX/benchmarks/single_node/agentic":/w \
  vllm/vllm-openai-rocm:nightly-f94666b60d4c58ec0807d22c837cfae322a1dde9 \
  bash -c 'bash /w/apply_kimi_k3_patches.sh'
```

Then commit the container. The script is marker-guarded and idempotent — safe to
re-run. Expected output:

```
[kimi-patches] applying in-container patches...
[kv-blockpool] patched .../v1/core/single_type_kv_cache_manager.py
[pr51705] applying vendored diff (c326df26b4eb4caa, 3830 lines)
[pr51705] applied
[pr51705-rejects] MultiHeadLatentAttention.__init__ accepts enable_dcp_q_replicate
[kimi-patches] done.
```

Anything reporting `target not found` or `skipping` means that patch did **not**
apply — do not ship the image.

`kimik3_fp4_mi355x_mtp.sh` does not call this script; the image ships the patches
pre-applied.

## Where things are

Both under `upstream/InferenceX/benchmarks/single_node/agentic/`:

| file | |
|---|---|
| patch script | [`apply_kimi_k3_patches.sh`](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/blob/chore/sa-agentx-v1.0/upstream/InferenceX/benchmarks/single_node/agentic/apply_kimi_k3_patches.sh) |
| vendored #51705 diff (170 KB) | [`pr51705_vllm.diff`](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/blob/chore/sa-agentx-v1.0/upstream/InferenceX/benchmarks/single_node/agentic/pr51705_vllm.diff) |

Inside the patch script:

| function | defined | invoked |
|---|---:|---:|
| `patch_kv_blockpool` | [L39](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/blob/chore/sa-agentx-v1.0/upstream/InferenceX/benchmarks/single_node/agentic/apply_kimi_k3_patches.sh#L39) | [L217](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/blob/chore/sa-agentx-v1.0/upstream/InferenceX/benchmarks/single_node/agentic/apply_kimi_k3_patches.sh#L217) |
| `patch_pr51705` | [L142](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/blob/chore/sa-agentx-v1.0/upstream/InferenceX/benchmarks/single_node/agentic/apply_kimi_k3_patches.sh#L142) | [L262](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/blob/chore/sa-agentx-v1.0/upstream/InferenceX/benchmarks/single_node/agentic/apply_kimi_k3_patches.sh#L262) |
| **`patch_pr51705_rejects`** | [**L164**](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/blob/chore/sa-agentx-v1.0/upstream/InferenceX/benchmarks/single_node/agentic/apply_kimi_k3_patches.sh#L164) | [**L265**](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/blob/chore/sa-agentx-v1.0/upstream/InferenceX/benchmarks/single_node/agentic/apply_kimi_k3_patches.sh#L265) |

Those three are the whole script — running it reproduces the reference image
exactly.

## Why `pr51705-rejects` is required

#51705's diff does not apply cleanly to `f94666b6`. Hunk 4 of
`vllm/models/kimi_k3/nvidia/mla.py` rejects — the PR's context line is
`run_gemm_rs_ar`, this nightly has `run_gemm_rs`.

The failure is asymmetric:

- the hunk that **uses** `enable_dcp_q_replicate` applies
- the hunk that **adds it to the signature** rejects

`MultiHeadLatentAttention` then references a parameter its `__init__` never
gained. Invisible with spec decoding off — nothing instantiates it — but
`dspark_mla.py` passes the kwarg, so **DCP + MTP fails at init**:

```
TypeError: MultiHeadLatentAttention.__init__() got an unexpected
           keyword argument 'enable_dcp_q_replicate'
```

`patch_pr51705_rejects` inserts `enable_dcp_q_replicate: bool = True` into the
signature.

## Caveats

- **Pin `e72380a5`.** #51705 is open and changing — a reviewer asked to retire
  `VLLM_ALLOW_DCP_FULL_CUDAGRAPH` and the author agreed. Today's PR head will not
  reproduce this image.
- **`pr51705-rejects` is base-specific**, needed only because #51705 is unmerged
  and its diff is stale against this nightly. It becomes unnecessary once #51705
  merges.
- **Once #51705 and #52707 both merge**, a stock nightly containing them is
  equivalent and this image can be retired. Budget one validation run at
  concurrency 52 first: a later nightly carries hundreds of unrelated commits,
  and DCP is the path most sensitive to its base moving.

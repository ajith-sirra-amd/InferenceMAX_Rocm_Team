# Reproducing `aigmkt/kimi-k3-vllm:latest`

## Recipe

| # | piece | source |
|---|---|---|
| 1 | base image | `vllm/vllm-openai-rocm:nightly-f94666b60d4c58ec0807d22c837cfae322a1dde9` |
| 2 | `kv-blockpool` | [#52707](https://github.com/vllm-project/vllm/pull/52707) — prevents negative external block allocation |
| 3 | `pr51705` | [#51705](https://github.com/vllm-project/vllm/pull/51705) @ `e72380a5` — DCP support for Kimi-K3 DSpark |

**Base + #52707 + #51705 is sufficient.** 
> Note : DCP cannot run without #51705.

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

Those three are the whole script — running it reproduces the reference image
exactly.

## Caveats

- **Pin `e72380a5`.** #51705 is open and changing — a reviewer asked to retire
  `VLLM_ALLOW_DCP_FULL_CUDAGRAPH` and the author agreed. Today's PR head will not
  reproduce this image.
- **Check `triton_mla.py` after any rebase.** The vendored diff raises
  `TritonMLAMetadataBuilder._cudagraph_support` from
  `UNIFORM_SINGLE_TOKEN_DECODE` to `UNIFORM_BATCH`. The DSpark draft can only
  run on TRITON_MLA — it is the sole ROCm MLA backend declaring
  `supports_non_causal_multi_token_decode` — and cudagraph capability is the
  **minimum across attention groups**, so without this line the draft demotes
  the whole engine `FULL_AND_PIECEWISE` → `PIECEWISE` and the drafter runs
  eager. Measured cost of losing it: **14.05 → 77.65 tok/s, ITL 71.16 → 12.88
  ms (5.52×)**. Confirm `Capturing model for DSpark speculator...` appears in
  the server log; if it is missing, this hunk did not land.
- **Once #51705 and #52707 both merge**, a stock nightly containing them is
  equivalent and this image can be retired. Budget one validation run at
  concurrency 52 first: a later nightly carries hundreds of unrelated commits,
  and DCP is the path most sensitive to its base moving.

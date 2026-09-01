# IMAGE RECIPE — Kimi-K3 FP4 on MI355X

How to reproduce the **10,632 tok/s/GPU** C72 result (T195/T198, n=2, 0.02%
spread) and the **9.70 ms** C1 TPOT (T200).

> Supersedes the `aigmkt/kimi-k3-vllm:latest` recipe, kept at the bottom for
> history. That image is no longer used by any current result.

## The short version

There is **no custom image.** Every number in the ledger — including the peak —
runs on the stock upstream nightly with two patch layers applied *inside the
container at launch*:

```
vllm/vllm-openai-rocm:nightly-46638857fdbb30e0c232c9e8f9cb1ff6d6f545c3
  + [k3-overlay]  Hyukjoon's 264 KB K3 overlay -> site-packages
  + [pr-stack]    5 per-file upstream PR hunks -> site-packages
```

Nothing is baked, nothing is pushed to a registry. That is deliberate: any node
can reproduce the number by pulling a public tag, and it satisfies the
no-Docker-Hub-push constraint. The local `kimi-k3-vllm:v2` / `:v3` images exist
but **no ledger result comes from them.**

Verified from the T195 job log (run 33418755100, job 99575730290):

```
IMAGE: vllm/vllm-openai-rocm:nightly-46638857fdbb30e0c232c9e8f9cb1ff6d6f545c3
Status: Image is up to date for vllm/vllm-openai-rocm:nightly-46638857...
[k3-overlay] applied=1 conc=72
[pr-stack] applied=5 files, skipped: none (#53940 a4w4-flydsl, #50813 SiTUv2-A8W4-MoE)
```

## Layer 0 — base image

| | |
|---|---|
| tag | `vllm/vllm-openai-rocm:nightly-46638857fdbb30e0c232c9e8f9cb1ff6d6f545c3` |
| digest | `sha256:8908b8ab5ba28c3b81f9f42bb72e2421f06a180e001c67c4f10ff7f127c5690b` |
| site-packages | `/usr/local/lib/python3.12/dist-packages` |
| set in | `upstream/InferenceX/configs/amd-master.yaml`, kimi block `image:` |

Immutable digest-pinned tag, so no drift. This replaced
`aigmkt/k3-unified-v2-from-cb810:validation-20260815-opusfix1`; that swap is
what moved the headline from 8,342 to 10,632, and **every** aigmkt-era
top-of-curve conclusion (C64 cliff, C72 death, mns-96-kills-engine, mutable-tag
drift) was falsified on this base.

## Layer 1 — K3 overlay (MANDATORY, fatal if it fails)

`upstream/InferenceX/benchmarks/single_node/agentic/k3_patches/`

| patch | bytes | sha256 | used when |
|---|--:|---|---|
| `vllm_nightly_46638857_k3_c16_c52_current.patch` | 264,116 | `90f975fa…f64dcc0` | CONC > 4 |
| `vllm_nightly_46638857_k3_c1_current.patch` | 232,005 | `554ec638…749a4e6d` | CONC ≤ 4 |

Hyukjoon's overlay, applied verbatim. It touches
`models/kimi_k3/amd/{mla,kda,linear,latent_moe_runner}.py`, `layers/mla.py`,
`mla_attention.py`, `fused_moe/runner/*`, `platforms/rocm.py`, `envs.py`.

**Without it the nightly is missing every Kimi-K3 kernel path and regresses.**
It is cut against `46638857` specifically — on any other image the dry-run
fails. `REQUIRE_K3_OVERLAY` defaults to `1` so that case is a hard exit rather
than a silent unpatched run producing a misleading number.

Applied by `kimik3_fp4_mi355x_mtp.sh` (~L50) as dry-run → apply →
`[k3-overlay] applied=N conc=C`. On success it also exports
`SKIP_KIMI_PATCHES=1`: the legacy `apply_kimi_k3_patches.sh` edits files the
overlay also carries, and running it afterwards shifts context and silently
breaks the overlay.

## Layer 2 — upstream PR stack (non-fatal, per-file)

`k3_patches/pr_stack/` — all pure Python, no `csrc/`, so `patch -p1` into
site-packages is sufficient. No rebuild, no push.

| file | bytes | PR |
|---|--:|---|
| `vllm___aiter_ops.py.patch` | 2,077 | #53940 |
| `vllm__envs.py.patch` | 1,058 | #53940 |
| `vllm__model_executor__layers__fused_moe__experts__rocm_aiter_moe.py.patch` | 1,221 | #53940 |
| `vllm__model_executor__layers__fused_moe__oracle__mxfp4.py.patch` | 880 | #53940 |
| `vllm__model_executor__layers__quantization__quark__quark_moe.py.patch` | 5,437 | #50813 |

- **#53940** a4w4 flydsl kernels for Kimi-K3 — the AMD MoE path.
- **#50813** opt-in K3 SiTUv2 A8W4 routed MoE. The launcher already exported
  `VLLM_ROCM_USE_AITER_MOE_SITUV2_A8W4=1` and `AITER_SITUV2_A8W4=1`, i.e. we
  were setting flags for a code path that was not present.

**Per-file, not one monolithic patch.** T187 proved why: `cudagraph_utils.py`
Hunk #2 failed at line 362 and `patch` is all-or-nothing per invocation, so that
one hunk vetoed all five files (`applied=0`). Splitting means a stale hunk costs
only its own file.

**Non-fatal by design**, unlike layer 1: a PR that stops applying degrades to
the overlay-only baseline instead of killing the run. The `[pr-stack]` gate line
records which way it went, so a number is never silently mis-attributed.

### Deliberately excluded

| PR | why |
|---|---|
| #54095 aiter per-stream workspace | `cudagraph_utils.py` hunk 2 is cut against a newer tree, FAILS at line 362 on 46638857 (T187/T188/T190). Removed as dead weight. |
| #53154, #37682 | edit files the K3 overlay rewrites (`amd/mla.py`, `layers/mla.py`, `rocm_aiter_mla.py`) — will not apply on top of it |
| #50647, #54255 | NVIDIA path |

## Numerics validation

GSM8K on this exact stack: **0.995** with the 4-file pr-stack, **0.99** with all
5. Any numerics-affecting change re-runs GSM8K at limit 200 before it is allowed
into a throughput run.

## Gate lines to verify on every run

A run is only trustworthy if these appear in the log:

```
[k3-overlay] applied=1 conc=<C>
[pr-stack] applied=5 files, skipped: none
[dcp] ENABLED size=8 backend=a2a interleave=1      # C>4; "DISABLED size=1" at C<=4
[mns] max_num_seqs=<N> conc=<C> offload=dram
[chunk] max_num_batched_tokens=<N> conc=<C>
graphs: dense ladder 1..<mns x SPEC_ROWS>
```

`applied=0` on either layer means the result is **not** comparable to the
ledger. The ladder must always cover `mns × SPEC_ROWS` — a capture smaller than
mns is the signature that precedes `HSA_STATUS_ERROR_OUT_OF_RESOURCES`.

## Config at the peak (T195, C72)

| knob | value | evidence |
|---|---|---|
| conc | 72 | curve 52/60/64/72/80 → peak at 72 |
| max-num-seqs | 80 | 96 measured identical (T198, 0.02%) |
| max-num-batched-tokens | 16384 | 8192 measured identical (T199, 0.07%) |
| dcp | 8, a2a, interleave 1 | |
| kv-offloading | dram, vllm-simple | three independent A/Bs favour it |
| gpu-memory-utilization | 0.9 | |
| load-format | fastsafetensors | |
| spec | none at C≥48; MTP only at CONC ∈ {1,2,4} | |

Concurrency, mns and chunk are all **closed as levers** — three independent
knobs, all flat at the operating point. The remaining 14.9% to 12,500 is not in
the launcher's argument space.

## Optional: bake it

`k3_patches/Dockerfile.kimi-k3-current` builds layer 0 + layer 1 as an image
(sha256-checked `COPY` + `patch -p1` into dist-packages).
`Dockerfile.kimi-k3-vllm.v2` and `.v3` are the local variants. **Build only — do
not push to Docker Hub.** The runtime path above is the supported one; the
Dockerfiles exist so a node without access to the patch files can still run.

---

# APPENDIX (SUPERSEDED) — reproducing `aigmkt/kimi-k3-vllm:latest`

Kept for history. This image backed the ledger up to T~180 and topped out at
**8,342 tok/s/GPU**. Not used by any current result.

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

## Where things are

Both under `upstream/InferenceX/benchmarks/single_node/agentic/`:

| file | |
|---|---|
| patch script | [`apply_kimi_k3_patches.sh`](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/blob/chore/sa-agentx-v1.0/upstream/InferenceX/benchmarks/single_node/agentic/apply_kimi_k3_patches.sh) |
| vendored #51705 diff (170 KB) | [`pr51705_vllm.diff`](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/blob/chore/sa-agentx-v1.0/upstream/InferenceX/benchmarks/single_node/agentic/pr51705_vllm.diff) |

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

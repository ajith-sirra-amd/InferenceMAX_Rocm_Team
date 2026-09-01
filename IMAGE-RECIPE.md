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

### What it is made of

Hyukjoon: *"The patch I made is a mixture of PRs."* It is **not** a single
upstream PR and has no PR number of its own. Matching hunks back to
`vllm-project/vllm` by symbol and title:

| overlay area | upstream PR | state |
|---|---|---|
| `v1/attention/ops/dcp.py` — `_DCPA2ABufferPool` | [#51705](https://github.com/vllm-project/vllm/pull/51705) [ROCm][MLA][DCP] Support causal multi-token verification | closed |
| KDA kernels — `ops/fused_qkv_conv1d.py`, `fused_sigmoid_gate.py`, `ops/third_party/kda/chunk.py`, `flash_linear_attention/ops/chunk_delta_h.py` | [#54038](https://github.com/vllm-project/vllm/pull/54038) [ROCm][Perf] Kimi-K3 Fused kernels for KDA prefill (reland) | open |
| `v1/attention/backends/mla/triton_mla.py` cudagraph support | [#54546](https://github.com/vllm-project/vllm/pull/54546) [ROCm][MLA][DCP] Advertise Triton MLA non-causal multi-token DCP | open |
| KV-offload / cache-manager group — `single_type_kv_cache_manager`, `simple_kv_offload/manager`, `kv_cache_coordinator`, `kv_cache_interface`, `block_pool` | [#53917](https://github.com/vllm-project/vllm/pull/53917) [Bugfix][DCP] Handle hybrid cache geometry in offload recovery · [#54457](https://github.com/vllm-project/vllm/pull/54457) [Bugfix] Do not adjust `dcp_kv_cache_interleave_size` for CPU offloading | open |
| decode-LSE correctness | [#54639](https://github.com/vllm-project/vllm/pull/54639) [AMD][DCP] Close two decode-LSE holes left by AITER DCP support | closed |

**Not matched to any public PR** — likely unpublished or squashed differently:
`models/kimi_k3/amd/latent_moe_runner.py`, `v1/attention/ops/rocm_aiter_mla_reduce.py`
(new file), and the `v1/core/sched/scheduler.py` hunks (+116/−65).

### If it were to be split for upstreaming

34 files / ~3,600 changed lines is not a reviewable PR. Natural split, easiest
first:

| # | scope | Δlines | note |
|---|---|--:|---|
| A | DCP a2a buffer pool (`dcp.py`) | +48/−10 | self-contained ROCm correctness fix; best first PR |
| B | spec-decode cudagraph (`dflash/cudagraph.py`, `speculator.py`, `config/speculative.py`) | +34/−11 | small |
| C | KV-offload + cache manager | +692/−170 | highest value (generic, not K3-specific), heaviest review |
| D | ROCm AITER MLA backend (`rocm_aiter_mla.py` +1077/−168, `rocm_aiter_mla_reduce.py` new) | +1241/−178 | ROCm-specific |
| E | Kimi-K3 AMD model path (`models/kimi_k3/amd/*`, MoE runner, `envs`, `platforms/rocm`) | +1550/−230 | depends on D |

Authorship is Hyukjoon's — nothing should be filed without their sign-off, and
the new-file hunks need re-checking against current upstream `main` since the
overlay is cut against a July nightly.

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

## `kimi-k3-vllm:v4` — everything baked, nothing at runtime

Built and verified on `b23_07`. **35.6 GB**, `docker build` exit 0, **0 failed
hunks**, all 5 pr-stack files applied.

```bash
cd upstream/InferenceX/benchmarks/single_node/agentic
docker build -f k3_patches/Dockerfile.kimi-k3-vllm.v4 -t kimi-k3-vllm:v4 .
```

Collapses both runtime layers into the image and **removes the C1-vs-C52
overlay split** — one overlay for every concurrency, the `c16_c52` cut, because
that is the one behind every throughput number in the ledger including 10,632.

It drops `/etc/k3-image-manifest`:

```
k3-image: kimi-k3-vllm:v4
base: vllm/vllm-openai-rocm:nightly-46638857fdbb30e0c232c9e8f9cb1ff6d6f545c3
base-digest: sha256:8908b8ab5ba28c3b81f9f42bb72e2421f06a180e001c67c4f10ff7f127c5690b
overlay: vllm_nightly_46638857_k3_c16_c52_current.patch (all concurrencies)
overlay-sha256: 90f975fad15722494366153ec3f32a14c4445bfa88c51ec53043b88eaf64dcc0
pr-stack: 5 files (#53940 a4w4-flydsl, #50813 SiTUv2-A8W4-MoE)
runtime-patching: none
```

`kimik3_fp4_mi355x_mtp.sh` short-circuits on that file: if it exists, the script
does not touch site-packages and logs

```
[k3-overlay] baked into image -- runtime patching SKIPPED
[pr-stack]   baked into image -- runtime patching SKIPPED
[k3-image] ...manifest...
```

The manifest is the contract — re-applying an already-applied overlay fails the
dry-run, and with `REQUIRE_K3_OVERLAY=1` that would kill the run.

Verified inside the image: `vllm 0.26.1rc1.dev1219+g46638857f`,
`kimi_k3/amd/latent_moe_runner.py` present, SiTUv2 markers in `quark_moe.py`,
flydsl markers in `_aiter_ops.py`.

### Two things to know before trusting a number from it

1. **C1 numerics are not yet validated.** Folding C1 onto the `c16_c52` overlay
   is a real change: the `c1` cut carried an online-quantization / weight-loading
   subsystem (`layers/quantization/online/*`, `config/quantization.py`,
   `model_loader/{base_loader,weight_utils}.py`, `layers/linear.py`,
   `routed_experts.py`) that this image does not have. C72 runs the same fp4
   model without it, so it is not load-bearing — but **C1 must re-pass GSM8K
   (limit 200) on this image** before any C1 result from it counts.
2. **C1 also *gains* the newer kernels**: `latent_moe_runner.py`, the KDA chunk
   kernels, `moe_runner`/`shared_experts`, `envs.py`, and much newer
   `simple_kv_offload/manager.py` (164 → 469 lines) and
   `single_type_kv_cache_manager.py` (243 → 478). Those are exactly what a
   batch-1 decode step is bound by, so C1 TPOT may well improve.

### Blocked on one shared-file change

`runners/launch_mi355x-amd.sh` cannot launch a local-only tag as written:

| line | problem |
|---|---|
| 35 `docker pull $IMAGE` | fails — `kimi-k3-vllm:v4` is in no registry |
| 36 `{{index .RepoDigests 0}}` | aborts *"index out of range"* — local images have no `RepoDigests` |
| 59 `--pull always` | re-pulls and fails again |

The fix is small and backwards compatible (tolerate a failed pull when the image
is present locally, fall back to `.Id` for the digest, and switch to
`--pull never` in that case), but that file is outside the standing edit bounds,
so **it needs explicit approval before `image:` can point at `kimi-k3-vllm:v4`.**

## Optional: bake overlay only

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

# REPRODUCE — 10,607 tok/s/GPU on 8× MI355X, and the path to a mergeable version

Two things are asked for here, and today they are **not the same artifact**:

| goal | status |
|---|---|
| **A. Reproduce 10,607 tok/s/GPU** from nightly + patches | **done, deterministic** — recipe below |
| **B. Get there with mergeable PRs only** | **DONE (T232)** — `pronly` = 10,692 at C72 |

**Update 2026-09-02:** B is no longer blocked. `kimi-k3-vllm:pronly` — 12
upstream PRs on `nightly-7c5dc571`, no vendor patch — measured **10,692
tok/s/GPU** at C72 (err 0.18%, GSM8K 0.99, T232), statistically identical to v4.
Section B's gap analysis below is kept for the reasoning and the PR mechanics,
but its premise ("not yet possible") is superseded. Prefer `pronly` for anything
that must be reproducible outside this team.

Read both. A is the pinned-overlay recipe. B is the mergeable path, now measured.

---

# A. Exact reproduction

## A1. Pinned inputs

| | value |
|---|---|
| base image | `vllm/vllm-openai-rocm:nightly-46638857fdbb30e0c232c9e8f9cb1ff6d6f545c3` |
| base digest | `sha256:8908b8ab5ba28c3b81f9f42bb72e2421f06a180e001c67c4f10ff7f127c5690b` |
| overlay | `k3_patches/vllm_nightly_46638857_k3_c16_c52_current.patch` |
| overlay sha256 | `90f975fad15722494366153ec3f32a14c4445bfa88c51ec53043b88eaf64dcc0` |
| overlay size | 264,116 B · 199 hunks · 34 files |
| PR stack | `k3_patches/pr_stack/` — 4 files, PR #53940 (a4w4 flydsl MoE) |
| model | `moonshotai/Kimi-K3` (FP4), TP8, EP1 |

The base digest is pinned deliberately: the `nightly-*` tag is mutable, and the
overlay is cut against this exact commit. It applies to **no other base**.

## A2. Build the image

From the `agentic/` directory (not from `k3_patches/`):

```bash
docker build -f k3_patches/Dockerfile.kimi-k3-vllm.v4 -t kimi-k3-vllm:v4 .
```

The Dockerfile verifies the overlay sha256 before applying, treats **both**
layers as fatal on failure, and writes `/etc/k3-image-manifest`. That marker is
the contract: if it exists, the image is fully patched and the launcher must not
touch site-packages. No runtime patching, one overlay for every concurrency.

Prebuilt and pushed: `aigmkt/kimi-k3-vllm:v4`, digest
`sha256:88c8438f5aa0fc2fa01ee1736eb0f8a88e478b26a93a733f535b4f964bb197f2`.

## A3. Server configuration — the C72 operating point

```
--tensor-parallel-size 8
--decode-context-parallel-size 8 --dcp-comm-backend a2a --cp-kv-cache-interleave-size 1
--max-num-seqs 80
--max-num-batched-tokens 16384
--gpu-memory-utilization 0.9
--max-model-len 1048576
--kv-cache-dtype fp8
--attention-backend ROCM_AITER_MLA
--attention-config '{"mla_prefill_backend":"ROCM_AITER_FA"}'
--moe-backend auto
--load-format fastsafetensors
--language-model-only
--enable-prefix-caching
--no-async-scheduling
--kv-transfer-config  SimpleCPUOffloadConnector, kv_both, lazy_offload=false
--compilation-config  mode 3, FULL_AND_PIECEWISE, cudagraph_capture_sizes = 1..80 dense
```

No speculative decoding at C72 (DSpark/MTP is enabled only at CONC ≤ 4).

Workload: agentic replay, concurrency **72**.

## A4. Gate lines to verify before trusting any number

The launcher prints these; all five must appear:

```
[k3-image] k3-image: kimi-k3-vllm:v4
[dcp] ENABLED size=8 backend=a2a interleave=1
[chunk] max_num_batched_tokens=16384 conc=72
[mns] max_num_seqs=80 conc=72 offload=dram
graphs: dense ladder 1..80 (mns=80 x 1 rows), DCP=8
```

The ladder line matters: `LADDER = MAX_NUM_SEQS × SPEC_ROWS`, and capturing
below max batch is the `HSA_STATUS_ERROR_OUT_OF_RESOURCES` signature.

## A5. Expected result

| metric | value |
|---|---|
| **Throughput per GPU** | **10,607 tok/s/GPU ± 1.2% (n=4)** |
| individual runs | 10,632 · 10,630 · 10,646 · 10,518 |
| request error rate | 0.22 – 0.31% |
| GSM8K exact_match @ C72 | 0.995 |

**Quote the band, not a point.** Two byte-identical runs 26 h apart (T206 /
T228) differ by 1.20%. Any single number reported without the band will look
irreproducible to the next person who runs it.

At C1 (MTP on, DCP off, mns 1, chunk 8192, ladder 1..9): TPOT **9.06 ms** mean /
9.10 median / 9.31 p99, spread 0.22 ms.

## A6. Config levers already exhausted — do not re-derive

| knob | swept | outcome |
|---|---|---|
| concurrency | 52 / 60 / 64 / 72 / 80 | peak **72**; 9,482 / 9,775 / **10,632** / 9,864 |
| max-num-seqs | 80 vs 96 @ C72 | flat |
| chunk | 16384 vs 8192 @ C72; 8192/4096/2048 @ C1 | 16384 @ C72, 8192 @ C1 |
| gpu-mem-util | 0.90 vs 0.92 @ C1 | 0.92 is catastrophic (9.06 → 21.61 ms) |
| KV offload | dram vs none @ C1 | flat |

Every delta below ~1.2% at C72 is inside the noise band and cannot be ranked.

---

# B. The mergeable-PR path

## B1. Why the current artifact is not mergeable

The overlay has **no PR of its own**, is authored outside this team (Hyukjoon's),
and is cut against `46638857`. Reconstructing the stack by applying the 17 PRs
from Hyukjoon's manifest to a pristine base **does not build** — verified, not
assumed: dry-run against pristine `nightly-46638857` applied only #54546;
#53917, #54038, #54457 and #54639 all failed as they are cut against much newer
bases. The PRs also collide with each other.

So today: reproducible ✅, mergeable ❌.

## B2. What is already merged and where it lives

Four of the seventeen are merged. All four are in `nightly-7c5dc571…` and
**none** are in our v4 base — ancestry-verified against each merge commit:

| PR | what | 46638857 | 7c5dc571 |
|---|---|---|---|
| #51705 | causal multi-token verification under DCP | behind | **ahead** |
| #53598 | per-group DCP cache geometry / prefix-cache hits | behind | **ahead** |
| #52707 | prevent negative external block allocation | behind | **ahead** |
| #52033 | ROCm dual-stream shared-expert | behind | **ahead** |

Moving the base to `7c5dc571` absorbs those four for free. It also **costs the
overlay**, which will not apply there. That is the whole difficulty.

## B3. What stock upstream actually delivers without the overlay

Measured on bare `nightly-7c5dc571`, zero patches:

| workload | result |
|---|---|
| C1 fixed-len TPOT | **8.52 ms** — better than our patched C1 |
| GSM8K @ C52 | 0.985 — numerics are sound |
| agentic C1 | 1,222 tok/s/GPU, err 0.54% |
| agentic C16 | 3,591 tok/s/GPU, err **10.05%** |
| agentic C32 | 1,843 tok/s/GPU, err **16.92%** |
| agentic C52 | **starves** |

**Stock upstream is not shippable for agentic; the ceiling is ~C16.** The overlay
is what makes high concurrency work at all — it is not a tuning nicety.

## B4. Ordered plan to close the gap

1. **File group A first** (+48/−10, `v1/attention/ops/dcp.py`). Self-contained
   ROCm correctness fix — RCCL bakes buffer addresses into a FULL graph and a
   function-local `torch.empty` can be freed post-capture → aperture violation.
   Small, reviewable, no K3 coupling. Measured neutral for throughput (T222), so
   it can be filed on correctness grounds alone.
2. **Group B** (+34/−11, spec-decode cudagraph). Small, no K3 coupling,
   measured inert at C72 (T223).
3. **Coordinate on C and D rather than filing competing patches** — #53917 and
   #54457 already cover C's ground; #54546 and #54639 cover D's.
4. **Rebase the surviving delta onto `7c5dc571`.** This is the load-bearing step
   and it is real work: the overlay is cut against a July-era nightly, so every
   new-file hunk needs re-checking against current `main`.
5. **Group E (K3 AMD model path) is the hardest** and may stay vendor-carried.

Nothing can be filed without Hyukjoon's sign-off — the authorship is theirs.

## B5. #54546 is a supplement, not a replacement

It is the one candidate that applies to our base, and it is not sufficient:

| | overlay | #54546 | nightly today |
|---|---|---|---|
| `supports_non_causal_multi_token_dcp` | yes | yes (ROCm-gated) | absent |
| `supports_dcp_with_varlen` | yes | yes | absent |
| **`_cudagraph_support = UNIFORM_BATCH`** | **yes** | **no** | old name |

The third row is load-bearing. Without it the DSpark draft demotes the engine
`FULL_AND_PIECEWISE → PIECEWISE`, measured at **14.05 → 77.65 tok/s**
(ITL 71.16 → 12.88 ms, 5.52×).

## B6. What the ablation contributes, and what it cannot

`OVERLAY-ABLATION.md` has the detail. Summary of relevance to B:

- File-group axis: only **8 KB of 264 KB** is detachable (A + B). C+D+E are one
  coupled unit — three runs (T224/T225/T226) proved they fail on imports.
- PR axis: **20% of the patch** splits into ten individually-detachable buckets
  (`k3_patches/pr_split/`, recombination byte-identical). Better than A–E gave
  us, still leaves 80% inseparable.
- **Neither axis found meaningful perf to prune.** The value is upstreaming
  surface reduction, not throughput.
- The three highest-perf PRs — #53166, #51437, #52190 — are **not separable at
  all**; they live entirely in contested files. This experiment can say what is
  droppable, not what is valuable.

---

## Honest summary

**A is deliverable now.** Pinned base digest + pinned overlay sha256 + baked
image + the C72 config above reproduces 10,607 ± 1.2%, and the gate lines make a
silently-unpatched run impossible to mistake for a good one.

**B is not, and no amount of measurement will make it so.** The blocker is
review throughput on thirteen open PRs plus a rebase of the unpublished
remainder onto a newer base — not something this ledger can close. The
encouraging part, from Hyukjoon's manifest, is that the overlay is *not* mostly
unpublished work: most of it has PRs already open. That is a much better
position than "264 KB of vendor patch with no provenance," which is what it
looked like before the manifest arrived.

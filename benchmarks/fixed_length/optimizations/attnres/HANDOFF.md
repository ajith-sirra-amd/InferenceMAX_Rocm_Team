# AttnRes cached score weights

Ready for isolated full-model qualification after the fixed-length baseline.
Set `KIMI_ATTNRES_PRECOMPUTE_WEIGHTS=1` on every server worker before Python
imports the model. Keep all SA server arguments and the image digest unchanged.
The flag is read once at module import; restart the process to change variants.

Prepare overlays from the original pinned image's site-packages files:

```bash
python install.py --source ORIGINAL_SITE_PACKAGES --output FRESH_RUN_DIRECTORY
```

The installer verifies both original hashes and the patch hash before writing.
It preserves the original files under `original/`, applies the patch under
`candidate/`, verifies the resulting hashes, and writes `manifest.json`.
It rejects existing output directories, output inside/above the original tree,
and mismatched input source. Partial failed output is retained for diagnosis;
retry into a new datetime/run directory.

Bind these individual candidate files read-only into the server container:

| Source under `FRESH_RUN_DIRECTORY/candidate/` | Target under `/opt/runtime-venv/lib/python3.12/site-packages/` |
| --- | --- |
| `vllm/models/kimi_k3/amd/linear.py` | `vllm/models/kimi_k3/amd/linear.py` |
| `vllm/models/kimi_k3/amd/ops/attn_res.py` | `vllm/models/kimi_k3/amd/ops/attn_res.py` |

Mount only for the candidate container. Retain the run-local original files,
manifest, actual container argv/environment and logs. Never apply the patch to
the installed baseline package. Do not reuse precompiled Python bytecode for
these modules; verify loaded source paths and candidate hashes before serving.

## Scope and lifetime

The runtime gate requires rows280/336, H7168, historical blocks4/8, BF16 input,
delta, block bank and norm/projection/output-norm weights, delta present, output
normalization with epsilon1e-5, no block write and no fused FP8. Other calls use
the original weight multiplication. Batch-invariant mode is excluded.

Native MLA post-load work runs first. The model hook then prepares FP32 products
for 19 eligible layers (37–47 and85–92), two per layer: at most38 buffers and
1,089,536 bytes/rank. Buffers are registered nonpersistent; rebuilding through
the hook copies into same-device/same-shape storage to preserve its address.
Preparation during graph capture is rejected. Keep weights immutable during
serving; direct weight updates bypassing the hook can stale the cache. Model
migration, sleep and parameter offload have not been qualified.

## Required evidence from the model run

- All eight ranks load the candidate source hashes and report
  `ATTNRES_PRODUCT_PREPARED buffers=38`. Fewer buffers mean unsupported weight
  preparation; investigate before claiming the intended optimization ran.
- Expect `ATTNRES_PRODUCT_SELECTED rows=... blocks=... submission=capture|eager`.
  This is logged once per row/block pair per worker. Capture/warmup messages do
  not prove measured replay activation. Bind actual benchmark row/graph keys
  separately; the unchanged `_attn_res_kernel` name cannot identify the variant.
- Confirm model startup and graph capture succeed, then validate all192 measured
  requests at C48 and8192/1024 with zero errors. Compare mean per-request E2E,
  TTFT, TPOT and throughput against the retained baseline.
- Preserve output/token correctness evidence and graph-size transitions. The
  synthetic acceptance3.57 setting is matched for this performance comparison;
  it does not validate natural acceptance or model quality.

`validation.json` records CPU execution of actual source AST with tensor/module
doubles: permitted/excluded gates, forwarding and FP8 contracts, native loader
ordering, buffer ceiling and in-place rebuild, plus installer checks. These are
not actual Torch registration, checkpoint or GPU graph tests. The kernel file
is byte-identical to the earlier GPU-tested product implementation: 833 exact
finite comparisons; BF16 residual/output-normalized mean durations improved
10.95–17.88%. No full-model E2E uplift is established by that operator evidence.

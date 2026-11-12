# Kimi merged MoE front: isolated model candidate

This opt-in candidate merges the shared gate/up, FP32-output router and routed
input projections into one GEMM, followed by the installed AITER SiTU/split
epilogue and native top-k. It adapts vLLM PR 66736 to the frozen SA image;
`manifest.json` pins the image, original dependencies, patch and output hashes.
It is an experimental performance candidate, not a qualified model-quality change.

The fixed-length wrapper selects it with `OPTIMIZATION=moe_front` and passes
`KIMI_MOE_MERGED_FRONT=1` at process start. `install.py --source <site-packages>
--output <fresh-run-directory>` prepares original/candidate copies. Mount only
the two candidate files at their recorded installed paths. Do not edit the image
or installed baseline packages. Preserve the separate unmodified control run.

## Runtime contract

- MI355X/gfx950, TP8, DP/PCP/EP1; BF16 inputs and weights, FP32 router logits.
- Explicit component-tested rows: 259, 266, 273, 280, 287, 294, 301, 308, 315,
  322, 329, 336. The merged path also requires no active shared auxiliary-stream
  overlap for that invocation. Every other row/precision uses the native helper.
- All native MLA post-load processing completes before one-time weight merging.
  Native weights become views of the merged allocation. No live reload, sleep,
  device migration or changing the flag after import is qualified.
- The opaque full-MoE operation resolves the layer exactly once. Native fallback
  calls `_forward_impl` inside that operation, avoiding a second implicit layer
  index advancement. Native large-prefill projection classes stay unchanged.
- Original correction bias, grouped top-k, experts, shared down projection and
  final routed/shared tail are retained.

Require one readiness marker for every rank 0–7, with 92 prepared layers:

```text
MOE_MERGED_FRONT_READY rank=0 active=True prepared=92 expected=92 rows=259,266,273,280,287,294,301,308,315,322,329,336
MOE_MERGED_FRONT_SELECTED rank=0 rows=336 submission=capture
```

The second marker is emitted only for the first eligible submission at each row
per worker. It proves submission during eager execution or graph capture; it does
not establish measured graph replay frequency. Keep that distinction when
interpreting a full-model timing result.

## Evidence and remaining gate

GPU component runs 43187278675 and 43188122614 completed. The wider run covers
all twelve admitted rows, original and changed inputs, finite outputs and exact
agreement between each implementation's own eager and graph outputs. Graph mean
time falls 27.59–41.47% for the measured front operation. The probe excludes
expert execution, shared down projection and collectives, so this is not a
whole-MoE or request E2E improvement.

The two implementations have small BF16/FP32 numerical differences. No sampled
expert sets changed, but one row's ordered IDs changed at M294 original, M301
changed-input and M336 original. Strict candidate parity is false. Synthetic
component inputs do not establish checkpoint quality or natural acceptance.

`validation.json` records clean patch installation, all twelve runtime gates,
native fallback equivalence, changing-shape dispatch, one layer lookup, shared
MLP input contract, post-load ordering, eight rank markers, disabled flag,
capture rejection and installer failure controls. These are CPU checks.

Next gate is the isolated fixed-length model run: all 192 measured requests
must complete, eight workers must initialize and submit the candidate path,
and frozen server/client/environment must match the control apart from the
explicit optimization and local ports. Inspect retained logs and timings before
claiming an E2E improvement; repeat promising changes against a fresh control.

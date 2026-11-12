# Kimi MoE sorter crossover probe

Run in the pinned SA vLLM image on one MI355X, using the existing bounded probe
workflow. No model load, serving patch, installed-source edit or allocation change:

```bash
/opt/runtime-venv/bin/python benchmarks/sa/profile/optimization_moe_sort.py \
  --output /logs/optimization-moe-sort --rows 259,280,336 \
  --rounds 30 --max-seconds 300
```

The output directory must be fresh. `/logs` should use the workflow's authorized
dated storage under `/share_nfs/ajith/benchmarks/`. Start with one representative
row (`--rows 280`) if only a short gap is available. The default is nine cases:
three token counts and balanced E896, clustered E64, and hot E16 histograms.
These are synthetic workloads, not measured checkpoint routing histograms.

| Arm | Installed operation | Interpretation |
|---|---|---|
| `native` | Opus auto, exact M, both aux maps | Current E896/topk16/BM16 sort and zeroing path |
| `native_local` | Same, with `num_local_tokens=M` | Control for dynamic local-row handling |
| `single` | Opus `dispatch_policy=1` | Existing single-kernel control |
| `byte512_with_copy` | Physical 512 rows, `num_local_tokens=M` | Existing byte-mesh path plus two active ID/weight copies |
| `adaptive` | Existing `mxfp4_moe_sort`, prologue0/BM16 | Installed fused sort and output-zeroing control |

The byte-mesh threshold is hard-coded at physical M512 in
`moe_sorting_mesh_byte_size`; dispatch policy2 only forces multiphase sorting.
Therefore the padded arm is not a pure threshold change. It keeps only M valid
rows through the device-side local count and gives the sorter the same M×3584
BF16 output as other arms. Inactive padding contains valid expert IDs and
nonzero weights deliberately: the checker rejects any leaked padding route.
Each timed padded invocation copies active IDs and weights into the larger
input buffers. Workspace capacity and graph launch/layout effects are retained.

The pinned dispatch permits the smaller M-row output: E896 selects multiphase,
physical512 selects the small uint8 path, and the non-null local count selects
P0_v2/P23 with `LocalToken=true`. P0_v2 has no output pointer; P23 reads M from
the device count and bounds zeroing by M×3584 elements. The separate P2 kernel
that zeros physical rows is not selected. Reverse-map initialization does use
physical512, so its allocation covers all 512×16 entries.

All arms emit sorted IDs, FP32 weights, expert IDs, padded/valid counts,
`m_indices`, reverse maps, and zero the complete output. Comparisons check the
full route/weight multiset, padding sentinels, both maps, output zeroing and
buffer canaries. Graph replay is rechecked with changed expert assignments,
weights and padded histograms. Within-expert order and own eager/graph bitwise
agreement are reported separately; ordering differences can affect subsequent
atomic accumulation even when routing semantics pass.

Timing covers the complete sort, auxiliary outputs, output zeroing and padded
copies. It excludes caller allocation, router top-k, quantization, GEMMs and
collectives. Both eager and graph samples are retained, with rotated/reversed
arm order. Never convert a sorter percentage directly to request E2E uplift.

If an arm wins repeatably, first validate its downstream full-MoE outputs and
timing using the same routes, weights and atomic output contract. A promising
byte result motivates a separate run-local source experiment changing the
shared sizing/dispatch helper for exact M; it does not authorize a serving
threshold change. Keep the frozen SA control unchanged.

Local preparation verified the pinned source hashes and 18 CPU checker cases
covering dense/sparse histograms, preserved/reversed route order and both native
padding encodings. Seven corrupted-output controls were rejected. GPU execution
and downstream numerical qualification remain pending.

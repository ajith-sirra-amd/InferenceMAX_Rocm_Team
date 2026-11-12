# Kimi MoE sorter crossover probe

Run in the pinned SA vLLM image on one MI355X, using the existing bounded probe
workflow. No model load, serving patch, installed-source edit or allocation change:

```bash
/opt/runtime-venv/bin/python benchmarks/sa/profile/optimization_moe_sort.py \
  --output /logs/optimization-moe-sort-native-smoke --rows 259 \
  --arms native --histograms balanced --rounds 6 --max-seconds 60
```

The output directory must be fresh. `/logs` should use the workflow's authorized
dated storage under `/share_nfs/ajith/benchmarks/`. Defaults are the single native
smoke above. A separate parent process enforces the deadline and kills the
worker process group; it waits at most five extra seconds for teardown.
`probe.log` retains stdout/stderr and periodic Python stack dumps. `result.json`
saves a checkpoint before each arm, eager synchronization, graph capture/replay
and timing phase, including a terminal timeout result if HIP blocks.

After native smoke passes, compare only `--arms native,byte512_with_copy` at the
same M259 before widening. Then explicitly select `--rows 259,280,336` and
`--histograms balanced,clustered64,hot16` for coverage. The single-kernel and
adaptive controls require explicit selection; neither runs by default. These
are synthetic workloads, not measured checkpoint routing histograms.

Run43196486628 was cancelled after its in-process SIGALRM failed to interrupt
a native call. It reached the first M259/balanced case, but old checkpoints did
not persist per-arm progress, so the exact stalled arm is unproven. Its saved
artifact is retained. Do not repeat all nine cases to diagnose that failure.

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

The pinned header's zeroing bounds support the M-row output allocation:
E896 auto dispatch selects multiphase; physical512 selects the `<2048` uint8
path; non-null `num_local_tokens` selects P0_v2 and P23 with `LocalToken=true`
(`moe_sorting_opus.h`:1450, 3679, 3722, 3594). P0_v2 has no output pointer.
P23 reads the device-side M and passes it to the zeroing helper (2914–2927),
which bounds stores to `M*3584*2/16` aligned vectors (1397). The physical-row
zeroing at2467 belongs to P2, which this dispatch does not invoke. Reverse-map
initialization does use physical512; that array is allocated for512×16 entries.
Canaries supplement this source proof; they are not its basis.

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

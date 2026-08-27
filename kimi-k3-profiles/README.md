# Raw profiling output — Kimi-K3 on 8× MI355X

Unprocessed profiler output, committed verbatim. Nothing here has been filtered,
recomputed, or reformatted. For the analysis derived from it see
[../Kimi-K3-Where-The-Time-Goes.md](../Kimi-K3-Where-The-Time-Goes.md).

## `T124_rocprofv3_k_kernel_stats_no-offload.csv` — current

| | |
|---|---|
| Run | [T124](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/33033974874) |
| Date | 2026-08-27 |
| Config | DCP=8, conc 52, `max_num_seqs` 65, dense ladder 1…65, TP=8, EP=1, **no KV offload**, no MTP, gmu 0.9 |
| Trace | 3.1 GB, **8,474,873 dispatches**, serving window 1430.7 s |
| Result | GPU busy **71.8%**, idle **28.2%** |
| MD5 | `9c779a75b49fbd33c198690355649210` |

This is the current profile. It supersedes T116 below, which was taken with the
DRAM KV offload enabled — the offload has since been removed, and removing it is
what took idle from 44.3% to 28.2%.

Same two caveats apply as for T116: the `Percentage` column is unusable because
of the wrapped `cross_device_reduce_2stage` row, and the stats aggregate all
8 GPUs.

## `T116_rocprofv3_k_kernel_stats.csv` — superseded (offload ON)

| | |
|---|---|
| Run | [T116](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/32964875218) |
| Date | 2026-08-26 |
| Hardware | 8× MI355X (gfx950), ROCm 7.2.3 |
| Config | DCP=8, conc 52, `max_num_seqs` 80, dense capture ladder 1…80, TP=8, EP=1, DRAM KV offload, gmu 0.9, no MTP |
| Model | `moonshotai/Kimi-K3` (MXFP4 routed experts) |
| Image | `aigmkt/kimi-k3-vllm:latest` |
| Command | `rocprofv3 --kernel-trace --stats -f csv -d "$RP_DIR" -o k -- vllm serve …` |
| Rows | 403 kernels |
| MD5 | `ad2f908c5dd580b972a75b2ffd9d1ccc` |

Columns are rocprofv3's own:
`Name, Calls, TotalDurationNs, AverageNs, Percentage, MinNs, MaxNs, StdDev`

### Two things to know before using it

**1. One row is corrupt, and it poisons the `Percentage` column for every row.**

```
"void aiter::cross_device_reduce_2stage<std::bfloat16_t, 8, false>(...)",
  1842876, 18444334160200519208, ..., 100.00, ...
```

`18444334160200519208` ns ≈ 2⁶⁴ — an unsigned wrap of a *negative* duration.
rocprofv3 mis-times the custom all-reduce because that kernel spin-waits on
peer signals, so its end timestamp can precede its start. Because rocprofv3
computes `Percentage` against the total, that single row takes 100.00% and every
other row is driven to ~1e-6. **Treat `Percentage` as unusable.**
`Calls`, `TotalDurationNs`, `AverageNs`, `MinNs`, `MaxNs` are sound on the other
**402** rows, which sum to **2052.1 s** of kernel time.

**2. These stats aggregate all 8 GPUs.** e.g. `ncclDevKernel_Generic_1` shows
1,191,325 calls here versus 732,464 on Agent 2 alone. Divide by device, or use
the trace, if you want per-GPU figures.

## The full kernel traces are not in git

`k_kernel_trace.csv` is **2.3 GB / 6,356,992 rows** — one row per dispatch, with
`Agent_Id`, `Stream_Id`, `Start_Timestamp`, `End_Timestamp`, grid/workgroup
dimensions. Too large to commit. It is on the benchmark host at:

```
T116 (offload ON) : /data/hf_hub_cache/kimi-profiles/rocprof_20260826-114528_dcp8_conc52/k_kernel_trace.csv
T124 (offload OFF): /data/hf_hub_cache/kimi-profiles/rocprof_20260827-024048_dcp8_conc52_kvnone/k_kernel_trace.csv
```

The occupancy and idle-gap numbers in the analysis doc come from this file, not
from the stats CSV, because they need per-dispatch timestamps: GPU-busy is the
**union of merged intervals** on a single agent, and the idle attribution needs
each gap's bracketing kernels.

### Regenerating it

Set `ROCPROF=1` when running the Kimi benchmark script (it defaults to `0`; see
the comment at `ROCPROF=` in `kimik3_fp4_mi355x_mtp.sh`). Output lands in
`/mnt/hf_hub_cache/kimi-profiles/rocprof_<UTC>_dcp<N>_conc<C>/` inside the
container. Both files are written when the server process exits, not
incrementally.

**Tracing costs ~17% of input throughput** (54–57k tok/s traced vs 65–68k
untraced), so a traced run's throughput number is not comparable with an
untraced one. Only kernel *ratios* transfer.

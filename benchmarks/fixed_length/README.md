# Kimi-K3.1-Pro fixed-length comparisons

This uses the existing
[`kimik3_fp4_mi355x_mtp.sh`](../../upstream/InferenceX/benchmarks/single_node/agentic/kimik3_fp4_mi355x_mtp.sh).
After the same server startup and readiness check, `TEST=1` selects
`run_benchmark_serving` in place of the AgentX replay. `TEST=0` keeps the replay.

Run on the MI355X host:

```bash
bash benchmarks/fixed_length/kimi_k31_c48.sh \
  /share_nfs/ajith/benchmarks/fixed-$(date -u +%Y%m%dT%H%M%SZ) baseline
```

The caller supplies **8192 input / 1024 output tokens, C48, 192 measured
requests, range ratio 1**. The shared helper performs 96 warmups. It uses its
native random workload, chat template, seed 0 and ignored EOS. Detailed results
must pass `validate.py` before this is accepted as a baseline.

The agentic script explicitly contains all 25 engine environment settings and
the server command from SA run 42638222682, including device IDs 0–7.
`baseline.json` retains the image and reference settings. `audit_server.py`
compares the actual command/environment against that reference and records source
hashes before launch. It does not configure the server. Older tuning defaults
have been removed; this script pins the C48 throughput point.
The `baseline` mode applies no optimization patch, profiler, CPU affinity change
or package installation. The `attnres` mode prepares two verified source overlays
in the new run directory and mounts them read-only for that server process.
It enables cached score weights for the qualified BF16 AttnRes calls;
`optimization.json` records exact originals, patch and candidate hashes. The SA synthetic acceptance length 3.57 is retained for this performance
comparison. This test does not establish model quality or natural draft acceptance.

Outputs stay in the supplied directory: `recipe.json`, `effective-server.json`,
`vllm_command.txt`, `baseline-command.txt`, `baseline.json`,
`validation.json`, `server.log` and `workflow.log`. Regenerable caches live in
`work/`. Prior directories are never overwritten. GitHub Actions uploads the
compact results and logs; the original data remain on the shared filesystem.

Use the **e2e Test** workflow on **IPAT-Script**, choose **Fixed-length Kimi
baseline**, **Fixed-length Kimi AttnRes** or **Fixed-length Kimi DSpark**, and give the run a descriptive name. It targets the existing
`cluster:mi355x-amds-2` runner without modifying allocation or reservation.

Collect and validate this unchanged baseline before applying candidates. Compare
the same fixed workload and configuration for each candidate; then validate a
promising change with the full agentic workload. Fixed-length and AgentX results
are not interchangeable because their cache reuse and request distributions differ.

The `dspark` mode shards the draft context projection over TP8, gathers the
output, then keeps the existing full-width normalization. It uses a single
verified read-only overlay and records native dependency hashes. Real-checkpoint
component checks pass, but BF16 rounding differs; serving performance under
synthetic acceptance does not establish natural acceptance or model quality.

`validate_optimization.py` requires candidate preparation/loading markers on
all eight workers. AttnRes capture-time selection is distinguished from measured
replay frequency. Run-local Python cache paths avoid installed bytecode reuse.

The `moe_front` mode combines shared gate/up, router and routed latent front
projections, with the installed SiTU/split epilogue. It preserves the native
path outside the twelve qualified decode row sizes. Two source overlays and
`KIMI_MOE_MERGED_FRONT=1` isolate this arm; all eight workers must prepare92
layers. Component tests preserve expert sets but show small floating-point
and occasional expert-order differences; full-model quality remains open.

The `moe_front_dspark` mode measures both candidates together. It composes their
unchanged installers, checks that the three source overlays are disjoint, and
retains both complete component manifests in `optimization.json`. Both flags
are enabled; all eight workers must show MoE preparation/selection and DSpark
construction/loading. Baseline and isolated modes remain available. The combined
result must be measured directly; isolated gains are not additive estimates.

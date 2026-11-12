# Kimi-K3.1-Pro fixed-length baseline

This uses the existing
[`kimik3_fp4_mi355x_mtp.sh`](../../upstream/InferenceX/benchmarks/single_node/agentic/kimik3_fp4_mi355x_mtp.sh).
After the same server startup and readiness check, `TEST=1` selects
`run_benchmark_serving` in place of the AgentX replay. `TEST=0` keeps the replay.

Run on the MI355X host:

```bash
bash benchmarks/fixed_length/kimi_k31_c48.sh \
  /share_nfs/ajith/benchmarks/fixed-$(date -u +%Y%m%dT%H%M%SZ)
```

The caller supplies **8192 input / 1024 output tokens, C48, 192 measured
requests, range ratio 1**. The shared helper performs 96 warmups. It uses its
native random workload, chat template, seed 0 and ignored EOS. Detailed results
must pass `validate.py` before this is accepted as a baseline.

`baseline.json` freezes the image, all 25 engine environment settings and server
arguments from the verified SA reference. `prepare.py` only serializes that
configuration for the existing Bash launcher and records installed source hashes.
The frozen configuration bypasses the older tuning defaults in the branch base.
No optimization patch, profiler, CPU affinity change or package installation is
applied. The SA synthetic acceptance length 3.57 is retained for this performance
comparison. This test does not establish model quality or natural draft acceptance.

Outputs stay in the supplied directory: `recipe.json`, `effective-server.json`,
`server-config.sh`, `vllm_command.txt`, `baseline-command.txt`, `baseline.json`,
`validation.json`, `server.log` and `workflow.log`. Regenerable caches live in
`work/`. Prior directories are never overwritten. GitHub Actions uploads the
compact results and logs; the original data remain on the shared filesystem.

Use the **e2e Test** workflow on **IPAT-Script**, choose **Fixed-length Kimi
baseline**, and give the run a descriptive name. It targets the existing
`cluster:mi355x-amds-2` runner without modifying allocation or reservation.

Collect and validate this unchanged baseline before applying candidates. Compare
the same fixed workload and configuration for each candidate; then validate a
promising change with the full agentic workload. Fixed-length and AgentX results
are not interchangeable because their cache reuse and request distributions differ.

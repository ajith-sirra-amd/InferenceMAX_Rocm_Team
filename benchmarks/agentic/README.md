# Kimi C48 SA configurations

The current
[`kimik3_fp4_mi355x_mtp.sh`](../../upstream/InferenceX/benchmarks/single_node/agentic/kimik3_fp4_mi355x_mtp.sh)
contains the resolved server command from SA run **44640765021**, job
**135971483924**, measured checkout
`d73305abce215bfb1cd46b639dedeffd4d87377f`. As checked on November 26, 2026,
datapoint **448458** (published November 23) is the latest published vLLM MI355X
agentic high-concurrency point: **C48, TP8/DCP8, EP1**. Its published output
throughput is **152.2541 tokens/s/GPU** on eight MI355X GPUs.

Use the matching image:

```text
vllm/vllm-openai-rocm:kimi-k31-712d9eb@sha256:7c167350f127a3019ac2c2fc99f6d3229356e8e8e033a7634c3201f072f0ecf4
```

The full image, model revisions, server environment, arguments and source
provenance are saved in [`sa44640765021.json`](sa44640765021.json). The script
keeps the environment and command explicit; this reference does not inject
configuration. It uses the released **Inferact/Kimi-K3.1-Pro-DSpark** revision
`e4e6e6261e9f8b6d5f249f9627eb6b109d762d5f`, four speculative tokens,
`TRITON_ATTN`, and the measured golden-planner synthetic acceptance **3.46**.
Prefix matching is **1152** tokens. The new graph sizes and prefill/DCP
environment settings match the measured job; its C48 command has no explicit
prefill-cadence or adaptive long-prefill flags.

Mount the host's **/data** into the container and supply `MODEL_PATH` and
`DRAFT_MODEL_PATH` pointing to the corresponding local snapshots. The script
requires `MODEL=moonshotai/Kimi-K3.1-Pro`, `TP=8`, `DCP_SIZE=8`, `CONC=48`,
`EP_SIZE=1`, `KV_OFFLOADING=dram`, `KV_OFFLOAD_BACKEND=vllm-simple`,
`TOTAL_CPU_DRAM_GB=1799`, `EVAL_ONLY=false`, `PORT` and `RESULT_DIR`.
For `TEST=1`, also supply `TEST_ISL`, `TEST_OSL`, `TEST_RANGE_RATIO`,
`TEST_NUM_PROMPTS` and `RESULT_FILENAME`; this runs the fixed-length client.
For native replay, supply `TEST=0`, `DURATION=3600` and a prepared
`SA_CLIENT_ENTRYPOINT` from the measured source with its `--check` contract.
The historical `client.py` below is pinned to the earlier run and must not be
used as evidence of latest-workload replication. The legacy generic E2E config
is also not a launcher for this latest point.

The change was checked locally against the measured command and environment.
No model startup, GPU benchmark or new throughput measurement was performed.
The MI355X published result is not a measured result for the MI350 runner.

## Historical native SA agentic comparison

This client hands off to the measured InferenceX `infx.bench agentic` implementation
from SA run **42638222682**, job **129550719685**, source
`903a3023682c6b27683839309cbd4b4d9153afcb`. Both comparison arms use the same prepared
CPython 3.11.18 / AgentX 1.0.13 runtime and all 129 package versions recovered from
that job. The wrapper selects the
[archived server script](../../upstream/InferenceX/benchmarks/single_node/agentic/deprecated/kimik3_fp4_mi355x_mtp_sa42638222682.sh)
to preserve that run's image, preview draft and optimization comparisons.

The client executes the complete native warmup, 3600-second measured replay,
aggregation, distribution analysis and validation. The native AgentX preset has
10 warmup requests per lane and seed 42. No fast-mode or shortened-warmup override
is added. This is a direct core-recipe comparison; SA's Slurm orchestration and
9405/power exporter are not replicated. Server metrics from `/metrics` are required;
power is explicitly unavailable. Synthetic acceptance 3.57 remains enabled, so
this workload does not establish natural acceptance or model quality.

## Prepare once, before loading GPUs

Use a clean checkout of the measured SHA; pass its **inferencex-e2e** subdirectory:

```bash
bash benchmarks/agentic/prepare.sh \
  /share_nfs/ajith/benchmarks/agentic-<run>-<attempt>/client \
  /path/to/sa-checkout/inferencex-e2e
```

Preparation uses the pinned serving image in a CPU-only container. It does not
install into `/opt/runtime-venv` or patch SA source. `HF_TOKEN`, if present, is
forwarded by environment name. Dataset snapshot
`23f152f6f0f9399a85901b89a6458def0ef16729` and tokenizer snapshot
`d9e99b0792a5e93b78ea9bea62a68167e36ddd9e` are cached under the pair directory.
Tokenizer loading fetches tokenizer assets, not model weights. The native dataset
has 393 rows; runtime reconstruction can produce a roughly 6.7 GB mmap under
the arm's `work/agentic-tmp`, so shared run storage must have space.

The pair-owned cache's `refs/main` binds the exact saved revisions. Preparation
then tests the native default-main `hf download`, dataset loader and tokenizer
offline, comparing the dataset fingerprint and token IDs to pinned online loads.
Only a fully successful preparation writes `prepared.json`; a failed directory
is preserved and a retry uses a new directory. Network/bootstrap work is bounded
to 30 minutes. Dependency availability and offline API compatibility must pass
this CPU-only gate before either server is allowed to start.

## Run both arms

```bash
bash benchmarks/fixed_length/kimi_k31_c48.sh \
  /share_nfs/ajith/benchmarks/agentic-<run>-<attempt>/baseline baseline agentic \
  /share_nfs/ajith/benchmarks/agentic-<run>-<attempt>/client \
  /path/to/sa-checkout/inferencex-e2e

bash benchmarks/fixed_length/kimi_k31_c48.sh \
  /share_nfs/ajith/benchmarks/agentic-<run>-<attempt>/combined moe_front_dspark agentic \
  /share_nfs/ajith/benchmarks/agentic-<run>-<attempt>/client \
  /path/to/sa-checkout/inferencex-e2e
```

The wrapper mounts pair storage at `/agentic-work`, measured source at `/sa-client`
and these helpers at `/agentic-tools`. `client.py --check` validates source/package
identity, cache hashes and native plan before GPU wait/start. After readiness,
`client.py` executes the exact prepared venv Python with `-m infx.bench agentic`.
That exact executable path prevents native bootstrap from recreating the runtime.
Client environment and `PYTHONPATH` are isolated from the serving runtime.
Offline dataset/tokenizer APIs can still create cache locks or dynamic-module
metadata. The root-owned serving arms therefore need final ownership repair of
the shared client's cache/tmp directories, as well as each arm's result files.

Each arm preserves `recipe.json`, `agentic-client-check.json`,
`agentic-client-start.json`, server/benchmark commands and logs, native
`agentic.json`, per-request `agentic/aiperf_artifacts/profile_export.jsonl`, server
metrics and native validation outputs. The native failure threshold is 0.10;
inspect raw request/error counts rather than treating a zero exit as zero errors.
Allow native replay's drain/grace period and export to finish. Request counts
need not match: this is fixed-time trace replay, not the fixed 8192/1024 test.

Compare both arms using the same native request filtering and timing definitions;
preserve warmup and error records as evidence. Only claim a full completed result
after native validation, required `vllm:` metrics, unchanged recipe audits and the
candidate source/activation checks have passed. Client preparation alone is not a
completed model benchmark.

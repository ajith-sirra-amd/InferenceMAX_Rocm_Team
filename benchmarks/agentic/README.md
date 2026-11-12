# Native SA agentic comparison

This client hands off to the measured InferenceX `infx.bench agentic` implementation
from SA run **42638222682**, job **129550719685**, source
`903a3023682c6b27683839309cbd4b4d9153afcb`. Both comparison arms use the same prepared
CPython 3.11.18 / AgentX 1.0.13 runtime and all 129 package versions recovered from
that job. The existing agentic server script supplies the unchanged SA core recipe.

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

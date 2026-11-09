# Native SA benchmark runs

Run one published SA datapoint using its pinned InferenceX recipe binder,
golden-acceptance planner, and SRT benchmark engine. The native generated batch
script owns server startup, readiness, benchmark warmup, measurement and cleanup.
The team launcher supplies local storage and scheduler settings.

In **End-to-End Tests → Run workflow**, set `sa-config` to a checked-in JSON
configuration, `allocation-id` to your existing single-node reservation, and
`test-name` to a descriptive model/framework/GPU/concurrency/pass name. When
`sa-config` is set, the sweep command is ignored. A replication run and a profile
run must have separate workflow IDs.

The same entrypoint works directly from this repository on the allocated host:

```bash
python3 benchmarks/sa/run.py \
  --config benchmarks/sa/kimi-k3.1-pro-vllm-mi355x-c48.json \
  --allocation YOUR_ALLOCATION_ID \
  --output /share_nfs/ajith/benchmarks/YOUR_UNIQUE_RUN_NAME
```

Each JSON freezes the InferenceX and srt-slurm SHAs, native recipe/variant,
image digest and workflow environment. The golden acceptance value is selected
by InferenceX; do not hard-code it. Keep model-specific settings in JSON files.

The launcher verifies the allocation belongs to the current account and contains
only the current host. It executes the native rendered batch script in that
allocation and does not cancel the reservation. The Spur adapter translates
unsupported Pyxis default flags and append-mode output routing; it retains the
native task command. Model/cache mounts, output paths, network interface and
telemetry differences are recorded in `native-plan.json`. Native GPU and RAPL telemetry are requested; unavailable permissions remain
explicit missing evidence. Release assets use the GitHub API route on this network. Recipe equality alone does not certify
performance equivalence.

Source checkouts and Python environments stay within this repository under
`.sa-runs/`. Run data goes to `/share_nfs/ajith/benchmarks/<run>-<attempt>/`:

- `request.json`: frozen inputs.
- `bound-sa-recipe.yaml`: recipe selected by the unmodified SA binder.
- `native-plan.json`: generated command, acceptance and infrastructure adaptations.
- `rendered/`: native batch script and resolved configuration.
- `native/<allocation>/`: native benchmark logs, metrics and telemetry.
- `work/`: regenerable caches; excluded from Actions artifacts.

Native replication is under live validation. Successful model startup,
benchmark completion and profiling must be established by the run artifacts;
a successful dispatch or setup does not establish them.

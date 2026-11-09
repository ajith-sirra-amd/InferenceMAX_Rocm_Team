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
allocation and does not cancel the reservation. Spur's overlap mode ignores
container image/mount options, as confirmed by the retained transport check.
The adapter therefore runs Docker through the real reservation, preserving the
native task command and mounts. Model/cache mounts, output paths, network interface and
telemetry differences are recorded in `native-plan.json`. Native GPU and RAPL telemetry are requested; unavailable permissions remain
explicit missing evidence. Release assets use the GitHub API route on this network. Recipe equality alone does not certify
performance equivalence.

Use `sa-mode: transport-check` for a small image/mount check before model startup.
Container writability follows the native SRT option. SA's writable container is
needed for JIT modules built and imported under the installed package directory;
the image itself remains unchanged. Frozen workflow inputs and SA's workload
environment names are explicitly forwarded into Docker, preserving the native
inheritance needed by the benchmark client. Runtime caches stay in the run's `work/`
directory, with compiler temporary files on bounded tmpfs. SRT owns the process lifecycle.

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

An optional `profile` object selects the capture hook, currently for vLLM:
`{"tool":"rocprofv3","mode":"launch","delay_seconds":CALIBRATED_DELAY,"seconds":60}`.
Use a separate configuration and workflow run after the baseline succeeds.
The hook retains native server arguments and captures kernel, HIP, copy and
marker traces as CSV/JSON under `native/<allocation>/logs/profiles/rocprof/`.
The delay must be calibrated and the actual window checked against fresh
steady-state traffic. `clock-sync.json` brackets a boot-clock sample in UTC so
native rocprof timestamps can be compared with client phase timestamps.
Attachment mode loads the installed registration library before framework startup
and enables `ROCP_TOOL_ATTACH=1` and `VLLM_WORKER_MULTIPROC_METHOD=spawn` only for
the profiling server. The explicit multiprocessing override gives each worker a
fresh interpreter for profiler registration; record this profiling difference
from the unmodified replication. Before waiting for the calibrated window, the
hook checks that all expected vLLM GPU workers have the `rocp-bg-attach` thread.
It records `attachment-preflight.json`, excludes non-worker parent processes,
and rechecks readiness before attaching to each worker separately. Missing
readiness or failed export stops the invalid profile attempt through the native
server shutdown. Synchronous detachment exports traces while the workers remain
alive, avoiding loss when the serving engine later force-kills workers during
shutdown. Native server arguments and workload duration are preserved.
`attachment-status.json` records export status and worker PIDs;
`profile-capture-validation.json` in the run root records fresh device/window
validation. The host report stays outside container-owned raw trace directories.
`mode: launch` uses rocprof with SDK signal handlers disabled, leaving root-first
shutdown to the application. Its profiling overlay gives vLLM workers 900 seconds
and the native orchestrator 1200 seconds to finalize; the exact orchestrator
timeout edit and source hashes are retained in `profile-shutdown-overlay.json`.
These are maximum shutdown allowances, not additional capture time.

For an explicitly shortened diagnostic, set the environment's `DURATION` and
the profile's `warmup_requests_per_lane`. The small client adapter reuses the
pinned native Plan, bootstrap, execution and scoring, changing only the warmup
count. Fast mode must remain disabled because it forces 1200 seconds. The
`kimi-k3.1-pro-vllm-mi355x-c48-profile-check.json` diagnostic selects 600 seconds,
one warmup request per lane, and a 60-second launch capture. Its launch delay
must be checked against fresh measurement boundaries. It is not a replacement
for the saved SA baseline. Native model coverage must always be established
from the resulting artifacts. Phase 3 saves and validates collection; Phase 4
post-processes those saved files.

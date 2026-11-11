# Native SA benchmark runs

For the current C48 measured profile, select
`benchmarks/sa/kimi-k3.1-pro-vllm-mi355x-c48-measured-profile-sa42638222682.json`.
It preserves the SA image, server environment, native warmup, client command and
3,600-second replay. The sole server profiling argument configures the installed
vLLM Torch profiler; no rocprof launch timer is used in this pass.

The independent controller reads `logs/agentic/benchmark.log`, recognizes
`Phase profiling (profiling) started`, resolves that timestamp using the native
client timezone record, and requests `/start_profile` at **measurement +2,400 s**.
It captures **130 s**, calls `/stop_profile`, and permits **1,200 s** for export.
An uncertain start response still causes a stop request. Control states and UTC
boundaries are saved continuously under `logs/profiles/torch/`. After the native
workload completes, validation requires eight complete worker traces, GPU IDs
0–7, and at least **120 s common GPU extent within measured sending**. Traces,
control logs and validation remain with the fresh workflow output; previous
captures are preserved. Timing/CPU/GPU activity is collected here; HBM counters,
tensor shapes/precision and request-to-kernel attribution require separate passes.

For a separate **metadata diagnostic**, select
`kimi-k3.1-pro-vllm-mi355x-c48-metadata-sa42638222682.json`. It retains that full
native C48 workload and the measured +2,400 s trigger, with a **20 s** capture to
bound shape/stack/allocation instrumentation overhead. Native Torch
`record_shapes`, `with_stack`, `with_memory` and `with_flops` are enabled; validation
requires actual shape, input-dtype and Python-frame events on every rank, in
addition to eight complete GPU traces. This does not replace the 130 s timing
capture. Graphs were built before the session: internal replayed operators may
have no tensor/call-stack attribution. Torch FLOP estimates do not cover all
custom kernels, and allocation records are not hardware HBM counters.

The bounded exact-image check `43010911535` collected real `FETCH_SIZE`,
`WRITE_SIZE` and BF16/F8/F6F4 math counters on synthetic graph-replayed GEMMs.
Read and write values use KiB (multiply by 1024 for bytes); MOPS values use 512
operations per count. `FETCH_SIZE` plus `WRITE_SIZE` in one pass exceeds this
installed counter configuration's hardware capacity (also verified in
`43011308200`), so read, write and math use separate passes. Do not join model
passes on symbol/dispatch number alone or label these synthetic results a Kimi
roofline. Native Proton graph attribution is not currently validated: the
installed backend fails to load its SDK library. All probe evidence remains
under the original run-specific output/artifact. The reusable bounded probe is
`profile/evidence_probe.py`; its result records individual check failures even
when the overall capability inventory completes.

`profile/extract_arguments.py SAVED_WORKER_TRACE --output NEW_JSON` extracts
directly correlated GEMM operator shape/stride/dtype/scalar records for a later
reviewed replay. It rejects `hipGraphLaunch` correlations: an outer graph replay
does not identify an inner operator's arguments. Review the installed operator
schema, fill no unknown argument by assumption, and require the replay kernel
symbol to match the captured variant before using its hardware counters. An
isolated kernel replay is separate evidence from the original model/cache state.

Pinned-image capability check `42987645614` captured 96 synthetic kernel events
on all eight GPUs. It establishes basic profiler capability, not full Kimi model
coverage. Twenty-three local adapter/controller/export checks pass, including
marker timing, a lost start response, missing-rank rejection and warmup rejection.
The marker-controlled configuration supersedes the historical launch-delay
configurations described below for this measured-phase request.

Run one published SA datapoint using its pinned InferenceX recipe binder,
golden-acceptance planner, and SRT benchmark engine. The native generated batch
script owns server startup, readiness, benchmark warmup, measurement and cleanup.
The team launcher supplies local storage and scheduler settings.

In **End-to-End Tests → Run workflow**, set `sa-config` to a checked-in JSON
configuration, leave `allocation-id` empty for direct execution on the runner, and
`test-name` to a descriptive model/framework/GPU/concurrency/pass name. When
`sa-config` is set, the sweep command is ignored. A replication run and a profile
run must have separate workflow IDs.

The same entrypoint works directly from this repository on the runner:

```bash
python3 benchmarks/sa/run.py \
  --config benchmarks/sa/kimi-k3.1-pro-vllm-mi355x-c48-sa42638222682.json \
  --output /share_nfs/ajith/benchmarks/YOUR_UNIQUE_RUN_NAME
```

Each JSON freezes the InferenceX and srt-slurm SHAs, native recipe/variant,
image digest and workflow environment. The golden acceptance value is selected
by InferenceX; do not hard-code it. Keep model-specific settings in JSON files.

The native job targets `self-hosted` and `cluster:mi355x-amds-2`; the runner does
not need a separate `nodes:1` label. Direct execution uses Docker on that host,
without looking up, creating, resizing or cancelling a reservation. A local
context ID supplies native output-directory naming; it is not an allocation.
The native engine owns process startup and shutdown through the transport adapter.
If `--allocation` is explicitly supplied, the launcher instead verifies that
existing reservation belongs to the current account and contains only this host.
Spur's overlap mode ignores
container image/mount options, as confirmed by the retained transport check.
In reservation mode, the adapter runs Docker through the real reservation, preserving the
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

The workflow also runs independent container cleanup after execution, including
on cancellation. Before a new run it stops leftovers only when both the `sa.run`
label and saved launch record identify them and GitHub confirms that attempt is
completed. Cleanup allows 1,200 seconds for graceful shutdown/export and records
the stopped containers and remaining GPU memory. It does not alter reservations.

Source checkouts and Python environments stay within this repository under
`.sa-runs/`. Run data goes to `/share_nfs/ajith/benchmarks/<run>-<attempt>/`:

- `request.json`: frozen inputs.
- `bound-sa-recipe.yaml`: recipe selected by the unmodified SA binder.
- `native-plan.json`: generated command, acceptance and infrastructure adaptations.
- `rendered/`: native batch script and resolved configuration.
- `execution-context.json`: local transport, host, and optional allocation.
- `native/<context-id>/`: native benchmark logs, metrics and telemetry.
- `work/`: regenerable caches; excluded from Actions artifacts.

The files suffixed `sa42638222682` pin the updated SA Kimi C48 reference,
including its new image, six speculative tokens and native acceptance 3.57.
The historical `profile-check` captured during warmup with its initial
1500-second delay. Use the linked `profile-check-120s-sa42638222682.json`
candidate for the retry: 600-second benchmark, one additional warmup request per
lane, 2700-second profiler delay and 130-second capture. It requires at least
120 seconds common measured coverage across eight GPUs. The paired
`profile-120s-sa42638222682.json` preserves the full native workload and uses a
3000-second delay; launch it only after the smoke passes. The
[timing record](profile/TIMING_SA42638222682.md) explains calibration and limits.
The original files preserve the prior SA reference.
These configs use `local_frontend_port: 0` to select an available host TCP port
because ports 8000 and 18000 are occupied on the runner. This pinned SRT version has no YAML port option,
so the binding changes its single shared port constant before native rendering.
Server, readiness checks and client inherit that same value; the original and
revised hashes are saved in `local-endpoint-binding.json`. Workload flags remain
unchanged.

Successful model startup,
benchmark completion and profiling must be established by the run artifacts;
a successful dispatch or setup does not establish them.

For a quick synthetic profiler check, use `sa-mode: transport-check`, an empty
allocation, the current image config, and `sa-check-command` set to
`["python3","/sa-tools/profile/dummy.py","--output","/logs/native/dummy-profile"]`.
This replays batch-8 BF16 graphs on eight GPUs for 15 seconds, requires at least
10 seconds of common steady GPU events, and validates complete CSV/JSON exports
after worker exit. It loads no model and does not validate Kimi/C48 steady-state
coverage. The real C48 profile retains its 120-second minimum.

An optional `profile` object selects the capture hook, currently for vLLM:
`{"tool":"rocprofv3","mode":"launch","delay_seconds":CALIBRATED_DELAY,"seconds":130,"minimum_seconds":120}`.
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
Launch collection records kernels, HIP graph launches, copies and markers in
CSV/JSON. Broad HIP API tracing is omitted because the installed SDK can crash
while serializing API arguments at exit. This reduces CPU API visibility;
graph records do not replace a full host call trace. Inspect complete JSON and
profiler exit evidence as well as CSV coverage before accepting the capture.
The launcher waits for native server shutdown, then checks complete, stable JSON
files against worker CSV counts and finalization metadata before artifact upload.
The capture timer is not an export timeout. Missing/truncated or still-changing
files fail validation; failed-run artifacts remain preserved for diagnosis.

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

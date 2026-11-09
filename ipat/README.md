# IPAT runner

**SA execution boundary:** the executor below reconstructs Docker execution and
does not establish native SA workflow replication. IPAT disables SA starts through
this route while the native handoff is prepared. The user and IPAT must invoke
the same team-owned SA workflow, with replication and steady-state profiling in
two independent workflow runs. See the [profiling and code-path README](https://github.com/ajith-sirra-amd/IPAT/blob/ajith-ipat/docs/PROFILING_README.md)
for commands, output files, evidence requirements and current validation status.

`e2e-tests.yml` accepts an optional `ipat-request` JSON input on branch `IPAT`.
It runs a single frozen request on the selected MI355X runner, in a dedicated
checkout. Run data goes to the user-approved MI355X server directory:

```text
/share_nfs/ajith/IPAT/runs/<workflow-id>-<attempt>/
  request.json, recipe.json, manifest.json
  index.html, report.md
  baseline/                  # Client results, logs and serving metrics
  profile/traces/            # Native profiler files
  work/                      # Regenerable caches and client environments
```

The workflow fails if `/share_nfs/ajith` is unavailable; it does not fall back to
storing run data in `_work`. The UI keeps its existing local data directory and
can still collect the Actions evidence artifact. Existing sweep dispatches keep
their original path. Shared-directory permissions and quota need runner validation.

`runner.py` records host capabilities and the resolved image/model metadata,
then runs independent baseline and profiling containers. `container.py` runs
the server and recorded client, scrapes serving metrics, and collects a bounded
Torch or rocprofv3 server trace. Torch profiling requires compatible vLLM/SGLang
endpoints; rocprofv3 attachment is checked against the installed version.
Missing GPU coverage fails the profiling pass. Baseline metrics and profiling
measurements are kept separate.

A preflight request may include `image_check: {"image": "<local pinned image>"}`
and an optional `environment` inside it. This checks small FP32/FP16/BF16 matrix
multiplies on every GPU and records the loaded ROCm libraries before model loading.
It uses the same isolated output, runtime storage and owned-container cleanup as normal runs.
Set `temporary_memory_filesystem: true` inside `image_check` to test the compiler
temporary mount used by execution. Normal runs use a 64 GiB maximum tmpfs at
`/work/tmp` path because the earlier runner NFS rejected colons in COMGR
intermediate filenames. This mount is temporary RAM-backed storage with a
64 GiB limit, allocated on demand. Persistent runtime files use the run's
`work/` directory, mounted at `/work`; retained evidence is mounted at `/output`.
BF16 GPU math is checked before client/model startup. The container root is
read-only. Compiler caches and AITER's fallback home directory live under `/work`.

IPAT prepares and saves the request through its chat tools. The request includes
argument arrays, environment, exact client package versions, model-cache paths,
timeouts, and the profiling window. No shell is used to interpret commands.
Model caches are mounted read-only; generated files stay in this run's shared
data directory. Direct local invocations may still use an output inside the
team checkout. Code changes stay inside the two repositories. Cleanup targets
only the run's own Docker container.
An occupied GPU container causes the request to fail rather than interrupt it.

The `ipat-<request-id>` Actions artifact contains the request, effective recipe,
manifests, image identity, model metadata hashes, logs, raw client output,
serving metrics, native traces, `report.md`, and a browsing `index.html`.
Package environments and caches remain in `work/` but are excluded from the
artifact and ownership repair. Automatic cache deletion is not implemented;
remove a completed attempt's `work/` only after its owned containers have stopped.
The UI data directory and existing collected evidence are not migrated.
Reports explicitly identify missing quality, HBM counters,
roofline and full tensor-flow evidence.

Local checks (set `TMPDIR` to a directory inside one of the two repositories):

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=ipat python3 -m unittest ipat/test_runner.py
```

## Aggregated kernel reports

`kernel_report.py` streams rocprof kernel CSVs or Torch Chrome traces. It exports
all kernel symbols ordered by summed GPU duration share, calls, mean/min/max/std,
per-device/shape variants, per-dispatch and explicit-iteration compressed CSVs,
and HTML/Markdown/Excel reports. The report is generated after profile shutdown.
Top-ten rooflines and best-among-tested-path verdicts require separately supplied
`kernel-evidence.json`, bound to the trace/variant and hashed evidence files.
Without counters, runtime tensor/source proof or compatible candidate benchmarks,
those sections stay unavailable and `top10-collection-plan.json` lists follow-ups.
See IPAT's `docs/KERNEL_REPORT.md` for the schema and interpretation limits.

`rocprof_capture.py` adds the launch wrapper from the user-supplied profiling
archive: kernel trace + stats, CSV, and a bounded collection period. New AgentX
preparations use launch mode; legacy attach and Torch configs remain supported.
The baseline observes the measured-phase marker offset from server launch. Its
offset plus `delay_seconds` calibrates a separate profile pass; an explicit
`launch_delay_seconds` can override it. The fresh profile marker must arrive before
the scheduled window or the attempt fails. `hip_trace`/`marker_trace` optionally
add domains, and capture is bounded to 1–180 seconds. Do not copy C96's 3300-second
delay to another workload. Capture timing, flush and device coverage require live
verification. Generated counter commands are templates, not automatic GPU jobs.

The workflow exposes Phase 2 replication and Phase 3 profiling as separate steps.
Both consume the same request; the profile stage verifies the preceding baseline
identity and retains its results before calibrating the bounded rocprof window.
Preflight image checks can explicitly pull the requested pinned image and record
GPU math, compiler-cache scope and profiler capabilities before model startup.

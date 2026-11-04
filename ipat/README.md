# IPAT runner

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

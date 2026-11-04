# IPAT runner

`e2e-tests.yml` accepts an optional `ipat-request` JSON input on branch `IPAT`.
It runs a single frozen request on the selected MI355X runner, in a dedicated
checkout. Existing sweep dispatches keep their original path.

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
It uses the same isolated output and owned-container cleanup as normal runs.
Set `temporary_memory_filesystem: true` inside `image_check` to test the compiler
temporary mount used by execution. Normal runs use a 64 GiB maximum tmpfs at
the existing repository-mounted `/output/tmp` path because the runner's NFS
rejects colons in COMGR intermediate filenames. Persistent evidence and caches
remain in the checkout. BF16 GPU math is checked before client/model startup.
The container root is read-only. Compiler caches are redirected under `/output`;
AITER uses its existing fallback to the repository-scoped home directory when
the installed package is read-only. This enforces the repository-only write scope.

IPAT prepares and saves the request through its chat tools. The request includes
argument arrays, environment, exact client package versions, model-cache paths,
timeouts, and the profiling window. No shell is used to interpret commands.
Model caches are mounted read-only; generated files stay under `evidence/`
inside this team checkout. Cleanup targets only the run's own Docker container.
An occupied GPU container causes the request to fail rather than interrupt it.

The `ipat-<request-id>` Actions artifact contains the request, effective recipe,
manifests, image identity, model metadata hashes, logs, raw client output,
serving metrics, native traces, `report.md`, and a browsing `index.html`.
Package environments and caches remain in the checkout but are excluded from
the artifact and ownership repair, avoiding NFS quota charges for generated caches.
Reports explicitly identify missing quality, HBM counters,
roofline and full tensor-flow evidence.

Local checks (set `TMPDIR` to a directory inside one of the two repositories):

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=ipat python3 -m unittest ipat/test_runner.py
```

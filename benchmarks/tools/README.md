# Benchmark diagnostics and kernel reports

`runner.py` executes an explicitly supplied custom benchmark or bounded diagnostic
request. Published SA datapoints use the independent native entrypoint in
[`../sa/`](../sa/); this Docker-based executor does not establish SA workflow
replication.

Use the `benchmark-request` input of `e2e-tests.yml` for diagnostics. Data stays
under `/share_nfs/ajith/benchmarks/`; owned containers and run-specific work
folders keep diagnostic files separate from model caches.

`kernel_report.py` aggregates native rocprof kernel CSVs and Torch traces into
all-kernel timing statistics, per-dispatch and marked-iteration exports, and
HTML/Markdown/Excel reports. Rooflines and optimized-path verdicts require matched
counter, tensor/source and candidate-run evidence. Missing evidence stays explicit.

Run focused local checks with `PYTHONPATH=benchmarks/tools python3 -m unittest
discover -s benchmarks/tools` and a temporary directory inside the checkout.

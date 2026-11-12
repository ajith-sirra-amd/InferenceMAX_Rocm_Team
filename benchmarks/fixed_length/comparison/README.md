# Compare fixed-length serving runs

Use the native `run_benchmark_serving` result saved with `--save-detailed`:

```bash
python3 benchmarks/fixed_length/comparison/compare.py \
  /share_nfs/ajith/benchmarks/BASELINE/baseline.json \
  /share_nfs/ajith/benchmarks/CANDIDATE/baseline.json \
  --output /share_nfs/ajith/benchmarks/CANDIDATE/comparison.json
```

Both runs must complete **192 measured requests, 8192 input tokens, 1024 output
tokens, concurrency 48, and zero errors**. The tool rejects mismatched exported
backend, model, tokenizer, best-of, request-rate or burstiness fields. Use the
native benchmark output, not the launcher's `recipe.json`.

The primary result is the change in native **mean request E2E**. Request throughput
and output tokens/second are recomputed from validated totals and duration.
`improvement_percent` is positive for lower latency or higher throughput. These
metrics can disagree; retain both. The tool evaluates neither statistical
significance nor semantic output quality.

The result initially remains `screening_only`: native JSON lacks prompt identity
and complete runtime provenance. Optionally pass `--baseline-prompts` and
`--candidate-prompts`, each a JSON array containing **all 192 actual measured
prompt strings in request order**. The comparator hashes their exact ordered
content and rejects mismatches or incomplete arrays. Four preflight prompts,
matching lengths, or matching seeds are insufficient. Independently establish
that supplied corpora were the inputs actually sent; do not substitute a
reconstruction unless its sampler revision, tokenizer, seed, worker count and
generation path have been verified.

Before accepting a performance claim, review the saved server/client commands,
image digest, model/draft/tokenizer revisions, graph mode, warmup completion,
profiler-disabled state, GPU host/settings, and absence of concurrent load. Keep
these fixed apart from the intended optimization. The native JSON cannot verify
all of them. In particular, warmup results and whether output lengths came from
server usage or retokenization are not exported by this client.

Start with at least **three alternating matched baseline/candidate pairs** to
measure run variability. For a **2–5%** claim, plan **5–10 or more independent
pairs**, balance AB/BA order, and adjust the count to the observed run variance.
No fixed count guarantees adequate precision. Treat complete runs as the
experimental units: the 192 concurrent requests are correlated and must not be
used as 192 independent replicates. This tool intentionally reports no p-values
or significance interval from their unpaired samples.

Run the behavior checks with:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover \
  -s benchmarks/fixed_length/comparison -p 'test_*.py'
```

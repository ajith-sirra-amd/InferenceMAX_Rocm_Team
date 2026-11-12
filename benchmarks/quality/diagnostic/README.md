# Short natural-verifier diagnostic

Prepared only. No dispatch or installed source change has occurred.

The failed baseline's responses are corrupted: all 48 contain repeated `up`
strings, usually six between target words. Forty-one responses contain at least
2000 literal `up` occurrences. The saved counter delta is truly 84363 accepted
draft tokens / 14065 verifications (AL 6.99808); only 27 of 84390 proposed tokens
were rejected. This does not establish useful natural acceptance.

Run on an otherwise idle authorized MI355x runner, using the team checkout with
the existing `benchmarks/quality` helpers. Copy this directory into that checkout
when publishing the proposal, then use:

```bash
bash benchmarks/quality/diagnostic/run.sh "$PWD" \
  /share_nfs/ajith/benchmarks/natural-diagnostic-<run>-<attempt>
```

The command runs a separate baseline server with the existing natural recipe,
same pinned image/weights/core settings, block verifier and graph execution.
It uses existing quality recipe construction, source/config audits, prompt assets
and response reader. Instrumentation is enabled through an isolated
`PYTHONPATH`/`sitecustomize`; installed model/runtime files are not changed.
The only added telemetry setting is `VLLM_COMPUTE_NANS_IN_LOGITS=1`.

Three sequential scenarios: C1 temperature0, C1 temperature1, then C48
temperature1, each at max128 output tokens, with API token IDs requested.
Sampling seeds and the ordinary GSM8K question prompts are retained. Output
truncation is expected at this limit and is not scored as a quality result.

Each worker saves its first two live proposal calls and native sampling calls
per scenario. The first two verification calls are recorded where native
sampling ownership executes them; C1 may have only one owner. C48 should expose
all eight owners. Saved JSON includes selected IDs, input/position mappings,
cache row ownership, raw/processed target and draft logit finite/NaN/Inf counts,
finite counts per row, selected-token draft logits, sampler classes, final IDs
and acceptance counts. Full logits are never copied to CPU or saved.

The hooks return every original output without alteration, leave CUDA graphs
enabled and do not perform an extra eager replay. GPU synchronization changes
timing: **this run cannot measure performance**. It can identify whether bad
IDs already exist in graph proposal/cache output, are introduced by verification,
or coincide with invalid target/draft logits or ownership.

Bounds: overall2400s +120s forced cleanup; readiness1800s; client360s;
server cleanup75s; owned Docker stop/remove/chown bounded. Artifacts stay in a
fresh shared directory. Upload all files except `work/**`.

Source SHA-256 values of the imported sampler, graph, draft and Kimi modules,
helper hashes, unchanged natural recipe and effective server argv/env are saved.
No synthetic fallback or model-specific fused Markov fastpath was found in the
static source chain. The default registry uses stock `K3DSparkForCausalLM`; the
fused fastpath requires `decode_backend=latency_specialized`, absent here.

Local validation is syntax and CPU structural checking only. Actual hook
imports have a separate90s CPU preflight inside the pinned image before model
startup. The existing evidence did not include logits or token IDs, so the
specific corruption mechanism remains unproven until this diagnostic runs.

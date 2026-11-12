# Short natural-verifier diagnostic

Diagnostic workflow; installed model/runtime source is not changed.

Current status (12 November): run `43254445132` completed, collected 192 worker
records and saved 50 responses without request errors. Execution is paused by
user instruction: do not use a GPU or runner until explicitly reauthorized.
The commands below document the existing helper; they are not a pending run.

The failed baseline's responses are corrupted: all 48 contain repeated `up`
strings, usually six between target words. Forty-one responses contain at least
2000 literal `up` occurrences. The saved counter delta is truly 84363 accepted
draft tokens / 14065 verifications (AL 6.99808); only 27 of 84390 proposed tokens
were rejected. This does not establish useful natural acceptance.

Run on an otherwise idle authorized MI355x runner, using the team checkout with
the existing `benchmarks/quality` helpers. In `e2e-tests.yml`, choose
`Natural verifier diagnostic Kimi baseline` for `generate-cli-command`.
The same helper can be invoked directly from the checkout:

```bash
bash benchmarks/quality/diagnostic/run.sh "$PWD" \
  /share_nfs/ajith/benchmarks/natural-diagnostic-<run>-<attempt>
```

The command runs a separate baseline server with the existing natural recipe,
same pinned image/weights/core settings, block verifier and graph execution.
It uses existing quality recipe construction, source/config audits, prompt assets
and response reader. Instrumentation adds
`--worker-extension-cls worker_extension.DiagnosticWorkerExtension` and makes that
module available through an isolated `PYTHONPATH`. The native worker initialization
hook installs observations after device/model-runner initialization and before
model loading or graph capture. Module import has no installation side effects;
client, Python startup, and helper subprocesses do not install hooks.
The additional telemetry setting is `VLLM_COMPUTE_NANS_IN_LOGITS=1`.

`natural-recipe.json` preserves the unchanged natural core configuration;
`recipe.json` and `effective-server.json` record the instrumented command. The
full instrumented command is explicitly marked as different from the natural
recipe and SA throughput recipe. The core model/server settings still match the
natural baseline.

Three sequential scenarios: C1 temperature0, C1 temperature1, then C48
temperature1, each at max16 output tokens, with API token IDs requested.
Sampling seeds and the ordinary GSM8K question prompts are retained. Output
truncation is expected at this limit and is not scored as a quality result.

Each worker saves its first two actual draft proposal calls and native sampling calls
per scenario. Context-only calls that request zero speculative tokens are skipped
because their returned draft buffer may be stale. The first two verification calls are recorded where native
sampling ownership executes them. Ownership and actual live batch size must be
read from the captured records; a C48 client label proves neither. Saved JSON
includes selected IDs, input/position mappings,
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
imports and explicit installation have a separate90s CPU preflight inside the
pinned image after Python startup and before model startup. The preflight uses
unbuffered progress markers and repeats a Python stack dump every30s until done.
An earlier attempt (`43237433849`) timed out in this preflight with an empty log
while eager `sitecustomize` imports were enabled. The revised helper removes
`sitecustomize` entirely; the exact cause of that import stall was not established.
Retry `43239357102` passed hook installation and collected 128 C1 worker records,
then exhausted the 360-second client budget. The greedy request produced 98 tokens
before its 300-second HTTP timeout; C48 did not start. All eight workers observed
all-NaN draft-cache values with finite raw target logits. Sampled block verification
accepted six repeated draft IDs; greedy verification rejected them. The exact
operation that first produces NaNs remains unresolved. The initial finite-cache
snapshot was context-only and is not evidence of successful eager drafting.

The completed 16-token revision caps requests at 16 output tokens, writes each response when
it completes, and reports readiness/stage progress to the workflow log. Stage
markers use the container's private `/tmp` mount so repeated hook checks do not
read the shared filesystem. Slow-generation cause has not been established.
Mounted target and draft `config.json` files are copied into the evidence directory.


For the first two actual C1 proposals, additional per-row finite/NaN/+Inf/-Inf
counts are retained in the native draft graph and read after the selected replay.
The observed boundaries include target auxiliaries, staged context, context
projection/norm/KV projection, draft-layer attention/MLP/norm, shared-head logits,
Markov projection and summed sampling logits. Live and padded rows are labelled
separately. Only native graphs of at most two requests are instrumented internally;
larger graphs, including C48, retain outer proposal/verifier observations and
explicitly report unavailable inner coverage. No eager rerun is substituted.
A C1 collection fails validation if those graph observations are missing.
These observations change timing and memory lifetime and do not prove numerical
correctness. The first invalid observed boundary may require a narrower follow-up
to identify the exact operation.

## Latest collected boundary and next evidence

Run `43254445132` used source `19d35942380cddac5981b11cf57c4fc6f270ae7b`.
All 48 captured proposal records first show nonfinite live output at
`draft.layer.0.attention.output`: six live rows × 7,168 values are NaN,
with finite attention inputs and finite padded output rows. The next draft
cache has 983,040 NaNs; raw target logits remain finite. This boundary includes
output projection and is before the following fused all-reduce/RMSNorm. The
exact internal operation is unresolved; the collection does not prove an AITER
kernel bug or a valid natural baseline. All responses hit the 16-token cap.

The C48 client submitted 48 requests within 16.701 ms. Its first two hooks per
worker ran 0.080–0.305 s after submission, before the first request finished at
0.490 s. The server later logged 47 running requests. Captured proposal and
verification batches each contain one request, padded to two. The capture
budget expired early; actual C48 decode execution is still missing.

The proposed follow-up separates capture budgets by actual live batch and decode
state, and adds bounded observations inside the first draft attention module:
Q/KV projection and norm; Q expansion/FP8/RoPE; DCP query gather; per-rank attention
partial output and LSE; DCP merge; value BMM; final output projection. Bind the
actual weights, scales, referenced cache slots and metadata. Larger-graph inner
observations and a longer C48 client cap are proposals, not implemented changes.
Preserve native graph execution and all original outputs. This remains a
diagnostic, separate from quality and unprofiled performance comparisons.

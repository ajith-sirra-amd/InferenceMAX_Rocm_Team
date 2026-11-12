# Short Kimi optimization benchmarks

Use these component tests before committing GPU time to a full agentic
comparison. They run synthetic tensors through the pinned image's native
implementations and isolated candidates. They do not load model weights,
change installed packages or prove per-request E2E improvement.

Use the existing `e2e-tests.yml` workflow on the research branch:

- `sa-config`: `benchmarks/sa/kimi-k3.1-pro-vllm-mi355x-c48-sa42638222682.json`
- `sa-mode`: `transport-check`
- `allocation-id`: empty for direct execution on the existing runner
- `generate-cli-command`: `Single benchmark`
- `test-name`: a descriptive component name
- `sa-check-command`: a JSON argv array, as below

The transport exposes the scripts read-only at `/sa-tools/profile`, saves
results under `/logs/native`, and enforces a 600-second command deadline.
Run one comparison at a time on a host to avoid competing for its GPUs.
Preserve the transport's temporary/cache paths.

| Script | Scope | Output argument |
|---|---|---|
| `optimization_sparse.py` | Direct page lookup and masked packing; one GPU with eight rank values | `/logs/native/optimization-sparse/result.json` |
| `optimization_attnres.py` | Cached constant products; BF16/FP8 output and mutation checks | `/logs/native/optimization-attnres` |
| `optimization_moe.py` | Native FP4 GEMM1/GEMM2 tile variants, full pair including sort/quantization | `/logs/native/optimization-moe` |
| `optimization_dspark.py` | Native TP8 projection, output gather and RMSNorm on eight GPUs | `/logs/native/optimization-dspark` |

Example sparse command:

```json
["python3", "/sa-tools/profile/optimization_sparse.py", "--output", "/logs/native/optimization-sparse/result.json", "--max-seconds", "480"]
```

The transport configuration contains orchestration environment values; it does
not itself install the engine-role environment from the native recipe. For a
comparison using the serving dispatch, prepend `env` and **every frozen
`roles.agg.env` assignment** from the selected `recipe.lock.yaml` to the argv,
before `python3`. Retain that full dispatch request as evidence. Do not silently
substitute the current branch's latest recipe for the measured recipe.

The DSpark probe initializes the saved effective compilation mode NONE with
FULL_AND_PIECEWISE graphs and installs native platform kernel priorities.
It still constructs component layers directly, so model constructor guards,
checkpoint loading, natural draft acceptance and complete serving graphs need
separate validation.

Follow each run until terminal, download `sa-benchmark-<run-id>`, and inspect
`result.json` plus logs. A completed MoE experiment means evidence was saved;
check `all_numerical_and_graph_smoke_checks_pass` separately. Keep numerical
differences, fallback warnings, failed candidates and all timing samples.
Event-time improvements are operation results, not request-latency estimates.

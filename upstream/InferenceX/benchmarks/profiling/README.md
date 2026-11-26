# Measured Kimi profiling

**English** | [中文](README_zh.md)

The direct [Kimi C48 script](../single_node/agentic/kimik3_fp4_mi355x_mtp.sh)
accepts **`--profile`**. Profiling is off when this flag is omitted. It uses
the pinned vLLM image's native Torch profiler on all eight workers. The SA
model, precision, graph settings, warmup and 3600-second AgentX replay remain
unchanged. Keep profiled measurements separate from throughput baselines.

Supply the existing server/model/client environment first, including local
snapshots mounted under `/data`, a fresh writable `RESULT_DIR`, and a prepared
`SA_CLIENT_ENTRYPOINT` matching the selected SA source. From the team repo root:

```bash
TEST=0 PROFILE=0 DURATION=3600 \
PROFILE_DELAY_SECONDS=1800 \
PROFILE_SECONDS=130 \
PROFILE_MIN_SECONDS=120 \
PROFILE_PHASE_TIMEOUT_SECONDS=7200 \
PROFILE_POLL_SECONDS=2 \
PROFILE_EXPORT_TIMEOUT_SECONDS=600 \
bash upstream/InferenceX/benchmarks/single_node/agentic/kimik3_fp4_mi355x_mtp.sh --profile
```

All six timing inputs are required. The example waits 1800 seconds **after
AgentX's measured-phase timestamp**, captures for 130 seconds, and requires
at least 120 seconds of common GPU trace extent. It does not time capture from
model startup. `PROFILE_PHASE_TIMEOUT_SECONDS` bounds waiting for measurement
after the client launches; it does not change the server startup timeout.
`PROFILE_EXPORT_TIMEOUT_SECONDS` bounds each start/stop API acknowledgement.
The capture must be at most 180 seconds, and delay, capture, both API budgets,
polling and 90 seconds of health/metrics budget must fit inside `DURATION`.
Invalid settings and reused profile directories fail before GPU startup.

This option supports native AgentX (`TEST=0`). It rejects `TEST=1` and the
legacy `PROFILE=1` mechanism, which changes fixed-length request counts.
The client must write its native `agentic/benchmark.log` and
`agentic/agentic_power_timezone_offset.txt` under `RESULT_DIR`. The controller
requires measured-phase activity and generated-token progress, then calls
`/start_profile` and `/stop_profile`. The server stays alive through export,
including a bounded stop/export attempt on cancellation. No CPU pinning or
second GPU profiler is added. Shapes, stacks and memory tracing are disabled
for this timing pass; HBM counters require a separate collection.

Outputs under `RESULT_DIR`:

| File | Contents |
| --- | --- |
| `vllm_core_command.txt` | SA command before the diagnostic profiler overlay; checked by the optional SA audit |
| `vllm_command.txt` | Actual server command including `--profiler-config` |
| `profiles/torch/settings.json`, `profiler-config.json` | Requested capture settings and native profiler configuration |
| `profiles/torch/controller.log`, `profile-control.json` | Phase detection, token progress, start/stop times, export state and errors |
| `profiles/torch/*.trace.json.gz` | Complete worker traces, including nested native output directories |
| `profile-validation.json` | Eight-worker coverage and actual capture containment within measured sending |
| `profiles/torch/kernels.csv`, `top10-kernels.csv` | All symbols and the top ten by summed GPU duration, with calls, mean/min/max duration and GPU-time share |

Validation fully parses the traces, requires devices 0–7 and nonzero kernels,
and checks that files remain unchanged while read. A missing worker, incomplete
export or capture outside measured sending fails the run. CSV summaries are
written only after validation passes. GPU-time shares sum overlapping worker
activity; they are not end-to-end latency percentages or proof of optimization
benefit. Keep sufficient writable space for both raw export temporaries and
compressed traces, and preserve failed attempts in their original directories.

Local validation uses mocked HTTP/GPU/client collaborators and the real shell,
controller and validator:

```bash
python3 -m unittest discover -s upstream/InferenceX/benchmarks/profiling -p 'test_*.py'
```

This verifies control flow and artifacts. A fresh full-model capture on the
selected runner is still required to establish real GPU coverage.

# Kimi benchmarks and profiling

**English** | [中文](README_zh.md)

The [Kimi script](../single_node/agentic/kimik3_fp4_mi355x_mtp.sh) supports three
workloads. Each works with `PROFILE=0` (default) or `PROFILE=1`.
`--profile` and `--no-profile` override the environment flag.

| `BENCHMARK_MODE` | Workload | Legacy selection |
| --- | --- | --- |
| `serving` | `run_benchmark_serving`, bundled random-request client | `TEST=1` |
| `vllm` | Native `vllm bench serve`, `prefix_repetition` dataset | Set `BENCHMARK_MODE=vllm` |
| `agentic` | `run_agentic_replay_and_write_outputs`, trace replay and result validation | `TEST=0`, also the default |

`BENCHMARK_MODE` takes precedence over `TEST`. All modes use the same SA-derived
server recipe and enable server prefix caching. Profiling preserves prompt
count, concurrency, warmup and replay duration. The old shared helper's
`PROFILE=1` behavior, which caps requests to concurrency, is bypassed by this
script's independent controller.

## Defaults and overrides

These defaults were explicitly requested for this direct script. Environment
values override them; empty or invalid required inputs fail validation.

| Setting | `serving` | `vllm` | `agentic` |
| --- | --- | --- | --- |
| `TEST_ISL` (random input / unique suffix tokens) | 8192 | 3584 | Trace-defined; ignored |
| `TEST_PREFIX_LEN` (shared prefix tokens) | 0 | 4608 | Trace-defined; ignored |
| `TEST_OSL` (output tokens) | 1024 | 1024 | Trace-defined; ignored |
| `CONC` (client concurrency) | 48 | 48 | 48 |
| `MAX_NUM_SEQS` (server scheduler limit) | 48 | 48 | 48 |
| `TEST_NUM_PROMPTS` | 192 | 192 | Not used |
| `DURATION` (measured workload seconds) | Not used; request-count based | Not used; request-count based | 3600 |
| `PROFILE_DELAY_SECONDS` | 10 | 10 | 1800 |
| `PROFILE_SECONDS` (capture seconds) | 10 | 10 | 130 |
| `PROFILE_EXPORT_TIMEOUT_SECONDS` (each start/stop acknowledgement) | 120 | 120 | 600 |

All modes default to `PROFILE_PHASE_TIMEOUT_SECONDS=7200`,
`PROFILE_POLL_SECONDS=0.5`, and `PROFILE_MIN_SECONDS=0.8 * PROFILE_SECONDS`.
`TEST_RANGE_RATIO=1` makes the bundled client's lengths fixed.
`VLLM_NUM_PREFIXES=1` selects one shared prefix in the native vLLM dataset;
raise it to create multiple prefix groups. Native vLLM runs the installed
`vllm bench serve` executable, with `/opt/runtime-venv/bin` first on `PATH`.

Client concurrency and server `MAX_NUM_SEQS` are independent. Changing either
from 48 creates a new operating point; it does not reproduce SA C48. These
profiling defaults are starting values, not measured optimal settings.

## Examples

Supply the server/model environment first: `MODEL`, `MODEL_PATH`,
`DRAFT_MODEL_PATH`, `TP=8`, `DCP_SIZE=8`, `EP_SIZE=1`, `KV_OFFLOADING=dram`,
`KV_OFFLOAD_BACKEND=vllm-simple`, `TOTAL_CPU_DRAM_GB=1799`, `EVAL_ONLY=false`,
`PORT`, `RESULT_FILENAME` and a fresh writable `RESULT_DIR`. Mount the model
snapshots under `/data`; use the matching image documented in the team's
[SA reference guide](../../../../benchmarks/agentic/README.md).

From the team repository root, with those inputs exported:

```bash
SCRIPT=upstream/InferenceX/benchmarks/single_node/agentic/kimik3_fp4_mi355x_mtp.sh

# Bundled serving client: ordinary random prompts, with or without capture.
BENCHMARK_MODE=serving PROFILE=0 RESULT_DIR=/data/results/serving-plain bash "$SCRIPT"
BENCHMARK_MODE=serving PROFILE=1 RESULT_DIR=/data/results/serving-profile bash "$SCRIPT"

# Bundled client with a shared prefix: 3584 + 4608 = 8192 input tokens.
BENCHMARK_MODE=serving TEST_ISL=3584 TEST_PREFIX_LEN=4608 PROFILE=0 \
RESULT_DIR=/data/results/shared-plain bash "$SCRIPT"
BENCHMARK_MODE=serving TEST_ISL=3584 TEST_PREFIX_LEN=4608 PROFILE=1 \
RESULT_DIR=/data/results/shared-profile bash "$SCRIPT"

# Native vLLM prefix_repetition: defaults target the same 8192-token total.
BENCHMARK_MODE=vllm PROFILE=0 RESULT_DIR=/data/results/vllm-plain bash "$SCRIPT"
BENCHMARK_MODE=vllm PROFILE=1 RESULT_DIR=/data/results/vllm-profile bash "$SCRIPT"

# Agentic trace replay, after the additional setup below.
BENCHMARK_MODE=agentic PROFILE=0 RESULT_DIR=/data/results/agentic-plain bash "$SCRIPT"
BENCHMARK_MODE=agentic PROFILE=1 RESULT_DIR=/data/results/agentic-profile bash "$SCRIPT"

# Example override: low concurrency, independent scheduler cap and capture timing.
BENCHMARK_MODE=serving CONC=8 MAX_NUM_SEQS=64 TEST_ISL=4096 TEST_OSL=512 \
PROFILE=1 PROFILE_DELAY_SECONDS=5 PROFILE_SECONDS=5 \
RESULT_DIR=/data/results/serving-c8-profile bash "$SCRIPT"
```

The shared prefix is **added to** `TEST_ISL`, not included in it. The bundled
client applies the chat template; native vLLM's `prefix_repetition` is a
completion workload. Tokenization and prefix-block alignment can affect actual
cache reuse. This is a prefix-caching benchmark, not an exact cache-hit target
or a reproduction of AA's complete workload.

For agentic mode, source `upstream/InferenceX/benchmarks/runtime_settings.sh`
**before your overrides**, and export `INFMAX_CONTAINER_WORKSPACE` pointing to
this checkout's `upstream/InferenceX`. Export `MODEL_PREFIX=kimik3.1` and
`FRAMEWORK=vllm` for trace selection and result metadata. Supply `IS_MULTINODE=false`,
`REQUIRE_POWER=0`, `ENABLE_AGENTX_POWER=0`, `AIPERF_EXPERIMENTAL_FAST=0` and
`AIPERF_RUNTIME_DIR` pointing to a writable client-runtime directory. Set
`AGENTIC_OUTPUT_DIR` to the desired aggregate-output directory. Existing
`resolve_trace_source` / `build_replay_cmd` helpers install their client and
prepare the dataset before GPU startup. `uv`, dataset access and enough disk
space are required. This uses the **checked-out shared helper's client**, not
the exact latest measured SA AgentX package set. `SA_CLIENT_ENTRYPOINT` is no
longer used by this direct script. Historical comparison launchers keep their
archived script and pinned client.

## Capture timing and validation

The delay starts at the **measured workload phase after warmup**, not at server
launch. It is not a client startup delay. `PROFILE_SECONDS` bounds capture;
it does not set total synthetic benchmark runtime. Synthetic benchmarks finish
all `TEST_NUM_PROMPTS`; agentic replay uses `DURATION`. The phase timeout starts
after client launch and does not change the server startup timeout.

The bundled client emits an atomic measurement timestamp. Native vLLM runs
directly, with unbuffered output saved to `vllm-benchmark.log` during profiling.
Its delay starts when the controller observes `Starting main benchmark run...`
after warmup, so it has log-polling precision. Final validation uses the native
JSON's detailed `start_times` and `latencies`, converted from the same host's
performance-counter clock to wall time. There is no Python client wrapper,
monkey-patching or client `--profile` flag. Agentic mode uses its native
measured-phase marker and completed phase records.

The controller confirms generated-token progress, calls `/start_profile`, waits
for the capture window, then calls `/stop_profile`. Capture must be positive,
at most 180 seconds, and no shorter than `PROFILE_MIN_SECONDS`; polling must be
shorter than the delay. Agentic delay, capture, both API budgets, polling and
90 seconds of health/metrics budget must fit inside its duration. For synthetic
runs, choose enough prompts to cover the window. If the workload ends too soon,
the run fails visibly; it does not reduce requests, repeat the workload or
silently shorten the capture. Use fresh output directories.

The server stays alive through export, including a bounded export attempt on
cancellation. Shapes, stacks and memory tracing are disabled. Keep profiled
results separate from unprofiled throughput comparisons.

| Output under `RESULT_DIR` | Contents |
| --- | --- |
| `vllm_core_command.txt` | Server command before profiler overlay |
| `vllm_command.txt` | Actual server command |
| `<RESULT_FILENAME>-command.txt` or `agentic/benchmark_command.txt` | Selected client command |
| `profiles/torch/settings.json`, `profiler-config.json` | Capture settings and profiler configuration |
| `profiles/torch/measurement-start.json` | Bundled serving client's measurement marker |
| `vllm-benchmark.log` | Native vLLM progress/output when profiling |
| `profiles/torch/controller.log`, `profile-control.json` | Phase, progress, start/stop and export state |
| `profiles/torch/*.trace.json.gz` | Worker traces, possibly in nested native output directories |
| `profile-validation.json` | Eight-worker coverage and measured-interval containment |
| `profiles/torch/kernels.csv`, `top10-kernels.csv` | Ranked symbols, calls, timing and GPU-time share |

Validation fully parses stable exports, requires devices 0–7 and nonzero kernels,
and rejects incomplete traces or capture outside measured activity. CSV summaries
are written only after validation passes. GPU-time shares sum overlapping worker
activity; they are not end-to-end latency shares or optimization gains. HBM
counters require a separate collection.

CPU-only checks (GPU/HTTP/client dependencies stubbed in the shell suite):

```bash
python3 -m unittest discover -s upstream/InferenceX/benchmarks/profiling -p 'test_*.py'
PYTHONPATH=upstream/InferenceX python3 -m pytest upstream/InferenceX/utils/test_benchmark_serving.py
```

These verify control flow, default/override forwarding, marker placement and
artifacts. No GPU run or new performance measurement accompanies this update.

# Kimi profiling handoff: commands, timing and recovery

Updated: 2026-11-11. Documentation only; runs remain stopped.

Use the phase-controlled vLLM profiler for the current native Kimi recipe. The archived rocprof wrapper is useful for kernel CSV collection, but its timer starts with the server and its reports depend on process shutdown.

## Source and scope

Instructions below refer to InferenceMAX revision [275f4eb8](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/tree/275f4eb8c718f0ba653f45e11e5049727808e515):

- [Active Kimi recipe: profiler configuration](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/blob/275f4eb8c718f0ba653f45e11e5049727808e515/upstream/InferenceX/benchmarks/single_node/agentic/kimik3_fp4_mi355x_mtp.sh#L434).
- [Controller launch](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/blob/275f4eb8c718f0ba653f45e11e5049727808e515/upstream/InferenceX/benchmarks/single_node/agentic/kimik3_fp4_mi355x_mtp.sh#L550).
- [Controller implementation](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/blob/275f4eb8c718f0ba653f45e11e5049727808e515/utils/atom_attention/native_steady_profile.py).
- [Workflow timing and artifact upload](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/blob/275f4eb8c718f0ba653f45e11e5049727808e515/.github/workflows/benchmark-tmpl.yml#L148).

Before any future dispatch, confirm authorization, reservation/cutoff and the exact source/image combination. Keep GMU 0.90, load format auto, weight quantization, graphs, model and CCD binding fixed. This is Kimi-K3 guidance; K3.1 requires separate attention-path validation.

## Current setup: measured-phase capture

In the root e2e-tests.yml workflow, select the existing Kimi C48 + MTP configuration and supply:

| Input | Value |
| --- | --- |
| generate-cli-command | test-config --config-files upstream/InferenceX/configs/amd-master.yaml --config-keys kimik3-fp4-mi355x-vllm-agentic-mtp |
| ref | Exact reviewed InferenceMAX source revision |
| native-vllm-sha | Exact vLLM commit matched to installed binaries |
| native-comparison | native |
| native-steady-profile | true |
| native-fp8-enabled | 1 |
| Replay duration | 3600 seconds; evaluation and path-probe modes disabled |

**Current limitation:** the active recipe explicitly rejects FP8-OFF plus profiling. It also rejects profiling with shortened replay, evaluation, probes or reference-comparison mode. A matched OFF/ON profile needs a reviewed harness change; changing the flag alone will fail. The ordinary unprofiled OFF control remains separate.

The workflow supplies these values; reproduce them explicitly if maintaining the recipe:

~~~bash
export NATIVE_STEADY_PROFILE=1
export NATIVE_PROFILE_SETTLE_SECONDS=300
export NATIVE_PROFILE_CAPTURE_SECONDS=20
export NATIVE_PROFILE_TIMEOUT_SECONDS=7200
export NATIVE_PROFILE_POLL_SECONDS=5
~~~

The following is the profiling portion of the existing recipe, not a complete standalone model launcher. VLLM_CMD, RESULT_DIR, PORT, SERVER_PID and VIRTUAL_ENV must come from that recipe:

~~~bash
PROFILE_OUTPUT="$RESULT_DIR/native_steady_profile"
VLLM_CMD+=(--profiler-config "{\"profiler\":\"torch\",\"torch_profiler_dir\":\"$PROFILE_OUTPUT\",\"torch_profiler_activities\":[\"CPU\",\"CUDA\"],\"ignore_frontend\":true,\"torch_profiler_with_stack\":false,\"torch_profiler_record_shapes\":false,\"torch_profiler_with_memory\":false,\"torch_profiler_use_gzip\":true,\"torch_profiler_dump_cuda_time_total\":false}")
~~~

The CUDA activity name is also used by PyTorch's ROCm profiler. This records worker CPU and GPU activity; it does not enable tensor-memory profiling or hardware bandwidth counters.

After server readiness and CCD binding, the recipe starts this controller in the background immediately before replay, then waits for it before final cleanup:

~~~bash
mkdir -p "$PROFILE_OUTPUT"
"$VIRTUAL_ENV/bin/python" /outputs/utils/atom_attention/native_steady_profile.py \
  --repository /outputs --output "$PROFILE_OUTPUT" \
  --benchmark-log "$RESULT_DIR/benchmark.log" --port "$PORT" \
  --server-pid "$SERVER_PID" --prefix native-steady \
  --settle-seconds "$NATIVE_PROFILE_SETTLE_SECONDS" \
  --capture-seconds "$NATIVE_PROFILE_CAPTURE_SECONDS" \
  --timeout-seconds "$NATIVE_PROFILE_TIMEOUT_SECONDS" \
  --poll-seconds "$NATIVE_PROFILE_POLL_SECONDS" \
  > "$PROFILE_OUTPUT/controller.log" 2>&1 &
PROFILE_PID=$!
~~~

The controller recognizes the benchmark line containing "Phase profiling ... started". Here "profiling" is AIPerf's measured phase, not confirmation that GPU tracing has started. It checks that generation tokens increase over the next 300 seconds, checks health, calls POST /start_profile, captures for 20 seconds, and calls POST /stop_profile to export the traces.

## How long to wait

| Milestone | Timing / meaning |
| --- | --- |
| Server loading, compilation, graph capture, warmup | Variable; do not include it in the 300-second settling window. Full replay uses 10 extra warmup requests per lane. |
| Measured-phase detection | Polled every 5 seconds after the controller starts. |
| Settling | 300 seconds after detection, with generation progress required. |
| GPU capture | 20 seconds after /start_profile returns. |
| Trace export | /stop_profile performs export; extra time depends on trace volume. Each controller HTTP request has a 120-second timeout. |
| Local reports | Inspect controller.log, profile-control.json and all worker trace files after export. |

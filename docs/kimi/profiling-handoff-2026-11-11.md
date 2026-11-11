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
| GitHub artifact | Uploaded after the full replay and launch step finish, not immediately at capture completion. |
| Phase-wait deadline | 7200 seconds from controller startup. This is a failure ceiling for finding the marker, not a recommended idle wait or whole-job timeout. |

Poll job progress every two minutes during an authorized run. Check server/warmup progress and the measured-phase marker. If progress stops, investigate rather than waiting for the two-hour ceiling. The current controller writes "waiting" initially and its detailed report at completion/failure; it does not update every intermediate phase in profile-control.json.

A file appearing is not enough: require status=captured, eight nonempty worker traces, successful export and subsequent kernel analysis. If an HTTP request reaches its 120-second timeout, inspect the controller/server error and cleanup; do not claim collection succeeded.

### Observed successful timeline

[Profile run 41478604149](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/41478604149) and [analysis run 41776530466](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/41776530466) both passed. Status and the analysis artifact were checked November 11.

Historical vLLM source: 492080779e9feea19308092b0aba0840e87c6115. This is not validation of the current a03796b8 head or the native-fallback candidate.

| Event on November 2 | UTC |
| --- | --- |
| GPU job started | 16:09:37 |
| Launch step started | 16:11:04 |
| Measured phase began | 16:51:34 |
| Controller observed marker | 16:51:38 |
| Start-profile request returned | 16:56:39 |
| Stop-profile/export request returned | 16:57:33 |
| Profile artifact upload completed | 17:53:51 |

The configured capture was 20 seconds; start-return to stop-return was about 54 seconds, including roughly 34 seconds of stop/export overhead. Reports were available locally about 6 minutes after the measured phase began. GitHub upload completed about 1 hour 44 minutes after GPU job start because the full replay continued.

Eight compressed traces totaled about 446 MB. Analysis found 417,128 kernel events per worker over approximately 20.12 seconds. The CPU analysis workflow took about 3 minutes. These are observed examples, not guaranteed deadlines.

## Earlier problems and the fixes

| Problem encountered | Explanation and solution | State |
| --- | --- | --- |
| Capture started during setup/warmup | A server-relative delay cannot predict loading, graph capture and warmup across configurations. Use the actual measured-phase marker, then a settling window with generation progress. | Implemented and used in the successful capture above. |
| Profiling environment overrides did not take effect | The launcher had already exported timing/warmup values; shell default expressions did not override existing values. The archived recipe used explicit values. The current workflow supplies explicit controller timings. Inspect the resolved command and logs. | Resolved configuration method; do not assume an outer export overrides a hardcoded recipe value. |
| rocprof stopped at "will wait for 2 children to exit" | Terminating only the wrapper left instrumented descendants alive. The archived cleanup signals the captured descendants from leaves to root, then the root, and permits finalization. | Code fix retained in the archive; not a promise every shutdown produces a valid report. |
| Trace export exceeded normal cleanup allowance | Keep the profile controller alive long enough to call /stop_profile and export before stopping vLLM. The controller is stopped before the server with a 125-second grace; historical profiling container allowance was extended to 210 seconds versus 90 normally. | Implemented; interruption/lost-start-response handling has CPU tests. Hard kill still cannot guarantee export. |
| Generic profile workflow used the wrong launch/storage path | The generic profile.yml used an older path and published outside the two authorized repositories. Use native replay integration and InferenceMAX Actions artifacts. | Dedicated path implemented. Do not dispatch generic profile.yml unchanged for this task. |
| Trace files mistaken for GPU coverage | CPU-only, stale, empty or missing-rank traces are insufficient. Use a fresh output directory, reject stale traces and run the analyzer requiring ranks 0-7 and nonzero kernel events. | Capture and analysis succeeded for the historical run above. |
| Output ownership / read-only mounts | The profile controller restores ownership only inside its new output tree. A separate November 6 diagnostic failed when editable installation wrote metadata into a read-only repository; its driver now copies the observer into a writable run directory before installation. | Distinct fixes; the latter was not an FP8 numerical failure. |
| Storage quota before Docker launch | The earlier runner reported "Disk quota exceeded" while copying run-helper.sh, despite free filesystem space. User/project quota can prevent output creation. Verify quota and output writability; the user moved runners. | No quota increase was performed by this session. This error was not established in the successful profile above. |

The later mixed-context diagnostic's TCPStore/FileStore failure is separate from profiling. Its TCP rendezvous fix remains pending; see the [main handoff](pr64288-handoff-2026-11-09.md#latest-failures-and-immediate-next-fix).

## Archived rocprof command and report finalization

[Exact archived command and timing](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/blob/275f4eb8c718f0ba653f45e11e5049727808e515/upstream/InferenceX/benchmarks/single_node/agentic/patches/kimik3_fp4_mi355x_mtp.profiling-archive.sh#L569):

~~~bash
ROCPROF_ENABLE=1
ROCPROF_START_DELAY=3300
ROCPROF_DURATION=60
ROCPROF_DIR="$RESULT_DIR/rocprof"
mkdir -p "$ROCPROF_DIR"
rocprofv3 --kernel-trace --stats \
  -d "$ROCPROF_DIR" -f csv \
  --summary-output-file "$RESULT_DIR/rocprof_summary.txt" \
  -P "${ROCPROF_START_DELAY}:${ROCPROF_DURATION}:1" \
  --collection-period-unit sec \
  -- "${VLLM_CMD[@]}"
~~~

The archived file hardcodes enable=1, delay=3300 and duration=60; its older "off by default" comment does not describe those assignments. This wrapper launches the server under profiling. It is not a command to attach to an already running server.

Capture occurs from server launch +55 minutes through +56 minutes. First verify that this window overlaps steady measured replay. Report files may not finalize until the wrapped server and workers exit. Capture ending does not mean the summary is immediately downloadable.

The [archived cleanup](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/blob/275f4eb8c718f0ba653f45e11e5049727808e515/upstream/InferenceX/benchmarks/single_node/agentic/patches/kimik3_fp4_mi355x_mtp.profiling-archive.sh#L69) allows up to 1200 seconds (20 minutes) after SIGTERM for the summary, polling every 5 seconds while the wrapper is alive, then performs bounded teardown. This is a flush cap, not mandatory waiting time. Its summary-file check alone is insufficient: inspect the actual nonempty kernel CSVs, rank/device coverage, timestamps and clean export.

### Exact shutdown change in the archived wrapper

The fix expanded shutdown from signaling the wrapper alone to signaling its captured process tree before waiting for export. In the existing recipe context, which defines the descendant helper, the core change is:

~~~bash
profiler_children=$(_background_process_descendants "$SERVER_PID")
while read -r profiler_child; do
  [[ -n "$profiler_child" ]] && kill -TERM "$profiler_child" 2>/dev/null || true
done < <(printf '%s\n' "$profiler_children" | tac)
kill -TERM "$SERVER_PID" 2>/dev/null || true

rocprof_deadline=$((SECONDS + 1200))
while [ ! -s "$RESULT_DIR/rocprof_summary.txt" ] \
  && [ $SECONDS -lt $rocprof_deadline ] \
  && kill -0 "$SERVER_PID" 2>/dev/null; do
  sleep 5
done
~~~

This is scoped to the launched server tree. It does not kill unrelated jobs. After the bounded wait, the recipe performs its normal scoped teardown. SIGKILL before export can lose the trace. The current native controller instead calls /stop_profile while the server is still alive; it need not wait for server exit to export.

## Reports and analysis

Current native capture outputs under results/native_steady_profile/:

- profile-control.json: capture status, timestamps, generation progress, file inventory and errors.
- controller.log: controller output/errors.
- Eight rank-specific .pt.trace.json.gz files: CPU/GPU timelines.
- GitHub artifact: native-steady-profile-<run_id>.

The separate CPU analyzer produces profile-analysis.json and kernel-times.csv. Its [workflow and command](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/blob/93b7624d154f38ddfa2e02bcb65654877e2ae11a/.github/workflows/native-profile-analysis.yml) are preserved by the existing diagnostic archive even though the old branch name was retired. Read source at the immutable revision; do not assume that deleted branch is still dispatchable.

Download already completed artifacts from within the authorized repository:

~~~bash
gh run download 41478604149 \
  --repo ajith-sirra-amd/InferenceMAX_Rocm_Team \
  --name native-steady-profile-41478604149 \
  --dir results/profile-41478604149
gh run download 41776530466 \
  --repo ajith-sirra-amd/InferenceMAX_Rocm_Team \
  --name native-profile-analysis-41478604149 \
  --dir results/profile-analysis-41478604149
~~~

Compute hotspot shares from valid kernel durations and GPU busy time from the union of intervals per device. Summed overlapping kernel durations are not wall-clock GPU utilization. Older rocprof statistics contained a wrapped duration that corrupted the Percentage column; retain the raw evidence and explicitly flag invalid rows before recomputing statistics.

Kernel traces do not establish HBM bandwidth or VRAM occupancy. For a future separate capture, --memory-copy-trace can add copy events; version-appropriate AMD SMI telemetry supplies memory-capacity/utilization samples. Hardware bandwidth counters require a separate supported counter configuration. Do not label copy timing or summed kernel time "memory utilization".

Use profiled throughput only as diagnostic data. Preserve unprofiled matched performance measurements, and compare whole gather/communication/scaling/projection/FMHA/merge pipelines before proposing another optimization.

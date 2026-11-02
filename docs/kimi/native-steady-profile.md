# Native Kimi steady-state profiling

[中文](native-steady-profile_zh.md)

Use the root `e2e-tests.yml` workflow with `native-steady-profile: true`, an exact `native-vllm-sha`, `native-comparison: native` and a full 3,600-second agentic replay. Profiling is off by default and is rejected for evaluation, path probes and reference runs.

The workflow explicitly supplies a 300-second settling interval, a 20-second capture, five-second controller polling and a 7,200-second deadline for reaching the measured phase. The controller starts after server readiness and CCD binding. It waits for the benchmark's measured-phase marker and verifies that the server's generation counter advances during settling.

The vLLM PyTorch profiler records worker CPU and GPU activities. Frontend, stack, shape and memory profiling are disabled. Traces are gzip-compressed and written below `/outputs/results/native_steady_profile`, within this repository. The existing graphs, cached FP8 path, MTP settings and CCD binding remain in use.

The controller stops profiling on normal completion and interruption, records failures in `profile-control.json`, and restores ownership of its output. The recipe waits for the controller and propagates its failure. GitHub uploads traces and the controller log as `native-steady-profile-<run_id>` even when the replay fails.

Inspect the exported events for all eight GPU workers, nonzero kernel duration and overlap with active generation before using the capture to choose an optimization. A captured file alone is not proof of kernel coverage or a steady workload mix. The profiled replay is diagnostic; profiler overhead makes its throughput unsuitable as an unprofiled performance baseline.

Validation: the controller's CPU suite covers phase timing, partial log lines, generation stalls, missing phase, server failure, output containment, SIGTERM cleanup and a lost start response. GPU profiling remains pending until a run and its trace analysis are recorded.

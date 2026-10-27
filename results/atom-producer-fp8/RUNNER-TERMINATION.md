# Moonshot producer-fusion benchmark: runner termination investigation

- Model: moonshotai/Kimi-K3.
- Requested benchmark: 10 warmup requests per lane; 3600 measured seconds; no GSM8K evaluation.
- Branch: perf/kimi-attnfp8-kv-allgather-20261021.
- Compute path: FP8 latent KV all-gather; fused FP8-weight KV up-projection with BF16 K/V output; BF16 FMHA.
- No completed benchmark metrics or KV-pool capacity are available for these attempts.

| Run | Verified result | Meaning |
|---|---|---|
| [40465923893](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/40465923893) | Docker SIGTERM, then SIGKILL; exit 137 | Docker stop sequence; no Docker OOM event |
| [40468464130](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/40468464130) | Unique container name; Docker SIGTERM at 07:45:03 UTC; exit 143 | Unique naming did not prevent the external stop |
| [40469496260](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/40469496260) | Docker SIGTERM at 07:55:02 UTC; exit 143 | Second stop approximately 10 minutes later |
| [40471466861](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/40471466861) | CPU-only container observation in progress | Tests whether a container without model/GPU work is also stopped |

## Checks that passed before loading

- Producer-fusion numerical preflight: all 8 GPUs, 64 cases per GPU, 512 total; actual FP8 scaled matrix multiplication passed.
- Fused KV reconstruction numerical preflight passed.
- Eight-rank RCCL preflight passed.
- Moonshot loader inventoried 513 target layers.
- Checkpoint loading began; no Python/kernel exception was recorded before shutdown.

## Root cause boundary

- Confirmed: a Docker API stop sent signal 15 to the benchmark container.
- Unknown: the process/user issuing that stop.
- Not established: GPU failure, OOM, or a kernel correctness problem.
- Visible user cron: absent.
- Visible system timers: do not establish the stop source.
- Docker spur-authz.py gates container creation; its inspected source does not periodically stop running containers.
- Root crontab is unreadable to the runner user.
- Runner registration mi355x-amd_pit2_p03_g07 maps to host pit2-p03-g09.
- The other registered MI355X runner is offline.

## Monitoring

- Script: [utils/monitor_kimi_run.py](../../utils/monitor_kimi_run.py).
- Interval: 600 seconds.
- Poller reports to an active Codex session; it cannot reopen a closed Codex session.
- Confirmed failures are investigated before another costly benchmark attempt.

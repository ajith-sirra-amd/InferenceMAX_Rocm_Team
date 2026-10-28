# Kimi-K3 attention experiment references

- Sources: original benchmark JSON artifacts and server logs from the runs below.
- Compare total tokens/sec/GPU and output tokens/sec/GPU separately. Total includes cached input tokens.
- Both references: 8 GPUs, TP8/DCP8, C48, FP8 KV cache, native vLLM image pinned at `1b2a66639`, auto loader, SimpleCPUOffload2249GB.
- Actual server configuration establishes the loaded checkpoint; historical informal labels are insufficient.

| Item | 15,476 reference | 23M KV-pool reference |
|---|---|---|
| GitHub run | [39239112271](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/39239112271) | [39601846940](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/39601846940) |
| Loaded checkpoint | Moonshot `moonshotai/Kimi-K3` | AMD `amd/Kimi-K3-Quark-MXFP4-AttnFP8` |
| Server quantization | MXFP4 plus FP8 override below | Quark checkpoint; no quantization-config override |
| FP8 override | `{"targets":{"*in_proj_qkvgfab":"fp8_per_channel"}}` | None |
| Added quantized layers | 69 `in_proj_qkvgfab` layers | Already quantized in checkpoint |
| GPU KV pool tokens | 20,951,355 | 23,098,919 |
| Total tokens/sec/GPU | 15,476.46881 | 15,258.77846 |
| Output tokens/sec/GPU | 103.05251 | 101.29363 |
| Actual reporting duration (sec) | 3,629.49550 | 3,629.98388 |
| GPU prefix-cache hit rate | 81.165% | 85.332% |

- A larger pool alone did not produce higher measured throughput in these two runs. Checkpoint, numerical behavior, request mix, cache hits and scheduling can affect performance; this comparison does not isolate a cause.
- Native FP8-prefill PS-buffer allocation prints do not establish FP8 FMHA execution. The new ATOM adapter has distinct engagement prints for causal and cached-context calls.
- Current ATOM integration retains native BF16 KV expansion and quantizes Q/K/V before FlyDSL FP8 FMHA. Direct FP8 up-projection and FP8 all-gather are not established by this experiment.
- Model smoke and full-run evidence will be recorded separately after artifacts become available.

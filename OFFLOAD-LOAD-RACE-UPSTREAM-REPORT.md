# Bug: DCP MLA prefill `kv_gather` deadlocks reading an in-flight `SimpleCPUOffloadConnector` async load

**Component:** `SimpleCPUOffloadConnector` (CPU KV offload) × DCP (decode context parallel) MLA prefill
**Severity:** hard hang (NCCL collective timeout → engine crash) under sustained load
**Image tested:** `vllm/vllm-openai-rocm:nightly-rocm100-81198e97ba7eee2a22540caaa756b7fdddcb4d93`
(vLLM `v0.31.1rc1.dev88+g81198e97b`; source refs below are against commit `81198e97`)

## Summary

Under DCP, an MLA **prefill** gathers each rank's KV shard for the cached context via
`MLADCPManager.kv_gather` (an `_ALLGATHER_BASE`). When that context KV is being brought
back from CPU by `SimpleCPUOffloadConnector` as an **async** load, there is **no ordering
between the load copy (on `load_stream`) and the prefill `kv_gather`**. If a request is
prefilled before its CPU→GPU load has completed, `kv_gather` reads KV that is still being
copied; the collective never completes and all ranks time out together after 600 s.

## Environment / repro config

- Model: Kimi-K3, FP4 (mxfp4), `--kv-cache-dtype fp8`
- Parallel: TP=8, **DCP=8** (`--decode-context-parallel-size 8 --dcp-comm-backend a2a --cp-kv-cache-interleave-size 1`)
- Attn backend: `ROCM_AITER_MLA`; `cudagraph_mode=FULL_DECODE_ONLY`
- Offload: `SimpleCPUOffloadConnector`, `kv_role=kv_both`, DRAM (`lazy_offload:false`)
- Long context (ISL ~100k), continuous/agentic load, conc≈96. Needs **sustained** load +
  a CPU-cache fetch-back burst to trigger (rare: ~5 loads/rank over 40 min in our trace).

## Symptom (observed)

All 8 ranks wedge on the **same** collective with identical parameters:

```
[Rank N] Watchdog caught collective operation timeout: WorkNCCL(SeqNum=86113,
OpType=_ALLGATHER_BASE, NumelIn=4755456, NumelOut=38043648, Timeout(ms)=600000)
```

Stack trace of the stuck collective (identical on every rank):

```
#2 kv_gather                              vllm/v1/attention/ops/dcp.py:1680
#3 _context_parallel_compute_prefill_context  vllm/model_executor/layers/attention/mla_attention.py:3163
#4 forward_mha                            vllm/model_executor/layers/attention/mla_attention.py:3279
#7 unified_mla_attention_with_output      vllm/model_executor/layers/attention/mla_attention.py:1535
#8 wrapper                                vllm/model_executor/layers/attention/kv_transfer_utils.py:54
```

## Root cause (the missing sync)

The offload path orders **stores** against compute but **not loads**:

- **Stores** record a compute-done event and the store copy waits on it
  (`worker.py: wait_for_save` → `copy_backend` `stream.wait_event(wait_event)`; cites #45704).
- **Loads** have **no** compute-ordering:
  - `SimpleCPUOffloadConnector.wait_for_layer_load()` is a **no-op** (`pass`)
    — `simple_cpu_offload_connector.py:222`.
  - The load copy runs async on a low-priority `load_stream` via a single background
    thread + `SimpleQueue` (`copy_backend.py` `launch_copy`/`_copy_loop`); only a
    completion `torch.Event` is recorded.
  - Load events are consumed **non-blocking** (`worker.py _poll_stream_events` uses
    `event.query()`); the only blocking sync (`_flush_and_sync_all`) runs **only on
    preemption** (`handle_preemptions` when `need_flush`).

So nothing guarantees a load has landed before the consuming prefill `kv_gather` reads it.
The design relies on the load completing "in time," which breaks under a fetch-back burst.

## Evidence

Instrumented runs (per-rank logging added around `kv_gather` and `start_load_kv`):

1. **The gather is fully symmetric** — all 8 ranks post the same `_ALLGATHER_BASE`
   (`SeqNum`, `NumelIn=4755456`, identical call count, identical
   `last enqueued/last completed` counters). So it is **not** a shape/count/ordering
   divergence — every rank enters the same collective and it still never completes
   ⇒ a GPU-side stall *below* the collective.
2. **It coincides with a large in-flight load.** Loads are rare (~5/rank over 40 min).
   Per-rank `start_load_kv` trace, all 8 ranks, immediately before the freeze:
   ```
   start_load_kv sync=False nblocks=499 event_idx=4 load_hwm=3   @ T
   (fatal kv_gather collective starts ~T+1s)
   load_hwm never reaches 4  → the 499-block load never completed
   ```
   A second run reproduced it with a 580-block load (`event_idx=5`, `load_hwm` stuck at 4).
   The fatal gather begins ~1 s after the big async load is issued, and that load never
   finishes ⇒ the gather read KV still being copied.
3. **A compute-stream `wait_event` is insufficient.** We tried making the **compute**
   stream wait on the load event before the forward. It still hung identically — because
   the `kv_gather` NCCL collective runs on its **own** stream, not the compute stream, so
   a compute-stream wait does not order the collective after the load. The fix must order
   the **collective's** stream (or the KV read) after the load-completion event.

## How to reproduce

1. Serve Kimi-K3 FP4 with the environment above (TP8/DCP8, `SimpleCPUOffloadConnector`
   DRAM offload, `FULL_DECODE_ONLY`, ISL ~100k) at conc≈96 under sustained agentic load.
2. Let it run until a CPU-cache fetch-back burst coincides with a large prefill
   (~40–50 min in our setup). The engine wedges; 600 s later all 8 ranks' watchdogs fire
   on the `_ALLGATHER_BASE` above.
3. To see the race directly, add temporary logging: per-rank in
   `DmaCopyBackend`/`worker.start_load_kv` log `(nblocks, event_idx, load_hwm)`, and in
   `_context_parallel_compute_prefill_context` log the `toks` right before `kv_gather`.
   The last load before the freeze is large and its `load_hwm` never advances.

Note: enabling DCP MLA **query replication** (vLLM #54494 / query-replicate) makes this
far more likely — it shrinks the on-GPU KV pool (~5%, per ROCm/ATOM #2290's measurement),
increasing offload/fetch-back churn. The race exists without it but rarely triggers.

## Suggested fix direction

Give loads the mirror of the store guard, on the **right** stream:

- Make the consuming prefill read / the `kv_gather` collective **wait on the pending
  load-completion event(s)** (order the collective's stream after the load), not just the
  compute stream; **or**
- Ensure a request is not scheduled into a prefill that reads offloaded KV until its load
  event has completed (close the gap in the async load gating so `_poll_stream_events`'
  non-blocking `query()` can't let a not-yet-landed load through); **or**
- For loads whose blocks are read in the same/next step, issue them synchronously.

Happy to share full `server.log` artifacts and the instrumentation diffs on request.

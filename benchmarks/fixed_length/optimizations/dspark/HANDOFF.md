# DSpark context projection sharding

Ready for isolated full-model performance qualification. Real-checkpoint native
operation test **43180069405 passed on all eight MI355X ranks**. Keep the pinned
SA image, all core server/client arguments and the retained baseline unchanged.
Select this candidate alone; root owns wrapper integration and dispatch.

## Install the isolated source

```sh
python install.py --source ORIGINAL_SITE_PACKAGES --output FRESH_RUN_DIRECTORY
```

The installer verifies the patch, original model file and five native dependency
hashes before writing. It copies the original model source under `original/`,
patches a separate `candidate/` copy, verifies the result and saves a manifest.
Existing output directories and output inside/above the original source are
rejected. Preserve failed preparation directories; retry into a fresh directory.

Bind only this candidate file read-only into the candidate server container:

```text
FRESH_RUN_DIRECTORY/candidate/vllm/models/kimi_k3/nvidia/dspark_mla.py
  -> /opt/runtime-venv/lib/python3.12/site-packages/vllm/models/kimi_k3/nvidia/dspark_mla.py
```

Set `KIMI_DSPARK_SHARD_CONTEXT_PROJ=1` before server/worker construction.
Use a fresh run-local Python cache prefix, for example
`PYTHONPYCACHEPREFIX=/work/cache/python-dspark`, so an installed stale bytecode
cache cannot bypass the overlay; `PYTHONDONTWRITEBYTECODE=1` alone prevents
writes but still permits reading old caches. Record loaded source hashes and
actual container environment alongside the run manifest. Never patch the
installed package or mount this overlay into the baseline container.

## Operation and activation

The opt-in constructor requires ROCm, unquantized draft, TP8, effective BF16
default dtype, H7168/target H7168 and five target layers. It replaces the native
replicated `[7168,35840]` projection with native `ColumnParallelLinear` storing
`[896,35840]` per rank, `gather_output=True`. Full K is retained per output; the
existing full-row RMSNorm runs after the output gather. The native loader slices
output rows from the same checkpoint key. No forward method, checkpoint mapper,
context KV operation, sampling configuration or graph method is changed.

Require these markers from ranks 0 through 7:

- `DSPARK_CONTEXT_SHARD phase=construct active=True`: TP8, local weight shape
  `(896,35840)`, global input35840/output7168, BF16, unquantized.
- `DSPARK_CONTEXT_SHARD phase=loaded active=True`: parameter
  `model.context_proj.weight`, same local shape/dtype and
  `ColumnParallelLinear.weight_loader_v2`.

Construction/load markers emit once; there are no forward hooks or per-call
logging costs. Missing/inactive markers invalidate a claimed candidate run.
They attest construction/loading, not per-request execution counts or numerical
quality. Retain complete startup, graph-capture and measured-request logs.

## Qualification evidence

The real checkpoint test loaded only `context_proj.weight` and
`context_norm.weight` from the local DSpark checkpoint. All eight ranks agreed
on selected tensor/config hashes and verified the exact native row shard.
All 512 FP32 scalar references, eager comparisons and changed-input HIP graph
checks passed. Inputs remained deterministic synthetic BF16 tensors.

| Rows | Baseline graph median | Candidate graph median | Reduction |
| --- | ---: | ---: | ---: |
| 280 | 316.463 us | 96.561 us | 69.49% |
| 336 | 323.742 us | 108.821 us | 66.39% |

Each value is the median of 16 per-iteration maximum-rank timings for the whole
projection, candidate gather and norm. Native N896 uses untuned `torch solution:0`;
there was no hipBLAS error. These are component results, not request gains.

Baseline repeats were exact. Candidate relative L2 error was 0.2966–0.3186%,
maximum absolute error .046875; 34.1–36.6% of output elements differed. Passing
atol .03125 / rtol .015625 is an approximate screen, not a quality guarantee.
The existing synthetic acceptance3.57 setting cannot test natural draft quality.

`validation.json` records actual candidate constructor/loader method execution
using CPU module doubles: nine activation cases, forwarding/order/mapper/load
marker checks, unchanged other method ASTs, installer roundtrip and five
rejection cases. This does not instantiate the actual Torch model.

## Full-model acceptance

Require all eight active markers, successful model loading and complete native
graph capture, then all 192 measured C48/8192-input/1024-output requests with
zero errors and valid output lengths. Compare per-request E2E, TTFT, TPOT and
throughput against the retained fixed-length baseline. Keep the complete
comparison data and any regressions. A faster full-model benchmark qualifies
performance only; natural acceptance, logits/output agreement and task quality
still need independent evidence before adopting this as a model optimization.

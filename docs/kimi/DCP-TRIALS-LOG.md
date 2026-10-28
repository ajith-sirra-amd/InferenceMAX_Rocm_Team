
## T27 (in flight) -- attack the collective itself, not its surroundings

Run: https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/32162233477
Config: DCP=4, c20, DRAM offload. **Sole variable:**
`VLLM_USE_DIRECT_DCP_{A2A,Q_GATHER,KV_GATHER}` **0 -> 1**.

Every DCP lever tried so far changed something *around* the collective --
how many ranks join it (T25), how big the batch feeding it is (T26), which
attention backend calls it (T21), whether cudagraphs wrap it (T7). All moved
TPOT by <=4%. That is consistent, and it points at the collective itself.

These three flags select the **symmetric-memory implementation** of that exact
gather/merge. They sat at 0 for one bad reason: I copied #52248's config
wholesale. Upstream disabled them so it could capture cudagraphs under DCP --
**we run cudagraphs NONE under DCP, so that constraint never applied to us.**

`VLLM_DCP_Q_REPLICATE` deliberately stays 0. It replicates Q rather than
gathering it -- a different tradeoff -- and flipping it in the same run would
repeat the T20 two-variable mistake.

Prediction, recorded before the result: this is the only remaining change that
can plausibly move TPOT more than a few percent. Either it does, or the ~2,000
ceiling is a property of the ROCm DCP implementation and not of my configuration
-- and that is a finding worth reporting to AMD either way.

Known risk: the direct paths are the suspected cause of the earlier
`dcp_utils all_gather(query)` deadlock. A hang shows up at server start, not at
80 minutes, so the downside is bounded.

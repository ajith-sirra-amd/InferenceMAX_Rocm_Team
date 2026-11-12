# Kimi-K3.1-Pro natural-verification screen

- Run the **End-to-End Tests** workflow on `IPAT-Script` with
  **Natural validation Kimi baseline + MoE front + DSpark**.
- It runs the baseline first, then the combined candidate, loading each server
  once on the same runner. A failed first arm stops the sequence.
- Both answer the same 48 pinned GSM8K questions at C48, with matching seeds,
  temperature 1, top-p 0.95 and a 2,048-token output limit.
- This separate configuration uses real block verification. Synthetic draft
  acceptance is removed; the other SA server settings and image are preserved.
  The original SA performance script and default fixed-length mode are unchanged.
- Results include all answers, per-question scores, token counts, server/source
  identities and raw acceptance counters. `comparison.json` lists answer
  regressions and acceptance changes. Missing or truncated answers invalidate
  collection. Successful collection does not prove unchanged model quality.

Files are saved under `/share_nfs/ajith/benchmarks/natural-<run>-<attempt>/`.
The workflow preserves both arms and uploads evidence, excluding regenerable
`work/` caches. Server readiness is capped at 30 minutes per arm and the client
at 390 seconds including forced termination; the job limit is 85 minutes.

To run a single arm manually, use a fresh output directory:

```bash
bash benchmarks/fixed_length/kimi_k31_c48.sh \
  /share_nfs/ajith/benchmarks/natural-example-baseline baseline natural
```

Use `moe_front_dspark` with a different directory for the second arm, then run
`python3 benchmarks/quality/compare.py BASELINE_DIR CANDIDATE_DIR RESULT_JSON`.
The workflow supplies the outer time limits and cleanup for the complete pair.

Local checks require no GPU or additional packages:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 benchmarks/quality/test_quality.py
```

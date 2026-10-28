# Kimi-K3 FP4 / 8× MI355X — C1 through C72

Best verified numbers per concurrency point, our repo unless marked SA. Not all
five are the same harness — noted per row.

## Summary

| conc | harness | best metric | run |
|---|---|---|---|
| **C1** | fixed-len, ISL 122k/OSL 500, k=8, DCP off | **P90 TPOT 7.91 ms** | [T147](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/33171360827) |
| **C4** | agentic | 2,857 tok/s/GPU, ITL p90 8.98 | [ATOM run](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/35487769456) (SA) |
| **C10** | fixed-len, ISL 8192/OSL 1024, MTP k=5 | P90 TPOT 16.76 ms, 593 tok/s | [run](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/35053890062) |
| **C48** | agentic, no-MTP | **11,048 tok/s/GPU**, ITL p90 77.58 | [run](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/34830611340) (SA) |
| **C72** | agentic, no-MTP | **12,484 tok/s/GPU (best throughput)**, ITL p90 119.67 | [run](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/34830611340) (SA) |

## C1 — lowest interactivity target

[Kimi-DCP-Experiemnts-Summary.md](Kimi-DCP-Experiemnts-Summary.md) T138–T151 sweep,
fixed-length, ISL 122k (~63.8k post-BPE) / OSL 500, 100 requests, DCP off, k=8 @
AL 4.00, draft KV fp8 — chosen to match the agentic replay's mean input length.

| run | image | TPOT mean | p50 | **p90** |
|---|---|--:|--:|--:|
| **[T147](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/33171360827)** | nightly 6f7df92a8e | 7.57 | 7.47 | **7.91** |
| [T148](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/33183801155) | nightly | 7.57 | 7.49 | 7.93 |
| [T150](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/33185946417) | nightly, mns 1 | 7.58 | 7.48 | 7.93 |

Nightly beats `aigmkt` by 13.7% (T145 8.77 → T147 7.57 mean).

## C4 — agentic, ATOM vs vLLM (SA cluster)

From [KIMI_vLLM_vs_Atom.md](KIMI_vLLM_vs_Atom.md):

| engine | tok/s/GPU | ITL p90 | run |
|---|--:|--:|---|
| **ATOM** | **2,857** | 8.98 | [35487769456](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/35487769456) |
| vLLM | 2,815 | 9.42 | [35157825655](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/35157825655) |

ATOM wins both axes at C4 (k=7 vs vLLM's k=5 — see that doc for the drafter-kernel gap).

## C10 — fixed-length, MTP k sweep

ISL 8192 / OSL 1024, 500 prompts, our repo, `mi355x-amd_b23_07`:

| k | tok/s | Mean TPOT | **P90 TPOT** | Mean ITL | run |
|---|--:|--:|--:|--:|---|
| 3 | 583.20 | 16.28 | 16.98 | 48.80 | [35053247421](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/35053247421) |
| 4 | 589.24 | 16.09 | 16.78 | 54.04 | [35051254195](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/35051254195) |
| **5** | **593.23** | **15.98** | **16.76** | 57.80 | [35053890062](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/35053890062) |

k=5 edges out k=3/4 on every axis at C10; gains are small (≤1.3%) and mean ITL
gets worse as k rises — same bunching effect seen at C12/C48.

## C48 — agentic, no-MTP vs MTP

Full breakdown in [HANDOFF.md](HANDOFF.md) (2026-09-24 sections) and
[Kimi-K3-Where-The-Time-Goes.md](Kimi-K3-Where-The-Time-Goes.md).

| arm | tok/s/GPU | ITL p90 | KV pool | GPU hit |
|---|--:|--:|--:|--:|
| **no-MTP** | **11,048** | 77.58 | 30,591,065 | 91.9% |
| MTP k=4 + #56861 | 10,549 | 90.47 | 14,256,600 | 31.9% |

No-MTP: [34830611340](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/34830611340) (SA). MTP k=4:
[35956289244](https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/actions/runs/35956289244)
(workflow shows `failure` — the benchmark itself completed clean; a harness
path bug dropped the result file after the numbers were already printed, fixed
in commit `3ad3a620`).
No-MTP wins throughput; MTP wins median ITL (46.55 vs 57.61) but loses P90 —
speculation bunches tokens, so the axis you optimize for picks the winner.

## C72 — agentic, best throughput achieved

| arm | tok/s/GPU | ITL p90 | KV pool | GPU hit |
|---|--:|--:|--:|--:|
| **no-MTP** | **12,484** | 119.67 | 27,867,046 | 76.0% |

Run: [34830611340](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/34830611340) (SA). Comparable to NV B300 c70 no-MTP (12,566 tok/s/GPU,
ITL p90 126.77) — we're at parity on throughput and ahead on P90 latency.
Best-ever target in [EXPERIMENT-QUEUE.md](EXPERIMENT-QUEUE.md) was 12,556; this
run is 0.6% short of that ceiling.

## What's missing between C10 and C48

**C14, C16, C32, C64** all have data scattered across `EXPERIMENT-QUEUE.md` and
scratch logs but none consolidated here. C14 in particular is the one point SA
ships MTP at k=3 in production ([KIMI_vLLM_vs_Atom.md](KIMI_vLLM_vs_Atom.md)) —
worth its own row if this table gets extended.

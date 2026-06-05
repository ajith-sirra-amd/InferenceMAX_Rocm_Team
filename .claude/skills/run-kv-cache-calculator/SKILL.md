---
name: run-kv-cache-calculator
description: Analytical KV-cache simulator that estimates the concurrency at which hicache/lmcache offloading becomes effective for a model on given hardware + parallelism (TP/EP/DEP). Use when the user asks where offloading pays off, the crossover concurrency, KV-pool size per GPU, KV bytes per token, or whether KV spills past HBM. Invoke with a model alias (glm-5.1-fp8) or an agentic launcher .sh path.
---

# run-kv-cache-calculator

Predicts the **concurrency at which the device KV pool overflows** — where hicache
(host RAM) / lmcache offloading starts to matter — for a model on given hardware +
parallelism. Paths below are relative to the repo root.

## Output: surface the raw stdout verbatim

The tool prints its own layout table + column legend + verdict. After running it,
paste that raw stdout in a code block and **stop**. Do NOT reformat it into a
markdown table, add prose summaries, takeaways, or verdicts of your own. Only
interpret if the user explicitly asks for analysis.

## Run

```bash
SK=.claude/skills/run-kv-cache-calculator/kv_cache_calc.py

# model alias + hardware:
python3 $SK glm-5.1-fp8 --hw mi355x

# from an agentic launcher (model/dtype/hw parsed from it):
python3 $SK upstream/InferenceX/benchmarks/single_node/agentic/gptoss_fp4_mi300x.sh

# override mean input length (env-dependent) and parallelism:
python3 $SK qwen3.5-fp8 --hw mi355x --tp 4 --gpu-util 0.8 --mean-isl 85000
```

Invoke with a model **alias** or a launcher `.sh` path — never a bare HF id, which
falls into a tableless per-token mode. If only an HF id is known, find or add its
alias in `vendor/aliases.json` first.

By default it reads the parallelism layouts **actually configured for that
model+hw in `.github/configs/amd-master.yaml`** (real `tp`/`ep`/`dp-attn` +
`conc-list`) and prints one row per layout:

```
mi355x | MiniMaxAI/MiniMax-M2.5 | KV=fp8 weights=fp4
params~228.6B  weight~106.5GB(node)  HBM 288GB x util 0.9  per-token KV 126,976 B  mean-ISL 250,000
layouts from amd-master.yaml: minimaxm2.5-fp4-mi355x-vllm-agentic-lmcache
┌────────┬────┬────┬────┬────────────┬─────────────┬───────────┬─────────────────┐
│ layout │ TP │ DP │ EP │ weight/GPU │ KVcache/GPU │ crossover │ configured conc │
├────────┼────┼────┼────┼────────────┼─────────────┼───────────┼─────────────────┤
│ tp     │ 2  │ 1  │ 1  │ 53.2 GB    │ 198.0 GB    │ ~13       │ 1…64            │
└────────┴────┴────┴────┴────────────┴─────────────┴───────────┴─────────────────┘
(KVcache/GPU = HBM left for KV per GPU = HBM × util − weight/GPU − ~8GB act.)
```

### Auto-inherited run conditions (util + ISL)

On an alias or `.sh` invocation, the calc locates the matching agentic launcher
and inherits its real server knobs, so you get the validated numbers without
flags:

- **`--gpu-util`** ← the launcher's `--mem-fraction-static` / `--gpu-memory-utilization`
  (varies per model/hw: 0.75–0.95).
- **`--mean-isl`** ← **85000** for any agentic launcher. The measured mean prompt
  length is ~85k across ALL CI runs (45k–112k). Note the `..._256k`
  `WEKA_LOADER_OVERRIDE` / `max_model_len=262144` is the trace's **context
  window**, NOT the mean ISL — don't confuse the two. Bare HF ids default to 250000.

When two launchers match (e.g. a vllm and an sglang variant), it prefers the
framework configured in `amd-master.yaml`. Explicit `--gpu-util` / `--mean-isl`
always override the inherited value.

**Validated against CI** (`SemiAnalysisAI/InferenceX` e2e, branch `amd/agentx-v0.4`):
predicted crossover vs the conc where the device overflows (vllm
`server_gpu_cache_hit_rate` collapse, or sglang `load_back_tokens` onset):
Kimi-fp4 ~41 vs 40 · MiniMax-fp8 ~32 vs 32 · GLM-fp4 ~9 vs 8 · Qwen-fp8 ~54 vs
48–56. Per-token KV / KVcache exact vs vllm's own startup log (Kimi: 35,136 B vs
35,137; 115 GB vs 119.58 GiB).

## Key flags

- `--hw {mi300x,mi325x,mi355x,h100,h200,b200,b300}` — sets per-GPU HBM (required).
- `--mean-isl N` — override the inherited mean input tokens/request (crossover
  scales inversely). Confirm against your run's `mean_input_tokens` in `agg_bmk.json`.
- `--quant {fp4,fp8,bf16,…}` — weight dtype; `--gpu-util` — override server mem-fraction.
- `--tp/--dp/--ep`, `--layout {tp,tp-ep,dep,all}`, `--no-config` — override the
  config-driven layouts with a generic sweep.
- `--json` — machine output. `--list` — known models.

## Files

- `kv_cache_calc.py` — the driver.
- `vendor/modelconfig.json` — model dims + MoE/param fields.
- `vendor/aliases.json` — aliases + launcher stems → HF id, plus `hw_hbm_gb`.

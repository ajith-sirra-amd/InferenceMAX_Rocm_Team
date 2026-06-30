# gg_agentic — InferenceMAX Agentic Orchestrator

Natural-language CLI interface powered by Claude for monitoring and managing
GitHub Actions workflows in the InferenceMAX_rocm project.

---

## Overview

`gg_agentic` provides two ways to interact with the CI/CD pipeline:

| Mode | Entry point | When to use |
|---|---|---|
| **Interactive chat** | `chat.py` | Ad-hoc queries, log analysis, flexible dispatch |
| **CLI launcher** | `run_and_watch.py` | Automated dispatch + live monitoring, scripting |

Both modes share the same GitHub Actions tools and Claude backend.

---

## Project layout

```
gg_agentic/
├── README.md              ← this file
├── __init__.py
├── config.py              ← centralized configuration (API keys, model, repo)
├── chat.py                ← interactive natural-language chat loop
├── run_and_watch.py       ← CLI: dispatch a workflow and monitor it live
├── run_config.yaml        ← default parameters for run_and_watch.py
└── tools/
    ├── __init__.py
    └── gh_actions.py      ← GitHub Actions REST API helpers + watch_run
```

Esempio di prompt:

```
lancia il default e monitoralo

lancia kimik2.7-code e monitoralo

Lancia e2e-tests.yml sul branch chore/giovanni_agentx-v0.4 \
con config kimik2.7-fp4-mi355x-vllm-agentic-lmcache e monitoralo

monitora la run 15042

```

---

## Analisi risultati benchmark

Dopo il completamento di un run, gli artifact vengono scaricati localmente (tipicamente in `results/`).
Questi script convertono i raw artifact in un json aggregato e generano il grafico Pareto.

### 1. Estrai `agg_bmk.json` da un run (o più run)

```bash
# Run singolo
python gg_agentic/extract_agg_bmk.py \
    --results-dir results/ \
    --hw mi355x \
    --model-prefix kimik2.7-code
# output: results/results_bmk/agg_bmk.json

# Più run (none + lmcache) → json combinato
python gg_agentic/extract_agg_bmk.py \
    --results-dir results_none/ results_lmcache/ \
    --hw mi355x \
    --model-prefix kimik2.7-code \
    --output results_combined/results_bmk/agg_bmk.json
```

Argomenti obbligatori: `--results-dir`, `--hw`, `--model-prefix`.
Opzionali: `--precision` (auto-rilevato dal nome modello), `--image`, `--output`.

Lo script legge automaticamente:
- `vllm_command.txt` / `sglang_command.txt` → tp, conc, model, offloading, ep, framework
- `benchmark_command.txt` → concurrency (verifica), durata, dataset
- `aiperf_artifacts/profile_export_aiperf.json` → tutte le metriche latenza/throughput
- `aiperf_artifacts/server_metrics_export.json` → breakdown token per cache source (vllm e sglang)

### 2. Genera il grafico Pareto a 3 pannelli

```bash
python gg_agentic/pareto_chart_kimik27.py
# output: gg_agentic/pareto_chart_kimik27.png
```

Il grafico mostra, per ogni json in `INPUT_FILES`:
- **Pannello sinistra** — Pareto frontier: asse X = p90 E2E latency (s), asse Y = throughput/GPU (tok/s)
- **Pannello centro** — TTFT bar: asse X = concurrency, asse Y = p90 TTFT (s)
- **Pannello destra** — Cache stacked bar: asse X = concurrency, asse Y = prompt token per source (local compute / cache hit / ext KV)

Per aggiungere nuovi punti (es. run lmcache o nuove concurrency):
1. Riesegui `extract_agg_bmk.py` con tutti i `--results-dir` necessari
2. Riesegui `pareto_chart_kimik27.py`

### File generati

| File | Contenuto |
|---|---|
| `results/results_bmk/agg_bmk.json` | Metriche aggregate (un oggetto per run) |
| `gg_agentic/pareto_chart_kimik27.png` | Grafico a 3 pannelli |
| `gg_agentic/GG_KimiK2.7code.md` | Guida alla configurazione del benchmark |

---

Convenience launchers in the project root:

| File | Description |
|---|---|
| `chat.bat` | Windows CMD: activate conda env and run `chat.py` |
| `gg_agentic/launch_kimik27.sh` | Bash: check runner + dispatch kimik2.7 (no monitor) |
| `gg_agentic/launch_kimik27.bat` | Windows CMD equivalent of the above |

---

## Prerequisites

| Requirement | Notes |
|---|---|
| Python ≥ 3.11 | available in the `auto_sglang` conda env |
| `anthropic` | Claude SDK |
| `requests` | GitHub REST API calls |
| `pyyaml` *(optional)* | for `run_config.yaml` parsing; falls back to a built-in parser |
| `rich` *(optional)* | coloured terminal UI in `chat.py`; falls back to plain `print` |

Install into the conda environment if needed:

```bash
conda activate auto_sglang
pip install anthropic requests rich pyyaml
```

---

## Configuration

### `.env` file (project root)

Create `.env` in the **project root** (git-ignored). Variables already set in
the shell environment take precedence.

```ini
# GitHub fine-grained PAT — needs repo + actions scopes on ROCm/InferenceMAX_rocm
# Note: classic PATs are blocked by the ROCm org; use fine-grained PATs.
GITHUB_TOKEN=github_pat_...

# --- Option A: public Anthropic API ---
ANTHROPIC_API_KEY=sk-ant-...

# --- Option B: AMD internal proxy (no API key needed) ---
ANTHROPIC_BASE_URL=https://<amd-proxy-host>/...
ANTHROPIC_CUSTOM_HEADERS=Ocp-Apim-Subscription-Key: <subscription-key>
# ANTHROPIC_API_KEY can be omitted or set to "dummy" in proxy mode

# Optional overrides
# ANTHROPIC_MODEL=claude-sonnet-4-6
# GITHUB_REPO=ROCm/InferenceMAX_rocm
# INFERENCEMAX_CONDA_ENV=auto_sglang
# INFERENCEMAX_CHAT_LOG=/path/to/session.log
```

### `run_config.yaml` (gg_agentic/)

Default parameters for `run_and_watch.py`. Edit this file to change the
target branch, config key, poll interval, etc. without passing CLI flags.

```yaml
workflow: e2e-tests.yml
ref: chore/giovanni_agentx-v0.4
config-files: .github/configs/amd-master.yaml
config-keys: kimik2.7-fp4-mi355x-vllm-agentic-lmcache
runner: mi355x
poll-interval: 30       # seconds between polls
job-filter: ""          # empty = all jobs; e.g. "agentic" to narrow scope
```

Priority: **CLI args > `run_config.yaml` > built-in defaults**.

---

## Usage

### 1. Interactive chat — `chat.py`

```bash
conda activate auto_sglang
python gg_agentic/chat.py
# or on Windows:
chat.bat
```

Type `exit` or press `Ctrl-C` to quit.

#### Input modes

**Single line** — the default:
```
You> mostrami le ultime 5 run di e2e-tests.yml
```

**Block mode** — for pasting log excerpts, error messages, or multi-line text.
Open and close with `"""` on its own line:
```
You> """
  [paste anything — multiple lines, blank lines, special characters]
  ERROR: container exited with code 1
  ...
"""
```

**Line continuation** — end a line with `\` to join the next:
```
You> analizza il job "agentic" della run 14823 \
...  e dimmi perché ha fallito
```

#### Example prompts

```
Mostrami le ultime 5 run di e2e-tests.yml sul branch corrente.

Cosa è andato storto nella run 14823?

Scarica i log del job "agentic" dalla run 14823 e dimmi perché ha fallito.

Lancia e2e-tests.yml sul branch chore/giovanni_agentx-v0.4 con config
kimik2.7-fp4-mi355x-vllm-agentic-lmcache e monitoralo.

Monitora la run 15042, solo il job "agentic".
```

---

### 2. CLI launcher — `run_and_watch.py`

A standalone script that combines runner check, workflow dispatch, and live
monitoring in a single command. Defaults are loaded from `run_config.yaml`.

```
python gg_agentic/run_and_watch.py [options]
```

#### Options

| Flag | Default (from `run_config.yaml`) | Description |
|---|---|---|
| `--config FILE` | `gg_agentic/run_config.yaml` | YAML file with default parameters |
| `--ref BRANCH` | `chore/giovanni_agentx-v0.4` | Branch or SHA to run on |
| `--config-keys KEY` | `kimik2.7-fp4-mi355x-vllm-agentic-lmcache` | Config key passed to `test-config` |
| `--config-files PATH` | `.github/configs/amd-master.yaml` | Config file path |
| `--workflow FILE` | `e2e-tests.yml` | Workflow filename |
| `--runner LABEL` | `mi355x` | Runner label for availability check |
| `--poll N` | `30` | Seconds between monitoring polls |
| `--job-filter STR` | *(all jobs)* | Monitor only jobs whose name contains this string |
| `--run-id N` | — | Skip dispatch; watch an existing run |
| `--force` | — | Dispatch even if the runner is busy |
| `--no-watch` | — | Dispatch but do not monitor |
| `--dry-run` | — | Print what would be dispatched, do nothing |

#### Common invocations

```bash
conda activate auto_sglang

# Use all defaults from run_config.yaml (zero args)
python gg_agentic/run_and_watch.py

# Override the branch for a one-off run
python gg_agentic/run_and_watch.py --ref chore/agentx-v0.4

# Run a different config key, keep the same branch
python gg_agentic/run_and_watch.py \
    --config-keys minimaxm2.5-fp4-mi355x-vllm-agentic-lmcache

# Watch only the "agentic" job, poll every 20 seconds
python gg_agentic/run_and_watch.py --job-filter agentic --poll 20

# Watch an already-running run (no dispatch)
python gg_agentic/run_and_watch.py --run-id 15042

# Dispatch without staying to monitor
python gg_agentic/run_and_watch.py --no-watch

# Preview: show what would be dispatched without doing it
python gg_agentic/run_and_watch.py --dry-run

# Use a different config file (e.g. for another model)
python gg_agentic/run_and_watch.py --config gg_agentic/run_config_minimax.yaml
```

#### Execution flow

```
run_and_watch.py
  │
  ├─ Load run_config.yaml (defaults)
  ├─ Parse CLI args (override defaults)
  │
  ├─ [skip if --run-id] Check runner availability
  │     gh API: list in_progress/queued jobs on mi355x
  │     → abort if busy (unless --force)
  │
  ├─ [skip if --run-id] Dispatch workflow
  │     trigger_workflow(workflow, ref, inputs)
  │     → wait up to 120s for new run ID to appear
  │
  └─ watch_run(run_id, poll_interval, job_filter)
        every N seconds:
          fetch run status + jobs
          for each job in_progress/completed:
            download job logs
            diff against previously seen lines
            filter lines by keyword heuristic
            ask Claude to interpret new lines (3-5 bullets)
            print live to terminal
        → print final summary when run is completed
        → Ctrl-C to interrupt
```

---

## Available tools (used by `chat.py` and `watch_run`)

| Tool | Description |
|---|---|
| `list_workflow_runs` | List recent runs of a workflow (status, conclusion, branch, URL) |
| `get_run_status` | Status + per-job summary of a single run |
| `get_run_logs` | Download full run logs as text (ZIP → concatenated); optional `job_name_filter` |
| `get_job_logs` | Download logs for a single job (lighter, no ZIP) |
| `trigger_workflow` | Dispatch a `workflow_dispatch` event — `chat.py` always asks for confirmation |
| `interpret_logs` | Ask Claude to summarize or answer a question about arbitrary log text |
| `watch_run` | Monitor a run live: poll → diff logs → filter → interpret → print |

### `watch_run` details

- **Log filtering** (`_filter_log_lines`): strips GitHub timestamps, keeps lines matching keywords: `error`, `warning`, `fail`, `exception`, `traceback`, `benchmark`, `throughput`, `tokens/s`, `ttft`, `tpot`, `conc`, `isl`, `osl`, `container`, `validation`, `pydantic`, `config`, `starting`, `finished`, `success`, plus 1 line of context around each match.
- **Incremental diff**: tracks how many lines have already been shown per job; only new lines are printed and sent to Claude each poll cycle.
- **Claude interpretation**: triggered when a new snippet is > 200 chars; uses a lightweight 512-token call to produce 3-5 bullet points.
- **Ctrl-C safe**: `KeyboardInterrupt` is caught; a summary URL is printed before exit.

### Live monitoring output example

```
[watch_run] Monitoring run #15042 — polling every 30s. Ctrl-C to stop.

  ⏳ [get-jobs] QUEUED
  🔄 [get-jobs] IN_PROGRESS
── get-jobs (new log lines) ──
  Generating matrix for kimik2.7-fp4-mi355x-vllm-agentic-lmcache…
  Matrix generated: 1 job

  🤖 Claude:
  • Matrix generation started — config key resolved successfully
  • 1 job queued: agentic (tp=8, offloading=lmcache)
  • No validation errors detected

  ✅ [get-jobs] COMPLETED → success
  🔄 [agentic] IN_PROGRESS
── agentic (new log lines) ──
  Pulling image vllm/vllm-openai-rocm:v0.21.0 ...
  Starting vLLM server on port 8911 ...
  Server ready. Starting benchmark...

  🤖 Claude:
  • Docker image pulled and vLLM server started on port 8911
  • Benchmark starting: tp=8, lmcache offloading, conc=16
  • No errors detected so far

  [watch_run] Run status: in_progress — next poll in 30s…
  ...
  ✅ Run #15042 completed — conclusion: success
     URL: https://github.com/ROCm/InferenceMAX_rocm/actions/runs/15042

Final job summary:
  ✅ get-jobs: completed / success
  ✅ agentic: completed / success
```

---

## Session log

All terminal output from `chat.py` is mirrored to `gg_agentic_session.log`
in the project root (git-ignored). Useful for reviewing long sessions.

Override the path:

```bash
export INFERENCEMAX_CHAT_LOG=/tmp/my_session.log
```

---

## Known issues / limitations

| Issue | Details |
|---|---|
| **Fine-grained PAT required** | The ROCm org blocks classic PATs (`ghp_...`). Use a fine-grained PAT (`github_pat_...`) with `repo` + `actions` scopes. New PATs require admin approval and may be `pending` for a short time. |
| **Port 8888 on m15-17** | `smci355-ccs-aus-m15-17` has a persistent process bound to `127.0.0.1:8888`. The runner script (`runners/launch_mi355x-amd.sh`) pins `PORT=8911` when `RUNNER_NAME` matches `*m15-17*`. |
| **Logs unavailable while queued** | GitHub does not expose logs for jobs in `queued` state. `watch_run` silently skips them and retries on the next poll. |
| **`watch_run` blocks the chat loop** | In `chat.py`, while `watch_run` is running the chat is unresponsive. Press `Ctrl-C` to interrupt monitoring and return to the prompt. |

---

## Adding new tools

1. Implement the function in `tools/gh_actions.py` (or a new file under `tools/`).
2. Import it in `chat.py` and add an entry to both `TOOLS_SCHEMA` and `TOOL_FN`.
3. Claude will discover and use the new tool automatically in the chat loop.
4. If the tool requires live output (like `watch_run`), inject `print_fn` via `_execute_tool`.

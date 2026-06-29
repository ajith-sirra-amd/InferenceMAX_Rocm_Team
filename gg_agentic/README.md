# gg_agentic — InferenceMAX Agentic Orchestrator

Natural-language CLI interface powered by Claude for monitoring and managing
GitHub Actions workflows in the InferenceMAX_rocm project.

---

## Overview

`gg_agentic` is a Claude-powered chat loop that lets you interact with the
CI/CD pipeline of InferenceMAX_rocm using plain language (Italian or English).

The assistant can:
- List recent GitHub Actions workflow runs
- Check the status and per-job summary of a run
- Download and display workflow logs
- Interpret logs to diagnose failures
- Trigger `workflow_dispatch` events (with explicit confirmation)

There are no autonomous background agents — every tool call is initiated by
Claude in response to your message, and results are shown immediately.

---

## Project layout

```
gg_agentic/
├── README.md              ← this file
├── __init__.py
├── config.py              ← centralized configuration (API keys, model, repo)
├── chat.py                ← interactive chat loop with agentic tool-use
└── tools/
    ├── __init__.py
    └── gh_actions.py      ← GitHub Actions REST API helpers
```

---

## Prerequisites

| Requirement | Notes |
|---|---|
| Python ≥ 3.11 | available in the `auto_sglang` conda env |
| `anthropic` | Claude SDK |
| `requests` | GitHub REST API calls |
| `rich` *(optional)* | coloured terminal UI; falls back to plain `print` |

Install into the conda environment if needed:

```bash
conda activate auto_sglang
pip install anthropic requests rich
```

---

## Configuration (`.env`)

Create a `.env` file in the **project root** (it is git-ignored):

```ini
# GitHub personal access token — needs repo + actions:read scope
GITHUB_TOKEN=ghp_...

# --- Option A: public Anthropic API ---
ANTHROPIC_API_KEY=sk-ant-...

# --- Option B: AMD internal proxy ---
ANTHROPIC_BASE_URL=https://...
ANTHROPIC_CUSTOM_HEADERS=Ocp-Apim-Subscription-Key: <key>
# ANTHROPIC_API_KEY can be left unset or set to "dummy" in proxy mode

# Optional overrides
# ANTHROPIC_MODEL=claude-sonnet-4-6
# GITHUB_REPO=ROCm/InferenceMAX_rocm
# INFERENCEMAX_CONDA_ENV=auto_sglang
# INFERENCEMAX_CHAT_LOG=/path/to/session.log
```

Variables already set in the shell environment take precedence over `.env`.

---

## Usage

```bash
conda activate auto_sglang
python gg_agentic/chat.py
```

Type `exit` or press `Ctrl-C` to quit.

### Input modes

**Single line** — the default:
```
You> mostrami le ultime 5 run di e2e-tests.yml
```

**Block mode** — for pasting log excerpts, error messages, or long text.
Open and close the block with `"""` on its own line:
```
You> """
  [paste anything here — multiple lines, blank lines, special characters]
  ERROR: container exited with code 1
  ...
"""
```
The entire block is sent as a single message once the closing `"""` is typed.

**Line continuation** — for long messages typed manually.
End a line with `\` to continue on the next:
```
You> analizza il job "agentic" della run 14823 \
...  e dimmi perche' ha fallito
```

### Example prompts

```
Mostrami le ultime 5 run di e2e-tests.yml sul branch corrente.

Cosa è andato storto nella run 14823?

Scarica i log del job "agentic" dalla run 14823 e dimmi perché ha fallito.

Lancia e2e-tests.yml sul branch chore/giovanni_agentx-v0.4
con input generate-cli-command="test-config --config-files
.github/configs/amd-master.yaml --config-keys glm5.1-fp8-mi300x-sglang-agentic".
```

---

## Available tools

| Tool | Description |
|---|---|
| `list_workflow_runs` | List recent runs of a workflow (status, conclusion, branch, URL) |
| `get_run_status` | Status + per-job summary of a single run |
| `get_run_logs` | Download full logs as text; optional `job_name_filter` to narrow scope |
| `get_job_logs` | Download logs for a single job (lighter than the full run ZIP) |
| `trigger_workflow` | Dispatch a `workflow_dispatch` event — always asks for confirmation |
| `interpret_logs` | Ask Claude to summarize or answer a question about log text |
| `watch_run` | **Monitor a running workflow live.** Polls every N seconds, downloads per-job logs, filters relevant lines, and asks Claude to interpret each update in real-time. Ctrl-C to stop. |

### Live monitoring

Invoke by asking Claude to "monitor", "watch", "follow", "ascolta i log", etc.
Claude will call `watch_run` automatically with the appropriate `run_id`.

```
You> lancia e2e-tests.yml con config kimik2.7-fp4-mi355x-vllm-agentic-lmcache e monitoralo
```

Output streams live to the terminal:
```
[watch_run] Monitoring run #15042 — polling every 30s. Ctrl-C to stop.

  ⏳ [get-jobs] QUEUED
  🔄 [get-jobs] IN_PROGRESS
── get-jobs (new log lines) ──
  Generating matrix for kimik2.7-fp4-mi355x-vllm-agentic-lmcache…

  🤖 Claude:
  • Matrix generation started — no validation errors
  • 1 job queued: agentic

  ✅ [get-jobs] COMPLETED → success
  🔄 [agentic] IN_PROGRESS
── agentic (new log lines) ──
  Pulling image vllm/vllm-openai-rocm:v0.21.0 ...
  …
```

You can also watch an already-running or already-dispatched run:
```
You> monitora la run 15042
You> monitora la run 15042, solo il job "agentic"
```

---

## Session log

All terminal output is mirrored to `gg_agentic_session.log` in the project
root (git-ignored). Useful for reviewing what happened after a long session.

Override the path via:

```bash
export INFERENCEMAX_CHAT_LOG=/tmp/my_session.log
```

---

## Adding new tools

1. Implement the function in `tools/gh_actions.py` (or a new file under `tools/`).
2. Import it in `chat.py` and add an entry to both `TOOLS_SCHEMA` and `TOOL_FN`.
3. Claude will discover and use the new tool automatically.

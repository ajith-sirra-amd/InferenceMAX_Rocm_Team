"""
InferenceMAX agentic orchestrator — natural-language chat interface.

Runs inside the `auto_sglang` conda environment.

Usage:
    python gg_agentic/chat.py

The user can type requests in any language. Claude interprets them and
calls the available tools (GitHub Actions: list runs, get status, fetch
logs, trigger workflow, interpret logs).

No autonomous agents are wired yet — the LLM decides which tools to call
based on the conversation; the human approves each step interactively.
"""

from __future__ import annotations

import io
import json
import sys
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Optional rich UI — fall back gracefully if not installed
# ---------------------------------------------------------------------------
try:
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.panel import Panel
    _RICH = True
except ImportError:
    _RICH = False


def _set_title(title: str) -> None:
    sys.stdout.write(f"\033]0;{title}\007")
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# TeeConsole: mirrors terminal output to a plain-text log file
# ---------------------------------------------------------------------------
class TeeConsole:
    def __init__(self, log_path: Path | None = None) -> None:
        if _RICH:
            self._term = Console()
        else:
            self._term = None
        self._log_fh = open(log_path, "a", encoding="utf-8") if log_path else None

    def _ts(self) -> str:
        return datetime.now().strftime("%H:%M:%S")

    def _write_log(self, text: str) -> None:
        if self._log_fh:
            self._log_fh.write(f"[{self._ts()}] {text}\n")
            self._log_fh.flush()

    def print(self, *args, markup: bool = True, **kwargs) -> None:
        if _RICH and self._term:
            self._term.print(*args, markup=markup, **kwargs)
        else:
            text = " ".join(str(a) for a in args)
            print(text)
        plain = " ".join(str(a) for a in args)
        self._write_log(plain)

    def input(self, prompt: str = "") -> str:
        if _RICH and self._term:
            value = self._term.input(prompt)
        else:
            value = input(prompt)
        self._write_log(f"> {value}")
        return value

    def rule(self, title: str = "") -> None:
        if _RICH and self._term:
            self._term.rule(title)
        else:
            print(f"{'─'*60}  {title}  {'─'*60}" if title else "─" * 80)

    def render_md(self, text: str) -> None:
        if _RICH and self._term:
            self._term.print(Markdown(text))
        else:
            print(text)


# ---------------------------------------------------------------------------
# Tool registry — maps tool name → Python callable
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent))
import config as cfg
from tools.gh_actions import (
    list_workflow_runs,
    get_run_status,
    get_run_logs,
    get_job_logs,
    trigger_workflow,
    interpret_logs,
    watch_run,
)

TOOLS_SCHEMA: list[dict] = [
    {
        "name": "list_workflow_runs",
        "description": (
            "List recent GitHub Actions workflow runs. "
            "Returns id, status, conclusion, branch, timestamps, url."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "workflow_id": {
                    "type": "string",
                    "description": "Workflow filename, e.g. 'e2e-tests.yml'.",
                    "default": "e2e-tests.yml",
                },
                "branch": {
                    "type": "string",
                    "description": "Filter by branch name. Omit for all branches.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max number of runs to return (1-100).",
                    "default": 10,
                },
            },
        },
    },
    {
        "name": "get_run_status",
        "description": "Get status, conclusion, and per-job summary of a workflow run.",
        "input_schema": {
            "type": "object",
            "properties": {
                "run_id": {
                    "type": "integer",
                    "description": "Numeric GitHub Actions run ID.",
                }
            },
            "required": ["run_id"],
        },
    },
    {
        "name": "get_run_logs",
        "description": (
            "Download full logs of a workflow run as plain text. "
            "Use job_name_filter to narrow down to a specific job."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "run_id": {
                    "type": "integer",
                    "description": "Numeric GitHub Actions run ID.",
                },
                "job_name_filter": {
                    "type": "string",
                    "description": "Optional substring to filter log files by job name.",
                },
            },
            "required": ["run_id"],
        },
    },
    {
        "name": "get_job_logs",
        "description": "Download logs for a single job (lighter than the full run zip).",
        "input_schema": {
            "type": "object",
            "properties": {
                "job_id": {
                    "type": "integer",
                    "description": "Numeric GitHub Actions job ID.",
                }
            },
            "required": ["job_id"],
        },
    },
    {
        "name": "trigger_workflow",
        "description": (
            "Trigger a GitHub Actions workflow_dispatch event. "
            "IMPORTANT: always confirm with the user before calling this tool."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "workflow_id": {
                    "type": "string",
                    "description": "Workflow filename, e.g. 'e2e-tests.yml'.",
                },
                "ref": {
                    "type": "string",
                    "description": "Branch or tag to run on.",
                },
                "inputs": {
                    "type": "object",
                    "description": "Key-value pairs matching the workflow's dispatch inputs.",
                },
            },
            "required": ["workflow_id", "ref"],
        },
    },
    {
        "name": "interpret_logs",
        "description": (
            "Ask Claude to summarize or answer a question about log text. "
            "Call get_run_logs first to obtain the log_text."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "log_text": {
                    "type": "string",
                    "description": "Raw log content to analyze.",
                },
                "question": {
                    "type": "string",
                    "description": "What to ask about the logs.",
                    "default": "Summarize errors and overall status.",
                },
            },
            "required": ["log_text"],
        },
    },
    {
        "name": "watch_run",
        "description": (
            "Monitor a GitHub Actions run in real-time until it completes. "
            "Polls every poll_interval seconds, downloads logs for each job as "
            "it progresses, filters relevant lines, and asks Claude to interpret "
            "them. Prints live updates and returns a final summary. "
            "Use this after triggering a workflow or when the user asks to 'monitor' or 'watch' a run."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "run_id": {
                    "type": "integer",
                    "description": "Numeric GitHub Actions run ID to monitor.",
                },
                "poll_interval": {
                    "type": "integer",
                    "description": "Seconds between polls (default 30).",
                    "default": 30,
                },
                "job_filter": {
                    "type": "string",
                    "description": (
                        "Optional substring: only monitor jobs whose name "
                        "contains this string (case-insensitive). "
                        "E.g. 'agentic', 'get-jobs'."
                    ),
                },
            },
            "required": ["run_id"],
        },
    },
]

TOOL_FN: dict[str, Any] = {
    "list_workflow_runs": list_workflow_runs,
    "get_run_status": get_run_status,
    "get_run_logs": get_run_logs,
    "get_job_logs": get_job_logs,
    "trigger_workflow": trigger_workflow,
    "interpret_logs": interpret_logs,
    "watch_run": watch_run,  # print_fn injected at call time via _execute_tool
}

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = f"""\
You are an assistant for the InferenceMAX_rocm project, an AMD benchmark \
and inference-optimization codebase.

You help the user:
  1. Launch and monitor GitHub Actions workflows (e2e-tests.yml, run-sweep.yml, …).
  2. Fetch, display, and interpret workflow logs to diagnose failures.
  3. Understand benchmark results and CI pipeline structure.
  4. Watch a running workflow live with `watch_run` — use it whenever the user
     says "monitor", "watch", "follow", "ascolta i log", or similar phrases,
     or right after triggering a workflow if the user asked to monitor it.

Conda environment on the host machine: `{cfg.CONDA_ENV}`.
GitHub repo: `{cfg.GITHUB_REPO}`.

Workflow IDs you know about:
  • e2e-tests.yml       — end-to-end agentic/single-node/multi-node tests
  • run-sweep.yml       — full sweep benchmarks
  • benchmark-tmpl.yml  — single-node benchmark template
  • profile.yml         — profiling runs

Guidelines:
  - Respond in the same language the user writes in.
  - For `trigger_workflow`, always ask for explicit confirmation before dispatching.
  - When showing logs, extract the most relevant lines rather than dumping everything.
  - Use `interpret_logs` to summarize failures clearly.
"""

# ---------------------------------------------------------------------------
# Main chat loop
# ---------------------------------------------------------------------------

def _execute_tool(name: str, inputs: dict, console: "TeeConsole | None" = None) -> str:
    fn = TOOL_FN.get(name)
    if fn is None:
        return f"[error] unknown tool: {name}"
    try:
        # Inject live print function for watch_run so output goes to TeeConsole
        if name == "watch_run" and console is not None:
            inputs = {**inputs, "print_fn": console.print}
        result = fn(**inputs)
        if isinstance(result, (dict, list)):
            return json.dumps(result, indent=2, ensure_ascii=False)
        return str(result)
    except Exception as exc:
        return f"[error] {name}: {exc}"


def _display_result(console: TeeConsole, name: str, result_text: str) -> None:
    header = f"[bold cyan]Tool:[/bold cyan] {name}"
    if _RICH:
        console._term.print(Panel(result_text[:4000], title=header, expand=False))  # type: ignore[attr-defined]
    else:
        print(f"--- {name} ---")
        print(result_text[:4000])
    console._write_log(f"TOOL RESULT [{name}]: {result_text[:2000]}")


def _read_user_input(console: TeeConsole) -> str | None:
    """Read one user turn, supporting multiline input.

    Modes:
      - Normal:       single line (default)
      - Block mode:   first line is exactly `\"\"\"` → read lines until a line
                      that is exactly `\"\"\"` closes the block.
      - Continuation: any line ending with `\\` is joined to the next line
                      (the backslash is removed).

    Returns the assembled string, or None on EOF/Ctrl-C.
    """
    try:
        first = console.input("[bold yellow]You>[/bold yellow] ")
    except (KeyboardInterrupt, EOFError):
        return None

    # --- block mode ---
    if first.strip() == '"""':
        console.print(
            '[dim]  (multiline mode — paste text, then type [bold]"""[/bold] on its own line to send)[/dim]'
        )
        lines: list[str] = []
        while True:
            try:
                line = console.input("")
            except (KeyboardInterrupt, EOFError):
                break
            if line.strip() == '"""':
                break
            lines.append(line)
        return "\n".join(lines)

    # --- line continuation mode ---
    parts = [first]
    while parts[-1].endswith("\\"):
        parts[-1] = parts[-1][:-1]   # strip trailing backslash
        try:
            next_line = console.input("[bold yellow]...[/bold yellow] ")
        except (KeyboardInterrupt, EOFError):
            break
        parts.append(next_line)

    return "\n".join(parts)


def main() -> None:
    _set_title("InferenceMAX Orchestrator")
    console = TeeConsole(log_path=cfg.CHAT_SESSION_LOG)

    console.rule("[bold green]InferenceMAX Agentic Orchestrator[/bold green]")
    console.print(
        f"[dim]Model:[/dim] {cfg.CLAUDE_MODEL}  |  "
        f"[dim]Repo:[/dim] {cfg.GITHUB_REPO}  |  "
        f"[dim]Log:[/dim] {cfg.CHAT_SESSION_LOG}"
    )
    console.print(
        '[dim]Type [bold]exit[/bold] to quit. '
        'Use [bold]"""[/bold] on its own line to open/close a multiline block. '
        'End a line with [bold]\\\\[/bold] to continue on the next.[/dim]\n'
    )

    client = cfg.make_anthropic_client()
    history: list[dict] = []

    while True:
        user_input = _read_user_input(console)
        if user_input is None:
            console.print("\n[dim]Goodbye.[/dim]")
            break

        user_input = user_input.strip()
        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit", "q"}:
            console.print("[dim]Goodbye.[/dim]")
            break

        history.append({"role": "user", "content": user_input})

        # Agentic loop: keep calling until no more tool_use blocks
        while True:
            response = client.messages.create(
                model=cfg.CLAUDE_MODEL,
                max_tokens=cfg.MAX_TOKENS,
                system=SYSTEM_PROMPT,
                tools=TOOLS_SCHEMA,
                messages=history,
            )

            # Collect assistant message (may mix text + tool_use blocks)
            assistant_blocks = response.content
            history.append({"role": "assistant", "content": assistant_blocks})

            # Display any text blocks immediately
            for block in assistant_blocks:
                if block.type == "text" and block.text.strip():
                    console.rule()
                    console.render_md(block.text)
                    console._write_log(f"ASSISTANT: {block.text}")

            # If stop_reason is not tool_use → conversation turn is complete
            if response.stop_reason != "tool_use":
                break

            # Execute tool calls and collect results
            tool_results: list[dict] = []
            for block in assistant_blocks:
                if block.type != "tool_use":
                    continue
                tool_name = block.name
                tool_inputs = block.input or {}
                console.print(
                    f"\n[bold magenta]→ Calling tool:[/bold magenta] "
                    f"[cyan]{tool_name}[/cyan] "
                    f"[dim]{json.dumps(tool_inputs, ensure_ascii=False)[:200]}[/dim]"
                )
                result_text = _execute_tool(tool_name, tool_inputs, console=console)
                _display_result(console, tool_name, result_text)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result_text,
                })

            history.append({"role": "user", "content": tool_results})
            # Loop continues → model will process tool results


if __name__ == "__main__":
    main()

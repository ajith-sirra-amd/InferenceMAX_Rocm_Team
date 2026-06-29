#!/usr/bin/env python3
"""
run_and_watch.py — dispatch a GitHub Actions workflow and monitor it live.

Defaults are loaded from  gg_agentic/run_config.yaml  (auto-detected).
Any value can be overridden on the command line.

Usage:
    python gg_agentic/run_and_watch.py [options]

Examples:
    # Use defaults from run_config.yaml (no args needed)
    python gg_agentic/run_and_watch.py

    # Override branch and/or config key
    python gg_agentic/run_and_watch.py --ref chore/agentx-v0.4
    python gg_agentic/run_and_watch.py --config-keys minimaxm2.5-fp4-mi355x-vllm-agentic-lmcache

    # Use a different config file
    python gg_agentic/run_and_watch.py --config gg_agentic/my_other_config.yaml

    # Only watch the "agentic" job, 20s poll interval
    python gg_agentic/run_and_watch.py --job-filter agentic --poll 20

    # Just watch a run that is already running (no dispatch)
    python gg_agentic/run_and_watch.py --run-id 15042

    # Dry-run: print what would be dispatched without doing it
    python gg_agentic/run_and_watch.py --dry-run
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

# Make the gg_agentic package importable when called from the project root
sys.path.insert(0, str(Path(__file__).parent))

import config as cfg
from tools.gh_actions import (
    _gh_get,
    trigger_workflow,
    watch_run,
)

import requests

# ---------------------------------------------------------------------------
# Default config file (sibling of this script)
# ---------------------------------------------------------------------------
_DEFAULT_CONFIG_FILE = Path(__file__).parent / "run_config.yaml"


# ---------------------------------------------------------------------------
# YAML loader — uses stdlib only (no pyyaml required)
# ---------------------------------------------------------------------------

def _load_yaml_simple(path: Path) -> dict:
    """Minimal YAML parser: handles  key: value  lines and # comments.

    Sufficient for flat config files like run_config.yaml.
    Falls back to pyyaml if available for robustness.
    """
    try:
        import yaml  # type: ignore
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except ImportError:
        pass

    result: dict = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.split("#")[0].strip()   # strip comments
            if not line or ":" not in line:
                continue
            key, _, value = line.partition(":")
            key = key.strip()
            value = value.strip().strip("'\"")
            result[key] = value
    return result


def _load_config(config_path: Path) -> dict:
    """Load run_config.yaml; return empty dict if not found."""
    if not config_path.exists():
        return {}
    try:
        data = _load_yaml_simple(config_path)
        print(f"[config] Loaded defaults from {config_path}")
        return data
    except Exception as exc:
        print(f"[config] Warning: could not parse {config_path}: {exc}")
        return {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _current_branch() -> str:
    """Return the current git branch name."""
    try:
        result = subprocess.check_output(
            ["git", "-C", str(Path(__file__).parent.parent),
             "rev-parse", "--abbrev-ref", "HEAD"],
            stderr=subprocess.DEVNULL,
        )
        return result.decode().strip()
    except Exception:
        return "chore/agentx-v0.4"


def _check_runner_busy(runner_label: str, workflow: str) -> list[str]:
    """Return a list of busy job descriptions, empty if the runner is free."""
    busy: list[str] = []
    for status in ("in_progress", "queued"):
        try:
            data = _gh_get(
                f"actions/workflows/{workflow}/runs",
                params={"status": status, "per_page": 20},
            )
        except Exception:
            continue
        for run in data.get("workflow_runs", []):
            rid = run["id"]
            try:
                jobs_data = _gh_get(f"actions/runs/{rid}/jobs")
            except Exception:
                continue
            for j in jobs_data.get("jobs", []):
                labels = j.get("labels", [])
                name   = j.get("name", "").lower()
                jstatus = j.get("status", "")
                if runner_label in labels or runner_label in name:
                    if jstatus in ("in_progress", "queued", "waiting"):
                        busy.append(f"  Run #{rid}: {j['name']} [{jstatus}]")
    return busy


def _latest_run_id(workflow: str) -> int | None:
    """Return the numeric ID of the most recent run of a workflow."""
    try:
        data = _gh_get(f"actions/workflows/{workflow}/runs", params={"per_page": 1})
        runs = data.get("workflow_runs", [])
        return runs[0]["id"] if runs else None
    except Exception:
        return None


def _wait_for_new_run(workflow: str, before_id: int | None, timeout: int = 90) -> int | None:
    """Poll until a new run appears (run ID > before_id). Return its ID."""
    deadline = time.time() + timeout
    print(f"  Waiting for new run to appear (up to {timeout}s)…", flush=True)
    while time.time() < deadline:
        current = _latest_run_id(workflow)
        if current is not None and (before_id is None or current > before_id):
            return current
        time.sleep(5)
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # ------------------------------------------------------------------
    # 1. Pre-parse to find --config so we can load YAML before argparse
    # ------------------------------------------------------------------
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=str(_DEFAULT_CONFIG_FILE))
    pre_args, _ = pre.parse_known_args()
    yaml_cfg = _load_config(Path(pre_args.config))

    # Resolve defaults: YAML file < CLI args
    def _d(key: str, fallback: str = "") -> str:
        return str(yaml_cfg.get(key, fallback))

    def _di(key: str, fallback: int) -> int:
        try:
            return int(yaml_cfg.get(key, fallback))
        except (ValueError, TypeError):
            return fallback

    # ------------------------------------------------------------------
    # 2. Full argument parser with YAML-sourced defaults
    # ------------------------------------------------------------------
    parser = argparse.ArgumentParser(
        description="Dispatch a GitHub Actions workflow and monitor it live.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config",
        default=str(_DEFAULT_CONFIG_FILE),
        metavar="FILE",
        help=f"YAML config file for defaults (default: {_DEFAULT_CONFIG_FILE.name}).",
    )
    parser.add_argument(
        "--ref",
        default=_d("ref"),
        help="Branch or SHA to run on (default from config, or current git branch).",
    )
    parser.add_argument(
        "--config-keys",
        default=_d("config-keys", "kimik2.7-fp4-mi355x-vllm-agentic-lmcache"),
        help="Config key(s) to pass to test-config.",
    )
    parser.add_argument(
        "--config-files",
        default=_d("config-files", ".github/configs/amd-master.yaml"),
        help="Config file path.",
    )
    parser.add_argument(
        "--workflow",
        default=_d("workflow", "e2e-tests.yml"),
        help="Workflow filename.",
    )
    parser.add_argument(
        "--runner",
        default=_d("runner", "mi355x"),
        help="Runner label used to check availability.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Dispatch even if the runner is busy.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be dispatched without actually doing it.",
    )
    parser.add_argument(
        "--run-id",
        type=int,
        default=None,
        help="Skip dispatch and monitor an existing run by ID.",
    )
    parser.add_argument(
        "--poll",
        type=int,
        default=_di("poll-interval", 30),
        help="Seconds between polls during monitoring.",
    )
    parser.add_argument(
        "--job-filter",
        default=_d("job-filter") or None,
        help="Only monitor jobs whose name contains this substring (e.g. 'agentic').",
    )
    parser.add_argument(
        "--no-watch",
        action="store_true",
        help="Dispatch the workflow but do not monitor it.",
    )

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 3. If --run-id given: skip dispatch, monitor directly
    # ------------------------------------------------------------------
    if args.run_id is not None:
        print(f"[run_and_watch] Monitoring existing run #{args.run_id} on {cfg.GITHUB_REPO}")
        watch_run(
            run_id=args.run_id,
            poll_interval=args.poll,
            job_filter=args.job_filter or None,
        )
        return

    # ------------------------------------------------------------------
    # 4. Resolve ref
    # ------------------------------------------------------------------
    ref = args.ref or _current_branch()

    print(f"\n{'='*60}")
    print(f"  run_and_watch.py")
    print(f"  Repo     : {cfg.GITHUB_REPO}")
    print(f"  Ref      : {ref}")
    print(f"  Workflow : {args.workflow}")
    print(f"  Config   : {args.config_keys}")
    print(f"{'='*60}\n")

    # ------------------------------------------------------------------
    # 5. Runner availability check
    # ------------------------------------------------------------------
    print(f"[*] Checking runner '{args.runner}' availability…")
    busy = _check_runner_busy(args.runner, args.workflow)

    if busy:
        if args.force:
            print(f"[!] Runner '{args.runner}' is busy but --force specified. Proceeding.")
        else:
            print(f"\n[ERROR] Runner '{args.runner}' is busy:")
            for b in busy:
                print(b)
            print("\n  Wait for active runs to finish, or use --force to override.")
            sys.exit(1)
    else:
        print(f"[OK] Runner '{args.runner}' is free.\n")

    # ------------------------------------------------------------------
    # 6. Dry-run
    # ------------------------------------------------------------------
    dispatch_input = (
        f"test-config --config-files {args.config_files} --config-keys {args.config_keys}"
    )
    if args.dry_run:
        print("[dry-run] Would dispatch:")
        print(f"  workflow : {args.workflow}")
        print(f"  ref      : {ref}")
        print(f"  input    : generate-cli-command={dispatch_input!r}")
        return

    # ------------------------------------------------------------------
    # 7. Dispatch
    # ------------------------------------------------------------------
    before_id = _latest_run_id(args.workflow)
    print(f"[*] Dispatching {args.workflow} on ref '{ref}'…")

    try:
        result = trigger_workflow(
            workflow_id=args.workflow,
            ref=ref,
            inputs={"generate-cli-command": dispatch_input},
        )
        print(f"[OK] {result['message']}")
    except Exception as exc:
        print(f"[ERROR] Dispatch failed: {exc}")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 8. Optionally skip monitoring
    # ------------------------------------------------------------------
    if args.no_watch:
        print("\n  (--no-watch: skipping monitoring)")
        return

    # ------------------------------------------------------------------
    # 9. Wait for the new run, then monitor
    # ------------------------------------------------------------------
    new_run_id = _wait_for_new_run(args.workflow, before_id, timeout=120)
    if new_run_id is None:
        print("[ERROR] New run did not appear within 120s. Check GitHub Actions manually.")
        print(f"  https://github.com/{cfg.GITHUB_REPO}/actions/workflows/{args.workflow}")
        sys.exit(1)

    print(f"[OK] New run detected: #{new_run_id}")
    print(f"     https://github.com/{cfg.GITHUB_REPO}/actions/runs/{new_run_id}\n")

    watch_run(
        run_id=new_run_id,
        poll_interval=args.poll,
        job_filter=args.job_filter or None,
    )


if __name__ == "__main__":
    main()

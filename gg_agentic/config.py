"""
Centralized configuration for the InferenceMAX agentic orchestrator.
Edit paths and env-vars here; everything else reads from this module.
"""

import os
from pathlib import Path


def _load_env_file() -> None:
    """Load <project_root>/.env into os.environ (existing vars are NOT overwritten)."""
    env_path = Path(__file__).parent.parent / ".env"
    if not env_path.exists():
        return
    with open(env_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip("'\"")
            if key and key not in os.environ:
                os.environ[key] = value


_load_env_file()

# ---------------------------------------------------------------------------
# Anthropic API / AMD internal proxy
# The SDK picks up ANTHROPIC_BASE_URL automatically when set.
# ---------------------------------------------------------------------------
ANTHROPIC_API_KEY: str = os.environ.get("ANTHROPIC_API_KEY", "")

CLAUDE_MODEL: str = (
    os.environ.get("ANTHROPIC_MODEL")
    or os.environ.get("ANTHROPIC_DEFAULT_SONNET_MODEL")
    or "claude-sonnet-4-6"
)

MAX_TOKENS: int = 8192


def _parse_custom_headers() -> dict[str, str]:
    raw = os.environ.get("ANTHROPIC_CUSTOM_HEADERS", "")
    if not raw:
        return {}
    headers: dict[str, str] = {}
    for part in raw.split(","):
        part = part.strip()
        if ":" in part:
            k, _, v = part.partition(":")
            headers[k.strip()] = v.strip()
    return headers


def make_anthropic_client():
    """Return a configured anthropic.Anthropic client (public API or AMD proxy)."""
    import anthropic

    extra_headers = _parse_custom_headers()
    base_url = os.environ.get("ANTHROPIC_BASE_URL", "")

    if base_url and extra_headers:
        return anthropic.Anthropic(
            api_key=ANTHROPIC_API_KEY or "dummy",
            default_headers=extra_headers,
        )

    if not ANTHROPIC_API_KEY:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY is not set.\n"
            "  • Public API: set ANTHROPIC_API_KEY.\n"
            "  • AMD proxy:  set ANTHROPIC_BASE_URL + ANTHROPIC_CUSTOM_HEADERS."
        )
    return anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)


# ---------------------------------------------------------------------------
# GitHub
# ---------------------------------------------------------------------------
GITHUB_TOKEN: str = os.environ.get("GITHUB_TOKEN", "")
GITHUB_REPO: str = os.environ.get(
    "GITHUB_REPO", "ROCm/InferenceMAX_rocm"
)

# ---------------------------------------------------------------------------
# Conda environment used on the remote host
# ---------------------------------------------------------------------------
CONDA_ENV: str = os.environ.get("INFERENCEMAX_CONDA_ENV", "auto_sglang")

# ---------------------------------------------------------------------------
# Chat session log
# ---------------------------------------------------------------------------
CHAT_SESSION_LOG: Path = Path(
    os.environ.get("INFERENCEMAX_CHAT_LOG", "")
    or Path(__file__).parent.parent / "gg_agentic_session.log"
)

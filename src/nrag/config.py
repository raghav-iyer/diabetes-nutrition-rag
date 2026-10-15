"""Config and secrets loading.

`config.yaml` holds every tunable number; secrets come only from the
environment (optionally via a gitignored `.env`).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "config.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    path = Path(path or os.environ.get("NRAG_CONFIG") or DEFAULT_CONFIG)
    with path.open() as f:
        return yaml.safe_load(f)


def resolve_path(p: str | Path) -> Path:
    """Config paths are relative to the repo root."""
    p = Path(p)
    return p if p.is_absolute() else REPO_ROOT / p


def env(name: str, required: bool = False) -> str | None:
    load_dotenv(REPO_ROOT / ".env", override=False)
    value = os.environ.get(name) or None
    if required and not value:
        raise SystemExit(f"Set {name} in your environment or .env (see .env.example).")
    return value

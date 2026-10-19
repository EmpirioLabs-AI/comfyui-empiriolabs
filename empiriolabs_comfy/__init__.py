"""EmpirioLabs AI nodes for ComfyUI."""

from __future__ import annotations

import re
from pathlib import Path


def _read_version() -> str:
    # pyproject.toml is the one place the version is written; it is read here for the User-Agent.
    try:
        text = (Path(__file__).resolve().parent.parent / "pyproject.toml").read_text(encoding="utf-8")
    except OSError:
        return "unknown"
    match = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE)
    return match.group(1) if match else "unknown"


__version__ = _read_version()

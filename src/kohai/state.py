"""Small persistent state in ~/.kohai (which models were downloaded, where)."""

from __future__ import annotations

import json
import os
from pathlib import Path


def home() -> Path:
    path = Path(os.environ.get("KOHAI_HOME", Path.home() / ".kohai"))
    if not path.exists():
        legacy = Path.home() / ".dojo"  # pre-rename state
        if legacy.is_dir() and "KOHAI_HOME" not in os.environ:
            legacy.rename(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _downloads_file() -> Path:
    return home() / "downloads.json"


def downloads() -> dict[str, str]:
    try:
        return json.loads(_downloads_file().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def mark_downloaded(model_id: str, path: str | Path) -> None:
    data = downloads()
    data[model_id] = str(path)
    _downloads_file().write_text(json.dumps(data, indent=2), encoding="utf-8")


def downloaded_path(model_id: str) -> Path | None:
    p = downloads().get(model_id)
    return Path(p) if p and Path(p).exists() else None

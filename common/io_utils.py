"""Filesystem helpers: project paths, JSON round-trips, model persistence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib


def project_root() -> Path:
    """Repository root (the folder that holds `common/` and the project folders).

    `absolute()` rather than `resolve()`, here and throughout the projects: `resolve()`
    follows symlinks and Windows reparse points, which can rewrite the repository path
    into a different view of itself where generated artefacts are not visible. Every
    module locates its data and model directories the same way, so the whole pipeline
    agrees on where things live.
    """
    return Path(__file__).absolute().parent.parent


def ensure_dirs(*paths: Path) -> None:
    """Create every directory in `paths`, parents included."""
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def save_json(obj: Any, path: Path) -> Path:
    """Write `obj` as indented JSON, creating the parent directory if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2, default=_json_default)
    return path


def load_json(path: Path) -> Any:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def save_model(model: Any, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return path


def load_model(path: Path) -> Any:
    return joblib.load(path)


def _json_default(value: Any) -> Any:
    """Make numpy scalars and Paths JSON-serialisable."""
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON-serialisable: {type(value)!r}")

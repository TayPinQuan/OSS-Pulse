"""Filesystem layout helpers.

All data paths are derived from a single root so tests can point at a tmp dir.
"""

from __future__ import annotations

import os
from pathlib import Path


def data_root() -> Path:
    return Path(os.environ.get("OSSPULSE_DATA_DIR", "data")).resolve()


def raw_dir() -> Path:
    return _ensure(data_root() / "raw")


def staging_dir() -> Path:
    return _ensure(data_root() / "staging")


def analytics_dir() -> Path:
    return _ensure(data_root() / "analytics")


def github_search_dir() -> Path:
    return _ensure(raw_dir() / "github_search")


def github_api_dir() -> Path:
    return _ensure(raw_dir() / "github_api")


def gharchive_dir() -> Path:
    return _ensure(raw_dir() / "gharchive")


def gharchive_parquet_dir() -> Path:
    return _ensure(raw_dir() / "gharchive_parquet")


def pypi_dir() -> Path:
    return _ensure(raw_dir() / "pypi")


def _ensure(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def figures_dir() -> Path:
    return _ensure(data_root().parent / "results" / "figures")

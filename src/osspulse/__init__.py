"""OSS Pulse: survival, forecasting and causal analysis of OSS ecosystems."""

from __future__ import annotations

from dotenv import find_dotenv, load_dotenv

# Load .env once at import time so every entry point (`python -m osspulse.*`,
# `make ...`, tests) sees the same variables without callers needing to
# `source .env` manually. `override=False` keeps any pre-set env vars intact.
_ENV_PATH = find_dotenv(usecwd=True)
if _ENV_PATH:
    load_dotenv(_ENV_PATH, override=False)

__version__ = "0.1.0"

# Re-export the resolved path so callers can log which file was loaded.
ENV_PATH: str | None = _ENV_PATH or None

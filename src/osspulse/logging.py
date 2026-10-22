"""Minimal structured logging. No external deps."""

from __future__ import annotations

import logging
import os
import sys


def setup(level: str | None = None) -> None:
    lvl = (level or os.environ.get("OSSPULSE_LOG_LEVEL", "INFO")).upper()
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s %(levelname)s %(name)s %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
    )
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(lvl)


def get(name: str) -> logging.Logger:
    return logging.getLogger(name)

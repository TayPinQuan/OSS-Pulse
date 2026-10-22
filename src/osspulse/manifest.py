"""Append-only JSONL manifest for hour-level download progress."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class Manifest:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._state: dict[str, dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                hour = rec.get("hour")
                if hour:
                    self._state[hour] = rec

    def get(self, hour: str) -> dict[str, Any] | None:
        return self._state.get(hour)

    def is_completed(self, hour: str) -> bool:
        rec = self._state.get(hour)
        return rec is not None and rec.get("status") == "completed"

    def mark(self, hour: str, status: str, **kwargs: Any) -> None:
        rec = {
            "hour": hour,
            "status": status,
            "ts": datetime.now(UTC).isoformat(),
            **kwargs,
        }
        self._state[hour] = rec
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, separators=(",", ":")) + "\n")

    def summary(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for rec in self._state.values():
            status = rec.get("status", "unknown")
            counts[status] = counts.get(status, 0) + 1
        return counts

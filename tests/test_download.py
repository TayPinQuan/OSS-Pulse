from __future__ import annotations

import gzip
import json
from datetime import UTC, date, datetime
from pathlib import Path

import pyarrow.parquet as pq

from osspulse.download import (
    SCHEMA,
    estimate_total_gb,
    filter_file,
    hour_label,
    hour_url,
    iter_hours,
)
from osspulse.manifest import Manifest


def test_iter_hours_count() -> None:
    hours = list(iter_hours(date(2019, 1, 1), date(2019, 1, 2)))
    assert len(hours) == 48
    assert hours[0] == datetime(2019, 1, 1, 0, tzinfo=UTC)
    assert hours[-1] == datetime(2019, 1, 2, 23, tzinfo=UTC)


def test_hour_label_and_url() -> None:
    dt = datetime(2019, 1, 1, 5, tzinfo=UTC)
    assert hour_label(dt) == "2019-01-01-5"
    assert hour_url("https://x/{date}-{hour}.json.gz", dt) == "https://x/2019-01-01-5.json.gz"


def test_estimate_total_gb_positive() -> None:
    gb = estimate_total_gb(date(2019, 1, 1), date(2019, 12, 31))
    assert 40 < gb < 100  # ~60 GB expected


def test_manifest_append_and_load(tmp_path: Path) -> None:
    p = tmp_path / "m.jsonl"
    m = Manifest(p)
    assert not m.is_completed("2019-01-01-0")
    m.mark("2019-01-01-0", "completed", rows=3)
    m.mark("2019-01-01-1", "failed", error="boom")
    assert m.is_completed("2019-01-01-0")
    assert not m.is_completed("2019-01-01-1")

    m2 = Manifest(p)
    assert m2.is_completed("2019-01-01-0")
    assert m2.summary() == {"completed": 1, "failed": 1}

    m2.mark("2019-01-01-1", "completed", rows=0)
    assert m2.is_completed("2019-01-01-1")


def test_filter_file_keeps_only_candidates(tmp_path: Path) -> None:
    raw = tmp_path / "2019-01-01-0.json.gz"
    events = [
        {
            "id": "1",
            "type": "PushEvent",
            "created_at": "2019-01-01T00:00:00Z",
            "actor": {"id": 10, "login": "alice"},
            "repo": {"id": 100, "name": "a/b"},
            "payload": {"commits": []},
        },
        {
            "id": "2",
            "type": "WatchEvent",
            "created_at": "2019-01-01T00:00:01Z",
            "actor": {"id": 11, "login": "bob"},
            "repo": {"id": 999, "name": "x/y"},
            "payload": {"action": "started"},
        },
    ]
    with gzip.open(raw, "wt", encoding="utf-8") as f:
        for ev in events:
            f.write(json.dumps(ev) + "\n")

    out = tmp_path / "out.parquet"
    n = filter_file(raw, {100}, out)
    assert n == 1
    table = pq.read_table(out)
    assert table.schema == SCHEMA
    assert table.num_rows == 1
    assert table.column("repo_id").to_pylist() == [100]
    assert table.column("type").to_pylist() == ["PushEvent"]

"""GH Archive hourly-dump downloader with streaming filter.

Flow per hour:
  1. Download https://data.gharchive.org/YYYY-MM-DD-H.json.gz to a .part file
  2. Atomically rename to .json.gz
  3. Open gzip, parse each line as JSON, keep events whose repo.id is in the
     candidate set
  4. Write kept events as a Parquet shard (one per hour)
  5. Delete the raw .json.gz
  6. Append a completion record to the manifest

Resumability: on restart, hours already marked `completed` in the manifest are
skipped. `--since` / `--until` override the cohort window for one run only.
"""

from __future__ import annotations

import gzip
import json
import time
from collections.abc import Iterator
from datetime import UTC, date, datetime, timedelta
from datetime import time as dt_time
from pathlib import Path

import orjson
import pyarrow as pa
import pyarrow.parquet as pq
import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from osspulse import paths
from osspulse.config import Config
from osspulse.logging import get
from osspulse.manifest import Manifest

log = get(__name__)

SCHEMA = pa.schema(
    [
        ("id", pa.string()),
        ("type", pa.string()),
        ("created_at", pa.timestamp("us", tz="UTC")),
        ("actor_id", pa.int64()),
        ("actor_login", pa.string()),
        ("repo_id", pa.int64()),
        ("repo_name", pa.string()),
        ("payload_json", pa.string()),
    ]
)

# Conservative per-hour compressed size by year (MB), used only for the
# startup estimate. Actual size is recorded in the manifest per hour.
_EST_MB_PER_HOUR = {
    2019: 7,
    2020: 11,
    2021: 17,
    2022: 20,
    2023: 27,
    2024: 32,
}


# ---------------------------------------------------------------------------
# Hour enumeration
# ---------------------------------------------------------------------------


def iter_hours(start: date, end: date) -> Iterator[datetime]:
    cur = datetime.combine(start, dt_time(0, 0), tzinfo=UTC)
    end_dt = datetime.combine(end, dt_time(23, 0), tzinfo=UTC)
    while cur <= end_dt:
        yield cur
        cur += timedelta(hours=1)


def hour_label(dt: datetime) -> str:
    return f"{dt.year:04d}-{dt.month:02d}-{dt.day:02d}-{dt.hour}"


def hour_url(template: str, dt: datetime) -> str:
    return template.format(date=f"{dt.year:04d}-{dt.month:02d}-{dt.day:02d}", hour=dt.hour)


def hour_parquet_path(dt: datetime) -> Path:
    return (
        paths.gharchive_parquet_dir()
        / f"{dt.year:04d}"
        / f"{dt.month:02d}"
        / f"{dt.day:02d}"
        / f"{dt.hour:02d}.parquet"
    )


def estimate_total_gb(start: date, end: date) -> float:
    total_mb = 0.0
    for h in iter_hours(start, end):
        total_mb += _EST_MB_PER_HOUR.get(h.year, 20)
    return total_mb / 1024.0


# ---------------------------------------------------------------------------
# Candidate pool
# ---------------------------------------------------------------------------


def load_candidate_repo_ids() -> set[int]:
    ids: set[int] = set()
    for p in sorted(paths.github_search_dir().glob("2*.json")):
        if p.name.endswith(".manifest.json"):
            continue
        try:
            data = json.loads(p.read_text())
        except json.JSONDecodeError:
            log.warning("skip unreadable shard: %s", p)
            continue
        for repo in data:
            rid = repo.get("id")
            if isinstance(rid, int):
                ids.add(rid)
    return ids


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------


@retry(
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=2, min=2, max=60),
    reraise=True,
)
def _download(url: str, dest: Path, timeout: float) -> int:
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    total = 0
    with requests.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        with part.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                if chunk:
                    f.write(chunk)
                    total += len(chunk)
    part.replace(dest)
    return total


# ---------------------------------------------------------------------------
# Filter
# ---------------------------------------------------------------------------


def _extract_row(ev: dict) -> dict | None:
    repo = ev.get("repo") or {}
    rid = repo.get("id")
    if rid is None:
        return None
    actor = ev.get("actor") or {}
    try:
        payload = orjson.dumps(ev.get("payload") or {}).decode("utf-8")
    except TypeError:
        payload = "{}"
    return {
        "id": str(ev.get("id", "")),
        "type": ev.get("type"),
        "created_at": _parse_ts(ev.get("created_at")),
        "actor_id": actor.get("id"),
        "actor_login": actor.get("login"),
        "repo_id": rid,
        "repo_name": repo.get("name"),
        "payload_json": payload,
    }


def _parse_ts(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None


def filter_file(raw_path: Path, repo_ids: set[int], out_path: Path) -> int:
    rows: list[dict] = []
    with gzip.open(raw_path, "rb") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                ev = orjson.loads(line)
            except orjson.JSONDecodeError:
                continue
            if (ev.get("repo") or {}).get("id") not in repo_ids:
                continue
            row = _extract_row(ev)
            if row is not None:
                rows.append(row)

    if not rows:
        return 0

    table = pa.Table.from_pylist(rows, schema=SCHEMA)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, out_path, compression="zstd")
    return len(rows)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run(
    cfg: Config,
    *,
    force: bool = False,
    since: date | None = None,
    until: date | None = None,
) -> dict:
    start = since or cfg.cohort.created_from
    end = until or cfg.cohort.observation_end
    if start > end:
        raise ValueError(f"since ({start}) must be <= until ({end})")

    # Warn if the latest cohort has too little follow-up to be classifiable.
    followup_days = (cfg.cohort.observation_end - cfg.cohort.created_to).days
    if followup_days < cfg.cohort.inactivity_days:
        min_end = cfg.cohort.created_to + timedelta(days=cfg.cohort.inactivity_days)
        log.warning(
            "follow-up after created_to is %d days (< inactivity_days=%d); "
            "repos created near %s cannot be classified as inactive and will be "
            "dropped or heavily censored in survival analysis. Consider "
            "observation_end >= %s.",
            followup_days,
            cfg.cohort.inactivity_days,
            cfg.cohort.created_to,
            min_end.isoformat(),
        )

    repo_ids = load_candidate_repo_ids()
    if not repo_ids:
        raise RuntimeError("No candidate repo ids found. Run `make candidates` first.")
    log.info("candidate repo ids: %d", len(repo_ids))

    est_gb = estimate_total_gb(start, end)
    total_hours = sum(1 for _ in iter_hours(start, end))
    log.info(
        "window %s -> %s: %d hours, estimated ~%.0f GB compressed",
        start,
        end,
        total_hours,
        est_gb,
    )

    manifest = Manifest(Path(cfg.download.manifest_path))
    raw_dir = paths.raw_dir() / "gharchive"
    raw_dir.mkdir(parents=True, exist_ok=True)

    done = skipped = failed = rows_total = 0
    bytes_total = 0
    t0 = time.time()

    for i, dt in enumerate(iter_hours(start, end), start=1):
        label = hour_label(dt)
        if not force and manifest.is_completed(label):
            skipped += 1
            if i % 500 == 0:
                log.info(
                    "progress %d/%d (skipped=%d done=%d failed=%d)",
                    i,
                    total_hours,
                    skipped,
                    done,
                    failed,
                )
            continue

        url = hour_url(cfg.download.url_template, dt)
        raw_path = raw_dir / f"{label}.json.gz"
        out_path = hour_parquet_path(dt)

        try:
            nbytes = _download(url, raw_path, cfg.download.request_timeout_sec)
            rows = filter_file(raw_path, repo_ids, out_path)
            if not cfg.download.keep_raw:
                raw_path.unlink(missing_ok=True)
            manifest.mark(label, "completed", rows=rows, bytes=nbytes)
            done += 1
            rows_total += rows
            bytes_total += nbytes
        except Exception as exc:  # noqa: BLE001
            log.warning("hour %s failed: %s", label, exc)
            manifest.mark(label, "failed", error=str(exc)[:500])
            failed += 1
            if not cfg.download.keep_raw:
                raw_path.unlink(missing_ok=True)

        if i % 100 == 0:
            elapsed = time.time() - t0
            rate = i / elapsed if elapsed > 0 else 0
            log.info(
                "progress %d/%d (done=%d skipped=%d failed=%d rows=%d) %.1f hours/s",
                i,
                total_hours,
                done,
                skipped,
                failed,
                rows_total,
                rate,
            )

    summary = {
        "window": {"from": start.isoformat(), "to": end.isoformat()},
        "hours_total": total_hours,
        "completed": done,
        "skipped": skipped,
        "failed": failed,
        "rows_written": rows_total,
        "bytes_downloaded": bytes_total,
        "manifest_status": manifest.summary(),
    }
    log.info("done: %s", summary)
    return summary


def main() -> None:
    import argparse

    from osspulse import logging as oss_log

    parser = argparse.ArgumentParser(description="Download and filter GH Archive hourly dumps.")
    parser.add_argument(
        "--since",
        type=date.fromisoformat,
        default=None,
        help="Override start date (YYYY-MM-DD). Default: cohort.created_from",
    )
    parser.add_argument(
        "--until",
        type=date.fromisoformat,
        default=None,
        help="Override end date (YYYY-MM-DD). Default: cohort.observation_end",
    )
    parser.add_argument(
        "--force", action="store_true", help="Re-download hours already marked completed"
    )
    parser.add_argument("--config", default="configs/cohort.yaml")
    args = parser.parse_args()

    cfg = Config.load(args.config)
    oss_log.setup(cfg.logging.level)
    run(cfg, force=args.force, since=args.since, until=args.until)


if __name__ == "__main__":
    main()

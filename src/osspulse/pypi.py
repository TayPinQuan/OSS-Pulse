"""pypistats.org downloader and daily->monthly aggregator.

Flow:
  1. For each package in configs/packages.yaml, GET
     https://pypistats.org/api/packages/{name}/overall?mirrors=false
  2. Cache the raw response to raw/pypi/{name}.json
  3. Aggregate daily series -> monthly sums, filling missing days with 0
     (pypistats drops zero-download days from the payload).
  4. Write analytics/package_downloads_month.parquet

Resumability: cached packages are skipped unless --force is passed. A JSONL
manifest records the status of each package (ok / 404 / failed).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import requests
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from osspulse import paths
from osspulse.config import Config, PackagesConfig
from osspulse.logging import get
from osspulse.manifest import Manifest

log = get(__name__)

API = "https://pypistats.org/api/packages/{name}/overall"
INTER_REQUEST_SEC = 1.5  # well below the ~30 req/min anonymous limit


class _ServerError(Exception):
    """5xx from pypistats; worth retrying."""


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=5, max=60),
    retry=retry_if_exception_type((requests.ConnectionError, requests.Timeout, _ServerError)),
    reraise=True,
)
def _fetch_one(name: str) -> dict:
    resp = requests.get(
        API.format(name=name),
        params={"mirrors": "false"},
        timeout=30,
        headers={"User-Agent": "osspulse/0.1 (research project)"},
    )
    if resp.status_code == 404:
        return {"_missing": True, "name": name}
    if resp.status_code == 429:
        retry_after = float(resp.headers.get("Retry-After", "60"))
        log.warning("429 on %s; sleeping %ss", name, retry_after)
        time.sleep(retry_after)
        raise _ServerError("rate-limited")
    if resp.status_code >= 500:
        raise _ServerError(f"server error {resp.status_code}")
    resp.raise_for_status()
    return resp.json()


def download_all(cfg: Config, pkg_cfg, *, force: bool = False) -> dict:
    cache_dir = paths.pypi_dir()
    manifest = Manifest(cache_dir / "_download_manifest.jsonl")

    names = pkg_cfg.all_names()
    log.info("packages to fetch: %d", len(names))

    ok = cached = missing = failed = 0
    for i, name in enumerate(names, start=1):
        p = cache_dir / f"{name}.json"
        if p.exists() and not force:
            cached += 1
            continue

        try:
            payload = _fetch_one(name)
            p.write_text(json.dumps(payload))
            if payload.get("_missing"):
                missing += 1
                manifest.mark(name, "missing")
            else:
                ok += 1
                manifest.mark(name, "completed", points=len(payload.get("data") or []))
        except Exception as exc:  # noqa: BLE001
            log.warning("pypi fetch %s failed: %s", name, exc)
            manifest.mark(name, "failed", error=str(exc)[:500])
            failed += 1

        time.sleep(INTER_REQUEST_SEC)
        if i % 10 == 0:
            log.info(
                "progress %d/%d (ok=%d cached=%d missing=%d failed=%d)",
                i,
                len(names),
                ok,
                cached,
                missing,
                failed,
            )

    summary = {
        "total": len(names),
        "ok": ok,
        "cached": cached,
        "missing": missing,
        "failed": failed,
    }
    log.info("download done: %s", summary)
    return summary


def _build_from_clickhouse_csv(csv_path: Path, pkg_cfg: PackagesConfig) -> pd.DataFrame:
    """Read a ClickHouse-exported CSV (project, month, downloads) -> monthly table."""
    df = pd.read_csv(csv_path)
    df.columns = [c.lower().strip() for c in df.columns]
    required = {"project", "month", "downloads"}
    if not required.issubset(df.columns):
        raise ValueError(
            f"ClickHouse CSV must have columns {required}; got {list(df.columns)}. "
            f"Re-export with: SELECT project, toStartOfMonth(date) AS month, count() AS downloads ..."
        )

    df = df.rename(columns={"project": "package"}).copy()
    df["month"] = pd.to_datetime(df["month"], utc=True)
    df["downloads"] = df["downloads"].astype(float)

    wanted = set(pkg_cfg.all_names())
    df = df[df["package"].isin(wanted)].copy()
    df = df.sort_values(["package", "month"]).reset_index(drop=True)

    out = paths.analytics_dir() / "package_downloads_month.parquet"
    df[["package", "month", "downloads"]].to_parquet(out, index=False)
    log.info(
        "clickhouse csv -> package_downloads_month.parquet: %d rows, %d packages, %s -> %s",
        len(df),
        df["package"].nunique(),
        df["month"].min().date(),
        df["month"].max().date(),
    )
    return df


def build_monthly(pkg_cfg: PackagesConfig) -> pd.DataFrame:
    # Prefer the ClickHouse export when present (pypistats only keeps 180 days).
    ch_csv = paths.pypi_dir() / "clickhouse" / "pypi_downloads.csv"
    if ch_csv.exists():
        log.info("using ClickHouse export at %s", ch_csv)
        return _build_from_clickhouse_csv(ch_csv, pkg_cfg)

    log.info("no ClickHouse CSV; falling back to pypistats 180-day cache")
    cache_dir = paths.pypi_dir()
    rows: list[dict] = []
    for name in pkg_cfg.all_names():
        p = cache_dir / f"{name}.json"
        if not p.exists():
            log.warning("missing cache for %s; skipping", name)
            continue
        try:
            payload = json.loads(p.read_text())
        except json.JSONDecodeError:
            log.warning("unreadable cache for %s; skipping", name)
            continue
        if payload.get("_missing"):
            continue

        data = payload.get("data") or []
        if not data:
            continue

        df = pd.DataFrame(data)
        df = df[df["category"] == "without_mirrors"].copy()
        if df.empty:
            continue

        df["date"] = pd.to_datetime(df["date"], utc=True)
        df = df.sort_values("date")

        # Fill missing days with 0 (pypistats drops zero-download days).
        full_range = pd.date_range(df["date"].min(), df["date"].max(), freq="D", tz="UTC")
        df = (
            df.set_index("date").reindex(full_range, fill_value=0).rename_axis("date").reset_index()
        )
        df["month"] = pd.to_datetime(df["date"].dt.strftime("%Y-%m-01"), utc=True)

        monthly = df.groupby("month", as_index=False)["downloads"].sum()
        monthly["package"] = name
        rows.append(monthly)

    if not rows:
        raise RuntimeError("No package data to aggregate. Run the download step first.")

    out = pd.concat(rows, ignore_index=True)[["package", "month", "downloads"]]
    out = out.sort_values(["package", "month"]).reset_index(drop=True)

    path = paths.analytics_dir() / "package_downloads_month.parquet"
    out.to_parquet(path, index=False)
    log.info(
        "analytics/package_downloads_month.parquet: %d rows, %d packages, %s -> %s",
        len(out),
        out["package"].nunique(),
        out["month"].min(),
        out["month"].max(),
    )
    return out


def main() -> None:
    import argparse

    from osspulse import logging as oss_log

    parser = argparse.ArgumentParser(description="Download pypistats data and build monthly table.")
    parser.add_argument("--force", action="store_true", help="Ignore cache and re-fetch")
    parser.add_argument("--packages", default="configs/packages.yaml")
    parser.add_argument("--config", default="configs/cohort.yaml")
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Skip fetching; rebuild the monthly table only",
    )
    args = parser.parse_args()

    cfg = Config.load(args.config)
    oss_log.setup(cfg.logging.level)
    pkg_cfg = Config.load_packages(args.packages)

    if not args.aggregate_only:
        download_all(cfg, pkg_cfg, force=args.force)
    build_monthly(pkg_cfg)


if __name__ == "__main__":
    main()

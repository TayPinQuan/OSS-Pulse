"""GitHub Search API candidate-pool extraction, sharded by creation month.

Output layout:
  data/raw/github_search/
    YYYY-MM.json          # list of repo dicts (fields we keep)
    YYYY-MM.manifest.json # {count, total_count, overflow, pages, fetched_at}
    _funnel.json          # aggregate counts across all shards
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from osspulse import paths
from osspulse.config import Config
from osspulse.logging import get

log = get(__name__)

SEARCH_URL = "https://api.github.com/search/repositories"
KEEP_FIELDS = (
    "id",
    "full_name",
    "name",
    "owner",
    "language",
    "stargazers_count",
    "forks_count",
    "created_at",
    "archived",
    "fork",
    "license",
    "topics",
    "size",
)


@dataclass(frozen=True)
class Shard:
    year: int
    month: int

    @property
    def label(self) -> str:
        return f"{self.year:04d}-{self.month:02d}"

    def date_range(self) -> tuple[date, date]:
        start = date(self.year, self.month, 1)
        end_excl = date(self.year + (self.month == 12), (self.month % 12) + 1, 1)
        return start, end_excl - timedelta(days=1)


def iter_shards(cfg: Config) -> Iterator[Shard]:
    c = cfg.cohort
    y, m = c.created_from.year, c.created_from.month
    while (y, m) <= (c.created_to.year, c.created_to.month):
        yield Shard(y, m)
        m += 1
        if m == 13:
            y, m = y + 1, 1


def build_query(cfg: Config, shard: Shard, min_stars: int) -> str:
    start, end = shard.date_range()
    return (
        f"language:{cfg.cohort.language} "
        f"created:{start.isoformat()}..{end.isoformat()} "
        f"stars:>={min_stars} "
        f"fork:false archived:false"
    )


def _headers() -> dict[str, str]:
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        from osspulse import ENV_PATH

        raise RuntimeError(
            "GITHUB_TOKEN is not set.\n"
            f"  cwd:      {Path.cwd()}\n"
            f"  .env:     {ENV_PATH or '<not found>'}\n"
            "  fix:      put GITHUB_TOKEN=ghp_... in .env at the project root, "
            "then re-run."
        )
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


@retry(
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=2, min=2, max=30),
    reraise=True,
)
def _get_page(url: str, params: dict, session: requests.Session) -> dict:
    resp = session.get(url, params=params, headers=_headers(), timeout=30)
    if resp.status_code == 403 and "rate limit" in resp.text.lower():
        reset = int(resp.headers.get("X-RateLimit-Reset", "0"))
        sleep_s = max(1, reset - int(time.time()))
        log.warning("search rate limited; sleeping %ss", sleep_s)
        time.sleep(sleep_s)
        resp.raise_for_status()
    resp.raise_for_status()
    return resp.json()


def _trim(repo: dict) -> dict:
    return {k: repo.get(k) for k in KEEP_FIELDS}


def fetch_shard(
    cfg: Config,
    shard: Shard,
    session: requests.Session,
    *,
    min_stars: int | None = None,
    force: bool = False,
) -> dict:
    out_dir = paths.github_search_dir()
    shard_path = out_dir / f"{shard.label}.json"
    manifest_path = out_dir / f"{shard.label}.manifest.json"

    if shard_path.exists() and manifest_path.exists() and not force:
        log.info("shard %s cached", shard.label)
        return json.loads(manifest_path.read_text())

    stars = min_stars if min_stars is not None else cfg.cohort.search_min_stars
    query = build_query(cfg, shard, stars)
    per_page = cfg.search_api.per_page
    max_pages = cfg.search_api.max_pages

    items: list[dict] = []
    total_count: int | None = None
    pages_fetched = 0
    overflow = False

    for page in range(1, max_pages + 1):
        payload = _get_page(
            SEARCH_URL,
            {"q": query, "per_page": per_page, "page": page},
            session,
        )
        if total_count is None:
            total_count = payload.get("total_count", 0)
        batch = payload.get("items", [])
        items.extend(_trim(r) for r in batch)
        pages_fetched += 1
        if len(batch) < per_page:
            break
        if page * per_page >= min(total_count, 1000):
            break
        time.sleep(cfg.search_api.min_interval_sec)

    if total_count is not None and total_count > len(items):
        overflow = True
        log.warning(
            "shard %s overflow: total_count=%d fetched=%d (raise stars threshold)",
            shard.label,
            total_count,
            len(items),
        )

    manifest = {
        "shard": shard.label,
        "query": query,
        "min_stars_used": stars,
        "count": len(items),
        "total_count": total_count,
        "overflow": overflow,
        "pages": pages_fetched,
        "fetched_at": datetime.now(UTC).isoformat(),
    }
    shard_path.write_text(json.dumps(items, indent=0))
    manifest_path.write_text(json.dumps(manifest, indent=2))
    log.info("shard %s: fetched %d (total=%s)", shard.label, len(items), total_count)
    return manifest


def fetch_all(cfg: Config, *, force: bool = False) -> dict:
    session = requests.Session()
    manifests: list[dict] = []
    for shard in iter_shards(cfg):
        m = fetch_shard(cfg, shard, session, force=force)
        manifests.append(m)

    funnel = {
        "generated_at": datetime.now(UTC).isoformat(),
        "cohort_window": {
            "from": cfg.cohort.created_from.isoformat(),
            "to": cfg.cohort.created_to.isoformat(),
        },
        "search_min_stars": cfg.cohort.search_min_stars,
        "shards": len(manifests),
        "total_candidates": sum(m["count"] for m in manifests),
        "overflow_shards": [m["shard"] for m in manifests if m["overflow"]],
    }
    (paths.github_search_dir() / "_funnel.json").write_text(json.dumps(funnel, indent=2))
    log.info(
        "funnel: %d shards, %d candidates, %d overflow shards",
        funnel["shards"],
        funnel["total_candidates"],
        len(funnel["overflow_shards"]),
    )
    return funnel


def main() -> None:
    import argparse

    from osspulse import logging as oss_log

    parser = argparse.ArgumentParser(description="Fetch GitHub Search API candidate pool.")
    parser.add_argument(
        "--force", action="store_true", help="Ignore cache and re-fetch every shard"
    )
    parser.add_argument("--config", default="configs/cohort.yaml")
    args = parser.parse_args()

    cfg = Config.load(args.config)
    oss_log.setup(cfg.logging.level)
    fetch_all(cfg, force=args.force)


if __name__ == "__main__":
    main()

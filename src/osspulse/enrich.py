"""GitHub REST API metadata enrichment.

Reads the Search API candidate pool (raw/github_search/*.json), fetches each
repo's metadata via GET /repositories/{id}, caches the trimmed response to
raw/github_api/{id}.json, and writes a Parquet snapshot at
staging/repo_metadata.parquet.

Why /repositories/{id} rather than /repos/{owner}/{name}:
  The Search API result may be years old. Repos get renamed, transferred or
  deleted. Querying by numeric id is stable across renames; querying by name
  would 404 on every renamed repo.

Resumability: each fetched repo is written to cache immediately and recorded
in a JSONL manifest. Re-running skips cached ids. Deleted/private repos are
cached as {"_deleted": true} so we don't re-query them every run.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from osspulse import ENV_PATH, paths
from osspulse.config import Config
from osspulse.logging import get
from osspulse.manifest import Manifest

log = get(__name__)

REPO_BY_ID_URL = "https://api.github.com/repositories/{repo_id}"

# Fields we keep from the API response. Everything else is dropped to keep the
# cache small. `license` is reduced to its spdx_id (see _trim).
KEEP_FIELDS = (
    "id",
    "full_name",
    "name",
    "description",
    "homepage",
    "language",
    "topics",
    "created_at",
    "updated_at",
    "pushed_at",
    "stargazers_count",
    "forks_count",
    "watchers_count",
    "open_issues_count",
    "default_branch",
    "archived",
    "disabled",
    "fork",
    "size",
    "has_issues",
    "has_projects",
    "has_wiki",
    "has_pages",
    "has_discussions",
    "visibility",
)

# Lowest remaining-quota threshold at which we pause and wait for reset.
_RATE_LIMIT_FLOOR = 200
# Small delay between requests to stay well under secondary rate limits.
_INTER_REQUEST_SEC = 0.15


class _ServerError(Exception):
    """5xx from GitHub, worth retrying."""


def _headers() -> dict[str, str]:
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        raise RuntimeError(
            "GITHUB_TOKEN is not set.\n"
            f"  cwd:  {Path.cwd()}\n"
            f"  .env: {ENV_PATH or '<not found>'}\n"
            "  fix: put GITHUB_TOKEN=ghp_... in .env at the project root."
        )
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


# ---------------------------------------------------------------------------
# Candidate loading
# ---------------------------------------------------------------------------


def load_candidates() -> list[dict[str, Any]]:
    """Return deduplicated [{id, full_name}, ...] from every Search shard."""
    seen: dict[int, str] = {}
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
            if isinstance(rid, int) and rid not in seen:
                seen[rid] = repo.get("full_name") or ""
    log.info("candidates: %d unique repos", len(seen))
    return [{"id": rid, "full_name": name} for rid, name in seen.items()]


# ---------------------------------------------------------------------------
# Rate-limit handling
# ---------------------------------------------------------------------------


def _respect_rate_limit(resp: requests.Response) -> None:
    """Sleep if GitHub says we're out of primary quota."""
    remaining = int(resp.headers.get("X-RateLimit-Remaining", "1"))
    reset = int(resp.headers.get("X-RateLimit-Reset", "0"))
    if remaining < _RATE_LIMIT_FLOOR:
        wait = max(1, reset - int(time.time()) + 1)
        log.warning(
            "rate limit low (remaining=%d); sleeping %ss until reset",
            remaining,
            wait,
        )
        time.sleep(wait)


def _trim(raw: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {k: raw.get(k) for k in KEEP_FIELDS if k != "license"}
    lic = raw.get("license") or {}
    out["license_spdx"] = lic.get("spdx_id") if isinstance(lic, dict) else None
    out["license_name"] = lic.get("name") if isinstance(lic, dict) else None
    owner = raw.get("owner") or {}
    out["owner_login"] = owner.get("login") if isinstance(owner, dict) else None
    return out


# ---------------------------------------------------------------------------
# Fetch with retry
# ---------------------------------------------------------------------------


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=2, max=30),
    retry=retry_if_exception_type((requests.ConnectionError, requests.Timeout, _ServerError)),
    reraise=True,
)
def _get_repo(session: requests.Session, repo_id: int) -> tuple[dict[str, Any] | None, int]:
    """Return (trimmed metadata or None if deleted, http_status)."""
    resp = session.get(
        REPO_BY_ID_URL.format(repo_id=repo_id),
        headers=_headers(),
        timeout=30,
    )

    # 404 => repo deleted or made private. Cache the negative result.
    if resp.status_code == 404:
        return None, 404

    # 403 with rate-limit body: sleep until reset, then retry.
    if resp.status_code == 403 and "rate limit" in resp.text.lower():
        reset = int(resp.headers.get("X-RateLimit-Reset", "0"))
        wait = max(1, reset - int(time.time()) + 1)
        log.warning("403 rate-limited on id=%d; sleeping %ss", repo_id, wait)
        time.sleep(wait)
        raise _ServerError(f"rate-limited; retry after {wait}s")

    # 429 secondary rate limit: honour Retry-After.
    if resp.status_code == 429:
        retry_after = float(resp.headers.get("Retry-After", "60"))
        log.warning("429 on id=%d; sleeping %ss", repo_id, retry_after)
        time.sleep(retry_after)
        raise _ServerError("secondary rate-limit")

    if resp.status_code >= 500:
        raise _ServerError(f"server error {resp.status_code}")

    resp.raise_for_status()
    _respect_rate_limit(resp)
    return _trim(resp.json()), resp.status_code


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def _cache_path(repo_id: int) -> Path:
    return paths.github_api_dir() / f"{repo_id}.json"


def _write_cache(repo_id: int, payload: dict[str, Any] | None) -> None:
    p = _cache_path(repo_id)
    if payload is None:
        p.write_text(json.dumps({"_deleted": True, "id": repo_id}))
    else:
        p.write_text(json.dumps(payload, separators=(",", ":")))


def _read_cache(repo_id: int) -> dict[str, Any] | None:
    p = _cache_path(repo_id)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError:
        return None


def run(cfg: Config, *, force: bool = False, limit: int | None = None) -> dict:
    candidates = load_candidates()
    if limit is not None:
        candidates = candidates[:limit]
        log.info("limit applied: %d repos", limit)

    manifest_path = paths.github_api_dir() / "_enrich_manifest.jsonl"
    manifest = Manifest(manifest_path)

    session = requests.Session()
    done = cached = deleted = failed = 0
    t0 = time.time()

    for i, c in enumerate(candidates, start=1):
        rid = c["id"]
        if not force:
            existing = _read_cache(rid)
            if existing is not None:
                cached += 1
                if i % 200 == 0:
                    log.info("progress %d/%d (cached=%d done=%d)", i, len(candidates), cached, done)
                continue

        try:
            payload, status = _get_repo(session, rid)
            _write_cache(rid, payload)
            if payload is None:
                deleted += 1
                manifest.mark(str(rid), "deleted", http=status, full_name=c.get("full_name"))
            else:
                done += 1
                manifest.mark(
                    str(rid),
                    "completed",
                    http=status,
                    language=payload.get("language"),
                    license=payload.get("license_spdx"),
                )
        except Exception as exc:  # noqa: BLE001
            log.warning("id=%d failed: %s", rid, exc)
            manifest.mark(str(rid), "failed", error=str(exc)[:500])
            failed += 1

        time.sleep(_INTER_REQUEST_SEC)
        if i % 100 == 0:
            elapsed = time.time() - t0
            rate = (done + deleted + failed) / elapsed if elapsed > 0 else 0
            log.info(
                "progress %d/%d (cached=%d done=%d deleted=%d failed=%d) %.1f req/s",
                i,
                len(candidates),
                cached,
                done,
                deleted,
                failed,
                rate,
            )

    summary = {
        "total_candidates": len(candidates),
        "cached": cached,
        "fetched": done,
        "deleted": deleted,
        "failed": failed,
        "manifest_status": manifest.summary(),
    }
    log.info("done: %s", summary)
    return summary


# ---------------------------------------------------------------------------
# Snapshot
# ---------------------------------------------------------------------------


def materialise() -> Path:
    """Aggregate the JSON cache into staging/repo_metadata.parquet."""
    rows: list[dict[str, Any]] = []
    for p in sorted(paths.github_api_dir().glob("*.json")):
        if p.name.startswith("_"):
            continue
        try:
            data = json.loads(p.read_text())
        except json.JSONDecodeError:
            continue
        if data.get("_deleted"):
            data = {"id": data.get("id"), "_deleted": True}
        else:
            data["_deleted"] = False
        rows.append(data)

    df = pd.DataFrame(rows)
    out = paths.staging_dir() / "repo_metadata.parquet"
    df.to_parquet(out, index=False)
    log.info("materialised %d rows -> %s", len(df), out)
    return out


def main() -> None:
    import argparse

    from osspulse import logging as oss_log

    parser = argparse.ArgumentParser(description="Fetch GitHub repo metadata (cached).")
    parser.add_argument("--force", action="store_true", help="Ignore cache and re-fetch")
    parser.add_argument(
        "--limit", type=int, default=None, help="Fetch only the first N repos (for smoke tests)"
    )
    parser.add_argument(
        "--materialise-only",
        action="store_true",
        help="Skip fetching; just rebuild the Parquet snapshot from cache",
    )
    parser.add_argument("--config", default="configs/cohort.yaml")
    args = parser.parse_args()

    cfg = Config.load(args.config)
    oss_log.setup(cfg.logging.level)

    if args.materialise_only:
        materialise()
        return

    run(cfg, force=args.force, limit=args.limit)
    materialise()


if __name__ == "__main__":
    main()

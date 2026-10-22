"""Staging layer: dedupe, bot filter, ForkEvent exclusion, payload parsing.

Inputs (read):
  data/raw/gharchive_parquet/**/*.parquet  — filtered cohort events
  data/raw/gharchive/_download_manifest.jsonl
  staging/repo_metadata.parquet            — from `make enrich`
  configs/cohort.yaml                      — bot list, inactivity days

Outputs (write):
  staging/events.parquet        — deduped, bot-filtered, ForkEvent excluded
  staging/events_typed.parquet  — events.parquet + parsed payload scalars
  staging/ci_commits.parquet    — PushEvent commits touching .github/workflows/
  staging/staging_report.json   — row counts at each step
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import duckdb
import orjson
import pandas as pd

from osspulse import paths
from osspulse.config import Config
from osspulse.logging import get

log = get(__name__)


# ---------------------------------------------------------------------------
# Events: dedupe + bot filter + ForkEvent exclusion
# ---------------------------------------------------------------------------


def _bot_sql(known: list[str]) -> str:
    quoted = ", ".join(f"'{x.lower()}'" for x in known)
    return f"""
        (actor_login IS NULL
         OR (actor_login NOT LIKE '%[bot]%'
             AND LOWER(actor_login) NOT IN ({quoted or "''"})))
    """


def build_events(cfg: Config) -> Path:
    src = paths.gharchive_parquet_dir()
    out = paths.staging_dir() / "events.parquet"

    con = duckdb.connect()
    bot_clause = _bot_sql(cfg.bots.known_logins)
    con.execute(
        f"""
        COPY (
            WITH deduped AS (
                SELECT *,
                       ROW_NUMBER() OVER (
                           PARTITION BY id ORDER BY created_at ASC
                       ) AS rn
                FROM read_parquet('{src}/**/*.parquet', hive_partitioning=true)
            )
            SELECT id, type, created_at, actor_id, actor_login, repo_id, repo_name, payload_json
            FROM deduped
            WHERE rn = 1
              AND type != 'ForkEvent'
              AND {bot_clause}
        ) TO '{out}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )
    row = con.execute(f"SELECT COUNT(*) FROM '{out}'").fetchone()
    assert row is not None
    n = row[0]
    log.info("staging/events.parquet: %d rows", n)
    con.close()
    return out


# ---------------------------------------------------------------------------
# Events typed: parse payload scalars with DuckDB JSON functions
# ---------------------------------------------------------------------------


def build_events_typed(cfg: Config) -> Path:
    src = paths.staging_dir() / "events.parquet"
    out = paths.staging_dir() / "events_typed.parquet"

    con = duckdb.connect()
    con.execute(
        f"""
        COPY (
            SELECT
                id,
                type,
                created_at,
                actor_id,
                actor_login,
                repo_id,
                repo_name,
                json_extract_string(payload_json, '$.action')      AS action,
                json_extract_string(payload_json, '$.issue.number') AS issue_number,
                TRY_CAST(json_extract_string(payload_json, '$.issue.created_at') AS TIMESTAMPTZ)
                    AS issue_created_at,
                TRY_CAST(json_extract(payload_json, '$.pull_request.merged') AS BOOLEAN)
                    AS pr_merged,
                TRY_CAST(json_extract_string(payload_json, '$.pull_request.number') AS BIGINT)
                    AS pr_number,
                TRY_CAST(json_extract_string(payload_json, '$.pull_request.created_at') AS TIMESTAMPTZ)
                    AS pr_created_at,
                payload_json
            FROM read_parquet('{src}')
        ) TO '{out}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )
    row = con.execute(f"SELECT COUNT(*) FROM '{out}'").fetchone()
    assert row is not None
    n = row[0]
    log.info("staging/events_typed.parquet: %d rows", n)
    con.close()
    return out


# ---------------------------------------------------------------------------
# CI commits: PushEvent commits touching .github/workflows/
# ---------------------------------------------------------------------------


def build_ci_commits(cfg: Config) -> Path:
    src = paths.staging_dir() / "events_typed.parquet"
    out = paths.staging_dir() / "ci_commits.parquet"

    push = pd.read_parquet(
        src,
        columns=["id", "repo_id", "created_at", "payload_json"],
        filters=[("type", "=", "PushEvent")],
    )
    log.info("PushEvent rows to scan: %d", len(push))

    rows: list[dict] = []
    for ev in push.itertuples(index=False):
        try:
            payload = orjson.loads(ev.payload_json or b"{}")
        except (orjson.JSONDecodeError, TypeError):
            continue
        commits = payload.get("commits") or []
        for c in commits:
            paths_in_commit = list(c.get("added") or []) + list(c.get("modified") or [])
            workflow_paths = [
                p
                for p in paths_in_commit
                if isinstance(p, str) and p.startswith(".github/workflows/")
            ]
            if workflow_paths:
                rows.append(
                    {
                        "event_id": ev.id,
                        "repo_id": ev.repo_id,
                        "event_created_at": ev.created_at,
                        "commit_sha": c.get("sha"),
                        "workflow_paths": workflow_paths,
                    }
                )
                break

    df = pd.DataFrame(
        rows, columns=["event_id", "repo_id", "event_created_at", "commit_sha", "workflow_paths"]
    )
    df.to_parquet(out, index=False)
    log.info("staging/ci_commits.parquet: %d commits touching .github/workflows/", len(df))
    return out


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def _scalar(con: duckdb.DuckDBPyConnection, sql: str) -> int:
    """Execute a COUNT-style query and return the single scalar result."""
    row = con.execute(sql).fetchone()
    if row is None:
        raise RuntimeError(f"Query returned no rows: {sql}")
    return int(row[0])

def write_report(cfg: Config) -> Path:
    con = duckdb.connect()
    events = paths.staging_dir() / "events.parquet"
    typed = paths.staging_dir() / "events_typed.parquet"
    ci = paths.staging_dir() / "ci_commits.parquet"
    metadata = paths.staging_dir() / "repo_metadata.parquet"

    report: dict[str, object] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "rows": {
            "events": _scalar(con, f"SELECT COUNT(*) FROM '{events}'"),
            "events_typed": _scalar(con, f"SELECT COUNT(*) FROM '{typed}'"),
            "ci_commits": _scalar(con, f"SELECT COUNT(*) FROM '{ci}'"),
            "repos_in_metadata": _scalar(con, f"SELECT COUNT(*) FROM '{metadata}'"),
            "repos_deleted": _scalar(
                con, f"SELECT COUNT(*) FROM '{metadata}' WHERE _deleted = TRUE"
            ),
        },
        "event_types": con.execute(
            f"SELECT type, COUNT(*) AS n FROM '{events}' GROUP BY type ORDER BY n DESC"
        ).df().to_dict(orient="records"),
        "distinct_repos_with_events": _scalar(
            con, f"SELECT COUNT(DISTINCT repo_id) FROM '{events}'"
        ),
    }
    con.close()

    out = paths.staging_dir() / "staging_report.json"
    out.write_text(json.dumps(report, indent=2))
    log.info("staging_report written to %s", out)
    return out


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    import argparse

    from osspulse import logging as oss_log

    parser = argparse.ArgumentParser(description="Build staging tables from raw Parquet.")
    parser.add_argument("--config", default="configs/cohort.yaml")
    args = parser.parse_args()

    cfg = Config.load(args.config)
    oss_log.setup(cfg.logging.level)

    build_events(cfg)
    build_events_typed(cfg)
    build_ci_commits(cfg)
    write_report(cfg)


if __name__ == "__main__":
    main()

"""Analytics tables: repo_survival, repo_month_panel, ci_adoption.

Inputs:
  staging/events_typed.parquet
  staging/ci_commits.parquet
  staging/repo_metadata.parquet

Outputs:
  analytics/repo_survival.parquet
  analytics/repo_month_panel.parquet
  analytics/ci_adoption.parquet
  analytics/cohort_funnel.json
"""

from __future__ import annotations

import json

import duckdb
import numpy as np
import pandas as pd

from osspulse import paths
from osspulse.config import Config
from osspulse.logging import get

log = get(__name__)

FEATURE_WINDOW_DAYS = 90


def _stratified_sample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Proportional stratified sample by creation_year, trimmed to exactly n."""
    df = df.copy()
    df["creation_year"] = pd.to_datetime(df["created_at"], utc=True).dt.year
    total = len(df)
    parts: list[pd.DataFrame] = []
    for _, group in df.groupby("creation_year"):
        target = max(1, int(round(n * len(group) / total)))
        target = min(target, len(group))
        parts.append(group.sample(n=target, random_state=seed))
    out = pd.concat(parts, ignore_index=True)
    if len(out) > n:
        out = out.sample(n=n, random_state=seed).reset_index(drop=True)
    return out.drop(columns=["creation_year"])


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


def _load_metadata() -> pd.DataFrame:
    """Load non-deleted metadata.

    The `_deleted` column is written by `make enrich`, but this function also
    accepts metadata files without it (e.g. hand-built test fixtures or older
    caches) and treats every row as non-deleted.
    """
    df = pd.read_parquet(paths.staging_dir() / "repo_metadata.parquet")
    df = df[~df["_deleted"]].copy() if "_deleted" in df.columns else df.copy()
    df["created_at_ts"] = pd.to_datetime(df["created_at"], utc=True)
    return df


# ---------------------------------------------------------------------------
# CI adoption
# ---------------------------------------------------------------------------


def build_ci_adoption() -> pd.DataFrame:
    """Repos whose first .github/workflows/ commit fell inside 2020-2022."""
    src = paths.staging_dir() / "ci_commits.parquet"
    if not src.exists():
        log.warning("ci_commits.parquet not found; writing empty ci_adoption")
        empty = pd.DataFrame(
            {
                "repo_id": pd.Series([], dtype="int64"),
                "first_workflow_date": pd.Series([], dtype="datetime64[ns, UTC]"),
            }
        )
        out = paths.analytics_dir() / "ci_adoption.parquet"
        empty.to_parquet(out, index=False)
        return empty

    con = duckdb.connect()
    df = con.execute(
        f"""
        SELECT repo_id, MIN(event_created_at) AS first_workflow_date
        FROM read_parquet('{src}')
        GROUP BY repo_id
        """
    ).df()
    con.close()

    df["first_workflow_date"] = pd.to_datetime(df["first_workflow_date"], utc=True)
    lo = pd.Timestamp("2020-01-01", tz="UTC")
    hi = pd.Timestamp("2022-12-31", tz="UTC")
    df = df[(df["first_workflow_date"] >= lo) & (df["first_workflow_date"] <= hi)].copy()

    out = paths.analytics_dir() / "ci_adoption.parquet"
    df.to_parquet(out, index=False)
    log.info("analytics/ci_adoption.parquet: %d treated repos", len(df))
    return df


# ---------------------------------------------------------------------------
# Survival
# ---------------------------------------------------------------------------


def build_repo_survival(cfg: Config) -> tuple[pd.DataFrame, dict]:
    """One row per candidate repo, with first-90-day features and survival fields.

    Applies the inclusion threshold (PRD 4.2) on first-90-day activity, then a
    stratified sample by creation year down to `cfg.cohort.sample_size`. The
    sample happens AFTER inclusion so the reported n is the analysed n.
    """
    meta = _load_metadata()
    obs_end = pd.Timestamp(cfg.cohort.observation_end, tz="UTC")
    inactivity_td = pd.Timedelta(days=cfg.cohort.inactivity_days)

    meta_df = meta[
        [
            "id",
            "created_at_ts",
            "language",
            "license_spdx",
            "description",
            "topics",
            "owner_login",
        ]
    ].rename(columns={"id": "repo_id", "created_at_ts": "created_at"})

    con = duckdb.connect()
    events_path = paths.staging_dir() / "events_typed.parquet"

    # Normalise string/boolean columns so that an all-null column in the
    # parquet file (common in test fixtures, possible in real data if a
    # column happens to be empty) is still usable in SQL string comparisons.
    # DuckDB maps an all-null parquet column to INT32 by default, which
    # breaks `action = 'started'`-style filters.
    con.execute(
        f"""
        CREATE OR REPLACE VIEW events AS
        SELECT
            id,
            CAST(type AS VARCHAR)          AS type,
            created_at,
            actor_id,
            CAST(actor_login AS VARCHAR)   AS actor_login,
            repo_id,
            repo_name,
            CAST(action AS VARCHAR)        AS action,
            issue_number,
            issue_created_at,
            CAST(pr_merged AS BOOLEAN)     AS pr_merged,
            pr_number,
            pr_created_at,
            payload_json
        FROM read_parquet('{events_path}')
        """
    )

    con.register("meta_df", meta_df)

    # ---- First 90 days aggregates ----------------------------------------
    con.execute(
        f"""
        CREATE TEMP TABLE early AS
        SELECT e.*, m.created_at AS repo_created_at
        FROM events e
        JOIN meta_df m USING (repo_id)
        WHERE e.created_at BETWEEN m.created_at
                               AND m.created_at + INTERVAL '{FEATURE_WINDOW_DAYS} days'
        """
    )

    features = con.execute(
        """
        SELECT
            m.repo_id,
            m.created_at,
            m.language,
            m.license_spdx,
            m.owner_login,
            LENGTH(COALESCE(m.description, ''))               AS description_length,
            COALESCE(len(m.topics), 0)                        AS topics_count,

            COALESCE(COUNT(*) FILTER (
                WHERE e.type = 'WatchEvent'
                  AND (e.action = 'started' OR e.action IS NULL)
            ), 0) AS stars_90d,

            COALESCE(COUNT(DISTINCT e.actor_login) FILTER (
                WHERE e.actor_login IS DISTINCT FROM m.owner_login
                  AND e.type IN ('PushEvent','PullRequestEvent','IssuesEvent','PullRequestReviewEvent')
            ), 0) AS contributors_90d,

            COALESCE(COUNT(*) FILTER (
                WHERE e.type = 'IssuesEvent' AND e.action = 'opened'
            ), 0) AS issues_opened_90d,

            COALESCE(COUNT(*) FILTER (
                WHERE e.type = 'PullRequestEvent' AND e.action = 'opened'
            ), 0) AS prs_opened_90d,

            COALESCE(COUNT(*) FILTER (
                WHERE e.type = 'PushEvent'
            ), 0) AS pushes_90d,

            COUNT(*) FILTER (
                WHERE e.type = 'PullRequestEvent' AND e.action = 'closed' AND e.pr_merged
            )::DOUBLE
              / NULLIF(COUNT(*) FILTER (
                    WHERE e.type = 'PullRequestEvent' AND e.action = 'closed'
                ), 0) AS pr_merge_rate_90d,

            COALESCE(COUNT(*), 0) AS total_activity_90d

        FROM meta_df m
        LEFT JOIN early e USING (repo_id)
        GROUP BY m.repo_id, m.created_at, m.language, m.license_spdx,
                 m.owner_login, m.description, m.topics
        """
    ).df()

    # ---- Last active timestamp per repo ----------------------------------
    last_active = con.execute(
        """
        SELECT repo_id, MAX(created_at) AS last_active_at
        FROM events
        WHERE type IN ('PushEvent','PullRequestEvent','IssuesEvent')
        GROUP BY repo_id
        """
    ).df()
    last_active["last_active_at"] = pd.to_datetime(last_active["last_active_at"], utc=True)

    # ---- Median first-response time on issues in first 90 days -----------
    responses = con.execute(
        """
        WITH opens AS (
            SELECT repo_id, issue_number, MIN(created_at) AS opened_at
            FROM early
            WHERE type = 'IssuesEvent' AND action = 'opened' AND issue_number IS NOT NULL
            GROUP BY repo_id, issue_number
        ),
        replies AS (
            SELECT e.repo_id, e.issue_number, MIN(e.created_at) AS replied_at
            FROM early e
            WHERE e.issue_number IS NOT NULL
              AND (e.type = 'IssueCommentEvent'
                   OR (e.type = 'IssuesEvent' AND e.action = 'closed'))
            GROUP BY e.repo_id, e.issue_number
        )
        SELECT o.repo_id,
               MEDIAN(date_diff('hour', o.opened_at, r.replied_at)) AS median_response_hours
        FROM opens o
        JOIN replies r USING (repo_id, issue_number)
        WHERE r.replied_at > o.opened_at
        GROUP BY o.repo_id
        """
    ).df()

    con.close()

    # ---- Assemble --------------------------------------------------------
    df = features.merge(last_active, on="repo_id", how="left")
    df = df.merge(responses, on="repo_id", how="left")

    # ---- Inclusion flags -------------------------------------------------
    df["included"] = (df["stars_90d"] >= cfg.cohort.inclusion_min_stars_90d) & (
        df["contributors_90d"] >= cfg.cohort.inclusion_min_contributors_90d
    )

    # ---- Survival fields -------------------------------------------------
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True)
    df["last_active_at"] = pd.to_datetime(df["last_active_at"], utc=True, errors="coerce")

    has_active = df["last_active_at"].notna()
    event_time = df["last_active_at"] + inactivity_td
    fallback = df["created_at"] + inactivity_td
    event_time = event_time.where(has_active, fallback)

    df["event_flag"] = event_time <= obs_end
    df["duration_days"] = (
        event_time.where(df["event_flag"], obs_end) - df["created_at"]
    ).dt.days.clip(lower=0)

    df["inactivity_days"] = cfg.cohort.inactivity_days
    df["observation_end"] = obs_end.date()

    # ---- Filter to included, then stratified sample ----------------------
    survival = df[df["included"]].copy().reset_index(drop=True)
    n_pre_sample = len(survival)

    if n_pre_sample > cfg.cohort.sample_size:
        survival = _stratified_sample(survival, cfg.cohort.sample_size, cfg.cohort.sample_seed)
        log.info(
            "sampled %d -> %d repos (stratified by creation year, seed=%d)",
            n_pre_sample,
            len(survival),
            cfg.cohort.sample_seed,
        )
    else:
        log.info(
            "no sampling needed: %d included <= sample_size %d",
            n_pre_sample,
            cfg.cohort.sample_size,
        )

    out = paths.analytics_dir() / "repo_survival.parquet"
    survival.to_parquet(out, index=False)
    log.info(
        "analytics/repo_survival.parquet: %d included (pre-sample %d) / %d candidates",
        len(survival),
        n_pre_sample,
        len(df),
    )

    # ---- Funnel ----------------------------------------------------------
    funnel = {
        "metadata_repos": int(len(meta)),
        "with_any_active_event": int(has_active.sum()),
        "included_stars_90d": int((df["stars_90d"] >= cfg.cohort.inclusion_min_stars_90d).sum()),
        "included_contributors_90d": int(
            (df["contributors_90d"] >= cfg.cohort.inclusion_min_contributors_90d).sum()
        ),
        "included_both_pre_sample": int(n_pre_sample),
        "sampled_to": int(len(survival)),
        "sample_seed": int(cfg.cohort.sample_seed),
        "events_observed": int(survival["event_flag"].sum()),
        "censored": int((~survival["event_flag"]).sum()),
        "median_duration_days": float(survival["duration_days"].median()) if len(survival) else 0.0,
    }
    return survival, funnel


# ---------------------------------------------------------------------------
# Monthly panel
# ---------------------------------------------------------------------------


def build_repo_month_panel(
    cfg: Config,
    survival: pd.DataFrame,
    ci: pd.DataFrame,
) -> pd.DataFrame:
    obs_end = pd.Timestamp(cfg.cohort.observation_end, tz="UTC")

    # ---- Month spine -----------------------------------------------------
    spine_rows: list[dict] = []
    for r in survival.itertuples(index=False):
        start = pd.Timestamp(r.created_at)
        start = start.tz_localize("UTC") if start.tzinfo is None else start.tz_convert("UTC")
        start = start.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        months = pd.date_range(start, obs_end, freq="MS", tz="UTC")
        for m in months:
            spine_rows.append({"repo_id": int(r.repo_id), "month": m})
    spine = pd.DataFrame(spine_rows)

    # ---- Monthly aggregates ---------------------------------------------
    con = duckdb.connect()
    events_path = paths.staging_dir() / "events_typed.parquet"
    con.execute(
        f"""
        CREATE OR REPLACE VIEW events AS
        SELECT
            id,
            CAST(type AS VARCHAR)          AS type,
            created_at,
            actor_id,
            CAST(actor_login AS VARCHAR)   AS actor_login,
            repo_id,
            repo_name,
            CAST(action AS VARCHAR)        AS action,
            issue_number,
            issue_created_at,
            CAST(pr_merged AS BOOLEAN)     AS pr_merged,
            pr_number,
            pr_created_at,
            payload_json
        FROM read_parquet('{events_path}')
        """
    )
    monthly = con.execute(
        """
        SELECT
            repo_id,
            DATE_TRUNC('month', created_at) AS month,
            COUNT(*) AS activity_count,
            COUNT(DISTINCT actor_login) AS contributor_count,
            COUNT(*) FILTER (WHERE type = 'PullRequestEvent' AND pr_merged) AS merged_pr_count,
            COUNT(*) FILTER (WHERE type = 'IssuesEvent' AND action = 'opened') AS issues_opened,
            COUNT(*) FILTER (WHERE type = 'PullRequestEvent' AND action = 'opened') AS prs_opened
        FROM events
        WHERE type IN ('PushEvent','PullRequestEvent','IssuesEvent','PullRequestReviewEvent')
        GROUP BY repo_id, DATE_TRUNC('month', created_at)
        """
    ).df()
    con.close()
    monthly["month"] = pd.to_datetime(monthly["month"], utc=True)

    panel = spine.merge(monthly, on=["repo_id", "month"], how="left")
    for col in [
        "activity_count",
        "contributor_count",
        "merged_pr_count",
        "issues_opened",
        "prs_opened",
    ]:
        panel[col] = panel[col].fillna(0).astype(int)

    # ---- Merge treatment timing ----------------------------------------
    panel = panel.merge(ci, on="repo_id", how="left")
    panel["first_workflow_date"] = pd.to_datetime(
        panel["first_workflow_date"], utc=True, errors="coerce"
    )
    panel["first_workflow_month"] = pd.to_datetime(
        panel["first_workflow_date"].dt.strftime("%Y-%m-01"), utc=True
    )

    panel["treated"] = panel["first_workflow_month"].notna() & (
        panel["month"] >= panel["first_workflow_month"]
    )

    rel = (
        (panel["month"].dt.year - panel["first_workflow_month"].dt.year) * 12
        + (panel["month"].dt.month - panel["first_workflow_month"].dt.month)
    ).astype("Float64")
    rel = rel.where(panel["first_workflow_month"].notna(), np.nan)
    panel["rel_month"] = rel

    out = paths.analytics_dir() / "repo_month_panel.parquet"
    panel.to_parquet(out, index=False)
    log.info(
        "analytics/repo_month_panel.parquet: %d rows, %d repos, %d treated repos",
        len(panel),
        panel["repo_id"].nunique(),
        (panel.groupby("repo_id")["treated"].any()).sum(),
    )
    return panel


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    import argparse

    from osspulse import logging as oss_log

    parser = argparse.ArgumentParser(description="Build analytics tables.")
    parser.add_argument("--config", default="configs/cohort.yaml")
    args = parser.parse_args()

    cfg = Config.load(args.config)
    oss_log.setup(cfg.logging.level)

    ci = build_ci_adoption()
    survival, funnel = build_repo_survival(cfg)
    build_repo_month_panel(cfg, survival, ci)

    funnel_out = paths.analytics_dir() / "cohort_funnel.json"
    funnel_out.write_text(json.dumps(funnel, indent=2))
    log.info("cohort funnel: %s", funnel)


if __name__ == "__main__":
    main()

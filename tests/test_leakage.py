"""Automated leakage test: no survival feature may use data from day 90 onwards.

This test reads the actual `events_typed.parquet` from a small fixture built in
the test, constructs survival features, and asserts that adding events after
day 90 does not change any feature value.
"""

from __future__ import annotations

import pandas as pd

from osspulse import analytics, paths


def test_features_do_not_change_with_post_90d_events(monkeypatch, tmp_path):
    """If we append events after day 90, the feature table must be identical."""
    monkeypatch.setenv("OSSPULSE_DATA_DIR", str(tmp_path))

    # Build a minimal metadata file: one repo, created 2019-01-01.
    staging = tmp_path / "staging"
    staging.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "id": 1,
                "created_at": "2019-01-01T00:00:00Z",
                "language": "Python",
                "license_spdx": "MIT",
                "description": "x",
                "topics": None,
                "owner_login": "alice",
                "deleted": False,
            }
        ]
    ).to_parquet(staging / "repo_metadata.parquet", index=False)

    def _row(event_id: str, ts: str, actor: str = "bob") -> dict:
        return {
            "id": event_id,
            "type": "PushEvent",
            "created_at": pd.Timestamp(ts, tz="UTC"),
            "actor_id": 1,
            "actor_login": actor,
            "repo_id": 1,
            "repo_name": "alice/repo",
            "payload_json": "{}",
            "action": None,
            "issue_number": None,
            "issue_created_at": None,
            "pr_merged": None,
            "pr_number": None,
            "pr_created_at": None,
        }

    inside = pd.DataFrame(
        [
            _row("1", "2019-01-10T00:00:00Z"),
            _row("2", "2019-03-01T00:00:00Z"),
        ]
    )
    inside.to_parquet(staging / "events_typed.parquet", index=False)

    cfg = _load_cfg()
    features_before, _ = analytics.build_repo_survival(cfg)
    paths.analytics_dir().joinpath("repo_survival.parquet").unlink(missing_ok=True)

    # Append events after day 90 (day 90 = 2019-04-01).
    with_outside = pd.concat(
        [inside, pd.DataFrame([_row("3", "2019-06-01T00:00:00Z", actor="carol")])],
        ignore_index=True,
    )
    with_outside.to_parquet(staging / "events_typed.parquet", index=False)

    features_after, _ = analytics.build_repo_survival(cfg)

    # Every first-90-day feature must be unchanged.
    for col in [
        "stars_90d",
        "contributors_90d",
        "issues_opened_90d",
        "prs_opened_90d",
        "pushes_90d",
        "total_activity_90d",
    ]:
        assert (features_before[col] == features_after[col]).all(), col


def _load_cfg():
    from osspulse.config import Config

    return Config.load("configs/cohort.yaml")

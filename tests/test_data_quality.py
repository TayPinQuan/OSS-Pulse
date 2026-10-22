"""Data-quality checks on staging and analytics tables.

These tests run against whatever is currently on disk under `data/`. They are
skipped automatically if the pipeline has not been run yet, so `make test`
works on a clean clone.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

STAGING = Path("data/staging")
ANALYTICS = Path("data/analytics")


@pytest.mark.skipif(not (STAGING / "events.parquet").exists(), reason="run make staging first")
def test_no_duplicate_event_ids() -> None:
    df = pd.read_parquet(STAGING / "events.parquet", columns=["id"])
    assert df["id"].is_unique


@pytest.mark.skipif(not (STAGING / "events.parquet").exists(), reason="run make staging first")
def test_no_bot_actors_remain() -> None:
    df = pd.read_parquet(STAGING / "events.parquet", columns=["actor_login"])
    logins = df["actor_login"].dropna().str.lower()
    assert not logins.str.contains(r"\[bot\]").any()
    assert not logins.isin(["dependabot", "renovate", "github-actions"]).any()


@pytest.mark.skipif(not (STAGING / "events.parquet").exists(), reason="run make staging first")
def test_fork_events_excluded() -> None:
    df = pd.read_parquet(STAGING / "events.parquet", columns=["type"])
    assert "ForkEvent" not in df["type"].unique()


@pytest.mark.skipif(
    not (ANALYTICS / "repo_survival.parquet").exists(), reason="run make analytics first"
)
def test_no_negative_durations() -> None:
    df = pd.read_parquet(ANALYTICS / "repo_survival.parquet", columns=["duration_days"])
    assert (df["duration_days"] >= 0).all()


@pytest.mark.skipif(
    not (ANALYTICS / "repo_survival.parquet").exists(), reason="run make analytics first"
)
def test_event_flag_and_censoring_partition() -> None:
    df = pd.read_parquet(ANALYTICS / "repo_survival.parquet", columns=["event_flag"])
    events = int(df["event_flag"].sum())
    censored = int((~df["event_flag"]).sum())
    assert events + censored == len(df)
    assert events > 0, "no events observed: check observation_end vs created_to"


@pytest.mark.skipif(
    not (ANALYTICS / "cohort_funnel.json").exists(), reason="run make analytics first"
)
def test_funnel_is_monotonic() -> None:
    with open(ANALYTICS / "cohort_funnel.json") as f:
        f_ = json.load(f)
    assert f_["metadata_repos"] >= f_["with_any_active_event"] >= f_["included_both_pre_sample"]
    assert f_["sampled_to"] == f_["included_both_pre_sample"]


@pytest.mark.skipif(
    not (ANALYTICS / "forecast_summary.parquet").exists(), reason="run make forecast first"
)
def test_forecast_summary_has_all_models() -> None:
    df = pd.read_parquet(ANALYTICS / "forecast_summary.parquet")
    assert set(df["model"].unique()) == {"naive", "seasonal_naive", "ets", "lightgbm"}
    assert set(df["horizon"].unique()) == {1, 3}

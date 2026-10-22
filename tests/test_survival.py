from __future__ import annotations

import numpy as np
import pandas as pd

from osspulse import survival


def _synthetic_survival(n: int = 200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "repo_id": range(n),
            "created_at": pd.Timestamp("2020-01-01", tz="UTC"),
            "last_active_at": pd.Timestamp("2020-06-01", tz="UTC"),
            "stars_90d": rng.integers(5, 100, n),
            "contributors_90d": rng.integers(1, 20, n),
            "pushes_90d": rng.integers(0, 200, n),
            "issues_opened_90d": rng.integers(0, 30, n),
            "prs_opened_90d": rng.integers(0, 20, n),
            "median_response_hours": rng.choice([np.nan, 1.0, 12.0, 48.0], n),
            "pr_merge_rate_90d": rng.choice([np.nan, 0.0, 0.5, 0.9], n),
            "license_spdx": rng.choice([None, "MIT", "Apache-2.0"], n),
            "language": "Python",
            "duration_days": rng.integers(30, 1000, n),
            "event_flag": rng.integers(0, 2, n).astype(bool),
        }
    )


def test_prepare_features_no_nans() -> None:
    df = _synthetic_survival()
    prep = survival.prepare_features(df)
    for f in survival.FEATURES:
        assert prep[f].notna().all(), f"{f} has NaNs"


def test_prepare_features_binary_flags() -> None:
    df = _synthetic_survival()
    prep = survival.prepare_features(df)
    assert set(prep["has_issue_response"].unique()) <= {0, 1}
    assert set(prep["has_closed_pr"].unique()) <= {0, 1}
    assert set(prep["license_present"].unique()) <= {0, 1}


def test_fit_cox_returns_ci_bounds() -> None:
    df = _synthetic_survival(seed=1)
    prep = survival.prepare_features(df)
    cox, _ = survival.fit_cox(prep)
    assert (cox["hr_lower"] <= cox["hazard_ratio"]).all()
    assert (cox["hazard_ratio"] <= cox["hr_upper"]).all()


def test_fit_km_returns_curves() -> None:
    df = _synthetic_survival(seed=2)
    curves, logrank = survival.fit_km_segments(df)
    assert set(curves["segment"]) == {"license_present", "has_issue_response", "stars_bucket"}
    assert "license_present" in logrank


def test_sensitivity_produces_three_definitions(cfg) -> None:
    df = _synthetic_survival(seed=3)
    sens = survival.sensitivity_table(df, cfg)
    assert set(sens["definition_days"]) == {90, 180, 365}

from __future__ import annotations

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter

from osspulse import survival_ph
from osspulse.survival import FEATURES, prepare_features


def _synth(n: int = 400, seed: int = 0) -> pd.DataFrame:
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
            "duration_days": rng.integers(180, 700, n),
            "event_flag": rng.integers(0, 2, n).astype(bool),
        }
    )


def _fit(df: pd.DataFrame) -> tuple[pd.DataFrame, CoxPHFitter]:
    prep = prepare_features(df)
    cph = CoxPHFitter(penalizer=0.01)
    cph.fit(
        prep[["duration_days", "event_flag", *FEATURES]],
        duration_col="duration_days",
        event_col="event_flag",
    )
    return prep, cph


def test_schoenfeld_test_returns_all_features() -> None:
    prep, cph = _fit(_synth())
    test = survival_ph.schoenfeld_test(prep, cph)
    assert set(test["feature"]) == set(FEATURES)
    assert test["p_value"].between(0, 1).all()


def test_schoenfeld_residuals_long_form() -> None:
    prep, cph = _fit(_synth(n=200))
    resid = survival_ph.schoenfeld_residuals(prep, cph)
    assert set(resid.columns) == {"duration_days", "feature", "residual"}
    assert resid["feature"].nunique() == len(FEATURES)


def test_time_varying_hr_positive() -> None:
    prep, _ = _fit(_synth())
    features_to_vary = ["log_contributors_90d"]
    _, cph_tv = survival_ph.fit_time_varying(prep, features_to_vary, time_transform="log")
    hr_df = survival_ph.hr_at_time(
        cph_tv, features_to_vary, [180.0, 240.0, 300.0], time_transform="log"
    )
    assert (hr_df["hazard_ratio"] > 0).all()
    assert (hr_df["hr_lower"] <= hr_df["hazard_ratio"]).all()
    assert (hr_df["hazard_ratio"] <= hr_df["hr_upper"]).all()
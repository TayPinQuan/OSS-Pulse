from __future__ import annotations

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter

from osspulse import survival_advanced
from osspulse.survival import FEATURES, prepare_features


def _synth(n: int = 300, seed: int = 0) -> pd.DataFrame:
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
            "duration_days": rng.integers(200, 1000, n),
            "event_flag": rng.integers(0, 2, n).astype(bool),
        }
    )


def test_calibration_returns_bins() -> None:
    df = _synth()
    prep = prepare_features(df)
    cph = CoxPHFitter(penalizer=0.01)
    cph.fit(
        prep[["duration_days", "event_flag", *FEATURES]],
        duration_col="duration_days",
        event_col="event_flag",
    )
    calib = survival_advanced.calibration_table(prep, cph, horizons=[300.0], n_bins=5)
    assert not calib.empty
    assert {"horizon", "bin", "n", "pred_mean", "obs_mean"} <= set(calib.columns)
    assert calib["n"].sum() == len(prep)


def test_decision_curve_monotone_thresholds() -> None:
    df = _synth()
    prep = prepare_features(df)
    cph = CoxPHFitter(penalizer=0.01)
    cph.fit(
        prep[["duration_days", "event_flag", *FEATURES]],
        duration_col="duration_days",
        event_col="event_flag",
    )
    dca = survival_advanced.decision_curve(
        prep, cph, horizon=360.0, thresholds=np.array([0.1, 0.3, 0.5])
    )
    assert len(dca) == 3
    assert (dca["nb_treat_none"] == 0).all()


def test_subgroup_cox_skips_small() -> None:
    df = _synth(n=300, seed=2)
    df["tier"] = np.where(df["stars_90d"] >= 20, "high", "low")
    prep = prepare_features(df)
    sub = survival_advanced.subgroup_cox(prep, "tier", ["high", "low", "tiny_empty"])
    # tiny_empty has 0 rows -> should be skipped
    assert set(sub["subgroup_value"].unique()) <= {"high", "low"}

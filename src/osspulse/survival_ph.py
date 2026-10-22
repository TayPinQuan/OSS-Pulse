"""Proportional hazards diagnostics: Schoenfeld residuals and time-varying coefficients.

Why this matters:
  The Cox model assumes each covariate's effect is constant over time. If
  that assumption is violated, the reported hazard ratio is an average of
  time-varying effects and its interpretation is misleading. This module
  tests the assumption per feature and, where it is violated, estimates
  HR(t) with a time-varying coefficient model.

Inputs:
  analytics/repo_survival.parquet

Outputs:
  analytics/survival_ph_test.parquet         — per-feature Schoenfeld test
  analytics/survival_schoenfeld_residuals.parquet
  analytics/survival_time_varying.parquet    — HR(t) for violated features
  results/figures/survival_schoenfeld_<feature>.png
  results/figures/survival_time_varying_<feature>.png
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.statistics import proportional_hazard_test

from osspulse import paths
from osspulse.config import Config
from osspulse.logging import get
from osspulse.survival import FEATURES, prepare_features

log = get(__name__)


# ---------------------------------------------------------------------------
# Schoenfeld residual test
# ---------------------------------------------------------------------------


def schoenfeld_test(prep: pd.DataFrame, cph: CoxPHFitter) -> pd.DataFrame:
    """Per-feature proportional-hazards test on scaled Schoenfeld residuals.

    `time_transform="rank"` is the standard choice for moderate sample sizes.
    A small p-value means the feature's effect changes over time (PH violated).
    """
    df = prep[["duration_days", "event_flag", *FEATURES]]
    results = proportional_hazard_test(
        cph, df, time_transform="rank"
    )
    out = results.summary.copy()
    out.index.name = "feature"
    out = out.reset_index()
    out = out.rename(columns={"test_statistic": "chi2", "p": "p_value"})
    out["ph_violated"] = out["p_value"] < 0.05
    return out[["feature", "chi2", "p_value", "ph_violated"]]


def schoenfeld_residuals(prep: pd.DataFrame, cph: CoxPHFitter) -> pd.DataFrame:
    """Scaled Schoenfeld residuals, long-form: one row per (event, feature).

    Schoenfeld residuals are only defined at event times, so the returned
    table has one row per observed event, not per repo. If PH holds, the
    residuals should be uncorrelated with event time.
    """
    df = prep[["duration_days", "event_flag", *FEATURES]].copy()
    resid = cph.compute_residuals(df, kind="scaled_schoenfeld")

    # compute_residuals preserves the original index for event rows only.
    # Align duration_days by that index, not by position.
    resid = resid.copy()
    resid["duration_days"] = df.loc[resid.index, "duration_days"].to_numpy()
    resid = resid.reset_index(drop=True)

    long = resid.melt(
        id_vars=["duration_days"],
        var_name="feature",
        value_name="residual",
    )
    return long


def plot_schoenfeld_residuals(
    resid: pd.DataFrame, test: pd.DataFrame, path_dir
) -> None:
    """One panel per feature: residual vs time with LOWESS smoothing."""
    from statsmodels.nonparametric.smoothers_lowess import lowess

    for feature in test["feature"]:
        sub = resid[resid["feature"] == feature].dropna(subset=["residual", "duration_days"])
        if len(sub) < 20:
            continue
        p = float(test.set_index("feature").loc[feature, "p_value"])

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.scatter(sub["duration_days"], sub["residual"], s=6, alpha=0.35, color="steelblue")
        smooth = lowess(sub["residual"], sub["duration_days"], frac=0.3, return_sorted=True)
        ax.plot(smooth[:, 0], smooth[:, 1], color="tab:red", linewidth=2, label="LOWESS")
        ax.axhline(0, color="grey", linestyle="--", alpha=0.5)
        ax.set_xlabel("Event time (days)")
        ax.set_ylabel("Scaled Schoenfeld residual")
        verdict = "PH violated" if p < 0.05 else "PH holds"
        ax.set_title(f"{feature} — {verdict} (p = {p:.3g})")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(path_dir / f"survival_schoenfeld_{feature}.png", dpi=120)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Time-varying coefficient model
# ---------------------------------------------------------------------------


def fit_time_varying(
    prep: pd.DataFrame,
    features_to_vary: list[str],
    *,
    time_transform: str = "log",
) -> tuple[pd.DataFrame, CoxPHFitter]:
    """Fit Cox with X × f(t) interaction terms for the given features.

    For a feature with PH violated, its coefficient is allowed to change with
    time. The HR at time t is exp(beta_x + beta_xt * f(t)), which we compute
    in `hr_at_time`.
    """
    df = prep[["duration_days", "event_flag", *FEATURES]].copy()
    t = df["duration_days"].clip(lower=1).astype(float)
    if time_transform == "log":
        tf = np.log(t)
    elif time_transform == "linear":
        tf = t / t.max()  # normalise to [0, 1]
    else:
        raise ValueError(f"unknown time_transform: {time_transform}")

    for f in features_to_vary:
        df[f"{f}_x_t"] = df[f] * tf

    cph = CoxPHFitter(penalizer=0.01)
    cph.fit(df, duration_col="duration_days", event_col="event_flag")
    return df, cph


def hr_at_time(
    cph_tv: CoxPHFitter,
    features_to_vary: list[str],
    times: list[float],
    *,
    time_transform: str = "log",
) -> pd.DataFrame:
    """Compute HR(t) with delta-method CIs from the time-varying Cox model."""
    params = cph_tv.params_
    cov = cph_tv.variance_matrix_
    rows: list[dict] = []

    for feature in features_to_vary:
        b1 = float(params[feature])
        b2_name = f"{feature}_x_t"
        if b2_name not in params:
            continue
        b2 = float(params[b2_name])

        for t in times:
            ft = float(np.log(max(t, 1.0))) if time_transform =="log" else t / max(times)

            log_hr = b1 + b2 * ft
            hr = float(np.exp(log_hr))

            # Delta method: Var(log HR) = g' V g, where g = [1, f(t)]
            v11 = float(cov.loc[feature, feature])
            v22 = float(cov.loc[b2_name, b2_name])
            v12 = float(cov.loc[feature, b2_name])
            var_log_hr = v11 + 2 * ft * v12 + (ft**2) * v22
            se = float(np.sqrt(max(var_log_hr, 0.0)))

            lower = float(np.exp(log_hr - 1.96 * se))
            upper = float(np.exp(log_hr + 1.96 * se))

            rows.append(
                {
                    "feature": feature,
                    "time": float(t),
                    "hazard_ratio": hr,
                    "hr_lower": lower,
                    "hr_upper": upper,
                }
            )
    return pd.DataFrame(rows)


def plot_time_varying_hr(
    hr_df: pd.DataFrame, path_dir
) -> None:
    for feature, sub in hr_df.groupby("feature"):
        sub = sub.sort_values("time")
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(sub["time"], sub["hazard_ratio"], color="steelblue", linewidth=2)
        ax.fill_between(
            sub["time"], sub["hr_lower"], sub["hr_upper"],
            color="steelblue", alpha=0.2, label="95% CI",
        )
        ax.axhline(1.0, color="grey", linestyle="--", alpha=0.6)
        ax.set_xlabel("Time since creation (days)")
        ax.set_ylabel("Hazard ratio")
        ax.set_title(f"{feature} — HR(t)")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(path_dir / f"survival_time_varying_{feature}.png", dpi=120)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    import argparse

    from osspulse import logging as oss_log

    parser = argparse.ArgumentParser(
        description="Proportional hazards diagnostics and time-varying Cox."
    )
    parser.add_argument("--config", default="configs/cohort.yaml")
    args = parser.parse_args()

    cfg = Config.load(args.config)
    oss_log.setup(cfg.logging.level)

    src = paths.analytics_dir() / "repo_survival.parquet"
    df = pd.read_parquet(src)
    if len(df) < 200:
        raise RuntimeError(f"Only {len(df)} repos; too few for PH diagnostics.")

    prep = prepare_features(df)
    cph = CoxPHFitter(penalizer=0.01)
    cph.fit(
        prep[["duration_days", "event_flag", *FEATURES]],
        duration_col="duration_days",
        event_col="event_flag",
    )

    # ---- Global PH test ------------------------------------------------
    # lifelines does not expose a clean global test; report a Bonferroni
    # pass if any feature fails.
    test = schoenfeld_test(prep, cph)
    test.to_parquet(paths.analytics_dir() / "survival_ph_test.parquet", index=False)
    log.info("Schoenfeld PH test:\n%s", test.round(4).to_string(index=False))

    violated = test.loc[test["ph_violated"], "feature"].tolist()
    log.info("features violating PH (p<0.05): %s", violated)

    # ---- Schoenfeld residuals ------------------------------------------
    resid = schoenfeld_residuals(prep, cph)
    resid.to_parquet(
        paths.analytics_dir() / "survival_schoenfeld_residuals.parquet", index=False
    )
    plot_schoenfeld_residuals(resid, test, paths.figures_dir())
    log.info("Schoenfeld residual plots written")

    # ---- Time-varying Cox for violated features ------------------------
    tv_summary: dict = {"violated_features": violated}
    if violated:
        _, cph_tv = fit_time_varying(prep, violated, time_transform="log")
        times = [180.0, 240.0, 300.0, 360.0]
        hr_df = hr_at_time(cph_tv, violated, times, time_transform="log")
        hr_df.to_parquet(
            paths.analytics_dir() / "survival_time_varying.parquet", index=False
        )
        plot_time_varying_hr(hr_df, paths.figures_dir())
        log.info("HR(t):\n%s", hr_df.round(3).to_string(index=False))
        tv_summary["hr_at_times"] = hr_df.round(4).to_dict(orient="records")
    else:
        log.info("No feature violates PH; time-varying model skipped")

    (paths.analytics_dir() / "survival_ph_summary.json").write_text(
        json.dumps(tv_summary, indent=2, default=str)
    )


if __name__ == "__main__":
    main()
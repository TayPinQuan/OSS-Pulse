"""RQ1: Survival analysis.

Inputs:
  analytics/repo_survival.parquet

Outputs:
  analytics/survival_km.parquet              — KM curves by segment
  analytics/survival_cox.parquet             — Cox HR table with 95% CI
  analytics/survival_sensitivity.parquet     — Cox HR across 90/180/365-day definitions
  analytics/survival_model_comparison.json   — C-index Cox vs RSF
  results/figures/survival_km.png
  results/figures/survival_forest.png
  results/figures/survival_sensitivity.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import concordance_index_censored
from sksurv.util import Surv

from osspulse import paths
from osspulse.config import Config
from osspulse.logging import get

log = get(__name__)

FEATURES = [
    "log_contributors_90d",
    "has_issue_response",
    "median_response_capped",
    "has_closed_pr",
    "pr_merge_rate_90d_filled",
    "log_stars_90d",
    "log_pushes_90d",
    "license_present",
]


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Turn raw survival table into model-ready numeric features."""
    out = df.copy()

    out["log_contributors_90d"] = np.log1p(out["contributors_90d"].fillna(0))
    out["log_stars_90d"] = np.log1p(out["stars_90d"].fillna(0))
    out["log_pushes_90d"] = np.log1p(out["pushes_90d"].fillna(0))

    out["has_issue_response"] = out["median_response_hours"].notna().astype(int)
    out["median_response_capped"] = out["median_response_hours"].fillna(10_000).clip(upper=10_000)

    out["has_closed_pr"] = out["pr_merge_rate_90d"].notna().astype(int)
    out["pr_merge_rate_90d_filled"] = out["pr_merge_rate_90d"].fillna(0.5).clip(0, 1)

    out["license_present"] = out["license_spdx"].notna().astype(int)

    return out


def _survival_arrays(df: pd.DataFrame):
    y = Surv.from_arrays(
        event=df["event_flag"].astype(bool).values,
        time=df["duration_days"].astype(float).values,
    )
    return y


# ---------------------------------------------------------------------------
# KM curves
# ---------------------------------------------------------------------------


def fit_km_segments(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Fit KM per segment and return long-form curve data + log-rank results."""
    segments = {
        "license_present": ("No license", "Has license"),
        "has_issue_response": ("No issue reply", "Issue reply"),
        "stars_bucket": ("stars 5-19", "stars 20+"),
    }

    prep = prepare_features(df)
    prep["stars_bucket"] = np.where(prep["stars_90d"] >= 20, "stars 20+", "stars 5-19")

    curves = []
    logrank: dict = {}
    for seg, (label_a, label_b) in segments.items():
        for value, label in [(0, label_a), (1, label_b)]:
            mask = prep[seg] == (value if seg != "stars_bucket" else label)
            if mask.sum() == 0:
                continue
            kmf = KaplanMeierFitter()
            kmf.fit(prep.loc[mask, "duration_days"], prep.loc[mask, "event_flag"], label=label)
            sf = kmf.survival_function_.reset_index()
            sf.columns = ["timeline", "survival"]
            sf["segment"] = seg
            sf["group"] = label
            curves.append(sf)

        # log-rank
        if seg == "stars_bucket":
            a = prep[prep[seg] == label_a]
            b = prep[prep[seg] == label_b]
        else:
            a = prep[prep[seg] == 0]
            b = prep[prep[seg] == 1]
        if len(a) > 0 and len(b) > 0:
            res = logrank_test(
                a["duration_days"],
                b["duration_days"],
                a["event_flag"],
                b["event_flag"],
            )
            logrank[seg] = {
                "p_value": float(res.p_value),
                "test_statistic": float(res.test_statistic),
            }

    curve_df = pd.concat(curves, ignore_index=True) if curves else pd.DataFrame()
    return curve_df, logrank


def plot_km(curve_df: pd.DataFrame, path: Path) -> None:
    segments = curve_df["segment"].unique()
    fig, axes = plt.subplots(1, len(segments), figsize=(5 * len(segments), 4), sharey=True)
    if len(segments) == 1:
        axes = [axes]
    for ax, seg in zip(axes, segments, strict=False):
        sub = curve_df[curve_df["segment"] == seg]
        for group, g in sub.groupby("group"):
            ax.step(g["timeline"], g["survival"], where="post", label=group)
        ax.set_title(seg)
        ax.set_xlabel("Days since creation")
        ax.set_ylabel("S(t)")
        ax.legend(loc="best", fontsize=8)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Cox
# ---------------------------------------------------------------------------


def fit_cox(prep: pd.DataFrame) -> tuple[pd.DataFrame, CoxPHFitter]:
    fit_df = prep[["duration_days", "event_flag", *FEATURES]].copy()
    cph = CoxPHFitter(penalizer=0.01)
    cph.fit(fit_df, duration_col="duration_days", event_col="event_flag")

    summary = cph.summary.copy()
    summary.index.name = "feature"
    summary = summary.reset_index()
    summary = summary.rename(
        columns={
            "exp(coef)": "hazard_ratio",
            "exp(coef) lower 95%": "hr_lower",
            "exp(coef) upper 95%": "hr_upper",
            "p": "p_value",
        }
    )
    keep = ["feature", "hazard_ratio", "hr_lower", "hr_upper", "p_value"]
    return summary[keep], cph


def plot_forest(cox_df: pd.DataFrame, path: Path) -> None:
    df = cox_df.sort_values("hazard_ratio")
    fig, ax = plt.subplots(figsize=(6, 0.4 * len(df) + 1.5))
    y = np.arange(len(df))
    hr = df["hazard_ratio"].values
    lo = df["hr_lower"].values
    hi = df["hr_upper"].values
    ax.errorbar(hr, y, xerr=[hr - lo, hi - hr], fmt="o", color="steelblue", capsize=3)
    ax.axvline(1.0, color="grey", linestyle="--", alpha=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(df["feature"].values, fontsize=9)
    ax.set_xscale("log")
    ax.set_xlabel("Hazard ratio (log scale, 95% CI)")
    ax.set_title("Cox proportional hazards")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# RSF
# ---------------------------------------------------------------------------


def fit_rsf(prep: pd.DataFrame, seed: int) -> tuple[float, np.ndarray]:
    """Fit RSF on 80/20 split; return (C-index, permutation importances)."""
    X = prep[FEATURES].values
    y = _survival_arrays(prep)

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(prep))
    cut = int(0.8 * len(prep))
    train, test = idx[:cut], idx[cut:]

    rsf = RandomSurvivalForest(
        n_estimators=300,
        min_samples_leaf=10,
        n_jobs=-1,
        random_state=seed,
    )
    rsf.fit(X[train], y[train])

    baseline = concordance_index_censored(y[test]["event"], y[test]["time"], rsf.predict(X[test]))[
        0
    ]

    # Permutation importance: how much does C-index drop when feature j is shuffled?
    importances = np.zeros(X.shape[1])
    X_test = X[test].copy()
    for j in range(X.shape[1]):
        X_perm = X_test.copy()
        rng.shuffle(X_perm[:, j])
        c_perm = concordance_index_censored(y[test]["event"], y[test]["time"], rsf.predict(X_perm))[
            0
        ]
        importances[j] = baseline - c_perm

    return float(baseline), importances


# ---------------------------------------------------------------------------
# Sensitivity: 90 / 180 / 365 days
# ---------------------------------------------------------------------------


def sensitivity_table(df_raw: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Recompute Cox HRs under 90/180/365-day inactivity definitions.

    Skips a definition if fewer than 20 events are observed, to avoid
    fitting Cox on a degenerate sample.
    """
    obs_end = pd.Timestamp(cfg.cohort.observation_end, tz="UTC")
    created = pd.to_datetime(df_raw["created_at"], utc=True)
    last_active = pd.to_datetime(df_raw["last_active_at"], utc=True, errors="coerce")

    rows: list[dict] = []
    for days in (90, 180, 365):
        inactivity = pd.Timedelta(days=days)
        has_active = last_active.notna()
        event_time = (last_active + inactivity).where(has_active, created + inactivity)
        event_flag = event_time <= obs_end
        duration = (event_time.where(event_flag, obs_end) - created).dt.days.clip(lower=0)

        n_events = int(event_flag.sum())
        if n_events < 20:
            log.warning(
                "skipping %d-day sensitivity: only %d events observed "
                "(need >= 20 for stable Cox). This is expected when "
                "observation_end is close to created_to.",
                days,
                n_events,
            )
            continue

        tmp = df_raw.copy()
        tmp["duration_days"] = duration
        tmp["event_flag"] = event_flag
        prep = prepare_features(tmp)

        try:
            cox, _ = fit_cox(prep)
            cox["definition_days"] = days
            cox["n_repos"] = len(prep)
            cox["n_events"] = n_events
            cox["event_rate"] = float(event_flag.mean())
            rows.append(cox)
            log.info(
                "%d-day definition: n=%d events=%d rate=%.3f",
                days,
                len(prep),
                n_events,
                float(event_flag.mean()),
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("Cox failed for %d-day definition: %s", days, exc)

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def plot_sensitivity(sens: pd.DataFrame, path: Path) -> None:
    if sens.empty:
        return
    pivot = sens.pivot_table(index="feature", columns="definition_days", values="hazard_ratio")
    fig, ax = plt.subplots(figsize=(7, 0.4 * len(pivot) + 1.5))
    pivot.plot(kind="barh", ax=ax, logx=True)
    ax.axvline(1.0, color="grey", linestyle="--", alpha=0.6)
    ax.set_xlabel("Hazard ratio (log scale)")
    ax.set_title("Sensitivity to inactivity definition (days)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    import argparse

    from osspulse import logging as oss_log

    parser = argparse.ArgumentParser(description="RQ1: survival analysis.")
    parser.add_argument("--config", default="configs/cohort.yaml")
    args = parser.parse_args()

    cfg = Config.load(args.config)
    oss_log.setup(cfg.logging.level)

    src = paths.analytics_dir() / "repo_survival.parquet"
    df = pd.read_parquet(src)
    log.info("loaded %d repos from %s", len(df), src)

    if len(df) < 50:
        raise RuntimeError(
            f"Only {len(df)} repos; too few for survival analysis. Check the cohort funnel."
        )

    prep = prepare_features(df)

    # KM
    curve_df, logrank = fit_km_segments(df)
    curve_df.to_parquet(paths.analytics_dir() / "survival_km.parquet", index=False)
    plot_km(curve_df, paths.figures_dir() / "survival_km.png")
    log.info("KM log-rank: %s", logrank)

    # Cox
    cox_df, cph = fit_cox(prep)
    cox_df.to_parquet(paths.analytics_dir() / "survival_cox.parquet", index=False)
    plot_forest(cox_df, paths.figures_dir() / "survival_forest.png")
    log.info("Cox:\n%s", cox_df.to_string(index=False))

    # RSF
    c_rsf, importances = fit_rsf(prep, cfg.cohort.sample_seed)
    c_cox = float(cph.concordance_index_)
    comparison = {
        "n_repos": int(len(prep)),
        "event_rate": float(prep["event_flag"].mean()),
        "c_index_cox": c_cox,
        "c_index_rsf": c_rsf,
        "rsf_importances": dict(zip(FEATURES, importances.tolist(), strict=False)),
        "km_logrank": logrank,
    }
    (paths.analytics_dir() / "survival_model_comparison.json").write_text(
        json.dumps(comparison, indent=2)
    )
    log.info(
        "C-index: Cox=%.3f  RSF=%.3f (RSF on 20%% holdout)",
        c_cox,
        c_rsf,
    )

    # Sensitivity
    sens = sensitivity_table(df, cfg)
    if not sens.empty:
        sens.to_parquet(paths.analytics_dir() / "survival_sensitivity.parquet", index=False)
        plot_sensitivity(sens, paths.figures_dir() / "survival_sensitivity.png")
        log.info(
            "sensitivity table:\n%s",
            sens.pivot(index="feature", columns="definition_days", values="hazard_ratio")
            .round(3)
            .to_string(),
        )


if __name__ == "__main__":
    main()

"""RQ1 v2: calibration, decision curve analysis, subgroup effects.

All three run against the existing `repo_survival.parquet`. No new data.

Outputs:
  analytics/survival_calibration.parquet
  analytics/survival_dca.parquet
  analytics/survival_subgroups.parquet
  results/figures/survival_calibration.png
  results/figures/survival_dca.png
  results/figures/survival_subgroups.png
"""

from __future__ import annotations

import json
from collections.abc import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter

from osspulse import paths
from osspulse.config import Config
from osspulse.logging import get
from osspulse.survival import FEATURES, fit_cox, prepare_features

log = get(__name__)


# ---------------------------------------------------------------------------
# Calibration: predicted vs observed survival at fixed horizons
# ---------------------------------------------------------------------------


def calibration_table(
    prep: pd.DataFrame,
    cph: CoxPHFitter,
    horizons: Sequence[float],
    n_bins: int = 10,
) -> pd.DataFrame:
    """Bin repos by predicted survival and compare to observed survival.

    For each horizon t: predicted S_i(t) from the Cox model; observed
    survival = fraction of repos in the bin whose event time > t (or censored
    after t). Reports both per-bin and the overall Integrated Calibration
    Index (ICI).
    """
    rows: list[dict] = []
    for t in horizons:
        surv_df = cph.predict_survival_function(prep[FEATURES], times=[t])
        pred = surv_df.iloc[0].to_numpy()

        # Observed survival: 1 if still alive at t
        observed = ((prep["duration_days"] > t) | (~prep["event_flag"])).astype(float).to_numpy()

        # Bin by predicted survival (quantile bins, fall back to equal-width if ties)
        try:
            bins = pd.qcut(pred, n_bins, labels=False, duplicates="drop")
        except ValueError:
            bins = pd.cut(pred, n_bins, labels=False)
        df = pd.DataFrame({"pred": pred, "obs": observed, "bin": bins}).dropna(subset=["bin"])
        for b, g in df.groupby("bin"):
            rows.append(
                {
                    "horizon": t,
                    "bin": int(b),
                    "n": int(len(g)),
                    "pred_mean": float(g["pred"].mean()),
                    "obs_mean": float(g["obs"].mean()),
                }
            )

    out = pd.DataFrame(rows)
    if not out.empty:
        ici = float(np.mean(np.abs(out["pred_mean"] - out["obs_mean"])))
        out.attrs["ici"] = ici
    return out


def plot_calibration(calib: pd.DataFrame, path) -> None:
    if calib.empty:
        return
    horizons = sorted(calib["horizon"].unique())
    fig, ax = plt.subplots(figsize=(6, 6))
    for t in horizons:
        sub = calib[calib["horizon"] == t]
        ax.plot(
            sub["pred_mean"],
            sub["obs_mean"],
            marker="o",
            label=f"t = {int(t)} days",
        )
    ax.plot([0, 1], [0, 1], color="grey", linestyle="--", label="perfect calibration")
    ax.set_xlabel("Predicted survival")
    ax.set_ylabel("Observed survival")
    ax.set_title("Calibration (Cox model)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Decision Curve Analysis
# ---------------------------------------------------------------------------


def decision_curve(
    prep: pd.DataFrame,
    cph: CoxPHFitter,
    horizon: float,
    thresholds: np.ndarray | None = None,
) -> pd.DataFrame:
    """Net benefit of using the model vs treat-all vs treat-none at a horizon.

    A "treat" decision means: act on the repo now (e.g. assign a maintainer).
    You want to treat repos that will die by `horizon`.
    """
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 99)

    surv = cph.predict_survival_function(prep[FEATURES], times=[horizon]).iloc[0].to_numpy()
    risk = 1.0 - surv

    # True "will die by t": duration <= t AND event_flag
    died = ((prep["duration_days"].to_numpy() <= horizon) & prep["event_flag"].to_numpy()).astype(
        int
    )
    n = len(died)
    prevalence = died.mean()

    rows: list[dict] = []
    for p_t in thresholds:
        # Model: treat if predicted risk >= p_t
        model_treat = risk >= p_t
        tp = int(np.sum(model_treat & (died == 1)))
        fp = int(np.sum(model_treat & (died == 0)))
        nb_model = tp / n - fp / n * (p_t / (1 - p_t)) if p_t < 1 else 0.0

        # Treat all
        tp_all = int(died.sum())
        fp_all = int(n - died.sum())
        nb_all = tp_all / n - fp_all / n * (p_t / (1 - p_t)) if p_t < 1 else 0.0

        # Treat none
        nb_none = 0.0

        rows.append(
            {
                "horizon": horizon,
                "threshold": float(p_t),
                "nb_model": nb_model,
                "nb_treat_all": nb_all,
                "nb_treat_none": nb_none,
                "prevalence": float(prevalence),
            }
        )
    return pd.DataFrame(rows)


def plot_dca(dca: pd.DataFrame, path) -> None:
    if dca.empty:
        return
    horizons = sorted(dca["horizon"].unique())
    fig, axes = plt.subplots(1, len(horizons), figsize=(6 * len(horizons), 4))
    if len(horizons) == 1:
        axes = [axes]
    for ax, t in zip(axes, horizons, strict=False):
        sub = dca[dca["horizon"] == t]
        ax.plot(sub["threshold"], sub["nb_model"], label="model", color="tab:blue")
        ax.plot(
            sub["threshold"],
            sub["nb_treat_all"],
            label="treat all",
            color="tab:orange",
            linestyle="--",
        )
        ax.axhline(0, label="treat none", color="grey", linestyle=":")
        ax.set_xlabel("Decision threshold (predicted risk)")
        ax.set_ylabel("Net benefit")
        ax.set_title(f"Decision curve, horizon = {int(t)} days")
        ax.legend()
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Subgroup analysis
# ---------------------------------------------------------------------------


def subgroup_cox(
    prep: pd.DataFrame,
    subgroup_col: str,
    subgroup_values: list[str],
    *,
    ci_level: float = 0.95,
) -> pd.DataFrame:
    """Fit Cox separately in each subgroup and report HRs."""
    rows: list[dict] = []
    for val in subgroup_values:
        sub = prep[prep[subgroup_col] == val]
        if len(sub) < 100 or sub["event_flag"].sum() < 20:
            log.warning(
                "skip subgroup %s=%s: n=%d events=%d",
                subgroup_col,
                val,
                len(sub),
                int(sub["event_flag"].sum()),
            )
            continue
        try:
            cox, _ = fit_cox(sub)
            cox["subgroup_col"] = subgroup_col
            cox["subgroup_value"] = val
            cox["n_repos"] = int(len(sub))
            cox["n_events"] = int(sub["event_flag"].sum())
            rows.append(cox)
        except Exception as exc:  # noqa: BLE001
            log.warning("Cox failed for %s=%s: %s", subgroup_col, val, exc)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def plot_subgroups(sub: pd.DataFrame, feature: str, path) -> None:
    s = sub[sub["feature"] == feature].copy()
    if s.empty:
        return
    s = s.sort_values("subgroup_value")
    fig, ax = plt.subplots(figsize=(7, 0.5 * len(s) + 1.5))
    y = np.arange(len(s))
    hr = s["hazard_ratio"].to_numpy()
    lo = s["hr_lower"].to_numpy()
    hi = s["hr_upper"].to_numpy()
    ax.errorbar(hr, y, xerr=[hr - lo, hi - hr], fmt="o", color="steelblue", capsize=3)
    ax.axvline(1.0, color="grey", linestyle="--", alpha=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(s["subgroup_value"].to_numpy(), fontsize=9)
    ax.set_xscale("log")
    ax.set_xlabel("Hazard ratio (log scale, 95% CI)")
    ax.set_title(f"{feature} — by {s['subgroup_col'].iloc[0]}")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    import argparse

    from osspulse import logging as oss_log

    parser = argparse.ArgumentParser(description="RQ1 v2: calibration, DCA, subgroups.")
    parser.add_argument("--config", default="configs/cohort.yaml")
    args = parser.parse_args()

    cfg = Config.load(args.config)
    oss_log.setup(cfg.logging.level)

    df = pd.read_parquet(paths.analytics_dir() / "repo_survival.parquet")
    if len(df) < 100:
        raise RuntimeError(f"Only {len(df)} repos; too few for advanced survival analysis.")

    prep = prepare_features(df)
    cph = CoxPHFitter(penalizer=0.01)
    cph.fit(
        prep[["duration_days", "event_flag", *FEATURES]],
        duration_col="duration_days",
        event_col="event_flag",
    )

    # ---- Calibration -----------------------------------------------------
    calib_horizons = [180, 270, 360]
    calib = calibration_table(prep, cph, horizons=calib_horizons, n_bins=10)
    calib.to_parquet(paths.analytics_dir() / "survival_calibration.parquet", index=False)
    plot_calibration(calib, paths.figures_dir() / "survival_calibration.png")
    ici = calib.attrs.get("ici", float("nan"))
    log.info("calibration: ICI = %.3f", ici)
    log.info("calibration table:\n%s", calib.to_string(index=False))

    # ---- Decision curve --------------------------------------------------
    dca_frames = [decision_curve(prep, cph, horizon=t) for t in (180, 360)]
    dca = pd.concat(dca_frames, ignore_index=True)
    dca.to_parquet(paths.analytics_dir() / "survival_dca.parquet", index=False)
    plot_dca(dca, paths.figures_dir() / "survival_dca.png")
    log.info("DCA written for horizons [180, 360]")

    # ---- Subgroups -------------------------------------------------------
    prep["stars_tier"] = np.where(prep["stars_90d"] >= 20, "stars_20plus", "stars_5_19")
    prep["contrib_tier"] = np.where(prep["contributors_90d"] >= 3, "contrib_3plus", "contrib_1_2")
    subgroups = []
    for col, values in [
        ("stars_tier", ["stars_5_19", "stars_20plus"]),
        ("contrib_tier", ["contrib_1_2", "contrib_3plus"]),
    ]:
        sub = subgroup_cox(prep, col, values)
        if not sub.empty:
            subgroups.append(sub)
    sub_df = pd.concat(subgroups, ignore_index=True) if subgroups else pd.DataFrame()
    if not sub_df.empty:
        sub_df.to_parquet(paths.analytics_dir() / "survival_subgroups.parquet", index=False)
        for feature in ("log_contributors_90d", "log_pushes_90d", "license_present"):
            plot_subgroups(
                sub_df, feature, paths.figures_dir() / f"survival_subgroups_{feature}.png"
            )
        log.info(
            "subgroups:\n%s",
            sub_df[
                [
                    "subgroup_col",
                    "subgroup_value",
                    "feature",
                    "hazard_ratio",
                    "hr_lower",
                    "hr_upper",
                ]
            ]
            .round(3)
            .to_string(index=False),
        )

    # ---- Summary ---------------------------------------------------------
    summary = {
        "calibration_ici": ici,
        "calibration_horizons": calib_horizons,
        "dca_horizons": [180, 360],
        "subgroups_n_models": int(len(sub_df)),
    }
    (paths.analytics_dir() / "survival_advanced_summary.json").write_text(
        json.dumps(summary, indent=2)
    )


if __name__ == "__main__":
    main()

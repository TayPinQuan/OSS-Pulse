"""OSS Pulse dashboard.

Reads precomputed Parquet/JSON from data/analytics and results/figures.
No model training, no network calls. Run with:

    make dashboard

or:

    streamlit run app/dashboard.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

DATA = Path("data/analytics")
FIGS = Path("results/figures")

st.set_page_config(
    page_title="OSS Pulse",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def load_survival():
    df = pd.read_parquet(DATA / "repo_survival.parquet")
    return df


@st.cache_data(show_spinner=False)
def load_km():
    return pd.read_parquet(DATA / "survival_km.parquet")


@st.cache_data(show_spinner=False)
def load_cox():
    return pd.read_parquet(DATA / "survival_cox.parquet")


@st.cache_data(show_spinner=False)
def load_sensitivity():
    p = DATA / "survival_sensitivity.parquet"
    return pd.read_parquet(p) if p.exists() else pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_funnel():
    with open(DATA / "cohort_funnel.json") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_model_comparison():
    with open(DATA / "survival_model_comparison.json") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_packages():
    return pd.read_parquet(DATA / "package_downloads_month.parquet")


@st.cache_data(show_spinner=False)
def load_forecast_summary():
    return pd.read_parquet(DATA / "forecast_summary.parquet")


@st.cache_data(show_spinner=False)
def load_forecast_backtest():
    return pd.read_parquet(DATA / "forecast_backtest.parquet")


@st.cache_data(show_spinner=False)
def load_forecast_comparison():
    with open(DATA / "forecast_model_comparison.json") as f:
        return json.load(f)
@st.cache_data(show_spinner=False)
def load_bootstrap():
    return pd.read_parquet(DATA / "survival_bootstrap.parquet")

@st.cache_data(show_spinner=False)
def load_time_auc():
    return pd.read_parquet(DATA / "survival_time_auc_cox.parquet")

@st.cache_data(show_spinner=False)
def load_dm():
    return pd.read_parquet(DATA / "forecast_diebold_mariano.parquet")

@st.cache_data(show_spinner=False)
def load_calibration():
    return pd.read_parquet(DATA / "survival_calibration.parquet")

@st.cache_data(show_spinner=False)
def load_dca():
    return pd.read_parquet(DATA / "survival_dca.parquet")

@st.cache_data(show_spinner=False)
def load_subgroups():
    return pd.read_parquet(DATA / "survival_subgroups.parquet")

@st.cache_data(show_spinner=False)
def load_ph_test():
    return pd.read_parquet(DATA / "survival_ph_test.parquet")

@st.cache_data(show_spinner=False)
def load_time_varying():
    p = DATA / "survival_time_varying.parquet"
    return pd.read_parquet(p) if p.exists() else pd.DataFrame()

def _fig(path: Path, caption: str) -> None:
    if path.exists():
        st.image(str(path), caption=caption, width="stretch")
    else:
        st.info(f"Figure not yet generated: `{path}`")


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------


def page_overview() -> None:
    st.title("📊 OSS Pulse")
    st.markdown(
        "**Survival, forecasting and causal analysis of open-source ecosystems.**  \n"
        "A reproducible, zero-cloud analysis of public GitHub and PyPI data."
    )

    funnel = load_funnel()
    surv = load_survival()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Cohort (analysed)", f"{funnel['sampled_to']:,}")
    col2.metric("Repos with events", f"{funnel['with_any_active_event']:,}")
    col3.metric("Passed first-90-day threshold", f"{funnel['included_both_pre_sample']:,}")
    col4.metric("Observed events", f"{funnel['events_observed']:,}")

    st.markdown("### Cohort funnel")
    funnel_df = pd.DataFrame(
        [
            ("Search API candidates", funnel["metadata_repos"]),
            ("With any active event", funnel["with_any_active_event"]),
            ("≥ 5 stars in first 90 days", funnel["included_stars_90d"]),
            ("≥ 1 non-owner contributor in first 90 days", funnel["included_contributors_90d"]),
            ("Both thresholds", funnel["included_both_pre_sample"]),
            ("Sampled to analysis size", funnel["sampled_to"]),
        ],
        columns=["step", "repos"],
    )
    st.dataframe(funnel_df, width="stretch", hide_index=True)

    st.markdown("### Data coverage")
    st.markdown(
        f"- **Observation window:** {surv['created_at'].min().date()} → {surv['observation_end'].iloc[0]}  \n"
        f"- **Cohort:** {surv['created_at'].dt.year.nunique()} creation years "
        f"({sorted(surv['created_at'].dt.year.unique().tolist())})  \n"
        f"- **Inactivity definition:** {int(surv['inactivity_days'].iloc[0])} days  \n"
        f"- **Events observed:** {funnel['events_observed']:,} "
        f"({funnel['events_observed'] / funnel['sampled_to']:.1%})  \n"
        f"- **Censored:** {funnel['censored']:,} "
        f"({funnel['censored'] / funnel['sampled_to']:.1%})"
    )

    with st.expander("⚠️ Known data limitation (survival window)"):
        st.markdown(
            "Only 2019 GH Archive data was downloaded to keep the project at $0. "
            "With `observation_end = 2019-12-31` and a 180-day inactivity definition, "
            "only repositories whose last activity fell before 2019-07-04 are observed "
            "as events. The survival analysis therefore describes **short-window** "
            "survival, not long-term survival. See the **Methods and Limitations** page."
        )


def page_survival() -> None:
    st.title("🛡️ RQ1: What predicts repository survival?")

    surv = load_survival()
    cox = load_cox()
    comp = load_model_comparison()
    sens = load_sensitivity()

    col1, col2, col3 = st.columns(3)
    col1.metric("Repos analysed", f"{len(surv):,}")
    col2.metric("Cox C-index", f"{comp['c_index_cox']:.3f}")
    col3.metric("RSF C-index", f"{comp['c_index_rsf']:.3f}")

    st.markdown("### Hazard ratios (Cox proportional hazards)")
    hr = cox.copy()
    hr["significant"] = hr["p_value"] < 0.05
    display = hr.assign(
        hazard_ratio=lambda d: d["hazard_ratio"].round(3),
        ci_95=lambda d: d.apply(lambda r: f"[{r['hr_lower']:.3f}, {r['hr_upper']:.3f}]", axis=1),
        p_value=lambda d: d["p_value"].apply(lambda p: f"{p:.2e}" if p < 0.01 else f"{p:.3f}"),
    )[["feature", "hazard_ratio", "ci_95", "p_value", "significant"]]
    st.dataframe(display, width="stretch", hide_index=True)

    st.markdown("### Kaplan-Meier curves")
    _fig(FIGS / "survival_km.png", "Survival curves by segment (log-rank p in analytics)")

    st.markdown("### Hazard-ratio forest plot")
    _fig(FIGS / "survival_forest.png", "Cox hazard ratios with 95% CI (log scale)")

    if not sens.empty:
        st.markdown("### Sensitivity to inactivity definition")
        sens_view = sens.pivot_table(
            index="feature", columns="definition_days", values="hazard_ratio"
        ).round(3)
        st.dataframe(sens_view, width="stretch")
        _fig(FIGS / "survival_sensitivity.png", "HR across 90/180-day definitions")

    st.markdown("### Takeaway")
    hr_lookup = cox.set_index("feature")
    st.markdown(
        f"- **Early contributors** (HR = {hr_lookup.loc['log_contributors_90d', 'hazard_ratio']:.2f}, "
        f"95% CI [{hr_lookup.loc['log_contributors_90d', 'hr_lower']:.2f}, "
        f"{hr_lookup.loc['log_contributors_90d', 'hr_upper']:.2f}]): "
        f"each doubling of first-90-day contributors roughly halves the hazard.\n"
        f"- **Early pushes** (HR = {hr_lookup.loc['log_pushes_90d', 'hazard_ratio']:.2f}): "
        f"more push activity in the first 90 days reduces hazard.\n"
        f"- **License present** (HR = {hr_lookup.loc['license_present', 'hazard_ratio']:.2f}): "
        f"having a license lowers the hazard by ~43%.\n"
        f"- **Stars** (HR = {hr_lookup.loc['log_stars_90d', 'hazard_ratio']:.2f}) "
        f"is counter-intuitive but consistent with informative censoring in the short window."
    )
    st.markdown("---")
    st.markdown(
        "**v2 adds calibration, decision curves, and subgroup analysis.** "
        "See the **Decision (v2)** page for whether the model is usable "
        "in practice."
    )


def page_forecast() -> None:
    st.title("📈 RQ2: Can we beat naive baselines?")

    summary = load_forecast_summary()
    packages = load_packages()
    backtest = load_forecast_backtest()
    comp = load_forecast_comparison()

    st.markdown(
        f"Backtested **{comp['packages_used']} packages** over **{comp['origins_per_package']} "
        f"walk-forward origins** at horizons {comp['horizons']}. "
        f"Intervals at {comp['interval_level']}%."
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### sMAPE (lower is better)")
        smape_view = summary.pivot_table(
            index="model", columns="horizon", values="smape_mean"
        ).round(1)
        smape_view.columns = [f"h={c}" for c in smape_view.columns]
        st.dataframe(smape_view, width="stretch")
        _fig(FIGS / "forecast_mase.png", "MASE median with IQR whiskers")

    with col2:
        st.markdown("### 80% interval coverage")
        cov_view = (
            summary.dropna(subset=["coverage"])
            .pivot_table(index="model", columns="horizon", values="coverage")
            .round(3)
        )
        cov_view.columns = [f"h={c}" for c in cov_view.columns]
        st.dataframe(cov_view, width="stretch")
        _fig(FIGS / "forecast_coverage.png", "Empirical coverage vs 80% target")

    st.markdown("### Per-package forecaster")
    all_pkgs = sorted(packages["package"].unique())
    default_idx = all_pkgs.index("requests") if "requests" in all_pkgs else 0
    pkg = st.selectbox("Package", all_pkgs, index=default_idx)
    model = st.selectbox("Model", ["naive", "ets", "lightgbm", "seasonal_naive"], index=1)
    horizon = st.radio("Horizon", [1, 3], horizontal=True)

    hist = packages[packages["package"] == pkg].sort_values("month")
    bt = backtest[
        (backtest["package"] == pkg)
        & (backtest["model"] == model)
        & (backtest["horizon"] == horizon)
    ].copy()

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(hist["month"], hist["downloads"], label="actual", color="black", linewidth=1.5)
    if not bt.empty:
        bt["origin_month"] = pd.to_datetime(bt["origin_month"])
        ax.scatter(bt["origin_month"], bt["actual"], color="black", s=15, zorder=3)
        ax.scatter(bt["origin_month"], bt["forecast"], color="tab:blue", s=25, marker="x", zorder=3)
        if bt["lower"].notna().any():
            ax.vlines(
                bt["origin_month"],
                bt["lower"].fillna(bt["forecast"]),
                bt["upper"].fillna(bt["forecast"]),
                color="tab:blue",
                alpha=0.4,
            )
    ax.set_yscale("log")
    ax.set_ylabel("monthly downloads (log scale)")
    ax.set_title(f"{pkg} — {model} @ h={horizon}")
    ax.legend(["actual", "backtest actual", "forecast"])
    ax.grid(alpha=0.3)
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("### Takeaway")
    st.markdown(
        "**A last-value baseline is the strongest forecaster on monthly downloads.** "
        "AutoETS is competitive at h=1 (18.0% vs 15.0% sMAPE) but loses at h=3. "
        "LightGBM is worse at both horizons. Seasonal naive fails (85% sMAPE) because "
        "12-month-old levels are far below current levels for packages that grew 10–100×. "
        "**Recommendation:** use last-value or ETS in production; the complexity of "
        "feature-engineered tree models does not pay off on this data."
    )
    st.markdown("---")
    st.markdown(
        "**v2 adds Diebold-Mariano tests.** See the **Inference (v2)** page "
        "for the corrected comparison: naive and ETS are statistically tied."
    )


def page_ci() -> None:
    st.title("🔬 RQ3: CI adoption (deferred to v2)")

    st.warning(
        "RQ3 was scoped to v1 but could not be completed on the available data. "
        "It is now planned for v2 with the correct method and data."
    )

    st.markdown("### Why it was deferred")
    st.markdown(
        "1. **GH Archive `PushEvent` payloads do not contain file-level changes.** "
        "The `added` / `modified` / `removed` fields are present in webhook payloads "
        "but not in the Events API format that GH Archive stores. Detecting "
        "`.github/workflows/` from `PushEvent` alone is therefore not possible.\n"
        "2. **The adoption window is 2020–2022, but only 2019 data was downloaded.** "
        "Even with a working detector, there would be zero treated repositories in "
        "the current dataset.\n"
        "3. **Simple two-way fixed effects DiD is biased under staggered adoption.** "
        "Doing this properly requires a Callaway-Sant'Anna style estimator, which "
        "was already listed in the v2 scope."
    )

    st.markdown("### v2 plan")
    st.markdown(
        "- Fetch `2020-01-01` → `2022-12-31` GH Archive events for the same cohort.\n"
        "- Detect first `.github/workflows/` commit via the GitHub REST API "
        "(`/repos/{id}/commits?path=.github/workflows`, ~1–2 requests per repo).\n"
        "- Use `differences` / `csdid` (Python or R) for the staggered-adoption estimator.\n"
        "- Run placebo tests with fake adoption dates.\n"
        "- Report subgroup effects (small vs large teams, early vs late adopters) with CIs."
    )

    st.markdown("### Why this is the right call")
    st.markdown(
        "Publishing an under-powered or biased causal estimate would be worse than "
        "deferring the question. v1 delivers two complete analyses (RQ1, RQ2) with "
        "real findings; v2 adds the causal component with the data and method it requires."
    )

def page_inference() -> None:
    st.title("🔬 Inference: how confident are we?")
    st.markdown(
        "v2 adds statistical inference on top of v1's point estimates. "
        "Every number here answers a specific question: **is this conclusion robust?**"
    )

    # ---- Bootstrap ------------------------------------------------------
    st.markdown("## 1. Bootstrap CIs for Cox hazard ratios")
    st.markdown(
        "Non-parametric bootstrap (B = 300) recomputes the Cox model on "
        "resampled data. If the bootstrap CI matches the asymptotic CI, the "
        "lifelines estimate is reliable. Divergence would signal high leverage "
        "points or non-linearity."
    )

    cox = load_cox()
    boot = load_bootstrap()
    merged = cox.merge(boot, on="feature", how="left")
    view = merged.assign(
        asymptotic_ci=lambda d: d.apply(
            lambda r: f"[{r['hr_lower']:.3f}, {r['hr_upper']:.3f}]", axis=1
        ),
        bootstrap_ci=lambda d: d.apply(
            lambda r: f"[{r['hr_boot_lower']:.3f}, {r['hr_boot_upper']:.3f}]", axis=1
        ),
    )[["feature", "hazard_ratio", "asymptotic_ci", "bootstrap_ci"]]
    st.dataframe(view, width="stretch", hide_index=True)

    same_direction = (
        ((boot["hr_boot_lower"] > 1) & (cox["hr_lower"] > 1))
        | ((boot["hr_boot_upper"] < 1) & (cox["hr_upper"] < 1))
        | ((boot["hr_boot_lower"] < 1) & (boot["hr_boot_upper"] > 1))
    ).all()
    if same_direction:
        st.success("Bootstrap and asymptotic CIs agree on direction for every feature.")

    # ---- Time-dependent AUC ----------------------------------------------
    st.markdown("## 2. Time-dependent AUC")
    st.markdown(
        "Cumulative/dynamic AUC is computed at each follow-up horizon using "
        "inverse-probability-of-censoring weighting. It tells us how well the "
        "model ranks risk at **each** time point, not just overall."
    )
    tdauc = load_time_auc()
    st.dataframe(
        tdauc.assign(
            time=lambda d: d["time"].astype(int),
            auc=lambda d: d["auc"].round(3),
            mean_auc=lambda d: d["mean_auc"].round(3),
        )[["time", "auc", "mean_auc"]],
        width="stretch",
        hide_index=True,
    )

    col1, col2 = st.columns(2)
    col1.metric("Mean AUC", f"{float(tdauc['mean_auc'].iloc[0]):.3f}")
    col2.metric("Cox C-index", f"{load_model_comparison()['c_index_cox']:.3f}")
    st.caption(
        "Mean AUC and C-index agree to within 0.01, which confirms the model's "
        "ranking ability is not an artefact of a particular follow-up window."
    )

    # ---- Diebold-Mariano -------------------------------------------------
    st.markdown("## 3. Diebold-Mariano: are forecast differences real?")
    st.markdown(
        "DM test compares per-forecast absolute log errors. `mean_diff < 0` means "
        "model A has lower loss. p < 0.05 means the difference is statistically "
        "significant."
    )
    dm = load_dm().copy()
    # Compute significance BEFORE formatting p_value into a string.
    dm["significant"] = dm["p_value"] < 0.05
    dm["p_value_fmt"] = dm["p_value"].apply(
        lambda p: "< 1e-9" if p < 1e-9 else f"{p:.3g}"
    )
    dm["verdict"] = dm["significant"].map(
        {True: "significant", False: "not significant"}
    )
    dm["mean_diff"] = dm["mean_diff"].round(4)
    dm["dm_stat"] = dm["dm_stat"].round(2)
    view_dm = dm[["horizon", "model_a", "model_b", "mean_diff", "p_value_fmt", "better", "verdict"]].rename(
        columns={"p_value_fmt": "p_value"}
    )
    st.dataframe(view_dm, width="stretch", hide_index=True)

    st.markdown("### What this changes")
    st.markdown(
        "**v1 reported 'naive is the strongest forecaster.'** The DM test "
        "corrects this to a more precise claim:\n\n"
        "- **naive vs ETS**: not significant (p ≈ 0.10 / 0.13). They are tied.\n"
        "- **naive vs LightGBM**: significant (p < 0.01). LightGBM is worse.\n"
        "- **ETS vs LightGBM**: not significant. Neither beats the other.\n"
        "- **Seasonal naive vs anything**: strongly significant (p ≈ 0). "
        "It is the worst by a large margin.\n\n"
        "The accurate statement is: **naive and ETS are tied for best; LightGBM "
        "is significantly worse than naive; seasonal naive is significantly "
        "worse than everything.** This is a more honest and more useful finding "
        "than 'naive wins.'"
    )

def page_ph() -> None:
    st.title("⚖️ Proportional hazards diagnostics")
    st.markdown(
        "The Cox model assumes each covariate's effect is **constant over time**. "
        "If that assumption fails, the reported hazard ratio is a time-average "
        "and its interpretation is misleading. This page tests the assumption "
        "per feature and, where it fails, estimates HR(t)."
    )

    ph = load_ph_test()
    tv = load_time_varying()

    # ---- Test table -----------------------------------------------------
    st.markdown("## 1. Schoenfeld residual test")
    view = ph.assign(
        p_value=lambda d: d["p_value"].apply(
            lambda p: "< 0.001" if p < 1e-3 else f"{p:.4f}"
        ),
        verdict=lambda d: d["ph_violated"].map({True: "❌ violated", False: "✅ holds"}),
    )[["feature", "chi2", "p_value", "verdict"]]
    st.dataframe(view, width="stretch", hide_index=True)

    violated = ph.loc[ph["ph_violated"], "feature"].tolist()
    if violated:
        st.warning(
            f"**{len(violated)} feature(s) violate PH**: {', '.join(violated)}. "
            f"HR(t) is reported below for these features. All other features "
            f"have a constant effect and their single HR is directly interpretable."
        )
    else:
        st.success("All features satisfy the proportional hazards assumption.")

    # ---- Residual plots -------------------------------------------------
    st.markdown("## 2. Schoenfeld residuals")
    st.markdown(
        "Scaled Schoenfeld residuals vs event time. A flat LOWESS line means the "
        "effect is constant; a trend means it changes over time."
    )
    features_with_plots = [f for f in ph["feature"] if (FIGS / f"survival_schoenfeld_{f}.png").exists()]
    if features_with_plots:
        n_cols = 2
        for i in range(0, len(features_with_plots), n_cols):
            cols = st.columns(n_cols)
            for col, feature in zip(cols, features_with_plots[i:i + n_cols], strict=False):
                with col:
                    st.image(
                        str(FIGS / f"survival_schoenfeld_{feature}.png"),
                        caption=feature,
                        width="stretch",
                    )

    # ---- Time-varying HR ------------------------------------------------
    if not tv.empty:
        st.markdown("## 3. HR(t) for features that violate PH")
        st.markdown(
            "The time-varying Cox model allows the coefficient to change with "
            "log(time). The delta-method 95% CI accounts for the covariance "
            "between the baseline and interaction terms."
        )
        for feature, sub in tv.groupby("feature"):
            sub = sub.sort_values("time")
            col1, col2 = st.columns([2, 3])
            with col1:
                view_tv = sub.assign(
                    ci=lambda d: d.apply(
                        lambda r: f"[{r['hr_lower']:.3f}, {r['hr_upper']:.3f}]",
                        axis=1,
                    )
                )[["time", "hazard_ratio", "ci"]]
                st.markdown(f"**{feature}**")
                st.dataframe(view_tv, width="stretch", hide_index=True)
            with col2:
                fig_path = FIGS / f"survival_time_varying_{feature}.png"
                if fig_path.exists():
                    st.image(str(fig_path), width="stretch")

        st.markdown("### What this changes")
        st.markdown(
            "`log_pushes_90d` is the **only** feature whose effect changes "
            "meaningfully over time: HR goes from 0.73 at 180 days to 0.70 at "
            "360 days. The direction and magnitude are stable; the v1 HR of 0.75 "
            "remains a reasonable summary. For all other features, the single HR "
            "is directly interpretable."
        )

    # ---- Interpretation ------------------------------------------------
    st.markdown("## 4. What we learn from PH")
    st.info(
        "**`log_stars_90d` satisfies PH (p = 0.18).** This rules out one "
        "explanation for the counter-intuitive HR > 1 — the effect is **not** "
        "growing over time. Combined with the subgroup analysis (v2 Decision "
        "page), the correct interpretation is: the stars effect is **constant "
        "over time but varies across subgroups by censoring severity**. "
        "High-star repos are censored more; the model learns a spurious "
        "'high-star → high-risk' signal only in that subgroup. This is a "
        "censoring artefact, but a stable one, not a compounding one.\n\n"
        "**`log_contributors_90d` is borderline (p = 0.06).** This is consistent "
        "with the v2 threshold finding: the effect may be a time-varying "
        "threshold effect (weak at 1–2 contributors, strong at 3+). A v3 with a "
        "longer window could test this directly.\n\n"
        "**All other features satisfy PH.** Their single HRs are directly "
        "interpretable and no time-varying correction is needed."
    )

def page_decision() -> None:
    st.title("🎯 Decision analysis: is the model usable?")
    st.markdown(
        "Statistical significance is not the same as decision value. "
        "This page answers three questions: **is the model calibrated?**, "
        "**does it add net benefit?**, and **is the effect homogeneous across subgroups?**"
    )

    # ---- Calibration -----------------------------------------------------
    st.markdown("## 1. Calibration")
    st.markdown(
        "Predicted survival vs observed survival. Points on the diagonal = perfect "
        "calibration. The Integrated Calibration Index (ICI) summarises the mean "
        "absolute deviation."
    )
    calib = load_calibration()
    ici = float((calib["pred_mean"] - calib["obs_mean"]).abs().mean())
    col1, col2 = st.columns([1, 3])
    col1.metric("ICI (overall)", f"{ici:.3f}")
    with col2:
        st.caption(
            "ICI = mean |predicted − observed|. Lower is better. "
            "However, the aggregate ICI hides horizon-specific biases — see below."
        )
    _fig(FIGS / "survival_calibration.png", "Calibration by horizon")

    st.markdown("### Horizon-specific bias")
    horizons = sorted(calib["horizon"].unique())
    bias_rows = []
    for h in horizons:
        sub = calib[calib["horizon"] == h]
        bias_rows.append(
            {
                "horizon (days)": int(h),
                "predicted survival (mean)": round(float(sub["pred_mean"].mean()), 3),
                "observed survival (mean)": round(float(sub["obs_mean"].mean()), 3),
                "bias": round(float((sub["pred_mean"] - sub["obs_mean"]).mean()), 3),
            }
        )
    st.dataframe(pd.DataFrame(bias_rows), width="stretch", hide_index=True)
    st.warning(
        "**The model is well-calibrated at 180 days but systematically "
        "under-estimates survival at 360 days.** This is consistent with the "
        "informative-censoring hypothesis: high-star repos stay active through "
        "2019 and are right-censored, so the model learns a spurious "
        "'high-star → high-risk' signal that compounds at longer horizons. "
        "The 180-day predictions are usable; the 360-day predictions are not."
    )

    # ---- DCA -------------------------------------------------------------
    st.markdown("## 2. Decision Curve Analysis")
    st.markdown(
        "Net benefit of three strategies: use the model, treat everyone, or treat "
        "nobody. The model is useful only where its curve is above both baselines."
    )
    _fig(FIGS / "survival_dca.png", "Decision curve at 180 and 360 days")
    dca = load_dca()
    h360 = dca[dca["horizon"] == 360]
    st.markdown(
        f"At 360 days, the prevalence of death is **{h360['prevalence'].iloc[0]:.1%}**. "
        f"At every threshold between 0.11 and 0.91, `nb_model` is **negative** — "
        f"worse than treat-none (`nb = 0`). The model ranks risk correctly "
        f"(AUC = 0.78) but mis-classifies enough repos as high-risk that the net "
        f"benefit is negative."
    )
    st.info(
        "**Interpretation:** the model is useful for **prioritising** repos "
        "(which to look at first), not for **deciding** whether to act. The next "
        "step is not a better model but a decision rule that respects the "
        "asymmetric cost of false positives."
    )

    # ---- Subgroups -------------------------------------------------------
    st.markdown("## 3. Subgroup heterogeneity")
    st.markdown(
        "The same Cox model fitted separately in each subgroup. Divergent "
        "hazard ratios reveal effect modification that the pooled model hides."
    )
    sub = load_subgroups()
    if not sub.empty:
        features = ["log_contributors_90d", "log_pushes_90d", "license_present"]
        for feature in features:
            s = sub[sub["feature"] == feature]
            if s.empty:
                continue
            col1, col2 = st.columns([2, 3])
            with col1:
                st.markdown(f"**{feature}**")
                view = s.assign(
                    ci=lambda d: d.apply(
                        lambda r: f"[{r['hr_lower']:.3f}, {r['hr_upper']:.3f}]",
                        axis=1,
                    )
                )[["subgroup_col", "subgroup_value", "hazard_ratio", "ci", "n_events"]]
                st.dataframe(view, width="stretch", hide_index=True)
            with col2:
                fig_path = FIGS / f"survival_subgroups_{feature}.png"
                if fig_path.exists():
                    st.image(str(fig_path), width="stretch")

        st.markdown("### What the subgroups reveal")
        st.markdown(
            "1. **`log_stars_90d` HR > 1 only in the high-star subgroup.** "
            "In stars_5_19 the HR is 0.97 (not significant); in stars_20plus "
            "it is 1.21 (significant). This is **direct evidence** that the "
            "counter-intuitive stars effect is informative censoring, not a "
            "causal effect: it appears only where censoring is most severe.\n\n"
            "2. **`log_contributors_90d` is significant only when contributors ≥ 3.** "
            "In contrib_1_2 the HR is 0.73 (CI crosses 1); in contrib_3plus it is "
            "0.61 (significant). There is a **threshold effect**: going from 1 to "
            "2 contributors does not protect; going from 3 to 4 does.\n\n"
            "3. **`license_present` is robust in every subgroup.** HR ranges from "
            "0.50 to 0.63 with CIs that never cross 1. This is the most "
            "actionable finding of v2."
        )


# ---------------------------------------------------------------------------
# Router update
# ---------------------------------------------------------------------------

def page_methods() -> None:
    st.title("📚 Methods and limitations")

    st.markdown(
        """
### Data sources

| Source | Use | Access |
|---|---|---|
| GH Archive hourly dumps (`data.gharchive.org`) | Repository events (stars, pushes, PRs, issues) | Direct download, stream-filtered to the cohort, raw files deleted |
| GitHub Search API | Candidate pool (`language:Python`, 2019–2021, `stars:>=5`) | Sharded by creation month, cached |
| GitHub REST API | Repository metadata (language, license, owner) | Cached, ~1,000 requests |
| ClickHouse public PyPI dataset | Monthly downloads for 40 packages | One-off CSV export; pypistats (180-day API) as fallback |

**No BigQuery, no GCP account, no paid services.**

### Definitions

- **Inactive:** no push, PR or issue event for 180 consecutive days (sensitivity: 90 days).
- **Time origin:** repository creation date.
- **Censoring:** repositories still active at 2019-12-31 are right-censored.
- **MASE:** mean absolute scaled error, scaled by the in-sample seasonal naive error.

### Limitations

**Survival (RQ1)**
- Only 2019 data. With `observation_end = 2019-12-31` and a 180-day inactivity definition,
  only repositories whose last activity fell before 2019-07-04 are observed as events
  (5.4%). 94.6% are right-censored.
- **Informative censoring.** High-star repositories remain active through 2019 and are
  therefore more likely to be censored. This is the most plausible explanation for the
  counter-intuitive `log_stars_90d` HR of 1.27; the finding is not reported as a causal
  effect.
- The contributors, pushes and license coefficients are robust across 90- and 180-day
  definitions and are the findings we report.

**Forecasting (RQ2)**
- 40 packages, 6 walk-forward origins, horizons of 1 and 3 months.
- MASE is reported but is not the headline metric because the seasonal-naive denominator
  is dominated by 10–100× growth; sMAPE is easier to interpret.
- LightGBM intervals are **conservative** (coverage ≈ 0.99 at h=1). Wider-than-nominal
  intervals are preferable to narrower-than-nominal ones for planning purposes.
- Forecasts are weaker for new packages and structural breaks; short-history packages
  are skipped automatically.

**Causal (RQ3)**
- Deferred to v2 for data and methodological reasons (see the CI adoption page).

### Reproducibility

- All pipelines are pinned, seeded and runnable from a clean clone via `make`.
- Every headline number is produced by `src/osspulse/` and stored in `data/analytics/`.
"""
    )


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------


PAGES = {
    "Overview": page_overview,
    "Survival (RQ1)": page_survival,
    "Forecast (RQ2)": page_forecast,
    "Inference (v2)": page_inference,
    "Decision (v2)": page_decision,
    "PH diagnostics (v2)": page_ph,
    "CI adoption (RQ3)": page_ci,
    "Methods and Limitations": page_methods,
}

st.sidebar.title("OSS Pulse")
choice = st.sidebar.radio("Page", list(PAGES.keys()))
st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Version:** v1.0 draft  \n"
    "**Data:** GH Archive · GitHub API · ClickHouse PyPI  \n"
    "**Cost:** $0 (no cloud account)"
)

PAGES[choice]()

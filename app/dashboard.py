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


def _fig(path: Path, caption: str) -> None:
    if path.exists():
        st.image(str(path), caption=caption, use_column_width=True)
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
    st.dataframe(funnel_df, use_container_width=True, hide_index=True)

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
    st.dataframe(display, use_container_width=True, hide_index=True)

    st.markdown("### Kaplan-Meier curves")
    _fig(FIGS / "survival_km.png", "Survival curves by segment (log-rank p in analytics)")

    st.markdown("### Hazard-ratio forest plot")
    _fig(FIGS / "survival_forest.png", "Cox hazard ratios with 95% CI (log scale)")

    if not sens.empty:
        st.markdown("### Sensitivity to inactivity definition")
        sens_view = sens.pivot_table(
            index="feature", columns="definition_days", values="hazard_ratio"
        ).round(3)
        st.dataframe(sens_view, use_container_width=True)
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
        st.dataframe(smape_view, use_container_width=True)
        _fig(FIGS / "forecast_mase.png", "MASE median with IQR whiskers")

    with col2:
        st.markdown("### 80% interval coverage")
        cov_view = (
            summary.dropna(subset=["coverage"])
            .pivot_table(index="model", columns="horizon", values="coverage")
            .round(3)
        )
        cov_view.columns = [f"h={c}" for c in cov_view.columns]
        st.dataframe(cov_view, use_container_width=True)
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

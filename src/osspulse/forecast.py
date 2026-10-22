"""RQ2: Walk-forward forecasting with MASE and 80% interval coverage.

Reads analytics/package_downloads_month.parquet (from `make pypi`), runs an
expanding-window backtest at up to 6 origins per package, and compares 4
models: naive, seasonal naive, AutoETS, and LightGBM.

Outputs:
  analytics/forecast_metrics.parquet      — one row per (package, model, horizon, origin)
  analytics/forecast_summary.parquet      — aggregated per (model, horizon)
  analytics/forecast_backtest.parquet     — actual vs forecast (long form, for dashboard)
  analytics/forecast_model_comparison.json
  results/figures/forecast_mase.png
  results/figures/forecast_coverage.png
"""

from __future__ import annotations

import json
import warnings
from dataclasses import asdict, dataclass

import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import AutoETS

from osspulse import paths
from osspulse.config import Config
from osspulse.logging import get

warnings.filterwarnings(
    "ignore",
    message="divide by zero encountered in scalar divide",
    module="statsforecast",
)
log = get(__name__)

MIN_TRAIN_MONTHS = 18
N_ORIGINS = 6
HORIZONS = (1, 3)
H_MAX = max(HORIZONS)
INTERVAL_LEVEL = 80
LGBM_LAGS = (1, 2, 3, 12)


# ---------------------------------------------------------------------------
# Simple baselines
# ---------------------------------------------------------------------------


def naive_forecast(history: np.ndarray, h: int) -> np.ndarray:
    return np.full(h, history[-1], dtype=float)


def seasonal_naive_forecast(history: np.ndarray, h: int, season: int = 12) -> np.ndarray:
    if len(history) < season:
        return naive_forecast(history, h)
    last = history[-season:]
    return np.array([last[i % season] for i in range(h)], dtype=float)


# ---------------------------------------------------------------------------
# LightGBM recursive forecaster
# ---------------------------------------------------------------------------


def _build_lgbm_dataset(series: np.ndarray, lags: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
    max_lag = max(lags)
    X, y = [], []
    for t in range(max_lag, len(series)):
        X.append([series[t - lag] for lag in lags])
        y.append(series[t])
    return np.array(X, dtype=float), np.array(y, dtype=float)


def lgbm_forecast(
    history: np.ndarray, h: int, seed: int = 42
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """LightGBM point forecast with calibrated relative-error intervals.

    Design: the point model is trained on ALL history so it captures the most
    recent level. A separate calibration model is trained on a truncated window
    to produce genuine one-step out-of-sample residuals; those residuals are
    turned into a relative error quantile and used to scale the point forecast
    into an interval. This avoids the trend-extrapolation failure of a single
    model fit on truncated data.
    """
    n = len(history)
    min_len = max(LGBM_LAGS) + 10
    if n < min_len:
        return np.full(h, np.nan), np.full(h, np.nan), np.full(h, np.nan)

    def _fit(series: np.ndarray) -> lgb.LGBMRegressor | None:
        if len(series) < max(LGBM_LAGS) + 5:
            return None
        X, y = _build_lgbm_dataset(series, LGBM_LAGS)
        if len(X) < 5:
            return None
        m = lgb.LGBMRegressor(
            n_estimators=200,
            learning_rate=0.05,
            num_leaves=15,
            min_child_samples=5,
            verbose=-1,
            random_state=seed,
        )
        m.fit(X, y)
        return m

    # 1) Point model: full history.
    point_model = _fit(history)
    if point_model is None:
        return np.full(h, np.nan), np.full(h, np.nan), np.full(h, np.nan)

    # 2) Calibration model: truncated history, used only to estimate errors.
    calib_n = max(6, min(24, n // 4))
    train_trunc = history[: n - calib_n]
    calib_model = _fit(train_trunc)

    rel_errors: list[float] = []
    if calib_model is not None:
        rolling = list(train_trunc.astype(float))
        for actual in history[n - calib_n :]:
            feats = [rolling[-lag] for lag in LGBM_LAGS]
            pred = float(calib_model.predict([feats])[0])
            if pred > 0 and actual > 0:
                rel_errors.append(abs(float(actual) - pred) / pred)
            rolling.append(float(actual))

    q_rel = float(np.percentile(rel_errors, 80)) if rel_errors else 0.1
    # Floor at 2% so degenerate fits do not produce zero-width intervals,
    # widen with sqrt(h) for multi-step horizons.
    q_rel = max(q_rel, 0.02) * float(np.sqrt(h))

    # 3) Recursive point forecast from end of full history.
    history_list = list(history.astype(float))
    point: list[float] = []
    for _ in range(h):
        feats = [history_list[-lag] for lag in LGBM_LAGS]
        pred = float(point_model.predict([feats])[0])
        point.append(pred)
        history_list.append(pred)

    point_arr = np.asarray(point, dtype=float)
    # 4) Interval: relative to each step's point forecast.
    lower = point_arr * (1.0 - q_rel)
    upper = point_arr * (1.0 + q_rel)
    return point_arr, lower, upper


# ---------------------------------------------------------------------------
# Statsforecast (AutoETS) — batched per package
# ---------------------------------------------------------------------------


def _ets_forecast_all_origins(
    series: np.ndarray, origins: list[int]
) -> dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Fit AutoETS once per origin, batched via StatsForecast.

    Returns {origin: (point, lower, upper)} with arrays of length h_max.
    Missing origins (fit failures) are absent from the dict.
    """
    dfs = []
    for i, origin in enumerate(origins):
        train = series[:origin]
        dfs.append(
            pd.DataFrame(
                {
                    "unique_id": [f"o{i}"] * len(train),
                    "ds": pd.date_range("2000-01-01", periods=len(train), freq="MS"),
                    "y": train,
                }
            )
        )
    combined = pd.concat(dfs, ignore_index=True)

    try:
        sf = StatsForecast(
            models=[AutoETS(season_length=12)],
            freq="MS",
            n_jobs=1,  # safer on macOS
        )
        sf.fit(combined)
        forecasts = sf.predict(h=H_MAX, level=[INTERVAL_LEVEL])
    except Exception as exc:  # noqa: BLE001
        log.warning("AutoETS batch fit failed: %s", exc)
        return {}

    point_col = next(
        c
        for c in forecasts.columns
        if c not in ("unique_id", "ds") and "lo" not in c and "hi" not in c
    )
    lo_col = f"{point_col}-lo-{INTERVAL_LEVEL}"
    hi_col = f"{point_col}-hi-{INTERVAL_LEVEL}"

    out: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for i, origin in enumerate(origins):
        sub = forecasts[forecasts["unique_id"] == f"o{i}"]
        if len(sub) != H_MAX:
            continue
        point = sub[point_col].to_numpy(dtype=float)
        lo = sub[lo_col].to_numpy(dtype=float) if lo_col in sub else np.full(H_MAX, np.nan)
        hi = sub[hi_col].to_numpy(dtype=float) if hi_col in sub else np.full(H_MAX, np.nan)
        out[origin] = (point, lo, hi)
    return out


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def mase(actual: np.ndarray, forecast: np.ndarray, train: np.ndarray, season: int = 12) -> float:
    mae = float(np.mean(np.abs(actual - forecast)))
    if len(train) > season:
        naive_err = np.abs(np.diff(train, n=season))
    else:
        naive_err = np.abs(np.diff(train, n=1))
    denom = float(np.mean(naive_err))
    if denom == 0 or np.isnan(denom):
        return float("nan")
    return mae / denom


def smape(actual: np.ndarray, forecast: np.ndarray) -> float:
    denom = (np.abs(actual) + np.abs(forecast)) / 2.0
    safe = np.where(denom > 0, denom, 1.0)
    return float(np.mean(np.abs(actual - forecast) / safe) * 100)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class ForecastRow:
    package: str
    model: str
    horizon: int
    origin_month: str
    origin_index: int
    train_months: int
    actual: float
    forecast: float
    lower: float | None
    upper: float | None
    mase: float
    smape: float
    covered: bool | None


# ---------------------------------------------------------------------------
# Per-package backtest
# ---------------------------------------------------------------------------


def backtest_package(
    package: str,
    series: np.ndarray,
    months: pd.Series,
    seed: int,
) -> list[ForecastRow]:
    T = len(series)
    if T < MIN_TRAIN_MONTHS + H_MAX:
        log.info("skip %s: only %d months (need >= %d)", package, T, MIN_TRAIN_MONTHS + H_MAX)
        return []

    max_origin = T - H_MAX
    possible = list(range(MIN_TRAIN_MONTHS, max_origin + 1))
    if not possible:
        return []
    if len(possible) <= N_ORIGINS:
        origins = possible
    else:
        idx = np.linspace(0, len(possible) - 1, N_ORIGINS).astype(int)
        origins = [possible[i] for i in idx]

    ets_results = _ets_forecast_all_origins(series, origins)

    rows: list[ForecastRow] = []
    for origin in origins:
        train = series[:origin].astype(float)
        actual_hmax = series[origin : origin + H_MAX].astype(float)
        origin_month = months.iloc[origin] if origin < len(months) else pd.NaT

        # Baselines
        naive = naive_forecast(train, H_MAX)
        snaive = seasonal_naive_forecast(train, H_MAX)

        # LightGBM
        lgbm_point, lgbm_lo, lgbm_hi = lgbm_forecast(train, H_MAX, seed=seed)

        # ETS
        ets_entry = ets_results.get(origin)
        if ets_entry is not None:
            ets_point, ets_lo, ets_hi = ets_entry
        else:
            ets_point = np.full(H_MAX, np.nan)
            ets_lo = np.full(H_MAX, np.nan)
            ets_hi = np.full(H_MAX, np.nan)

        candidates = {
            "naive": (naive, np.full(H_MAX, np.nan), np.full(H_MAX, np.nan)),
            "seasonal_naive": (snaive, np.full(H_MAX, np.nan), np.full(H_MAX, np.nan)),
            "ets": (ets_point, ets_lo, ets_hi),
            "lightgbm": (lgbm_point, lgbm_lo, lgbm_hi),
        }

        for model_name, (point, lo, hi) in candidates.items():
            if np.all(np.isnan(point)):
                continue
            for h in HORIZONS:
                a = actual_hmax[:h]
                f = point[:h]
                if np.any(np.isnan(f)) or len(a) != h:
                    continue
                m = mase(a, f, train)
                s = smape(a, f)
                covered: bool | None = None
                lower_val: float | None = None
                upper_val: float | None = None
                if not np.any(np.isnan(lo[:h])):
                    l_arr = lo[:h]
                    u_arr = hi[:h]
                    inside = (a >= l_arr) & (a <= u_arr)
                    covered = bool(np.all(inside))
                    # report the h-th-step interval for the aggregated row
                    lower_val = float(l_arr[-1])
                    upper_val = float(u_arr[-1])
                rows.append(
                    ForecastRow(
                        package=package,
                        model=model_name,
                        horizon=h,
                        origin_month=str(origin_month)[:7],
                        origin_index=origin,
                        train_months=origin,
                        actual=float(a[-1]),
                        forecast=float(f[-1]),
                        lower=lower_val,
                        upper=upper_val,
                        mase=m,
                        smape=s,
                        covered=covered,
                    )
                )
    return rows


# ---------------------------------------------------------------------------
# Aggregation and plots
# ---------------------------------------------------------------------------


def summarise(rows: list[ForecastRow]) -> pd.DataFrame:
    df = pd.DataFrame([asdict(r) for r in rows])
    if df.empty:
        return df
    agg = (
        df.groupby(["model", "horizon"])
        .agg(
            n_forecasts=("mase", "size"),
            mase_mean=("mase", "mean"),
            mase_median=("mase", "median"),
            mase_p25=("mase", lambda x: np.nanpercentile(x, 25)),
            mase_p75=("mase", lambda x: np.nanpercentile(x, 75)),
            smape_mean=("smape", "mean"),
            coverage=(
                "covered",
                lambda x: (
                    float(np.mean([v for v in x if v is not None]))
                    if any(v is not None for v in x)
                    else np.nan
                ),
            ),
        )
        .reset_index()
    )
    return agg


def plot_mase_by_model(summary: pd.DataFrame, path) -> None:
    if summary.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    models = sorted(summary["model"].unique())
    width = 0.2
    x = np.arange(len(models))
    for j, h in enumerate(HORIZONS):
        sub = summary[summary["horizon"] == h].set_index("model").reindex(models)
        ax.bar(x + (j - 0.5) * width, sub["mase_median"], width, label=f"h={h}")
        ax.errorbar(
            x + (j - 0.5) * width,
            sub["mase_median"],
            yerr=[
                sub["mase_median"] - sub["mase_p25"],
                sub["mase_p75"] - sub["mase_median"],
            ],
            fmt="none",
            ecolor="black",
            capsize=3,
        )
    ax.axhline(1.0, color="grey", linestyle="--", alpha=0.6, label="seasonal naive = 1.0")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel("MASE (median, IQR whiskers)")
    ax.set_title("Forecast accuracy by model")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_coverage(summary: pd.DataFrame, path) -> None:
    if summary.empty:
        return
    cov = summary.dropna(subset=["coverage"])
    if cov.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    for h in HORIZONS:
        sub = cov[cov["horizon"] == h]
        ax.bar(
            [f"{m} (h={h})" for m in sub["model"]],
            sub["coverage"],
            label=f"h={h}",
        )
    ax.axhline(0.80, color="grey", linestyle="--", alpha=0.6, label="target 80%")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("80% interval coverage")
    ax.set_title("Prediction interval coverage")
    ax.legend()
    plt.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    import argparse

    from osspulse import logging as oss_log

    parser = argparse.ArgumentParser(description="RQ2: walk-forward forecast backtest.")
    parser.add_argument("--config", default="configs/cohort.yaml")
    parser.add_argument("--packages", default="configs/packages.yaml")
    args = parser.parse_args()

    cfg = Config.load(args.config)
    oss_log.setup(cfg.logging.level)
    pkg_cfg = Config.load_packages(args.packages)

    src = paths.analytics_dir() / "package_downloads_month.parquet"
    if not src.exists():
        raise FileNotFoundError(f"{src} not found. Run `make pypi` first.")

    df = pd.read_parquet(src)
    df["month"] = pd.to_datetime(df["month"], utc=True)
    log.info(
        "loaded %d rows, %d packages, %s -> %s",
        len(df),
        df["package"].nunique(),
        df["month"].min().date(),
        df["month"].max().date(),
    )

    wanted = set(pkg_cfg.all_names())
    all_rows: list[ForecastRow] = []
    skipped = []
    for package in sorted(wanted & set(df["package"].unique())):
        sub = df[df["package"] == package].sort_values("month")
        series = sub["downloads"].to_numpy(dtype=float)
        rows = backtest_package(
            package, series, sub["month"].reset_index(drop=True), cfg.cohort.sample_seed
        )
        if not rows:
            skipped.append(package)
            continue
        all_rows.extend(rows)
        log.info("%s: %d forecast rows", package, len(rows))

    if not all_rows:
        raise RuntimeError("No forecasts produced. Check packages.yaml and data.")

    metrics = pd.DataFrame([asdict(r) for r in all_rows])
    summary = summarise(all_rows)

    # Diebold-Mariano: pairwise model comparison per horizon
    from osspulse.inference import absolute_log_loss, diebold_mariano

    dm_rows: list[dict] = []
    models = sorted(metrics["model"].unique())
    for h in HORIZONS:
        sub = metrics[metrics["horizon"] == h]
        for a, b in [(m1, m2) for m1 in models for m2 in models if m1 < m2]:
            pivot = (
                sub.pivot_table(
                    index=["package", "origin_month"],
                    columns="model",
                    values=["actual", "forecast"],
                    aggfunc="first",
                )
                .dropna()
                .reset_index()
            )
            if a not in pivot["forecast"] or b not in pivot["forecast"]:
                continue
            actual = pivot["actual"][a].to_numpy()
            fa = pivot["forecast"][a].to_numpy()
            fb = pivot["forecast"][b].to_numpy()
            loss_a = absolute_log_loss(actual, fa)
            loss_b = absolute_log_loss(actual, fb)
            res = diebold_mariano(loss_a, loss_b, lag=max(0, h - 1))
            dm_rows.append(
                {
                    "horizon": h,
                    "model_a": a,
                    "model_b": b,
                    "n": res.n,
                    "mean_diff": res.mean_diff,
                    "dm_stat": res.dm_stat,
                    "p_value": res.p_value,
                    "better": res.better,
                }
            )
    dm_df = pd.DataFrame(dm_rows)
    if not dm_df.empty:
        dm_df.to_parquet(paths.analytics_dir() / "forecast_diebold_mariano.parquet", index=False)
        log.info("Diebold-Mariano:\n%s", dm_df.to_string(index=False))

    metrics.to_parquet(paths.analytics_dir() / "forecast_metrics.parquet", index=False)
    summary.to_parquet(paths.analytics_dir() / "forecast_summary.parquet", index=False)
    backtest_cols = [
        "package",
        "model",
        "horizon",
        "origin_month",
        "actual",
        "forecast",
        "lower",
        "upper",
    ]
    metrics[backtest_cols].to_parquet(
        paths.analytics_dir() / "forecast_backtest.parquet", index=False
    )

    plot_mase_by_model(summary, paths.figures_dir() / "forecast_mase.png")
    plot_coverage(summary, paths.figures_dir() / "forecast_coverage.png")

    comparison = {
        "packages_used": int(metrics["package"].nunique()),
        "packages_skipped": skipped,
        "origins_per_package": N_ORIGINS,
        "horizons": list(HORIZONS),
        "interval_level": INTERVAL_LEVEL,
        "summary": summary.to_dict(orient="records"),
    }
    (paths.analytics_dir() / "forecast_model_comparison.json").write_text(
        json.dumps(comparison, indent=2, default=str)
    )

    log.info("forecast summary:\n%s", summary.to_string(index=False))
    log.info("skipped packages (short history): %s", skipped)


if __name__ == "__main__":
    main()

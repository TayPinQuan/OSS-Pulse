"""Statistical inference utilities for v2.

- bootstrap_ci: non-parametric bootstrap CIs for any statistic
- diebold_mariano: pairwise forecast comparison with HAC variance
- time_dependent_auc: cumulative/dynamic AUC for survival models
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore", category=RuntimeWarning)


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------


@dataclass
class BootstrapResult:
    estimate: float
    lower: float
    upper: float
    n_boot: int
    level: float
    boot_distribution: np.ndarray


def bootstrap_ci(
    data: pd.DataFrame,
    statistic: Callable[[pd.DataFrame], float],
    *,
    n_boot: int = 500,
    level: float = 0.95,
    seed: int = 42,
    cluster_col: str | None = None,
) -> BootstrapResult:
    """Bootstrap CI for an arbitrary statistic.

    `cluster_col` resamples entire clusters (e.g. repo_id) instead of rows,
    which is the correct procedure when rows within a cluster are dependent.
    """
    rng = np.random.default_rng(seed)
    n = len(data)
    alpha = (1 - level) / 2

    boots = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        if cluster_col is not None:
            clusters = data[cluster_col].dropna().unique()
            chosen = rng.choice(clusters, size=len(clusters), replace=True)
            idx = np.concatenate([data.index[data[cluster_col] == c].to_numpy() for c in chosen])
            sample = data.loc[idx]
        else:
            idx = rng.integers(0, n, size=n)
            sample = data.iloc[idx]
        try:
            boots[i] = statistic(sample)
        except Exception:  # noqa: BLE001
            boots[i] = np.nan

    valid = boots[~np.isnan(boots)]
    if len(valid) < 10:
        return BootstrapResult(
            estimate=float(statistic(data)),
            lower=np.nan,
            upper=np.nan,
            n_boot=n_boot,
            level=level,
            boot_distribution=boots,
        )

    return BootstrapResult(
        estimate=float(statistic(data)),
        lower=float(np.percentile(valid, 100 * alpha)),
        upper=float(np.percentile(valid, 100 * (1 - alpha))),
        n_boot=int(len(valid)),
        level=level,
        boot_distribution=boots,
    )


# ---------------------------------------------------------------------------
# Diebold-Mariano
# ---------------------------------------------------------------------------


@dataclass
class DMResult:
    n: int
    mean_diff: float
    dm_stat: float
    p_value: float
    better: str  # "model_a" if model_a has lower loss


def diebold_mariano(
    loss_a: np.ndarray,
    loss_b: np.ndarray,
    *,
    lag: int = 0,
    h0: float = 0.0,
) -> DMResult:
    """Diebold-Mariano test for equal predictive accuracy.

    loss_a and loss_b are per-forecast loss arrays of the same length and in
    the same order. A negative mean difference means model A has lower loss.
    `lag` sets the Newey-West truncation; use horizon - 1 for h-step forecasts.

    Edge case: if the two loss series are identical (mean and variance of the
    difference both zero), the test statistic is 0/0. We return dm_stat = 0
    and p_value = 1.0, which is the correct interpretation of "no difference".
    """
    a = np.asarray(loss_a, dtype=float)
    b = np.asarray(loss_b, dtype=float)
    mask = ~(np.isnan(a) | np.isnan(b))
    a, b = a[mask], b[mask]
    n = len(a)
    if n < 5:
        return DMResult(n=n, mean_diff=np.nan, dm_stat=np.nan, p_value=np.nan, better="tie")

    d = a - b
    mean_d = float(d.mean())

    # If the two series are identical, there is nothing to test.
    if np.allclose(d, 0.0, atol=1e-12):
        return DMResult(n=n, mean_diff=0.0, dm_stat=0.0, p_value=1.0, better="tie")

    # Newey-West long-run variance
    gamma0 = float(np.var(d, ddof=0))
    lrv = gamma0
    for k in range(1, lag + 1):
        cov = float(np.cov(d[k:], d[:-k], ddof=0)[0, 1])
        lrv += 2 * (1 - k / (lag + 1)) * cov
    if lrv <= 0:
        lrv = max(gamma0, 1e-12)

    dm = mean_d / np.sqrt(lrv / n)

    # Two-sided p-value against standard normal
    p = float(2 * (1 - stats.norm.cdf(abs(dm))))
    better = "model_a" if mean_d < -h0 else ("model_b" if mean_d > h0 else "tie")
    return DMResult(n=n, mean_diff=mean_d, dm_stat=float(dm), p_value=p, better=better)


def absolute_log_loss(actual: np.ndarray, forecast: np.ndarray) -> np.ndarray:
    """|log1p(actual) - log1p(forecast)|, the DM loss used in this project."""
    a = np.asarray(actual, dtype=float)
    f = np.asarray(forecast, dtype=float)
    a = np.clip(a, 0, None)
    f = np.clip(f, 0, None)
    return np.abs(np.log1p(a) - np.log1p(f))


# ---------------------------------------------------------------------------
# Time-dependent AUC
# ---------------------------------------------------------------------------


def time_dependent_auc(
    durations: np.ndarray,
    events: np.ndarray,
    risk_scores: np.ndarray,
    times: Sequence[float],
) -> pd.DataFrame:
    """Cumulative/dynamic AUC at a set of time points.

    Uses the standard IPCW estimator (Uno et al. 2007). The risk score is the
    predicted *hazard*: higher = higher risk of an event by time t. If your
    model outputs survival probability, pass 1 - S(t).
    """
    from sksurv.metrics import cumulative_dynamic_auc
    from sksurv.util import Surv

    y = Surv.from_arrays(event=events.astype(bool), time=durations.astype(float))
    t_arr = np.asarray(times, dtype=float)
    auc, mean_auc = cumulative_dynamic_auc(y, y, risk_scores, t_arr)
    return pd.DataFrame(
        {
            "time": t_arr,
            "auc": np.asarray(auc),
            "mean_auc": float(mean_auc),
        }
    )

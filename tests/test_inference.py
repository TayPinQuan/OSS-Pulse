from __future__ import annotations

import numpy as np
import pandas as pd

from osspulse.inference import absolute_log_loss, bootstrap_ci, diebold_mariano


def test_bootstrap_ci_contains_estimate() -> None:
    df = pd.DataFrame({"x": np.random.default_rng(0).normal(10, 1, size=200)})
    res = bootstrap_ci(df, lambda d: float(d["x"].mean()), n_boot=200)
    assert res.lower < res.estimate < res.upper
    assert res.n_boot > 100


def test_bootstrap_cluster() -> None:
    df = pd.DataFrame(
        {
            "repo_id": np.repeat(np.arange(50), 20),
            "x": np.random.default_rng(1).normal(0, 1, size=1000),
        }
    )
    res = bootstrap_ci(df, lambda d: float(d["x"].mean()), n_boot=100, cluster_col="repo_id")
    assert not np.isnan(res.lower)


def test_diebold_mariano_identical_models() -> None:
    rng = np.random.default_rng(2)
    a = rng.normal(0, 1, size=100)
    res = diebold_mariano(a, a)
    assert abs(res.mean_diff) < 1e-9
    assert res.p_value > 0.99


def test_diebold_mariano_different_models() -> None:
    rng = np.random.default_rng(3)
    a = rng.normal(0.0, 0.1, size=200)
    b = rng.normal(0.5, 0.1, size=200)
    res = diebold_mariano(a, b)
    assert res.mean_diff < 0
    assert res.p_value < 0.01
    assert res.better == "model_a"


def test_absolute_log_loss_zero() -> None:
    a = np.array([10.0, 100.0])
    assert np.allclose(absolute_log_loss(a, a), 0.0)

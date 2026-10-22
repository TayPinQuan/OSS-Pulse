from __future__ import annotations

import json
from pathlib import Path

from osspulse import pypi
from osspulse.config import PackagesConfig


def _fake_payload(name: str, dates: list[str]) -> dict:
    return {
        "package": name,
        "data": [
            {"category": "without_mirrors", "date": d, "downloads": 10 * (i + 1)}
            for i, d in enumerate(dates)
        ],
    }


def test_build_monthly_fills_missing_days(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("OSSPULSE_DATA_DIR", str(tmp_path))
    cache = tmp_path / "raw" / "pypi"
    cache.mkdir(parents=True)
    analytics_dir = tmp_path / "analytics"
    analytics_dir.mkdir(parents=True)

    # 3 days in Jan, one missing between 01 and 05
    payload = _fake_payload("pkgA", ["2020-01-01", "2020-01-05"])
    (cache / "pkgA.json").write_text(json.dumps(payload))

    pkg_cfg = PackagesConfig(established=["pkgA"], newer=[])
    df = pypi.build_monthly(pkg_cfg)
    assert df["package"].iloc[0] == "pkgA"
    assert len(df) == 1  # only Jan 2020
    # Jan total = day1 (10) + day2..4 (0) + day5 (20) = 30
    assert int(df["downloads"].iloc[0]) == 30


def test_build_monthly_skips_missing(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("OSSPULSE_DATA_DIR", str(tmp_path))
    cache = tmp_path / "raw" / "pypi"
    cache.mkdir(parents=True)
    (tmp_path / "analytics").mkdir()

    (cache / "pkgB.json").write_text(json.dumps({"_missing": True, "name": "pkgB"}))
    (cache / "pkgC.json").write_text(json.dumps(_fake_payload("pkgC", ["2020-02-01"])))

    pkg_cfg = PackagesConfig(established=["pkgB", "pkgC"], newer=[])
    df = pypi.build_monthly(pkg_cfg)
    assert set(df["package"]) == {"pkgC"}

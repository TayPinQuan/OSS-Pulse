from __future__ import annotations

from pathlib import Path

import pytest

from osspulse.config import Config


def test_load_valid(valid_cfg: Path) -> None:
    cfg = Config.load(valid_cfg)
    assert cfg.cohort.sample_size == 1000
    assert cfg.download.source == "gharchive_hourly"


def test_url_template_missing_hour(valid_cfg: Path) -> None:
    text = valid_cfg.read_text().replace("{date}-{hour}", "{date}")
    bad = valid_cfg.parent / "bad.yaml"
    bad.write_text(text)
    with pytest.raises(Exception, match="hour"):
        Config.load(bad)


def test_bad_date_order(valid_cfg: Path) -> None:
    text = valid_cfg.read_text().replace("created_from: 2019-01-01", "created_from: 2022-01-01")
    bad = valid_cfg.parent / "bad2.yaml"
    bad.write_text(text)
    with pytest.raises(ValueError):
        Config.load(bad)

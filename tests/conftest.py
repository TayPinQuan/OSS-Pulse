from __future__ import annotations

from pathlib import Path

import pytest

VALID_YAML = """
cohort:
  language: Python
  created_from: 2019-01-01
  created_to: 2021-12-31
  search_shard_by: month
  search_min_stars: 5
  search_max_stars_escalation: 10
  inclusion_min_stars_90d: 5
  inclusion_min_contributors_90d: 1
  inactivity_days: 180
  sample_size: 1000
  sample_seed: 42
  observation_end: 2021-12-31
download:
  source: gharchive_hourly
  url_template: "https://data.gharchive.org/{date}-{hour}.json.gz"
  keep_raw: false
  manifest_path: data/raw/gharchive/_download_manifest.jsonl
  max_parallel: 2
  retry_attempts: 3
  request_timeout_sec: 60.0
search_api:
  per_page: 100
  max_pages: 10
  min_interval_sec: 5.0
sources:
  pypi: pypistats
  ci_detection: gharchive_push
logging:
  level: INFO
bots:
  known_logins: []
"""


@pytest.fixture()
def valid_cfg(tmp_path: Path) -> Path:
    p = tmp_path / "cohort.yaml"
    p.write_text(VALID_YAML)
    return p


@pytest.fixture()
def cfg(valid_cfg: Path):
    from osspulse.config import Config

    return Config.load(valid_cfg)

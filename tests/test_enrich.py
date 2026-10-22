from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

from osspulse import enrich


def _fake_repo(rid: int, lang: str = "Python", license_spdx: str | None = "MIT") -> dict:
    return {
        "id": rid,
        "full_name": f"user/repo{rid}",
        "name": f"repo{rid}",
        "description": "x",
        "homepage": None,
        "language": lang,
        "topics": ["ml"],
        "created_at": "2019-01-01T00:00:00Z",
        "updated_at": "2021-01-01T00:00:00Z",
        "pushed_at": "2021-01-01T00:00:00Z",
        "stargazers_count": 10,
        "forks_count": 2,
        "watchers_count": 10,
        "open_issues_count": 3,
        "default_branch": "main",
        "archived": False,
        "disabled": False,
        "fork": False,
        "size": 1000,
        "has_issues": True,
        "has_projects": True,
        "has_wiki": False,
        "has_pages": False,
        "has_discussions": False,
        "visibility": "public",
        "license": {"spdx_id": license_spdx, "name": license_spdx} if license_spdx else None,
        "owner": {"login": "user"},
    }


def test_trim_keeps_expected_fields() -> None:
    raw = _fake_repo(42)
    trimmed = enrich._trim(raw)
    assert trimmed["id"] == 42
    assert trimmed["language"] == "Python"
    assert trimmed["license_spdx"] == "MIT"
    assert trimmed["owner_login"] == "user"
    assert "license" not in trimmed  # raw license dict is replaced by spdx


def test_trim_handles_null_license() -> None:
    raw = _fake_repo(1, license_spdx=None)
    raw["license"] = None
    trimmed = enrich._trim(raw)
    assert trimmed["license_spdx"] is None


def test_get_repo_404_returns_none(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("GITHUB_TOKEN", "fake")
    session = MagicMock()
    resp = MagicMock()
    resp.status_code = 404
    session.get.return_value = resp
    payload, status = enrich._get_repo(session, 999)
    assert payload is None
    assert status == 404


def test_get_repo_success(monkeypatch) -> None:
    monkeypatch.setenv("GITHUB_TOKEN", "fake")
    session = MagicMock()
    resp = MagicMock()
    resp.status_code = 200
    resp.headers = {"X-RateLimit-Remaining": "4000", "X-RateLimit-Reset": "0"}
    resp.json.return_value = _fake_repo(7)
    session.get.return_value = resp
    payload, status = enrich._get_repo(session, 7)
    assert status == 200
    assert payload is not None
    assert payload["id"] == 7


def test_get_repo_5xx_raises(monkeypatch) -> None:
    monkeypatch.setenv("GITHUB_TOKEN", "fake")
    session = MagicMock()
    resp = MagicMock()
    resp.status_code = 503
    session.get.return_value = resp
    with pytest.raises(enrich._ServerError):
        enrich._get_repo(session, 1)


def test_materialise_aggregates_cache(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("OSSPULSE_DATA_DIR", str(tmp_path))
    api_dir = tmp_path / "raw" / "github_api"
    api_dir.mkdir(parents=True)
    (api_dir / "1.json").write_text(json.dumps(_fake_repo(1)))
    (api_dir / "2.json").write_text(json.dumps(_fake_repo(2, lang="Go")))
    (api_dir / "3.json").write_text(json.dumps({"_deleted": True, "id": 3}))

    out = enrich.materialise()
    df = pd.read_parquet(out)
    assert len(df) == 3
    assert set(df["id"]) == {1, 2, 3}
    row3 = df[df["id"] == 3].iloc[0]
    assert bool(row3["_deleted"]) is True

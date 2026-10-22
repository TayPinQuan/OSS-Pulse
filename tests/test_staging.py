from __future__ import annotations

from pathlib import Path

import pandas as pd

from osspulse import analytics


def test_ci_adoption_window(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("OSSPULSE_DATA_DIR", str(tmp_path))
    staging = tmp_path / "staging"
    analytics_dir = tmp_path / "analytics"
    staging.mkdir(parents=True)
    analytics_dir.mkdir(parents=True)

    rows = pd.DataFrame(
        [
            {
                "event_id": "1",
                "repo_id": 100,
                "event_created_at": pd.Timestamp("2020-06-01T00:00Z"),
                "commit_sha": "x",
                "workflow_paths": ["a"],
            },
            {
                "event_id": "2",
                "repo_id": 200,
                "event_created_at": pd.Timestamp("2019-06-01T00:00Z"),
                "commit_sha": "y",
                "workflow_paths": ["a"],
            },
            {
                "event_id": "3",
                "repo_id": 300,
                "event_created_at": pd.Timestamp("2023-06-01T00:00Z"),
                "commit_sha": "z",
                "workflow_paths": ["a"],
            },
        ]
    )
    rows.to_parquet(staging / "ci_commits.parquet", index=False)

    df = analytics.build_ci_adoption()
    # 2019 and 2023 are outside the 2020-2022 window
    assert list(df["repo_id"]) == [100]

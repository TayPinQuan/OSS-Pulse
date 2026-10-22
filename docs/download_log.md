# Download Log

This file records every GH Archive download session so the total bandwidth cost is auditable without a cloud billing dashboard. See `PRD.md` section 9 ("Storage and download").

## Sessions

| Date (UTC) | Window | Hours requested | Hours completed | Hours failed | Bytes downloaded | Cumulative |
|---|---|---|---|---|---|---|
| 2026-10-21 | 2019-01-01 → 2019-01-07 | 168 | 168 | 0 | 1.95 GB | 1.95 GB |
| 2026-10-21 to 2026-10-22 | 2019-01-01 → 2019-12-31 | 8,760 | 8,568 (+168 skipped from session 1) | 24 | 267.9 GB | **269.9 GB** |

## Total

- **Compressed download:** ~270 GB
- **Wall clock:** ~18.5 hours for the full-year session (rate ≈ 4 MB/s, single connection)
- **Local disk peak:** < 1 GB (`raw/gharchive_parquet/` after filtering, 2019 only)
- **Local disk after 2019:** ~1.2 GB (2.28M events in Parquet, zstd)

## Failed hours

24 hours were not downloaded because the source returned HTTP 404. These are **source-side gaps**, not local errors. GH Archive publishes most hours but occasionally loses archives to crawler outages. Known examples:

- `2019-05-08-12` (HTTP 404)
- `2019-05-08-13` (HTTP 404)
- 22 additional hours, distribution documented in `raw/gharchive/_download_manifest.jsonl`

The manifest records these as `status: failed` with the HTTP error. Re-running `make download` does not retry them (the manifest treats them as terminal). Their absence removes a few hours of events from the cohort, which is negligible at monthly aggregation and does not affect any reported finding.

## Reproducing the download

```bash
# Set GITHUB_TOKEN in .env first
make candidates                  # 36 monthly Search API shards
make download --since 2019-01-01 --until 2019-12-31
```

Resumability is provided by `raw/gharchive/_download_manifest.jsonl`: every completed hour is recorded and skipped on re-run.

## Data source attribution

GH Archive is a public project that archives the GitHub public event stream. It is not affiliated with GitHub. Event payload schemas may change; the downloader is pinned to the Events API schema observed in the 2019 data and validated against it in `tests/test_download.py`.
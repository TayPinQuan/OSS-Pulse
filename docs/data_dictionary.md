# Data Dictionary

Every analytics table produced by `make analytics`, `make survival`, `make forecast`. All timestamps are UTC. All paths are relative to the repository root unless otherwise noted.

---

## `data/analytics/repo_survival.parquet`

**Grain:** one row per repository in the analysed cohort (n = 5,421).
**Produced by:** `osspulse.analytics.build_repo_survival`.

| Column | Type | Description |
|---|---|---|
| `repo_id` | int64 | GitHub numeric repository id (stable across renames) |
| `created_at` | timestamp[us, tz=UTC] | Repository creation timestamp from the GitHub REST API |
| `language` | string | Primary language from the GitHub REST API (filtered to `Python` by the cohort) |
| `license_spdx` | string \| null | SPDX identifier (`MIT`, `Apache-2.0`, …) or null if no license |
| `owner_login` | string | GitHub login of the repository owner (used to exclude the owner from contributor counts) |
| `description_length` | int | Character count of the repository description |
| `topics_count` | int | Number of GitHub topics |
| `stars_90d` | int | `WatchEvent` count in the first 90 days |
| `contributors_90d` | int | Distinct non-owner actors emitting Push/PR/Issue/PRReview events in the first 90 days |
| `issues_opened_90d` | int | Issues opened in the first 90 days |
| `prs_opened_90d` | int | Pull requests opened in the first 90 days |
| `pushes_90d` | int | Push events in the first 90 days |
| `pr_merge_rate_90d` | float \| null | Merged / closed PRs in the first 90 days; null if no PRs closed |
| `total_activity_90d` | int | Total events of any type in the first 90 days |
| `last_active_at` | timestamp[us, tz=UTC] \| null | Timestamp of the last Push/PR/Issue event observed in the data window; null if no active event was observed |
| `median_response_hours` | float \| null | Median hours between issue open and first comment/close, computed on the first 90 days only; null if no issues received a response |
| `included` | bool | Whether the repo passes both first-90-day thresholds (stars ≥ 5 **and** contributors ≥ 1). Always `true` in this file |
| `event_flag` | bool | `true` if the repo is observed as **inactive** (no active event for `inactivity_days` before `observation_end`) |
| `duration_days` | int | Days from creation to `event_flag` (if true) or to `observation_end` (if censored) |
| `inactivity_days` | int | Inactivity definition used (180 by default) |
| `observation_end` | date | Fixed observation window end (2019-12-31 in v1) |

**Cohort funnel** (see `cohort_funnel.md`): 35,998 candidates → 5,421 included → 294 events observed.

---

## `data/analytics/repo_month_panel.parquet`

**Grain:** one row per (repo, calendar month).
**Produced by:** `osspulse.analytics.build_repo_month_panel`.

| Column | Type | Description |
|---|---|---|
| `repo_id` | int64 | GitHub numeric repository id |
| `month` | timestamp[us, tz=UTC] | First day of the month |
| `activity_count` | int | Push + PR + Issue + PRReview events in the month |
| `contributor_count` | int | Distinct actors emitting active events in the month |
| `merged_pr_count` | int | PRs merged in the month |
| `issues_opened` | int | Issues opened in the month |
| `prs_opened` | int | PRs opened in the month |
| `first_workflow_date` | timestamp[us, tz=UTC] \| null | First `.github/workflows/` commit date. **Always null in v1** — CI detection is deferred to v2 |
| `first_workflow_month` | timestamp[us, tz=UTC] \| null | Month of `first_workflow_date`. **Always null in v1** |
| `treated` | bool | Whether the row is post-`first_workflow_date`. **Always false in v1** |
| `rel_month` | float \| null | Months relative to `first_workflow_month`. **Always null in v1** |

---

## `data/analytics/package_downloads_month.parquet`

**Grain:** one row per (package, calendar month).
**Produced by:** `osspulse.pypi._build_from_clickhouse_csv`.

| Column | Type | Description |
|---|---|---|
| `package` | string | PyPI package name |
| `month` | timestamp[us, tz=UTC] | First day of the month |
| `downloads` | int64 | Monthly download count (sum of daily counts, mirrors excluded) |

**Coverage:** 40 packages, 2016-01 → 2026-10. Missing months (no downloads that month) are omitted rather than zero-filled; downstream consumers should reindex if a continuous spine is needed.

---

## `data/analytics/cohort_funnel.json`

Single JSON object with the funnel counts used in the README and dashboard.

```json
{
  "metadata_repos": 35998,
  "with_any_active_event": 11079,
  "included_stars_90d": 7097,
  "included_contributors_90d": 7120,
  "included_both_pre_sample": 5421,
  "sampled_to": 5421,
  "sample_seed": 42,
  "events_observed": 294,
  "censored": 5127,
  "median_duration_days": 192.0
}
```

---

## `data/analytics/survival_km.parquet`

**Grain:** one row per (segment, group, survival step).
**Produced by:** `osspulse.survival.fit_km_segments`.

| Column | Type | Description |
|---|---|---|
| `timeline` | float | Days since creation |
| `survival` | float | Kaplan-Meier survival estimate S(t) |
| `segment` | string | One of `license_present`, `has_issue_response`, `stars_bucket` |
| `group` | string | Segment label (e.g. `Has license`, `stars 20+`) |

---

## `data/analytics/survival_cox.parquet`

**Grain:** one row per feature.
**Produced by:** `osspulse.survival.fit_cox`.

| Column | Type | Description |
|---|---|---|
| `feature` | string | Feature name (log-transformed numerics, binary flags) |
| `hazard_ratio` | float | `exp(coef)` from lifelines CoxPHFitter |
| `hr_lower` | float | Lower bound of 95% CI |
| `hr_upper` | float | Upper bound of 95% CI |
| `p_value` | float | Wald p-value |

**Model:** `CoxPHFitter(penalizer=0.01)`, `duration_col="duration_days"`, `event_col="event_flag"`.
**Holdout C-index:** 0.773 (see `survival_model_comparison.json`).

---

## `data/analytics/survival_sensitivity.parquet`

**Grain:** one row per (feature, inactivity definition).
**Produced by:** `osspulse.survival.sensitivity_table`.

Same columns as `survival_cox.parquet` plus:

| Column | Type | Description |
|---|---|---|
| `definition_days` | int | Inactivity definition (90 or 180) |
| `n_repos` | int | Repos used in the Cox fit for this definition |
| `n_events` | int | Events observed under this definition |
| `event_rate` | float | `n_events / n_repos` |

**Note:** the 365-day definition is not present because with `observation_end = 2019-12-31` no event can be observed at that threshold.

---

## `data/analytics/survival_model_comparison.json`

Single JSON object. Fields: `n_repos`, `event_rate`, `c_index_cox`, `c_index_rsf`, `rsf_importances` (permutation importance, drop in C-index), `km_logrank` (per-segment p-values).

---

## `data/analytics/forecast_metrics.parquet`

**Grain:** one row per (package, model, horizon, origin).
**Produced by:** `osspulse.forecast.backtest_package`.

| Column | Type | Description |
|---|---|---|
| `package` | string | PyPI package name |
| `model` | string | `naive`, `seasonal_naive`, `ets`, `lightgbm` |
| `horizon` | int | 1 or 3 (months) |
| `origin_month` | string | Backtest origin, `YYYY-MM` |
| `origin_index` | int | Index of the origin in the series |
| `train_months` | int | Months of training data at that origin |
| `actual` | float | Actual value at `origin + horizon` |
| `forecast` | float | Point forecast |
| `lower` | float \| null | 80% interval lower bound (null for naive and seasonal naive) |
| `upper` | float \| null | 80% interval upper bound |
| `mase` | float | Mean absolute scaled error |
| `smape` | float | Symmetric mean absolute percentage error |
| `covered` | bool \| null | Whether `actual` falls in `[lower, upper]` |

---

## `data/analytics/forecast_summary.parquet`

**Grain:** one row per (model, horizon).
**Produced by:** `osspulse.forecast.summarise`.

| Column | Type | Description |
|---|---|---|
| `model` | string | Same as `forecast_metrics` |
| `horizon` | int | 1 or 3 |
| `n_forecasts` | int | Number of forecast rows aggregated |
| `mase_mean` | float | Mean MASE across all rows |
| `mase_median` | float | Median MASE |
| `mase_p25` | float | 25th percentile |
| `mase_p75` | float | 75th percentile |
| `smape_mean` | float | Mean sMAPE |
| `coverage` | float \| null | Fraction of intervals covering `actual` |

---

## `data/analytics/forecast_backtest.parquet`

**Grain:** same as `forecast_metrics`, reduced to the columns needed by the dashboard: `package, model, horizon, origin_month, actual, forecast, lower, upper`.

---

## `data/analytics/forecast_model_comparison.json`

Single JSON object. Fields: `packages_used`, `packages_skipped`, `origins_per_package`, `horizons`, `interval_level`, `summary` (the `forecast_summary` table serialized).

---

## Not produced in v1

- `ci_adoption.parquet` — created by `osspulse.analytics.build_ci_adoption` but empty (0 treated repos). See RQ3 deferral in `PRD.md` section 5.3.
- `ci_commits.parquet` — created by `osspulse.staging.build_ci_commits` but empty (GH Archive `PushEvent` payloads do not contain file-level changes).

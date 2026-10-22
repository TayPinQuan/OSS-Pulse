# Cohort Funnel

This document records every filter step that produced the analysed cohort. The numbers are reproduced from `data/analytics/cohort_funnel.json` by `make analytics`.

## Final counts

| Step | Repositories | Retained |
|---|---|---|
| Search API candidates (metadata fetched via REST) | 35,998 | — |
| With at least one active event in the 2019 data window | 11,079 | 30.8% |
| ≥ 5 stars in the first 90 days | 7,097 | 19.7% |
| ≥ 1 non-owner contributor in the first 90 days | 7,120 | 19.8% |
| **Both thresholds met (analysed cohort)** | **5,421** | **15.1%** |
| Events observed (not censored) | 294 | 5.4% of cohort |
| Right-censored | 5,127 | 94.6% of cohort |

## Interpretation

**The two inclusion thresholds are not independent.** 7,097 repos pass the stars threshold and 7,120 pass the contributors threshold, but only 5,421 pass both. The overlap is not large because the two criteria measure different kinds of early traction: stars measure attention, contributors measure collaboration. A repository that attracts attention but no collaborators is common in the cohort.

**The 15.1% pass rate is expected.** The Search API screen uses `stars:>=5` on *current total stars*, which is a looser filter than the real inclusion rule (`stars >= 5` **in the first 90 days**). Many repos reach 5 stars only after their first year. Dropping them keeps the cohort free of survivorship bias: inclusion does not depend on what happens after day 90.

**The 5.4% event rate is a consequence of the short observation window.** With `observation_end = 2019-12-31` and `inactivity_days = 180`, a repo can only be observed as an event if its last active event fell before 2019-07-04. Repos that remained active through 2019 are censored at the window end. This is documented in the README limitations and is the main reason RQ3 (CI adoption) is deferred to v2.

## Reproducing this funnel

```bash
make analytics
cat data/analytics/cohort_funnel.json
```

The counts are also logged at `INFO` level by `python -m osspulse.analytics`.

## Historical note

An earlier run with `sample_size: 1000` produced n = 1,000 and 55 events. That sample was replaced by the full included cohort (5,421) to raise the events-per-variable ratio from 6.9 to 36.7, well above the conventional threshold of 10 for Cox models.

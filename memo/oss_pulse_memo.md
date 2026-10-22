# OSS Pulse — Decision Memo

**Prepared for:** Developer-platform leadership
**Question:** What predicts open-source repository survival, and can adoption be forecast better than naive baselines?

**Data:** 5,421 Python repositories created in 2019, included only if they reached 5 stars and ≥ 1 non-owner contributor in their first 90 days. GH Archive events for 2019; monthly PyPI downloads for 40 packages (2016–2026). No cloud services; cost $0.

---

## Finding 1 — Early contributors matter more than anything else

**Number:** A doubling of distinct non-owner contributors in the first 90 days is associated with a **48% lower hazard of death** (HR = 0.52, 95% CI [0.43, 0.65], p < 10⁻⁸). The effect is stable across 90-day and 180-day inactivity definitions (HR 0.42 / 0.52).

Early push activity is a second robust predictor (HR = 0.75, 95% CI [0.69, 0.82]), and having a **license** lowers hazard by 43% (HR = 0.57, 95% CI [0.46, 0.71]).

Issue-response speed and PR merge rate are **not** independently predictive once contributor count is controlled for.

**Recommendation:** Redesign onboarding to explicitly target **multiple contributors in the first 90 days** — for example, a "second maintainer" checklist, structured first-issue programs, and contributor-facing docs promoted on the repo landing page. Add license selection to the same onboarding flow. Do not over-invest in response-time SLAs; they do not move survival once contributor counts are held fixed.

---

## Finding 2 — Complex forecasters do not beat naive on package downloads

**Number:** Across 40 packages × 6 walk-forward origins:

| Model | sMAPE h=1 | sMAPE h=3 | 80% coverage |
|---|---|---|---|
| **Last-value (naive)** | **15.0%** | **23.6%** | — |
| AutoETS | 18.0% | 26.5% | 0.82 / 0.68 |
| LightGBM | 22.0% | 28.5% | 0.99 / 0.96 |
| Seasonal naive | 84.6% | 79.4% | — |

**Interpretation:** For mature Python packages with strong trend and weak seasonality, a last-value baseline is hard to beat. Seasonal naive fails because 12-month-old levels are 10–100× below current levels. LightGBM is worse than both naive and ETS despite feature engineering and tuning.

**Recommendation:** In production adoption forecasting, start with **last-value or AutoETS**. Reserve tree-based models for cases with strong calendar seasonality, stable scale, and long history. The saved engineering cost and the simpler operational footprint outweigh any small accuracy gain tree models might deliver elsewhere.

---

## Finding 3 — CI adoption analysis is deferred to v2

**Status:** Not delivered in v1.

**Reason:** GH Archive `PushEvent` payloads do not include file-level changes (`added`/`modified` are webhook-only), so `.github/workflows/` cannot be detected from archived events. The 2020–2022 CI adoption window also does not overlap the 2019 dataset we downloaded.

**Plan for v2:** Detect first workflow commit via the GitHub REST API, download 2020–2022 events, and use a Callaway-Sant'Anna staggered-adoption estimator with placebo tests. Publishing an under-powered or biased DiD would be worse than deferring the question.

**Recommendation:** Do not make CI-adoption decisions on the basis of this project's v1 output. Revisit once v2 is published with the correct design.

---

## Caveats

1. **Short survival window.** With only 2019 events and a 180-day inactivity definition, 94.6% of repositories are right-censored. The survival findings describe **short-window** survival (6–12 months), not long-term survival. A counter-intuitive hazard ratio for stars (HR = 1.27) is most plausibly a censoring artefact, not a causal effect, and is not reported as a finding.
2. **Forecast scope.** The naive-beats-complex result is specific to monthly download data with strong trend and weak seasonality. It may not generalise to series with strong calendar seasonality or mean-reverting behaviour.

---

*Every number in this memo is reproducible from `data/analytics/` via `make survival` and `make forecast`. See the dashboard and README for the full pipeline.*
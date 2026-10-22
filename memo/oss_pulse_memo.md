# OSS Pulse — Decision Memo (v2)

**Prepared for:** Developer-platform leadership
**Question:** What predicts open-source repository survival, and can adoption be forecast better than naive baselines?

**Data:** 5,421 Python repositories created in 2019, included only if they reached 5 stars and ≥ 1 non-owner contributor in their first 90 days. GH Archive events for 2019; monthly PyPI downloads for 40 packages (2016–2026). No cloud services; cost $0.

**v2 changes:** statistical inference (bootstrap, time-dependent AUC, Diebold-Mariano), calibration and decision curve analysis, subgroup heterogeneity.

---

## Finding 1 — Early contributors matter, but only beyond a threshold

**Number:** A doubling of distinct non-owner contributors in the first 90 days is associated with a **48% lower hazard of death** (HR = 0.52, 95% CI [0.43, 0.65]; bootstrap CI [0.45, 0.62]). The effect is stable across 90-day and 180-day inactivity definitions (HR 0.42 / 0.52).

**Subgroup caveat (new in v2):** the effect is **not significant** in repos with 1–2 contributors (HR = 0.73, CI crosses 1) but is **significant** in repos with 3+ contributors (HR = 0.61, [0.42, 0.89]). **Going from 1 to 2 contributors does not protect; going from 3 to 4 does.**

Early push activity is a second robust predictor (HR = 0.75, 95% CI [0.69, 0.82]). Having a **license** lowers hazard by 43% (HR = 0.57, 95% CI [0.46, 0.71]) and is the **only effect that is robust in every subgroup**.

Issue-response speed and PR merge rate are not independently predictive once contributor count is controlled for.

**Recommendation:** Redesign onboarding to target **3+ contributors in the first 90 days**, not just "more than zero." A "second maintainer" checklist, structured first-issue programs, and contributor-facing docs help; a single co-maintainer does not. Add license selection to the same onboarding flow. Do not over-invest in response-time SLAs.

**Proportional-hazards check (new in v2):** only `log_pushes_90d` violates the
PH assumption (p = 0.011); its HR is 0.73 at 180 days and 0.70 at 360 days, so
the reported HR of 0.75 is a reasonable time-average. `log_stars_90d`
satisfies PH (p = 0.18), which rules out a compounding-over-time explanation
for its HR > 1 — the effect is a stable subgroup difference driven by
differential censoring, not a growing bias.

---

## Finding 2 — Naive and ETS are tied for best; LightGBM is significantly worse

**Number:** Across 40 packages × 6 walk-forward origins:

| Model | sMAPE h=1 | sMAPE h=3 | DM vs naive |
|---|---|---|---|
| **Naive** | **15.0%** | **23.6%** | — |
| AutoETS | 18.0% | 26.5% | p = 0.10 / 0.13 (tied) |
| LightGBM | 22.0% | 28.5% | p < 0.01 (naive better) |
| Seasonal naive | 84.6% | 79.4% | p ≈ 0 (worse) |

**Diebold-Mariano tests (new in v2):** naive vs ETS is **not significant** at either horizon. The correct statement is "naive and ETS are tied for best," not "naive wins." LightGBM is significantly worse than naive but not significantly different from ETS. Seasonal naive is significantly worse than everything.

**Recommendation:** In production adoption forecasting, use **naive or ETS**. Reserve tree-based models for cases with strong calendar seasonality and stable scale. The engineering cost of feature-engineering and tuning does not pay off on this data.

---

## Finding 3 — The survival model is not yet decision-usable

**v2 calibration (new):** Cox calibration is near-perfect at 180 days but **systematically under-estimates survival at 360 days** (predicted 0.52 vs observed 0.84 for the highest-risk bin). The Integrated Calibration Index (ICI) of 0.064 is dragged down by the well-calibrated 180-day bin; at 360 days the model is meaningfully mis-calibrated.

**v2 decision curve (new):** at 360 days, the model's net benefit is **negative** at every threshold between 0.11 and 0.91 — worse than treating nobody. The model still beats treating everyone (net benefit at threshold 0.21: model −0.043 vs treat-all −0.197), so it ranks risk correctly (AUC 0.78), but the false-positive cost exceeds the true-positive benefit.

**Why:** informative censoring. High-star repositories remain active through 2019 and are right-censored; the model learns a spurious "high-star → high-risk" signal that compounds at longer horizons. Subgroup analysis confirms this: `log_stars_90d` HR = 0.97 (not significant) in the low-star subgroup, but HR = 1.21 (significant) in the high-star subgroup — exactly where censoring is most severe.

**Recommendation:** Do not use this survival model to decide whether to intervene on a repository. Use it to **prioritise** which repositories to look at first. A decision rule that accounts for asymmetric costs (better to miss a death than to waste maintainer time on a false alarm) would require a different model or a v3 data window with 2020–2022 events.

---

## Caveats

1. **Short survival window.** Only 2019 events. 94.6% of repos are right-censored. Findings describe short-window survival (6–12 months), not long-term survival. The stars hazard ratio is a censoring artefact, not a causal effect.
2. **Calibration diverges at 360 days.** The 180-day predictions are usable; the 360-day predictions are not. This is a direct consequence of the short window and is not fixable without more data.
3. **Forecast scope.** The naive/ETS tie holds for monthly download data with strong trend and weak seasonality. It may not generalise to series with strong calendar seasonality.

---

*Every number in this memo is reproducible from `data/analytics/` via `make survival`, `make survival-advanced`, and `make forecast`. See the dashboard's Inference and Decision pages for the full v2 evidence.*
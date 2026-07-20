# Career Intelligence System

**Honest, calibrated wage guidance for people who don't have industry insiders to ask.**

## Problem

Early-career job seekers — students without industry networks, first-generation
professionals, international students unfamiliar with a foreign labor market —
rely on fragmented, anecdotal, or self-interested sources for salary expectations:
friends' half-remembered numbers, recruiter-provided ranges (which skew low), or
crowdsourced sites with unclear methodology and no stated confidence.

This project gives a wage **range**, not a false-precision point estimate, states
its own confidence explicitly (calibrated, not decorative), and explains *why* it
produced that range.

## Data

- **O\*NET 30.3 Database** (U.S. Dept. of Labor, CC BY 4.0) — skill-importance
  profiles per occupation (SOC code).
- **BLS OEWS, May 2025** — real wage percentiles (10th/25th/median/75th/90th) and
  employment counts per occupation.

Both are free, direct-download, government-sourced — no scraping, no ToS risk,
no API keys.

## Method

1. **Skill → Occupation Matching** — cosine similarity between a user's
   self-reported skill profile and O*NET's combined Essential + Transferable
   skill-importance matrix (35 elements across 894 occupations with skill
   survey data).
2. **Wage Range Output** — pull BLS OEWS percentiles for matched occupation(s).
3. **Model-Based Refinement** — a monotonic-constrained gradient-boosted
   regressor (XGBoost) trained on melted (occupation × percentile) rows
   predicts finer-grained position within the percentile band than a raw
   lookup allows, and extends wage estimates to occupations with no direct
   BLS match.
4. **Conformal Prediction Layer** — wraps point predictions in a
   split-conformal interval calibrated on held-out data; coverage is
   validated both marginally and conditionally by subgroup (Job Zone, SOC
   major group, percentile level).
5. **Explainability (SHAP)** — SHAP attributes each prediction to individual
   skills and to distributional "context" (percentile, job zone), reported
   separately so a lay user isn't shown "50th percentile" as if it were a
   skill. Dollar effects are reported multiplicatively (`exp(shap) - 1`,
   e.g. "+12%"), since the model predicts log-wage and SHAP values are
   additive in log-space, not dollar-space.

## Validated Results

**Crosswalk coverage (Phase 0):** 94.1% of O*NET occupations (956/1,016) find
a direct BLS OEWS wage match via the SOC crosswalk.

**Skill-matching sanity check (Phase 0):** cosine similarity over the combined
35-element skill space surfaces plausible, distinct occupation clusters for
both a quantitative test profile (Biostatisticians, Statisticians, Computer
Programmers) and a people-management test profile (Social and Community
Service Managers, IT Project Managers, First-Line Supervisors) — see
`notebooks/01_eda_onet_oews.ipynb`.

**Baseline wage model (Phase 1):** trained on 4,310 melted training rows
(862 occupations × 5 percentiles, after dropping 94 occupations with no
skill-survey data), split by occupation to prevent leakage.

| Metric | Model | Naive (percentile-only) |
|---|---|---|
| RMSE (log-wage) | 0.269 | 0.456 |
| MAE (dollars) | $19,586 | $31,943 |
| Improvement | **38.7%** | — |

Monotonicity (predicted wage never decreases 10th → 90th percentile): **0
violations across 173 held-out test occupations.**

**Conformal prediction layer (Phase 2):** split-conformal calibration on a
held-out calibration set (173 occupations, disjoint from train and test),
target 90% coverage.

- **Marginal coverage: 90.3%** (target 90%), mean interval width $69,017.
- **By Job Zone:** 94.7% (Zone 2) to 85.0% (Zone 4) — notably, Job Zone 2,
  the population closest to this project's early-career focus, is the
  *best*-covered group.
- **By percentile level:** a fully-powered, monotonic decline from 92.5%
  (10th percentile) to 86.7% (90th percentile) — see Limitations.
- Group-conditional (Mondrian) calibration was implemented and tested, but
  not applied by default — no subgroup crossed the 85% undercoverage
  threshold used as its trigger.

**Explainability layer (Phase 3):** SHAP (`TreeExplainer`) additivity holds
cleanly across all 865 held-out test rows — max error 1.24e-05, mean error
4.95e-06, both far inside a 1e-3 tolerance.

- **Context dominates the naive ranking:** `percentile` and `job_zone`
  together account for **38.1% of total |SHAP| mass**, with `percentile`
  alone outranking every individual skill. This is why context features are
  reported separately from skill drivers rather than mixed into one ranked
  list — a naive top-5 feature importance list would otherwise show "50th
  percentile" as the single largest driver, which is meaningless to a lay
  user. The skill-only ranking, once separated, is coherent: Critical
  Thinking, Complex Problem Solving, and Judgment and Decision Making top
  the list.
- **Real-wage sanity check is mixed, and reported honestly rather than
  cherry-picked:** for Light Truck Drivers, the model predicted $44,882
  against an actual BLS median of $44,860 — a near-exact match. For Chemical
  Technicians, it predicted $76,142 against an actual $60,390 — a ~26%
  overprediction, driven up by Science and Critical Thinking. The
  explanation makes this discrepancy interpretable even when the point
  estimate itself misses, which is the actual value proposition of this
  layer.

**Test coverage:** 50 tests passing across crosswalk logic, skill-matching
math, occupation feature construction, the baseline model, conformal
calibration, and SHAP explainability.

## Status

✅ Phase 0 — Setup, data acquisition, EDA — complete
✅ Phase 1 — Skill matching + baseline wage model — complete
✅ Phase 2 — Conformal prediction layer — complete
✅ Phase 3 — Explainability (SHAP) — complete
⬜ Phase 4 — API + tests
⬜ Phase 5 — Frontend + deploy
⬜ Phase 6 — Real-user testing
⬜ Phase 7 — Polish + docs

See `notebooks/` for the full exploratory record — each phase's notebook
documents its own validation before the logic moved into `src/`.

## Repo Structure

```
src/
├── data/       # O*NET + BLS ingestion and cleaning
├── features/   # skill-matching logic
├── models/     # training (wage_model.py), conformal wrapper (conformal.py),
│               # SHAP integration (Phase 3)
├── api/        # FastAPI endpoints (Phase 4)
└── app/        # frontend dashboard (Phase 5)
tests/          # pytest coverage
notebooks/      # exploration and real-data validation — never the final
                # deliverable, but the record of how each design decision
                # was checked before being trusted
docs/           # setup and reference docs
```

## How to Run

See `docs/SETUP.md` (full run instructions land once Phase 4's API is in
place).

## Limitations

Documented honestly as the project develops, not hidden:

- **Crosswalk gap:** 5.9% of O*NET occupations (60/1,016) have no direct BLS
  OEWS wage match — a mix of legislative, postsecondary-teacher subtypes,
  performing-arts, and lab-technician specialties, without one single cause.
- **Skill-survey gap:** 122 occupations (12%) have no O*NET skill-survey data
  at all (confirmed as a genuine O*NET data-collection lag for newer/renamed
  SOC titles, not a code bug) and are excluded from model training, though
  wage lookups still work for them where a BLS match exists.
- **Conformal calibration used melted (occupation × percentile) rows.** The
  calibration set's 865 rows represent only 173 independent occupations (5
  correlated rows each). The marginal coverage guarantee still holds, but the
  finite-sample correction's effective sample size is smaller than its row
  count suggests.
- **Job Zone 4 sits exactly at the Mondrian correction threshold** (85.0%
  coverage against an 85% trigger) — not corrected, since it didn't formally
  cross the bar, but not a clean pass either.
- **Coverage declines at higher wage percentiles** (92.5% → 86.7% from 10th
  to 90th) — a real, well-powered pattern not addressed by the current
  Job-Zone-only Mondrian check. Documented rather than corrected for now.
- **SOC-major-group fairness claims are inconclusive at this test-set size.**
  Only 2 of 22 SOC major groups have enough held-out occupations (n≥15) to
  trust their coverage numbers.
- **Skill-matching precision** is a cosine-similarity heuristic over 35
  O*NET elements, not a learned or validated ranking — plausible on manual
  spot-checks, not yet tested against real user judgments (planned for
  Phase 6's user testing).
- **Explanations can make a wrong prediction interpretable, not correct.**
  The Chemical Technicians spot-check (Phase 3) overpredicted the real BLS
  median wage by ~26%; SHAP shows Science and Critical Thinking driving the
  overprediction, which is useful for understanding *why* the model erred,
  but doesn't itself fix the error. Documented here rather than smoothed
  over, in the same spirit as the coverage and calibration gaps above.

## License

Code: MIT (see `LICENSE`).
Data: O\*NET is licensed CC BY 4.0 by USDOL/ETA; BLS OEWS data is U.S. government
public domain. Attribution is included in `data/raw/*/SOURCE.md`.

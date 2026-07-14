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

1. **Skill → Occupation Matching** — match user-input skills against O\*NET's
   full skill-importance taxonomy (Essential + Transferable Skills combined,
   35 elements covering both Basic Skills and Cross-Functional Skills) using
   cosine similarity. A visible warning fires if any requested skill isn't
   found in the matrix, so a partial match is never silent.
2. **Wage Range Output** — pull BLS OEWS percentiles (10th/25th/median/75th/90th)
   for matched occupation(s) via a SOC crosswalk (O\*NET-SOC → BLS SOC-2018).
3. **Model-Based Refinement** — a gradient-boosted regression (XGBoost) trained
   on melted per-occupation-percentile rows (skill profile + Job Zone +
   percentile → log-wage), with a monotonic constraint on the percentile
   feature so predicted wage never decreases as percentile increases. Trained
   with an occupation-level train/test split to prevent leakage. On real data,
   this beats a naive percentile-only baseline by **38.7% on dollar-MAE**
   (see `notebooks/03_baseline_model.ipynb`) — confirming skill profile and
   Job Zone add genuine predictive value, not just restating the naive
   baseline. This also lets the model estimate wages for occupations with a
   skill profile but no direct BLS wage match, which a pure lookup table
   cannot do.
4. **Conformal Prediction Layer** *(planned, Phase 2)* — wraps predictions in
   a calibrated interval; coverage validated and reported, including by
   subgroup.
5. **Explainability (SHAP)** *(planned, Phase 3)* — shows which input skills
   drove each prediction.

## Status

✅ Phase 0 complete — repo skeleton, O\*NET/BLS loaders, SOC crosswalk, skill
matching (validated on real data).
✅ Phase 1 complete — Job Zone loader, occupation feature table, melted
training dataset, baseline XGBoost wage model (validated: beats naive
baseline by 38.7% MAE, zero monotonicity violations on held-out data).
🚧 Next: Phase 2 — conformal prediction layer.

See `notebooks/` for the exploratory work behind each phase.

## Repo Structure

```
src/
├── data/      # O*NET + BLS ingestion and cleaning
├── features/  # skill-matching logic
├── models/    # training, conformal wrapper, SHAP integration
├── api/       # FastAPI endpoints
└── app/       # frontend dashboard
tests/         # pytest coverage
notebooks/     # exploration only — never the final deliverable
```

## How to Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

pytest  # 26 tests, all passing as of Phase 1

# Data must be downloaded first — see data/raw/onet/SOURCE.md and
# data/raw/oews/SOURCE.md for exact download links and placement.
```

See `docs/SETUP.md` for full setup details, and `notebooks/03_baseline_model.ipynb`
for the end-to-end pipeline (load → feature table → train → evaluate).

## Limitations

Documented honestly, as they were actually discovered building this — not
retrofitted for appearances:

- **Crosswalk coverage: 94.1%** (956/1,016 O\*NET occupations find a BLS wage
  match). The unmatched 60 don't share one obvious cause — a mix of
  legislative, some postsecondary-teacher subtypes, performing-arts, and a
  couple of lab-tech specialties.
- **~12% of occupations (122 of 1,016) have no O\*NET skill survey data at
  all**, in either the Essential or Transferable Skills files — confirmed to
  be a genuine data-collection lag (O\*NET updates skill profiles for a
  rolling subset of occupations each release, so newer/renamed SOC titles can
  lack survey data entirely), not a bug in this codebase. These occupations
  are excluded from model training but kept in the feature table for
  potential future handling.
- **Job Zone (experience-level proxy) is missing for a smaller subset** of
  occupations; where present alongside skill data, it's imputed with the
  dataset median — a simple, defensible choice given it's a coarse 1-5
  ordinal signal, but a real modeling choice, not a neutral one.
- **Skill-matching is small-subset sensitive**: cosine similarity computed
  over only the skills a user specifies (not the full 35-element space) can
  occasionally surface a directionally-matched but contextually odd
  occupation when only 1-2 skill dimensions actually overlap.
- **Wage estimates for previously-unmatched occupations vary more than
  expected across closely related titles** (e.g. different buyer/purchasing-
  agent variants), reflecting the limits of interpolating from a
  35-dimensional skill-similarity space rather than a flaw to hide.
- **National-level wages only** — BLS OEWS state/metro files exist but aren't
  used yet; real wages vary meaningfully by location, and the current range
  doesn't reflect that.
- **No individual-level wage data exists in either source dataset** — the
  model is trained on aggregate occupation-level percentiles (melted into
  per-percentile rows), not real individual salary observations. This is a
  deliberate, disclosed design choice, not an oversight.

## License

Code: MIT (see `LICENSE`).
Data: O\*NET is licensed CC BY 4.0 by USDOL/ETA; BLS OEWS data is U.S. government
public domain. Attribution is included in `data/raw/*/SOURCE.md`.

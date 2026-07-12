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

1. **Skill → Occupation Matching** — match user-input skills against O\*NET
   skill-importance profiles.
2. **Wage Range Output** — pull BLS OEWS percentiles for matched occupation(s).
3. **Model-Based Refinement** — gradient-boosted regression predicts finer
   position within the percentile band.
4. **Conformal Prediction Layer** — wraps predictions in a calibrated interval;
   coverage is validated and reported, including by subgroup.
5. **Explainability (SHAP)** — shows which input skills drove each prediction.

## Status

🚧 In development (Phase 0 — data acquisition and repo setup). See `notebooks/`
for exploratory work in progress.

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

See `docs/SETUP.md` (coming once Phase 1 lands).

## Limitations

To be documented honestly as the model develops — skill-matching precision,
occupation coverage gaps, and conformal coverage caveats will be reported here,
not hidden.

## License

Code: MIT (see `LICENSE`).
Data: O\*NET is licensed CC BY 4.0 by USDOL/ETA; BLS OEWS data is U.S. government
public domain. Attribution is included in `data/raw/*/SOURCE.md`.

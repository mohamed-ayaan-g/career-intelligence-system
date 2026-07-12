# BLS Occupational Employment and Wage Statistics (OEWS), May 2025

**Source:** U.S. Bureau of Labor Statistics
**License:** U.S. government work — public domain, no permission required.

## Download

National estimates (what we need for the MVP — wage percentiles by SOC code):
https://www.bls.gov/oes/special-requests/oesm25nat.zip

State-level (for later, if regional granularity is added):
https://www.bls.gov/oes/special-requests/oesm25st.zip

Metro/non-metro area (stretch goal — city-level estimates):
https://www.bls.gov/oes/special-requests/oesm25ma.zip

Technical notes / methodology: https://www.bls.gov/oes/current/oes_tec.htm

## Notes on the data

- ~830 occupations, SOC 2018 classification.
- Gives 10th/25th/median(50th)/75th/90th percentile wages + mean, plus
  employment counts — this is the ground-truth distribution the conformal
  layer calibrates against.
- Estimates are model-based using 3 years of survey panels (MB3 method) —
  worth a one-line mention in the README's limitations section, since it means
  the "percentile" isn't a single-year snapshot.
- Some cells are suppressed for small samples (<10 employment) — handle nulls
  explicitly in the loader, don't silently drop or zero-fill.

## Placement

```
data/raw/oews/oesm25nat.zip   # unzip to get the national XLSX table(s)
```

Gitignored — not committed (public data, easily re-downloaded, keeps repo lean).

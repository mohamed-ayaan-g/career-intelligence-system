"""
Loader for BLS OEWS (Occupational Employment and Wage Statistics) national data.

Expects the unzipped national XLSX table in data/raw/oews/ (see
data/raw/oews/SOURCE.md for download instructions).

Note: OEWS suppresses cells for occupations with small samples. These show as
'*' or '#' in the raw file — handle explicitly, don't silently coerce to NaN
and drop without logging how many rows were affected (this matters for the
README's honesty-about-limitations section).
"""

from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "oews" / "oesm25nat"

# Columns we actually need for the wage-range model; OEWS national files have
# many more (employment_prse, wage RSE, etc.) that aren't required for MVP.
WAGE_PERCENTILE_COLS = [
    "occ_code",
    "occ_title",
    "tot_emp",
    "a_pct10",
    "a_pct25",
    "a_median",
    "a_pct75",
    "a_pct90",
]


def load_national_wages(path: Path = RAW_DIR / "national_M2025_dl.xlsx") -> pd.DataFrame:
    """Load BLS OEWS national wage percentile data by SOC occupation code.

    Returns
    -------
    pd.DataFrame with occ_code, occ_title, employment, and annual wage
    percentiles (10th/25th/median/75th/90th).
    """
    df = pd.read_excel(path)
    df.columns = [c.strip().lower() for c in df.columns]

    missing = [c for c in WAGE_PERCENTILE_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Expected columns not found: {missing}. "
            "BLS occasionally renames columns between releases — check the "
            "actual header row against WAGE_PERCENTILE_COLS above."
        )

    df = df[WAGE_PERCENTILE_COLS].copy()

    # Suppressed/non-disclosable cells are marked '*' or '#' in OEWS files.
    suppressed_mask = df[["a_pct10", "a_pct25", "a_median", "a_pct75", "a_pct90"]].apply(
        lambda col: col.astype(str).str.contains(r"[*#]", na=False)
    )
    n_suppressed = suppressed_mask.any(axis=1).sum()
    if n_suppressed:
        print(f"[oews_loader] {n_suppressed} rows have suppressed wage data — kept as NaN, not dropped.")

    for col in ["a_pct10", "a_pct25", "a_median", "a_pct75", "a_pct90", "tot_emp"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


if __name__ == "__main__":
    wages = load_national_wages()
    print(f"Occupations with wage data: {wages['a_median'].notna().sum()} / {len(wages)}")

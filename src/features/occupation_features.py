"""
Build the occupation-level feature table used to train the baseline wage
model, and reshape it into the melted training format the model actually
learns from.

Why melted rows, not one row per occupation: neither O*NET nor OEWS provides
individual-level wage records (no "this specific person earned this specific
salary"). All we have is each occupation's 5 wage percentiles (10th/25th/
median/75th/90th) as aggregate ground truth. So each occupation's 5
percentiles become 5 separate training rows, each tagged with which
percentile it represents. The model then learns a general function: given an
occupation's skill profile + experience level + a target percentile, what
wage does that imply? This also lets the trained model estimate wages for
occupations with NO direct OEWS match at all (the ~60 crosswalk-unmatched
occupations) by interpolating from similar occupations' skill profiles —
something a pure lookup table could never do.
"""

import numpy as np
import pandas as pd

from src.data.soc_crosswalk import join_onet_to_oews

PERCENTILE_COLS = {
    "a_pct10": 10,
    "a_pct25": 25,
    "a_median": 50,
    "a_pct75": 75,
    "a_pct90": 90,
}


def build_occupation_feature_table(
    occ: pd.DataFrame,
    skill_matrix: pd.DataFrame,
    job_zones: pd.DataFrame,
    wages: pd.DataFrame,
) -> pd.DataFrame:
    """Combine skill profile + Job Zone + BLS wage percentiles into one
    occupation-level feature table.

    Parameters
    ----------
    occ : from onet_loader.load_occupation_data()
    skill_matrix : from onet_loader.build_full_skill_importance_matrix(),
        indexed by onetsoc_code
    job_zones : from onet_loader.load_job_zones()
    wages : from oews_loader.load_national_wages()

    Returns
    -------
    pd.DataFrame indexed by onetsoc_code with columns: title, job_zone,
    <skill columns>, tot_emp, a_pct10, a_pct25, a_median, a_pct75, a_pct90.
    One row per O*NET occupation (all 1,016, not just crosswalk-matched
    ones) — occupations without a BLS wage match keep NaN wage columns
    rather than being dropped, since they're valid inference targets for
    the trained model later, not just unusable rows.
    """
    base = occ[["onetsoc_code", "title"]].copy()

    base = base.merge(job_zones[["onetsoc_code", "job_zone"]], on="onetsoc_code", how="left")

    skills_reset = skill_matrix.reset_index()
    base = base.merge(skills_reset, on="onetsoc_code", how="left")

    wage_cols = ["onetsoc_code", "tot_emp"] + list(PERCENTILE_COLS.keys())
    joined_wages = join_onet_to_oews(occ, wages)[wage_cols]
    base = base.merge(joined_wages, on="onetsoc_code", how="left")

    return base.set_index("onetsoc_code")


def build_training_rows(feature_table: pd.DataFrame, skill_cols: list) -> pd.DataFrame:
    """Melt each occupation's 5 wage percentiles into 5 separate training
    rows, each tagged with its percentile level (10/25/50/75/90) as a
    feature, with log(wage) as the regression target.

    Only occupations with at least one non-null wage percentile contribute
    rows — occupations with no BLS wage match are excluded from TRAINING
    (there's nothing to learn from for them) but remain usable for
    INFERENCE later via the full feature_table.

    IMPORTANT for Step 6 (model training): when splitting train/test, split
    by onetsoc_code, not by row. Splitting by row lets the model see one
    percentile of an occupation in training and another in test, which
    leaks information and makes the model look better than it actually is.

    Parameters
    ----------
    feature_table : from build_occupation_feature_table(), indexed by
        onetsoc_code
    skill_cols : list of skill column names to carry through as features
        (typically skill_matrix.columns from build_full_skill_importance_matrix)

    Returns
    -------
    pd.DataFrame with columns: onetsoc_code, job_zone, <skill columns>,
    percentile, log_wage. One row per (occupation, percentile) pair that had
    a non-null wage value.
    """
    df = feature_table.reset_index()
    wage_cols = list(PERCENTILE_COLS.keys())
    has_any_wage = df[wage_cols].notna().any(axis=1)
    df = df[has_any_wage].copy()

    id_cols = ["onetsoc_code", "job_zone"] + list(skill_cols)

    rows = []
    for wage_col, percentile in PERCENTILE_COLS.items():
        sub = df[id_cols + [wage_col]].copy()
        sub = sub[sub[wage_col].notna()]
        sub["percentile"] = percentile
        sub["log_wage"] = np.log(sub[wage_col])
        sub = sub.drop(columns=[wage_col])
        rows.append(sub)

    return pd.concat(rows, ignore_index=True)


def clean_training_rows(training_rows: pd.DataFrame, skill_cols: list) -> pd.DataFrame:
    """Resolve nulls in melted training rows before model training.

    Confirmed against the real Phase 0/1 data (see notebooks/02_feature_table_check):
    two DISTINCT null sources exist, and they get different treatment:

    1. Occupations missing ALL skill values (no O*NET skill survey data at
       all — the same ~122-occupation gap identified in the Phase 0 EDA,
       confirmed to be all-or-nothing, not partial). These rows have zero
       signal to learn from and are DROPPED from training entirely. This is
       NOT the same as the 60 crosswalk-unmatched occupations from Phase 0 —
       these occupations DO have a wage value, they just lack skill data.
       They will need separate handling at inference time later (flagged as
       "cannot estimate" rather than silently predicting from missing
       features) — tracked as a Phase 2+ item, not solved here.

    2. Occupations with skill data but missing job_zone only (~69
       occupations, smaller and separate gap). job_zone is imputed with the
       dataset median. This is a simple, defensible choice given job_zone is
       a coarse 1-5 ordinal signal, but it IS a real modeling choice worth
       stating honestly in the README's limitations section, not treating as
       neutral.

    Returns
    -------
    pd.DataFrame with zero remaining nulls, ready for model training.
    Prints a summary of what was dropped/imputed and why.
    """
    n_before = len(training_rows)

    has_all_skills = training_rows[skill_cols].notna().all(axis=1)
    n_dropped = int((~has_all_skills).sum())
    cleaned = training_rows[has_all_skills].copy()

    n_missing_job_zone = int(cleaned["job_zone"].isna().sum())
    median_job_zone = None
    if n_missing_job_zone > 0:
        median_job_zone = cleaned["job_zone"].median()
        cleaned["job_zone"] = cleaned["job_zone"].fillna(median_job_zone)

    print(
        f"Dropped {n_dropped} rows ({n_dropped // 5} occupations) missing all "
        f"skill values, out of {n_before} total rows."
    )
    if n_missing_job_zone > 0:
        print(
            f"Imputed job_zone with dataset median ({median_job_zone}) for "
            f"{n_missing_job_zone} rows ({n_missing_job_zone // 5} occupations)."
        )
    else:
        print("No job_zone imputation needed.")

    remaining_nulls = int(cleaned.isna().sum().sum())
    assert remaining_nulls == 0, (
        f"{remaining_nulls} nulls remain after cleaning — an unexpected null "
        "source exists beyond the two handled here; investigate before training."
    )

    return cleaned

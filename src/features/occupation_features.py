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

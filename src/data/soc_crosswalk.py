"""
Crosswalk between O*NET-SOC codes and BLS SOC-2018 codes.

IMPORTANT — this is a real gotcha, not boilerplate:

O*NET uses O*NET-SOC codes, which are MORE granular than the BLS Standard
Occupational Classification (SOC). Example:
    O*NET-SOC: 15-1252.00  "Software Developers"
    BLS SOC:   15-1252     "Software Developers"

But some O*NET-SOC codes split a single SOC code into several detailed
occupations that BLS reports as one aggregate, e.g.:
    O*NET-SOC: 29-1141.01  "Acute Care Nurses"
    O*NET-SOC: 29-1141.02  "Advanced Practice Psychiatric Nurses"
    O*NET-SOC: 29-1141.03  "Critical Care Nurses"
    -> all roll up to BLS SOC: 29-1141  "Registered Nurses"

So the join key is: take the first 7 characters of the O*NET-SOC code
(the SOC-2018 portion) and drop the ".XX" detail suffix. When multiple
O*NET-SOC occupations map to one BLS SOC code, the wage percentiles are the
SAME for all of them (BLS doesn't distinguish) — but the skill profiles
differ. This is actually a feature, not a bug: it means the "match confidence"
can legitimately differ across sibling occupations sharing one wage range,
which is worth a line in the README's method section.
"""

import pandas as pd


def onet_soc_to_bls_soc(onet_soc_code: str) -> str:
    """Convert an O*NET-SOC code (e.g. '15-1252.00') to a BLS SOC code (e.g. '15-1252')."""
    return onet_soc_code.split(".")[0]


def add_bls_soc_column(df: pd.DataFrame, onet_soc_col: str = "onetsoc_code") -> pd.DataFrame:
    """Add a bls_soc_code column derived from an O*NET-SOC code column."""
    df = df.copy()
    df["bls_soc_code"] = df[onet_soc_col].apply(onet_soc_to_bls_soc)
    return df


def join_onet_to_oews(onet_df: pd.DataFrame, oews_df: pd.DataFrame,
                       onet_soc_col: str = "onetsoc_code",
                       oews_soc_col: str = "occ_code") -> pd.DataFrame:
    """Join an O*NET-derived dataframe to OEWS wage data via the SOC crosswalk.

    Many-to-one is expected and fine (see module docstring) — do NOT dedupe
    away the O*NET-SOC granularity, since that's what preserves per-occupation
    skill differentiation even where wage data is shared.
    """
    onet_df = add_bls_soc_column(onet_df, onet_soc_col)
    return onet_df.merge(
        oews_df, left_on="bls_soc_code", right_on=oews_soc_col, how="left"
    )

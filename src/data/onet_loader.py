"""
Loaders for the O*NET 30.3 Database (full flat-file text bundle).

Expects the bundle extracted at data/raw/onet/db_30_3_text/ with original
filenames intact (spaces included, e.g. "Occupation Data.txt") — this is
what you get from unzipping db_30_3_text.zip directly, as opposed to the
individual per-table Excel downloads. See data/raw/onet/SOURCE.md for
download instructions and attribution requirements.
"""

from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).resolve(
).parents[2] / "data" / "raw" / "onet" / "db_30_3_text"


def _read_onet_txt(filename: str) -> pd.DataFrame:
    path = RAW_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Confirm the flat-file bundle is unzipped to "
            f"{RAW_DIR} with original filenames intact (spaces included)."
        )
    return pd.read_csv(path, sep="\t", encoding="utf-8")


def _rename_or_raise(df: pd.DataFrame, rename_map: dict, source_name: str) -> pd.DataFrame:
    missing = [c for c in rename_map if c not in df.columns]
    if missing:
        raise ValueError(
            f"{source_name}: expected column(s) {missing} not found. "
            f"Actual columns are: {list(df.columns)}. "
            "O*NET occasionally tweaks header text between releases — "
            "update the rename_map in onet_loader.py to match."
        )
    return df.rename(columns=rename_map)


def load_occupation_data() -> pd.DataFrame:
    """Load Occupation Data.txt: O*NET-SOC code, title, description.

    Returns
    -------
    pd.DataFrame with columns: onetsoc_code, title, description
    """
    df = _read_onet_txt("Occupation Data.txt")
    return _rename_or_raise(
        df,
        {"O*NET-SOC Code": "onetsoc_code",
            "Title": "title", "Description": "description"},
        "Occupation Data.txt",
    )


def load_essential_skills() -> pd.DataFrame:
    """Load Essential Skills.txt: importance/level ratings per occupation.

    NOTE: as of O*NET 30.3, this covers only the "Basic Skills" half of
    O*NET's skills taxonomy (10 elements: Active Learning, Active Listening,
    Critical Thinking, Learning Strategies, Mathematics, Monitoring, Reading
    Comprehension, Science, Speaking, Writing). The other half — Cross-
    Functional Skills (Programming, Complex Problem Solving, Systems
    Analysis, Negotiation, Persuasion, Coordination, Management of Personnel
    Resources, etc.) — lives in Transferable Skills.txt (see
    load_transferable_skills below). Use build_full_skill_importance_matrix()
    to get the complete ~34-element skill space; using this loader alone
    silently restricts matching to just the 10 basic skills.

    Returns
    -------
    pd.DataFrame with columns: onetsoc_code, element_id, element_name,
    scale_id, data_value, ...
    """
    df = _read_onet_txt("Essential Skills.txt")
    return _rename_or_raise(
        df,
        {
            "O*NET-SOC Code": "onetsoc_code",
            "Element ID": "element_id",
            "Element Name": "element_name",
            "Scale ID": "scale_id",
            "Data Value": "data_value",
        },
        "Essential Skills.txt",
    )


def load_transferable_skills() -> pd.DataFrame:
    """Load Transferable Skills.txt: the Cross-Functional Skills half of
    O*NET's skills taxonomy (Programming, Complex Problem Solving, Systems
    Analysis, Negotiation, Persuasion, Coordination, Management of Personnel
    Resources, and similar — roughly 24 elements).

    Same schema as Essential Skills.txt. See load_essential_skills() docstring
    for why both files are needed together.

    Returns
    -------
    pd.DataFrame with columns: onetsoc_code, element_id, element_name,
    scale_id, data_value, ...
    """
    df = _read_onet_txt("Transferable Skills.txt")
    return _rename_or_raise(
        df,
        {
            "O*NET-SOC Code": "onetsoc_code",
            "Element ID": "element_id",
            "Element Name": "element_name",
            "Scale ID": "scale_id",
            "Data Value": "data_value",
        },
        "Transferable Skills.txt",
    )


def load_job_zones() -> pd.DataFrame:
    """Load Job Zones.txt: O*NET's 1-5 scale for the level of preparation
    (education, related experience, on-the-job training) an occupation
    typically requires. Used as our proxy for "experience level" in the
    wage model, since neither O*NET nor OEWS provides individual-level
    experience data directly.

    Job Zone scale (per O*NET documentation):
      1 = Little or no preparation needed
      2 = Some preparation needed
      3 = Medium preparation needed
      4 = Considerable preparation needed
      5 = Extensive preparation needed

    Returns
    -------
    pd.DataFrame with columns: onetsoc_code, job_zone (int, 1-5)
    """
    df = _read_onet_txt("Job Zones.txt")
    return _rename_or_raise(
        df,
        {"O*NET-SOC Code": "onetsoc_code", "Job Zone": "job_zone"},
        "Job Zones.txt",
    )


def build_skill_importance_matrix(essential_skills: pd.DataFrame) -> pd.DataFrame:
    """Pivot a single skills dataframe into an occupation x skill importance
    matrix. Kept for backward compatibility / single-file use — prefer
    build_full_skill_importance_matrix() for actual matching, since a single
    file (Essential Skills.txt alone) only covers half of O*NET's skill
    taxonomy (see load_essential_skills docstring).

    O*NET reports two scales per skill: Importance (IM) and Level (LV).
    For the matching step we primarily want Importance, since that reflects
    how central a skill is to the occupation regardless of the required
    proficiency level.

    Returns
    -------
    pd.DataFrame indexed by onetsoc_code, columns = skill element names,
    values = importance rating (1-5 scale per O*NET documentation).
    """
    im = essential_skills[essential_skills["scale_id"] == "IM"]
    matrix = im.pivot_table(
        index="onetsoc_code",
        columns="element_name",
        values="data_value",
        aggfunc="mean",
    )
    return matrix


def build_full_skill_importance_matrix(
    essential_skills: pd.DataFrame, transferable_skills: pd.DataFrame
) -> pd.DataFrame:
    """Pivot BOTH Essential Skills and Transferable Skills into one combined
    occupation x skill importance matrix covering the full ~34-element O*NET
    skills taxonomy (Basic Skills + Cross-Functional Skills together).

    This is the matrix that should actually be used for skill-to-occupation
    matching — using Essential Skills alone silently drops Programming,
    Complex Problem Solving, Negotiation, Persuasion, Coordination, and
    similar cross-functional skills from the match space entirely.

    Returns
    -------
    pd.DataFrame indexed by onetsoc_code, columns = skill element names
    (both Basic and Cross-Functional), values = importance rating.
    """
    combined = pd.concat(
        [essential_skills, transferable_skills], ignore_index=True)
    return build_skill_importance_matrix(combined)


if __name__ == "__main__":
    # Quick manual check once data is downloaded — not part of the test suite.
    occ = load_occupation_data()
    skills = load_essential_skills()
    print(f"Occupations: {len(occ)} rows")
    print(f"Essential skill ratings: {len(skills)} rows")
    print(occ.head())
    print(skills.head())

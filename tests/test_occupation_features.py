import numpy as np
import pandas as pd
import pytest

from src.features.occupation_features import build_occupation_feature_table, build_training_rows


@pytest.fixture
def occ():
    return pd.DataFrame(
        {
            "onetsoc_code": ["15-1252.00", "11-1011.00", "99-9999.00"],
            "title": ["Software Developers", "Chief Executives", "Legislators"],
        }
    )


@pytest.fixture
def skill_matrix():
    return pd.DataFrame(
        {"Programming": [5.0, 1.0, 0.5], "Mathematics": [4.5, 2.0, 1.0]},
        index=pd.Index(["15-1252.00", "11-1011.00", "99-9999.00"], name="onetsoc_code"),
    )


@pytest.fixture
def job_zones():
    return pd.DataFrame(
        {"onetsoc_code": ["15-1252.00", "11-1011.00", "99-9999.00"], "job_zone": [4, 5, 5]}
    )


@pytest.fixture
def wages():
    # Note: 99-9999.00 has NO wage row here, simulating a crosswalk-unmatched occupation
    return pd.DataFrame(
        {
            "occ_code": ["15-1252", "11-1011"],
            "a_pct10": [70000, 90000],
            "a_pct25": [85000, 120000],
            "a_median": [100000, 160000],
            "a_pct75": [120000, 210000],
            "a_pct90": [140000, 280000],
            "tot_emp": [1000000, 300000],
        }
    )


def test_feature_table_includes_all_occupations_even_without_wage_match(
    occ, skill_matrix, job_zones, wages
):
    """Occupations with no BLS wage match must NOT be dropped — they're
    valid inference targets for the trained model later."""
    table = build_occupation_feature_table(occ, skill_matrix, job_zones, wages)
    assert len(table) == 3
    assert "99-9999.00" in table.index
    assert table.loc["99-9999.00", "a_median"] is None or pd.isna(table.loc["99-9999.00", "a_median"])


def test_feature_table_preserves_skill_and_job_zone_values(occ, skill_matrix, job_zones, wages):
    table = build_occupation_feature_table(occ, skill_matrix, job_zones, wages)
    assert table.loc["15-1252.00", "Programming"] == 5.0
    assert table.loc["15-1252.00", "job_zone"] == 4


def test_training_rows_excludes_unmatched_occupation(occ, skill_matrix, job_zones, wages):
    table = build_occupation_feature_table(occ, skill_matrix, job_zones, wages)
    training = build_training_rows(table, skill_cols=["Programming", "Mathematics"])
    assert "99-9999.00" not in training["onetsoc_code"].values


def test_training_rows_has_five_rows_per_matched_occupation(occ, skill_matrix, job_zones, wages):
    table = build_occupation_feature_table(occ, skill_matrix, job_zones, wages)
    training = build_training_rows(table, skill_cols=["Programming", "Mathematics"])
    # 2 wage-matched occupations x 5 percentiles each = 10 rows
    assert len(training) == 10
    assert (training["onetsoc_code"].value_counts() == 5).all()


def test_training_rows_percentiles_are_correct_set(occ, skill_matrix, job_zones, wages):
    table = build_occupation_feature_table(occ, skill_matrix, job_zones, wages)
    training = build_training_rows(table, skill_cols=["Programming", "Mathematics"])
    assert set(training["percentile"].unique()) == {10, 25, 50, 75, 90}


def test_training_rows_log_wage_is_monotonic_with_percentile_per_occupation(
    occ, skill_matrix, job_zones, wages
):
    """Sanity check on the raw data itself: within one occupation, log_wage
    must increase strictly with percentile, since that's how OEWS reports
    percentiles in the first place (not something the model guarantees yet —
    this just confirms the melt didn't scramble the pairing)."""
    table = build_occupation_feature_table(occ, skill_matrix, job_zones, wages)
    training = build_training_rows(table, skill_cols=["Programming", "Mathematics"])
    for code in training["onetsoc_code"].unique():
        sub = training[training["onetsoc_code"] == code].sort_values("percentile")
        assert sub["log_wage"].is_monotonic_increasing


def test_log_wage_matches_manual_calculation(occ, skill_matrix, job_zones, wages):
    table = build_occupation_feature_table(occ, skill_matrix, job_zones, wages)
    training = build_training_rows(table, skill_cols=["Programming", "Mathematics"])
    row = training[(training["onetsoc_code"] == "15-1252.00") & (training["percentile"] == 50)]
    assert row["log_wage"].iloc[0] == pytest.approx(np.log(100000))

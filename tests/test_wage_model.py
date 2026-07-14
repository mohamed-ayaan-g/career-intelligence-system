import numpy as np
import pandas as pd
import pytest

from src.models.wage_model import (
    check_monotonicity,
    evaluate,
    naive_baseline_predict,
    occupation_train_test_split,
    train_baseline_model,
)

SKILL_COLS = ["skill_a", "skill_b"]
FEATURE_COLS = SKILL_COLS + ["job_zone", "percentile"]


@pytest.fixture
def synthetic_training_rows():
    """A small but structured synthetic dataset: wage genuinely depends on
    skills/job_zone/percentile, so a real model should clearly beat the
    naive baseline — this is what test_model_beats_naive_baseline checks."""
    np.random.seed(0)
    n_occ = 60
    occ_codes = [f"occ-{i:03d}" for i in range(n_occ)]
    skill_a = np.random.uniform(1, 5, n_occ)
    skill_b = np.random.uniform(1, 5, n_occ)
    job_zone = np.random.randint(1, 6, n_occ)
    base_log_wage = 10.5 + 0.15 * (skill_a + skill_b) + 0.1 * job_zone

    percentile_effect = {10: -0.5, 25: -0.2, 50: 0.0, 75: 0.2, 90: 0.45}
    rows = []
    for i, code in enumerate(occ_codes):
        for p, effect in percentile_effect.items():
            rows.append(
                {
                    "onetsoc_code": code,
                    "skill_a": skill_a[i],
                    "skill_b": skill_b[i],
                    "job_zone": job_zone[i],
                    "percentile": p,
                    "log_wage": base_log_wage[i] + effect + np.random.normal(0, 0.02),
                }
            )
    return pd.DataFrame(rows)


def test_split_has_no_occupation_overlap(synthetic_training_rows):
    train_df, test_df = occupation_train_test_split(synthetic_training_rows, test_size=0.25)
    train_codes = set(train_df["onetsoc_code"])
    test_codes = set(test_df["onetsoc_code"])
    assert train_codes.isdisjoint(test_codes)


def test_split_keeps_all_five_rows_per_occupation_together(synthetic_training_rows):
    train_df, test_df = occupation_train_test_split(synthetic_training_rows, test_size=0.25)
    assert (train_df["onetsoc_code"].value_counts() == 5).all()
    assert (test_df["onetsoc_code"].value_counts() == 5).all()


def test_train_baseline_model_requires_percentile_last(synthetic_training_rows):
    with pytest.raises(ValueError, match="must end with 'percentile'"):
        train_baseline_model(synthetic_training_rows, ["percentile"] + SKILL_COLS + ["job_zone"])


def test_model_beats_naive_baseline(synthetic_training_rows):
    train_df, test_df = occupation_train_test_split(synthetic_training_rows, test_size=0.25)
    model = train_baseline_model(train_df, FEATURE_COLS)
    result = evaluate(model, train_df, test_df, FEATURE_COLS)
    assert result.model_mae_dollars < result.naive_mae_dollars
    assert result.model_rmse_log < result.naive_rmse_log


def test_evaluate_reports_correct_test_counts(synthetic_training_rows):
    train_df, test_df = occupation_train_test_split(synthetic_training_rows, test_size=0.25)
    model = train_baseline_model(train_df, FEATURE_COLS)
    result = evaluate(model, train_df, test_df, FEATURE_COLS)
    assert result.n_test_rows == len(test_df)
    assert result.n_test_occupations == test_df["onetsoc_code"].nunique()


def test_naive_baseline_ignores_occupation_features(synthetic_training_rows):
    """The naive baseline should give the SAME prediction to two different
    occupations at the same percentile, since it ignores skill/job_zone
    entirely by design."""
    train_df, test_df = occupation_train_test_split(synthetic_training_rows, test_size=0.25)
    preds = naive_baseline_predict(train_df, test_df)
    same_percentile = test_df["percentile"] == test_df["percentile"].iloc[0]
    assert len(set(np.round(preds[same_percentile.values], 10))) == 1


def test_monotonicity_holds_across_all_occupations(synthetic_training_rows):
    train_df, test_df = occupation_train_test_split(synthetic_training_rows, test_size=0.25)
    model = train_baseline_model(train_df, FEATURE_COLS)
    for code in test_df["onetsoc_code"].unique():
        row = test_df[test_df["onetsoc_code"] == code].iloc[0]
        feature_values = {sc: row[sc] for sc in SKILL_COLS}
        feature_values["job_zone"] = row["job_zone"]
        assert check_monotonicity(model, feature_values, FEATURE_COLS)

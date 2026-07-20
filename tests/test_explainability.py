import numpy as np
import pandas as pd
import pytest
import shap

from src.models.wage_model import occupation_train_test_split, train_baseline_model
from src.models.explainability import (
    CONTEXT_FEATURES,
    build_explainer,
    explain_prediction,
    global_feature_importance,
)

SKILL_COLS = ["skill_a", "skill_b", "skill_c"]
FEATURE_COLS = SKILL_COLS + ["job_zone", "percentile"]


@pytest.fixture
def synthetic_training_rows():
    """Wage depends STRONGLY and POSITIVELY on skill_a, has no relationship
    to skill_c (pure noise feature), and depends on job_zone/percentile —
    structured so direction-sanity and skill-vs-context separation are both
    checkable against a known ground truth, not just "did it run"."""
    np.random.seed(0)
    n_occ = 80
    occ_codes = [f"occ-{i:03d}" for i in range(n_occ)]
    skill_a = np.random.uniform(1, 5, n_occ)
    skill_b = np.random.uniform(1, 5, n_occ)
    skill_c = np.random.uniform(1, 5, n_occ)  # noise -- should get near-zero SHAP on average
    job_zone = np.random.randint(1, 6, n_occ)
    base_log_wage = 10.0 + 0.4 * skill_a + 0.05 * skill_b + 0.1 * job_zone

    percentile_effect = {10: -0.5, 25: -0.2, 50: 0.0, 75: 0.2, 90: 0.45}
    rows = []
    for i, code in enumerate(occ_codes):
        for p, effect in percentile_effect.items():
            rows.append(
                {
                    "onetsoc_code": code,
                    "skill_a": skill_a[i],
                    "skill_b": skill_b[i],
                    "skill_c": skill_c[i],
                    "job_zone": job_zone[i],
                    "percentile": p,
                    "log_wage": base_log_wage[i] + effect + np.random.normal(0, 0.01),
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def trained_model_and_data(synthetic_training_rows):
    train_df, test_df = occupation_train_test_split(synthetic_training_rows, test_size=0.25, random_state=1)
    model = train_baseline_model(train_df, FEATURE_COLS)
    return model, train_df, test_df


def _row_dict(df, i):
    row = df.iloc[i]
    return {col: row[col] for col in FEATURE_COLS}


def test_additivity_holds_within_tolerance(trained_model_and_data):
    model, _, test_df = trained_model_and_data
    explainer = build_explainer(model)
    result = explain_prediction(model, explainer, _row_dict(test_df, 0), FEATURE_COLS, SKILL_COLS)
    assert result.additivity_ok
    assert result.additivity_max_abs_error < 1e-3


def test_context_and_skill_drivers_reconcile_to_full_prediction(trained_model_and_data):
    """Sum of ALL skill effects + ALL context effects + base value should
    equal the model's raw log-wage prediction -- confirming the presentation
    split (context vs. skill_drivers) never drops or double-counts any of
    the SHAP mass, it only reorganizes it."""
    model, _, test_df = trained_model_and_data
    explainer = build_explainer(model)
    result = explain_prediction(
        model, explainer, _row_dict(test_df, 3), FEATURE_COLS, SKILL_COLS, top_n=None
    )
    total = result.base_value_log
    total += sum(e.shap_log for e in result.skill_drivers)
    total += sum(e.shap_log for e in result.context.values())
    raw_log_pred = np.log(result.predicted_wage)
    assert abs(total - raw_log_pred) < 1e-3


def test_percentile_and_job_zone_are_context_not_skill_drivers(trained_model_and_data):
    model, _, test_df = trained_model_and_data
    explainer = build_explainer(model)
    result = explain_prediction(
        model, explainer, _row_dict(test_df, 0), FEATURE_COLS, SKILL_COLS, top_n=None
    )
    driver_names = {e.name for e in result.skill_drivers}
    assert driver_names == set(SKILL_COLS)
    assert set(result.context.keys()) == set(CONTEXT_FEATURES)
    assert driver_names.isdisjoint(result.context.keys())


def test_skill_direction_sanity_high_vs_low_skill_a(trained_model_and_data):
    """skill_a is the dominant positive driver by construction (coefficient
    0.4, vs 0.05 for skill_b). A row with skill_a near the top of its range
    should show a clearly positive SHAP effect; a row with skill_a near the
    bottom should show a clearly negative one."""
    model, train_df, _ = trained_model_and_data
    explainer = build_explainer(model)

    base_row = _row_dict(train_df, 0)
    high_row = dict(base_row, skill_a=4.8)
    low_row = dict(base_row, skill_a=1.2)

    high_result = explain_prediction(model, explainer, high_row, FEATURE_COLS, SKILL_COLS, top_n=None)
    low_result = explain_prediction(model, explainer, low_row, FEATURE_COLS, SKILL_COLS, top_n=None)

    high_effect = next(e for e in high_result.skill_drivers if e.name == "skill_a")
    low_effect = next(e for e in low_result.skill_drivers if e.name == "skill_a")

    assert high_effect.shap_log > 0
    assert low_effect.shap_log < 0
    assert high_effect.shap_log > low_effect.shap_log


def test_pct_effect_is_multiplicative_not_additive(trained_model_and_data):
    """pct_effect must equal exp(shap_log) - 1, not shap_log itself -- this
    is the log-space-vs-dollar-space distinction the module docstring
    warns about. A wrong implementation (pct_effect = shap_log) would still
    pass most other tests since both are monotonic in the same direction,
    so this checks the actual formula."""
    model, _, test_df = trained_model_and_data
    explainer = build_explainer(model)
    result = explain_prediction(model, explainer, _row_dict(test_df, 0), FEATURE_COLS, SKILL_COLS, top_n=None)
    for effect in result.skill_drivers + list(result.context.values()):
        # abs=1e-6 accounts for float32-vs-float64 rounding in SHAP's XGBoost
        # backend -- the formula itself is exact, this is precision noise.
        assert effect.pct_effect == pytest.approx(np.exp(effect.shap_log) - 1, abs=1e-6)


def test_top_n_sorts_by_absolute_shap_descending(trained_model_and_data):
    model, _, test_df = trained_model_and_data
    explainer = build_explainer(model)
    result = explain_prediction(model, explainer, _row_dict(test_df, 2), FEATURE_COLS, SKILL_COLS, top_n=None)
    abs_shaps = [abs(e.shap_log) for e in result.skill_drivers]
    assert abs_shaps == sorted(abs_shaps, reverse=True)


def test_top_n_limits_returned_driver_count(trained_model_and_data):
    model, _, test_df = trained_model_and_data
    explainer = build_explainer(model)
    result_top1 = explain_prediction(model, explainer, _row_dict(test_df, 0), FEATURE_COLS, SKILL_COLS, top_n=1)
    result_all = explain_prediction(model, explainer, _row_dict(test_df, 0), FEATURE_COLS, SKILL_COLS, top_n=None)
    assert len(result_top1.skill_drivers) == 1
    assert len(result_all.skill_drivers) == len(SKILL_COLS)
    # top_n=1 should keep the single largest-magnitude driver from the full list
    assert result_top1.skill_drivers[0].name == result_all.skill_drivers[0].name


def test_global_feature_importance_matches_manual_mean_abs_shap(trained_model_and_data):
    model, _, test_df = trained_model_and_data
    explainer = build_explainer(model)
    result = global_feature_importance(explainer, test_df, FEATURE_COLS, SKILL_COLS, top_n=None)

    manual_shap_values = explainer.shap_values(test_df[FEATURE_COLS], check_additivity=False)
    manual_mean_abs = dict(zip(FEATURE_COLS, np.abs(manual_shap_values).mean(axis=0)))

    for _, row in result.iterrows():
        assert row["mean_abs_shap_log"] == pytest.approx(manual_mean_abs[row["feature"]])


def test_global_feature_importance_categorizes_context_vs_skill(trained_model_and_data):
    model, _, test_df = trained_model_and_data
    explainer = build_explainer(model)
    result = global_feature_importance(explainer, test_df, FEATURE_COLS, SKILL_COLS, top_n=None)

    context_rows = result[result["feature"].isin(CONTEXT_FEATURES)]
    skill_rows = result[result["feature"].isin(SKILL_COLS)]
    assert (context_rows["category"] == "context").all()
    assert (skill_rows["category"] == "skill").all()
    assert len(context_rows) + len(skill_rows) == len(FEATURE_COLS)


def test_global_feature_importance_sorted_descending(trained_model_and_data):
    model, _, test_df = trained_model_and_data
    explainer = build_explainer(model)
    result = global_feature_importance(explainer, test_df, FEATURE_COLS, SKILL_COLS, top_n=None)
    values = result["mean_abs_shap_log"].tolist()
    assert values == sorted(values, reverse=True)


def test_global_feature_importance_respects_top_n(trained_model_and_data):
    model, _, test_df = trained_model_and_data
    explainer = build_explainer(model)
    result = global_feature_importance(explainer, test_df, FEATURE_COLS, SKILL_COLS, top_n=2)
    assert len(result) == 2


def test_build_explainer_is_reusable_across_calls(trained_model_and_data):
    """One explainer should serve both a local explanation and the global
    importance pass without rebuilding -- this is the whole point of
    separating build_explainer() out, not just an implementation detail."""
    model, _, test_df = trained_model_and_data
    explainer = build_explainer(model)
    local_result = explain_prediction(model, explainer, _row_dict(test_df, 0), FEATURE_COLS, SKILL_COLS)
    global_result = global_feature_importance(explainer, test_df, FEATURE_COLS, SKILL_COLS)
    assert local_result.additivity_ok
    assert len(global_result) > 0

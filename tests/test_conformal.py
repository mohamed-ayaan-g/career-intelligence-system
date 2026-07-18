"""
Tests for src/models/conformal.py — synthetic-data verification, following the
same pattern as test_wage_model.py: prove the math is correct on data where the
right answer is known, before trusting it on real O*NET/OEWS data (that real-data
check lives in notebooks/04_conformal_calibration.ipynb).
"""

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb

from src.models.wage_model import occupation_train_cal_test_split, occupation_train_test_split
from src.models.conformal import (
    calibrate,
    calibrate_by_group,
    predict_interval,
    predict_interval_by_group,
    evaluate_coverage,
    soc_major_group,
    _finite_sample_quantile,
)


def make_synthetic_training_rows(n_occupations=200, noise_scale=0.2, seed=0):
    """Occupations with a job_zone-like feature and log_wage that's a
    deterministic function of it plus known-scale Gaussian noise, melted
    into 5 percentile rows per occupation exactly like the real pipeline —
    so the model has something to learn and leaves a residual whose scale
    we know exactly, which is what the coverage tests below rely on.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_occupations):
        code = f"{10 + i // 50}-{1000 + i:04d}.00"
        job_zone = int(rng.integers(1, 6))
        base = 10.0 + 0.15 * job_zone
        for pct_idx, percentile in enumerate([10, 25, 50, 75, 90]):
            pct_offset = 0.1 * pct_idx
            noise = rng.normal(0, noise_scale)
            rows.append(
                {
                    "onetsoc_code": code,
                    "job_zone": job_zone,
                    "percentile": percentile,
                    "log_wage": base + pct_offset + noise,
                }
            )
    return pd.DataFrame(rows)


def fit_simple_model(train_df, feature_cols):
    model = xgb.XGBRegressor(n_estimators=50, max_depth=3, random_state=42)
    model.fit(train_df[feature_cols], train_df["log_wage"])
    return model


# ---------------------------------------------------------------------------
# 1. Quantile formula correctness
# ---------------------------------------------------------------------------


class TestQuantileFormula:
    def test_finite_sample_quantile_matches_hand_calculation(self):
        # n=9, alpha=0.1 -> level = ceil(10*0.9)/9 = ceil(9.0)/9 = 9/9 = 1.0
        # so q_hat should equal the max score exactly.
        scores = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=float)
        q_hat = _finite_sample_quantile(scores, alpha=0.10)
        assert q_hat == 9.0

    def test_finite_sample_quantile_level_above_raw_quantile(self):
        # n=19, alpha=0.1 -> level = ceil(20*0.9)/19 = ceil(18)/19 = 18/19 ~= 0.947,
        # strictly above the raw (1 - alpha) = 0.90 level -- confirms the
        # finite-sample correction actually inflates the quantile level.
        rng = np.random.default_rng(1)
        scores = rng.exponential(scale=1.0, size=19)
        q_hat_corrected = _finite_sample_quantile(scores, alpha=0.10)
        q_hat_raw = np.quantile(scores, 0.90)
        assert q_hat_corrected >= q_hat_raw

    def test_empty_calibration_set_raises(self):
        with pytest.raises(ValueError):
            _finite_sample_quantile(np.array([]), alpha=0.10)


# ---------------------------------------------------------------------------
# 2. Coverage guarantee on synthetic data (averaged over repeated trials)
# ---------------------------------------------------------------------------


class TestCoverageGuarantee:
    def test_marginal_coverage_near_target_across_repeated_trials(self):
        # Split conformal's guarantee is marginal (averaged over the
        # randomness of the calibration draw) and probabilistic -- a single
        # trial can land anywhere in a fairly wide band, so we average
        # empirical coverage across many independent trials and check the
        # AVERAGE lands close to 90%, rather than expecting any one trial to
        # hit it exactly.
        alpha = 0.10
        coverages = []
        for seed in range(30):
            data = make_synthetic_training_rows(n_occupations=150, noise_scale=0.2, seed=seed)
            feature_cols = ["job_zone", "percentile"]
            train_df, cal_df, test_df = occupation_train_cal_test_split(
                data, test_size=0.3, cal_size=0.3, random_state=seed
            )
            model = fit_simple_model(train_df, feature_cols)
            q_hat = calibrate(model, cal_df, feature_cols, alpha=alpha)
            result = evaluate_coverage(
                model, test_df, feature_cols, q_hat, target_coverage=1 - alpha
            )
            coverages.append(result.coverage)

        mean_coverage = np.mean(coverages)
        assert 0.85 <= mean_coverage <= 0.99, (
            f"Mean coverage {mean_coverage:.3f} across 30 trials is outside "
            f"the expected band around the 90% target."
        )


# ---------------------------------------------------------------------------
# 3. Interval validity
# ---------------------------------------------------------------------------


class TestIntervalValidity:
    def test_interval_bounds_are_ordered_and_contain_point_prediction(self):
        data = make_synthetic_training_rows(n_occupations=100, seed=2)
        feature_cols = ["job_zone", "percentile"]
        train_df, cal_df, test_df = occupation_train_cal_test_split(
            data, test_size=0.3, cal_size=0.3, random_state=2
        )
        model = fit_simple_model(train_df, feature_cols)
        q_hat = calibrate(model, cal_df, feature_cols, alpha=0.10)

        lo, hi = predict_interval(model, test_df, feature_cols, q_hat)
        point_pred_dollars = np.exp(model.predict(test_df[feature_cols]))

        assert np.all(lo < hi), "Interval lower bound must be strictly below upper bound."
        assert np.all(lo <= point_pred_dollars), "Point prediction must fall within interval (lo side)."
        assert np.all(point_pred_dollars <= hi), "Point prediction must fall within interval (hi side)."

    def test_dollar_space_interval_is_asymmetric_around_point_prediction(self):
        # Documents the expected (not buggy) consequence of exponentiating a
        # symmetric log-space interval: the high-side gap should exceed the
        # low-side gap whenever q_hat > 0.
        data = make_synthetic_training_rows(n_occupations=100, seed=3)
        feature_cols = ["job_zone", "percentile"]
        train_df, cal_df, test_df = occupation_train_cal_test_split(
            data, test_size=0.3, cal_size=0.3, random_state=3
        )
        model = fit_simple_model(train_df, feature_cols)
        q_hat = calibrate(model, cal_df, feature_cols, alpha=0.10)
        assert q_hat > 0

        lo, hi = predict_interval(model, test_df, feature_cols, q_hat)
        point_pred_dollars = np.exp(model.predict(test_df[feature_cols]))

        high_side_gap = hi - point_pred_dollars
        low_side_gap = point_pred_dollars - lo
        assert np.all(high_side_gap > low_side_gap)


# ---------------------------------------------------------------------------
# 4. Three-way split disjointness + test-set equivalence to the 2-way split
# ---------------------------------------------------------------------------


class TestThreeWaySplitDisjointness:
    def test_train_cal_test_are_pairwise_disjoint_by_occupation(self):
        data = make_synthetic_training_rows(n_occupations=200, seed=4)
        train_df, cal_df, test_df = occupation_train_cal_test_split(
            data, test_size=0.2, cal_size=0.25, random_state=4
        )
        train_codes = set(train_df["onetsoc_code"])
        cal_codes = set(cal_df["onetsoc_code"])
        test_codes = set(test_df["onetsoc_code"])

        assert train_codes.isdisjoint(cal_codes)
        assert train_codes.isdisjoint(test_codes)
        assert cal_codes.isdisjoint(test_codes)

        # Every occupation lands in exactly one split -- no drops, no dupes.
        all_codes = set(data["onetsoc_code"])
        assert train_codes | cal_codes | test_codes == all_codes

    def test_test_set_identical_to_two_way_split_with_same_params(self):
        # The three-way split is documented to carve calibration OUT OF train,
        # leaving test_df identical to what occupation_train_test_split would
        # produce with the same test_size/random_state -- this is what keeps
        # Phase 1 baseline numbers valid after adding the conformal layer.
        data = make_synthetic_training_rows(n_occupations=200, seed=5)
        _, _, test_df_3way = occupation_train_cal_test_split(
            data, test_size=0.2, cal_size=0.25, random_state=5
        )
        _, test_df_2way = occupation_train_test_split(data, test_size=0.2, random_state=5)

        assert set(test_df_3way["onetsoc_code"]) == set(test_df_2way["onetsoc_code"])


# ---------------------------------------------------------------------------
# 5. Mondrian (group-conditional) calibration
# ---------------------------------------------------------------------------


class TestMondrianCalibration:
    def test_group_specific_q_hats_reflect_different_residual_scales(self):
        # One job_zone's residuals are deliberately much noisier than the
        # other's; calibrate_by_group should give the noisier group a
        # strictly LARGER q_hat -- proving it uses group-specific residuals
        # rather than silently falling back to a shared/global calibration.
        rng = np.random.default_rng(6)
        rows = []
        for i in range(150):
            code = f"15-{2000 + i:04d}.00"
            job_zone = 2 if i % 2 == 0 else 4
            noise_scale = 0.05 if job_zone == 2 else 0.6
            base = 10.5
            for pct_idx, percentile in enumerate([10, 25, 50, 75, 90]):
                noise = rng.normal(0, noise_scale)
                rows.append(
                    {
                        "onetsoc_code": code,
                        "job_zone": job_zone,
                        "percentile": percentile,
                        "log_wage": base + 0.1 * pct_idx + noise,
                    }
                )
        data = pd.DataFrame(rows)
        feature_cols = ["job_zone", "percentile"]
        train_df, cal_df, test_df = occupation_train_cal_test_split(
            data, test_size=0.2, cal_size=0.3, random_state=6
        )
        model = fit_simple_model(train_df, feature_cols)

        q_hats = calibrate_by_group(model, cal_df, feature_cols, group_col="job_zone", alpha=0.10)

        assert set(q_hats.keys()) == {2, 4}
        assert q_hats[4] > q_hats[2], (
            "The noisier job_zone (4) should get a strictly larger calibrated "
            "q_hat than the cleaner one (2) -- otherwise group-specific "
            "residuals aren't actually being used."
        )

    def test_predict_interval_by_group_uses_correct_group_q_hat(self):
        q_hats = {2: 0.1, 4: 0.5}

        class ConstantModel:
            # Predicts a constant log-wage regardless of input, so any
            # difference in interval width is attributable ONLY to q_hat
            # lookup -- isolating exactly what this test needs to check.
            def predict(self, X):
                return np.full(len(X), 10.0)

        x = pd.DataFrame({"job_zone": [2, 4], "percentile": [50, 50]})
        lo, hi = predict_interval_by_group(
            ConstantModel(), x, ["job_zone", "percentile"], q_hats, "job_zone"
        )

        width = hi - lo
        assert width[1] > width[0]

    def test_predict_interval_by_group_raises_for_unseen_group(self):
        q_hats = {2: 0.1}

        class ConstantModel:
            def predict(self, X):
                return np.full(len(X), 10.0)

        x = pd.DataFrame({"job_zone": [4], "percentile": [50]})
        with pytest.raises(KeyError):
            predict_interval_by_group(ConstantModel(), x, ["job_zone", "percentile"], q_hats, "job_zone")


# ---------------------------------------------------------------------------
# 6. SOC major group extraction
# ---------------------------------------------------------------------------


class TestSocMajorGroup:
    def test_extracts_two_digit_prefix(self):
        assert soc_major_group("15-1251.00") == "15"
        assert soc_major_group("11-1011.03") == "11"

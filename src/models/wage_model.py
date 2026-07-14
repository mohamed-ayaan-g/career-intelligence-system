"""
Baseline wage prediction model.

Trains a gradient-boosted regressor on the melted training rows (skill
profile + job_zone + percentile -> log_wage). See occupation_features.py
for why this melted-row design exists (no individual-level wage data).

Key design choices:

- Train/test split is done by OCCUPATION, not by row. Splitting by row
  would let the model see one percentile of an occupation in training and
  a different percentile of the SAME occupation in test — that's leakage,
  and makes the model look better than it actually is.

- A monotonic constraint is applied to the `percentile` feature
  specifically, forcing predicted wage to never decrease as percentile
  increases. This structurally guarantees a sane ordering (10th <= 25th
  <= ... <= 90th for the same occupation) rather than relying on the model
  to learn it from data alone — pooling percentiles into one model doesn't
  otherwise guarantee this by construction.

- The naive baseline predicts log_wage using ONLY the training set's mean
  log_wage for that percentile level, ignoring occupation features
  entirely. This tests whether skill profile + job_zone actually add
  predictive value beyond "what percentile is this" — a real question
  worth answering explicitly, not assuming.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split


@dataclass
class EvalResult:
    model_rmse_log: float
    naive_rmse_log: float
    model_mae_dollars: float
    naive_mae_dollars: float
    n_test_rows: int
    n_test_occupations: int

    def summary(self) -> str:
        improvement = 100 * (1 - self.model_mae_dollars / self.naive_mae_dollars)
        return (
            f"Model RMSE (log-wage):  {self.model_rmse_log:.4f}\n"
            f"Naive RMSE (log-wage):  {self.naive_rmse_log:.4f}\n"
            f"Model MAE (dollars):    ${self.model_mae_dollars:,.0f}\n"
            f"Naive MAE (dollars):    ${self.naive_mae_dollars:,.0f}\n"
            f"Model improvement over naive: {improvement:.1f}%\n"
            f"Test set: {self.n_test_rows} rows across {self.n_test_occupations} held-out occupations"
        )


def occupation_train_test_split(
    training_rows: pd.DataFrame, test_size: float = 0.2, random_state: int = 42
):
    """Split melted training rows by OCCUPATION, not by row, to prevent
    leakage (see module docstring). All 5 percentile rows for a given
    occupation end up entirely in train or entirely in test.

    Returns
    -------
    (train_df, test_df) — both subsets of training_rows with all original
    columns intact.
    """
    unique_codes = training_rows["onetsoc_code"].unique()
    train_codes, test_codes = train_test_split(
        unique_codes, test_size=test_size, random_state=random_state
    )
    train_df = training_rows[training_rows["onetsoc_code"].isin(train_codes)].copy()
    test_df = training_rows[training_rows["onetsoc_code"].isin(test_codes)].copy()
    return train_df, test_df


def train_baseline_model(train_df: pd.DataFrame, feature_cols: list) -> xgb.XGBRegressor:
    """Train the baseline gradient-boosted wage model.

    Parameters
    ----------
    train_df : the train half from occupation_train_test_split()
    feature_cols : column names to use as features. MUST end with
        "percentile" as the last entry — the monotonic constraint below is
        applied positionally to the last feature.

    Returns
    -------
    Fitted XGBRegressor, trained to predict log_wage.
    """
    if feature_cols[-1] != "percentile":
        raise ValueError(
            "feature_cols must end with 'percentile' so the monotonic "
            "constraint (applied to the last feature) lines up correctly."
        )

    monotone_constraints = tuple([0] * (len(feature_cols) - 1) + [1])

    model = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        monotone_constraints=monotone_constraints,
        random_state=42,
    )
    model.fit(train_df[feature_cols], train_df["log_wage"])
    return model


def naive_baseline_predict(train_df: pd.DataFrame, test_df: pd.DataFrame) -> np.ndarray:
    """Predict log_wage using ONLY the training set's mean log_wage per
    percentile level, ignoring occupation features entirely. This is the
    bar the real model needs to clear to prove skill profile + job_zone
    add genuine predictive value beyond "which percentile is this".
    """
    percentile_means = train_df.groupby("percentile")["log_wage"].mean()
    return test_df["percentile"].map(percentile_means).values


def evaluate(
    model: xgb.XGBRegressor, train_df: pd.DataFrame, test_df: pd.DataFrame, feature_cols: list
) -> EvalResult:
    """Evaluate the trained model against held-out occupations, and against
    the naive percentile-only baseline, in both log-space (RMSE, what the
    model actually optimizes) and real dollar terms (MAE, what's actually
    interpretable to a person reading the result).
    """
    y_true_log = test_df["log_wage"].values
    y_pred_log = model.predict(test_df[feature_cols])
    naive_pred_log = naive_baseline_predict(train_df, test_df)

    model_rmse_log = float(np.sqrt(mean_squared_error(y_true_log, y_pred_log)))
    naive_rmse_log = float(np.sqrt(mean_squared_error(y_true_log, naive_pred_log)))

    y_true_dollars = np.exp(y_true_log)
    model_mae_dollars = float(mean_absolute_error(y_true_dollars, np.exp(y_pred_log)))
    naive_mae_dollars = float(mean_absolute_error(y_true_dollars, np.exp(naive_pred_log)))

    return EvalResult(
        model_rmse_log=model_rmse_log,
        naive_rmse_log=naive_rmse_log,
        model_mae_dollars=model_mae_dollars,
        naive_mae_dollars=naive_mae_dollars,
        n_test_rows=len(test_df),
        n_test_occupations=test_df["onetsoc_code"].nunique(),
    )


def check_monotonicity(model: xgb.XGBRegressor, feature_values: dict, feature_cols: list) -> bool:
    """Sanity check: for one occupation's feature values, predicted wage
    must never decrease as percentile increases (10 -> 25 -> 50 -> 75 -> 90).

    This SHOULD always hold given the monotonic constraint applied at
    training time — but is worth asserting explicitly on real predictions
    rather than trusting the constraint blindly, since constraints can
    behave unexpectedly at the edges of the training data's feature space.

    Parameters
    ----------
    feature_values : dict of feature name -> value, EXCLUDING "percentile"
        (e.g. skill values + job_zone for one occupation)
    feature_cols : full feature column list, ending in "percentile"

    Returns
    -------
    True if predictions are non-decreasing across all 5 percentile levels.
    """
    percentiles = [10, 25, 50, 75, 90]
    rows = pd.DataFrame([feature_values] * len(percentiles))
    rows["percentile"] = percentiles
    preds = model.predict(rows[feature_cols])
    return bool(np.all(np.diff(preds) >= -1e-6))

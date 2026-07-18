"""
Conformal prediction layer for the baseline wage model (Phase 2).

Wraps the trained XGBoost model's point predictions in a calibrated prediction
interval using split conformal regression, then validates coverage — both
marginally and conditionally by subgroup (Job Zone, SOC major group,
percentile level) — to check whether the model is overconfident for any
particular population. This is where the project's "honest uncertainty for
people without industry access" thesis becomes technically concrete: a
calibrated 90% interval that actually contains the true wage ~90% of the time,
checked directly against real held-out data rather than assumed.

Key design choices:

- The nonconformity score is the ABSOLUTE RESIDUAL IN LOG-WAGE SPACE, not
  dollars. Log space matches what the model was trained on, keeps residuals
  roughly homoscedastic across the wage distribution (dollar-space residuals
  blow up for high earners, making a single global quantile meaningless), and
  is the space the monotonicity constraint already operates in.

- The calibration quantile uses the FINITE-SAMPLE CORRECTED level
  ceil((n+1)(1-alpha)) / n, not the raw (1-alpha) quantile. This correction is
  what makes the marginal coverage guarantee (>= 1-alpha) hold in the
  split-conformal sense — skipping it would silently produce an interval that
  slightly undercovers.

- Exponentiating a symmetric log-space interval produces an ASYMMETRIC
  dollar-space interval (wider on the high side, since exp is convex). This is
  a real, correct consequence of calibrating in log space — not a bug — and
  is worth stating plainly wherever intervals are reported in dollars, since a
  reader would otherwise reasonably expect a symmetric range.

- Group-conditional (Mondrian) calibration — calibrate_by_group /
  predict_interval_by_group — is a separate path, not the default. It exists
  to respond to a documented finding from evaluate_conditional_coverage() (a
  subgroup, e.g. an early-career Job Zone, being meaningfully undercovered),
  following the same validate-the-simple-version-before-adding-complexity
  pattern used throughout this project (essential-vs-transferable skills,
  weighted-overlap-vs-cosine matching).
"""

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd
import xgboost as xgb


@dataclass
class CoverageResult:
    group_value: object
    n_occupations: int
    n_rows: int
    coverage: float
    mean_width_dollars: float
    target_coverage: float

    def summary(self) -> str:
        flag = " (n<15, LOW CONFIDENCE)" if self.n_occupations < 15 else ""
        return (
            f"{self.group_value!r}: coverage={self.coverage:.1%} "
            f"(target {self.target_coverage:.0%}), "
            f"mean width=${self.mean_width_dollars:,.0f}, "
            f"n_occ={self.n_occupations}{flag}"
        )


def soc_major_group(onetsoc_code: str) -> str:
    """Extract the 2-digit SOC major group prefix from an O*NET-SOC code,
    e.g. '15-1251.00' -> '15'. Intended use is adding it as a column before
    calling evaluate_conditional_coverage(), e.g.:

        test_df['soc_major_group'] = test_df['onetsoc_code'].map(soc_major_group)
    """
    return onetsoc_code.split("-")[0]


def _nonconformity_scores(model: xgb.XGBRegressor, df: pd.DataFrame, feature_cols: list) -> np.ndarray:
    """Absolute residual in log-wage space: |true_log_wage - predicted_log_wage|."""
    preds = model.predict(df[feature_cols])
    return np.abs(df["log_wage"].values - preds)


def _finite_sample_quantile(scores: np.ndarray, alpha: float) -> float:
    """The split-conformal finite-sample-corrected quantile.

    Uses level ceil((n+1)(1-alpha)) / n, clipped to at most 1.0. When n is too
    small for the requested alpha, this returns the maximum observed score
    (the most conservative valid choice) rather than erroring — but a caller
    relying on a very small calibration set should treat that q_hat with
    appropriate suspicion (see calibrate_by_group's docstring).
    """
    n = len(scores)
    if n == 0:
        raise ValueError("Cannot calibrate on an empty calibration set.")
    level = math.ceil((n + 1) * (1 - alpha)) / n
    level = min(level, 1.0)
    return float(np.quantile(scores, level, method="higher"))


def calibrate(
    model: xgb.XGBRegressor, cal_df: pd.DataFrame, feature_cols: list, alpha: float = 0.10
) -> float:
    """Split conformal calibration: compute a single global q_hat such that
    [pred - q_hat, pred + q_hat] in log-wage space achieves marginal coverage
    >= 1 - alpha on exchangeable held-out data (Vovk et al.'s split conformal
    guarantee).

    Parameters
    ----------
    model : the trained XGBRegressor (from train_baseline_model)
    cal_df : calibration split from occupation_train_cal_test_split — must NOT
        overlap with the data the model was trained on
    feature_cols : same feature_cols used at training time
    alpha : miscoverage rate (0.10 -> target 90% coverage)

    Returns
    -------
    q_hat : the calibrated half-width, in LOG-WAGE space
    """
    scores = _nonconformity_scores(model, cal_df, feature_cols)
    return _finite_sample_quantile(scores, alpha)


def calibrate_by_group(
    model: xgb.XGBRegressor,
    cal_df: pd.DataFrame,
    feature_cols: list,
    group_col: str,
    alpha: float = 0.10,
) -> dict:
    """Mondrian (group-conditional) calibration: a separate q_hat per group
    value, computed only from that group's own calibration residuals.

    Use this ONLY after evaluate_conditional_coverage() on the global q_hat
    shows real miscoverage by group_col — it's a response to a documented
    finding, not a default choice.

    Splitting calibration by group shrinks the effective n per group, which
    widens the finite-sample-correction penalty for smaller groups — a real
    cost of the fairness fix, not something to gloss over. Concretely: a
    small group will tend to get a LARGER q_hat (wider interval) than the
    global version would have given it, all else equal, precisely because
    there's less calibration data to pin the quantile down tightly.

    Returns
    -------
    dict mapping group value -> q_hat (log-wage space). Check group sizes in
    cal_df before trusting a small group's q_hat — this function does not
    suppress or flag small groups itself; evaluate_conditional_coverage()
    does that flagging when reporting results.
    """
    q_hats = {}
    for group_value, group_df in cal_df.groupby(group_col):
        scores = _nonconformity_scores(model, group_df, feature_cols)
        q_hats[group_value] = _finite_sample_quantile(scores, alpha)
    return q_hats


def predict_interval(
    model: xgb.XGBRegressor, x: pd.DataFrame, feature_cols: list, q_hat: float
):
    """Build a dollar-space prediction interval from a single global q_hat.

    NOTE: the interval is symmetric in LOG-WAGE space (+/- q_hat around the
    point prediction) but becomes ASYMMETRIC once exponentiated into dollars —
    wider on the high side. This is correct, expected behavior of calibrating
    in log space, not a bug.

    Parameters
    ----------
    x : DataFrame of feature values (one or more rows)
    q_hat : from calibrate()

    Returns
    -------
    (lo, hi) : dollar-space bounds (numpy arrays, one value per row of x)
    """
    log_pred = model.predict(x[feature_cols])
    lo = np.exp(log_pred - q_hat)
    hi = np.exp(log_pred + q_hat)
    return lo, hi


def predict_interval_by_group(
    model: xgb.XGBRegressor,
    x: pd.DataFrame,
    feature_cols: list,
    q_hats: dict,
    group_col: str,
):
    """Same as predict_interval, but looks up a group-specific q_hat per row
    based on x[group_col], for use with calibrate_by_group's output.

    Raises KeyError if a row's group value wasn't present in the calibration
    set, rather than silently falling back to some default — silently
    defaulting would hide exactly the "no calibration data for this
    population" case that matters most for the fairness thesis.
    """
    log_pred = model.predict(x[feature_cols])
    group_values = x[group_col].values
    q_hat_arr = np.array([q_hats[g] for g in group_values])
    lo = np.exp(log_pred - q_hat_arr)
    hi = np.exp(log_pred + q_hat_arr)
    return lo, hi


def evaluate_coverage(
    model: xgb.XGBRegressor,
    test_df: pd.DataFrame,
    feature_cols: list,
    q_hat: float,
    target_coverage: float = 0.90,
) -> CoverageResult:
    """Marginal coverage check on held-out test data: does the interval
    actually contain the true wage ~target_coverage of the time?

    Reports coverage AND mean interval width in dollars together, always —
    coverage alone is a cheap trick to game (a maximally wide interval always
    "covers"), so width is what tells you whether the interval is actually
    useful to someone reading it.
    """
    log_pred = model.predict(test_df[feature_cols])
    true_log = test_df["log_wage"].values
    lo = log_pred - q_hat
    hi = log_pred + q_hat
    covered = (true_log >= lo) & (true_log <= hi)

    width_dollars = np.exp(hi) - np.exp(lo)

    return CoverageResult(
        group_value="ALL",
        n_occupations=test_df["onetsoc_code"].nunique(),
        n_rows=len(test_df),
        coverage=float(covered.mean()),
        mean_width_dollars=float(width_dollars.mean()),
        target_coverage=target_coverage,
    )


def evaluate_conditional_coverage(
    model: xgb.XGBRegressor,
    test_df: pd.DataFrame,
    feature_cols: list,
    q_hat,
    group_col: str,
    target_coverage: float = 0.90,
    min_n_occupations: int = 15,
) -> pd.DataFrame:
    """Conditional coverage by subgroup (group_col e.g. 'job_zone', a derived
    SOC-major-group column via soc_major_group(), or 'percentile'). Accepts
    either a single global q_hat (float, from calibrate()) or a dict of
    per-group q_hats (from calibrate_by_group()) — pass whichever you're
    evaluating.

    Groups with fewer than min_n_occupations held-out occupations are still
    reported (nothing is hidden), but CoverageResult.summary() flags them —
    a coverage number computed from a handful of occupations is noisy enough
    to be close to meaningless, and presenting it with the same apparent
    authority as a well-populated group would be misleading.

    Returns
    -------
    DataFrame, one row per group value, sorted by group_value, with columns
    matching CoverageResult's fields (plus an is_low_confidence flag).
    """
    rows = []
    for group_value, group_df in test_df.groupby(group_col):
        this_q_hat = q_hat[group_value] if isinstance(q_hat, dict) else q_hat

        log_pred = model.predict(group_df[feature_cols])
        true_log = group_df["log_wage"].values
        lo = log_pred - this_q_hat
        hi = log_pred + this_q_hat
        covered = (true_log >= lo) & (true_log <= hi)
        width_dollars = np.exp(hi) - np.exp(lo)

        n_occ = group_df["onetsoc_code"].nunique()
        rows.append(
            {
                "group_value": group_value,
                "n_occupations": n_occ,
                "n_rows": len(group_df),
                "coverage": float(covered.mean()),
                "mean_width_dollars": float(width_dollars.mean()),
                "target_coverage": target_coverage,
                "is_low_confidence": n_occ < min_n_occupations,
            }
        )

    result_df = pd.DataFrame(rows)
    return result_df.sort_values("group_value").reset_index(drop=True)

"""
SHAP-based explainability layer for the baseline wage model.

See Phase 3 design notes (notebooks/05_shap_explainability.ipynb) for the
real-data verification this module was built against. Two things worth
knowing before reading the code:

1. DOLLAR EFFECTS ARE MULTIPLICATIVE, NOT ADDITIVE.
   The model predicts log(wage), so SHAP values are additive in LOG-wage
   space: sum(shap_i) + base_value == model's raw log-wage prediction.
   That means each feature's effect in DOLLAR space is multiplicative:
   exp(shap_i) is the factor that feature applied to the base wage.
   We report this as `pct_effect = exp(shap_i) - 1` (e.g. 0.12 == "+12%").
   Treating a SHAP value as a dollar amount directly would be wrong — it
   would silently misstate every effect except ones very close to zero.

2. `percentile` AND `job_zone` ARE CONTEXT, NOT SKILL DRIVERS.
   `percentile` is an artifact of this project's melted-row training design
   (see occupation_features.py) — it describes WHICH POINT in the wage
   distribution a row represents, not something the user reported about
   themselves. `job_zone` is a structural property of the occupation (how
   much preparation it typically requires), not a self-reported skill
   either. SHAP is computed over ALL features, including these two — we
   are not hiding their effect or explaining a different function than the
   one the model actually learned. They are only separated at the
   PRESENTATION layer, into `PredictionExplanation.context`, so a lay user
   isn't told "the 90th percentile" is a skill that drove their estimate.
   The full explanation (context + skill_drivers) still reconciles to the
   real prediction — see `additivity_max_abs_error`.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import shap

# Features that describe distributional/structural context rather than a
# self-reported skill. Kept as a tuple (not inferred) so the split is an
# explicit, reviewable design decision — not something that silently shifts
# if a feature gets renamed upstream.
CONTEXT_FEATURES = ("percentile", "job_zone")


@dataclass
class SkillEffect:
    """One feature's contribution to a single prediction."""

    name: str
    shap_log: float
    pct_effect: float  # exp(shap_log) - 1 -- e.g. 0.12 means "+12% vs base_wage"


@dataclass
class PredictionExplanation:
    """Local (per-row) SHAP explanation for one occupation/percentile prediction."""

    predicted_wage: float
    base_wage: float  # exp(base_value_log) -- the model's "average" prediction before any feature effects
    base_value_log: float
    skill_drivers: list  # list[SkillEffect], sorted by |shap_log| descending
    context: dict  # {"percentile": SkillEffect, "job_zone": SkillEffect}
    additivity_max_abs_error: float  # |sum(all shap) + base_value - raw_margin_prediction|, log-wage space
    additivity_ok: bool  # additivity_max_abs_error < ADDITIVITY_TOLERANCE


ADDITIVITY_TOLERANCE = 1e-3  # log-wage space; ~0.1% dollar-space effect, well below anything user-visible


def build_explainer(model) -> shap.TreeExplainer:
    """Build one TreeExplainer for reuse across many calls.

    Building a TreeExplainer parses the full ensemble structure, so it's
    wasteful to rebuild it per-prediction if explaining many rows (e.g. a
    batch of spot-check occupations). Callers should build once and pass
    the same explainer to explain_prediction() / global_feature_importance().
    """
    return shap.TreeExplainer(model)


def _compute_shap(explainer: shap.TreeExplainer, X: pd.DataFrame, feature_cols: list):
    """Run the explainer once. Returns (shap_values ndarray, base_value float),
    both in log-wage space (the model's native output space).

    check_additivity=False here is deliberate: we always compute our OWN
    additivity check afterward (see additivity_max_abs_error) and surface it
    as data on the result, rather than letting SHAP raise partway through a
    batch. This matches the "raise loudly, but as a reportable check" pattern
    used elsewhere in this project (see clean_training_rows's assertion) --
    except here the caller may reasonably want the explanation even when a
    single edge-case row has a slightly elevated error, so we report rather
    than hard-fail.
    """
    shap_values = explainer.shap_values(X[feature_cols], check_additivity=False)
    base_value = float(explainer.expected_value)
    return shap_values, base_value


def explain_prediction(
    model,
    explainer: shap.TreeExplainer,
    feature_values: dict,
    feature_cols: list,
    skill_cols: list,
    top_n: Optional[int] = 5,
) -> PredictionExplanation:
    """Explain a single occupation/percentile prediction.

    Parameters
    ----------
    model : the trained xgb.XGBRegressor (needed to get the raw log-wage
        margin prediction for the additivity check -- NOTE: `explainer.model`
        is SHAP's internal TreeEnsemble wrapper, not the real model, and does
        not behave like it; the original model must be passed separately).
    explainer : built via build_explainer(model)
    feature_values : dict of feature_name -> value for ONE row. Must contain
        every entry in feature_cols (skills + job_zone + percentile).
    feature_cols : full feature column list (as used to train the model)
    skill_cols : the O*NET skill element columns only (excludes percentile,
        job_zone) -- used to build skill_drivers, separate from context
    top_n : how many top skill drivers to keep, sorted by |shap_log|
        descending. None returns all skill drivers.
    """
    row = pd.DataFrame([feature_values])[feature_cols]
    shap_values, base_value = _compute_shap(explainer, row, feature_cols)
    shap_row = shap_values[0]

    raw_margin = float(model.predict(row[feature_cols], output_margin=True)[0])
    reconstructed = float(shap_row.sum() + base_value)
    additivity_max_abs_error = abs(reconstructed - raw_margin)

    effects = {
        name: SkillEffect(name=name, shap_log=float(val), pct_effect=float(np.exp(val) - 1))
        for name, val in zip(feature_cols, shap_row)
    }

    context = {name: effects[name] for name in CONTEXT_FEATURES if name in effects}
    skill_effects = [effects[name] for name in skill_cols if name in effects]
    skill_effects.sort(key=lambda e: abs(e.shap_log), reverse=True)
    if top_n is not None:
        skill_effects = skill_effects[:top_n]

    return PredictionExplanation(
        predicted_wage=float(np.exp(raw_margin)),
        base_wage=float(np.exp(base_value)),
        base_value_log=base_value,
        skill_drivers=skill_effects,
        context=context,
        additivity_max_abs_error=additivity_max_abs_error,
        additivity_ok=additivity_max_abs_error < ADDITIVITY_TOLERANCE,
    )


def global_feature_importance(
    explainer: shap.TreeExplainer,
    X: pd.DataFrame,
    feature_cols: list,
    skill_cols: list,
    top_n: Optional[int] = 15,
) -> pd.DataFrame:
    """Mean(|SHAP|) across a dataset (typically the test set), split into
    skill vs. context categories so the aggregate view carries the same
    honesty principle as the per-prediction view.

    Returns a DataFrame sorted by mean_abs_shap_log descending, with columns:
    rank, feature, category ("skill" | "context"), mean_abs_shap_log.

    This is also where a real, checkable finding surfaces: if context
    features (percentile/job_zone) dominate the top of this ranking, that's
    worth reporting honestly in the notebook and README, not reclassifying
    away -- see module docstring point 2.
    """
    shap_values, _ = _compute_shap(explainer, X, feature_cols)
    mean_abs = np.abs(shap_values).mean(axis=0)

    df = pd.DataFrame({"feature": feature_cols, "mean_abs_shap_log": mean_abs})
    df["category"] = df["feature"].apply(lambda f: "context" if f in CONTEXT_FEATURES else "skill")
    df = df.sort_values("mean_abs_shap_log", ascending=False).reset_index(drop=True)
    df.insert(0, "rank", df.index + 1)

    return df.head(top_n) if top_n is not None else df

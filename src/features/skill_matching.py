"""
Skill -> occupation matching.

MVP approach: weighted overlap between user-input skills and O*NET essential
skill importance profiles. This is the fallback / baseline — simple, fully
interpretable, and a fair comparison point once embedding similarity is added.

Stretch: sentence-transformer embedding similarity, to handle free-text skill
input that doesn't exactly match O*NET's skill vocabulary (e.g. user types
"data viz" instead of "Programming" / "Mathematics").
"""

import warnings
from dataclasses import dataclass

import pandas as pd


@dataclass
class MatchResult:
    onetsoc_code: str
    title: str
    score: float


def weighted_overlap_match(
    user_skills: dict[str, float],
    skill_importance_matrix: pd.DataFrame,
    occupation_titles: pd.DataFrame,
    top_n: int = 10,
) -> list[MatchResult]:
    """Match user skills to occupations via cosine similarity.

    Parameters
    ----------
    user_skills : dict mapping O*NET skill element name -> user-reported
        strength (0-5 scale, matching O*NET's Importance scale).
    skill_importance_matrix : occupation x skill matrix from
        onet_loader.build_skill_importance_matrix().
    occupation_titles : DataFrame with onetsoc_code, title (from
        onet_loader.load_occupation_data()).
    top_n : number of top matches to return.

    Returns
    -------
    List of MatchResult, sorted by score descending. Score is cosine
    similarity in [-1, 1] (in practice [0, 1] here since importance ratings
    are non-negative) restricted to the skill dimensions the user provided.

    Note on the fix (2026-07): the original version divided the dot product
    only by the user vector's norm, not the occupation vector's. That meant
    occupations scoring uniformly high across ALL specified skills could
    outrank occupations that were a sharper, more distinctive match — e.g.
    "Postsecondary Teachers" (broadly high on Speaking/Coordination/
    Persuasion/Personnel Management across many subjects) outscored
    "Human Resources Managers" for a management-skill test case, purely
    because teaching roles rate moderately-high on more of the specified
    skills at once. True cosine similarity (normalizing by BOTH vectors'
    magnitude) rewards matching the *shape* of the input profile, not just
    scoring high in absolute terms — confirmed against the same Test 1/
    Test 2 sanity checks from the Phase 0 EDA notebook before this replaced
    the original.

    Still intentionally the simplest defensible baseline, not the final
    method — embedding similarity remains the stretch upgrade for handling
    free-text skill input that doesn't match O*NET's vocabulary exactly.
    """
    user_vec = pd.Series(user_skills)
    common_cols = skill_importance_matrix.columns.intersection(user_vec.index)

    if len(common_cols) == 0:
        raise ValueError(
            "None of the provided skill names match O*NET skill element "
            "names. Check spelling against the Essential Skills element "
            "list (34 standard elements, e.g. 'Reading Comprehension', "
            "'Programming', 'Critical Thinking')."
        )

    if len(common_cols) < len(user_vec):
        missing = sorted(set(user_vec.index) - set(common_cols))
        warnings.warn(
            f"weighted_overlap_match: {len(missing)} of {len(user_vec)} provided "
            f"skill(s) not found in skill_importance_matrix and were silently "
            f"dropped from the match: {missing}. This usually means the matrix "
            f"was built from Essential Skills.txt alone (Basic Skills only) — "
            f"use build_full_skill_importance_matrix() (Essential + "
            f"Transferable Skills combined) to cover the full O*NET skill space.",
            stacklevel=2,
        )

    matrix = skill_importance_matrix[common_cols].fillna(0)
    user_v = user_vec[common_cols].fillna(0)

    dot = matrix.dot(user_v)
    occ_norms = matrix.pow(2).sum(axis=1).pow(0.5)
    user_norm = (user_v.pow(2).sum() ** 0.5)

    scores = dot / (occ_norms * user_norm + 1e-9)
    top = scores.sort_values(ascending=False).head(top_n)

    titles = occupation_titles.set_index("onetsoc_code")["title"]
    return [
        MatchResult(onetsoc_code=code, title=titles.get(
            code, "Unknown"), score=float(score))
        for code, score in top.items()
    ]

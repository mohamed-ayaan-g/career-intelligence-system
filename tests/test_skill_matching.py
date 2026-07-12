import warnings

import pandas as pd
import pytest

from src.features.skill_matching import weighted_overlap_match


@pytest.fixture
def matrix():
    return pd.DataFrame(
        {
            "Programming": [5.0, 1.0],
            "Mathematics": [5.0, 2.0],
            "Negotiation": [1.0, 5.0],
        },
        index=["15-1252.00", "11-1011.00"],
    )


@pytest.fixture
def titles():
    return pd.DataFrame(
        {
            "onetsoc_code": ["15-1252.00", "11-1011.00"],
            "title": ["Software Developers", "Chief Executives"],
        }
    )


def test_scores_are_bounded_cosine_similarity(matrix, titles):
    """Cosine similarity must fall in [-1, 1] — in practice [0, 1] here since
    O*NET importance ratings are non-negative."""
    results = weighted_overlap_match({"Programming": 5, "Mathematics": 5}, matrix, titles, top_n=2)
    for r in results:
        assert -1.0 <= r.score <= 1.0 + 1e-9


def test_direction_match_outranks_magnitude_only_match(matrix, titles):
    """This is the regression test for the original bug: an occupation whose
    profile SHAPE matches the query should outrank one that merely has high
    absolute values on the query's skills without matching direction."""
    results = weighted_overlap_match({"Programming": 5, "Mathematics": 5}, matrix, titles, top_n=2)
    scores_by_code = {r.onetsoc_code: r.score for r in results}
    assert scores_by_code["15-1252.00"] > scores_by_code["11-1011.00"]


def test_perfect_direction_match_scores_near_one():
    """An occupation whose profile is a scalar multiple of the query should
    score ~1.0 under true cosine similarity."""
    matrix = pd.DataFrame({"Programming": [10.0], "Mathematics": [10.0]}, index=["15-1252.00"])
    titles = pd.DataFrame({"onetsoc_code": ["15-1252.00"], "title": ["Software Developers"]})
    results = weighted_overlap_match({"Programming": 5, "Mathematics": 5}, matrix, titles, top_n=1)
    assert results[0].score == pytest.approx(1.0, abs=1e-6)


def test_no_warning_when_all_skills_found(matrix, titles):
    """Regression test for the silent-partial-overlap bug: when every
    provided skill exists in the matrix, no warning should fire."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        weighted_overlap_match({"Programming": 5, "Mathematics": 5}, matrix, titles, top_n=2)
        assert len(caught) == 0


def test_warning_fires_on_partial_overlap(matrix, titles):
    """The specific bug we found and fixed: silently matching on fewer skills
    than the caller specified must now raise a visible warning naming exactly
    which skill(s) were dropped."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        weighted_overlap_match(
            {"Programming": 5, "Nonexistent Skill": 3}, matrix, titles, top_n=2
        )
        assert len(caught) == 1
        assert "Nonexistent Skill" in str(caught[0].message)


def test_no_common_skills_raises_value_error(matrix, titles):
    with pytest.raises(ValueError, match="None of the provided skill names"):
        weighted_overlap_match({"Nonexistent Skill": 5}, matrix, titles, top_n=2)

"""Unit tests for §8 Explainability metrics (path coverage/validity + NL-claim grounding).

Everything is deterministic and needs only small hand-built persona/recipe dicts and
representative templated explanation strings — no LLM, no graph, no 7GB hummus.
"""
from __future__ import annotations

import math

import pytest

from foodrag.baselines import constraints, explainability as ex

pytestmark = pytest.mark.unit


# ── helpers ──────────────────────────────────────────────────────────────────

def _recipe(**kw):
    base = {
        "recipe_url": kw.get("recipe_url", "u"),
        "ingredient_names": kw.get("ingredient_names", []),
        "tags": kw.get("tags", []),
        "servingSize [g]": kw.get("serving", 100),
    }
    base.update({k: v for k, v in kw.items()
                 if k not in ("recipe_url", "ingredient_names", "tags", "serving")})
    return base


# ══ §8.1 path coverage ═══════════════════════════════════════════════════════

def test_path_coverage_fully_covered():
    persona = {
        "relative_nutrients": {"totalFat [g]": {"limits": {"lower": 0, "upper": 15}}},
        "nutrition_scores": {"fsa_score": 0.2},
        "total_nutrients": {"calories [kcal]": 500},
        "tags": {"likes": ["vegan"], "dislikes": ["spicy"]},
        "ingredients": {"likes": ["tofu"], "dislikes": ["peanut"]},
    }
    # one recipe supplies every inclusion requirement and avoids every exclusion
    rec = _recipe(**{
        "totalFat [g]": 10, "fsa_score": 0.5, "calories [kcal]": 300, "serving": 100,
        "tags": ["vegan"], "ingredient_names": ["tofu", "rice"],
    })
    assert ex.path_coverage(persona, [rec]) == pytest.approx(1.0)
    assert ex.path_validity(persona, [rec]) == pytest.approx(1.0)


def test_path_coverage_partially_covered():
    persona = {
        "relative_nutrients": {"totalFat [g]": {"limits": {"lower": 0, "upper": 15}}},
        "nutrition_scores": {"fsa_score": 0.9},   # NOT met -> unsupported
        "tags": {"likes": ["vegan"]},
    }
    rec = _recipe(**{"totalFat [g]": 10, "fsa_score": 0.3, "tags": ["vegan"]})
    # 3 requirements, 2 supported (fat + vegan tag), fsa min unmet -> 2/3
    assert ex.path_coverage(persona, [rec]) == pytest.approx(2 / 3)
    # >=1 covered -> validity 1.0 by construction
    assert ex.path_validity(persona, [rec]) == pytest.approx(1.0)


def test_path_coverage_exclusion_only_supported_when_avoided():
    persona = {"ingredients": {"dislikes": ["peanut"]}, "tags": {"dislikes": ["spicy"]}}
    clean = _recipe(ingredient_names=["rice"], tags=["vegan"])
    # both exclusions hold across all recipes -> both supported -> coverage 1.0
    assert ex.path_coverage(persona, [clean]) == pytest.approx(1.0)
    # a recipe that DOES contain peanut breaks the peanut exclusion -> 1/2
    dirty = _recipe(ingredient_names=["peanut"], tags=["vegan"])
    assert ex.path_coverage(persona, [clean, dirty]) == pytest.approx(1 / 2)


def test_path_zero_requirements_is_nan():
    persona = {"tags": {"likes": [], "dislikes": []},
               "ingredients": {"likes": [], "dislikes": []}}
    rec = _recipe()
    assert math.isnan(ex.path_coverage(persona, [rec]))
    # no covered requirement -> validity NaN too
    assert math.isnan(ex.path_validity(persona, [rec]))


def test_path_validity_nan_when_nothing_covered():
    # a single unmet nutrient requirement -> coverage 0, no covered path -> validity NaN
    persona = {"nutrition_scores": {"fsa_score": 0.9}}
    rec = _recipe(**{"fsa_score": 0.1})
    assert ex.path_coverage(persona, [rec]) == pytest.approx(0.0)
    assert math.isnan(ex.path_validity(persona, [rec]))


def test_path_exclusion_with_no_recipes_unsupported():
    persona = {"ingredients": {"dislikes": ["peanut"]}}
    # nothing recommended -> exclusion vacuous but nothing to explain -> unsupported
    assert ex.path_coverage(persona, []) == pytest.approx(0.0)
    assert math.isnan(ex.path_validity(persona, []))


def test_path_ignores_unresolved_none_rows():
    persona = {"tags": {"likes": ["vegan"]}}
    rec = _recipe(tags=["vegan"])
    assert ex.path_coverage(persona, [None, rec]) == pytest.approx(1.0)


# ══ §8.2 claim extraction ════════════════════════════════════════════════════

# The wording mirrors ExplanationTool.EXPLANATION_TEMPLATE's worked example.
_REAL_EXPLANATION = (
    "The recommended meal plan includes a selection of fish-based dishes that meet your "
    "specific dietary needs. Each meal maintains an FSA score of at least 0.06, supporting "
    "better nutritional quality. The fat content is within your requested range of 30 to 300 "
    "grams, and the cholesterol levels stay below your 356.06mg limit. Additionally, all meals "
    "exclude any ingredients with Californian tags, ensuring they align with your preferences."
)


def test_extract_claims_from_real_template_wording():
    claims = ex.extract_claims(_REAL_EXPLANATION)
    types = {c["type"] for c in claims}
    assert "score_min" in types
    assert "nutrient_range" in types
    assert "nutrient_max" in types
    assert "tag_exclude" in types

    score = next(c for c in claims if c["type"] == "score_min")
    assert score["column"] == "fsa_score" and score["value"] == pytest.approx(0.06)

    rng = next(c for c in claims if c["type"] == "nutrient_range")
    assert rng["column"] == "totalFat [g]"
    assert rng["low"] == pytest.approx(30) and rng["high"] == pytest.approx(300)

    mx = next(c for c in claims if c["type"] == "nutrient_max")
    assert mx["column"] == "cholesterol [mg]" and mx["value"] == pytest.approx(356.06)

    tag = next(c for c in claims if c["type"] == "tag_exclude")
    assert tag["tag"] == "californian"


def test_extract_nutriscore_letter_grade():
    claims = ex.extract_claims("Every dish keeps a NutriScore of B or better.")
    grade = next(c for c in claims if c["type"] == "score_min")
    assert grade["column"] == "nutri_score"
    assert grade["value"] == pytest.approx(0.8)  # B -> 0.8


def test_extract_ingredient_include_exclude():
    claims = ex.extract_claims("The plan includes spinach and excludes peanut for safety.")
    inc = [c for c in claims if c["type"] == "ingredient_include"]
    exc = [c for c in claims if c["type"] == "ingredient_exclude"]
    assert any(c["ingredient"] == "spinach" for c in inc)
    assert any(c["ingredient"] == "peanut" for c in exc)


def test_extract_claims_empty_text():
    assert ex.extract_claims("") == []
    assert ex.extract_claims("   ") == []


# ══ §8.2 claim grounding ═════════════════════════════════════════════════════

def test_claim_grounding_all_supported():
    claims = [
        {"type": "score_min", "column": "fsa_score", "value": 0.06},
        {"type": "nutrient_range", "column": "totalFat [g]", "low": 30, "high": 300},
        {"type": "ingredient_include", "ingredient": "salmon"},
        {"type": "tag_exclude", "tag": "californian"},
    ]
    rec = _recipe(**{"fsa_score": 0.5, "totalFat [g]": 100,
                     "ingredient_names": ["salmon"], "tags": ["healthy"]})
    out = ex.claim_grounding(claims, [rec])
    assert out["claim_grounding_rate"] == pytest.approx(1.0)
    assert out["unsupported_claim_rate"] == pytest.approx(0.0)


def test_claim_grounding_hallucinated_claim_unsupported():
    # explanation claims a NutriScore >= B but the recipe scores 0.4 (D) -> unsupported;
    # and claims it includes tofu, which it doesn't
    claims = [
        {"type": "score_min", "column": "nutri_score", "value": 0.8},   # hallucinated
        {"type": "ingredient_include", "ingredient": "tofu"},           # hallucinated
        {"type": "score_min", "column": "fsa_score", "value": 0.1},     # true
    ]
    rec = _recipe(**{"nutri_score": 0.4, "fsa_score": 0.5, "ingredient_names": ["beef"]})
    out = ex.claim_grounding(claims, [rec])
    assert out["claim_grounding_rate"] == pytest.approx(1 / 3)
    assert out["unsupported_claim_rate"] == pytest.approx(2 / 3)


def test_claim_grounding_no_claims_is_nan():
    out = ex.claim_grounding([], [_recipe()])
    assert math.isnan(out["claim_grounding_rate"])
    assert math.isnan(out["unsupported_claim_rate"])


def test_claim_grounding_end_to_end_real_wording():
    claims = ex.extract_claims(_REAL_EXPLANATION)
    # recipes that satisfy the extracted claims
    rec = _recipe(**{"fsa_score": 0.5, "totalFat [g]": 100, "cholesterol [mg]": 100,
                     "ingredient_names": ["salmon"], "tags": ["healthy"]})
    out = ex.claim_grounding(claims, [rec])
    assert out["claim_grounding_rate"] == pytest.approx(1.0)


def test_exclusion_claim_unsupported_when_a_recipe_violates():
    claims = [{"type": "ingredient_exclude", "ingredient": "peanut"}]
    clean = _recipe(ingredient_names=["rice"])
    dirty = _recipe(ingredient_names=["peanut"])
    # one recipe contains peanut -> exclusion claim not grounded
    out = ex.claim_grounding(claims, [clean, dirty])
    assert out["claim_grounding_rate"] == pytest.approx(0.0)


# ══ one-shot block ═══════════════════════════════════════════════════════════

def test_explainability_metrics_block_with_explanation():
    persona = {"nutrition_scores": {"fsa_score": 0.2}, "tags": {"likes": ["vegan"]}}
    rec = _recipe(**{"fsa_score": 0.5, "tags": ["vegan"]})
    block = ex.explainability_metrics(
        persona, [rec], explanation_text="Each meal maintains an FSA score of at least 0.2.")
    assert block["path_coverage"] == pytest.approx(1.0)
    assert block["path_validity"] == pytest.approx(1.0)
    assert block["claim_grounding_rate"] == pytest.approx(1.0)
    assert block["unsupported_claim_rate"] == pytest.approx(0.0)


def test_explainability_metrics_block_without_explanation_claims_nan():
    persona = {"tags": {"likes": ["vegan"]}}
    rec = _recipe(tags=["vegan"])
    block = ex.explainability_metrics(persona, [rec], explanation_text="")
    assert block["path_coverage"] == pytest.approx(1.0)
    assert math.isnan(block["claim_grounding_rate"])
    assert math.isnan(block["unsupported_claim_rate"])

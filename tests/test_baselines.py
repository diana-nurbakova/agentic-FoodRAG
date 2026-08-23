"""Unit tests for the baseline ladder.

Uses a tiny synthetic recipe corpus so the deterministic rungs (0a, 0b), the
constraint predicates and the metric/aggregation plumbing are
checked without loading the full HUMMUS frame.
"""
from __future__ import annotations

import math

import pandas as pd
import pytest

from foodrag.baselines import (PopularityHealthAware, PopularityRaw, aggregate,
                               requirement_count, requirement_types, run_baseline)
from foodrag.baselines import constraints, metrics
from foodrag.baselines.base import answer_set_stratum, gold_urls

pytestmark = pytest.mark.unit


@pytest.fixture
def recipes_df() -> pd.DataFrame:
    # r1: healthy, peanut-free, top rated   r2: contains peanut   r3: high sugar
    return pd.DataFrame([
        {"recipe_id": 1, "recipe_url": "r1", "title": "Veg Bowl", "tags": ["vegan"],
         "ingredient_names": ["spinach", "rice"], "servingSize [g]": 100,
         "sugars [g]": 5, "who_score": 0.9, "fsa_score": 0.8, "nutri_score": 0.9,
         "average_rating": 4.9, "number_of_ratings": 1000},
        {"recipe_id": 2, "recipe_url": "r2", "title": "Peanut Stew", "tags": ["vegan"],
         "ingredient_names": ["peanut", "rice"], "servingSize [g]": 100,
         "sugars [g]": 5, "who_score": 0.7, "fsa_score": 0.6, "nutri_score": 0.7,
         "average_rating": 5.0, "number_of_ratings": 50},
        {"recipe_id": 3, "recipe_url": "r3", "title": "Sugar Cake", "tags": ["dessert"],
         "ingredient_names": ["sugar", "flour"], "servingSize [g]": 100,
         "sugars [g]": 80, "who_score": 0.2, "fsa_score": 0.1, "nutri_score": 0.1,
         "average_rating": 4.8, "number_of_ratings": 2000},
    ])


@pytest.fixture
def qa_item() -> dict:
    # vegan, no peanut, total sugar <= 10 g/serving → only r1 is valid
    return {
        "answer": ["r1"],
        "persona": {
            "tags": {"likes": ["vegan"], "dislikes": []},
            "ingredients": {"likes": [], "dislikes": ["peanut"]},
            "relative_nutrients": {},
            "total_nutrients": {"sugars [g]": 10},
            "nutrition_scores": {},
        },
    }


# ── constraint predicates / valid-set filter ─────────────────────────────────

def test_filter_valid_matches_only_compliant_recipe(recipes_df, qa_item):
    valid = constraints.filter_valid(qa_item["persona"], recipes_df)
    assert valid["recipe_url"].tolist() == ["r1"]


def test_predicates_flag_allergen_and_hard_nutrient(recipes_df, qa_item):
    persona = qa_item["persona"]
    rows = {r["recipe_url"]: r for r in recipes_df.to_dict("records")}
    assert constraints.violates_allergen(rows["r2"], persona)
    assert not constraints.violates_allergen(rows["r1"], persona)
    assert constraints.violates_hard_nutrient(rows["r3"], persona)  # 80g sugar > 10
    assert constraints.satisfies_all(rows["r1"], persona)
    assert not constraints.satisfies_all(rows["r3"], persona)  # wrong tag + sugar


# ── retrieval metrics ────────────────────────────────────────────────────────

def test_retrieval_metrics():
    gold = {"r1"}
    assert metrics.hit_at_k(["x", "r1"], gold, 10) == 1.0
    assert metrics.mrr_at_k(["x", "r1"], gold, 10) == 0.5
    assert metrics.ndcg_at_k(["r1"], gold, 10) == pytest.approx(1.0)
    assert metrics.hit_at_k(["x", "y"], gold, 10) == 0.0


# ── popularity rungs (0a / 0b) ───────────────────────────────────────────────

def test_popularity_raw_ignores_constraints(recipes_df, qa_item):
    rec = PopularityRaw(recipes_df).recommend(qa_item, k=3)
    # ranked by rating then #ratings: r2 (5.0) > r1 (4.9) > r3 (4.8)
    assert rec == ["r2", "r1", "r3"]


def test_popularity_health_aware_filters_first(recipes_df, qa_item):
    rec = PopularityHealthAware(recipes_df).recommend(qa_item, k=3)
    assert rec == ["r1"]  # only valid recipe survives the constraint filter


def test_raw_popularity_has_safety_failures_health_aware_does_not(recipes_df, qa_item):
    raw = run_baseline(PopularityRaw(recipes_df), [qa_item], recipes_df, k=3)
    aware = run_baseline(PopularityHealthAware(recipes_df), [qa_item], recipes_df, k=3)
    # the deliberately-unaware rung serves the peanut recipe; the aware one does not
    assert raw[0]["metrics"]["allergen_violation"] > 0
    assert aware[0]["metrics"]["allergen_violation"] == 0.0
    assert aware[0]["metrics"]["constraint_satisfaction"] == 1.0


# ── gold answers (both question types) ───────────────────────────────────────

def test_gold_urls_reads_flat_recipe_answer(qa_item):
    assert gold_urls(qa_item) == set(qa_item["answer"])


def test_gold_urls_flattens_meal_plan_answer_over_slots():
    # QuestionType.MEAL_PLAN gold: the per-slot mapping from _find_valid_combinations.
    qa = {"answer": {
        "breakfast": {"meal_type": "breakfast", "valid_recipes": ["r1", "r2"]},
        "lunch": {"meal_type": "lunch", "valid_recipes": ["r2", "r3"]},
        "snack 1": {"meal_type": "snack", "valid_recipes": []},
    }}
    # Union over slots, de-duped — NOT the meal names.
    assert gold_urls(qa) == {"r1", "r2", "r3"}


def test_gold_urls_never_returns_meal_names():
    """Regression: `set(answer)` on the meal-plan dict yields slot names, which
    scored every retrieval metric at 0 instead of raising."""
    qa = {"answer": {"breakfast": {"meal_type": "breakfast", "valid_recipes": ["r1"]}}}
    assert "breakfast" not in gold_urls(qa)


def test_gold_urls_empty_answers_are_falsy():
    assert gold_urls({}) == set()
    assert gold_urls({"answer": None}) == set()
    assert gold_urls({"answer": []}) == set()
    # All slots empty → empty stratum, not a phantom 1-item gold set.
    assert gold_urls({"answer": {"lunch": {"meal_type": "lunch", "valid_recipes": []}}}) == set()


# ── strata + records + aggregation ───────────────────────────────────────────

def test_stratum_and_requirements(qa_item):
    assert answer_set_stratum(0) == "empty"
    assert answer_set_stratum(1) == "tight"
    assert answer_set_stratum(500) == "medium"
    assert answer_set_stratum(9999) == "broad"
    assert requirement_count(qa_item["persona"]) == 3  # 1 tag + 1 dislike + 1 total-nutrient
    assert set(requirement_types(qa_item["persona"])) == {"tag", "allergen", "nutrient_bound"}


def test_record_and_aggregate_schema(recipes_df, qa_item):
    records = run_baseline(PopularityHealthAware(recipes_df), [qa_item], recipes_df,
                           k=10, experiment_id="EXP-7", run=1)
    rec = records[0]
    assert rec["config_id"] == "popularity_health_aware"
    assert rec["stratum"] == "tight"
    assert rec["metrics"][f"hit_at_10"] == 1.0

    agg = aggregate(records)
    assert list(agg.columns)[:4] == ["experiment_id", "config_id", "stratum", "metric"]
    # aggregate stratum is always present; tight too (single tight question)
    assert {"aggregate", "tight"} <= set(agg["stratum"])
    hit_row = agg[(agg["stratum"] == "aggregate") & (agg["metric"] == "hit_at_10")].iloc[0]
    assert hit_row["mean"] == 1.0
    assert hit_row["n_runs"] == 1


def test_empty_gold_makes_retrieval_nan(recipes_df, qa_item):
    qa_item = {**qa_item, "answer": []}
    rec = run_baseline(PopularityRaw(recipes_df), [qa_item], recipes_df, k=3)[0]
    assert math.isnan(rec["metrics"]["hit_at_3"])
    # safety metrics are still defined without a gold set
    assert not math.isnan(rec["metrics"]["allergen_violation"])

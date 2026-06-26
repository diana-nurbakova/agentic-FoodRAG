"""Deterministic evaluation-metric unit tests (Evaluation Protocol §4.3–4.5)."""
import pytest

from ui.services import eval_metrics as M

pytestmark = [pytest.mark.unit, pytest.mark.gui]

R_GLUTEN = {"title": "Wheat Pasta", "ingredients": "wheat flour, tomato", "sugars [g]": 5,
            "calories [kcal]": 400, "nutri_score": 0.75, "fsa_score": 0.6, "tags": "italian"}
R_SWEET = {"title": "Sweet Bowl", "ingredients": "oats, honey", "sugars [g]": 18,
           "calories [kcal]": 350, "nutri_score": 0.5, "fsa_score": 0.5, "tags": "breakfast"}
R_CLEAN = {"title": "Clean Salad", "ingredients": "lettuce, olive oil", "sugars [g]": 3,
           "calories [kcal]": 300, "nutri_score": 1.0, "fsa_score": 0.9, "tags": "fresh"}


def test_allergen_substring_and_expand():
    assert M.recipe_violates_allergen(R_GLUTEN, ["wheat"]) is True       # substring net
    assert M.recipe_violates_allergen(R_CLEAN, ["wheat"]) is False
    # FoodOn descendant expansion: blacklist class "gluten" → leaf "wheat flour"
    expand = lambda terms: {"wheat flour"} if "gluten" in {t.lower() for t in terms} else set()
    assert M.recipe_violates_allergen(R_GLUTEN, ["gluten"], expand=expand) is True
    assert M.recipe_violates_allergen(R_CLEAN, ["gluten"], expand=expand) is False
    assert M.allergen_violation_rate([R_GLUTEN, R_CLEAN], ["wheat"]) == 0.5


def test_hard_constraint_and_satisfaction():
    cons = [("sugars [g]", "<=", 4)]
    assert M.hard_constraint_violation_rate([R_GLUTEN, R_CLEAN], cons) == 0.5  # 5 > 4
    # R_GLUTEN fails (wheat + sugar 5>4); R_CLEAN passes (no wheat, sugar 3<=4) → 0.5
    assert M.constraint_satisfaction_rate([R_GLUTEN, R_CLEAN], ["wheat"], cons) == 0.5
    assert M.constraint_satisfaction_rate([R_CLEAN], ["wheat"], cons) == 1.0
    assert M.constraint_satisfaction_rate([R_GLUTEN], ["wheat"], cons) == 0.0


def test_health_metrics():
    assert round(M.avg_nutriscore([R_GLUTEN, R_CLEAN]), 3) == 0.875
    assert 0 <= M.avg_fsa([R_GLUTEN, R_CLEAN, R_SWEET]) <= 1


def test_plan_quality():
    plan = {
        "breakfast": {"title": "B", "ingredients": "oats, milk", "calories [kcal]": 290, "tags": "quick"},
        "lunch": {"title": "L", "ingredients": "lettuce, oil", "calories [kcal]": 420, "tags": "fresh"},
        "dinner": {"title": "D", "ingredients": "salmon, rice", "calories [kcal]": 420, "tags": "fish"},
        "snack": {"title": "S", "ingredients": "apple", "calories [kcal]": 180, "tags": "fruit"},
    }
    assert M.meal_slot_completeness(plan) == 1.0
    assert M.meal_slot_completeness({"breakfast": plan["breakfast"]}) == 0.25
    assert 0.0 <= M.intra_plan_diversity(plan) <= 1.0
    assert M.calorie_distribution_adherence(plan) > 0.95  # close to 22/32/32/14
    assert M.structural_validity(plan, explanation="because") is True
    assert M.structural_validity(plan, explanation="") is False
    assert M.structural_validity({"breakfast": {"title": ""}}, explanation="x") is False


def test_foodon_expander_uses_translate_ingredients():
    # No FoodOn → None (caller falls back to substring)
    assert M.foodon_expander({}) is None
    assert M.foodon_expander({"food_on": object()}) is None

    calls = {}

    class FakeFoodOn:
        def translate_ingredients(self, ingredients, include_food_on, include_food_com):
            calls["args"] = (list(ingredients), include_food_on, include_food_com)
            return ["Wheat Flour", "whole wheat bread"] if "gluten" in ingredients else []

    expand = M.foodon_expander({"food_on": FakeFoodOn()})
    out = expand(["gluten"])
    # genuine ontology call: include_food_on must be True (the §4.3 upgrade)
    assert calls["args"][1] is True
    assert "wheat flour" in out and "gluten" in out          # leaves + original, lower-cased
    assert M.recipe_violates_allergen(R_GLUTEN, ["gluten"], expand=expand) is True


def test_parse_nutrient_constraints():
    assert M.parse_nutrient_constraints("Sugar <= 10g/meal") == [("sugars [g]", "<=", 10.0)]
    assert M.parse_nutrient_constraints("Protein >= 30g/meal") == [("protein [g]", ">=", 30.0)]
    assert M.parse_nutrient_constraints("max 500 kcal and Fat under 20g") == [
        ("calories [kcal]", "<=", 500.0), ("totalFat [g]", "<=", 20.0)]
    assert M.parse_nutrient_constraints("None") == []


def test_safety_report():
    # descendant expander: blacklist "gluten" → leaf "wheat flour"
    expand = lambda terms: {"wheat flour"} if "gluten" in {t.lower() for t in terms} else set()
    rep = M.safety_report([R_GLUTEN, R_SWEET, R_CLEAN], blacklist=["gluten"],
                          constraints=[("sugars [g]", "<=", 10)], expand=expand)
    assert abs(rep["allergen_rate"] - 1 / 3) < 1e-9
    assert abs(rep["hard_rate"] - 1 / 3) < 1e-9
    assert abs(rep["satisfaction_rate"] - 1 / 3) < 1e-9
    # offending ingredient surfaced (the leaf actually present), not the class label
    assert rep["per_recipe"][0]["allergens"] == ["wheat flour"]
    assert rep["per_recipe"][1]["constraints"] == ["sugars ≤ 10"]
    assert rep["per_recipe"][2]["ok"] is True
    assert rep["any_violation"] is True

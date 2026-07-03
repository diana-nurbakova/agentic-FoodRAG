"""Unit tests for the added deterministic metrics (Evaluation Protocol §4.3, §4.5, §4.5.1).

Covers:
  * §4.3 hard/soft constraint split (metrics + constraints decomposition),
  * §4.5 slot_appropriateness,
  * §4.5.1 guideline adherence (class mapping, band scoring, hard-ceiling → violation),
and the full-pipeline record schema additions (plan + explanation persistence).

Everything here is deterministic and needs only small hand-built recipe dicts / a
tiny recipes_df — no LLM, no graph, no 7GB hummus.
"""
from __future__ import annotations

import math

import pandas as pd
import pytest

from foodrag.baselines import constraints, guideline_adherence as ga, meal_plan_metrics as mpm, metrics

pytestmark = pytest.mark.unit


# ── helpers ──────────────────────────────────────────────────────────────────

def _recipe(**kw):
    base = {
        "recipe_url": kw.get("recipe_url", "u"),
        "ingredient_names": kw.get("ingredient_names", []),
        "tags": kw.get("tags", []),
        "meal_types": kw.get("meal_types", []),
        "servingSize [g]": kw.get("serving", 100),
    }
    base.update({k: v for k, v in kw.items()
                 if k not in ("recipe_url", "ingredient_names", "tags", "meal_types", "serving")})
    return base


# ══ §4.3 hard/soft constraint split ══════════════════════════════════════════

def test_hard_constraints_decomposition_counts_each_bound():
    persona = {
        "relative_nutrients": {"totalFat [g]": {"limits": {"lower": 0, "upper": 15}}},
        "nutrition_scores": {"fsa_score": 0.2},
        "total_nutrients": {"calories [kcal]": 500},
        "ingredients": {"dislikes": ["peanut"]},
    }
    preds = constraints.hard_constraints(persona)
    assert len(preds) == 4  # 1 relative + 1 score + 1 total + 1 allergen


def test_soft_constraints_only_added_when_stated():
    # tags.likes + tags.dislikes + ingredients.likes all present -> 3 soft constraints
    persona = {"tags": {"likes": ["vegan"], "dislikes": ["spicy"]},
               "ingredients": {"likes": ["tofu"]}}
    assert len(constraints.soft_constraints(persona)) == 3
    # nothing stated -> 0 soft constraints
    assert constraints.soft_constraints({"tags": {"likes": [], "dislikes": []},
                                         "ingredients": {"likes": []}}) == []


def test_hard_sat_perconstraint_partial_credit():
    # persona: fat in [0,15] (per-100g) AND fsa>=0.2 AND allergen-free (no peanut)
    persona = {
        "relative_nutrients": {"totalFat [g]": {"limits": {"lower": 0, "upper": 15}}},
        "nutrition_scores": {"fsa_score": 0.2},
        "ingredients": {"dislikes": ["peanut"]},
    }
    # rec satisfies fat(10 ok) + allergen(no peanut) but fails fsa(0.1<0.2) -> 2/3
    rec = _recipe(**{"totalFat [g]": 10, "fsa_score": 0.1, "ingredient_names": ["rice"]})
    val = metrics.hard_constraint_sat_perconstraint([rec], persona, constraints.identity_expand, 10)
    assert val == pytest.approx(2 / 3)


def test_hard_sat_allornothing_and_unresolved_none():
    persona = {"relative_nutrients": {"totalFat [g]": {"limits": {"lower": 0, "upper": 15}}}}
    good = _recipe(**{"totalFat [g]": 10})
    bad = _recipe(**{"totalFat [g]": 30})
    # 1 of 2 real recs fully satisfies; None counts as failure -> 1/3
    rows = [good, bad, None]
    assert metrics.hard_constraint_sat_allornothing(rows, persona, constraints.identity_expand, 10) == pytest.approx(1 / 3)
    # per-constraint: good=1.0, bad=0.0, None=0.0 -> mean 1/3
    assert metrics.hard_constraint_sat_perconstraint(rows, persona, constraints.identity_expand, 10) == pytest.approx(1 / 3)


def test_soft_sat_perconstraint():
    persona = {"tags": {"likes": ["vegan"], "dislikes": ["spicy"]}, "ingredients": {"likes": ["tofu"]}}
    # rec has vegan tag (ok), not spicy (ok), but no tofu (fail) -> 2/3
    rec = _recipe(tags=["vegan"], ingredient_names=["rice"])
    assert metrics.soft_constraint_sat([rec], persona, constraints.identity_expand, 10) == pytest.approx(2 / 3)


def test_zero_hard_constraints_is_nan():
    persona = {"tags": {"likes": ["vegan"]}}  # only soft
    rec = _recipe(tags=["vegan"])
    assert math.isnan(metrics.hard_constraint_sat_perconstraint([rec], persona, constraints.identity_expand, 10))
    assert math.isnan(metrics.hard_constraint_sat_allornothing([rec], persona, constraints.identity_expand, 10))
    assert metrics.soft_constraint_sat([rec], persona, constraints.identity_expand, 10) == pytest.approx(1.0)


def test_zero_soft_constraints_is_nan():
    persona = {"relative_nutrients": {"totalFat [g]": {"limits": {"lower": 0, "upper": 15}}}}
    rec = _recipe(**{"totalFat [g]": 10})
    assert math.isnan(metrics.soft_constraint_sat([rec], persona, constraints.identity_expand, 10))


def test_deterministic_metrics_includes_new_keys_and_keeps_old():
    persona = {"relative_nutrients": {"totalFat [g]": {"limits": {"lower": 0, "upper": 15}}},
               "tags": {"likes": ["vegan"]}}
    rec = _recipe(**{"totalFat [g]": 10, "tags": ["vegan"], "nutri_score": 0.5, "fsa_score": 0.5})
    block = metrics.deterministic_metrics(["u"], [rec], {"g"}, persona, 10)
    for key in ("hard_constraint_sat_perconstraint", "hard_constraint_sat_allornothing",
                "soft_constraint_sat"):
        assert key in block
    # existing keys unchanged
    for key in ("hit_at_10", "allergen_violation", "hard_constraint_violation",
                "constraint_satisfaction", "hallucination_rate"):
        assert key in block


def test_avg_rating_at_k_is_relevance_axis():
    # §9.4 Pareto relevance axis: mean of average_rating over resolved top-k
    persona = {}
    r1 = _recipe(**{"average_rating": 4.0})
    r2 = _recipe(**{"average_rating": 5.0})
    block = metrics.deterministic_metrics(["a", "b"], [r1, r2], set(), persona, 10)
    assert block["avg_rating_at_10"] == pytest.approx(4.5)
    # NaN when nothing resolves
    block_none = metrics.deterministic_metrics(["a"], [None], set(), persona, 10)
    assert math.isnan(block_none["avg_rating_at_10"])


# ══ §4.5 slot_appropriateness ════════════════════════════════════════════════

def test_slot_appropriateness_matches_synonyms():
    plan = {
        "breakfast": _recipe(meal_types=["brunch"]),        # brunch ∈ breakfast synonyms -> ok
        "lunch": _recipe(meal_types=["main-dish"]),          # ok
        "dinner": _recipe(meal_types=["dessert"]),           # dessert NOT in dinner synonyms -> not ok
        "snack": _recipe(meal_types=["snack"]),              # ok
    }
    assert mpm.slot_appropriateness(plan) == pytest.approx(3 / 4)


def test_slot_appropriateness_unknown_meal_types_not_penalised():
    plan = {"breakfast": _recipe(meal_types=[]), "lunch": _recipe(meal_types=None)}
    assert mpm.slot_appropriateness(plan) == pytest.approx(1.0)


def test_slot_appropriateness_accepts_string_repr_meal_types():
    # recipes_df stores meal_types as a string repr of a list
    plan = {"breakfast": _recipe(meal_types="['breakfast', 'brunch']"),
            "dinner": _recipe(meal_types="['dessert']")}
    assert mpm.slot_appropriateness(plan) == pytest.approx(1 / 2)


def test_slot_appropriateness_empty_plan_is_nan():
    assert math.isnan(mpm.slot_appropriateness({"breakfast": None, "lunch": None}))


def test_plan_quality_metrics_has_slot_appropriateness_key():
    plan = {"breakfast": _recipe(meal_types=["breakfast"], **{"calories [kcal]": 100})}
    block = mpm.plan_quality_metrics(plan, require_explanation=False)
    assert "slot_appropriateness" in block
    # old keys preserved
    for key in ("structural_validity", "meal_slot_completeness", "intra_plan_diversity",
                "calorie_adherence"):
        assert key in block


# ══ §4.5.1 guideline adherence ═══════════════════════════════════════════════

@pytest.fixture
def ref_tables():
    return ga.load_reference_tables()


def _nutrient_recipe(**totals):
    """A recipe whose per-100g values equal the given whole-serving totals (serving=100)."""
    cols = {
        "energy_kcal": "calories [kcal]", "protein_g": "protein [g]",
        "carbohydrate_g": "totalCarbohydrate [g]", "total_fat_g": "totalFat [g]",
        "saturated_fat_g": "saturatedFat [g]", "fibre_g": "dietaryFiber [g]",
        "sodium_g": "sodium [mg]", "sugars_g": "sugars [g]",
    }
    rec = _recipe(serving=100)
    for nut, val in totals.items():
        col = cols[nut]
        rec[col] = val * 1000 if nut == "sodium_g" else val  # sodium stored in mg
    return rec


def test_class_mapping_default_general_adult(ref_tables):
    persona = {"tags": {}, "ingredients": {}}  # no condition
    label, bands = ga.map_persona_to_class(persona, ref_tables)
    assert label == "general_adult"
    assert bands["energy_kcal"]["low"] == 1800


def test_class_mapping_diabetic_override(ref_tables):
    persona = {"condition": "type 2 diabetes"}
    label, bands = ga.map_persona_to_class(persona, ref_tables)
    assert label == "diabetic_adult"
    # diabetic carb band is tighter (200-250) than general (225-300)
    assert bands["carbohydrate_g"]["high"] == 250
    assert bands["sugars_g"]["high"] == 25  # stricter sugar


def test_class_mapping_comorbid_conflict_resolution(ref_tables):
    # diabetic + hypertension -> stricter per nutrient (min upper / max lower)
    persona = {"condition": "type 2 diabetes and hypertension"}
    label, bands = ga.map_persona_to_class(persona, ref_tables)
    assert label.startswith("composite(")
    assert "diabetic_adult" in label and "hypertension_adult" in label
    # carb from diabetic (250), sodium ceiling from hypertension (2.0), sugars from diabetic (25)
    assert bands["carbohydrate_g"]["high"] == 250
    assert bands["sodium_g"]["hard_ceiling"] == 2.0
    assert bands["sugars_g"]["high"] == 25


def test_guideline_within_band_scores_one(ref_tables):
    # single recipe hitting every general_adult band mid-range
    rec = _nutrient_recipe(energy_kcal=2000, protein_g=80, carbohydrate_g=260, total_fat_g=60,
                           saturated_fat_g=10, fibre_g=30, sodium_g=2.0, sugars_g=30)
    plan = {"breakfast": rec}
    out = ga.guideline_adherence(plan, {}, ref_tables)
    assert out["guideline_adherence_overall"] == pytest.approx(1.0)
    assert all(v == 1 for v in out["guideline_adherence_per_nutrient"].values())
    assert out["guideline_hard_ceiling_breach"] is False


def test_guideline_under_and_over_band(ref_tables):
    # energy far under (500 < 1800), sodium over-band but within ceiling (3.0 in [0.575,3.5])
    rec = _nutrient_recipe(energy_kcal=500, protein_g=80, carbohydrate_g=260, total_fat_g=60,
                           saturated_fat_g=10, fibre_g=30, sodium_g=3.0, sugars_g=30)
    out = ga.guideline_adherence({"lunch": rec}, {}, ref_tables)
    assert out["guideline_adherence_per_nutrient"]["energy"] == 0
    assert out["guideline_direction"]["energy"] == "under"
    assert out["guideline_adherence_per_nutrient"]["sodium"] == 1  # 3.0 within [0.575,3.5]
    assert out["guideline_hard_ceiling_breach"] is False


def test_guideline_hard_ceiling_breach(ref_tables):
    # sodium 4.0 g > 3.5 g general_adult hard_ceiling -> breach True
    rec = _nutrient_recipe(energy_kcal=2000, protein_g=80, carbohydrate_g=260, total_fat_g=60,
                           saturated_fat_g=10, fibre_g=30, sodium_g=4.0, sugars_g=30)
    out = ga.guideline_adherence({"dinner": rec}, {}, ref_tables)
    assert out["guideline_hard_ceiling_breach"] is True
    assert out["guideline_adherence_per_nutrient"]["sodium"] == 0  # also out of band


def test_guideline_empty_plan_is_nan(ref_tables):
    out = ga.guideline_adherence({"breakfast": None, "lunch": None}, {}, ref_tables)
    assert math.isnan(out["guideline_adherence_overall"])
    assert out["guideline_adherence_per_nutrient"] == {}


def test_guideline_missing_nutrient_column_scored_zero(ref_tables):
    # recipe with no nutrient columns at all -> totals None -> each per-nutrient 0
    rec = _recipe(serving=100)  # no nutrient cols
    out = ga.guideline_adherence({"snack": rec}, {}, ref_tables)
    # at least one recipe present -> not NaN; every nutrient missing -> 0
    assert out["guideline_adherence_overall"] == pytest.approx(0.0)
    assert set(out["guideline_adherence_per_nutrient"].values()) == {0}


def test_guideline_plan_totals_sum_across_slots(ref_tables):
    # two recipes each 1000 kcal -> plan total 2000 (within band)
    r1 = _nutrient_recipe(energy_kcal=1000, protein_g=40, carbohydrate_g=130, total_fat_g=30,
                          saturated_fat_g=5, fibre_g=15, sodium_g=1.0, sugars_g=15)
    r2 = _nutrient_recipe(energy_kcal=1000, protein_g=40, carbohydrate_g=130, total_fat_g=30,
                          saturated_fat_g=5, fibre_g=15, sodium_g=1.0, sugars_g=15)
    totals = ga.plan_daily_totals({"breakfast": r1, "dinner": r2})
    assert totals["energy_kcal"] == pytest.approx(2000)
    assert totals["sodium_g"] == pytest.approx(2.0)


def test_short_keys_are_exact(ref_tables):
    rec = _nutrient_recipe(energy_kcal=2000, protein_g=80, carbohydrate_g=260, total_fat_g=60,
                           saturated_fat_g=10, fibre_g=30, sodium_g=2.0, sugars_g=30)
    out = ga.guideline_adherence({"breakfast": rec}, {}, ref_tables)
    assert set(out["guideline_adherence_per_nutrient"].keys()) == {
        "energy", "protein", "carb", "fat", "satfat", "fibre", "sodium", "sugar"}


def test_reference_tables_injectable_for_tests():
    # a minimal synthetic ref-tables dict should work without touching the packaged file
    fake = {
        "reference_classes": {"general_adult": {"nutrients": {
            "energy_kcal": {"low": 100, "high": 200, "band_type": "range"}}}},
        "persona_to_class_mapping": {"rules": [{"default": "general_adult"}]},
        "metric_computation": {"tracked_nutrients": ["energy_kcal"]},
    }
    rec = _nutrient_recipe(energy_kcal=150)
    out = ga.guideline_adherence({"breakfast": rec}, {}, fake)
    assert out["guideline_adherence_overall"] == pytest.approx(1.0)
    assert out["guideline_adherence_per_nutrient"] == {"energy": 1}


# ══ full-pipeline record schema (plan + explanation persistence, guideline wiring) ══

from foodrag.baselines.full_pipeline import AssembledPlan, MealPlanAssembler, run_full_pipeline
from foodrag.baselines.judge import Judge


class _FakeAssembler(MealPlanAssembler):
    config_id = "foodrag_h1"
    require_explanation = True

    def __init__(self, plan, urls, explanation="", structured=None):
        self._plan, self._urls = plan, urls
        self._expl, self._struct = explanation, structured or {}

    def assemble(self, qa_item):
        return AssembledPlan(
            question=qa_item.get("question", ""), answer_text="plan text", plan=self._plan,
            recipe_urls=self._urls, contexts=["r1"], has_explanation=bool(self._expl),
            explanation_text=self._expl, explanation_structured=self._struct,
            llm_usage=3, llm_seq_usage=2, llm_parsing=0, latency_s=1.0)


def _pfp_recipes_df():
    return pd.DataFrame([
        {"recipe_id": 1, "recipe_url": "u1", "title": "Bowl", "tags": ["vegan"],
         "ingredient_names": ["rice"], "meal_types": ["breakfast"], "servingSize [g]": 100,
         "calories [kcal]": 2000, "protein [g]": 80, "totalCarbohydrate [g]": 260,
         "totalFat [g]": 60, "saturatedFat [g]": 10, "dietaryFiber [g]": 30,
         "sodium [mg]": 2000, "sugars [g]": 30, "nutri_score": 0.9, "fsa_score": 0.8,
         "average_rating": 4.9, "number_of_ratings": 10},
    ])


def _persona():
    return {"tags": {"likes": [], "dislikes": []}, "ingredients": {"likes": [], "dislikes": []},
            "relative_nutrients": {}, "total_nutrients": {}, "nutrition_scores": {}}


def test_record_persists_plan_and_explanation():
    df = _pfp_recipes_df()
    plan = {"breakfast": df.iloc[0].to_dict(), "lunch": None, "dinner": None, "snack": None}
    asm = _FakeAssembler(plan, ["u1"], explanation="This plan is balanced.",
                         structured={"breakfast": "VALID: fits requirements"})
    qa = {"question": "q", "answer": ["u1"], "persona": _persona()}
    rec = run_full_pipeline(asm, [qa], df, judges=[], k=10)[0]

    assert rec["plan"] == {"breakfast": "u1", "lunch": None, "dinner": None, "snack": None}
    assert rec["explanation_text"] == "This plan is balanced."
    assert rec["explanation_structured"] == {"breakfast": "VALID: fits requirements"}
    # new plan metrics present in the record
    for key in ("slot_appropriateness", "guideline_adherence_overall",
                "guideline_adherence_per_nutrient", "guideline_class"):
        assert key in rec["metrics"]
    assert rec["metrics"]["guideline_class"] == "general_adult"


def test_record_hard_ceiling_breach_raises_hard_constraint_violation():
    df = _pfp_recipes_df()
    df.loc[0, "sodium [mg]"] = 5000  # 5 g sodium > 3.5 g general_adult hard_ceiling
    plan = {"breakfast": df.iloc[0].to_dict(), "lunch": None, "dinner": None, "snack": None}
    asm = _FakeAssembler(plan, ["u1"], explanation="x")
    qa = {"question": "q", "answer": ["u1"], "persona": _persona()}
    rec = run_full_pipeline(asm, [qa], df, judges=[], k=10)[0]
    # guideline hard-ceiling breach folds into hard_constraint_violation (max -> 1.0)
    assert rec["metrics"]["hard_constraint_violation"] == pytest.approx(1.0)

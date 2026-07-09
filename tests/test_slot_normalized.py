"""Unit tests for the slot-normalized plan-level metrics (spec §3).

Deterministic: hand-built recipe dicts / a tiny recipes_df + the real reference-tables
JSON (bands). No LLM, no graph, no 7GB hummus.

Covers: 2-of-2 filled -> completeness_norm 1.0; band scaling by rho; ceiling scaling for
the *adherence* score but NOT for the safety hard-ceiling flag; diversity null on a single
filled slot; Q sourced from persisted requested_meal_types vs approximated from the plan.
"""
from __future__ import annotations

import math

import pandas as pd
import pytest

from foodrag.baselines import guideline_adherence as ga
from foodrag.baselines import slot_normalized as sn

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def ref():
    return ga.load_reference_tables()


# ── helpers ──────────────────────────────────────────────────────────────────

def _recipe(*, meal_types=(), serving=100, **nutrients):
    """A recipe row; nutrient kwargs are per-100 g so serving=100 → whole-serving == value."""
    row = {
        "recipe_url": nutrients.pop("recipe_url", "u"),
        "ingredient_names": nutrients.pop("ingredient_names", []),
        "tags": nutrients.pop("tags", []),
        "meal_types": list(meal_types),
        "servingSize [g]": serving,
    }
    cols = {
        "cal": "calories [kcal]", "protein": "protein [g]", "carb": "totalCarbohydrate [g]",
        "fat": "totalFat [g]", "satfat": "saturatedFat [g]", "fibre": "dietaryFiber [g]",
        "sodium_mg": "sodium [mg]", "sugar": "sugars [g]",
    }
    for k, col in cols.items():
        if k in nutrients:
            row[col] = nutrients[k]
    return row


# ── §3.1 completeness_norm ────────────────────────────────────────────────────

def test_completeness_norm_2_of_2_filled_is_one():
    plan = {"lunch": _recipe(meal_types=["lunch"]), "dinner": _recipe(meal_types=["dinner"])}
    assert sn.meal_slot_completeness_norm(plan, ["lunch", "dinner"]) == 1.0


def test_completeness_norm_half_when_one_requested_slot_empty():
    plan = {"lunch": _recipe(meal_types=["lunch"]), "dinner": None}
    assert sn.meal_slot_completeness_norm(plan, ["lunch", "dinner"]) == 0.5


def test_completeness_norm_nan_when_nothing_requested():
    assert math.isnan(sn.meal_slot_completeness_norm({}, []))


# ── §3.2 structural_validity_norm ─────────────────────────────────────────────

def test_structural_norm_valid_when_all_requested_filled_and_admissible():
    plan = {"lunch": _recipe(meal_types=["lunch"]), "dinner": _recipe(meal_types=["dinner"])}
    assert sn.structural_validity_norm(plan, ["lunch", "dinner"]) == 1.0


def test_structural_norm_zero_when_recipe_does_not_admit_slot():
    # a breakfast-only recipe dropped into the dinner slot -> not admissible
    plan = {"lunch": _recipe(meal_types=["lunch"]), "dinner": _recipe(meal_types=["breakfast"])}
    assert sn.structural_validity_norm(plan, ["lunch", "dinner"]) == 0.0


def test_structural_norm_zero_when_requested_slot_missing():
    plan = {"lunch": _recipe(meal_types=["lunch"])}
    assert sn.structural_validity_norm(plan, ["lunch", "dinner"]) == 0.0


# ── §3.3 rho ──────────────────────────────────────────────────────────────────

def test_rho_lunch_dinner():
    assert sn.rho(["lunch", "dinner"]) == pytest.approx(0.64)


def test_rho_full_day_is_one():
    assert sn.rho(["breakfast", "lunch", "dinner", "snack"]) == pytest.approx(1.0)


# ── §3.4 guideline_adherence_norm: band scaling by rho ────────────────────────

def test_band_target_energy_in_scaled_band_but_out_of_full_band(ref):
    """1300 kcal over lunch+dinner: inside the rho-scaled energy band [1152,1408] (norm=1),
    outside the full-day band [1800,2200] (full=0)."""
    plan = {"lunch": _recipe(meal_types=["lunch"], cal=650),
            "dinner": _recipe(meal_types=["dinner"], cal=650)}
    totals = ga.plan_daily_totals(plan)
    bands = sn.bands_for_class_label("general_adult", ref)

    per_norm, _ = sn.guideline_adherence_norm(totals, bands, sn.rho(["lunch", "dinner"]), ref)
    assert per_norm["energy"] == 1

    full = ga.guideline_adherence(plan, {})  # empty persona -> general_adult, full-day band
    assert full["guideline_adherence_per_nutrient"]["energy"] == 0


def test_band_scaling_degrades_to_full_at_rho_one(ref):
    """rho == 1 (all four slots requested) -> scaled band == full band."""
    plan = {"breakfast": _recipe(meal_types=["breakfast"], cal=2000)}
    totals = ga.plan_daily_totals(plan)
    bands = sn.bands_for_class_label("general_adult", ref)
    per, _ = sn.guideline_adherence_norm(totals, bands, sn.rho(list(sn.DELTA)), ref)
    assert per["energy"] == 1  # 2000 in [1800,2200]


# ── §3.4 ceiling scaling for adherence but NOT for the safety flag ────────────

def test_ceiling_scaled_for_adherence_but_safety_flag_unscaled(ref):
    """Sodium 3.0 g over lunch+dinner (rho=.64): scaled adherence ceiling = 2.24 g -> the
    adherence score marks it 0, yet 3.0 <= 3.5 g real UL so the safety hard-ceiling breach
    stays False. The safety ceiling is never scaled (spec §3.4)."""
    plan = {"lunch": _recipe(meal_types=["lunch"], sodium_mg=1500),
            "dinner": _recipe(meal_types=["dinner"], sodium_mg=1500)}  # 3000 mg = 3.0 g
    totals = ga.plan_daily_totals(plan)
    bands = sn.bands_for_class_label("general_adult", ref)

    per_norm, _ = sn.guideline_adherence_norm(totals, bands, sn.rho(["lunch", "dinner"]), ref)
    assert per_norm["sodium"] == 0                       # 3.0 > rho*3.5 = 2.24 -> not adherent

    # safety hard-ceiling (unscaled) is NOT breached: 3.0 <= 3.5
    assert ga.hard_ceiling_breach(plan, {}) is False


def test_ceiling_adherent_at_full_day(ref):
    """Same 3.0 g sodium judged over the full day (rho=1) IS adherent (<= 3.5)."""
    plan = {"lunch": _recipe(meal_types=["lunch"], sodium_mg=1500),
            "dinner": _recipe(meal_types=["dinner"], sodium_mg=1500)}
    totals = ga.plan_daily_totals(plan)
    bands = sn.bands_for_class_label("general_adult", ref)
    per, _ = sn.guideline_adherence_norm(totals, bands, 1.0, ref)
    assert per["sodium"] == 1


# ── §3.5 calorie_adherence_norm ───────────────────────────────────────────────

def test_calorie_norm_perfect_at_renormalized_split():
    """lunch/dinner target renormalizes to .41/.41... equal-calorie halves deviate a little
    but a .5/.5 split against .5/.5 (lunch+dinner equal weights) is exact -> ~1.0."""
    # lunch & dinner share delta .32/.32 -> renorm .5/.5; equal calories -> deviation 0
    plan = {"lunch": _recipe(meal_types=["lunch"], cal=500),
            "dinner": _recipe(meal_types=["dinner"], cal=500)}
    val = sn.calorie_adherence_norm(plan, ["lunch", "dinner"], sn.rho(["lunch", "dinner"]))
    assert val == pytest.approx(1.0)


def test_calorie_norm_nan_without_calories():
    plan = {"lunch": _recipe(meal_types=["lunch"])}
    assert math.isnan(sn.calorie_adherence_norm(plan, ["lunch"], sn.rho(["lunch"])))


# ── §3.6 diversity guard ──────────────────────────────────────────────────────

def test_diversity_null_on_single_filled_slot(ref):
    single = _recipe(recipe_url="L", meal_types=["lunch"], cal=500, ingredient_names=["a", "b"])
    indexed = _indexed_from([single])
    rec = {"plan": {"lunch": "L"}, "requested_meal_types": ["lunch"], "guideline_class": "general_adult"}
    assert math.isnan(sn.compute_record_norm(rec, indexed, ref)["intra_plan_diversity_norm"])


def test_diversity_defined_with_two_filled_slots():
    from foodrag.baselines import meal_plan_metrics as mpm
    plan = {"lunch": _recipe(ingredient_names=["a", "b"]), "dinner": _recipe(ingredient_names=["c", "d"])}
    assert not math.isnan(mpm.intra_plan_diversity(plan))
    single = {"lunch": _recipe(ingredient_names=["a", "b"])}
    assert math.isnan(mpm.intra_plan_diversity(single))


# ── Q source: persisted vs approx (end-to-end compute_record_norm) ────────────

def _indexed_from(recipes):
    df = pd.DataFrame(recipes) if recipes else pd.DataFrame(columns=["recipe_url"])
    return df.set_index("recipe_url", drop=False)


def test_q_source_persisted(ref):
    lunch = _recipe(recipe_url="L", meal_types=["lunch"], cal=650, ingredient_names=["a"])
    dinner = _recipe(recipe_url="D", meal_types=["dinner"], cal=650, ingredient_names=["b"])
    indexed = _indexed_from([lunch, dinner])
    rec = {
        "plan": {"lunch": "L", "dinner": "D"},
        "requested_meal_types": ["lunch", "dinner"],
        "guideline_class": "general_adult",
    }
    out = sn.compute_record_norm(rec, indexed, ref)
    assert out["requested_slots_source"] == "persisted"
    assert out["meal_slot_completeness_norm"] == 1.0
    assert out["n_requested_slots"] == 2
    assert out["requested_day_energy_share"] == pytest.approx(0.64)


def test_q_source_persisted_penalizes_unfilled_request(ref):
    lunch = _recipe(recipe_url="L", meal_types=["lunch"], cal=650)
    indexed = _indexed_from([lunch])
    rec = {"plan": {"lunch": "L"}, "requested_meal_types": ["lunch", "dinner"],
           "guideline_class": "general_adult"}
    out = sn.compute_record_norm(rec, indexed, ref)
    assert out["requested_slots_source"] == "persisted"
    assert out["meal_slot_completeness_norm"] == 0.5       # requested 2, filled 1
    assert out["structural_validity_norm"] == 0.0


def test_q_source_approx_from_plan(ref):
    lunch = _recipe(recipe_url="L", meal_types=["lunch"], cal=650)
    dinner = _recipe(recipe_url="D", meal_types=["dinner"], cal=650)
    indexed = _indexed_from([lunch, dinner])
    rec = {"plan": {"lunch": "L", "dinner": "D"}, "guideline_class": "general_adult"}  # no requested_meal_types
    out = sn.compute_record_norm(rec, indexed, ref)
    assert out["requested_slots_source"] == "approx_from_plan"
    assert out["meal_slot_completeness_norm"] == 1.0       # trivially: Q == filled slots
    assert out["structural_validity_norm"] == 1.0

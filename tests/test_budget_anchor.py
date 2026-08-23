"""Per-slot calorie budgets for meal-plan questions.

A meal-plan question states a cap per slot. Those caps are only interpretable if they
are shares of the *same* daily budget the question states as its total — otherwise the
question contradicts itself and a plan cannot satisfy it. These tests pin that
invariant, plus the slot vocabulary the caps are expressed in.
"""
from __future__ import annotations

import numpy as np
import pytest

from foodrag.baselines.guideline_adherence import resolve_daily_energy
from foodrag.food_rag.meal_plan.meal_type import MealType
from foodrag.persona import persona_generator
from foodrag.persona.qa_generator import QuestionType, _generate_question, _meal_plan_restriction

pytestmark = pytest.mark.unit

CAL = "calories [kcal]"


@pytest.fixture
def persona_config():
    return {"calories_per_meal_distribution": {
        "breakfast": 0.22, "lunch": 0.32, "dinner": 0.32, "snack 1": 0.07, "snack 2": 0.07}}


def _persona(**overrides):
    base = {"tags": {"likes": [], "dislikes": []}, "ingredients": {"likes": [], "dislikes": []},
            "relative_nutrients": {}, "total_nutrients": {}, "nutrition_scores": {}}
    base.update(overrides)
    return base


def _caps(restriction):
    return {slot: spec["guidelines"][CAL] for slot, spec in restriction.items()}


# ── the daily-energy anchor ──────────────────────────────────────────────────

def test_resolve_daily_energy_prefers_the_personas_own_cap():
    assert resolve_daily_energy(_persona(total_nutrients={CAL: 1650})) == pytest.approx(1650)


def test_resolve_daily_energy_falls_back_to_the_class_reference():
    # Shipped personas carry no condition field → general_adult → 2000 kcal.
    assert resolve_daily_energy(_persona()) == pytest.approx(2000)


def test_resolve_daily_energy_ignores_unusable_caps():
    for bad in ({}, {CAL: None}, {CAL: 0}, {CAL: "n/a"}):
        assert resolve_daily_energy(_persona(total_nutrients=bad)) == pytest.approx(2000)


# ── the coherence invariant ──────────────────────────────────────────────────

@pytest.mark.parametrize("stated_cap", [None, 1650, 2400])
def test_per_slot_caps_sum_to_the_personas_daily_budget(persona_config, stated_cap):
    """The invariant. Caps used to derive from randint(1600, 3000), independent of the
    persona, so for ~71% of personas they summed to more than the stated daily total."""
    persona = _persona(total_nutrients={CAL: stated_cap} if stated_cap else {})
    restriction = _meal_plan_restriction(persona, persona_config)
    budget = resolve_daily_energy(persona)
    # Rounding is to the nearest whole kcal per slot, so allow a few kcal of slack.
    assert sum(_caps(restriction).values()) == pytest.approx(budget, abs=5)


def test_caps_never_contradict_the_stated_total_in_the_question(persona_config):
    persona = _persona(total_nutrients={CAL: 1650})
    restriction = _meal_plan_restriction(persona, persona_config)
    question = _generate_question(persona, restriction, QuestionType.MEAL_PLAN)
    assert "Total Nutrition Limits" in question and "Meal Plan Restriction" in question
    assert sum(_caps(restriction).values()) <= 1650 + 5


def test_restriction_is_deterministic_given_the_persona(persona_config):
    """Full-day by construction: no random current-meal or already-eaten subtraction,
    so the same persona yields the same budget on every generation run."""
    persona = _persona(total_nutrients={CAL: 1800})
    np.random.seed(0)
    first = _caps(_meal_plan_restriction(persona, persona_config))
    np.random.seed(99)
    assert _caps(_meal_plan_restriction(persona, persona_config)) == first


# ── slot vocabulary ──────────────────────────────────────────────────────────

def test_restriction_slots_match_the_plan_meal_types(persona_config):
    """Caps are expressed in the four-slot MealType vocabulary that plans use; a
    five-slot restriction would cap two snacks a four-slot plan can never both fill."""
    restriction = _meal_plan_restriction(_persona(), persona_config)
    assert set(restriction) == {m.value for m in MealType.get_all()}


def test_merge_snacks_sums_the_two_snack_shares(persona_config):
    split = persona_generator.get_random_meal_plan_nutrients(
        persona_config, current_meal="breakfast", today_intaken_kcal=0, kcal_per_day=2000)
    merged = persona_generator.get_random_meal_plan_nutrients(
        persona_config, current_meal="breakfast", today_intaken_kcal=0, kcal_per_day=2000,
        merge_snacks=True)
    assert {"snack 1", "snack 2"} <= set(split) and "snack" not in split
    assert "snack" in merged and not {"snack 1", "snack 2"} & set(merged)
    assert merged["snack"]["guidelines"][CAL] == pytest.approx(
        split["snack 1"]["guidelines"][CAL] + split["snack 2"]["guidelines"][CAL])
    assert merged["snack"]["meal_type"] == "snack"


def test_explicit_kcal_per_day_overrides_the_random_draw(persona_config):
    caps = _caps(persona_generator.get_random_meal_plan_nutrients(
        persona_config, current_meal="breakfast", today_intaken_kcal=0,
        kcal_per_day=2000, merge_snacks=True))
    assert caps == {"breakfast": 440, "lunch": 640, "dinner": 640, "snack": 280}

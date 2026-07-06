"""Unit tests for the two FoodRAG safety fixes (error-analysis follow-ups).

FIX 1 — deterministic hard allergen/blacklist post-filter
    ``constraints.enforce_allergen_safety`` drops any plan slot whose recipe
    contains a persona-disliked (blacklisted) ingredient, guaranteeing the FINAL
    plan is allergen-safe even when the LLM validation lets one through
    (error analysis: persona q_01495 recommends 'green onions' across all 3 runs).

FIX 2 — gate explanation on a non-empty plan
    ``ExplanationTool.explain`` returns a fixed safe message and does NOT call the
    LLM when the recommendation is empty, so the Enhance agent never narrates
    recipes that were never selected (15.4% of H1-t2p explanations were fully
    hallucinated on empty-plan records).

All memory-light: no hummus load, LLM is mocked.
"""
from __future__ import annotations

import pytest

from foodrag.baselines import constraints
from foodrag.food_rag.agents.workflow_runner import get_dummy_state
from foodrag.food_rag.meal_plan.meal_plan_structures import (MealPlan,
                                                             MealPlanSubstitution)
from foodrag.food_rag.tools.explanation_tools import ExplanationTool
from foodrag.food_rag.utils.recipe_context import RecipeState

pytestmark = pytest.mark.unit


# ── helpers ──────────────────────────────────────────────────────────────────

def _recipe(url, ingredients):
    return {"recipe_url": url, "title": url, "ingredient_names": ingredients}


def _persona(dislikes):
    return {"ingredients": {"dislikes": list(dislikes)}}


# ── FIX 1: enforce_allergen_safety semantics ─────────────────────────────────

def test_drops_slot_with_disliked_ingredient():
    plan = {"breakfast": _recipe("b", ["egg", "onion"]),
            "lunch": _recipe("l", ["rice", "chicken"])}
    persona = _persona(["onion"])
    safe = constraints.enforce_allergen_safety(plan, persona)
    assert "breakfast" not in safe          # violating slot dropped
    assert "lunch" in safe                  # safe slot kept
    assert safe["lunch"] is plan["lunch"]


def test_persona_without_dislikes_leaves_plan_unchanged():
    plan = {"breakfast": _recipe("b", ["egg"]), "lunch": _recipe("l", ["rice"])}
    safe = constraints.enforce_allergen_safety(plan, _persona([]))
    assert safe == plan
    # a NEW dict is returned (input not mutated)
    assert safe is not plan


def test_empty_slot_kept_as_is():
    # Documented choice: an empty slot ({}) is falsy → treated as "nothing to filter"
    # and kept as-is (the filter only removes UNSAFE non-empty recipes).
    plan = {"breakfast": {}, "lunch": _recipe("l", ["rice"])}
    safe = constraints.enforce_allergen_safety(plan, _persona(["onion"]))
    assert "breakfast" in safe and safe["breakfast"] == {}
    assert "lunch" in safe


def test_recipe_missing_ingredient_names_treated_safe():
    plan = {"breakfast": {"recipe_url": "b", "title": "b"}}  # no ingredient_names
    safe = constraints.enforce_allergen_safety(plan, _persona(["onion"]))
    assert "breakfast" in safe               # kept (treated safe)


def test_does_not_mutate_input():
    plan = {"breakfast": _recipe("b", ["onion"])}
    original_keys = set(plan.keys())
    constraints.enforce_allergen_safety(plan, _persona(["onion"]))
    assert set(plan.keys()) == original_keys  # original untouched


def test_q01495_green_onion_case_is_filtered():
    """Reproduce the error-analysis q_01495 case: a persona that blacklists
    'green onions' and a recipe whose ingredient_names include it → dropped."""
    persona = _persona(["green onions"])
    plan = {"dinner": _recipe("green-onion-stirfry",
                              ["green onions", "soy sauce", "garlic"]),
            "lunch": _recipe("safe-salad", ["lettuce", "tomato"])}
    safe = constraints.enforce_allergen_safety(plan, persona)
    assert "dinner" not in safe, "green-onion recipe must be filtered out"
    assert "lunch" in safe


# ── FIX 2: ExplanationTool empty-plan gate ───────────────────────────────────

class _RecordingLLM:
    """Minimal LLM stand-in that records whether .complete was invoked."""

    def __init__(self, text="LLM explanation."):
        self.text = text
        self.complete_called = False

    def complete(self, prompt):
        self.complete_called = True

        class _Resp:
            pass

        r = _Resp()
        r.text = self.text
        return r


def _ctx_with_plan(meal_plan_dict, substitution_dict=None):
    """Build a real (dummy-workflow) context seeded with a plan (and optional
    substitution) so ExplanationTool.explain can read/write it."""
    state = get_dummy_state()
    state[RecipeState.MEAL_PLAN_RECOMMENDATION_KEY] = MealPlan.from_dict(meal_plan_dict)
    if substitution_dict is not None:
        state[RecipeState.SUBSTITUTED_MEAL_PLAN_RECOMMENDATION_KEY] = \
            MealPlanSubstitution.from_dict(substitution_dict)
    state.save_to_context()
    return state.get_context()


def test_empty_plan_gate_skips_llm_and_returns_fixed_message():
    llm = _RecordingLLM()
    tool = ExplanationTool(llm=llm, hummus_data=None, verbose=False)
    # No substitution, and a meal plan whose only slot maps to an empty recipe → empty.
    ctx = _ctx_with_plan({"breakfast": {}})
    result = tool.explain(ctx=ctx)
    assert result == ExplanationTool.EMPTY_PLAN_MESSAGE
    assert llm.complete_called is False, "LLM must NOT be called for an empty plan"
    # persisted
    st = RecipeState.from_context(ctx)
    assert st.get_meal_plan_explanation() == ExplanationTool.EMPTY_PLAN_MESSAGE


def test_completely_empty_plan_gate():
    llm = _RecordingLLM()
    tool = ExplanationTool(llm=llm, hummus_data=None, verbose=False)
    ctx = _ctx_with_plan({})  # no slots at all
    result = tool.explain(ctx=ctx)
    assert result == ExplanationTool.EMPTY_PLAN_MESSAGE
    assert llm.complete_called is False


def test_non_empty_plan_runs_llm_path():
    llm = _RecordingLLM(text="This plan is great.")
    tool = ExplanationTool(llm=llm, hummus_data=None, verbose=False)
    ctx = _ctx_with_plan({"breakfast": {"recipe_url": "b", "title": "Eggs"}})
    result = tool.explain(ctx=ctx)
    assert llm.complete_called is True, "LLM must run for a non-empty plan"
    assert result == "This plan is great."


def test_is_empty_recommendation_helper():
    assert ExplanationTool._is_empty_recommendation(MealPlan.from_dict({})) is True
    assert ExplanationTool._is_empty_recommendation(MealPlan.from_dict({"breakfast": {}})) is True
    assert ExplanationTool._is_empty_recommendation(
        MealPlan.from_dict({"breakfast": {"title": "Eggs"}})) is False
    assert ExplanationTool._is_empty_recommendation(MealPlanSubstitution.from_dict({})) is True
    assert ExplanationTool._is_empty_recommendation(
        MealPlanSubstitution.from_dict({"breakfast": "swap eggs"})) is False

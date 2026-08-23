"""Unit tests for the FULL-DAY meal-plan mode.

The default meal-plan tool reformulates the query into meal-plan questions for only
the meal types "relevant to the current time" (partial-day, mean ~2.3 meals). When the
tool is constructed with ``full_day=True`` it must instead request a meal-plan question
for ALL FOUR meal types (breakfast/lunch/dinner/snack), regardless of the current time,
so the pipeline can produce a complete 4-slot plan.

These tests mock the LLM to return a fixed meal-plan-questions JSON, so no real
API / hummus / retriever is needed (memory-light).
"""
from __future__ import annotations

import pytest

from foodrag.food_rag.agents.workflow_runner import get_dummy_state
from foodrag.food_rag.meal_plan.meal_type import MealType
from foodrag.food_rag.tools.meal_plan_tools import (
    FULL_DAY_QUERY_AUGMENTATION_TEMPLATE, QUERY_AUGMENTATION_TEMPLATE,
    MealPlanRecommendationTool)
from foodrag.food_rag.utils.recipe_context import RecipeState

pytestmark = pytest.mark.unit


class _FakeCompletion:
    def __init__(self, text):
        self.text = text
        self.additional_kwargs = {}


class _RecordingLLM:
    """Records every prompt it is asked to complete and returns a fixed meal-plan JSON."""

    def __init__(self, response_text):
        self._response_text = response_text
        self.prompts = []

    def complete(self, prompt):
        self.prompts.append(str(prompt))
        return _FakeCompletion(self._response_text)


class _StubQueryEngine:
    """Just enough surface for MealPlanRecommendationTool.__init__ (JsonExtractorTool + llm)."""

    def __init__(self, llm):
        self.llm = llm
        self.llm_calls = 1


_ALL_FOUR_JSON = ('```json\n{"breakfast": "vegan oats", "lunch": "vegan salad", '
                  '"dinner": "vegan curry", "snack": "fruit"}\n```')
_PARTIAL_JSON = '```json\n{"breakfast": "oats", "lunch": "salad"}\n```'


# ── (a) full_day=True → all four meal types requested in prompt AND parsed out ─

def test_full_day_selects_full_day_template():
    llm = _RecordingLLM(_ALL_FOUR_JSON)
    tool = MealPlanRecommendationTool(llm=llm, recipe_query_engine=_StubQueryEngine(llm),
                                      retrieve_only=True, full_day=True, verbose=False)
    assert tool.full_day is True
    assert tool.query_augmentation_template is FULL_DAY_QUERY_AUGMENTATION_TEMPLATE


def test_full_day_prompt_requests_all_four_meal_types():
    llm = _RecordingLLM(_ALL_FOUR_JSON)
    tool = MealPlanRecommendationTool(llm=llm, recipe_query_engine=_StubQueryEngine(llm),
                                      retrieve_only=True, full_day=True,
                                      time="12:00", date="2026-06-28", verbose=False)
    state = get_dummy_state()
    ctx = state.get_context()
    tool.get_meal_plan_question(ctx=ctx, question="a healthy day of vegan meals")

    assert llm.prompts, "the reformulation LLM was never called"
    prompt = llm.prompts[0]
    # The prompt instructs the model to cover EVERY meal type and to ignore the time.
    assert "EVERY meal type" in prompt
    assert "IGNORE the current time" in prompt
    for mt in MealType.get_all_str():
        assert mt in prompt


def test_full_day_yields_meal_plan_questions_with_all_four_keys():
    llm = _RecordingLLM(_ALL_FOUR_JSON)
    tool = MealPlanRecommendationTool(llm=llm, recipe_query_engine=_StubQueryEngine(llm),
                                      retrieve_only=True, full_day=True,
                                      time="00:01", date="2026-06-28", verbose=False)
    state = get_dummy_state()
    ctx = state.get_context()
    tool.get_meal_plan_question(ctx=ctx, question="a healthy day of vegan meals")

    final = RecipeState.from_context(ctx)
    mpq = final.get_meal_plan_questions()
    keys = {str(k) for k in mpq.keys()}
    assert keys == set(MealType.get_all_str())  # all four slots
    assert len(mpq) == 4
    # Resolved recommendation time is still captured (reproducibility).
    assert final.get_recommendation_time() == "2026-06-28 00:01"


# ── (b) full_day=False unchanged (time-based partial-day) ─────────────────────

def test_partial_day_is_default_and_uses_time_template():
    llm = _RecordingLLM(_PARTIAL_JSON)
    tool = MealPlanRecommendationTool(llm=llm, recipe_query_engine=_StubQueryEngine(llm),
                                      retrieve_only=True, verbose=False)
    assert tool.full_day is False
    assert tool.query_augmentation_template is QUERY_AUGMENTATION_TEMPLATE


def test_partial_day_prompt_is_time_based_not_full_day():
    llm = _RecordingLLM(_PARTIAL_JSON)
    tool = MealPlanRecommendationTool(llm=llm, recipe_query_engine=_StubQueryEngine(llm),
                                      retrieve_only=True, time="08:00", date="2026-06-28",
                                      verbose=False)
    state = get_dummy_state()
    ctx = state.get_context()
    tool.get_meal_plan_question(ctx=ctx, question="q")

    prompt = llm.prompts[0]
    # The default (partial-day) template keys off the current time and does NOT force all four.
    assert "relevant to the current time" in prompt
    assert "EVERY meal type" not in prompt
    assert "IGNORE the current time" not in prompt

    # Only the two returned meal types are parsed out (unchanged, time-based behaviour).
    final = RecipeState.from_context(ctx)
    keys = {str(k) for k in final.get_meal_plan_questions().keys()}
    assert keys == {"breakfast", "lunch"}


def test_explicit_template_overrides_full_day():
    """An explicitly-passed template wins over the full_day default (documented behaviour)."""
    custom = QUERY_AUGMENTATION_TEMPLATE
    llm = _RecordingLLM(_PARTIAL_JSON)
    tool = MealPlanRecommendationTool(llm=llm, recipe_query_engine=_StubQueryEngine(llm),
                                      retrieve_only=True, full_day=True,
                                      query_augmentation_template=custom, verbose=False)
    assert tool.query_augmentation_template is custom

"""Unit tests for pinning + persisting the meal-plan recommendation time.

The goal of the feature is reproducibility: the meal-plan tool resolves a "current
time/date" that decides which meal types are requested. We now CAPTURE the resolved
value into RecipeState (RECOMMENDATION_TIME_KEY) and carry it — plus the requested
meal types (the keys of MEAL_PLAN_QUESTIONS_KEY) — onto AssembledPlan and into the
per-question record.

These tests mock the LLM to return a fixed meal-plan-questions JSON, so no real
API / hummus / retriever is needed (memory-light).
"""
from __future__ import annotations

import pandas as pd
import pytest

from foodrag.baselines.full_pipeline import (AssembledPlan, MealPlanAssembler,
                                             WorkflowMealPlanAssembler, _requested_meal_types,
                                             run_full_pipeline)
from foodrag.food_rag.agents.workflow_runner import get_dummy_state
from foodrag.food_rag.meal_plan.meal_plan_structures import MealPlanQuestions
from foodrag.food_rag.tools.meal_plan_tools import MealPlanRecommendationTool
from foodrag.food_rag.utils.recipe_context import RecipeState
from foodrag.baselines.judge import Judge

pytestmark = pytest.mark.unit


# ── (a) a pinned time flows into the prompt and is captured in state ─────────

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


_MEAL_JSON = '```json\n{"breakfast": "oats", "lunch": "salad"}\n```'


def test_pinned_time_flows_into_prompt_and_is_captured_in_state():
    llm = _RecordingLLM(_MEAL_JSON)
    qe = _StubQueryEngine(llm)
    tool = MealPlanRecommendationTool(llm=llm, recipe_query_engine=qe, retrieve_only=True,
                                      time="00:01", date="2026-06-28", verbose=False)

    state = get_dummy_state()
    ctx = state.get_context()
    tool.get_meal_plan_question(ctx=ctx, question="a healthy day of vegan meals")

    # The pinned time+date reached the reformulation prompt.
    assert any("00:01" in p for p in llm.prompts)
    assert any("2026-06-28" in p for p in llm.prompts)

    # The resolved "{date} {time}" was captured + persisted into the state.
    final = RecipeState.from_context(ctx)
    assert final.get_recommendation_time() == "2026-06-28 00:01"


def test_now_time_is_resolved_and_captured_not_left_as_now():
    llm = _RecordingLLM(_MEAL_JSON)
    qe = _StubQueryEngine(llm)
    tool = MealPlanRecommendationTool(llm=llm, recipe_query_engine=qe, retrieve_only=True,
                                      time="now", date="today", verbose=False)
    state = get_dummy_state()
    ctx = state.get_context()
    tool.get_meal_plan_question(ctx=ctx, question="q")

    captured = RecipeState.from_context(ctx).get_recommendation_time()
    # Resolved to a concrete "YYYY-MM-DD HH:MM", never the literal placeholders.
    assert captured and "now" not in captured.lower() and "today" not in captured.lower()
    assert len(captured.split(" ")) == 2


# ── (b) requested_meal_types is populated from MEAL_PLAN_QUESTIONS_KEY ────────

def test_requested_meal_types_from_meal_plan_questions_key():
    llm = _RecordingLLM(_MEAL_JSON)
    qe = _StubQueryEngine(llm)
    tool = MealPlanRecommendationTool(llm=llm, recipe_query_engine=qe, retrieve_only=True,
                                      time="00:01", date="2026-06-28", verbose=False)
    state = get_dummy_state()
    ctx = state.get_context()
    tool.get_meal_plan_question(ctx=ctx, question="q")

    final = RecipeState.from_context(ctx)
    assert _requested_meal_types(final.get_meal_plan_questions()) == ["breakfast", "lunch"]


def test_requested_meal_types_helper_handles_empty_and_dict():
    assert _requested_meal_types(None) == []
    assert _requested_meal_types(MealPlanQuestions()) == []
    assert _requested_meal_types({"dinner": "q", "snack": "q"}) == ["dinner", "snack"]


# ── (c) AssembledPlan / record carry both fields ─────────────────────────────

def test_workflow_assembler_carries_time_and_meals_from_state():
    state = {
        "meal_plan_recommendation": {"breakfast": {"recipe_url": "u1", "title": "Oats"}},
        "meal_plan_questions": MealPlanQuestions.from_dict({"breakfast": "oats", "lunch": "salad"}),
        "recommendation_time": "2026-06-28 00:01",
        "meal_plan_explanation": "Vegan and balanced.",
    }
    asm = WorkflowMealPlanAssembler(agent_workflow=None, retriever=object(),
                                    config_id="tog", run_workflow=lambda w, q, m: (state, 1.0))
    plan = asm.assemble({"question": "q"})
    assert plan.recommendation_time == "2026-06-28 00:01"
    assert plan.requested_meal_types == ["breakfast", "lunch"]


def test_workflow_assembler_defaults_when_state_lacks_time():
    state = {"meal_plan_recommendation": {}, "meal_plan_explanation": "Not written yet."}
    asm = WorkflowMealPlanAssembler(agent_workflow=None, retriever=object(),
                                    config_id="rog", run_workflow=lambda w, q, m: (state, 0.0))
    plan = asm.assemble({"question": "q"})
    assert plan.recommendation_time == ""
    assert plan.requested_meal_types == []


class _FixedAssembler(MealPlanAssembler):
    config_id = "tog"
    require_explanation = False

    def assemble(self, qa_item):
        return AssembledPlan(
            question=qa_item.get("question", ""), answer_text="plan text",
            plan={"breakfast": {"recipe_url": "u1", "title": "Veg Bowl",
                                "ingredient_names": ["spinach"], "tags": ["vegan"],
                                "servingSize [g]": 100, "calories [kcal]": 200}},
            recipe_urls=["u1"], contexts=["r1"],
            recommendation_time="2026-06-28 00:01",
            requested_meal_types=["breakfast", "lunch"])


def test_record_carries_recommendation_time_and_meals():
    recipes_df = pd.DataFrame([
        {"recipe_id": 1, "recipe_url": "u1", "title": "Veg Bowl", "tags": ["vegan"],
         "ingredient_names": ["spinach"], "servingSize [g]": 100, "calories [kcal]": 200,
         "sugars [g]": 5, "nutri_score": 0.9, "fsa_score": 0.8, "average_rating": 4.9,
         "number_of_ratings": 10},
    ])
    qa_item = {"question": "vegan", "answer": ["u1"],
               "persona": {"tags": {"likes": ["vegan"], "dislikes": []},
                           "ingredients": {"likes": [], "dislikes": []},
                           "relative_nutrients": {}, "total_nutrients": {}, "nutrition_scores": {}}}
    records = run_full_pipeline(_FixedAssembler(), [qa_item], recipes_df,
                               judges=[Judge("judgeA", None)], k=10)
    rec = records[0]
    assert rec["recommendation_time"] == "2026-06-28 00:01"
    assert rec["requested_meal_types"] == ["breakfast", "lunch"]

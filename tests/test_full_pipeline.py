"""Unit tests for the Use-2 full-pipeline driver (§5.7, §4.4–4.5, §7).

The assembly layer and judges need a live LLM/graph, so they are faked here: a
``FakeAssembler`` returns a fixed meal plan and a ``FakeJudge`` returns canned
verdicts. That isolates and pins down the deterministic meal-plan metrics, the
judge-metric shaping (dual-judge key naming + 1–5 → [0,1]) and the §13 record the
orchestrator emits — none of which need real models.
"""
from __future__ import annotations

import math

import pandas as pd
import pytest

from foodrag.baselines import meal_plan_metrics as mpm
from foodrag.baselines.full_pipeline import (AssembledPlan, MealPlanAssembler,
                                             WorkflowMealPlanAssembler, run_full_pipeline)
from foodrag.baselines.judge import Judge, judge_plan

pytestmark = pytest.mark.unit


# ── deterministic meal-plan metrics (§4.4–4.5) ───────────────────────────────

def _recipe(ingredients, tags, cal, serving=100):
    return {"recipe_url": "u", "ingredient_names": ingredients, "tags": tags,
            "calories [kcal]": cal, "servingSize [g]": serving}


def test_meal_slot_completeness_counts_filled_slots():
    plan = {"breakfast": _recipe(["egg"], [], 200), "lunch": None,
            "dinner": _recipe(["rice"], [], 200), "snack": None}
    assert mpm.meal_slot_completeness(plan) == 0.5


def test_intra_plan_diversity_rewards_distinct_meals():
    same = {"breakfast": _recipe(["egg"], ["a"], 100), "lunch": _recipe(["egg"], ["a"], 100)}
    diff = {"breakfast": _recipe(["egg"], ["a"], 100), "lunch": _recipe(["rice"], ["b"], 100)}
    assert mpm.intra_plan_diversity(same) == pytest.approx(0.0)   # identical → no diversity
    assert mpm.intra_plan_diversity(diff) == pytest.approx(1.0)   # disjoint → full diversity
    assert math.isnan(mpm.intra_plan_diversity({"breakfast": _recipe(["egg"], [], 100)}))


def test_calorie_adherence_perfect_split_scores_one():
    # calories per serving = value/100*serving; serving=100 → equals the value.
    plan = {"breakfast": _recipe([], [], 22), "lunch": _recipe([], [], 32),
            "dinner": _recipe([], [], 32), "snack": _recipe([], [], 14)}
    assert mpm.calorie_distribution_adherence(plan) == pytest.approx(1.0)


def test_structural_validity_requires_slots_and_explanation():
    full = {s: _recipe([], [], 1) for s in mpm.STANDARD_SLOTS}
    assert mpm.structural_validity(full, has_explanation=False, require_explanation=False) == 1.0
    assert mpm.structural_validity(full, has_explanation=False, require_explanation=True) == 0.0
    assert mpm.structural_validity({"breakfast": _recipe([], [], 1)}) == 0.0  # missing slots


# ── dual-judge shaping (§7) ──────────────────────────────────────────────────

class FakeJudgeLLM:
    """Returns a fixed completion text regardless of prompt."""

    def __init__(self, text):
        self._text = text

    def complete(self, prompt):
        return type("R", (), {"text": self._text})()


def test_judge_plan_emits_dual_judge_keys_and_scales():
    judges = [Judge("judgeA", FakeJudgeLLM('{"passing": true}')),
              Judge("judgeB", FakeJudgeLLM('{"score": 5}'))]
    out = judge_plan(judges, question="q", answer="a", contexts=["r1"], gold=["g"],
                     require_explanation=True)
    # judgeA answers the binary metrics true, the scaled ones unparsable → NaN
    assert out["faithfulness_judgeA"] == 1.0
    assert math.isnan(out["correctness_judgeA"])
    # judgeB answers score=5 → scaled to 1.0 for correctness/explanation
    assert out["correctness_judgeB"] == pytest.approx(1.0)
    assert out["explanation_quality_judgeB"] == pytest.approx(1.0)
    assert math.isnan(out["faithfulness_judgeB"])  # "score" payload isn't a passing verdict


def test_judge_without_llm_is_nan():
    out = judge_plan([Judge("judgeA", None)], question="q", answer="a", contexts=[])
    assert all(math.isnan(v) for v in out.values())


# ── orchestrator: assemble → metrics → §13 record ────────────────────────────

class FakeAssembler(MealPlanAssembler):
    config_id = "tog"
    require_explanation = False

    def __init__(self, plan, urls):
        self._plan, self._urls = plan, urls

    def assemble(self, qa_item):
        return AssembledPlan(question=qa_item.get("question", ""), answer_text="plan text",
                             plan=self._plan, recipe_urls=self._urls, contexts=["r1"],
                             llm_usage=12, llm_seq_usage=9, llm_parsing=0, latency_s=3.4)


@pytest.fixture
def recipes_df():
    return pd.DataFrame([
        {"recipe_id": 1, "recipe_url": "u1", "title": "Veg Bowl", "tags": ["vegan"],
         "ingredient_names": ["spinach", "rice"], "servingSize [g]": 100, "calories [kcal]": 200,
         "sugars [g]": 5, "nutri_score": 0.9, "fsa_score": 0.8, "average_rating": 4.9, "number_of_ratings": 10},
        {"recipe_id": 2, "recipe_url": "u2", "title": "Peanut Stew", "tags": ["vegan"],
         "ingredient_names": ["peanut"], "servingSize [g]": 100, "calories [kcal]": 300,
         "sugars [g]": 5, "nutri_score": 0.5, "fsa_score": 0.5, "average_rating": 4.0, "number_of_ratings": 5},
    ])


@pytest.fixture
def qa_item():
    return {"question": "vegan, no peanut", "answer": ["u1"],
            "persona": {"tags": {"likes": ["vegan"], "dislikes": []},
                        "ingredients": {"likes": [], "dislikes": ["peanut"]},
                        "relative_nutrients": {}, "total_nutrients": {}, "nutrition_scores": {}}}


def test_run_full_pipeline_emits_combined_record(recipes_df, qa_item):
    plan = {"breakfast": recipes_df.iloc[0].to_dict(), "lunch": recipes_df.iloc[1].to_dict(),
            "dinner": None, "snack": None}
    assembler = FakeAssembler(plan, urls=["u1", "u2"])
    judges = [Judge("judgeA", FakeJudgeLLM('{"passing": true}'))]

    records = run_full_pipeline(assembler, [qa_item], recipes_df, judges=judges, k=10,
                                experiment_id="EXP-7", rung="4")
    rec = records[0]
    assert rec["config_id"] == "tog" and rec["rung"] == "4" and rec["stratum"] == "tight"
    m = rec["metrics"]
    # deterministic safety on the assembled plan: u2 contains the peanut allergen
    assert m["allergen_violation"] == pytest.approx(0.5)
    # deterministic meal-plan quality present
    assert m["meal_slot_completeness"] == 0.5 and "intra_plan_diversity" in m
    # judge metric shaped + cost metrics carried through
    assert m["faithfulness_judgeA"] == 1.0
    assert m["llm_usage"] == 12 and m["latency_s"] == pytest.approx(3.4)
    assert rec["retrieved_ids"] == ["u1", "u2"]


# ── full agentic pipeline (Heuristic 1) state extraction ─────────────────────

def test_workflow_assembler_extracts_enhanced_plan(recipes_df, qa_item):
    # Final agent state: a recommended plan + an explanation, default final text.
    state = {
        "meal_plan_recommendation": {"breakfast": recipes_df.iloc[0].to_dict(),
                                     "lunch": None, "dinner": None, "snack": None},
        "meal_plan_explanation": "This plan is vegan and peanut-free.",
        "final_recommendation": "Final recommendation required.",  # default → fall back to plan
        "llm_usage_counter": 7, "llm_sequential_usage_counter": 5, "llm_parsing_usage_counter": 1,
    }

    def fake_run(workflow, question, memory):
        return state, 2.5

    asm = WorkflowMealPlanAssembler(agent_workflow=None, retriever=object(),
                                    config_id="tog", run_workflow=fake_run)
    plan = asm.assemble(qa_item)
    assert asm.require_explanation is True              # explanation is judged for Heuristic 1
    assert plan.recipe_urls == ["u1"]
    assert plan.has_explanation is True
    assert "vegan" in plan.answer_text and "Veg Bowl" in plan.answer_text
    assert plan.llm_usage == 7 and plan.latency_s == pytest.approx(2.5)


def test_workflow_assembler_detects_missing_explanation(qa_item):
    state = {"meal_plan_recommendation": {}, "meal_plan_explanation": "Not written yet."}

    asm = WorkflowMealPlanAssembler(agent_workflow=None, retriever=object(), config_id="rog",
                                    run_workflow=lambda w, q, m: (state, 0.0))
    plan = asm.assemble(qa_item)
    assert plan.has_explanation is False
    assert plan.recipe_urls == []


# ── turnkey agentic factory wiring ───────────────────────────────────────────

class _NullKGClient:
    def run(self, *a, **k):
        return []


def test_build_agentic_pipeline_invokes_builder_with_shared_retriever(recipes_df):
    from foodrag.baselines.kbqa import build_agentic_pipeline
    from foodrag.baselines.kbqa.kg import HummusKG

    captured = {}

    def fake_builder(query_engine, retriever):
        captured["query_engine"] = query_engine
        captured["retriever"] = retriever
        return "WORKFLOW"

    asm = build_agentic_pipeline("tfs_kbqa", HummusKG(_NullKGClient()), recipes_df,
                                 workflow_builder=fake_builder, llm=None)
    assert asm.config_id == "tfs_kbqa" and asm.agent_workflow == "WORKFLOW"
    # the retriever the assembler binds is the same one the workflow was built on
    assert asm.retriever is captured["retriever"]
    assert captured["query_engine"].retriever is captured["retriever"]


def test_build_agentic_pipeline_requires_hummus_data_for_turnkey(recipes_df):
    from foodrag.baselines.kbqa import build_agentic_pipeline
    from foodrag.baselines.kbqa.kg import HummusKG

    with pytest.raises(ValueError, match="hummus_data"):
        build_agentic_pipeline("tog", HummusKG(_NullKGClient()), recipes_df,
                               workflow_builder=None, hummus_data=None)

"""Build a MEAL_PLAN QA set paired to an existing RECIPE tight set.

The full-day FoodRAG runs impose four slots on ``data/qa/main_10k_tight`` — a set of
RECIPE questions that carry no per-slot budget. This script produces the matching
budget-constrained MEAL_PLAN set: it reuses the SAME personas (in the same order, so
``question_id`` still refers to the same persona) and, for each, builds the per-slot
calorie restriction and the gold meal-plan answer.

Reusing the personas rather than drawing fresh ones makes the two sets paired: the only
thing that changes between the recipe-framed and budget-framed run is the question, so a
before/after on any metric is attributable to the budget, not to persona variation.

The per-slot caps are anchored to each persona's daily budget
(``guideline_adherence.resolve_daily_energy`` — the persona's own stated cap when known,
else the class reference), so they sum to the daily total the question states.

    python scripts/gen_mealplan_tight.py \
        --src data/qa/main_10k_tight/recipe_question_answer_set.pkl \
        --out data/qa/main_10k_tight_mealplan/meal_plan_question_answer_set.pkl
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from foodrag.baselines.base import answer_set_stratum, gold_urls  # noqa: E402
from foodrag.persona import qa_generator as qg  # noqa: E402
from foodrag.persona.qa_generator import QuestionType, _generate_question, _meal_plan_restriction  # noqa: E402
from foodrag.utils import load_util, rdf_util  # noqa: E402

DEFAULT_SRC = "data/qa/main_10k_tight/recipe_question_answer_set.pkl"
DEFAULT_OUT = "data/qa/main_10k_tight_mealplan/meal_plan_question_answer_set.pkl"


def build(src: str, out: str, persona_config_path: str, data_path: str) -> dict:
    with open(src, "rb") as fh:
        recipe_qa = pickle.load(fh)
    print(f"source personas: {len(recipe_qa)} (from {src})")

    persona_config = load_util.get_config(persona_config_path)
    print("loading hummus + FoodOn (~2 min) ...")
    hummus = load_util.get_current_hummus(data_path)
    food_on = rdf_util.PreProcessedFoodOn(hummus)
    recipes_df = hummus["recipes_df"]

    out_set = []
    strata = {"empty": 0, "tight": 0, "medium": 0, "broad": 0}
    for i, item in enumerate(recipe_qa):
        persona = item["persona"]
        restriction = _meal_plan_restriction(persona, persona_config)
        question = _generate_question(persona, restriction, QuestionType.MEAL_PLAN)
        answer = qg._generate_valid_meal_plans_for_persona(
            food_on, persona_config, persona, restriction, recipes_df)
        rec = {"question": question, "llm_questions": [], "answer": answer,
               "persona": persona, "meal_plan": restriction}
        out_set.append(rec)
        strata[answer_set_stratum(len(gold_urls(rec)))] += 1
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(recipe_qa)}  strata so far: {strata}")

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as fh:
        pickle.dump(out_set, fh)
    # Mirror the generator's stage marker so downstream loaders treat it as complete.
    (out_path.parent / "stage_meal_plan.json").write_text('"done"', encoding="utf-8")

    print(f"\nwrote {len(out_set)} meal-plan QA items -> {out_path}")
    print(f"final strata: {strata}")
    return {"n": len(out_set), "strata": strata, "out": str(out_path)}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build a MEAL_PLAN QA set paired to a RECIPE tight set.")
    p.add_argument("--src", default=DEFAULT_SRC)
    p.add_argument("--out", default=DEFAULT_OUT)
    p.add_argument("--persona-config", default="config/persona_config.yml")
    p.add_argument("--data-path", default="data")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    build(args.src, args.out, args.persona_config, args.data_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

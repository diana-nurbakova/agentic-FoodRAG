"""``generate_meal_plan`` — the single entrypoint the GUI calls (spec §11.5).

Builds a natural-language question from the persona + free-text query, runs the
agentic pipeline, and maps the resulting RecipeState into the GUI plan shape:

    {
      "recipes":        [ {meal, ...card fields}, ... ],   # Breakfast..Snack
      "correction_log": [ {meal, round, status, reason, action}, ... ],
      "splits":         [ {meal, q}, ... ],                 # per-meal query split
      "explanation":    str,
      "llm_calls":      int,
      "elapsed":        float,
    }
"""

from __future__ import annotations

import os
import time
from typing import Any

from . import backend, eval_metrics, mapping


def real_enabled() -> bool:
    """True if the live backend should be used (env override or admin toggle)."""
    if os.environ.get("FOODRAG_REAL_BACKEND", "").lower() in ("1", "true", "yes", "on"):
        return True
    try:
        import streamlit as st

        cfg = st.session_state.get("admin_config", {})
        return bool(cfg.get("pipeline_defaults", {}).get("use_real_backend"))
    except Exception:  # noqa: BLE001
        return False


def build_question(persona: dict, query: str = "") -> str:
    """Compose a meal-plan request from the persona constraints + free text."""
    parts: list[str] = []
    cond = persona.get("condition")
    if cond and cond != "None":
        parts.append(f"I have {cond}.")
    allergies = persona.get("allergies") or []
    if allergies:
        parts.append("I am allergic to " + ", ".join(allergies) + ".")
    liked = persona.get("likedTags") or []
    if liked:
        parts.append("I like " + ", ".join(liked) + ".")
    disliked = (persona.get("dislikedIng") or []) + (persona.get("dislikedTags") or [])
    if disliked:
        parts.append("I dislike " + ", ".join(disliked) + ".")
    nc = persona.get("nutrientConstraint")
    if nc and nc != "None":
        parts.append(f"Please respect this nutritional constraint: {nc}.")

    base = query.strip() if query and query.strip() else (
        "Suggest a full day meal plan with a breakfast, lunch, dinner and a snack."
    )
    return (base + " " + " ".join(parts)).strip()


def _coerce_meal_map(value: Any) -> dict:
    """Coerce a MealPlan/MealPlanValidation/... (or its dict) to {meal_str: value}."""
    if value is None:
        return {}
    if hasattr(value, "to_dict"):
        try:
            value = value.to_dict()
        except Exception:  # noqa: BLE001
            return {}
    if not hasattr(value, "items"):
        return {}
    out = {}
    for k, v in value.items():
        key = k.value if hasattr(k, "value") else str(k)
        out[key.lower()] = v
    return out


def generate_meal_plan(persona: dict, query: str, config: dict) -> dict:
    """Run the live pipeline and return the GUI plan result. Raises on failure
    (callers fall back to mock)."""
    from foodrag.food_rag.utils.recipe_context import RecipeState

    question = build_question(persona, query)
    retriever = (config or {}).get("retriever", "text-to-pandas")
    t0 = time.time()
    state = backend.run_pipeline(question, retriever=retriever)
    elapsed = time.time() - t0

    meal_plan = _coerce_meal_map(state.get(RecipeState.MEAL_PLAN_RECOMMENDATION_KEY))
    validation = _coerce_meal_map(state.get(RecipeState.MEAL_PLAN_VALIDATION_KEY))
    subs = _coerce_meal_map(state.get(RecipeState.SUBSTITUTED_MEAL_PLAN_RECOMMENDATION_KEY))
    questions = _coerce_meal_map(state.get(RecipeState.MEAL_PLAN_QUESTIONS_KEY))

    explanation = state.get(RecipeState.MEAL_PLAN_EXPLANATION_KEY) or ""
    if not explanation or explanation == "Not written yet.":
        explanation = state.get(RecipeState.FINAL_RECOMMENDATION_KEY) or ""

    cards = mapping.meal_plan_to_recipes(meal_plan)
    safety = _safety_check(persona, meal_plan, cards)

    return {
        "recipes": cards,
        "safety": safety,
        "correction_log": mapping.validation_to_log(validation, subs),
        "splits": mapping.questions_to_splits(questions),
        "explanation": explanation,
        "overall_validation": state.get(RecipeState.OVERALL_MEAL_PLAN_VALIDATION_KEY, ""),
        "llm_calls": int(state.get(RecipeState.LLM_USAGE_COUNTER_KEY, 0) or 0),
        "elapsed": round(elapsed, 1),
        "question": question,
    }


def _safety_check(persona: dict, meal_plan: dict, cards: list[dict]) -> dict | None:
    """Deterministic FoodOn-aware safety check on the generated plan (protocol §4.3).

    Runs on the RAW backend recipes (HUMMUS columns), attaches a per-card
    ``safety`` flag (aligned to ``cards``), and returns the aggregate report.
    Best-effort: never raises into the result.
    """
    try:
        blacklist = list(persona.get("allergies") or [])
        constraints = eval_metrics.parse_nutrient_constraints(persona.get("nutrientConstraint", ""))
        try:
            expand = eval_metrics.foodon_expander(backend.get_engine().hummus_data)
        except Exception:  # noqa: BLE001
            expand = None
        raw_aligned = [meal_plan.get(card["meal"].lower(), {}) for card in cards]
        report = eval_metrics.safety_report(raw_aligned, blacklist, constraints, expand)
        for card, pr in zip(cards, report["per_recipe"]):
            card["safety"] = {"ok": pr["ok"], "allergens": pr["allergens"],
                              "constraints": pr["constraints"]}
        report["foodon"] = expand is not None
        return report
    except Exception:  # noqa: BLE001
        return None

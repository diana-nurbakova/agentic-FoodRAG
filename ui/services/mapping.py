"""Convert backend results into the GUI's recipe/plan shapes.

Pure functions (no Streamlit / no backend imports) so they're unit-testable.
The backend recipe dicts use HUMMUS column names (e.g. ``calories [kcal]``,
``nutri_score`` as a 0–1 float); the GUI recipe-card shape uses the keys defined
in ``ui.mock_data`` (``cal``, ``nutriscore`` as an A–E letter, etc.).
"""

from __future__ import annotations

import re
from typing import Any

# Meal-type → emoji (the GUI cards show an emoji; backend recipes have none)
_MEAL_EMOJI = {"breakfast": "🥣", "lunch": "🥗", "dinner": "🍽", "snack": "🍎"}


def _num(recipe: dict, *keys: str, default: float = 0.0) -> float:
    """First present numeric column among ``keys`` (handles the ``x [unit]`` names)."""
    for k in keys:
        v = recipe.get(k)
        if v is not None and v != "":
            try:
                return float(v)
            except (TypeError, ValueError):
                continue
    return default


def nutri_letter(score: float) -> str:
    """Map the normalised nutri_score (A→1, B→.75, C→.5, D→.25, E→0) to a letter."""
    try:
        s = float(score)
    except (TypeError, ValueError):
        return "C"
    if s >= 0.875:
        return "A"
    if s >= 0.625:
        return "B"
    if s >= 0.375:
        return "C"
    if s >= 0.125:
        return "D"
    return "E"


def _who_10(who_score: float) -> int:
    """WHO score (0–1, higher healthier) → 0–10 scale used by the card."""
    return max(0, min(10, round(who_score * 10)))


def _fsa_points(fsa_score: float) -> int:
    """FSA: GUI shows traffic-light points where LOWER = healthier. The HUMMUS
    ``fsa_score`` is normalised 0–1 higher=healthier, so invert to points."""
    return max(0, min(10, round((1 - fsa_score) * 10)))


def _skill_from_duration(minutes: float) -> str:
    if minutes <= 15:
        return "Easy"
    if minutes <= 35:
        return "Medium"
    return "Hard"


def _split_csv(value: Any) -> list[str]:
    if not value:
        return []
    if isinstance(value, list):
        items = value
    else:
        items = re.split(r"[;,]", str(value))
    return [s.strip() for s in items if s and s.strip()]


def _split_steps(directions: Any) -> list[str]:
    if not directions:
        return []
    if isinstance(directions, list):
        steps = directions
    else:
        # Split on sentence boundaries; keep it readable
        steps = re.split(r"(?<=[.!?])\s+", str(directions).strip())
    return [s.strip() for s in steps if s and s.strip()]


def recipe_to_card(recipe: dict, meal_type: str = "") -> dict:
    """Map one backend recipe dict to the GUI recipe-card shape (``ui.mock_data``)."""
    duration = _num(recipe, "duration [min]", "duration", default=0)
    ingredients = _split_csv(recipe.get("ingredient_quantities")) or _split_csv(recipe.get("ingredients"))
    return {
        "name": str(recipe.get("title", "Untitled recipe")),
        "recipe_id": recipe.get("recipe_id"),
        "time": f"{int(duration)}min" if duration else "—",
        "nutriscore": nutri_letter(recipe.get("nutri_score", 0.5)),
        "who": _who_10(_num(recipe, "who_score", default=0.5)),
        "fsa": _fsa_points(_num(recipe, "fsa_score", default=0.5)),
        "cal": round(_num(recipe, "calories [kcal]", "calories")),
        "protein": round(_num(recipe, "protein [g]", "protein")),
        "carbs": round(_num(recipe, "totalCarbohydrate [g]", "carbs")),
        "fat": round(_num(recipe, "totalFat [g]", "fat")),
        "fiber": round(_num(recipe, "dietaryFiber [g]", "fiber")),
        "sugar": round(_num(recipe, "sugars [g]", "sugar")),
        "sodium": round(_num(recipe, "sodium [mg]", "sodium")),
        "img": _MEAL_EMOJI.get(meal_type.lower(), "🍽"),
        "skill": _skill_from_duration(duration),
        "servings": int(_num(recipe, "serves", default=2)) or 2,
        "tags": _split_csv(recipe.get("tags"))[:3],
        "ingredients": ingredients,
        "steps": _split_steps(recipe.get("directions")),
        "substitutions": [],  # structured per-ingredient subs not exposed by backend yet
    }


# GUI meal order (the cards iterate this order)
GUI_MEALS = ["Breakfast", "Lunch", "Dinner", "Snack"]


def meal_plan_to_recipes(meal_plan: dict) -> list[dict]:
    """Ordered list of GUI recipe cards for Breakfast/Lunch/Dinner/Snack.

    ``meal_plan`` maps meal-type string → backend recipe dict
    (``MealPlan.to_dict()``). Missing meals are skipped.
    """
    cards = []
    for meal in GUI_MEALS:
        recipe = meal_plan.get(meal.lower())
        if recipe:
            cards.append({"meal": meal, **recipe_to_card(recipe, meal)})
    return cards


def _status_from_validation(text: str) -> tuple[str, str]:
    """('OK'|'INVALID', short reason) parsed from a validation string."""
    t = (text or "").strip()
    status = "INVALID" if t.upper().startswith("INVALID") else "OK"
    reason = re.sub(r"^(IN)?VALID[:\s]*", "", t, flags=re.IGNORECASE).strip()
    reason = re.sub(r"\s+", " ", reason)
    if len(reason) > 220:
        reason = reason[:217] + "…"
    return status, reason or "—"


def validation_to_log(validation: dict, substitutions: dict | None = None) -> list[dict]:
    """Per-meal correction-log rows for the GUI ``correction_log`` component."""
    subs = substitutions or {}
    rows = []
    for meal, text in (validation or {}).items():
        status, reason = _status_from_validation(text)
        action = ""
        sub_text = subs.get(meal)
        if sub_text:
            action = re.sub(r"\s+", " ", str(sub_text)).strip()[:200]
        rows.append({"meal": meal.capitalize(), "round": 1, "status": status,
                     "reason": reason, "action": action})
    return rows


def questions_to_splits(questions: dict) -> list[dict]:
    """Per-meal query split rows for the query inspector."""
    return [{"meal": meal.capitalize(), "q": q} for meal, q in (questions or {}).items()]

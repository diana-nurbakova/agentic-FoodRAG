"""Deterministic meal-plan quality metrics (Evaluation Protocol §4.4–§4.5).

The Use-2 full-pipeline (EXP-7 rung 4 / EXP-3) judges an *assembled meal plan*,
not a flat ranked list. Three of its quality signals are deterministic — computed
straight off the plan's recipes + the HUMMUS attributes — and so live here next to
the other rule-based metrics rather than going through an LLM judge:

* **Structural Validity** (§4.4) — does the output parse as a meal plan with the
  required slots (and an explanation field, when the config is expected to emit
  one)? A schema check, not an LLM call.
* **Intra-Plan Diversity** (§4.5) — 1 − mean pairwise ingredient/tag Jaccard
  across the meals (penalises four-near-identical-dishes plans).
* **Calorie-Distribution Adherence** (§4.5) — 1 − mean abs. deviation from the
  target split (Breakfast 22% / Lunch 32% / Dinner 32% / Snack 14%).
* **Meal-Slot Completeness** (§4.5) — fraction of the four required slots filled.

A plan is passed in normalised form: ``{meal_type_str: recipe_row | None}`` where a
recipe row is a mapping of HUMMUS columns (``ingredient_names``, ``tags``,
``calories [kcal]``, ``servingSize [g]``, …) and ``None`` marks an unfilled slot.
"""
from __future__ import annotations

from itertools import combinations
from typing import Mapping, Optional

# §4.5 calorie target split for adults 19–59 (thesis Table 6.3).
CALORIE_TARGETS = {"breakfast": 0.22, "lunch": 0.32, "dinner": 0.32, "snack": 0.14}
STANDARD_SLOTS = tuple(CALORIE_TARGETS)
CALORIE_COL = "calories [kcal]"
SERVING_COL = "servingSize [g]"

Plan = Mapping[str, Optional[dict]]


# ── §4.4 structural validity ─────────────────────────────────────────────────

def structural_validity(plan: Plan, *, has_explanation: bool = True,
                        require_explanation: bool = False) -> float:
    """1.0 iff the plan parses with all four required slots present (and, when
    ``require_explanation``, an explanation field is present); else 0.0.

    This is the deterministic half of the thesis "Guideline" metric, split out
    from explanation *quality* (§4.4) which stays LLM-judged.
    """
    slots_ok = all(slot in plan for slot in STANDARD_SLOTS)
    explanation_ok = has_explanation or not require_explanation
    return 1.0 if (slots_ok and explanation_ok) else 0.0


# ── §4.5 meal-plan quality ───────────────────────────────────────────────────

def meal_slot_completeness(plan: Plan) -> float:
    """Fraction of the four required slots filled with a real recipe."""
    return sum(1 for s in STANDARD_SLOTS if _filled(plan.get(s))) / len(STANDARD_SLOTS)


def intra_plan_diversity(plan: Plan) -> float:
    """1 − mean pairwise Jaccard over each meal's ingredient+tag feature set.

    Undefined (NaN) with fewer than two filled meals — diversity needs a pair.
    """
    feature_sets = [_features(plan[s]) for s in plan if _filled(plan.get(s))]
    feature_sets = [f for f in feature_sets if f]
    if len(feature_sets) < 2:
        return float("nan")
    sims = [_jaccard(a, b) for a, b in combinations(feature_sets, 2)]
    return 1.0 - (sum(sims) / len(sims))


def calorie_distribution_adherence(plan: Plan) -> float:
    """1 − mean abs. deviation of each slot's calorie share from its target.

    Targets are renormalised over the *filled* slots so a partial plan is judged
    on the shape it does provide. NaN if no slot has a usable calorie value.
    """
    cals = {s: _total_calories(plan.get(s)) for s in STANDARD_SLOTS}
    cals = {s: c for s, c in cals.items() if c is not None and c > 0}
    if not cals:
        return float("nan")
    total = sum(cals.values())
    target_norm = sum(CALORIE_TARGETS[s] for s in cals)
    deviations = [abs(cals[s] / total - CALORIE_TARGETS[s] / target_norm) for s in cals]
    return max(0.0, 1.0 - sum(deviations) / len(deviations))


def plan_quality_metrics(plan: Plan, *, has_explanation: bool = True,
                         require_explanation: bool = False) -> dict:
    """The full deterministic §4.4/§4.5 block for a §13.1 record."""
    return {
        "structural_validity": structural_validity(
            plan, has_explanation=has_explanation, require_explanation=require_explanation),
        "meal_slot_completeness": meal_slot_completeness(plan),
        "intra_plan_diversity": intra_plan_diversity(plan),
        "calorie_adherence": calorie_distribution_adherence(plan),
    }


# ── helpers ──────────────────────────────────────────────────────────────────

def _filled(recipe: Optional[dict]) -> bool:
    return bool(recipe)


def _features(recipe: Optional[dict]) -> set:
    if not recipe:
        return set()
    ingredients = recipe.get("ingredient_names") or []
    tags = recipe.get("tags") or []
    return {f"i:{i}" for i in ingredients} | {f"t:{t}" for t in tags}


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def _total_calories(recipe: Optional[dict]) -> Optional[float]:
    """Whole-serving calories (HUMMUS nutrient values are per 100 g)."""
    if not recipe:
        return None
    cal = recipe.get(CALORIE_COL)
    if cal is None:
        return None
    try:
        cal = float(cal)
    except (TypeError, ValueError):
        return None
    serving = recipe.get(SERVING_COL)
    try:
        serving = float(serving)
    except (TypeError, ValueError):
        serving = None
    return cal / 100.0 * serving if serving else cal

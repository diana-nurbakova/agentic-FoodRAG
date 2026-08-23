"""Deterministic meal-plan quality metrics.

The Use-2 full-pipeline (EXP-7 rung 4 / EXP-3) judges an *assembled meal plan*,
not a flat ranked list. Three of its quality signals are deterministic — computed
straight off the plan's recipes + the HUMMUS attributes — and so live here next to
the other rule-based metrics rather than going through an LLM judge:

* **Structural Validity** — does the output parse as a meal plan with the
  required slots (and an explanation field, when the config is expected to emit
  one)? A schema check, not an LLM call.
* **Intra-Plan Diversity** — 1 − mean pairwise ingredient/tag Jaccard
  across the meals (penalises four-near-identical-dishes plans).
* **Calorie-Distribution Adherence** — 1 − mean abs. deviation from the
  target split (Breakfast 22% / Lunch 32% / Dinner 32% / Snack 14%).
* **Meal-Slot Completeness** — fraction of the four required slots filled.

A plan is passed in normalised form: ``{meal_type_str: recipe_row | None}`` where a
recipe row is a mapping of HUMMUS columns (``ingredient_names``, ``tags``,
``calories [kcal]``, ``servingSize [g]``, …) and ``None`` marks an unfilled slot.
"""
from __future__ import annotations

from itertools import combinations
from typing import Mapping, Optional

# calorie target split for adults 19–59 (thesis Table 6.3).
CALORIE_TARGETS = {"breakfast": 0.22, "lunch": 0.32, "dinner": 0.32, "snack": 0.14}
STANDARD_SLOTS = tuple(CALORIE_TARGETS)
CALORIE_COL = "calories [kcal]"
SERVING_COL = "servingSize [g]"

# slot-appropriateness synonym map: which HUMMUS ``meal_types`` labels count
# as "suitable" for each plan slot. A recipe suits a slot if its meal_types overlap
# the slot's synonym set (or its meal_types are unknown/empty → not penalised).
SLOT_SYNONYMS = {
    "breakfast": {"breakfast", "brunch"},
    "lunch": {"lunch", "main-dish", "main course", "dinner-party"},
    "dinner": {"dinner", "main-dish", "main course", "dinner-party"},
    "snack": {"snack", "snacks", "appetizer", "appetizers", "dessert", "side-dish"},
}

Plan = Mapping[str, Optional[dict]]


# ── structural validity ──────────────────────────────────────────────────────

def structural_validity(plan: Plan, *, has_explanation: bool = True,
                        require_explanation: bool = False) -> float:
    """1.0 iff the plan parses with all four required slots present (and, when
    ``require_explanation``, an explanation field is present); else 0.0.

    This is the deterministic half of the thesis "Guideline" metric, split out
    from explanation *quality* which stays LLM-judged.
    """
    slots_ok = all(slot in plan for slot in STANDARD_SLOTS)
    explanation_ok = has_explanation or not require_explanation
    return 1.0 if (slots_ok and explanation_ok) else 0.0


# ── meal-plan quality ────────────────────────────────────────────────────────

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


def slot_appropriateness(plan: Plan) -> float:
    """Fraction of FILLED slots whose recipe suits the slot's meal type.

    A recipe suits its slot when its HUMMUS ``meal_types`` overlap the slot's
    :data:`SLOT_SYNONYMS`; a recipe with unknown/empty ``meal_types`` is *not*
    penalised (counts as appropriate). Denominator is the filled slots; NaN if the
    plan has no filled slot.
    """
    filled = [(s, plan[s]) for s in STANDARD_SLOTS if _filled(plan.get(s))]
    if not filled:
        return float("nan")
    appropriate = 0
    for slot, recipe in filled:
        mtypes = _meal_types(recipe)
        if not mtypes or (SLOT_SYNONYMS.get(slot, set()) & mtypes):
            appropriate += 1
    return appropriate / len(filled)


def budget_compliance(plan: Plan, restriction: Optional[Mapping] = None) -> dict:
    """Did the plan respect the calorie caps its own question stated?

    Distinct from ``guideline_adherence``, which scores the plan against population
    reference bands the persona was never told. This scores it against the per-slot
    caps actually present in its prompt, so it answers "did it obey the budget it was
    given?" — and unlike :func:`calorie_distribution_adherence`, which normalises by
    the plan total and so only sees the *shape* of the split, it is sensitive to
    magnitude: a plan with a perfect 22/32/32/14 shape at triple the calories scores 0.

    ``restriction`` is the QA item's ``meal_plan`` mapping
    ``{slot: {"meal_type": str, "guidelines": {"calories [kcal]": cap}}}``. All keys
    are NaN when it is absent (a recipe-type question, or a run predating per-slot
    budgets) — absent is not the same as complied.
    """
    nan = float("nan")
    empty = {"budget_slot_compliance": nan, "budget_total_compliance": nan,
             "budget_overshoot_ratio": nan, "budget_daily_kcal": nan, "budget_plan_kcal": nan}
    if not restriction:
        return empty

    caps = {}
    for slot, spec in restriction.items():
        cap = ((spec or {}).get("guidelines") or {}).get(CALORIE_COL)
        if cap is None:
            continue
        try:
            caps[slot] = float(cap)
        except (TypeError, ValueError):
            continue
    if not caps:
        return empty

    daily_budget = sum(caps.values())
    if daily_budget <= 0:
        return empty

    # Per-slot: only slots that are both filled and capped can be judged.
    judged = [(_total_calories(plan.get(slot)), cap) for slot, cap in caps.items()
              if _filled(plan.get(slot))]
    judged = [(kcal, cap) for kcal, cap in judged if kcal is not None]
    slot_compliance = (sum(1 for kcal, cap in judged if kcal <= cap) / len(judged)
                       if judged else nan)

    # Daily total: over ALL filled slots, including any the restriction never capped —
    # calories eaten off-budget still count against the day.
    totals = [_total_calories(r) for r in plan.values() if _filled(r)]
    totals = [t for t in totals if t is not None]
    if not totals:
        return {**empty, "budget_daily_kcal": daily_budget}
    plan_kcal = sum(totals)

    return {
        "budget_slot_compliance": slot_compliance,
        "budget_total_compliance": 1.0 if plan_kcal <= daily_budget else 0.0,
        "budget_overshoot_ratio": plan_kcal / daily_budget,
        "budget_daily_kcal": daily_budget,
        "budget_plan_kcal": plan_kcal,
    }


def plan_quality_metrics(plan: Plan, *, has_explanation: bool = True,
                         require_explanation: bool = False,
                         restriction: Optional[Mapping] = None) -> dict:
    """The full deterministic meal-plan-quality block for a per-question record."""
    return {
        "structural_validity": structural_validity(
            plan, has_explanation=has_explanation, require_explanation=require_explanation),
        "meal_slot_completeness": meal_slot_completeness(plan),
        "intra_plan_diversity": intra_plan_diversity(plan),
        "calorie_adherence": calorie_distribution_adherence(plan),
        "slot_appropriateness": slot_appropriateness(plan),
        **budget_compliance(plan, restriction),
    }


# ── helpers ──────────────────────────────────────────────────────────────────

def _filled(recipe: Optional[dict]) -> bool:
    return bool(recipe)


def _meal_types(recipe: Optional[dict]) -> set:
    """Normalise a recipe's ``meal_types`` to a lower-cased set.

    HUMMUS stores ``meal_types`` as a real list, but the CSV-loaded frame keeps it as a
    string repr of a list (``"['lunch', 'dinner']"``); accept both, plus a bare list.
    """
    if not recipe:
        return set()
    raw = recipe.get("meal_types")
    if raw is None:
        return set()
    if isinstance(raw, (list, tuple, set)):
        items = raw
    elif isinstance(raw, str):
        s = raw.strip()
        if not s:
            return set()
        try:
            import ast
            parsed = ast.literal_eval(s)
            items = parsed if isinstance(parsed, (list, tuple, set)) else [parsed]
        except (ValueError, SyntaxError):
            items = [p.strip() for p in s.split(",")]
    else:
        return set()
    return {str(m).strip().lower() for m in items if str(m).strip()}


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

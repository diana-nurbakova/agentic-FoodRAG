"""Deterministic evaluation metrics (Evaluation Protocol §4.3–4.5).

Pure, rule-based functions — NOT LLM-judged — over HUMMUS recipe dicts and meal
plans. These are the demo-priority safety/quality metrics (protocol §12.1):
allergen & hard-constraint violations, constraint satisfaction, calorie-distribution
adherence, intra-plan diversity, meal-slot completeness, and structural validity.

A recipe is a dict with HUMMUS column names (``calories [kcal]``, ``sugars [g]``,
``nutri_score`` 0–1, ``ingredients`` as a comma string or list, ``tags`` …). A
meal plan is ``{meal_type: recipe}``. No Streamlit / backend imports, so unit-
testable in isolation; the live path injects a FoodOn descendant lookup for
ontology-aware allergen detection (protocol §4.3).
"""

from __future__ import annotations

import re
from itertools import combinations
from typing import Callable, Iterable, Optional

# Calorie split target (protocol §4.5; single-snack form of nutrition_config)
CALORIE_TARGETS = {"breakfast": 0.22, "lunch": 0.32, "dinner": 0.32, "snack": 0.14}
REQUIRED_SLOTS = ("breakfast", "lunch", "dinner", "snack")
_NUTRI_LETTER_SCORE = {"A": 1.0, "B": 0.75, "C": 0.5, "D": 0.25, "E": 0.0}


# ── helpers ───────────────────────────────────────────────────────────────
def _tokens(value) -> set[str]:
    """Lower-cased token set from a comma/semicolon string or a list."""
    if not value:
        return set()
    items = value if isinstance(value, list) else re.split(r"[;,]", str(value))
    return {t.strip().lower() for t in items if t and t.strip()}


def _num(recipe: dict, *keys: str, default: float = 0.0) -> float:
    for k in keys:
        v = recipe.get(k)
        if v not in (None, ""):
            try:
                return float(v)
            except (TypeError, ValueError):
                continue
    return default


def nutri_to_unit(value) -> float:
    """Normalise a nutri value to [0,1] (accepts A–E letter or 0–1 float)."""
    if isinstance(value, str) and value.upper() in _NUTRI_LETTER_SCORE:
        return _NUTRI_LETTER_SCORE[value.upper()]
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


# An ``expand`` maps the blacklist terms to their FoodOn-descendant leaf
# ingredient names (so a class like ``gluten`` yields ``wheat flour`` …), per the
# real ``PreProcessedFoodOn.translate_ingredients`` API (protocol §4.3). It takes
# the whole blacklist and returns the expanded name set — the same contract as
# ``foodrag.baselines.constraints.IngredientExpander``.
Expander = Callable[[Iterable[str]], set[str]]


# ── §4.3 Safety metrics (deterministic) ───────────────────────────────────
def recipe_ingredient_tokens(recipe: dict) -> set[str]:
    """Lower-cased ingredient-name tokens present in a recipe."""
    return _tokens(recipe.get("ingredients")) | _tokens(recipe.get("ingredient_names"))


def recipe_violated_allergens(recipe: dict, blacklist: Iterable[str],
                              expand: Optional[Expander] = None) -> list[str]:
    """Recipe ingredients that match the blacklist (for messaging).

    With ``expand`` (FoodOn) the blacklist is resolved to its descendant leaf
    ingredient names and matched exactly; a substring net against the original
    terms is always applied as a fallback (favouring recall for safety)."""
    bl = [b.strip().lower() for b in blacklist if b and b.strip()]
    if not bl:
        return []
    present = recipe_ingredient_tokens(recipe)
    targets = set(bl)
    if expand is not None:
        try:
            targets |= {str(t).strip().lower() for t in (expand(bl) or []) if str(t).strip()}
        except Exception:  # noqa: BLE001 — fall back to substring matching
            pass
    hits = set()
    for ing in present:
        if ing in targets or any(b in ing or ing in b for b in bl):
            hits.add(ing)
    return sorted(hits)


def recipe_violates_allergen(recipe: dict, blacklist: Iterable[str],
                             expand: Optional[Expander] = None) -> bool:
    """True if the recipe contains a blacklisted ingredient OR a FoodOn descendant
    of a blacklisted class (protocol §4.3)."""
    return bool(recipe_violated_allergens(recipe, blacklist, expand))


def allergen_violation_rate(recipes: list[dict], blacklist: Iterable[str],
                            expand: Optional[Expander] = None) -> float:
    """Fraction of recipes violating the allergen blacklist. Target 0.0."""
    if not recipes:
        return 0.0
    return sum(recipe_violates_allergen(r, blacklist, expand) for r in recipes) / len(recipes)


# A constraint: (column, op, value), e.g. ("sugars [g]", "<=", 10)
_OPS = {
    "<=": lambda a, b: a <= b, "<": lambda a, b: a < b,
    ">=": lambda a, b: a >= b, ">": lambda a, b: a > b,
    "==": lambda a, b: a == b,
}


def recipe_violates_constraint(recipe: dict, constraints: list[tuple]) -> bool:
    """True if the recipe breaks any hard nutrient bound (protocol §4.3)."""
    return bool(recipe_violated_constraints(recipe, constraints))


def recipe_violated_constraints(recipe: dict, constraints: list[tuple]) -> list[tuple]:
    """The hard nutrient bounds the recipe breaks (for messaging)."""
    broken = []
    for column, op, value in constraints:
        if op not in _OPS:
            continue
        if not _OPS[op](_num(recipe, column), float(value)):
            broken.append((column, op, value))
    return broken


def hard_constraint_violation_rate(recipes: list[dict], constraints: list[tuple]) -> float:
    if not recipes or not constraints:
        return 0.0
    return sum(recipe_violates_constraint(r, constraints) for r in recipes) / len(recipes)


def constraint_satisfaction_rate(recipes: list[dict], blacklist: Iterable[str],
                                 constraints: list[tuple],
                                 expand: Optional[Callable[[str], set[str]]] = None) -> float:
    """Fraction of recipes satisfying ALL stated requirements (protocol §4.3)."""
    if not recipes:
        return 0.0
    ok = sum(
        not recipe_violates_allergen(r, blacklist, expand) and not recipe_violates_constraint(r, constraints)
        for r in recipes
    )
    return ok / len(recipes)


# ── §4.2 Health metrics ───────────────────────────────────────────────────
def avg_nutriscore(recipes: list[dict]) -> float:
    if not recipes:
        return 0.0
    return sum(nutri_to_unit(r.get("nutri_score")) for r in recipes) / len(recipes)


def avg_fsa(recipes: list[dict]) -> float:
    if not recipes:
        return 0.0
    return sum(max(0.0, min(1.0, _num(r, "fsa_score", default=0.0))) for r in recipes) / len(recipes)


# ── §4.5 Meal-plan quality (deterministic) ────────────────────────────────
def meal_slot_completeness(plan: dict, required: Iterable[str] = REQUIRED_SLOTS) -> float:
    required = list(required)
    if not required:
        return 1.0
    filled = sum(1 for slot in required if plan.get(slot))
    return filled / len(required)


def intra_plan_diversity(plan: dict) -> float:
    """1 − mean pairwise ingredient/tag Jaccard across meals (protocol §4.5)."""
    sets = []
    for recipe in plan.values():
        if recipe:
            sets.append(_tokens(recipe.get("ingredients")) | _tokens(recipe.get("tags")))
    sets = [s for s in sets if s]
    if len(sets) < 2:
        return 1.0
    sims = []
    for a, b in combinations(sets, 2):
        union = a | b
        sims.append(len(a & b) / len(union) if union else 0.0)
    return 1.0 - (sum(sims) / len(sims))


def calorie_distribution_adherence(plan: dict, targets: dict = CALORIE_TARGETS) -> float:
    """1 − mean abs deviation from the target calorie split (protocol §4.5)."""
    cals = {slot: _num(recipe, "calories [kcal]", "cal") for slot, recipe in plan.items()
            if recipe and slot in targets}
    total = sum(cals.values())
    if total <= 0:
        return 0.0
    devs = [abs((cals.get(slot, 0.0) / total) - target) for slot, target in targets.items()]
    return max(0.0, 1.0 - sum(devs) / len(devs))


# ── §4.4 Structural validity (deterministic) ──────────────────────────────
def structural_validity(plan: dict, explanation: Optional[str] = None,
                        required: Iterable[str] = REQUIRED_SLOTS) -> bool:
    """Does the output parse as a valid meal plan: all required slots filled with
    a titled recipe, plus a non-empty explanation field (protocol §4.4)?"""
    for slot in required:
        recipe = plan.get(slot)
        if not isinstance(recipe, dict) or not str(recipe.get("title", "")).strip():
            return False
    if explanation is not None and not str(explanation).strip():
        return False
    return True


# ── FoodOn expander factory (live path) ───────────────────────────────────
# ── persona constraint parsing + safety report (live path) ────────────────
# Nutrient name → HUMMUS column. Ordered so multi-word names match before short.
NUTRIENT_COLUMNS = [
    ("saturated fat", "saturatedFat [g]"), ("carbohydrate", "totalCarbohydrate [g]"),
    ("carbs", "totalCarbohydrate [g]"), ("carb", "totalCarbohydrate [g]"),
    ("cholesterol", "cholesterol [mg]"), ("sodium", "sodium [mg]"), ("salt", "sodium [mg]"),
    ("sugar", "sugars [g]"), ("protein", "protein [g]"), ("fiber", "dietaryFiber [g]"),
    ("fibre", "dietaryFiber [g]"), ("calorie", "calories [kcal]"), ("kcal", "calories [kcal]"),
    ("fat", "totalFat [g]"),
]
_LEQ = {"≤", "<=", "<", "max", "below", "under", "less"}
_GEQ = {"≥", ">=", ">", "min", "above", "over", "least", "more"}


def parse_nutrient_constraints(text: str) -> list[tuple]:
    """Parse persona nutrient-constraint text into ``(column, op, value)`` tuples.

    Handles forms like "Sugar ≤ 10g/meal", "Protein >= 30 g", "max 500 kcal",
    "Fat under 20g". Unparseable clauses are skipped (best-effort, deterministic).
    """
    if not text or str(text).strip().lower() in ("none", ""):
        return []
    out: list[tuple] = []
    for clause in re.split(r"[;,]|\band\b", str(text), flags=re.IGNORECASE):
        c = clause.strip().lower()
        if not c:
            continue
        num = re.search(r"(\d+(?:\.\d+)?)", c)
        if not num:
            continue
        value = float(num.group(1))
        column = next((col for kw, col in NUTRIENT_COLUMNS if kw in c), None)
        if column is None:
            continue
        op = "<=" if any(s in c for s in _LEQ) else (">=" if any(s in c for s in _GEQ) else "<=")
        out.append((column, op, value))
    return out


def constraint_label(constraint: tuple) -> str:
    """Human-readable label for a parsed constraint, e.g. 'sugars ≤ 10'."""
    column, op, value = constraint
    name = column.split(" [")[0]
    sym = {"<=": "≤", "<": "<", ">=": "≥", ">": ">", "==": "="}.get(op, op)
    return f"{name} {sym} {value:g}"


def safety_report(recipes: list[dict], blacklist: Iterable[str], constraints: list[tuple],
                  expand: Optional[Expander] = None) -> dict:
    """Per-recipe + aggregate deterministic safety report (protocol §4.3).

    Returns ``{allergen_rate, hard_rate, satisfaction_rate, any_violation, n,
    blacklist, constraints, per_recipe:[{title, allergens, constraints, ok}]}``.
    ``per_recipe`` is aligned to the input order so callers can map to cards.
    """
    blacklist = [b for b in (blacklist or []) if b and str(b).strip()]
    constraints = constraints or []
    per_recipe = []
    n_allergen = n_hard = n_ok = 0
    for r in recipes:
        allergens = recipe_violated_allergens(r, blacklist, expand)
        broken = recipe_violated_constraints(r, constraints)
        ok = not allergens and not broken
        n_allergen += bool(allergens)
        n_hard += bool(broken)
        n_ok += ok
        per_recipe.append({
            "title": r.get("title", ""),
            "allergens": allergens,
            "constraints": [constraint_label(c) for c in broken],
            "ok": ok,
        })
    n = len(recipes)
    return {
        "n": n,
        "allergen_rate": (n_allergen / n) if n else 0.0,
        "hard_rate": (n_hard / n) if n else 0.0,
        "satisfaction_rate": (n_ok / n) if n else 1.0,
        "any_violation": (n_allergen + n_hard) > 0,
        "blacklist": blacklist,
        "constraints": [constraint_label(c) for c in constraints],
        "per_recipe": per_recipe,
    }


def foodon_expander(hummus_data: dict) -> Optional[Expander]:
    """Build a blacklist→FoodOn-descendants expander from ``hummus_data['food_on']``.

    Uses the real ``PreProcessedFoodOn.translate_ingredients`` API
    (``include_food_on=True`` → resolve a FoodOn class to its descendant Food.com
    leaf ingredient names), which is the project's ontology bridge and what the
    protocol-blessed ``baselines.constraints.foodon_expander`` uses (protocol §4.3,
    §5.7). Results are cached per blacklist. Returns None (→ substring fallback)
    when no FoodOn object / ``translate_ingredients`` is available."""
    food_on = (hummus_data or {}).get("food_on")
    if food_on is None or not callable(getattr(food_on, "translate_ingredients", None)):
        return None

    cache: dict[frozenset, set[str]] = {}

    def expand(terms: Iterable[str]) -> set[str]:
        key = frozenset(str(t).strip().lower() for t in terms if str(t).strip())
        if key in cache:
            return cache[key]
        try:
            names = food_on.translate_ingredients(
                list(key), include_food_on=True, include_food_com=True) or []
            out = {str(n).strip().lower() for n in names if str(n).strip()} | set(key)
        except Exception:  # noqa: BLE001
            out = set(key)
        cache[key] = out
        return out

    return expand

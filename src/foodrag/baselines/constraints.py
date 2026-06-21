"""Deterministic persona-constraint logic shared by every baseline rung.

This mirrors ``qa_generator._get_valid_recipes_for_persona`` (the function that
defines the gold answer set) so that the safety metrics (§4.3 of the Evaluation
Protocol) are computed against *exactly* the same notion of "valid recipe" used
to build the ground truth — there is no second, drifting definition.

Two views are provided:

* :func:`filter_valid` — vectorised filter over a recipes DataFrame (used to
  build the candidate set for the health-aware popularity rung 0b and for the
  constraint-satisfaction metric).
* per-recipe predicates (:func:`violates_allergen`, :func:`violates_hard_nutrient`,
  :func:`satisfies_all`) — used to score the top-K recommendations of *any* rung.

Ingredient matching uses a pluggable ``expand`` callable. The default
(:func:`identity_expand`) does plain Food.com-name matching, which is exactly
what the shipped QA set encodes (``persona_config.ingredients.include_food_on =
false``). Passing a FoodOn-backed expander upgrades allergen detection to the
hierarchy lookup described in §4.3 ("wheat flour → wheat → gluten") without any
change to the metric code.
"""
from __future__ import annotations

from typing import Callable, Iterable

import pandas as pd

# A persona's disliked/liked ingredient names → the concrete Food.com ingredient
# names to match against ``recipe['ingredient_names']``.
IngredientExpander = Callable[[Iterable[str]], "set[str]"]

SERVING_COL = "servingSize [g]"


def identity_expand(ingredients: Iterable[str]) -> set[str]:
    """Default expander: match Food.com ingredient names verbatim (no hierarchy)."""
    return set(ingredients)


def foodon_expander(food_on, persona_config) -> IngredientExpander:
    """Build an expander that resolves FoodOn classes to leaf Food.com names.

    Delegates to ``food_on.translate_ingredients`` with the project's persona
    config, so the expansion is identical to QA-set generation. Use this for the
    full §4.3 allergen story; omit it for the lightweight string-matching path.
    """
    include_food_on = persona_config["ingredients"]["include_food_on"]
    include_food_com = persona_config["ingredients"]["include_food_com"]

    def _expand(ingredients: Iterable[str]) -> set[str]:
        ingredients = list(ingredients)
        if not ingredients:
            return set()
        return set(food_on.translate_ingredients(
            ingredients=ingredients,
            include_food_on=include_food_on,
            include_food_com=include_food_com,
        ))

    return _expand


# ── per-recipe predicates (score the top-K recommendations) ──────────────────

def _total(row, nutrient: str) -> float:
    """Whole-serving amount of ``nutrient`` (recipe values are per 100 g)."""
    return row[nutrient] / 100 * row[SERVING_COL]


def violates_hard_nutrient(row, persona: dict) -> bool:
    """True if the recipe breaks any numeric bound stated by the persona.

    Covers relative-nutrient bounds, minimum nutrition-score thresholds and
    total-nutrient maxima — the deterministic "hard constraint" family (§4.3).
    """
    for nutrient, limits in persona.get("relative_nutrients", {}).items():
        lower = limits["limits"]["lower"]
        upper = limits["limits"]["upper"]
        if not (lower <= row[nutrient] <= upper):
            return True

    for score_name, score_min in persona.get("nutrition_scores", {}).items():
        if row[score_name] < score_min:
            return True

    for nutrient, total_max in persona.get("total_nutrients", {}).items():
        if _total(row, nutrient) > total_max:
            return True

    return False


def violates_allergen(row, persona: dict, expand: IngredientExpander = identity_expand) -> bool:
    """True if the recipe contains a disliked (blacklisted) ingredient.

    Disliked ingredients are the QA set's proxy for allergens/blacklist; with a
    FoodOn ``expand`` this becomes hierarchy-aware descendant detection (§4.3).
    """
    disliked = expand(persona.get("ingredients", {}).get("dislikes", []))
    if not disliked:
        return False
    return any(ingr in disliked for ingr in row["ingredient_names"])


def _tags_ok(row, persona: dict) -> bool:
    liked = persona.get("tags", {}).get("likes", [])
    disliked = persona.get("tags", {}).get("dislikes", [])
    tags = row["tags"]
    if liked and not any(t in liked for t in tags):
        return False
    if any(t in disliked for t in tags):
        return False
    return True


def _liked_ingredients_ok(row, persona: dict, expand: IngredientExpander) -> bool:
    liked = expand(persona.get("ingredients", {}).get("likes", []))
    if not liked:  # no liked ingredients → any ingredient allowed
        return True
    return any(ingr in liked for ingr in row["ingredient_names"])


def satisfies_all(row, persona: dict, expand: IngredientExpander = identity_expand) -> bool:
    """True iff the recipe satisfies EVERY stated requirement (= in the gold set).

    Equivalent to membership in :func:`filter_valid`; provided as a row-wise
    predicate so the constraint-satisfaction metric can score arbitrary
    recommendations without rebuilding the full valid set.
    """
    return (
        not violates_hard_nutrient(row, persona)
        and not violates_allergen(row, persona, expand)
        and _tags_ok(row, persona)
        and _liked_ingredients_ok(row, persona, expand)
    )


# ── vectorised filter (build the valid candidate set) ────────────────────────

def filter_valid(persona: dict, recipes_df: pd.DataFrame,
                 expand: IngredientExpander = identity_expand) -> pd.DataFrame:
    """Return the sub-frame of recipes satisfying every persona requirement.

    Vectorised mirror of ``qa_generator._get_valid_recipes_for_persona`` (with
    ``as_df=True``); short-circuits to an empty frame as soon as a filter empties
    the set, matching the generator's behaviour.
    """
    df = recipes_df

    for nutrient, limits in persona.get("relative_nutrients", {}).items():
        lower = limits["limits"]["lower"]
        upper = limits["limits"]["upper"]
        df = df[(df[nutrient] >= lower) & (df[nutrient] <= upper)]
        if df.empty:
            return df

    for score_name, score_min in persona.get("nutrition_scores", {}).items():
        df = df[df[score_name] >= score_min]
        if df.empty:
            return df

    for nutrient, total_max in persona.get("total_nutrients", {}).items():
        df = df[(df[nutrient] / 100 * df[SERVING_COL]) <= total_max]
        if df.empty:
            return df

    liked_tags = persona.get("tags", {}).get("likes", [])
    disliked_tags = persona.get("tags", {}).get("dislikes", [])
    if liked_tags:
        df = df[df["tags"].apply(lambda x: any(t in liked_tags for t in x))]
        if df.empty:
            return df
    df = df[~df["tags"].apply(lambda x: any(t in disliked_tags for t in x))]
    if df.empty:
        return df

    liked_ingr = expand(persona.get("ingredients", {}).get("likes", []))
    disliked_ingr = expand(persona.get("ingredients", {}).get("dislikes", []))
    if liked_ingr:
        df = df[df["ingredient_names"].apply(lambda x: any(i in liked_ingr for i in x))]
        if df.empty:
            return df
    df = df[~df["ingredient_names"].apply(lambda x: any(i in disliked_ingr for i in x))]

    return df

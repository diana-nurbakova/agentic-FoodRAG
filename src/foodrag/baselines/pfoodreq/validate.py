"""Structural validator for a generated ``recipe_kg.json`` — no torch required.

Replays the exact field accesses pFoodReq performs so we can prove a converted KB
is consumable *without* installing its 2018-era dependency stack (torch 0.4.1
etc.). It mirrors:

* ``io_utils.load_ndjson_to_dict`` — each line is ``json.loads`` then ``dict.update``;
* ``generate_all_qa.generate_constraint_qas`` — tag → ``tagged_dishes`` → dish
  ``name``/``uri`` → ``contains_ingredients`` → ingredient ``name``; the three fat
  components and the ``LIMIT_NUTRIENT_VALUES`` nutrients (protein/carbohydrates)
  read as ``float(...[0])``.

Returns a report with counts and the first offending entries (if any).
"""
from __future__ import annotations

import json
from pathlib import Path

# nutrients pFoodReq reads as float(dish.neighbors[n][0]); 'fat' is derived in
# generate_all_qa from the three components, so the components must be present.
REQUIRED_DISH_NUTRIENTS = ["protein", "carbohydrates", "sugar", "calories",
                           "polyunsaturated fat", "monounsaturated fat", "saturated fat"]


def load_recipe_kg(path: str | Path) -> dict:
    """Load ndjson KB the way pFoodReq does (line → JSON object → merge)."""
    kb: dict = {}
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                kb.update(json.loads(line))
    return kb


def validate_recipe_kg(path: str | Path, max_errors: int = 20) -> dict:
    """Validate KB structure against pFoodReq's access pattern; return a report."""
    kb = load_recipe_kg(path)
    errors: list[str] = []
    n_tags = n_dishes = n_ingredients = 0

    def err(msg: str) -> None:
        if len(errors) < max_errors:
            errors.append(msg)

    for tag_uri, entity in kb.items():
        for field in ("name", "type", "neighbors"):
            if field not in entity:
                err(f"tag {tag_uri!r} missing {field!r}")
        if not entity.get("name"):
            err(f"tag {tag_uri!r} has empty name")
        neighbors = entity.get("neighbors", {})
        if "tagged_dishes" not in neighbors:
            err(f"tag {tag_uri!r} missing neighbors.tagged_dishes")
            continue
        n_tags += 1

        for dish_wrap in neighbors["tagged_dishes"]:
            if not isinstance(dish_wrap, dict) or len(dish_wrap) != 1:
                err(f"tag {tag_uri!r}: tagged_dishes entry not a single-key dict")
                continue
            dish = list(dish_wrap.values())[0]
            n_dishes += 1
            if not dish.get("name"):
                err(f"dish in {tag_uri!r} missing name")
            if "uri" not in dish:
                err(f"dish in {tag_uri!r} missing uri")
            dn = dish.get("neighbors", {})
            if "contains_ingredients" not in dn:
                err(f"dish {dish.get('uri')!r} missing contains_ingredients")
            for ing_wrap in dn.get("contains_ingredients", []):
                ing = list(ing_wrap.values())[0]
                n_ingredients += 1
                if not ing.get("name"):
                    err(f"ingredient in dish {dish.get('uri')!r} missing name")
            for nutrient in REQUIRED_DISH_NUTRIENTS:
                if nutrient not in dn:
                    err(f"dish {dish.get('uri')!r} missing nutrient {nutrient!r}")
                    continue
                try:
                    float(dn[nutrient][0])
                except (TypeError, ValueError, IndexError):
                    err(f"dish {dish.get('uri')!r} nutrient {nutrient!r} not float-like: {dn[nutrient]!r}")

    return {
        "ok": not errors,
        "n_tags": n_tags,
        "n_dish_entries": n_dishes,
        "n_ingredient_refs": n_ingredients,
        "n_errors": len(errors),
        "errors": errors,
    }

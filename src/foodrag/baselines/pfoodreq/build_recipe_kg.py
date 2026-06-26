"""HUMMUS → pFoodReq ``recipe_kg.json`` converter (baseline ladder rung 4, §5.8).

pFoodReq/BAMnet consumes a FoodKG-style knowledge graph as newline-delimited
JSON, **tag-rooted and nested**: each line is ``{tag_uri: tag_entity}``, and the
tag entity embeds *all of its dishes* under ``neighbors.tagged_dishes``; each dish
embeds its ingredients under ``neighbors.contains_ingredients`` plus per-serving
nutrient values. (Reverse-engineered from ``data_builder/src/generate_all_qa.py``
and ``BAMnet/src/core/build_data/foodkg/build_data.py``.)

Two consequences drive the design:

* **Duplication / fan-out.** A recipe is repeated under every tag it carries, and
  per question BAMnet's candidate-answer set is *all dishes under the topic tag*.
  At full HUMMUS scale this is multi-GB and is the literal source of pFoodReq's
  slowness. ``recipe_filter`` / ``max_dishes_per_tag`` bound the build so it can
  run on the tight stratum (§9.1) — the protocol's bounded-evaluation stance
  (§5.5, §5.8).
* **Nutrient mapping.** pFoodReq expects ``protein``, ``carbohydrates``, ``sugar``,
  ``calories`` and the three fat components (``polyunsaturated/monounsaturated/
  saturated fat``); it derives ``fat`` as their sum. HUMMUS has total + saturated
  fat only, so the unsaturated split is approximated (all non-saturated fat →
  polyunsaturated). All values are emitted **per serving** to match pFoodReq's
  per-meal limit semantics (``LIMIT_NUTRIENT_VALUES``). Documented in NUTRIENT_MAP.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

ID_COL = "recipe_url"
SERVING_COL = "servingSize [g]"

# pFoodReq nutrient key → HUMMUS per-100g column. Values are converted to
# per-serving on emit (value / 100 * servingSize), matching qa_generator and
# pFoodReq's per-meal LIMIT_NUTRIENT_VALUES.
NUTRIENT_MAP = {
    "protein": "protein [g]",
    "carbohydrates": "totalCarbohydrate [g]",
    "sugar": "sugars [g]",
    "calories": "calories [kcal]",
    "saturated fat": "saturatedFat [g]",
}


def _slug(prefix: str, value: str) -> str:
    return f"{prefix}:{value}"


def _per_serving(row, col: str) -> float:
    val = row.get(col)
    serving = row.get(SERVING_COL)
    if val is None or serving is None or pd.isna(val) or pd.isna(serving):
        return 0.0
    return float(val) / 100.0 * float(serving)


def _dish_entity(row) -> tuple[str, dict]:
    """Build one ``{dish_uri: dish_graph}`` entry for ``neighbors.tagged_dishes``."""
    dish_uri = _slug("recipe", str(row[ID_COL]))

    contains = [
        {_slug("ingredient", name):
            {"name": [name], "alias": [], "type": ["ingredient"], "uri": _slug("ingredient", name)}}
        for name in (row.get("ingredient_names") or [])
    ]

    neighbors: dict = {"contains_ingredients": contains}
    for nutrient, col in NUTRIENT_MAP.items():
        neighbors[nutrient] = [_per_serving(row, col)]

    # HUMMUS lacks the unsaturated split: attribute all non-saturated fat to
    # polyunsaturated, none to monounsaturated, so pFoodReq's derived
    # fat = poly + mono + saturated == total fat.
    total_fat = _per_serving(row, "totalFat [g]")
    neighbors["polyunsaturated fat"] = [max(total_fat - neighbors["saturated fat"][0], 0.0)]
    neighbors["monounsaturated fat"] = [0.0]

    dish_graph = {
        "name": [str(row.get("title", dish_uri))],
        "uri": dish_uri,
        "alias": [],
        "type": ["dish_recipe"],
        "neighbors": neighbors,
    }
    return dish_uri, dish_graph


def build_recipe_kg(
    recipes_df: pd.DataFrame,
    out_path: str | Path,
    *,
    recipe_filter: Optional[Iterable[str]] = None,
    tag_filter: Optional[Iterable[str]] = None,
    max_dishes_per_tag: Optional[int] = None,
) -> dict:
    """Write ``recipe_kg.json`` (ndjson) and return build statistics.

    :param recipes_df: HUMMUS recipes (``tags`` / ``ingredient_names`` as lists).
    :param recipe_filter: if given, only these ``recipe_url`` dishes are included
        (bounded build — e.g. the union of QA answer sets).
    :param tag_filter: if given, only these tags become topic entities.
    :param max_dishes_per_tag: cap dishes per tag (drops the fan-out tail; the
        number dropped is reported in ``stats['truncated_tags']``).
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = recipes_df
    if recipe_filter is not None:
        df = df[df[ID_COL].isin(set(recipe_filter))]
    tag_filter = set(tag_filter) if tag_filter is not None else None

    # tag → list of dish row-dicts
    tag_to_rows: dict[str, list] = {}
    for row in df.to_dict("records"):
        for tag in (row.get("tags") or []):
            if tag_filter is not None and tag not in tag_filter:
                continue
            tag_to_rows.setdefault(tag, []).append(row)

    stats = {"n_tags": 0, "n_dish_entries": 0, "n_unique_dishes": 0,
             "truncated_tags": 0, "out_path": str(out_path)}
    unique_dishes: set[str] = set()

    with open(out_path, "w", encoding="utf-8") as fh:
        for tag, rows in tag_to_rows.items():
            if max_dishes_per_tag is not None and len(rows) > max_dishes_per_tag:
                rows = rows[:max_dishes_per_tag]
                stats["truncated_tags"] += 1

            tagged_dishes = []
            for row in rows:
                dish_uri, dish_graph = _dish_entity(row)
                tagged_dishes.append({dish_uri: dish_graph})
                unique_dishes.add(dish_uri)

            tag_uri = _slug("tag", tag)
            tag_entity = {
                tag_uri: {
                    "name": [tag],
                    "alias": [],
                    "type": ["tag"],
                    "neighbors": {"tagged_dishes": tagged_dishes},
                }
            }
            fh.write(json.dumps(tag_entity, ensure_ascii=False) + "\n")
            stats["n_tags"] += 1
            stats["n_dish_entries"] += len(tagged_dishes)

    stats["n_unique_dishes"] = len(unique_dishes)
    return stats

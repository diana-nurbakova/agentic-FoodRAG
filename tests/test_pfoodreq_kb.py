"""Unit tests for the pFoodReq KB converter (baseline ladder rung 4).

Builds a tiny recipe_kg.json from a synthetic frame and checks it against the
structural validator that mirrors pFoodReq's own field access — so the schema
contract is enforced without the 2018-era torch stack or the full HUMMUS frame.
"""
from __future__ import annotations

import pandas as pd
import pytest

from foodrag.baselines.pfoodreq import build_recipe_kg, load_recipe_kg, validate_recipe_kg

pytestmark = pytest.mark.unit


@pytest.fixture
def recipes_df() -> pd.DataFrame:
    return pd.DataFrame([
        {"recipe_url": "r1", "title": "Veg Bowl", "tags": ["vegan", "easy"],
         "ingredient_names": ["spinach", "rice"], "servingSize [g]": 200,
         "protein [g]": 10, "totalCarbohydrate [g]": 30, "sugars [g]": 5,
         "calories [kcal]": 150, "totalFat [g]": 8, "saturatedFat [g]": 2},
        {"recipe_url": "r2", "title": "Easy Toast", "tags": ["easy"],
         "ingredient_names": ["bread"], "servingSize [g]": 100,
         "protein [g]": 4, "totalCarbohydrate [g]": 20, "sugars [g]": 2,
         "calories [kcal]": 120, "totalFat [g]": 3, "saturatedFat [g]": 1},
    ])


def test_build_and_validate(tmp_path, recipes_df):
    out = tmp_path / "recipe_kg.json"
    stats = build_recipe_kg(recipes_df, out)
    # 'easy' has both recipes, 'vegan' has one → 2 tags, 3 dish entries, 2 unique
    assert stats["n_tags"] == 2
    assert stats["n_dish_entries"] == 3
    assert stats["n_unique_dishes"] == 2

    report = validate_recipe_kg(out)
    assert report["ok"], report["errors"]
    assert report["n_dish_entries"] == 3


def test_schema_shape_and_nutrient_mapping(tmp_path, recipes_df):
    out = tmp_path / "recipe_kg.json"
    build_recipe_kg(recipes_df, out)
    kb = load_recipe_kg(out)

    assert set(kb) == {"tag:vegan", "tag:easy"}
    vegan = kb["tag:vegan"]
    assert vegan["type"] == ["tag"] and vegan["name"] == ["vegan"]

    dish = list(vegan["neighbors"]["tagged_dishes"][0].values())[0]
    assert dish["type"] == ["dish_recipe"] and dish["uri"] == "recipe:r1"
    nb = dish["neighbors"]
    # per-serving (value/100 * servingSize): protein 10 -> 10/100*200 = 20
    assert nb["protein"][0] == pytest.approx(20.0)
    assert nb["calories"][0] == pytest.approx(300.0)
    # fat split: saturated = 2/100*200 = 4; poly = total(8/100*200=16) - 4 = 12; mono = 0
    assert nb["saturated fat"][0] == pytest.approx(4.0)
    assert nb["polyunsaturated fat"][0] == pytest.approx(12.0)
    assert nb["monounsaturated fat"][0] == pytest.approx(0.0)
    # derived fat (poly+mono+sat) == total fat, the invariant pFoodReq relies on
    derived = nb["polyunsaturated fat"][0] + nb["monounsaturated fat"][0] + nb["saturated fat"][0]
    assert derived == pytest.approx(16.0)

    ing = list(nb["contains_ingredients"][0].values())[0]
    assert ing["type"] == ["ingredient"] and ing["name"][0] in {"spinach", "rice"}


def test_filters_and_caps(tmp_path, recipes_df):
    out = tmp_path / "recipe_kg.json"
    # recipe_filter keeps only r1 → 'easy' and 'vegan' each have 1 dish
    stats = build_recipe_kg(recipes_df, out, recipe_filter={"r1"})
    assert stats["n_unique_dishes"] == 1

    # cap drops the fan-out tail and reports it
    stats = build_recipe_kg(recipes_df, out, max_dishes_per_tag=1)
    assert stats["truncated_tags"] == 1  # only 'easy' had >1 dish
    assert stats["n_dish_entries"] == 2  # one per tag

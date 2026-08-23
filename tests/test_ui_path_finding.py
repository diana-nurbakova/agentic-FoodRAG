"""Grounded KG-path extraction tests."""
import pytest

from ui.services import path_finding as P

pytestmark = [pytest.mark.unit, pytest.mark.gui]

PERSONA = {"allergies": ["Gluten"], "nutrientConstraint": "Sugar ≤ 10g/meal",
           "likedTags": ["Mediterranean"]}

# GUI card shape
CARD_OK = {"name": "Chickpea Salad", "sugar": 4, "ingredients": ["chickpeas", "tomato", "olive oil"],
           "tags": ["Mediterranean", "vegan"]}
CARD_BAD = {"name": "Wheat Pasta", "sugar": 18, "ingredients": ["wheat flour", "tomato"],
            "tags": ["italian"]}


def test_persona_requirements():
    reqs = P.persona_requirements(PERSONA)
    kinds = sorted(r["kind"] for r in reqs)
    assert kinds == ["allergen", "nutrient", "tag"]
    labels = {r["label"] for r in reqs}
    assert "No Gluten" in labels and "sugars ≤ 10" in labels and "Mediterranean" in labels


def test_paths_all_satisfied():
    rep = P.build_paths(CARD_OK, PERSONA)
    assert rep["n"] == 3
    assert rep["coverage"] == 1.0          # all 3 requirements supported
    assert rep["validity"] == 1.0          # every presented edge is real
    by_req = {p["requirement"]: p for p in rep["paths"]}
    assert by_req["No Gluten"]["satisfied"] is True
    # excludes path: recipe → excludes Gluten
    assert by_req["No Gluten"]["nodes"][-1] == ("excludes Gluten", "allergen")
    assert by_req["sugars ≤ 10"]["satisfied"] is True
    assert by_req["Mediterranean"]["satisfied"] is True


def test_paths_with_violations():
    # Direct-name allergy "wheat" → substring catches "wheat flour" (no FoodOn needed)
    persona = {"allergies": ["wheat"], "nutrientConstraint": "Sugar ≤ 10g/meal",
               "likedTags": ["Mediterranean"]}
    rep = P.build_paths(CARD_BAD, persona)
    by_req = {p["requirement"]: p for p in rep["paths"]}
    assert by_req["No wheat"]["satisfied"] is False
    assert ("wheat flour", "ingredient") in by_req["No wheat"]["nodes"]
    assert by_req["sugars ≤ 10"]["satisfied"] is False     # 18 > 10
    assert by_req["Mediterranean"]["satisfied"] is False   # tag absent
    assert rep["coverage"] == 0.0
    assert rep["validity"] == 1.0          # paths still grounded in real attributes


def test_foodon_expander_grounds_allergen_path():
    # blacklist "gluten" expands to leaf "wheat flour" via the (mock) expander
    expand = lambda terms: {"wheat flour"} if "gluten" in {t.lower() for t in terms} else set()
    card = {"name": "Bread", "sugar": 2, "ingredients": ["wheat flour"], "tags": []}
    persona = {"allergies": ["gluten"], "nutrientConstraint": "", "likedTags": []}
    rep = P.build_paths(card, persona, expand=expand)
    gluten = rep["paths"][0]
    assert gluten["satisfied"] is False
    assert ("wheat flour", "ingredient") in gluten["nodes"]


def test_no_requirements():
    rep = P.build_paths(CARD_OK, {"allergies": [], "nutrientConstraint": "None", "likedTags": []})
    assert rep["n"] == 0 and rep["coverage"] == 1.0 and rep["paths"] == []


def test_ingredient_class_cypher_builder():
    from ui.services import backend
    q = backend.ingredient_class_cypher(7362, max_ingredients=8, depth=2)
    assert "recipe_id: 7362" in q and "SUB_CLASS_OF*1..2" in q and "LIMIT 8" in q
    # int coercion + clamps (defends against bad input in the live query string)
    assert "SUB_CLASS_OF*1..4" in backend.ingredient_class_cypher(1, depth=99)
    assert "LIMIT 25" in backend.ingredient_class_cypher(1, max_ingredients=999)

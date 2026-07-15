"""GUI mapping unit tests: backend HUMMUS recipe dict → GUI card shapes."""
import pytest

pytestmark = [pytest.mark.unit, pytest.mark.gui]

RECIPE = {
    "title": "Layered Hummus Dip", "calories [kcal]": 382.2, "protein [g]": 18.8,
    "totalCarbohydrate [g]": 22.9, "totalFat [g]": 25.5, "dietaryFiber [g]": 8.3,
    "sugars [g]": 3.9, "sodium [mg]": 1199.0, "nutri_score": 0.0, "fsa_score": 0.25,
    "who_score": 0.2325, "duration [min]": 15.0, "serves": 2.0,
    "ingredient_quantities": "8 ounces hummus; 0.5 cup tomato; feta cheese",
    "ingredients": "hummus, tomatoes, feta cheese", "tags": "easy, low-carb, dips",
    "directions": "Combine hummus and cumin. Top with tomatoes. Add feta and olives.",
}


def test_nutri_letter_thresholds():
    from ui.services.mapping import nutri_letter

    assert [nutri_letter(v) for v in (1.0, 0.75, 0.5, 0.25, 0.0)] == ["A", "B", "C", "D", "E"]


def test_recipe_to_card():
    from ui.services.mapping import recipe_to_card

    card = recipe_to_card(RECIPE, "lunch")
    assert card["name"] == "Layered Hummus Dip"
    assert card["nutriscore"] == "E"        # 0.0 → E
    assert card["cal"] == 382 and card["protein"] == 19
    assert card["time"] == "15min" and card["skill"] == "Easy" and card["servings"] == 2
    assert card["who"] == 2                   # round(0.2325*10)
    assert card["fsa"] == 8                   # round((1-0.25)*10)
    assert len(card["ingredients"]) == 3 and len(card["steps"]) >= 3
    assert card["img"] == "🥗"               # lunch emoji


def test_recipe_to_card_per_serving_and_food_com():
    from ui.services.mapping import recipe_to_card

    recipe = dict(RECIPE, **{
        "servingSize [g]": 250.0, "recipe_id": 42,
        "recipe_url": "https://www.food.com/recipe/hummus-dip-42",
    })
    card = recipe_to_card(recipe, "lunch")
    # HUMMUS nutrients are per-100 g; per serving = value/100 × servingSize.
    assert card["cal"] == round(382.2 / 100 * 250)      # 956
    assert card["protein"] == round(18.8 / 100 * 250)   # 47
    assert card["serving_size_g"] == 250
    assert card["recipe_url"] == "https://www.food.com/recipe/hummus-dip-42"


def test_food_com_url_from_locator():
    from ui.services.mapping import food_com_url

    assert food_com_url({"recipe_url": "https://www.food.com/recipe/x-1"}) == "https://www.food.com/recipe/x-1"
    assert food_com_url({"food_kg_locator": "http://idea.rpi.edu/heals/kb/recipe/x-2"}) == "https://www.food.com/recipe/x-2"
    assert food_com_url({}) == ""


def test_meal_plan_to_recipes_order():
    from ui.services.mapping import meal_plan_to_recipes

    plan = {"dinner": dict(RECIPE), "breakfast": dict(RECIPE)}
    cards = meal_plan_to_recipes(plan)
    # Always Breakfast→Lunch→Dinner→Snack order; only present meals
    assert [c["meal"] for c in cards] == ["Breakfast", "Dinner"]


def test_validation_to_log_and_splits():
    from ui.services.mapping import questions_to_splits, validation_to_log

    log = validation_to_log({"breakfast": "VALID it works", "lunch": "INVALID\nbad"},
                            {"lunch": "relaxed tag"})
    by_meal = {r["meal"]: r for r in log}
    assert by_meal["Breakfast"]["status"] == "OK"
    assert by_meal["Lunch"]["status"] == "INVALID" and by_meal["Lunch"]["action"] == "relaxed tag"

    splits = questions_to_splits({"breakfast": "a gluten-free breakfast"})
    assert splits == [{"meal": "Breakfast", "q": "a gluten-free breakfast"}]

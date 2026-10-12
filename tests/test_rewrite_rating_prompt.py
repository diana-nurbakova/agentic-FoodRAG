"""The judge prompt that scores ``llm_questions`` must show the judge everything it rates against.

Regression: an inline conditional bound across implicitly-concatenated string literals, so
recipe questions were scored without the original question or persona, and meal-plan
questions without the reformulations or the rating instruction.
"""
from foodrag.persona.qa_generator import _parse_scores, _rating_prompt

PERSONA = {"ingredients": {"likes": ["baking apples"], "dislikes": ["raw tomatoes"]}}
REWRITES = [{"question": "Apple recipes without raw tomatoes?"},
            {"question": "Anything with baking apples?"}]


def _qa(meal_plan=None):
    return {"question": "What are the recipes that meet the following criteria?\n - Ingredients Liked: baking apples",
            "persona": PERSONA, "meal_plan": meal_plan, "llm_questions": REWRITES}


def test_recipe_prompt_contains_original_persona_rewrites_and_instruction():
    prompt = _rating_prompt(_qa())
    assert "Original Question: What are the recipes" in prompt
    assert "baking apples" in prompt and "raw tomatoes" in prompt  # persona
    for rewrite in REWRITES:
        assert rewrite["question"] in prompt
    assert "Provide a score between 1 (poor) and 5 (excellent)" in prompt
    assert "Meal Plan" not in prompt


def test_meal_plan_prompt_also_contains_rewrites_and_instruction():
    prompt = _rating_prompt(_qa(meal_plan={"breakfast": {"calories [kcal]": 500}}))
    assert "Meal Plan:\n{'breakfast'" in prompt
    assert "and \n" not in prompt
    for rewrite in REWRITES:
        assert rewrite["question"] in prompt
    assert "Return the results as a list of integer scores" in prompt


def test_parse_scores():
    assert _parse_scores("[5, 4, 3]") == [5, 4, 3]
    assert _parse_scores("Here are the scores: [5, 2]\nThanks") == [5, 2]
    assert _parse_scores("no list here") is None
    assert _parse_scores("[5, 7]") is None
    assert _parse_scores("[]") is None

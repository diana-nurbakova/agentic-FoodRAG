import numpy as np

MIN_BIAS = np.float32(0)
WEIGHT_INIT = np.float32(1)
WEIGHT_INCREASE = np.float32(1.1)
WEIGHT_DECREASE = np.float32(0.9)
MIN_NUTRIENT_PENALTY = 1
MAX_NUTRIENT_PENALTY = 3
MAX_NUTRIENT_UPPER_BOUND_MULTIPLIER = 2


def match_restrictions(recipe, recipe_restrictions, reward, constraint_config):
    """
    Matches a recipe to a user's restrictions.
    :param recipe: recipe to be matched
    :param recipe_restrictions: restrictions of the user
    :param reward: reward for the recipe
    :param constraint_config: configuration of the general constraints
    :return: The weighted reward of the recipe. MIN_BIAS if the recipe does not match the restrictions at all.
    """
    # Check if the recipe contains any hard restrictions
    if not _match_hard_restrictions(recipe, recipe_restrictions):
        return MIN_BIAS

    # Check if the recipe contains any soft restrictions and adjust the reward accordingly
    weight = WEIGHT_INIT  # Initial weight, it must always be greater than 0
    weight = _match_soft_restrictions(recipe, recipe_restrictions, weight, constraint_config)

    # Adjust the reward
    assert weight > 0
    if reward >= 0:
        new_reward = reward * weight
    else:
        new_reward = reward / weight

    return new_reward


def _match_hard_restrictions(recipe, recipe_restrictions):
    # Disliked Ingredients
    disliked_ingredients = recipe_restrictions['ingredients'].get('dislikes', [])
    if any(ingredient in recipe['ingredient_names'] for ingredient in disliked_ingredients):
        return False

    # Guidelines (given as total values for the whole meal/serving)
    for nutrient, restriction in recipe_restrictions['guidelines'].items():
        whole_meal_value = recipe[nutrient] / 100 * recipe['servingSize [g]']

        if whole_meal_value < restriction['meal']['lower']:
            return False
        elif whole_meal_value > restriction['meal']['upper']:
            return False

    return True


def _match_soft_restrictions(recipe, recipe_restrictions, weight, constraint_config):
    # Liked Ingredients
    liked_ingredients = recipe_restrictions['ingredients'].get('likes', [])
    weight = _increase_weight(weight, times=sum(  # count the number of liked ingredients in the recipe
        1 for ingredient in liked_ingredients if ingredient in recipe['ingredient_names']))

    # Nutrients (Given per 100g per serving of the recipe)
    for nutrient, restriction in recipe_restrictions['nutrients'].items():
        restriction_values = constraint_config['LIMIT_NUTRIENT_VALUES'][nutrient][restriction]

        if recipe[nutrient] < restriction_values['lower']:
            times = _calculate_penalty(recipe[nutrient], restriction_values['lower'])
            weight = _decrease_weight(weight, times=times)
        elif recipe[nutrient] > restriction_values['upper']:
            times = _calculate_penalty(recipe[nutrient], restriction_values['upper'])
            weight = _decrease_weight(weight, times=times)

    return weight


def _increase_weight(weight, times=1):
    return weight * WEIGHT_INCREASE ** np.float32(times)


def _decrease_weight(weight, times=1):
    return weight * WEIGHT_DECREASE ** np.float32(times)


def _calculate_penalty(value, bound):
    if value < bound:  # bound == lower_bound
        distance = bound - value
        max_distance = max(bound, 1)
    elif value > bound:  # bound == upper_bound
        distance = value - bound
        max_distance = max(value - bound, bound * MAX_NUTRIENT_UPPER_BOUND_MULTIPLIER, 1)
    else:
        return 0

    # Calculate the proportion of the distance compared to the maximum distance
    proportion = min(1, distance / max_distance)

    # Calculate penalty points linearly based on the proportion
    penalty_points = max(MIN_NUTRIENT_PENALTY, min(MAX_NUTRIENT_PENALTY, round(proportion * MAX_NUTRIENT_PENALTY)))

    return penalty_points

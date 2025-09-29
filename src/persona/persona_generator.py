import functools
import numpy as np
import random
import sys

from src.utils import rdf_util


def generate_persona_preferences(persona_config, nutrition_config, hummus_data, food_on=None, print_persona=False):
    if food_on is None:
        food_on = rdf_util.PreProcessedFoodOn(hummus_data)

    # Generate persona consisting of random ingredient preferences, random nutrient preferences, and random tags
    tags = _get_random_items(food_on, persona_config, hummus_data, item_type='tags')
    ingredients = _get_random_items(food_on, persona_config, hummus_data, item_type='ingredients')
    relative_nutrients = _get_random_relative_nutrients(persona_config, nutrition_config)
    total_nutrients = _get_random_total_nutrients(persona_config, nutrition_config)
    nutrition_scores = _get_random_nutrition_scores(nutrition_config)

    persona = {
        'tags': tags,
        'ingredients': ingredients,
        'relative_nutrients': relative_nutrients,
        'total_nutrients': total_nutrients,
        'nutrition_scores': nutrition_scores
    }

    if print_persona:
        _print_persona(persona)

    return persona


def _random(n_min, n_max):
    if n_min >= n_max:
        return n_max  # ==> min(n_min, n_max)
    else:
        return np.random.choice(range(n_min, n_max + 1)).tolist()


def _get_random_items(food_on: rdf_util.PreProcessedFoodOn, persona_config, hummus_data, item_type):
    # Get generation params from configs
    min_likes = persona_config[item_type]['likes']['min_num']
    max_likes = persona_config[item_type]['likes']['max_num']
    min_dislikes = persona_config[item_type]['dislikes']['min_num']
    max_dislikes = persona_config[item_type]['dislikes']['max_num']
    invalid_items = persona_config[item_type]['invalid_items']

    # Get valid items
    if item_type == 'tags':
        available_items = hummus_data['recipe_tags']
    elif item_type == 'ingredients':
        include_food_on = persona_config[item_type]['include_food_on']
        include_food_com = persona_config[item_type]['include_food_com']
        available_items = []

        if include_food_on:
            available_items.extend(food_on.get_all_nodes(remove_unconnected=True))
        if include_food_com:
            available_items.extend(hummus_data['ingredients_df'][['ing_name']].drop_duplicates()['ing_name'].to_list())
    else:
        raise ValueError('Invalid item type: {}'.format(item_type))

    # Remove invalid items from valid items
    valid_items = list(set(available_items) - set(invalid_items))

    # max number of likes and dislikes must be less than the number of items
    if max_likes + max_dislikes > len(valid_items):
        # If this is the case, the corresponding items are not enough to generate
        if len(valid_items) > 2:  # As fallback, lets use for likes and dislikes at most only 1 item
            max_likes = 1
            max_dislikes = 1
        else:  # If there are only 2 items, we do not generate any likes and dislikes
            return {'likes': [],
                    'dislikes': []}

    # Randomly choose number of liked and disliked items
    num_likes = _random(min_likes, max_likes)
    num_dislikes = _random(min_dislikes, max_dislikes)

    # Randomly choose items
    item_likes = np.random.choice(valid_items, num_likes, replace=False).tolist()
    remaining_items = list(set(valid_items) - set(item_likes))
    item_dislikes = np.random.choice(remaining_items, num_dislikes, replace=False).tolist()

    return {'likes': item_likes,
            'dislikes': item_dislikes}


def _get_random_relative_nutrients(persona_config, nutrition_config):
    # Get generation params from configs
    min_nutr = persona_config['relative_nutrients']['min_num']
    max_nutr = persona_config['relative_nutrients']['max_num']
    possible_nutrients = persona_config['relative_nutrients']['possible_nutrients']
    all_nutrients = nutrition_config['ALL_NUTRITION_TYPES']
    valid_nutrients = list(set(all_nutrients).intersection(possible_nutrients))

    # Adapt limits if the number of valid nutrients is too small
    max_nutr = min(max_nutr, len(valid_nutrients))

    # Randomly choose number of liked and disliked nutrients
    num_nutr = _random(min_nutr, max_nutr)

    # Randomly choose nutrition types
    chosen_nutrients = np.random.choice(valid_nutrients, num_nutr, replace=False).tolist()
    nutrient_preferences = _get_relative_nutrient_preferences(chosen_nutrients, nutrition_config)

    return nutrient_preferences


def _get_relative_nutrient_preferences(sampled_nutrients, nutrition_config):
    # Randomly choose nutrition preference values and explicit ranges
    result = {}
    for nutrient in sampled_nutrients:
        # Randomly choose a general preference value
        preference = np.random.choice(['low', 'medium', 'high']).tolist()
        limits = nutrition_config['RELATIVE_NUTRIENTS'][nutrient][preference]

        # Add to result
        result[nutrient] = {
            'preference': preference,
            'limits': limits
        }

    return result


def _get_random_total_nutrients(persona_config, nutrition_config):
    # Get generation params from configs
    min_nutr = persona_config['total_nutrients']['min_num']
    max_nutr = persona_config['total_nutrients']['max_num']
    deviation_min = persona_config['total_nutrients']['deviation_min']
    deviation_max = persona_config['total_nutrients']['deviation_max']
    possible_nutrients = persona_config['total_nutrients']['possible_nutrients']
    all_nutrients = nutrition_config['ALL_NUTRITION_TYPES']
    valid_nutrients = list(set(all_nutrients).intersection(possible_nutrients))

    # Adapt limits if the number of valid nutrients is too small
    max_nutr = min(max_nutr, len(valid_nutrients))

    # Randomly choose number of liked and disliked nutrients
    num_nutr = _random(min_nutr, max_nutr)

    # Randomly choose nutrition types
    chosen_nutrients = np.random.choice(valid_nutrients, num_nutr, replace=False).tolist()
    nutrient_preferences = _get_total_nutrient_preferences(chosen_nutrients, nutrition_config, deviation_min,
                                                           deviation_max)

    return nutrient_preferences


def _get_total_nutrient_preferences(sampled_nutrients, nutrition_config, deviation_min, deviation_max):
    # Randomly choose nutrition preference values and explicit ranges
    result = {}

    for nutrient in sampled_nutrients:
        # Choose a random value between deviation_min and deviation_max
        preference_percentage = np.random.uniform(deviation_min, deviation_max)

        # Calculate the total nutrition
        total_nutrition = nutrition_config['TOTAL_NUTRIENTS'][nutrient] * preference_percentage

        # Add to result
        result[nutrient] = total_nutrition

    return result


def _get_random_nutrition_scores(nutrition_config):
    # Choose a random nutrition score
    score_name = np.random.choice(nutrition_config['ALL_NUTRITION_SCORES']).tolist()

    # Choose a random value between 0 and 1
    if score_name == 'nutri_score':  # Nutri score is the only one that is categorized with a small number of categories
        # score_value = np.random.choice(['A', 'B', 'C', 'D', 'E']).tolist()
        score_value = np.random.choice([1.0, 0.75, 0.5, 0.25, 0.0]).tolist()  # same but normalized
    else:  # All other scores are (semi) continuous (or have at least a large number of categories)
        score_value = np.random.uniform(0, 1)

    return {score_name: score_value}


def get_random_meal_plan_nutrients(persona_config, current_meal=None, today_intaken_kcal=None):
    """
    Generate a random meal plan for a persona based on the given persona configuration.

    :param persona_config: The persona configuration.
    :param current_meal: The current meal for which the meal plan should be generated. Randomly generated if None.
    :param today_intaken_kcal: The total calories already intaken today. If None, it is assumed that no calories have
        been intaken yet. If also current_meal is None, some random calories are calculated.
    :return: A dictionary containing the meal plan for the persona.
    """
    # Get user configurations
    kcal_per_day = persona_config.get('calories_per_day', np.random.randint(1600, 3000))
    calories_distribution_dict = {
        'breakfast': persona_config['calories_per_meal_distribution'].get('breakfast', 0.22),
        'lunch': persona_config['calories_per_meal_distribution'].get('lunch', 0.32),
        'dinner': persona_config['calories_per_meal_distribution'].get('dinner', 0.32),
        'snack 1': persona_config['calories_per_meal_distribution'].get('snack 1', 0.07),
        'snack 2': persona_config['calories_per_meal_distribution'].get('snack 2', 0.07)
    }

    # If no current meal is given, choose one randomly
    if current_meal is None:
        current_meal = np.random.choice(['breakfast', 'lunch', 'dinner', 'snack 1', 'snack 2']).tolist()

    # If no calories have been intaken yet, calculate some random calories
    if today_intaken_kcal is None:
        # Calculate how many calories could have been intaken today
        if current_meal == 'breakfast':
            possible_calories = 0
        elif current_meal == 'lunch':
            possible_calories = kcal_per_day * calories_distribution_dict['breakfast']
        elif current_meal == 'dinner':
            possible_calories = kcal_per_day * (
                    calories_distribution_dict['breakfast'] + calories_distribution_dict['lunch'])
        elif current_meal == 'snack 1':
            possible_calories = kcal_per_day * (
                    calories_distribution_dict['breakfast'] + calories_distribution_dict['lunch'] +
                    calories_distribution_dict['dinner'])
        elif current_meal == 'snack 2':
            possible_calories = kcal_per_day * (
                    calories_distribution_dict['breakfast'] + calories_distribution_dict['lunch'] +
                    calories_distribution_dict['dinner'] + calories_distribution_dict['snack 1'])
        else:
            raise ValueError('Invalid meal name: {}'.format(current_meal))

        if possible_calories > 0:
            today_intaken_kcal = np.random.randint(possible_calories * 0.8, possible_calories)
        else:
            today_intaken_kcal = 0

    # Get today's already intake calories and subtract them from the daily intake calories
    if today_intaken_kcal > 0:
        kcal_per_day -= today_intaken_kcal

    # Distribute calories to meals
    # If there are no information of the current meal also start with the breakfast
    if current_meal == 'breakfast' or current_meal == '':
        left_meals = ['breakfast', 'lunch', 'dinner', 'snack 1', 'snack 2']
        cal_dict = _update_cal_distributions_dict(left_meals, calories_distribution_dict)
        meal_plan = {
            'breakfast':
                {'meal_type': 'breakfast',
                 'guidelines': _get_cal_per_meal(kcal_per_day, 'breakfast', cal_dict)},
            'lunch':
                {'meal_type': 'lunch',
                 'guidelines': _get_cal_per_meal(kcal_per_day, 'lunch', cal_dict)},
            'dinner':
                {'meal_type': 'dinner',
                 'guidelines': _get_cal_per_meal(kcal_per_day, 'dinner', cal_dict)},
            'snack 1':
                {'meal_type': 'snack',
                 'guidelines': _get_cal_per_meal(kcal_per_day, 'snack 1', cal_dict)},
            'snack 2':
                {'meal_type': 'snack',
                 'guidelines': _get_cal_per_meal(kcal_per_day, 'snack 2', cal_dict)}}
    elif current_meal == 'lunch':
        left_meals = ['lunch', 'dinner', 'snack 1', 'snack 2']
        cal_dict = _update_cal_distributions_dict(left_meals, calories_distribution_dict)
        meal_plan = {
            'lunch':
                {'meal_type': 'lunch',
                 'guidelines': _get_cal_per_meal(kcal_per_day, 'lunch', cal_dict)},
            'dinner':
                {'meal_type': 'dinner',
                 'guidelines': _get_cal_per_meal(kcal_per_day, 'dinner', cal_dict)},
            'snack 1':
                {'meal_type': 'snack',
                 'guidelines': _get_cal_per_meal(kcal_per_day, 'snack 1', cal_dict)},
            'snack 2':
                {'meal_type': 'snack',
                 'guidelines': _get_cal_per_meal(kcal_per_day, 'snack 2', cal_dict)}}
    elif current_meal == 'dinner':
        left_meals = ['dinner', 'snack 1', 'snack 2']
        cal_dict = _update_cal_distributions_dict(left_meals, calories_distribution_dict)
        meal_plan = {
            'dinner':
                {'meal_type': 'dinner',
                 'guidelines': _get_cal_per_meal(kcal_per_day, 'dinner', cal_dict)},
            'snack 1':
                {'meal_type': 'snack',
                 'guidelines': _get_cal_per_meal(kcal_per_day, 'snack 1', cal_dict)},
            'snack 2':
                {'meal_type': 'snack',
                 'guidelines': _get_cal_per_meal(kcal_per_day, 'snack 2', cal_dict)}}
    elif current_meal == 'snack 1' or current_meal == 'snack':
        left_meals = ['snack 1', 'snack 2']
        cal_dict = _update_cal_distributions_dict(left_meals, calories_distribution_dict)
        meal_plan = {
            'snack 1':
                {'meal_type': 'snack',
                 'guidelines': _get_cal_per_meal(kcal_per_day, 'snack 1', cal_dict)},
            'snack 2':
                {'meal_type': 'snack',
                 'guidelines': _get_cal_per_meal(kcal_per_day, 'snack 2', cal_dict)}}
    elif current_meal == 'snack 2':
        left_meals = ['snack 2']
        cal_dict = _update_cal_distributions_dict(left_meals, calories_distribution_dict)
        meal_plan = {
            'snack 2':
                {'meal_type': 'snack',
                 'guidelines': _get_cal_per_meal(kcal_per_day, 'snack 2', cal_dict)}}
    else:
        raise ValueError('Invalid meal name: {}'.format(current_meal))

    return meal_plan


def _get_cal_per_meal(calories_per_day, meal_name, calories_distribution_dict):
    calories_per_meal = calories_per_day * calories_distribution_dict[meal_name]

    # Round to the nearest 10
    calories_per_meal = round(calories_per_meal)

    nutrients = {'calories [kcal]': calories_per_meal}

    return nutrients


def _update_cal_distributions_dict(left_meals, calories_distribution_dict):
    # Remove all meals that are not left
    filtered_dict = {key: value for key, value in calories_distribution_dict.items() if key in left_meals}

    # Update the distribution of the left meals
    reweighed_dict = _reweigh_cal_dist(filtered_dict)

    return reweighed_dict


def _reweigh_cal_dist(distributions):
    total = sum(distributions.values())
    if total != 1:
        distributions = {meal_name: dist / total for meal_name, dist in distributions.items()}
    return distributions


def choose_randomly(occurrence_dict, weight_pow=1.0, k_min=1, k_max=5):
    """
    Choose k random keys (in range of k_min and k_max) from a dictionary, where the probability of choosing a key is
    proportional to its value.
    :param occurrence_dict: A dict consisting of keys to choose from and integer values representing their number of
     occurrences. The more frequent a key is, the higher the probability of choosing it.
    :param weight_pow: The power to which the weights/values/occurrences of the keys are raised. The higher the power,
    the more likely it is to choose. E.g. if weight_pow=2, then the probability of choosing a key is proportional to its
     number of occurrences squared.
    :param k_min: Minimum number of keys to choose.
    :param k_max: Maximum number of keys to choose.
    :return: The set of randomly chosen keys.
    """
    k = random.randint(k_min, k_max)
    pow_func = functools.partial(lambda p, x: int(x ** p), weight_pow)
    weights = list(map(pow_func, list(occurrence_dict.values())))

    try:
        return set(random.choices(list(occurrence_dict.keys()), weights=weights, k=k))
    except IndexError:
        print('Error in choosing randomly. Returning the first k items.', file=sys.stderr)
        first_k = list(occurrence_dict.keys())[:k]
        return set(first_k)


def _print_persona(persona):
    print("\n\n".join("{}\n{}".format(k, v) for k, v in persona.items()))

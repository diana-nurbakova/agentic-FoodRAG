import numpy as np
import pandas as pd

from foodrag.utils import graphDB_util

CAL_TO_KJ_FACTOR = 4.184

FSA_SCORE = "fsa"
WHO_SCORE = "who"
NUTRI_SCORE = "nutri"


def _normalize(score, best_value):
    if best_value == 0:
        return 0
    else:
        return score / best_value


def _is_cheese(url):
    return graphDB_util.is_only_type(url, graphDB_util.FOOD_ON_CHEESE)


def _is_water(url):
    return graphDB_util.is_only_type(url, graphDB_util.FOOD_ON_WATER)


def _is_beverage(url):
    return graphDB_util.is_also_type(url, graphDB_util.FOOD_ON_CHEESE)


def _get_fruit_and_veggie_ratio(url):
    results = graphDB_util.get_specific_ingredients_types(url,
                                                          [graphDB_util.FOOD_ON_FRUIT, graphDB_util.FOOD_ON_VEGGIE])

    if len(results) > 0:
        # This assumption is a vague approximation
        ratio = results[results['type'].str.contains('true')]['type'].count() / len(results)
    else:
        ratio = 0

    return ratio


def _score_who_value(value, lower_bound, upper_bound, minimize=True, normalize=False):
    """
    Calculates a score for a single value. The resulting score is between [0,1] while 0 is the worst possible score, and
     1 the best.
    :param value: The to-be-scored nutrient value.
    :param lower_bound: The lower bound (based on the 100g normalization).
    :param upper_bound: The upper bound (based on the 100g normalization).
    :param minimize: Optional parameter which inverts the result when the nutrient value shall be maximized instead.
    :param normalize: If the result shall be normalized to [0,1]. Does only work for results which are numbers.
    :return: A score is 0, 1, or 2 (between [0,1] when normalized) while 0 is the worst possible score, and 1 the best.
    """
    if normalize:
        if value < lower_bound:  # score is very good (low/green category)
            score = 1
        elif value > upper_bound:  # score is very bad (high/red category)
            score = 0
        else:  # scores in the medium/amber category
            score = 1 - (value - lower_bound) / (upper_bound - lower_bound)

        if minimize:
            return score
        else:  # a score which maximizes the value (thus a high value instead a low one is good) is inverted
            return 1 - score
    else:
        if minimize:
            if value < lower_bound:
                return 2
            elif value < upper_bound:
                return 1
            else:
                return 0
        else:
            if value < lower_bound:
                return 0
            elif value < upper_bound:
                return 1
            else:
                return 2


def _score_fsa_value(value, lower_bound, upper_bound, upper_bound_per_portion, serving_size, normalization_comment,
                     normalize=False):
    """
    Calculates a score for a single value. The resulting score is between [0,1] while 0 is the worst possible score, and
     1 the best.

    :param value: The to-be-scored nutrient value.
    :param lower_bound: The lower bound (based on the 100g normalization).
    :param upper_bound: The upper bound (based on the 100g normalization).
    :param upper_bound_per_portion: The upper bound per portion.
    :param serving_size: The size in grams one serving has.
    :param normalization_comment: The comment from the normalization.
    :param normalize: If the result shall be normalized to [0,1]. Does only work for results which are numbers.
    :return: A score is 0, 1, or 2 (between [0,1] when normalized) while 0 is the worst possible score, and 1 the best.
    """
    # re-factor the normalization process and calculate the nutrients as originally given (as nutrients per portion)
    if normalization_comment == '' and serving_size > 0:
        value_per_portion = value * serving_size / 100
    else:  # for recipes with problematic normalization return an invalid score
        return float('nan')

    if value_per_portion > upper_bound_per_portion:
        return 0
    else:
        return _score_who_value(value, lower_bound, upper_bound, normalize=normalize)


def _score_nutri_value(value, boundaries, results, else_result, op='smaller_equal', linear_ascent=False,
                       normalize=False):
    """
    Checks in which boundary the value is. Returns the calculated score.

    Also refer to:
    Chantal, J., & Hercberg, S. (2017). Development of a new front-of-pack nutrition label in France: the five-colour
    Nutri-Score. Public Health Panorama, 03(04), 712–725.

    :param value: The to-be-scored nutrient value.
    :param boundaries: All boundaries which shall be checked. The boundaries MUST be ascending.
    :param results: All results as a set corresponding to the boundaries. MUST be of the same size as boundaries.
    :param else_result: The result to return if no boundary was valid.
    :param op: The operation which defines how the value shall be checked against the boundary. Available: smaller,
        smaller_equal, equal
    :param linear_ascent: If the resulting score should be the plain value as given in result, or linear weighted. Does
        only work for results which are numbers.
    :param normalize: If the result shall be normalized to [0,1]. Does only work for results which are numbers.
    :return: Returns the calculated score, in bounds of minimal and maximal values of the results set and the
        else_result.
    """
    # look for the index of the result
    try:  # get first element (all valid indices) and get first element (first fitting boundary)
        if op == 'smaller':
            idx = np.where(value < np.array(boundaries))[0][0]
        elif op == 'smaller_equal':
            idx = np.where(value <= np.array(boundaries))[0][0]
        elif op == 'equal':
            idx = np.where(value == np.array(boundaries))[0][0]
        else:
            raise AttributeError('Do not support operation: ' + op)
    except IndexError:  # no index was found, return the else result
        return else_result

    # get the corresponding result element
    result = results[idx]

    if linear_ascent:
        upper_bound = boundaries[idx]
        if idx - 1 >= 0:
            lower_bound = boundaries[idx - 1]
        elif upper_bound != 0:
            lower_bound = 0
        else:
            raise AttributeError('Do not support upper bounds of value 0')
        result = result * (value - lower_bound) / (upper_bound - lower_bound)

    if normalize:
        return _normalize(result, max(max(results), else_result))
    else:
        return result


def who_score(protein, total_carbohydrate, sugars, total_fat, saturated_fat, dietary_fiber, sodium, serving_size,
              normalization_comment, normalize=False):
    """
    Calculates the WHO-Score. The range is 0-14, with 14 as the best.

    :param protein: The proteins in g per 100 g.
    :param total_carbohydrate: The carbohydrates in g per 100 g.
    :param sugars: The sugar in g per 100 g.
    :param total_fat: The fat in g per 100 g.
    :param saturated_fat: The saturated fat in g per 100 g.
    :param dietary_fiber: The dietary fiber in g per 100 g.
    :param sodium: The sodium in g per 100 g.
    :param serving_size: The size of a single portion.
    :param normalization_comment: The comment from the normalization.
    :param normalize: Whether the result shall be normalized in the range [0,1] or not. Default is False.
    :return: The WHO-Score
    """

    # WHO score requires the daylie sodium value. As there are no user information here, we take the next best value-
    # the sodium amount per portion (assuming the user eats one portion)
    # Also, re-factor the normalization process.
    if normalization_comment == '' and serving_size > 0:
        sodium_per_serving = sodium / 100 * serving_size
    else:  # for recipes with problematic normalization return an invalid score
        sodium_per_serving = float('nan')

    score = sum([_score_who_value(protein, lower_bound=10, upper_bound=15, normalize=normalize, minimize=False),
                 _score_who_value(total_carbohydrate, lower_bound=55, upper_bound=75, normalize=normalize,
                                  minimize=False),
                 _score_who_value(sugars, lower_bound=0, upper_bound=10, normalize=normalize),
                 _score_who_value(total_fat, lower_bound=15, upper_bound=30, normalize=normalize),
                 _score_who_value(saturated_fat, lower_bound=0, upper_bound=10, normalize=normalize),
                 _score_who_value(dietary_fiber, lower_bound=0, upper_bound=3, normalize=normalize, minimize=False),
                 _score_who_value(sodium_per_serving, lower_bound=0, upper_bound=2)])

    if normalize:
        return _normalize(score, 14)
    else:
        return score


def fsa_score(total_fat, saturated_fat, sugars, sodium, normalization_comment, serving_size, normalize=False):
    """
    Calculates the FSA-Score. The range is 0-8, with 8 as the best.

    :param total_fat: The fat in g per 100 g.
    :param saturated_fat: The saturated fat in g per 100 g.
    :param sugars: The sugar in g per 100 g.
    :param sodium: The sodium in g per 100 g.
    :param normalization_comment: The comment for the normalization process. Invalidates invalid scores.
    :param serving_size: The size of one serving in g.
    :param normalize: Whether the result shall be normalized in the range [0,1] or not. Default is False.
    :return: The FSA-Score
    """
    score = sum([_score_fsa_value(total_fat, lower_bound=3, upper_bound=17.5, upper_bound_per_portion=21,
                                  serving_size=serving_size, normalization_comment=normalization_comment,
                                  normalize=False),
                 _score_fsa_value(saturated_fat, lower_bound=1.5, upper_bound=5, upper_bound_per_portion=6.0,
                                  serving_size=serving_size, normalization_comment=normalization_comment,
                                  normalize=False),
                 _score_fsa_value(sugars, lower_bound=5, upper_bound=22.5, upper_bound_per_portion=27,
                                  serving_size=serving_size, normalization_comment=normalization_comment,
                                  normalize=False),
                 _score_fsa_value(sodium, lower_bound=0.3, upper_bound=1.5, upper_bound_per_portion=1.8,
                                  serving_size=serving_size, normalization_comment=normalization_comment,
                                  normalize=False)])
    if normalize:
        return _normalize(score, 8)
    else:
        return score


def nutri_score(calories, saturated_fat, sugars, protein, fiber, sodium, food_com_url, normalize=False,
                complete_score=False):
    """
    Calculates the Nutri-Score. A is the best result, E is the worst result.

    Note: As the lipids are not known, the fat/lipids ratio cannot be calculated and thus the NutriScore version without
    lipids is chosen.

    :param calories: The total calories in cal.
    :param saturated_fat: The saturated fat in g per 100 g.
    :param sugars: The sugar in g per 100 g.
    :param protein: The proteins in g per 100 g.
    :param fiber: The dietary fiber in g per 100 g.
    :param sodium: The sodium in g per 100 g.
    :param food_com_url: The url of the recipe in the knowledge graph.
    :param normalize: Whether the result shall be normalized in the range [0,1] or not. Default is False.
    :param complete_score: Whether the full score shall be calculated or not. If True, it requires a connection to the
     HUMMUS KG to get information related to cheese, beverages and fruits. Default is False. The expected repository
      name is stored in the graphDB_util.
    :return: The Nutri-Score
    """
    # Retrieves the recipes types for the complete score
    if complete_score:
        is_beverage = _is_beverage(food_com_url)
        is_cheese = _is_cheese(food_com_url)
        is_water = _is_water(food_com_url)
        fruit_veggie_ratio = _get_fruit_and_veggie_ratio(food_com_url) * 100  # in %
    else:  # default values
        is_beverage = False
        is_cheese = False
        is_water = False
        fruit_veggie_ratio = 0

    # Define the dict for the NUTRI score
    nutri_to_int_dict = {"A": 4, "B": 3, "C": 2, "D": 1, "E": 0}

    # Points A.a
    energy = calories * CAL_TO_KJ_FACTOR  # Calc. the energy from the calories value
    if is_beverage:
        energy_score = _score_nutri_value(energy,
                                          boundaries=[0, 30, 60, 90, 120, 150, 180, 210, 240, 270],
                                          results=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], else_result=10)
    else:  # if no beverage or no complete score
        energy_score = _score_nutri_value(energy,
                                          boundaries=[335, 670, 1005, 1340, 1675, 2010, 2345, 2680, 3015, 3350],
                                          results=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], else_result=10)

    # Points A.b
    if is_beverage:
        sugar_score = _score_nutri_value(sugars,
                                         boundaries=[0, 1.5, 3, 4.5, 6, 7.5, 9, 10.5, 12, 13.5],
                                         results=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], else_result=10)
    else:  # if no beverage or no complete score
        sugar_score = _score_nutri_value(sugars,
                                         boundaries=[4.5, 9, 13.5, 18, 22.5, 27, 31, 36, 40, 45],
                                         results=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], else_result=10)

    # Points A.c
    # Lipids are not available, just takes the default fat score calculation scheme
    fat_score = _score_nutri_value(saturated_fat,
                                   boundaries=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                                   results=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], else_result=10)

    # Points A.d
    sodium_score = _score_nutri_value(sodium,
                                      boundaries=[0.09, 0.18, 0.27, 0.36, 0.45, 0.54, 0.63, 0.72, 0.81, 0.9],
                                      results=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], else_result=10)

    # Points C.a
    if complete_score:
        # Calculate the fruit_veg_score
        if is_beverage:
            fruit_veg_score = _score_nutri_value(fruit_veggie_ratio,
                                                 boundaries=[40, 60, 80],
                                                 results=[0, 2, 4], else_result=10)
        else:
            fruit_veg_score = _score_nutri_value(fruit_veggie_ratio,
                                                 boundaries=[40, 60, 80],
                                                 results=[0, 1, 2], else_result=5)
    else:
        # We assume a fruit_veg_score of 0 which is true for all recipes which  consist of <= 40% fruits/vegetables
        fruit_veg_score = 0

    # Points C.b
    fiber_score = _score_nutri_value(fiber,
                                     boundaries=[0.7, 1.4, 2.1, 2.8, 3.5],
                                     results=[0, 1, 2, 3, 4], else_result=5)

    # Points C.c
    protein_score = _score_nutri_value(protein,
                                       boundaries=[1.6, 3.2, 4.8, 6.4, 8],
                                       results=[0, 1, 2, 3, 4], else_result=5)

    # Get final score. Refer to the official NutriScore scheme
    points_a = energy_score + sugar_score + fat_score + sodium_score
    points_c = fruit_veg_score + fiber_score + protein_score

    if points_a < 11 or is_cheese:
        points = points_a - points_c
    else:
        if fruit_veg_score >= 5:
            points = points_a - points_c
        else:
            points = points_a - (fiber_score + fruit_veg_score)

    # Get score class. Refer to the official NutriScore scheme
    if is_beverage:
        score = _score_nutri_value(points, boundaries=[-1, 2, 10, 18], results=['A', 'B', 'C', 'D'], else_result='E')
    else:
        if is_water:
            score = 'A'
        else:
            score = _score_nutri_value(points, boundaries=[1, 5, 9], results=['B', 'C', 'D'], else_result='E')

    # Return the result and normalize if necessary
    if normalize:
        return _normalize(nutri_to_int_dict[score], 4)
    else:
        return score


def calculate_food_scores(recipes, normalized_ingredients, score_names, normalize=False,
                          complete_score=False):
    """
    Calculates a score for all recipes.

    :param recipes: The recipes including the nutrient features.
    :param normalized_ingredients: A dataframe of normalized ingredients to use. Must have same number of rows and
        ordering as recipes.
    :param score_names: The names of the to-be-used scores. Available: 'who', 'fsa', 'nutri'
    :param normalize: Whether to normalize the scores to a range of 0 to 1.
    :param complete_score: Whether the full score shall be calculated or not. If True, it requires a connection to the
        HUMMUS KG to get information related to cheese, beverages and fruits. Default is False. The expected repository
        name is stored in the graphDB_util.
    :return: The original recipe dataframe in addition to the new score column.
    """

    if WHO_SCORE in score_names:
        nutrients = normalized_ingredients[['protein [g]', 'totalCarbohydrate [g]', 'sugars [g]', 'totalFat [g]',
                                            'saturatedFat [g]', 'dietaryFiber [g]', 'sodium [g]', 'servingSize [g]',
                                            'normalization_comment']]
        scores = [who_score(row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7], row[8], normalize=normalize)
                  for row in nutrients.to_numpy()]
        recipes['who_score'] = scores

    if FSA_SCORE in score_names:
        nutrients = pd.concat([normalized_ingredients[['totalFat [g]', 'saturatedFat [g]', 'sugars [g]', 'sodium [g]',
                                                       'normalization_comment']],
                               recipes[['servingSize [g]']]], axis=1, join="inner")
        scores = [fsa_score(row[0], row[1], row[2], row[3], row[4], row[5], normalize=normalize) for row in
                  nutrients.to_numpy()]
        recipes['fsa_score'] = scores

    if NUTRI_SCORE in score_names:
        nutrients = pd.concat([normalized_ingredients[['calories [kcal]', 'saturatedFat [g]', 'sugars [g]',
                                                       'protein [g]', 'dietaryFiber [g]', 'sodium [g]']],
                               recipes[['recipe_url']]], axis=1, join="inner")
        scores = [nutri_score(row[0], row[1], row[2], row[3], row[4], row[5], row[6], normalize=normalize,
                              complete_score=complete_score) for row in nutrients.to_numpy()]
        recipes['nutri_score'] = scores

    recipes['normalization_comment'] = normalized_ingredients[['normalization_comment']]

    return recipes

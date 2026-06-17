import collections
from itertools import chain

import pandas as pd

from foodrag.utils import hummus_util


def get_recipe_interactions(user_id, hummus_data):
    recipe_user_interactions = hummus_data['reviews_df']
    recipes = hummus_data['recipes_df']

    # Filter all interactions of given user
    filtered_interactions = recipe_user_interactions[recipe_user_interactions['member_id'] == user_id]

    # Merge interactions with corresponding recipes
    recipe_interactions = pd.merge(filtered_interactions, recipes, on='recipe_id')

    return recipe_interactions


def get_tag_interactions(interactions, query_config):
    # Get all interacted tags and their occurrences
    tag_occurrence_dict = dict(collections.Counter(tag for tags in interactions['tags'] for tag in tags))

    # Remove ignored tags
    tag_occurrence_dict = dict((key, value) for key, value in tag_occurrence_dict.items()
                               if key not in query_config['ignore_tags'])

    return tag_occurrence_dict


def get_ingredient_interactions(interactions, hummus_data, query_config):
    # Ingredient ID to FoodKG FoodOn-Super-Classes
    food_class_dict = hummus_util.merge_dicts(hummus_data['ingredients_dict'], hummus_data['label_dict'])

    # Get all interacted ingredients
    ingredients = hummus_data['ingredients_df']
    ingredients = ingredients[ingredients['recipe_id'].isin(interactions['recipe_id'].unique())]
    # Remove ignored ingredients
    ingredients = ingredients[~ingredients['ing_name'].isin(query_config['ignore_ingredients'])]

    # Get the ingredient occurrences
    ing_occurrence_dict = dict(collections.Counter(ingredient for ingredient in ingredients['ing_name']))

    # Get all interacted FoodOn-Super-Classes and their occurrences
    all_categories = list(chain.from_iterable([food_class_dict[ing] for ing in ingredients['ingredient_id']]))
    food_classes_occurrence_dict = dict(collections.Counter(category for category in all_categories))
    # Remove ignored classes
    food_classes_occurrence_dict = dict((key, value) for key, value in food_classes_occurrence_dict.items()
                                        if key not in query_config['ignore_food_classes'])

    return ing_occurrence_dict, food_classes_occurrence_dict

import ast
import json
import random
import re
import warnings
from llama_index.core import Prompt
from llama_index.core.llms import LLM
from typing import Dict

from src.food_rag.meal_plan.meal_type import MealType

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pandarallel import pandarallel
from src.utils import recipe_similarity_util
from tqdm.notebook import tqdm
from src.utils import score_util

import sys, os
from src.utils import tag_util
from src.utils import load_util
from src.utils import cluster_util

MEAL_TYPE_ANNOTATE_TEMPLATE = Prompt(
    'For given recipe find fitting meal types of the recipe.\n'
    'The meal type can be one or multiple of ' + ', '.join(MealType.get_all_str()) +
    '.\nReturn the corresponding meal type/types as a json, e.g.\n'
    '```json\n["breakfast", "snack"]\n```.'
    '\nRecipe: {recipe}\nMeal Types: ')


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


class IdDict(dict):
    def __missing__(self, key):
        return key


def combine_dicts(dict1: Dict, dict2: Dict) -> Dict:
    """
    Combines two dictionaries by merging the values of the same keys, e.g.:
    dict1 = {'a': 1, 'b': 2}
    dict2 = {'a': 3, 'b': 4}
    result_dict = {'a': [1, 3], 'b': [2, 4]}

    Note: Does not handle nested dictionaries or dictionaries with lists as values. They are kept as they are
    and treated as a single value.

    :param dict1: The first dictionary.
    :param dict2: The second dictionary.
    :return: The combined dictionary.
    """
    return {key: dict1.get(key, []) + dict2.get(key, []) for key in set(list(dict1.keys()) + list(dict2.keys()))}


def merge_dicts(dict1: Dict, dict2: Dict, none_value=[]) -> Dict:
    """
    Merge/chain the dictionaries using dictionary unpacking in a comprehension, e.g.:
    dict1 = {'a': 1, 'b': 2}
    ict2 = {'1': I, '2': II}
    result_dict = {'a': I, 'b': II}

    :param dict1: The first dictionary.
    :param dict2: The second dictionary.
    :param none_value: The value to be used when a key is not present in the first dictionary.
    :return: The merged dictionary.
    """

    if dict1 is None or isinstance(dict1, IdDict):
        return dict2
    else:
        result_dict = {key: dict2.get(dict1[key], none_value) for key in dict1}

    return result_dict


def invert_dict(original_dict: Dict) -> Dict:
    """
    Inverts a dictionary, e.g.:
    original_dict = {'a': 1, 'b': 2}
    inverted_dict = {1: 'a', 2: 'b'}

    When the values are not unique, the inverted dictionary will have lists as values, e.g.:
    original_dict = {'a': 1, 'b': 1}
    inverted_dict = {1: ['a', 'b']}

    :param original_dict: The original dictionary.
    :return: The inverted dictionary.
    """
    if original_dict is None or isinstance(original_dict, IdDict):
        return original_dict
    else:
        lists_as_values = any(isinstance(value, list) for value in original_dict.values())  # Test if values are lists

        if not lists_as_values:
            unique_values = len(original_dict.values()) == len(set(original_dict.values()))  # Test if values are unique

            if unique_values:  # dict is default invertible
                return {v: k for k, v in original_dict.items()}
            else:  # no unique values
                return _invert_not_unique_dict(original_dict)

        else:  # values as lists
            return _invert_list_dict(original_dict)


def _invert_not_unique_dict(original_dict):
    inverted_dict = {}

    for key, value in original_dict.items():
        inverted_dict.setdefault(value, []).append(key)

    return inverted_dict


def _invert_list_dict(original_dict):
    inverted_dict = {}
    all_values = [value for value_list in original_dict.values() for value in value_list]
    values_distinct = len(all_values) == len(set(all_values))

    if values_distinct:  # all values are distinct
        for key, values in original_dict.items():
            for item in values:
                inverted_dict[item] = key
    else:
        for key, values in original_dict.items():
            for item in values:
                inverted_dict.setdefault(item, []).append(key)

    return inverted_dict


def replace_nan_with_empty_list(value):
    return [] if isinstance(value, float) else value


def try_parse(string_object):
    try:
        return json.loads(string_object)
    except json.JSONDecodeError:
        try:
            return ast.literal_eval(string_object)
        except SyntaxError:
            raise SyntaxError('String not parseable.')
        except ValueError:
            raise SyntaxError('String not parseable.')


def _parse_string_list(string_object):
    try:
        return try_parse(string_object)
    except SyntaxError:
        try:
            return try_parse(string_object.replace(',Serve.', ''))
        except SyntaxError:
            try:
                return [s.replace('"', "").replace('\'', "").replace('[', "").replace(']', "").strip() for s in
                        string_object.split('","')]
            except Exception:
                return None


def _parse_string_triple(string_object):
    try:
        return try_parse(string_object)
    except SyntaxError:
        try:
            # try to parse fraction numbers
            try:
                temp_r = string_object.split('",', 1)[1]
            except IndexError:
                temp_r = string_object.split(',', 1)[1]
            try:
                temp_fraction = temp_r.split(',"', 1)[0]
                if len(temp_fraction) == 1:
                    temp_fraction = temp_r.split(',', 1)[0]
            except IndexError:
                temp_fraction = temp_r.split(',', 1)[0]
            numbers = temp_fraction.split('/')
            fraction = float(numbers[0]) / float(numbers[1])

            # rebuild string
            split_string_object = string_object.split(temp_fraction, 1)
            new_string_object = split_string_object[0] + str(fraction) + split_string_object[1]

            # parse with ast again....
            return ast.literal_eval(new_string_object)

        except Exception:
            splits = re.split('",|,"|\(,|,[0-9]|[0-9],', string_object)

            if len(splits) > 3:
                number_of_join = len(splits) - 2
                new_splits = []
                new_splits.append(splits[0])
                new_splits.append(splits[1])
                new_splits.append(''.join(splits[number_of_join:]))  # add the last n'th elements together
                splits = new_splits

            return [s.replace('"', "").replace('\'', "").replace('[', "")
                    .replace(']', "").replace('(', "").replace(')', "").strip() for s in splits]


def _try_split(split_words, name):
    for split_word in split_words:
        split_word = split_word + ' '
        try:
            quantity_part, name_part = name.split(split_word, 1)
            quantity_part = quantity_part + split_word
            return quantity_part, name_part
        except ValueError:
            pass

    # if nothing worked use default case
    return '', name


def _shift_quantity_information(ingredient_triple):
    category = ingredient_triple[0]
    quantity = ingredient_triple[1]
    name = ingredient_triple[2]

    # Build sequential list to split, start with entities like 'bag' to split, if this does not work use the units
    # and else ')' to split
    split_words = ['bag', 'can', 'cup', 'bags', 'cans', 'cups', 'ounces', 'ounce', 'pound', 'pounds', 'teaspoon',
                   'teaspoons', 'tablespoon', 'tablespoons', 'g', 'kg', 'lb', 'lbs', 'ml', 'oz', 'pint', 'tbsp', 'tsp',
                   ')']

    # Split quantity from name
    quantity_part, name_part = _try_split(split_words, name)

    # Add quantity part to the overall quantity
    quantity = str(quantity) + ' time(s) ' + quantity_part

    # Remove quantity part from name
    name = name_part.strip()

    return category, name, quantity


def _is_ingredient_in_key(ingredient, key):
    if len(ingredient) == 3:
        ingr_key = ingredient[0]

        if ingr_key == key:
            try:
                amount = float(ingredient[1])
            except ValueError:
                amount = ingredient[1]
            name = ingredient[2]
            return amount, name
        else:
            return None
    else:  # ingredient not valid, parse was not successful
        return None


def _get_ingredients_for_key(ingredients, key):
    keyed_ingredients = [_is_ingredient_in_key(ingredient, key) for ingredient in ingredients]
    keyed_ingredients = [x for x in keyed_ingredients if x is not None]

    return keyed_ingredients


def _count_parsed_list(parsed_list):
    if parsed_list is not None:
        return int(len(parsed_list))
    else:
        return -1


def _count_ingredient_list(ingredient_dict, parse_as_dict):
    if ingredient_dict is not None:
        if parse_as_dict:
            return sum([len(sub_ingredients) for sub_ingredients in ingredient_dict.values()])
        else:
            return len(ingredient_dict)
    else:
        return -1


def _parse_layered_ingredients(string_object, parse_as_dict=True):
    try:
        string_ingredients = json.loads(string_object)
        parsed_ingredients = [_parse_string_triple(ingredient) for ingredient in string_ingredients]

        # shift quantity information from ingredient name to quantity field
        parsed_ingredients = [_shift_quantity_information(ingredient_triple) for ingredient_triple in
                              parsed_ingredients]

        if parse_as_dict:
            # get all keys, set comprehension to ensure uniqueness, also do not include invalid keys
            keys = {ingredient[0] for ingredient in parsed_ingredients if isinstance(ingredient[0], str)}

            # add ingredients to keys
            ingredient_dict = {key: _get_ingredients_for_key(parsed_ingredients, key) for key in keys}

            return ingredient_dict
        else:
            ingredient_list = [ingredient[1] for ingredient in parsed_ingredients]
            return ingredient_list
    except json.JSONDecodeError:
        return None


def _lookup_index(row, dict):
    try:
        return dict[row[0]]
    except KeyError:
        return -1


def _row_is_in_tuple(row, item_user_set):
    is_in = (row[0], row[1]) in item_user_set
    return is_in


def food_locator_to_food_com(food_locator_url):
    return food_locator_url.replace('http://idea.rpi.edu/heals/kb/recipe/', 'https://www.food.com/recipe/')


def load_and_clean_data(data_location, graph_location, additional_location, s3=None, add_recipe_columns=None,
                        add_review_columns=None, comment_relations=2.5, authorship_relations=-1, recipe_tags=False,
                        add_additional_meal_type_labels=True, discard_non_tagged_recipes=False, k_user=2, k_recipe=2,
                        n_clusters=-1, n_components=2, clean_nan_strings=True, exclude_author_ids=True,
                        keep_old_ids=False, parse_as_dict=True, make_titles_unique=False, food_class_strategy='None',
                        clean_food_category=True, similarity_thresholds=False, feature_weights=None, include_usda=True,
                        llm: LLM = None, do_annotate_meal_type=False, create_random_meal_types=False, debug=False):
    """
    Loads and cleans the dataset from food.com (including possibly some information from other sources)

    0. Load raw data.
    1. [Optional] Either removes comments (== ratings of value 0) or modifies those to a certain rating value.
    2. [Optional] Adds authorship relations (from the recipe data) as rating relations.
    3. [Optional] Adds tags for recipes from another food.com dataset (https://www.aclweb.org/anthology/D19-1613/) and
        possibly discards recipes without tags.
    4. [Optional] Uses threshold discarding with specified minimum of user (k_user) and recipe (k_recipe) interactions.
    5. [Optional] Clusters users into max. n_clusters clusters with n_components dimensions. When either n_clusters or
        n_components are 0 or negative, no clustering is performed.
    6. Reset indices of members and recipes to be in a continuous space starting by 0.
    7. [Optional] Cleans nan strings (replaces them by empty strings).
    8. Parse average recipe rating and amount
    9. Parses nutrients (multiple columns in grams and milligrams respectively)
    10. Computes a dataframe consisting of nutrients normalized on 100g.
    11. Calculates food scores.
    12. Cleans durations dates (minutes) into easy-to-access formats.
    13. Counts the directions and ingredients sizes for each recipe.
    14. [Optional] Makes titles unique by adding a number to the end of the title.
    15. Get all filtered foodKG ingredient links.
    16. Process food/graph labels.
    17. Processes tags.
    18. Adds foodKG recipe ingredients to recipes_df.
    19. [Optional] Calculates similarities between recipes based on the features of the recipes.
    20. [Optional] Annotates recipes with meal types.

    :param data_location: Name of the directory where the data is located in.
    :param graph_location: Name of the directory where the graph data is located in.
    :param additional_location: Name of the directory where the additional food.com dataset is located in
        (https://www.aclweb.org/anthology/D19-1613/).
    :param s3: The s3 connection. When the connection is not None, tries to use this connection for data loading instead
        local data loading.
    :param add_recipe_columns: List of additional/non-standard columns/features which shall be present in the
        recipe dataset. None or [] if not necessary.
    :param add_review_columns: List of additional/non-standard columns/features which shall be present in the
        review dataset. None or [] if not necessary.
    :param comment_relations: Defines whether the comment relations shall be added:
        -1 --> not added,
        any other number >= 0 --> add those relations with the given rating.
    :param authorship_relations: Defines whether the authorship relations shall be added:
        -1 --> not added,
        any other number >= 0 --> add those relations with the given rating.
    :param recipe_tags: When recipe_tags are true, add a tags column from another food.com dataset for most recipes,
        skips otherwise.
    :param add_additional_meal_type_labels: If True, additional meal type labels will be added.
    :param discard_non_tagged_recipes: Whether recipes without tags shall be discarded. Is only relevant when
        additional_location is given to load tags from another dataset.
    :param k_recipe: The required minimum recipe interactions.
    :param k_user: The required minimum user interactions.
    :param n_clusters: Number of clusters for the user clustering. Should be at least 2. 0 or any negative number
     disables clustering. N_clusters is the maximum number of clusters to be created, some clusters may be empty and
     therefore not created
    :param n_components: Number of components for the user clustering for latent features. Must be at least 2. 0 or any
     negative number disables clustering.
    :param clean_nan_strings: Whether nan strings shall be replaced by empty strings.
    :param exclude_author_ids: Whether the author ids shall also be remapped. When True and given authors did not write
        reviews (or excluded via threshold discarding), then the author ids will be invalid.
    :param keep_old_ids: Whether old ids should be kept or discarded.
    :param parse_as_dict: Whether ingredients shall be parsed as a dict including additional information instead of a
        simple list of ingredients.
    :param make_titles_unique: Whether titles shall be made unique by adding a number to the end of the title.
    :param food_class_strategy: If 'None' the food classes will not be integrated directly in tha dataset (but still
     returned). If 'Ingredients' the food classes will be integrated as ingredients. If 'Tags' the food classes will
        be integrated as tags. If 'Both' the food classes will be integrated as ingredients and tags.
    :param clean_food_category: If True, the food categories will be cleaned.
    :param similarity_thresholds: The thresholds for the similarities of recipes. If None default thresholds are chosen.
     If false, no similarities are calculated. Note: This process takes a lot of RAM.
    :param feature_weights: The weights dict for the features of the recipes for similarity calculation. If None default
     weights are chosen.
    :param include_usda: If True, the usda data will be included in the output. Note: Requires a lot of RAM.
    :param llm: The LLM instance for the meal type annotation. If None, no meal type annotation will be performed.
    :param do_annotate_meal_type: If True, the meal types will be annotated.
    :param create_random_meal_types: If True, random meal types will be created if do_annotate_meal_type is True.
    :param debug: If debug messages shall be printed.
    :return:
        A dict including all relevant hummus data like e.g. recipes, reviews, users, recipes_dict_inv.
    """
    # Default column names of recipes & reviews/ratings
    recipe_column_names = ['recipe_id', 'title', 'description', 'author_id', 'duration', 'nutrition', 'directions',
                           'ingredients', 'serves', 'rating', 'last_changed_date', 'food_kg_locator', 'recipe_url']
    review_column_names = ['member_id', 'recipe_id', 'rating', 'text', 'likes', 'last_modified_date']
    member_column_names = ['member_id', 'member_url', 'member_name', 'member_description', 'member_status',
                           'member_avg_rating', 'member_map_pin', 'member_joined', 'follows_count', 'follow_me_count',
                           'avatar_url']
    if add_recipe_columns is not None:
        recipe_column_names.extend(add_recipe_columns)
    if add_review_columns is not None:
        review_column_names.extend(add_review_columns)

    # 0. Load raw data
    raw_data = _load_raw_hummus(data_location, graph_location, additional_location, recipe_column_names,
                                review_column_names, member_column_names, include_usda, s3, debug)

    # 1. Reparse invalid ratings/comments with value 0 to another value based on param
    reviews_df = _parse_comments(raw_data['reviews_df'], comment_relations, debug)

    # 2. Add authorship relations as rating based on param
    reviews_df = _parse_authorship(raw_data['recipes_df'], reviews_df, authorship_relations, debug)

    # 3. Add recipe tags from other dataset based on param (and possibly discards non-tagged recipes)
    recipes_df, reviews_df = _add_external_tags(additional_location, raw_data['recipes_df'], reviews_df, recipe_tags,
                                                add_additional_meal_type_labels, discard_non_tagged_recipes, s3, debug)

    # 4. Threshold discarding
    recipes_df, reviews_df, users_df = _threshold_discarding_init(k_recipe, k_user, recipes_df, reviews_df,
                                                                  raw_data['users_df'], debug)

    # 5. Generate dicts to fastly lookup the new indices
    recipes_df, reviews_df, users_df, recipes_dict, users_dict = _reindex(recipes_df, reviews_df, users_df,
                                                                          exclude_author_ids, keep_old_ids, debug)

    # 6. Clustering and reindexing
    recipes_df, reviews_df, users_df, recipes_dict, users_dict = _cluster_users(recipes_df, reviews_df, users_df,
                                                                                n_clusters, n_components, recipes_dict,
                                                                                users_dict, debug)

    # 7. Parse average recipe rating and amount
    recipes_df, reviews_df, users_df = _clean_nan_strings(recipes_df, reviews_df, users_df, clean_nan_strings, debug)

    # 8. Parse average recipe rating and amount
    _parse_recipe_rating_and_amount(recipes_df, debug)

    # 9. Parse nutrition elements
    _parse_nutrients(recipes_df, debug)

    # 10. Compute a dataframe consisting of nutrients normalized on 100g (takes ~15min for k=(1,1))
    normalized_nutrients = _init_normalized_nutrients(recipes_df, debug)

    # 11. Calculate food scores
    recipes_df = _calc_food_scores(recipes_df, normalized_nutrients, debug)

    # 12. Clean durations
    _parse_durations(recipes_df, debug)

    # 13. Count Directions and Ingredients
    _count_directions_and_ingredients(recipes_df, debug, parse_as_dict)

    # 14. Make titles unique
    _make_titles_unique(recipes_df, debug, make_titles_unique)

    # 15. Get all filtered foodKG ingredient links
    ingredient_names_dict, ingredients_df, ingredients_dict, mapped_recipes, recipe_ingredients_dict = _get_kg_links(
        recipes_df, raw_data['ingredients_df'], debug)

    # 16. Process food/graph labels
    ingredient_names_dict, ingredients_df, label_dict, recipe_ingredients_dict = _process_graph_labels(
        clean_food_category, food_class_strategy, ingredient_names_dict, ingredients_dict, recipe_ingredients_dict,
        ingredients_df, recipes_df, raw_data['label_ingredients_df'], debug)

    # 17. Process tags
    recipe_tag_dict, recipe_tag_urls, recipe_tags, tag_recipe_dict = _process_tags(recipes_df, debug)

    # 18. Add recipe ingredients from ingredients_df, recipe_ingredients_dict, ingredient_names_dict to recipes_df
    recipes_df = _add_foodKG_ingredients(ingredients_df, recipes_df, debug)

    # 19. Calculate similarities
    similar_recipes_dict, cosine_similarities = _get_similar_recipes(recipes_df, similarity_thresholds, feature_weights,
                                                                     debug)

    # 20. Annotate recipes with meal types
    recipes_df = _annotate_meal_types(recipes_df, llm, do_annotate_meal_type, '[20] ',
                                      create_random_meal_types, debug)

    # Define params to store and return
    hummus_data = {'recipes_df': recipes_df,
                   'reviews_df': reviews_df,
                   'users_df': users_df,
                   'recipes_dict': recipes_dict,
                   'user_dict': users_dict,
                   'old_food_locator_dict': raw_data['old_food_locator_dict'],
                   'old_food_com_dict': raw_data['old_food_com_dict'],
                   'recipe_tags': recipe_tags,
                   'recipe_tag_urls': recipe_tag_urls,
                   'recipe_tag_dict': recipe_tag_dict,
                   'tag_recipe_dict': tag_recipe_dict,
                   'food_on_ontology': raw_data['food_on_ontology'],
                   'ingredients_df': ingredients_df,
                   'ingredients_dict': ingredients_dict,
                   'mapped_recipes': mapped_recipes,
                   'recipe_ingredients_dict': recipe_ingredients_dict,
                   'ingredient_names_dict': ingredient_names_dict,
                   'label_dict': label_dict,
                   'food_on_mappings': raw_data['food_on_mappings'],
                   'usda_mappings': raw_data['usda_mappings'],
                   'usda_df': raw_data['usda_df'],
                   'usda_json': raw_data['usda_json'],
                   'similar_recipes_dict': similar_recipes_dict,
                   'cosine_similarities': cosine_similarities}

    return hummus_data


def _load_raw_hummus(data_location, graph_location, additional_location, recipe_column_names, review_column_names,
                     member_column_names, include_usda, s3, debug):
    if debug:
        print("[0] Load raw HUMMUS data ...")

    # Load the hummus data
    recipes_df, reviews_df, users_df = load_util.load_raw_hummus(data_location, recipe_column_names,
                                                                 review_column_names, member_column_names, s3=s3)

    # Get the additional food mappings. Note: Both are performed before reindexing methods, thus the old ids are used.
    # recipe_id --> food_kg_locator
    old_food_locator_dict = recipes_df.set_index('recipe_id').to_dict()['food_kg_locator']
    old_food_com_dict = recipes_df.set_index('recipe_id').to_dict()['recipe_url']  # recipe_id --> recipe_url

    # Load the additional data
    if additional_location is not None:
        additional_data_df = load_util.load_additional_data(additional_location, s3)
    else:
        additional_data_df = None

    # Load the graph data
    if graph_location is not None:
        (ingredients_df, labeled_ingredients_df, food_on_ontology, food_on_mappings, usda_df, usda_json,
         usda_mappings) = load_util.load_graph_data(graph_location, include_usda, s3=s3)
    else:
        (ingredients_df, labeled_ingredients_df, food_on_ontology, food_on_mappings, usda_df, usda_json,
         usda_mappings) = None, None, None, None, None, None, None

    # Put raw data into the dict
    raw_data = {'recipes_df': recipes_df,
                'reviews_df': reviews_df,
                'users_df': users_df,
                'additional_data_df': additional_data_df,
                'ingredients_df': ingredients_df,
                'label_ingredients_df': labeled_ingredients_df,
                'usda_df': usda_df,
                'usda_json': usda_json,
                'food_on_ontology': food_on_ontology,
                'old_food_locator_dict': old_food_locator_dict,
                'old_food_com_dict': old_food_com_dict,
                'food_on_mappings': food_on_mappings,
                'usda_mappings': usda_mappings}

    return raw_data


def _parse_comments(reviews, comment_relations, debug):
    if comment_relations == -1:  # remove the 0 ratings (==comments)
        if debug:
            print("[1] Remove comment relations ...")
        reviews = reviews[reviews.rating != 0]
    if comment_relations >= 0:  # set the new value for the 0 ratings
        if debug:
            print("[1] Reparse comment relations ...")
        reviews.loc[reviews.rating == 0, 'rating'] = comment_relations
    return reviews


def _parse_authorship(recipes, reviews, authorship_relations, debug):
    if authorship_relations >= 0:
        if debug:
            print("[2] Add authorship relations ...")
        # compute new potential ratings and rename columns such that they are similar to the reviews
        relations = recipes.loc[:, ['author_id', 'recipe_id', 'last_changed_date']]
        relations['rating'] = authorship_relations
        relations['text'] = 'NaN'
        relations = relations.rename(columns={'author_id': 'member_id', 'last_changed_date': 'last_modified_date'})

        # get a matrix of already existing item-user ratings, calc valid indices, remove invalid ones from relation
        review_set = set(zip(reviews.recipe_id, reviews.member_id))
        relation_set = set(map(tuple, relations[['recipe_id', 'member_id']].to_numpy().tolist()))
        intersection = relation_set.intersection(review_set)
        valid_indices = [not _row_is_in_tuple(row, intersection) for row in
                         relations[['recipe_id', 'member_id']].to_numpy().tolist()]
        relations = relations[valid_indices]

        # add relations to reviews
        reviews = pd.concat([reviews, relations], ignore_index=True)
    else:
        if debug:
            print("[2] Skip authorship relations ...")
    return reviews


def _add_external_tags(additional_location, recipes, reviews, recipe_tags, add_additional_meal_type_labels,
                       discard_non_tagged_recipes, s3, debug):
    if recipe_tags is True:
        if debug:
            status = "[3] Add recipe tags"

            if add_additional_meal_type_labels:
                status += " and add additional meal type labels"

            if discard_non_tagged_recipes:
                status += " and discard non-tagged recipes"

            print(status + " ...")

        # Load additional data
        if s3 is None:
            additional_data = pd.read_csv(additional_location + 'RAW_recipes.csv', sep=',').loc[:,
                              ['id', 'name', 'tags']]
        else:
            with s3.open(additional_location + 'RAW_recipes.csv') as file:
                additional_data = pd.read_csv(file, sep=',').loc[:, ['id', 'name', 'tags']]

        # Add tags for same recipes
        recipes['tags'] = recipes['recipe_id'].map(additional_data.set_index('id')['tags'])
        recipes['tags'] = [try_parse(row) if isinstance(row, str) else row for row in
                           recipes['tags'].to_numpy().tolist()]

        # Add additional meal type labels
        if add_additional_meal_type_labels:
            recipes['tags'] = recipes.apply(_append_additional_tags, axis=1)

        # Remove recipes without tags
        if discard_non_tagged_recipes:
            # Remove recipes without tags
            recipes = recipes.dropna(subset=['tags'])

            # Remove reviews for recipes without tags
            reviews = reviews[reviews['recipe_id'].isin(recipes['recipe_id'])]
    else:
        if debug:
            print("[3] Skip recipe tags ...")

    return recipes, reviews


def _append_additional_tags(row, additional_tags=None):
    if additional_tags is None:  # Default additional tags
        additional_tags = ['breakfast', 'lunch', 'dinner', 'snack']

    tags = row['tags'] if isinstance(row['tags'], list) else []
    title = row['title'] if isinstance(row['title'], str) else ''
    description = row['description'] if isinstance(row['description'], str) else ''
    directions = ' '.join(row['directions']) if isinstance(row['directions'], list) else ''

    for additional_tag in additional_tags:
        if additional_tag in title.lower() or additional_tag in description.lower() or additional_tag in directions.lower():
            tags.append(additional_tag)

    # Return list of unique tags
    return list(set(tags))


def _threshold_discarding_init(k_recipe, k_user, recipes, reviews, users, debug):
    if (k_user > 1 and k_recipe > 1) or (k_user >= 1 and k_recipe > 1) or (k_user > 1 and k_recipe >= 1):
        if debug:
            print("[4] Start Threshold Discarding ...")
        recipes, reviews, users = threshold_discarding(k_user, k_recipe, recipes, reviews, users, debug)
    else:
        if debug:
            print("[4] Skip Threshold Discarding ...")
    return recipes, reviews, users


def threshold_discarding(k_user, k_recipe, recipes, reviews, users, debug):
    """
    Executes the threshold discarding on recipe and review data frames.

    :param k_user: Number of minimal user interactions. Must be at least 2.
    :param k_recipe: Number of minimal recipe interactions. Must be at least 2.
    :param recipes: Dataframe of the recipes.
    :param reviews: Dataframe of the reviews.
    :param users: Dataframe of the users.
    :param debug: If debug messages shall be printed.
    :return:
        reduced_recipes: The recipe data frame with recipes with have at least k_recipe interactions,
        reduced_reviews: The review data frame with reviews with have at least k_user interactions,
    """
    # Count occurrence of each user and recipe in reviews
    counts_col_user = reviews.groupby("member_id")["member_id"].transform(len)
    counts_col_recipes = reviews.groupby("recipe_id")["recipe_id"].transform(len)

    # Bool masks
    mask_user = counts_col_user >= k_user
    mask_recipes = counts_col_recipes >= k_recipe

    # Count reviews with non-sparse users and non-sparse recipes
    _count(k_user, mask_user, 'user')
    _count(k_recipe, mask_recipes, 'recipe')

    # Apply masks for reviews
    if debug:
        print(f"   Number of reviews before: {reviews.shape[0]}")
    reduced_reviews = reviews[mask_user & mask_recipes]
    if debug:
        print(f"   Number of reviews after: {reduced_reviews.shape[0]}")

    # Apply masks for recipes
    if debug:
        print(f"   Number of recipes before: {recipes.shape[0]}")
    reduced_recipes = recipes[recipes.recipe_id.isin(reduced_reviews.recipe_id.unique())]
    if debug:
        print(f"   Number of recipes after: {reduced_recipes.shape[0]}")

    # Apply masks for users
    if debug:
        print(f"   Number of users before: {users.shape[0]}")
    reduced_users = users[users.member_id.isin(reduced_reviews.member_id.unique())]
    if debug:
        print(f"   Number of users after: {reduced_users.shape[0]}")

    # resets indices
    reduced_reviews = reduced_reviews.reset_index()
    reduced_reviews.drop(['index'], axis=1, inplace=True)
    reduced_recipes = reduced_recipes.reset_index()
    reduced_recipes.drop(['index'], axis=1, inplace=True)
    reduced_users = reduced_users.reset_index()
    reduced_users.drop(['index'], axis=1, inplace=True)

    # Assert that the shared recipe ids are the same ones (each recipe has a review after threshold discarding)
    assert set(reduced_recipes['recipe_id'].tolist()) == set(reduced_reviews['recipe_id'].tolist())

    return reduced_recipes, reduced_reviews, reduced_users


def _count(k, mask, name):
    counts = mask.value_counts()
    try:
        counts_true = counts[True]
    except KeyError:
        counts_true = 0
    try:
        counts_false = counts[False]
    except KeyError:
        counts_false = 0
    print("   Data with non-sparse " + name + " (k_" + name + ": " + str(k) + "):")
    print(f"       #True: {int(counts_true / len(mask) * 100)}%")
    print(f"       #False: {int(counts_false / len(mask) * 100)}%")


def _reindex(recipes, reviews, users, exclude_author_ids, keep_old_ids, debug):
    if debug:
        print("[5] Reset indices ...")

    # Get new member and recipe ids (continuous space starting by 0): old_id --> new_id
    recipes_dict = _get_new_recipe_ids(recipes)
    members_dict = _get_new_member_ids(recipes, reviews, exclude_author_ids)

    # lookup the new indices, add them as a new column and remove old indices
    recipes, reviews, users = _update_ids(recipes, reviews, users, recipes_dict, members_dict, keep_old_ids,
                                          new_column_str='')

    return recipes, reviews, users, recipes_dict, members_dict


def _get_new_recipe_ids(recipes):
    # resets index twice to end up with continuous indices in the dictionaries
    recipes_dict = recipes.reset_index().reset_index().set_index('recipe_id').to_dict()['level_0']

    # old_id --> new_id
    return recipes_dict


def _get_new_member_ids(recipes, reviews, exclude_author_ids):
    # resets index twice to end up with continuous indices in the dictionaries
    if exclude_author_ids:
        # Convert to Series
        all_member_ids = reviews['member_id'].squeeze()

        # remove -1 (invalid ids)
        removed_minus_one, all_member_ids = remove_minus_one(all_member_ids)

        # Reset indices
        single_member_ids = all_member_ids.drop_duplicates().reset_index().reset_index()
        member_dict = single_member_ids.set_index('member_id').to_dict()['level_0']
    else:  # also add author ids
        # Convert to Series
        all_member_ids = pd.concat([recipes['author_id'].squeeze(), reviews['member_id'].squeeze()])

        # remove -1 (invalid ids)
        removed_minus_one, all_member_ids = remove_minus_one(all_member_ids)

        # Reset indices
        single_member_ids = all_member_ids.drop_duplicates().reset_index().reset_index()
        member_dict = single_member_ids.set_index(0).to_dict()['level_0']

    # Add invalid id if it was removed previously
    if removed_minus_one:
        member_dict[-1] = -1

    # old_id --> new_id
    return member_dict


def remove_minus_one(data_series):
    if -1 in data_series:
        return True, data_series[data_series != -1]
    else:
        return False, data_series


def _update_ids(recipes, reviews, users, recipes_dict, member_dict, keep_old_ids, new_column_str='new_',
                old_column_str='_old', original_column_str=''):
    # lookup the new indices and add them as a new column
    reviews[new_column_str + 'member_id'] = [_lookup_index(row, member_dict) for row in
                                             reviews[[original_column_str + 'member_id']].to_numpy().tolist()]
    users[new_column_str + 'member_id'] = [_lookup_index(row, member_dict) for row in
                                           users[[original_column_str + 'member_id']].to_numpy().tolist()]
    reviews[new_column_str + 'recipe_id'] = [_lookup_index(row, recipes_dict) for row in
                                             reviews[[original_column_str + 'recipe_id']].to_numpy().tolist()]
    recipes[new_column_str + 'recipe_id'] = [_lookup_index(row, recipes_dict) for row in
                                             recipes[[original_column_str + 'recipe_id']].to_numpy().tolist()]
    # Also update author_ids, invalid ids will be mapped with -1
    recipes[new_column_str + 'author_id'] = [_lookup_index(row, member_dict) for row in
                                             recipes[[original_column_str + 'author_id']].to_numpy().tolist()]
    if keep_old_ids:  # keep old indices but rename them
        recipes = recipes.rename(columns={original_column_str + 'recipe_id': old_column_str + 'recipe_id',
                                          original_column_str + 'author_id': old_column_str + 'author_id',
                                          original_column_str + 'member_id': old_column_str + 'member_id'})
    elif original_column_str != new_column_str:  # drop old indices
        recipes, reviews, users = _remove_old_ids(recipes, reviews, users, original_column_str)
    # else when original and new column names are the same, just keep it/do nothing

    return recipes, reviews, users


def _remove_old_ids(recipes, reviews, users, original_column_str):
    recipes = recipes.drop(columns=[original_column_str + 'recipe_id', original_column_str + 'author_id'])
    reviews = reviews.drop(columns=[original_column_str + 'member_id', original_column_str + 'recipe_id'])
    users = users.drop(columns=[original_column_str + 'member_id'])
    return recipes, reviews, users


def _cluster_users(recipes, reviews, users, n_clusters, n_components, recipes_dict, members_dict, debug):
    if n_clusters > 0 and n_components > 0:
        if debug:
            print("[6] Cluster users ...")
        # For clustering need to remove all reviews which are invalid (member_id == -1) as otherwise the clusters
        # would be faulty
        reviews = reviews[reviews['member_id'] != -1]

        # Assert that the dicts are valid in recipes and reviews
        assert set(recipes['recipe_id'].tolist()) == set(recipes_dict.values())  # new recipe ids
        assert set(reviews['recipe_id'].tolist()) - {-1} == set(recipes_dict.values())  # new recipe ids

        # Recipe clustering
        recipes, reviews, users, user_cluster_dict, _, _ = cluster_util.full_clustering(recipes, reviews, users,
                                                                                        n_clusters=n_clusters,
                                                                                        random_state=42)

        # Update the member dict (old_id --> new_id) + (new_id --> cluster_id) = (old_id --> cluster_id)
        members_dict = merge_dicts(members_dict, user_cluster_dict)

        if debug:
            print("   Number of user clusters: " + str(len(set(user_cluster_dict.values()))))
    else:
        if debug:
            print("[6] Skip user clustering ...")

    return recipes, reviews, users, recipes_dict, members_dict


def _clean_nan_strings(recipes, reviews, users, clean_nan_strings, debug):
    if clean_nan_strings:
        if debug:
            print("[7] Clean nan strings ...")

        recipes['title'] = recipes['title'].fillna('')
        recipes['description'] = recipes['description'].fillna('')

        reviews['text'] = reviews['text'].fillna('')

        users['member_name'] = users['member_name'].fillna('')
        users['member_description'] = users['member_description'].fillna('')
        users['member_map_pin'] = users['member_map_pin'].fillna('')
    else:
        if debug:
            print("[7] Skip cleaning nan strings ...")

    return recipes, reviews, users


def _parse_recipe_rating_and_amount(recipes, debug):
    if debug:
        print("[8] Parse average recipe rating and amount ...")
    json_data = recipes['rating']
    parsed_json = [ast.literal_eval(json_string) for json_string in json_data]
    recipes['average_rating'] = [x[0] for x in parsed_json]
    recipes['number_of_ratings'] = [x[1] for x in parsed_json]

    # remove old rating column
    recipes.drop(['rating'], axis=1, inplace=True)


def _parse_nutrients(recipes, debug):
    if debug:
        print("[9] Parse nutrients ...")
    json_data = recipes.loc[:, ['nutrition']]['nutrition']
    parsed_json = [json.loads(json_string) for json_string in json_data]
    nutrition_data = pd.json_normalize(parsed_json)

    # remove units from data
    nutrition_normalized = pd.DataFrame()
    nutrition_normalized['servingSize [g]'] = nutrition_data['servingSize'].map(
        lambda x: x.rsplit('(')[1].rsplit(')')[0])
    nutrition_normalized['servingsPerRecipe'] = nutrition_data['servingPerRecipe']
    nutrition_normalized['calories [kcal]'] = nutrition_data['calories']
    nutrition_normalized['caloriesFromFat [kcal]'] = nutrition_data['caloriesFromFat'].map(lambda x: x.rsplit(' ')[0])
    nutrition_normalized['totalFat [g]'] = nutrition_data['totalFat'].map(lambda x: x.rsplit(' ')[0])
    nutrition_normalized['saturatedFat [g]'] = nutrition_data['saturatedFat'].map(lambda x: x.rsplit(' ')[0])
    nutrition_normalized['cholesterol [mg]'] = nutrition_data['cholesterol'].map(lambda x: x.rsplit(' ')[0])
    nutrition_normalized['sodium [mg]'] = nutrition_data['sodium'].map(lambda x: x.rsplit(' ')[0])
    nutrition_normalized['totalCarbohydrate [g]'] = nutrition_data['totalCarbohydrate'].map(lambda x: x.rsplit(' ')[0])
    nutrition_normalized['dietaryFiber [g]'] = nutrition_data['dietaryFiber'].map(lambda x: x.rsplit(' ')[0])
    nutrition_normalized['sugars [g]'] = nutrition_data['sugars'].map(lambda x: x.rsplit(' ')[0])
    nutrition_normalized['protein [g]'] = nutrition_data['protein'].map(lambda x: x.rsplit(' ')[0])

    # add nutrition data to the recipes and convert to float/int
    recipes[['servingsPerRecipe']] = nutrition_normalized[['servingsPerRecipe']].astype(int)
    recipes[['servingSize [g]', 'calories [kcal]', 'caloriesFromFat [kcal]', 'totalFat [g]', 'saturatedFat [g]',
             'cholesterol [mg]', 'sodium [mg]', 'totalCarbohydrate [g]', 'dietaryFiber [g]', 'sugars [g]',
             'protein [g]']] = \
        nutrition_normalized[
            ['servingSize [g]', 'calories [kcal]', 'caloriesFromFat [kcal]', 'totalFat [g]', 'saturatedFat [g]',
             'cholesterol [mg]', 'sodium [mg]', 'totalCarbohydrate [g]', 'dietaryFiber [g]', 'sugars [g]',
             'protein [g]']].astype(float)

    # remove old nutrition column
    recipes.drop(['nutrition'], axis=1, inplace=True)


def _init_normalized_nutrients(recipes, debug):
    if debug:
        print('[10] Normalize nutrients ...')

    return normalize_nutrients(recipes)


def normalize_nutrients(recipes):
    """
    Normalizes all nutrients of the recipe dataframe by changing all mg units into g, and normalizing all values based
     on a recipe weight of 100g.
    :param recipes: The recipe dataframe
    :return: The normalized ingredients with a new comment row describing the type of normalization (refer to
        '_check_and_normalize_weights()')
    """

    # Init the parallel lib
    pandarallel.initialize(verbose=0)

    # Copy all ingredient values in a new data frame and change all mg-based nutrients into g-based ones. Keep cal
    # values as they are.
    normalized_ingredients = pd.DataFrame()
    normalized_ingredients[
        ['servingsPerRecipe', 'servingSize [g]', 'calories [kcal]', 'caloriesFromFat [kcal]', 'totalFat [g]',
         'saturatedFat [g]']] = recipes[
        ['servingsPerRecipe', 'servingSize [g]', 'calories [kcal]', 'caloriesFromFat [kcal]', 'totalFat [g]',
         'saturatedFat [g]']]
    normalized_ingredients['cholesterol [g]'] = recipes['cholesterol [mg]'].map(lambda x: x / 100)
    normalized_ingredients['sodium [g]'] = recipes['sodium [mg]'].map(lambda x: x / 100)
    normalized_ingredients[['totalCarbohydrate [g]', 'dietaryFiber [g]', 'sugars [g]', 'protein [g]']] = recipes[
        ['totalCarbohydrate [g]', 'dietaryFiber [g]', 'sugars [g]', 'protein [g]']]

    # Computation expensive (~15min on my laptop), might change to vectorized implementation...
    # Normalize all ingredients to 100g.
    normalized_ingredients = normalized_ingredients. \
        parallel_apply(lambda row: _check_and_normalize_weights(row), axis=1)

    return normalized_ingredients


def _check_and_normalize_weights(nutrient_row):
    """
    Normalize the weights via the following heuristic.

    A) Make sure that all ingredients weight less than the serving size (because this is, according to food.com, the
    scale) --> normalize to 100g
    B) The serving size is faulty. Test if the ingredients summed weight is less than 'serving size' *
    'servings per recipe'. --> normalize based on that value & mark the value with one star '*'.
    C) The serving size does not correspond to both alternatives. --> skip normalization & mark the value with two
    stars '**'.

    :param nutrient_row: A single row of the nutrient dataframe.
    :return: The normalized rows including the star-markings.
    """
    # Note: Saturated fat is part of total fat, so we do not need to take that into consideration. Also, dietary fiber
    # and sugars are both parts of total carbs, so we can also discard those.
    sum_of_nutrient_weights = sum(
        nutrient_row[['totalFat [g]', 'cholesterol [g]', 'sodium [g]', 'totalCarbohydrate [g]', 'protein [g]']])

    serving_size = nutrient_row[['servingSize [g]']][0]
    servings_per_recipe = nutrient_row[['servingsPerRecipe']][0]

    # Get factor, and if necessary, the marking.
    if serving_size >= sum_of_nutrient_weights:  # A
        normalization_function = lambda x: x * 100 / serving_size
        marking = ''
    elif serving_size * servings_per_recipe >= sum_of_nutrient_weights:  # B
        normalization_function = lambda x: (x / servings_per_recipe) * 100 / (serving_size / servings_per_recipe)
        marking = '*'
    else:  # C
        normalization_function = lambda x: x
        marking = '**'

    # Normalize values based on the factor
    nutrient_row[['totalFat [g]', 'saturatedFat [g]', 'cholesterol [g]', 'sodium [g]', 'totalCarbohydrate [g]',
                  'dietaryFiber [g]', 'sugars [g]', 'protein [g]']] \
        = nutrient_row[['totalFat [g]', 'saturatedFat [g]', 'cholesterol [g]', 'sodium [g]', 'totalCarbohydrate [g]',
                        'dietaryFiber [g]', 'sugars [g]', 'protein [g]']].map(normalization_function)

    # Add marking
    nutrient_row['normalization_comment'] = marking

    return nutrient_row


def _calc_food_scores(recipes_df, normalized_nutrients, debug):
    if debug:
        print('[11] Calculate Food scores ...')
    recipes_df = score_util.calculate_food_scores(recipes_df, normalized_nutrients,
                                                  score_names=['who', 'fsa', 'nutri'], normalize=True)
    return recipes_df


def _parse_durations(recipes, debug):
    if debug:
        print("[12] Clean durations ...")
    date_time = pd.to_datetime(recipes['duration'].replace({'min': ' min'}, regex=True), format='%M min',
                               errors="coerce").fillna(
        pd.to_datetime(recipes['duration'].replace({'mins': ' min'}, regex=True), format='%M min',
                       errors="coerce")).fillna(
        pd.to_datetime(recipes['duration'].replace({'hr': ' hour'}, regex=True), format='%H hour',
                       errors="coerce")).fillna(
        pd.to_datetime(recipes['duration'].replace({'hrs': ' hour'}, regex=True), format='%H hour',
                       errors="coerce")).fillna(
        pd.to_datetime(recipes['duration'].replace({'hr': ' hour', 'min': ' min'}, regex=True), format='%H hour %M min',
                       errors="coerce")).fillna(
        pd.to_datetime(recipes['duration'].replace({'hr': ' hour', 'mins': ' min'}, regex=True),
                       format='%H hour %M min',
                       errors="coerce")).fillna(
        pd.to_datetime(recipes['duration'].replace({'hrs': ' hour', 'min': ' min'}, regex=True),
                       format='%H hour %M min',
                       errors="coerce")).fillna(
        pd.to_datetime(recipes['duration'].replace({'hrs': ' hour', 'mins': ' min'}, regex=True),
                       format='%H hour %M min',
                       errors="coerce"))
    time = date_time.dt.hour * 60 + date_time.dt.minute
    recipes['duration'] = time


def _count_directions_and_ingredients(recipes, debug, parse_as_dict):
    if debug:
        print("[13] Parse & Count directions and ingredients ...")

    # parse directions
    list_data_directions = recipes.loc[:, ['directions']]['directions'].map(
        lambda x: x.replace('{', '[').replace('}', ']').replace('\r', '').replace('\t', ''))
    list_data_directions = [_parse_string_list(direction) for direction in list_data_directions]
    direction_sizes = [_count_parsed_list(direction) for direction in list_data_directions]

    # parse ingredients
    list_data_ingredients = recipes.loc[:, ['ingredients']]['ingredients'].map(
        lambda x: x.replace('{', '[').replace('}', ']').replace('\r', '').replace('\t', ''))
    list_data_ingredients = [_parse_layered_ingredients(ingredient, parse_as_dict) for ingredient in
                             list_data_ingredients]
    ingredients_sizes = [_count_ingredient_list(ingredient, parse_as_dict) for ingredient in list_data_ingredients]

    # add both sizes and parsed lists to recipes
    recipes['directions'] = list_data_directions
    recipes['direction_size'] = direction_sizes
    recipes['ingredients'] = list_data_ingredients
    recipes['ingredients_sizes'] = ingredients_sizes


def _make_titles_unique(recipes, debug, make_titles_unique):
    if make_titles_unique:
        if debug:
            print("[14] Make titles unique ...")

        # Get the counts of each title
        title_counts = recipes['title'].value_counts()

        #  Modify the titles to be unique
        recipes['title'] = recipes['title'].apply(_add_suffix, title_counts=title_counts)
    else:
        if debug:
            print("[14] Skip making titles unique ...")


def _add_suffix(title, title_counts):  # Function to add suffix to non-unique titles

    count = title_counts[title]

    if count > 1:
        title_counts[title] = title_counts[title] - 1
        return f"{title} ({count})"
    else:
        return title


def _get_kg_links(recipes_df, ingredients_df, debug):
    if debug:
        print('[15] Get ingredient KG links ...')

    ingredients_df, ingredients_dict, mapped_recipes = get_ingredient_dict(recipes_df, ingredients_df)
    recipe_ingredients_dict = get_recipe_ingredients_dict(recipes_df, ingredients_df)
    ingredient_names_dict = get_ingredient_names_dict(ingredients_df)

    return ingredient_names_dict, ingredients_df, ingredients_dict, mapped_recipes, recipe_ingredients_dict


def get_recipe_ingredients_dict(recipes_df, ingredients_df):
    """
    Returns a dictionary with the recipe urls as keys and the ingredients as values.

    :param recipes_df: The recipe dataframe.
    :param ingredients_df: The ingredients dataframe.
    :return: The dictionary with the recipe urls as keys and the ingredients as values.
    """
    # get the recipe ids
    recipe_ids = recipes_df['recipe_id'].to_numpy().tolist()

    # get the recipe urls
    recipe_url_dict = recipes_df.set_index('recipe_id')['recipe_url'].to_dict()

    # get the ingredients
    recipe_ingredient_dict = ingredients_df.groupby('recipe_id')['ingredient_id'].agg(list).to_dict()

    # build the dictionary
    recipe_ingredients_dict = {recipe_url_dict[recipe_id]: recipe_ingredient_dict.get(recipe_id, []) for recipe_id in
                               recipe_ids}

    return recipe_ingredients_dict


def get_ingredient_names_dict(ingredients_df):
    """
    Returns a dictionary with the ingredient ids as keys and the ingredient names as values.

    :param ingredients_df: The ingredients dataframe.
    :return: The dictionary with the ingredient ids as keys and the ingredient names as values.
    """
    # get the ingredient names
    ingredient_name_dict = ingredients_df.set_index('ingredient_id')['ing_name'].to_dict()

    return ingredient_name_dict


def get_ingredient_dict(recipes, ingredients_df):
    """
    Loads ingredient specific data.

    :param recipes: Those recipe define which ingredients shall be fetched- only connected ingredients will be included.
    :param ingredients_df: The ingredients dataframe.
    :return:
        reduced_ingredients: A dataframe with following features: ing_name, recipe_id, ingredient_id
        ingredients_dict: A dictionary for the ingredient ids (ingredient_id -> foodKG_mapped_ingredient)
        mapped_recipes: A list of all mapped recipes (might be shorter than the size of param recipes as not mappings
            might be incomplete)
    """
    # build food_kg_locator_dict such as food_kg_locator --> new_recipe_id
    # Note: As the recipes are the crawled ones, there might be the case that they don't exist (because they were not
    # present on food.com anymore)
    food_kg_locator_dict = recipes.reset_index().set_index('food_kg_locator').to_dict()['recipe_id']

    # use recipes_dict (food_kg_locator --> new_recipe_id)
    ingredients_df['recipe_id'] = [_lookup_index(row, food_kg_locator_dict) for row in
                                   ingredients_df[['recipe']].to_numpy().tolist()]

    # Get the un- & mapped recipes
    unmapped_recipes = ingredients_df.loc[ingredients_df['recipe_id'] == -1]['recipe'].unique()
    mapped_recipes = ingredients_df.loc[ingredients_df['recipe_id'] != -1]['recipe'].unique()

    # remove the unmapped recipes
    reduced_ingredients = ingredients_df[~ingredients_df['recipe'].isin(unmapped_recipes)]

    # get all distinct foodKG_mapped_ingredient and assign ids (ingredient_id -> foodKG_mapped_ingredient)
    ingredients_dict = {index[0]: url for index, url in
                        np.ndenumerate(reduced_ingredients['foodKG_mapped_ingredient'].unique())}
    inv_ingredients_dict = {url: index[0] for index, url in
                            np.ndenumerate(reduced_ingredients['foodKG_mapped_ingredient'].unique())}

    # use inv_ingredients_dict (foodKG_mapped_ingredient --> ingredient_id)
    reduced_ingredients['ingredient_id'] = [_lookup_index(row, inv_ingredients_dict) for row in
                                            reduced_ingredients[['foodKG_mapped_ingredient']].to_numpy().tolist()]

    # remove unnecessary columns
    reduced_ingredients.drop(['foodKG_mapped_ingredient'], axis=1, inplace=True)
    reduced_ingredients.drop(['recipe'], axis=1, inplace=True)

    return reduced_ingredients, ingredients_dict, list(mapped_recipes)


def _process_graph_labels(clean_food_category, food_class_strategy, ingredient_names_dict, ingredients_dict,
                          recipe_ingredients_dict, ingredients_df, recipes_df, label_ingredients_df, debug):
    if debug:
        print('[16] Process the graph-based label dict ...')

    label_dict = get_ingredient_tags(label_ingredients_df)

    # Integrate food classes from the FoodOn Ontology
    inv_ing_dict = invert_dict(ingredients_dict)
    food_category_dict = {}

    # Create new `pandas` methods which use `tqdm` progress
    tqdm.pandas()
    if food_class_strategy == 'None':
        if debug:
            print('   No food classes will be integrated. Instead, they will be returned as a dictionary.')
    elif food_class_strategy == 'Ingredients':
        if debug:
            print('   Food classes will be integrated as ingredients.')
        ingredients_df, recipe_ingredients_dict, ingredient_names_dict = _extend_ingredients(
            recipes_df, ingredients_df, inv_ing_dict, label_dict, clean_food_category, food_category_dict)
    elif food_class_strategy == 'Tags':
        if debug:
            print('   Food classes will be integrated as tags.')
        recipes_df['tags'] = recipes_df.progress_apply(_extend_tags, axis=1,
                                                       args=(label_dict, recipe_ingredients_dict, ingredients_dict,
                                                             clean_food_category, food_category_dict))
    elif food_class_strategy == 'Both':
        if debug:
            print('   Food classes will be integrated as ingredients and tags.')
        ingredients_df, recipe_ingredients_dict, ingredient_names_dict = _extend_ingredients(
            recipes_df, ingredients_df, inv_ing_dict, label_dict, clean_food_category, food_category_dict)
        recipes_df['tags'] = recipes_df.progress_apply(_extend_tags, axis=1,
                                                       args=(label_dict, recipe_ingredients_dict, ingredients_dict,
                                                             clean_food_category, food_category_dict))
    else:
        if debug:
            print('   Food class strategy not recognized. No food classes will be integrated.')

    return ingredient_names_dict, ingredients_df, label_dict, recipe_ingredients_dict


def get_ingredient_tags(label_ingredients_df):
    """
    Processes the ingredient tag dict.

    :param label_ingredients_df: The ingredient labels dataframe.

    :return: The ingredient tags (ingredient_url --> tag_list)
    """
    # create empty dict
    label_dict = {}

    # convert to dict (ingredient)
    for _, row in label_ingredients_df.iterrows():
        _add_labels(row, label_dict)

    return label_dict


def _add_labels(row, label_dict):
    ingredient_id = row[1]
    label = row[2]

    if ingredient_id not in label_dict:
        label_dict[ingredient_id] = [label]
    else:
        label_dict[ingredient_id].append(label)


def _extend_ingredients(recipes_df, ingredients_df, inv_ing_dict, label_dict, clean_food_category, food_category_dict):
    last_valid_index = ingredients_df['ingredient_id'].max()
    ingredient_recipe_map = ingredients_df.groupby('ingredient_id')['recipe_id'].apply(list).to_dict()

    # Extend ingredients with food classes
    for original_ingredient_url, ingredient_labels in tqdm(label_dict.items(), desc='Extending ingredients'):
        original_ingredient_id = inv_ing_dict[original_ingredient_url]

        # Get all recipes that contain the ingredient
        recipie_ids = ingredient_recipe_map[original_ingredient_id]

        for recipe_id in recipie_ids:
            for ingredient_label in ingredient_labels:
                if clean_food_category:
                    ingredient_label = _clean_food_category(ingredient_label, food_category_dict)

                if ingredient_label is not None:
                    # Check if the ingredient label is already present and get the ingredient id
                    if ingredient_label in ingredients_df['ing_name'].values:  # Ingr. label is same as original ingr.
                        # Check if the ingredient label is not already present for the recipe
                        if not ingredients_df.loc[(ingredients_df['ing_name'] == ingredient_label) &
                                                  (ingredients_df['recipe_id'] == recipe_id)].empty:
                            # Get the ingredient id
                            ingredient_id = ingredients_df.loc[
                                ingredients_df['ing_name'] == ingredient_label]['ingredient_id'].iloc[0]

                            # Update recipes
                            ingredients_df = ingredients_df.append(
                                {'ing_name': ingredient_label, 'recipe_id': recipe_id, 'ingredient_id': ingredient_id},
                                ignore_index=True)
                    else:  # Ingredient label is not present and has not been added before
                        # Assign a new ingredient id
                        ingredient_id = last_valid_index + 1
                        last_valid_index = ingredient_id

                        # Update recipes
                        ingredients_df = ingredients_df.append(
                            {'ing_name': ingredient_label, 'recipe_id': recipe_id, 'ingredient_id': ingredient_id},
                            ignore_index=True)

    # Update dicts
    recipe_ingredients_dict = get_recipe_ingredients_dict(recipes_df, ingredients_df)
    ingredient_names_dict = get_ingredient_names_dict(ingredients_df)

    return ingredients_df, recipe_ingredients_dict, ingredient_names_dict


def _extend_tags(row, label_dict, recipe_ingredients_dict, ingredients_dict, clean_food_category, food_category_dict):
    recipe_ingredients = recipe_ingredients_dict.get(row['recipe_url'], [])
    combined_tags = row['tags']
    additional_tags = []

    for ingredient_id in recipe_ingredients:
        ingredient_url = ingredients_dict[ingredient_id]
        additional_tags = label_dict.get(ingredient_url, [])
        combined_tags += additional_tags

    # Clean additional tags and remove duplicates
    if clean_food_category:
        additional_tags = set(filter(lambda x: x is not None,
                                     map(lambda tag: _clean_food_category(tag, food_category_dict), additional_tags)))

    return additional_tags


def _clean_food_category(food_category, food_category_dict):
    # If category includes 'food product by' remove it completely
    if 'food product by' in food_category or 'food product component' in food_category:
        return None

    # Removes 'food product' from the food category. If we have something like 'seafood product' only 'product' will be
    # removed. If we have 'food product' it will be removed completely.
    result = re.sub(r'(\b(?<!\bfood\s)product\b)|(\bfood(?!\b\s*sea)\b.*product)', '', food_category).strip()

    if result.isspace() or len(result) == 0:
        # If the category is empty or only has space chars after removing 'food product' remove it completely
        return None
    else:
        food_category_dict[food_category] = result
        return result


def _process_tags(recipes_df, debug):
    if debug:
        print('[17] Process tags ...')

    recipes_df['tags'] = recipes_df['tags'].apply(replace_nan_with_empty_list)
    unique_tags_temp = recipes_df[['tags']].explode('tags')['tags'].dropna().unique()
    recipe_tags = unique_tags_temp.tolist()
    recipe_tag_urls = np.vectorize(tag_util.get_url_from_tag)(unique_tags_temp).tolist()

    tag_recipe_dict = tag_util.get_tag_recipe_dict(recipes_df)
    recipe_tag_dict = tag_util.get_recipe_tag_dict(recipes_df)

    return recipe_tag_dict, recipe_tag_urls, recipe_tags, tag_recipe_dict


def _add_foodKG_ingredients(ingredients_df, recipes_df, debug):
    if debug:
        print('[18] Add foodKG recipe ingredients to recipes_df ...')

    # Perform the join and aggregation
    grouped_df = \
        pd.merge(recipes_df, ingredients_df, left_on='recipe_id', right_on='recipe_id').groupby('recipe_id')[
            'ing_name'].apply(list).reset_index()

    # Merge the aggregated strings with the original recipes DataFrame
    recipes_with_ingredients_df = pd.merge(recipes_df, grouped_df, on='recipe_id', how='left')

    # Set default value to recipes without ingredients in the foodKG
    recipes_with_ingredients_df['ing_name'] = recipes_with_ingredients_df['ing_name'].fillna('[]')

    # Rename the column to 'recipe_strings'
    recipes_with_ingredients_df.rename(columns={'ing_name': 'ingredient_names'}, inplace=True)

    return recipes_with_ingredients_df


def _get_similar_recipes(recipes_df, similarity_thresholds, feature_weights, debug):
    if debug:
        print('[19] Load and process similarity thresholds ...')

    if similarity_thresholds is not False:
        cosine_similarities = recipe_similarity_util.calc_tfidf_similarities(recipes_df,
                                                                             feature_weights=feature_weights)
        similar_recipes_dict = recipe_similarity_util.get_similar_recipes(recipes_df, cosine_similarities,
                                                                          similarity_thresholds)
    else:
        if debug:
            print('   Similar recipes will not be calculated.')
        similar_recipes_dict = {}
        cosine_similarities = {}

    return similar_recipes_dict, cosine_similarities


def annotate_meal_types(recipes_df, llm, annotate_meal_type, debug):
    return _annotate_meal_types(recipes_df, llm, annotate_meal_type, '', debug)


def _annotate_meal_types(recipes_df, llm, annotate_meal_type, print_str_prefix='', create_random=False, debug=True):
    """
    Annotates the meal type of the recipes with the LLM model.
    :param recipes_df: The recipes dataframe.
    :param llm: The LLM model.
    :param annotate_meal_type: Whether to annotate the meal type.
    :param print_str_prefix: The print string prefix.
    :param create_random: Whether to create random meal types for testing purposes.
    :param debug: Whether to print debug information.
    :return: The annotated recipes dataframe.
    """
    if not annotate_meal_type or (llm is None and not create_random):
        if debug:
            print(print_str_prefix + 'Skip annotating meal types ...')
        return recipes_df
    elif create_random:
        if debug:
            print(print_str_prefix + 'Create random meal types (for testing purposes)...')

        # Create new feature for meal types with empty lists if not present
        if 'meal_types' not in recipes_df.columns:
            recipes_df['meal_types'] = recipes_df.apply(lambda x: [], axis=1)

        # Create and set random meal types from the list MEAL_TYPES
        meal_types = MealType.get_all_str()
        random_meal_types = [random.sample(meal_types, random.randint(0, len(meal_types)))
                             for _ in range(recipes_df.shape[0])]
        recipes_df['meal_types'] = random_meal_types
        return recipes_df
    else:
        if debug:
            print(print_str_prefix + 'Annotate meal types ...')
            print('   Important Note: As meal types are annotated by using a llm, this process might take a while and'
                  ' also the annotations are not deterministic and will be change from run to run!')

        # Create new feature for meal types with empty lists if not present
        if 'meal_types' not in recipes_df.columns:
            recipes_df['meal_types'] = recipes_df.apply(lambda x: [], axis=1)

        for index, row in tqdm(recipes_df.iterrows(), total=recipes_df.shape[0], desc='Annotating meal types',
                               disable=not debug):
            if debug:
                # print every 100th recipe a status update
                if index % 100 == 0:
                    print(f'   Annotating meal types for recipe #{index} of {recipes_df.shape[0]}...')

            row_dict = row.to_dict()
            row_str = str(row_dict)

            # Annotate the meal type with the llm
            meal_plan_prompt = MEAL_TYPE_ANNOTATE_TEMPLATE.format(recipe=row_str)
            response = llm.complete(meal_plan_prompt)

            # Try to extract the json from the response (output between  ```json and ```).
            try:
                response_json_str = response.text.split('```json')[1].split('```')[0].strip()
            except Exception:
                response_json_str = response.text

            #  Try to parse the meal type JSON
            try:
                response_json = try_parse(response_json_str)

                if isinstance(response_json, list):
                    meal_type_list = response_json
                elif isinstance(response_json, str):
                    meal_type_list = [response_json]
                else:
                    meal_type_list = []
            except SyntaxError:
                meal_type_list = []

            # Add the meal type to the recipes dataframe
            recipes_df.at[index, 'meal_types'] = meal_type_list

        return recipes_df


def clean_serves_to_int(recipes_df, serves_column_name='serves', default_serves=2):
    """
    Cleans the serves column to integers. If the serves column cannot be parsed as a digit, the default serves will be
    used.
    :param recipes_df: The recipes dataframe.
    :param serves_column_name: The column name of the serves column. Default: 'serves'.
    :param default_serves: The default/fallback serves value. Default: 2.
    :return:
    """

    def try_int_cast(x):
        if isinstance(x, str):
            if '-' in x:
                x = x.split('-')[-1]

            if x.isdigit():
                x_digit = int(x)
                if x_digit > 0:
                    return x_digit
                else:
                    return default_serves
            else:
                return default_serves

    recipes_df[serves_column_name] = recipes_df[serves_column_name].apply(lambda x: try_int_cast(x))

    # Set NaN values to default serves
    recipes_df[serves_column_name] = recipes_df[serves_column_name].fillna(default_serves)

    return recipes_df

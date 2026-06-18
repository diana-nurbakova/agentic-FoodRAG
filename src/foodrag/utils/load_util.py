import ast
import gzip
import json
import logging
import numpy as np
import os
import pandas as pd
import pickle
import rdflib
import ruamel.yaml as ryaml
import yaml
from llama_index.core.llms import LLM
from pathlib import Path

from foodrag.utils import hummus_util

SUB_DIRECTORY_NAME = 'hummus'


def serialize(string):
    try:
        return ast.literal_eval(string)
    except Exception:
        return []


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def get_config(config_path="config.yml"):
    with open(config_path, "r") as setting:
        config = yaml.load(setting, yaml.FullLoader)
    return config


def parse_zip(path):
    g = gzip.open(path, 'rb')
    for l in g:
        yield json.loads(l)


def get_df(path):
    i = 0
    df = {}
    for d in parse_zip(path):
        df[i] = d
        i += 1
    return pd.DataFrame.from_dict(df, orient='index')


def replace_nan_with_empty_list(value):
    return [] if isinstance(value, float) else value


def get_hummus(data_location, graph_location, additional_location, data_path='', discard_non_tagged_recipes=False,
               k_user=1, k_recipe=1, n_clusters=-1, n_components=2, food_class_strategy='None',
               add_additional_meal_type_labels=True, clean_food_category=True, feature_weights=None,
               similarity_thresholds=False, include_usda=True, do_annotate_meal_type=False,
               create_random_meal_types=False, llm: LLM = None, s3=None, debug=True):
    """
    Loads the hummus data set. If the data has not been initialized yet, it will be initialized with the given
    parameters. If the data has already been initialized, it will be loaded from the given data_path. If the data has
    already been initialized with different parameters, it will be re-initialized with the given parameters.
    :param data_location: The location of the hummus data set.
    :param graph_location: The location of the hummus graph.
    :param additional_location: The location of the additional data (tags).
    :param data_path: The location where the data should be stored.
    :param discard_non_tagged_recipes: Whether recipes without tags should be discarded.
    :param k_user: The parameter k for the user clustering.
    :param k_recipe: The parameter k for the recipe clustering.
    :param n_clusters: The number of clusters to use for the user clustering. If <= 1 no clustering will be performed.
    :param n_components: The number of components to use for the recipe clustering. If <= no clustering will be
        performed.
    :param food_class_strategy: If 'None' the food classes will not be integrated directly in tha dataset (but still
     returned). If 'Ingredients' the food classes will be integrated as ingredients. If 'Tags' the food classes will
        be integrated as tags. If 'Both' the food classes will be integrated as ingredients and tags.
    :param add_additional_meal_type_labels: If True, additional meal type labels will be added.
    :param clean_food_category: If True, the food categories will be cleaned.
    :param feature_weights: The weights dict for the features of the recipes for similarity calculation. If None default
     weights are chosen.
    :param similarity_thresholds: The thresholds for the similarities of recipes. If None default thresholds are chosen.
     If False no similarity is calculated.
    :param include_usda: If True, the usda data will be included. Note: Requires a lot of RAM.
    :param llm: The LLM instance for the meal type annotation. If None, no meal type annotation will be performed.
    :param do_annotate_meal_type: If True, the meal types will be annotated.
    :param create_random_meal_types: If True, random meal types will be created when do_annotate_meal_type is True.
    :param s3: The s3 connection. When the connection is not None, tries to use this connection for data loading instead
        local data loading.
    :param debug: If True, debug messages will be printed.
    :return: A dictionary containing the data.
    """
    # check if the data has already been initialized by loading the loading_state file
    ensure_dir(Path(data_path, SUB_DIRECTORY_NAME))
    file_path = Path(data_path, SUB_DIRECTORY_NAME, 'loading_state.json')
    is_initialized = os.path.exists(file_path) and os.path.isfile(file_path)

    if is_initialized:
        # Check if the current parameters match those of the present data
        loading_state = json.load(open(file_path))

        try:
            do_params_match = (loading_state['n_clusters'] == n_clusters and
                               loading_state['n_components'] == n_components and
                               loading_state['k_user'] == k_user and
                               loading_state['k_recipe'] == k_recipe and
                               loading_state['food_class_strategy'] == food_class_strategy and
                               loading_state['add_additional_meal_type_labels'] == add_additional_meal_type_labels and
                               loading_state['clean_food_category'] == clean_food_category and
                               loading_state['feature_weights'] == feature_weights and
                               loading_state['similarity_thresholds'] == similarity_thresholds and
                               loading_state['include_usda'] == include_usda and
                               loading_state['do_annotate_meal_type'] == do_annotate_meal_type)
        except KeyError:
            do_params_match = False

        if do_params_match:
            print('The data has already been initialized. Loading ...')
            data = _load_hummus(data_path)
        else:
            print('The data has been initialized with different parameters. Re-initializing ...')
            data = _init_hummus(data_location, graph_location, additional_location, data_path,
                                discard_non_tagged_recipes, k_user, k_recipe, n_clusters, n_components,
                                food_class_strategy, add_additional_meal_type_labels, clean_food_category,
                                feature_weights, similarity_thresholds, include_usda, do_annotate_meal_type,
                                create_random_meal_types, llm, s3, debug)
    else:
        # If data was not present or of other configuration initialize it
        print('The data has not been initialized. Initializing ...')
        data = _init_hummus(data_location, graph_location, additional_location, data_path, discard_non_tagged_recipes,
                            k_user, k_recipe, n_clusters, n_components, food_class_strategy,
                            add_additional_meal_type_labels, clean_food_category, feature_weights,
                            similarity_thresholds, include_usda, do_annotate_meal_type, create_random_meal_types, llm,
                            s3, debug)

    # Store the loading state file
    loading_state = {'n_clusters': n_clusters, 'n_components': n_components, 'k_user': k_user, 'k_recipe': k_recipe,
                     'food_class_strategy': food_class_strategy,
                     'add_additional_meal_type_labels': add_additional_meal_type_labels,
                     'clean_food_category': clean_food_category, 'feature_weights': feature_weights,
                     'similarity_thresholds': similarity_thresholds, 'include_usda': include_usda,
                     'do_annotate_meal_type': do_annotate_meal_type}
    json.dump(loading_state, open(file_path, 'w'))

    return data


def get_current_hummus(data_path=''):
    # check if the data has already been initialized by loading the loading_state file
    ensure_dir(Path(data_path, SUB_DIRECTORY_NAME))
    file_path = Path(data_path, SUB_DIRECTORY_NAME, 'loading_state.json')
    is_initialized = os.path.exists(file_path) and os.path.isfile(file_path)

    if is_initialized:
        return _load_hummus(data_path)
    else:
        raise Exception('The data has not been initialized. Please initialize it first. Path: ' + str(file_path))


def _init_hummus(data_location, graph_location, additional_location, data_path, discard_non_tagged_recipes, k_user,
                 k_recipe, n_clusters, n_components, food_class_strategy, add_additional_meal_type_labels,
                 clean_food_category, feature_weights, similarity_thresholds, include_usda, do_annotate_meal_type,
                 create_random_meal_types, llm, s3, debug):
    # Import & clean data
    hummus_data = hummus_util.load_and_clean_data(data_location, graph_location, additional_location,
                                                  add_additional_meal_type_labels=add_additional_meal_type_labels,
                                                  discard_non_tagged_recipes=discard_non_tagged_recipes, k_user=k_user,
                                                  k_recipe=k_recipe, n_clusters=n_clusters, n_components=n_components,
                                                  authorship_relations=6, recipe_tags=True, make_titles_unique=True,
                                                  food_class_strategy=food_class_strategy,
                                                  clean_food_category=clean_food_category,
                                                  similarity_thresholds=similarity_thresholds,
                                                  feature_weights=feature_weights, include_usda=include_usda,
                                                  do_annotate_meal_type=do_annotate_meal_type,
                                                  create_random_meal_types=create_random_meal_types, llm=llm, s3=s3,
                                                  debug=debug)

    # Store preprocessed data
    if debug:
        print('[20] Store data ...')
    store_data(hummus_data, Path(data_path, SUB_DIRECTORY_NAME))

    return hummus_data


def _load_hummus(data_path):
    data_path = Path(data_path, SUB_DIRECTORY_NAME)
    data = load_data(data_path)

    # Normalize tags, directions, ingredient_names, ingredients, and meal types
    data['recipes_df']['tags'] = data['recipes_df']['tags'].apply(lambda x: serialize(x))
    data['recipes_df']['directions'] = data['recipes_df']['directions'].apply(lambda x: serialize(x))
    data['recipes_df']['ingredient_names'] = data['recipes_df']['ingredient_names'].apply(lambda x: serialize(x))
    data['recipes_df']['ingredients'] = data['recipes_df']['ingredients'].apply(lambda x: serialize(x))

    if 'meal_types' in data['recipes_df']:
        data['recipes_df']['meal_types'] = data['recipes_df']['meal_types'].apply(lambda x: serialize(x))

    return data


def store_data(data, data_path, force_suffix_dict=None):
    ensure_dir(data_path)

    for name, value in data.items():
        if force_suffix_dict is not None and name in force_suffix_dict:
            force_suffix = force_suffix_dict[name]
            store_file(value, Path(data_path, name), force_suffix=force_suffix)
        else:
            store_file(value, Path(data_path, name))


def store_file(data, path, force_suffix=None):
    path = Path(path)

    if force_suffix is not None:
        forced_path = path.with_suffix(force_suffix)

        if force_suffix == '.yml':
            with open(forced_path, 'w') as file:
                yaml.dump(data, file)
        elif force_suffix == '.csv':
            data.to_csv(forced_path)
        elif force_suffix == '.pkl':
            with open(forced_path, 'wb') as pickle_file:
                pickle.dump(data, pickle_file)
        elif force_suffix == '.ttl':
            data.serialize(forced_path, format='turtle')
        elif force_suffix == '.owl':
            data.serialize(forced_path, format='xml')
        elif force_suffix == '.npy':
            np.save(forced_path, data)
        elif force_suffix == '.json':
            with open(forced_path, 'w', encoding='utf-8') as json_file:
                json.dump(data, json_file, ensure_ascii=False, indent=4)
        else:
            print('Extension "' + force_suffix + '" not possible with file "' + str(path.with_suffix('')) + '".')
    else:
        if isinstance(data, pd.DataFrame):
            path = path.with_suffix('.csv')
            data.to_csv(path)
        elif isinstance(data, dict) or isinstance(data, list):
            path = path.with_suffix('.pkl')
            with open(path, 'wb') as pickle_file:
                pickle.dump(data, pickle_file)
        elif isinstance(data, rdflib.graph.Graph):
            path = path.with_suffix('.ttl')
            data.serialize(path, format='turtle')
        elif isinstance(data, np.ndarray):
            path = path.with_suffix('.npy')
            np.save(path, data)
        else:
            try:
                path = path.with_suffix('.json')
                with open(path, 'w', encoding='utf-8') as json_file:
                    json.dump(data, json_file, ensure_ascii=False, indent=4)
            except:
                print('File "' + str(path) + '" is of an unsupported type.')


def update_config(new_value_pairs, config_path):
    with open(config_path, "r") as file:
        data = ryaml.YAML(typ='rt').load(file)

    for key, value in new_value_pairs.items():
        data[key] = value

    with open(config_path, "w") as file:
        ryaml.YAML(typ='rt').dump(data, file)


def load_data(path, s3=None):
    loaded_data = {}

    # Check if the directory exists and get all files in the directory
    if s3 is None:
        try:
            file_names = os.listdir(path)
        except FileNotFoundError:
            print(f"Path {str(path)} does not exist.")
            return loaded_data
    else:
        try:
            file_names = [object_meta.key for object_meta in s3.Bucket(path).objects.filter(Prefix="path/")]
        except Exception:
            print(f"S3-Path {str(path)} does not exist.")
            return loaded_data

    # Load all files in the directory
    for file_name in file_names:
        file_path = Path(path, file_name)

        if file_path.is_file():
            loaded_data[file_name.split('.')[0]] = load_file(file_path)

    return loaded_data


def load_folder_files(path, s3):
    file_names = []
    files = {}

    # Check if the directory exists and get all files in the directory
    if s3 is None:
        try:
            file_names = os.listdir(path)
        except FileNotFoundError:
            print(f"Path {str(path)} does not exist.")
    else:
        try:
            file_names = [object_meta.key for object_meta in s3.Bucket(path).objects.filter(Prefix="path/")]
        except Exception:
            print(f"S3-Path {str(path)} does not exist.")

    # Load all ontology imports
    for file_name in file_names:
        files[file_name] = load_file(Path(path, file_name), s3=None)

    return files


def load_file(path, s3=None):
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == '.yml':
        # Load YAML file as a dictionary
        if s3 is None:
            with open(path, 'r') as yaml_file:
                return yaml.load(yaml_file, Loader=yaml.CBaseLoader)
        else:
            with s3.open(path) as yaml_file:
                return yaml.load(yaml_file, Loader=yaml.CBaseLoader)
    elif suffix == '.csv':
        # Load CSV file as a Pandas DataFrame
        if s3 is None:
            try:
                df = pd.read_csv(path)
            except pd.errors.ParserError:
                df = pd.read_csv(path, sep=';')  # german csv file seperator
        else:
            with s3.open(path) as csv_file:
                try:
                    df = pd.read_csv(csv_file)
                except pd.errors.ParserError:
                    df = pd.read_csv(csv_file, sep=';')  # german csv file seperator

        # Clean old indices columns
        if 'Unnamed: 0' in df.columns:
            df.drop(columns=['Unnamed: 0'], inplace=True)

        return df
    elif suffix == '.pkl':
        # Load pickle file as a dictionary
        if s3 is None:
            with open(path, 'rb') as pickle_file:
                try:
                    return pickle.load(pickle_file)
                except EOFError as e:
                    print(f"EOFError: {e}.\n\n"
                          f"The pickle file {str(path)} is empty or corrupted. "
                          f"Continuing (if possible) with empty dict.")
                    return {}
        else:
            with s3.open(path, 'rb') as pickle_file:
                try:
                    return pickle.load(pickle_file)
                except EOFError as e:
                    print(f"EOFError: {e}.\n\n"
                          f"The pickle file {str(path)} is empty or corrupted. "
                          f"Continuing (if possible) with empty dict.")
                    return {}

    elif suffix == '.ttl':
        # Suppress RDFLib logging messages
        logging.getLogger("rdflib.term").setLevel(logging.CRITICAL)

        # Load Turtle file as a graph
        if s3 is None:
            rdf_graph = rdflib.Graph().parse(path, format='turtle')
        else:
            with s3.open(path) as ttl_file:
                rdf_graph = rdflib.Graph().parse(ttl_file, format='turtle')

        # Enable logging messages again
        logging.getLogger("rdflib.term").setLevel(logging.WARNING)

        return rdf_graph
    elif suffix == '.owl':
        # Suppress RDFLib logging messages
        logging.getLogger("rdflib.term").setLevel(logging.CRITICAL)

        # Load Owl file as a graph
        if s3 is None:
            rdf_graph = rdflib.Graph().parse(path, format='xml')
        else:
            with s3.open(path) as owl_file:
                rdf_graph = rdflib.Graph().parse(owl_file, format='xml')

        # Enable logging messages again
        logging.getLogger("rdflib.term").setLevel(logging.WARNING)

        return rdf_graph
    elif suffix == '.json':
        # Load JSON file as a dictionary
        if s3 is None:
            with open(path, 'r', encoding='utf-8') as json_file:
                return json.load(json_file)
        else:
            with s3.open(path, 'r', encoding='utf-8') as json_file:
                return json.load(json_file)
    elif suffix == '.npy':
        # Load Numpy file as an array
        if s3 is None:
            return np.load(path)
        else:
            with s3.open(path) as npy_file:
                return np.load(npy_file)
    elif suffix == '':
        # Load file and just read utf-8 content
        if s3 is None:
            with open(path, 'r', encoding='utf-8') as file:
                return file.read()
        else:
            with s3.open(path, 'r', encoding='utf-8') as file:
                return file.read()
    else:
        print(f"Unsupported file type for loading: {str(path)}")
        return None


def load_rdf_imports(path, rdf_graph, s3=None):
    # Suppress RDFLib logging messages
    logging.getLogger("rdflib.term").setLevel(logging.CRITICAL)

    # Check if the directory exists and get all files in the directory
    if s3 is None:
        try:
            file_names = os.listdir(path)
        except FileNotFoundError:
            print(f"Path {str(path)} does not exist.")
            return rdf_graph
    else:
        try:
            file_names = [object_meta.key for object_meta in s3.Bucket(path).objects.filter(Prefix="path/")]
        except Exception:
            print(f"S3-Path {str(path)} does not exist.")
            return rdf_graph

    # Load all ontology imports
    for file_name in file_names:
        file_path = Path(path, file_name)
        suffix = file_path.suffix.lower()

        if s3 is None:
            if suffix == '.owl':
                rdf_graph.parse(file_path, format='xml')
            elif suffix == '.ttl':
                rdf_graph.parse(file_path, format='turtle')
        else:
            with s3.open(file_path) as graph_file:
                if suffix == '.owl':
                    rdf_graph.parse(graph_file, format='xml')
                elif suffix == '.ttl':
                    rdf_graph.parse(graph_file, format='turtle')

    # Enable logging messages again
    logging.getLogger("rdflib.term").setLevel(logging.WARNING)

    return rdf_graph


def load_raw_hummus(data_location, recipe_column_names, review_column_names, member_column_names, s3=None):
    date_parser = lambda date_str: pd.to_datetime(date_str, format='%d/%m/%Y')
    user_date_parser = lambda date_str: pd.to_datetime(date_str, format='%m/%Y')

    if s3 is None:
        recipes = pd.read_csv(Path(data_location, 'recipes.csv'), parse_dates=['last_changed_date'],
                              date_parser=date_parser).loc[:, recipe_column_names]
        reviews = pd.read_csv(Path(data_location, 'reviews.csv'), parse_dates=['last_modified_date'],
                              date_parser=date_parser).loc[:, review_column_names]
        users = pd.read_csv(Path(data_location, 'members.csv'), parse_dates=['member_joined'],
                            date_parser=user_date_parser).loc[:, member_column_names]
    else:
        with s3.open(Path(data_location, 'recipes.csv')) as file:
            recipes = pd.read_csv(file, parse_dates=['last_changed_date'], date_parser=date_parser).loc[:,
                      recipe_column_names]
        with s3.open(Path(data_location, 'reviews.csv')) as file:
            reviews = pd.read_csv(file, parse_dates=['last_modified_date'], date_parser=date_parser).loc[:,
                      review_column_names]
        with s3.open(Path(data_location, 'members.csv')) as file:
            users = pd.read_csv(file, parse_dates=['member_joined'], date_parser=user_date_parser).loc[:,
                    member_column_names]

    return recipes, reviews, users


def load_additional_data(additional_location, s3=None):
    # Load additional data
    if s3 is None:
        additional_data_df = pd.read_csv(Path(additional_location, 'RAW_recipes.csv')).loc[:, ['id', 'name', 'tags']]
    else:
        with s3.open(Path(additional_location, 'RAW_recipes.csv')) as file:
            additional_data_df = pd.read_csv(file).loc[:, ['id', 'name', 'tags']]

    return additional_data_df


def load_graph_data(graph_location, include_usda, s3=None):
    # Load ingredient data
    if s3 is None:
        ingredients_df = pd.read_csv(Path(graph_location, 'graph_ingredient_mapping.csv')).loc[:,
                         ['ing_name', 'recipe', 'foodKG_mapped_ingredient']]
        labeled_ingredients_df = pd.read_csv(Path(graph_location, 'graph_ingredient_labels.csv')).loc[:,
                                 ['ingredient_name', 'ingredient_id', 'label']]
    else:
        with s3.open(Path(graph_location, 'graph_ingredient_mapping.csv')) as file:
            ingredients_df = pd.read_csv(file).loc[:, ['ing_name', 'recipe', 'foodKG_mapped_ingredient']]
        with s3.open(Path(graph_location, 'graph_ingredient_labels.csv')) as file:
            labeled_ingredients_df = pd.read_csv(file).loc[:, ['ingredient_name', 'ingredient_id', 'label']]

    # Load FoodOn ontology and its mapping
    food_on_ontology = load_file(Path(graph_location, 'foodon.owl'), s3=s3)
    food_on_ontology = load_rdf_imports(Path(graph_location, 'imports/'), food_on_ontology, s3=s3)
    food_on_mappings = load_file(Path(graph_location, 'food_kg_to_food_on.pkl'), s3=s3)

    # Load usda data and its mapping
    if include_usda:
        usda_df, usda_json = _load_usda_data(Path(graph_location, 'usda_data/'), s3=s3)
        usda_mappings = load_file(Path(graph_location, 'food_kg_to_usda.pkl'), s3=s3)
    else:
        usda_df, usda_json, usda_mappings = None, None, None

    return ingredients_df, labeled_ingredients_df, food_on_ontology, food_on_mappings, usda_df, usda_json, usda_mappings


# USDA datasets excluded by default. branded_foods.json is ~3.3GB of commercial
# packaged products; the FoodKG ingredient mappings resolve to SR Legacy foods
# (the KG build filters to dataType == 'SR Legacy'), so branded is never used.
# Set to () (or pass exclude=()) to include it.
USDA_EXCLUDED_FILES = ('branded_foods.json',)


def _load_usda_data(path, s3, exclude=USDA_EXCLUDED_FILES):
    # Load usda data, skipping excluded files (don't even read them into memory).
    exclude = set(exclude or ())
    if s3 is None:
        file_names = [f for f in os.listdir(path) if f.endswith('.json') and f not in exclude]
        usda_jsons = {f: load_file(Path(path, f), s3=None) for f in file_names}
    else:
        usda_data = load_folder_files(path, s3)
        usda_jsons = {k: v for k, v in usda_data.items()
                      if k.endswith('.json') and Path(k).name not in exclude}

    # Merge usda csv data on feature fdc_id
    # alt_usda_df = reduce(lambda left, right: _conditional_merge(left, right, on='fdc_id'), usda_csvs.values())

    # Merge usda json data
    usda_json = {nested_key: nested_value for outer_key, inner_dict in usda_jsons.items() for nested_key, nested_value
                 in inner_dict.items()}

    # Merge usda sources and convert to df
    usda_df = pd.DataFrame([food_desc for outer_key, food_list in usda_json.items() for food_desc in food_list])

    # Drop bulky/irrelevant columns. errors='ignore' because some of these
    # (foodUpdateLog, labelNutrients) only exist in the Branded dataset, which
    # is optional and may be excluded.
    usda_df.drop(columns=['foodUpdateLog', 'foodAttributes', 'labelNutrients'], inplace=True, errors='ignore')

    # Drop columns with only NaN values
    usda_df.dropna(axis=1, how='all', inplace=True)

    # For each column foodNutrients get the dict, retrieve the nutrition values and add them as columns
    usda_df = _retrieve_nutrition_from_usda(usda_df)

    return usda_df, usda_json


def _retrieve_nutrition_from_usda(usda_df):
    # Usda documentation: https://fdc.nal.usda.gov/data-documentation.html
    # And e.g. for branded food:
    # https://app.swaggerhub.com/apis/fdcnal/food-data_central_api/1.0.1#/BrandedFoodItem

    # Extract nutrients from foodNutrients column
    nutrients_df = pd.DataFrame(usda_df['foodNutrients'].apply(_extract_nutrients).tolist())
    usda_df = pd.concat([usda_df, nutrients_df], axis=1)

    # Drop column 'foodNutrients'
    usda_df.drop(columns=['foodNutrients'], inplace=True)

    return usda_df


def _extract_nutrients(nutrient_list):
    nutrients = {
        f"{nutrient_dict['nutrient']['name'].lower()} [{nutrient_dict['nutrient']['unitName'].lower()}]":
            nutrient_dict['amount'] for nutrient_dict in nutrient_list
        if nutrient_dict['type'] == 'FoodNutrient' and 'amount' in nutrient_dict
    }

    return nutrients


def _conditional_merge(left, right, on='fdc_id'):
    if on in left.columns and on in right.columns:
        # Ensure only one id column is kept
        merged = pd.merge(left, right, on=on, suffixes=('', '_drop'))
        merged = merged[[col for col in merged.columns if not col.endswith('_drop')]]
        return merged
    else:
        # print(f"Skipping merge as '{on}' column is missing in one of the DataFrames: {left.columns}, {right.columns}")
        return left


def remove_directory(path):
    # Check if the directory exists
    if os.path.exists(path) and os.path.isdir(path):
        # Get a list of all files in the directory
        files = os.listdir(path)

        # Loop through the files and delete each one
        for file in files:
            file_path = Path(path, file)
            os.remove(file_path)
            print(f"File {str(file_path)} has been deleted.")

        print(f"All files in {str(path)} have been deleted.")
    else:
        print(f"The directory at {str(path)} does not exist.")

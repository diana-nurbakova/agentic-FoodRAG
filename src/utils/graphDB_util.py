import json
import pandas as pd
import requests

from src.utils import rdf_util

# Graph DB Values
GRAPHDB_URL = 'http://localhost:7200/'
GRAPHDB_REPOSITORY_NAME = 'HUMMUS'

# Entity Types
FOOD_ON_FOOD_PRODUCT = rdf_util.FOOD_ON_FOOD_PRODUCT
FOOD_ON_CHEESE = rdf_util.FOOD_ON_CHEESE
FOOD_ON_BEVERAGE = rdf_util.FOOD_ON_BEVERAGE
FOOD_ON_PLANT = rdf_util.FOOD_ON_PLANT  # But also includes e.g. mushrooms
FOOD_ON_VEGGIE = rdf_util.FOOD_ON_VEGGIE
FOOD_ON_FRUIT = rdf_util.FOOD_ON_FRUIT
FOOD_ON_WATER = rdf_util.FOOD_ON_WATER

# RDF Properties
RDF_TYPE = rdf_util.RDF_TYPE
RDFS_LABEL = rdf_util.RDFS_LABEL
RDFS_PROPERTY = rdf_util.RDFS_PROPERTY
RDFS_SUBCLASS = rdf_util.RDFS_SUBCLASS
OWL_EQUIVALENT_CLASS = rdf_util.OWL_EQUIVALENT_CLASS

# FoodKG URLS
FOOD_KG_USES = 'http://idea.rpi.edu/heals/kb/uses'
FOOD_KG_RECIPE = 'http://idea.rpi.edu/heals/kb/recipe'
FOOD_COM_USER = 'https://www.food.com/user'
FOOD_COM_REVIEW = 'https://www.food.com/review'
FOOD_COM_TAG = 'https://www.food.com/tag'
FOOD_COM_JOINED_AT = 'https://www.food.com/user/joined_at'
FOOD_COM_FOLLOWS_COUNT = 'https://www.food.com/user/follows_count'
FOOD_COM_FOLLOWER_COUNT = 'https://www.food.com/user/follower_count'
FOOD_COM_HAS_AUTHOR = 'https://www.food.com/recipe/has_author'
FOOD_COM_HAS_DESCRIPTION = 'https://www.food.com/recipe/has_description'
FOOD_COM_HAS_DURATION = 'https://www.food.com/recipe/has_duration'
FOOD_COM_HAS_DIRECTIONS = 'https://www.food.com/recipe/has_directions'
FOOD_COM_SERVES = 'https://www.food.com/recipe/serves'
FOOD_COM_LAST_CHANGED_AT = 'https://www.food.com/recipe/last_changed_at'
FOOD_COM_HAS_TAG = 'https://www.food.com/recipe/has_tag'
FOOD_COM_AMOUNT_OF_SERVINGS = 'https://www.food.com/recipe/amount_of_servings'
FOOD_COM_HAS_SERVING_SIZE = 'https://www.food.com/recipe/has_serving_size'
FOOD_COM_HAS_CALORIES = 'https://www.food.com/recipe/has_calories'
FOOD_COM_HAS_CALORIES_FROM_FAT = 'https://www.food.com/recipe/has_calories_from_fat'
FOOD_COM_HAS_TOTAL_FAT = 'https://www.food.com/recipe/has_total_fat'
FOOD_COM_HAS_SATURATED_FAT = 'https://www.food.com/recipe/has_saturated_fat'
FOOD_COM_HAS_CHOLESTEROL = 'https://www.food.com/recipe/has_cholesterol_mg'
FOOD_COM_HAS_SODIUM = 'https://www.food.com/recipe/has_sodium_mg'
FOOD_COM_HAS_TOTAL_CARBS = 'https://www.food.com/recipe/has_total_carbohydrate'
FOOD_COM_HAS_DIETARY_FIBER = 'https://www.food.com/recipe/has_dietary_fiber'
FOOD_COM_HAS_SUGARS = 'https://www.food.com/recipe/has_sugars'
FOOD_COM_HAS_PROTEIN = 'https://www.food.com/recipe/has_protein'
FOOD_COM_HAS_REVIEW = 'https://www.food.com/recipe/has_review'
FOOD_COM_HAS_REVIEWER = 'https://www.food.com/review/has_reviewer'
FOOD_COM_HAS_RATING = 'https://www.food.com/review/has_rating'
FOOD_COM_HAS_TEXT_REVIEW = 'https://www.food.com/review/has_text_review'
FOOD_COM_LAST_MODIFIED_AT = 'https://www.food.com/review/last_modified_at'
FOOD_COM_HAS_LIKES = 'https://www.food.com/review/has_likes'


def get_specific_ingredients_types(entity_url, food_on_types):
    return _get_ingredients_types(entity_url, True, food_on_types)


def get_ingredients_types(entity_url):
    return _get_ingredients_types(entity_url, False)


def _get_ingredients_types(entity_url, specific, food_on_types=None):
    part0 = """PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX owl: <http://www.w3.org/2002/07/owl#>
        PREFIX foodkg: <http://idea.rpi.edu/heals/kb/>
        PREFIX foodon: <http://purl.obolibrary.org/obo/FOODON_>
        SELECT DISTINCT ?food_kg_ingredient ?type
        WHERE {
        """
    if specific:
        part1 = 'VALUES ?types {' + '\n'.join(map(lambda x: '<' + x + '>', food_on_types)) + '}\n'
    else:
        part1 = ''

    part2 = '<' + entity_url + '> a foodkg:recipe;\n' + \
            'foodkg:uses ?food_kg_ingredient.\n'

    if specific:
        part3 = 'BIND (exists{?food_kg_ingredient owl:equivalentClass/rdfs:subClassOf* ?types} AS ?type)}'
    else:
        part3 = '?food_kg_ingredient owl:equivalentClass/rdfs:subClassOf ?type\n' + \
                'FILTER (isIRI(?type))}'

    query = part0 + part1 + part2 + part3
    json_raw, _ = sparql_request(query, False)

    ingredients = [row['food_kg_ingredient']['value'] for row in json_raw['results']['bindings']]
    types = [row['type']['value'] for row in json_raw['results']['bindings']]
    result = [[i, t] for i, t in zip(ingredients, types)]
    return pd.DataFrame(result, columns=['food_kg_ingredient', 'type'])


def is_also_type(entity_url, food_on_type):
    return _is_type(entity_url, food_on_type, False)


def is_only_type(entity_url, food_on_type):
    return _is_type(entity_url, food_on_type, True)


def _is_type(entity_url, food_on_type, solely):
    part0 = """PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX owl: <http://www.w3.org/2002/07/owl#>
        PREFIX foodkg: <http://idea.rpi.edu/heals/kb/>
        PREFIX foodon: <http://purl.obolibrary.org/obo/FOODON_>
        ASK
        WHERE {
        """
    part1 = '<' + entity_url + '''> a foodkg:recipe;
       foodkg:uses/owl:equivalentClass ?ingredient.
       ?ingredient rdfs:subClassOf* <''' + food_on_type + '>.\n'

    part2 = 'FILTER NOT EXISTS{ #remove all recipes having other ingredients than cheese\n' + \
            '<' + entity_url + '> foodkg:uses/owl:equivalentClass ?nonCheeseIngredient.\n' + \
            'FILTER NOT EXISTS {\n' + \
            '?nonCheeseIngredient rdfs:subClassOf* <' + food_on_type + '>.}}'

    part3 = '}'

    if solely:
        query = part0 + part1 + part2 + part3
    else:
        query = part0 + part1 + part3

    json_raw, _ = sparql_request(query, False)
    return json_raw['boolean']


def sparql_request(query, show_pretty):
    url = GRAPHDB_URL + "repositories/" + GRAPHDB_REPOSITORY_NAME

    params = dict(
        repositoryID=GRAPHDB_REPOSITORY_NAME,
        query=query,
        Accept='application/sparql-results+json'
    )

    answer = requests.get(url=url, params=params)

    try:
        json_raw = json.loads(answer.content)

        if show_pretty:
            pretty_string = json.dumps(json_raw, indent=4, sort_keys=True)
        else:
            pretty_string = ''
    except Exception:
        json_raw = None
        pretty_string = None

    if not answer.ok:
        raise requests.exceptions.RequestException(answer, json_raw)

    return json_raw, pretty_string

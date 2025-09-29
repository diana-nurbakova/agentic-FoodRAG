import json
import pandas as pd
from llama_index.core import VectorStoreIndex
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.indices.vector_store import VectorIndexRetriever
from llama_index.core.llms import LLM
from pandas import DataFrame
from pydantic import Field
from pydantic.fields import FieldInfo
from typing import List, Dict, Set, Any

from src.food_rag.retriever import util_retriever
from src.food_rag.tools.tool_base import RecipeToolBase
from src.food_rag.utils import hummus_docify
from src.utils import hummus_util
from src.utils.rdf_util import PreProcessedFoodOn

DEBUG = "debug"
EMBEDDING = "embedding"
DOC_2_VEC = "doc2vec"
BOW_COSINE_RETRIEVER = "bow_cosine_retriever"
RETRIEVING_MODES = [DEBUG, EMBEDDING, DOC_2_VEC]


class IngredientSubstitutionTool(RecipeToolBase):
    """
    A tool that substitutes ingredients using different data sources.

    Possible sources:
    - Ingredients dataframe: The ingredients of the food.com recipes.
    - Food-on ontology: The food-on food product ontology/hierarchy.
    - USDA food products: The usda food products dataframe including nutritional information for each
    product/ingredient.
    """
    TOOL_NAME = "ingredient_substitution_tool"

    def __init__(self, hummus_data, llm: LLM, embed_model: BaseEmbedding, similarity_top_k=3, min_similarity_score=0.75,
                 n_ingredients=20, ingredient_embeddings=[], food_product_embeddings=[], food_on_embeddings=[],
                 parent_depth=1, child_depth=1, retrieving_mode=EMBEDDING, verbose: bool = True):
        """
        Initialize the tool with a recipe query engine. Note: Requires ~10 minutes to init on my laptop.

        :param hummus_data: The hummus data object containing all necessary data.
        :param LLM llm: The language model instance.
        :param BaseEmbedding embed_model: The embedding model.
        :param int similarity_top_k: The number of similar ingredients to retrieve for the index itself.
        :param float min_similarity_score: The minimum similarity score to consider an ingredient as similar.
        :param int n_ingredients: The number of ingredients to return.
        :param dict ingredient_embeddings: The ingredient embeddings.
        :param dict food_product_embeddings: The food product embeddings.
        :param dict food_on_embeddings: The food-on embeddings.
        :param int parent_depth: The depth to search for parent nodes in the food-on ontology. Default is 1.
        :param int child_depth: The depth to search for child nodes in the food-on ontology. Default is 1.
        :param str retrieving_mode: The mode to use for retrieving similar ingredients. Can be any item of
        RETRIEVING_MODES. The embeddings are only used with embedding mode EMBEDDING. Default is EMBEDDING.
        :param bool verbose: Whether to print the ingredient substitution step.
        """
        # Check if the retrieving mode is valid
        if retrieving_mode not in RETRIEVING_MODES:
            raise ValueError(f"Invalid retrieving mode: {retrieving_mode}. Must be one of {RETRIEVING_MODES}.")

        super().__init__(fn=self.find_ingredient_substitutions, tool_name=self.TOOL_NAME, verbose=verbose)
        self.hummus_data = hummus_data
        self.ingredients_df: DataFrame = hummus_data['ingredients_df']
        self.food_on: PreProcessedFoodOn = PreProcessedFoodOn(hummus_data)
        self.similarity_top_k = similarity_top_k
        self.min_similarity_score = min_similarity_score
        self.n_ingredients = n_ingredients
        self.parent_depth = parent_depth
        self.child_depth = child_depth
        self.retrieving_mode = retrieving_mode
        self.llm = llm

        # Clean the usda dataframe
        self.usda_df, _, _ = hummus_docify.clean_df(
            df=hummus_data['usda_df'],
            id_column=None,
            doc_type="food_product",
            rename_columns={'id': 'food_product_id', 'ndbNumber': 'usda_id', 'description': 'name',
                            'brandOwner': 'brand_owner', 'servingSize': 'serving_size',
                            'servingSizeUnit': 'serving_size_unit', 'brandedFoodCategory': 'food_category',
                            'total lipid (fat) [g]': 'total fat [g]',
                            'carbohydrate, by difference [g]': 'carbohydrate [g]',
                            'fiber, total dietary [g]': 'fiber [g]', 'calcium, ca [mg]': 'calcium [mg]',
                            'iron, fe [mg]': 'iron [mg]', 'sodium, na [mg]': 'sodium [mg]',
                            'vitamin c, total ascorbic acid [mg]': 'vitamin c [mg]',
                            'fatty acids, total trans [g]': 'trans fat [g]',
                            'fatty acids, total saturated [g]': 'saturated fat [g]'},
            drop_columns=[],
            string_columns=['name', 'brand_owner', 'ingredients', 'serving_size_unit', 'food_category'],
            number_columns=['serving_size', 'protein [g]', 'total fat [g]', 'carbohydrate [g]', 'energy [kcal]',
                            'total sugars [g]', 'fiber [g]', 'calcium [mg]', 'iron [mg]', 'sodium [mg]',
                            'vitamin c [mg]', 'cholesterol [mg]', 'trans fat [g]', 'saturated fat [g]'],
            list_columns=[],
            date_columns=[],
            included_content_features=['type', 'name', 'food_product_id', 'usda_id', 'serving_size',
                                       'serving_size_unit', 'food_category', 'serving_size', 'protein [g]',
                                       'total fat [g]', 'carbohydrate [g]', 'energy [kcal]', 'total sugars [g]',
                                       'fiber [g]', 'calcium [mg]', 'iron [mg]', 'sodium [mg]', 'vitamin c [mg]',
                                       'cholesterol [mg]', 'trans fat [g]', 'saturated fat [g]'],
            content_functions={},
        )

        # Map the usda mappings to the ids directly, now: food.com url --> usda id
        self.usda_mappings = {key: int(value.split("#")[-1]) for key, value in hummus_data['usda_mappings'].items()}

        # Invert the  usda mappings, now: usda id --> food.com url
        self.usda_mappings_inv = hummus_util.invert_dict(self.usda_mappings)

        # Map the food_on mappings to the ids directly, now: food.com url --> food_on url
        self.food_on_mappings = hummus_data['food_on_mappings']

        # Invert the food_on mappings, now: food_on url --> food.com url
        self.food_on_mappings_inv = hummus_util.invert_dict(self.food_on_mappings)

        # ingredient id --> food.com url
        self.ingredients_mapping = hummus_data['ingredients_dict']

        # Concat all nodes with embeddings
        self.node_embeddings = ([node for node_list in ingredient_embeddings.values() for node in node_list]
                                + food_product_embeddings + food_on_embeddings)

        # Build index & retriever for querying. Skip it when debugging.
        if retrieving_mode == EMBEDDING:
            self.retriever = VectorIndexRetriever(  # Takes some time to build the index
                index=VectorStoreIndex(self.node_embeddings),
                similarity_top_k=similarity_top_k,
                embed_model=embed_model
            )
        elif retrieving_mode == DEBUG:
            self.retriever = util_retriever.RandomRetriever(
                nodes=self.node_embeddings,
                k=similarity_top_k,
                score_method="1"
            )
            print("Warning: Selecting random similar ingredients because of DEBUG mode.")
        elif retrieving_mode == DOC_2_VEC:
            self.retriever = util_retriever.Doc2VecRetriever(
                nodes=self.node_embeddings,
                similarity_top_k=similarity_top_k,
            )
        elif retrieving_mode == BOW_COSINE_RETRIEVER:
            self.retriever = util_retriever.BoWCosineRetriever(
                nodes=self.node_embeddings,
                similarity_top_k=similarity_top_k,
            )
        else:
            raise ValueError(f"Invalid retrieving mode: {retrieving_mode}. Must be one of {RETRIEVING_MODES}.")

        # For backward compatibility possibly create an usda_id dict (for the case it is not part of the metadata of the
        # embedded usda data). Dict created from self.usda_df with format: food_product_id --> usda_id
        self.usda_id_dict = self.usda_df.set_index('food_product_id')['usda_id'].to_dict()

    def find_ingredient_substitutions(self, ingredients: List[str] =
    Field(description="The original ingredients which shall be substituted or for whose similar ones should be"
                      " fetched.")) -> str:
        """
        Find substitutions or similar ingredients.
        """
        if isinstance(ingredients, FieldInfo):
            raise ValueError("The 'ingredients' parameter must be provided as a list of strings.")

        ingredients_dict = {}
        for ingredient in ingredients:
            # Get the most similar ingredients
            similar_ingredients = self.retriever.retrieve(ingredient)

            # Filter by similarity score
            similar_ingredients = [ingredient for ingredient in similar_ingredients if
                                   ingredient.score >= self.min_similarity_score]

            # Split ingredients into different categories
            recipe_ingredients_df = pd.DataFrame([ingredient.metadata for ingredient in similar_ingredients if
                                                  ingredient.metadata.get("type") == "ingredient"])
            usda_ingredients_df = pd.DataFrame([ingredient.metadata for ingredient in similar_ingredients if
                                                ingredient.metadata.get("type") == "food_product"])
            food_on_ingredients_df = pd.DataFrame([ingredient.metadata for ingredient in similar_ingredients if
                                                   ingredient.metadata.get("type") == "food_on_product"])

            # Convert all ingredients to food_on
            food_on_urls = set(
                _recipe_ingredients_to_food_on(recipe_ingredients_df, self.ingredients_mapping, self.food_on_mappings)
                + _usda_to_food_on(usda_ingredients_df, self.usda_id_dict, self.usda_mappings_inv,
                                   self.food_on_mappings)
                + _food_on_to_urls(food_on_ingredients_df))

            # Explore the food_on ingredients
            food_on_ingredients = _food_on_explore(food_on_urls, self.food_on, self.parent_depth, self.child_depth)

            # Convert all ingredients back to usda (and also keep the original usda ingredients which might have been lost
            # if not mapped)
            converted_usda_ingredients_df = _food_on_to_usda(food_on_ingredients, self.food_on_mappings_inv,
                                                             self.usda_mappings, self.usda_df)

            if usda_ingredients_df.empty:
                combined_ingredients_ids = set(converted_usda_ingredients_df['food_product_id'])
            else:
                combined_ingredients_ids = set(converted_usda_ingredients_df['food_product_id']).union(
                    set(usda_ingredients_df['food_product_id']))

            usda_ingredients_df = self.usda_df[self.usda_df['food_product_id'].isin(combined_ingredients_ids)]

            # Filter and sort the usda ingredients
            usda_ingredient_df = usda_ingredients_df[:self.n_ingredients]

            # Check for duplicate column names and remove them
            if usda_ingredient_df.columns.duplicated().any():
                usda_ingredient_df = usda_ingredient_df.loc[:, ~usda_ingredient_df.columns.duplicated()]

            # Docify the usda ingredients
            ingredient_documents = hummus_docify.food_products_to_docs(usda_ingredient_df, verbose=False)
            ingredient_texts = [doc.metadata.get('name', 'unnamed ingredient') for doc in ingredient_documents]

            # Add the ingredient texts to the dict
            ingredients_dict[ingredient] = ingredient_texts

        return json.dumps(ingredients_dict, indent=4, sort_keys=True)


class IngredientHierarchyTool(RecipeToolBase):
    """
    A tool that retrieves the ingredient hierarchy for given ingredients from the food-on ontology.
    """

    TOOL_NAME = "ingredient_hierarchy_tool"

    def __init__(self, hummus_data, llm: LLM, embed_model: BaseEmbedding, similarity_top_k=3, min_similarity_score=0.9,
                 food_on_embeddings=[], parent_depth=3, child_depth=3, retrieving_mode=EMBEDDING, verbose: bool = True):
        """
        Initialize the tool with a recipe query engine. Note: Requires ~10 minutes to init on my laptop.

        :param hummus_data: The hummus data object containing all necessary data.
        :param LLM llm: The language model instance.
        :param BaseEmbedding embed_model: The embedding model.
        :param int similarity_top_k: The number of similar ingredients to retrieve for the index itself.
        :param float min_similarity_score: The minimum similarity score to consider an ingredient as similar.
        :param dict food_on_embeddings: The food-on embeddings.
        :param int parent_depth: The depth to search for parent nodes in the food-on ontology. Default is 3.
        :param int child_depth: The depth to search for child nodes in the food-on ontology. Default is 3.
        :param str retrieving_mode: The mode to use for retrieving similar ingredients. Can be any item of
        RETRIEVING_MODES. The embeddings are only used with embedding mode EMBEDDING. Default is EMBEDDING.
        :param bool verbose: Whether to print the ingredient substitution step.
        """
        # Check if the retrieving mode is valid
        if retrieving_mode not in RETRIEVING_MODES:
            raise ValueError(f"Invalid retrieving mode: {retrieving_mode}. Must be one of {RETRIEVING_MODES}.")

        super().__init__(fn=self.get_ingredient_hierarchy, tool_name=self.TOOL_NAME, verbose=verbose)
        self.hummus_data = hummus_data
        self.ingredients_df: DataFrame = hummus_data['ingredients_df']
        self.food_on: PreProcessedFoodOn = PreProcessedFoodOn(hummus_data)
        self.similarity_top_k = similarity_top_k
        self.min_similarity_score = min_similarity_score
        self.parent_depth = parent_depth
        self.child_depth = child_depth
        self.retrieving_mode = retrieving_mode
        self.llm = llm

        # Map the food_on mappings to the ids directly, now: food.com url --> food_on url
        self.food_on_mappings = hummus_data['food_on_mappings']

        # Invert the food_on mappings, now: food_on url --> food.com url
        self.food_on_mappings_inv = hummus_util.invert_dict(self.food_on_mappings)

        # ingredient id --> food.com url
        self.ingredients_mapping = hummus_data['ingredients_dict']

        # Concat all nodes with embeddings
        self.food_on_embeddings = food_on_embeddings

        # Build index & retriever for querying. Skip it when debugging.
        if retrieving_mode == EMBEDDING:
            self.retriever = VectorIndexRetriever(  # Takes some time to build the index
                index=VectorStoreIndex(self.food_on_embeddings),
                similarity_top_k=similarity_top_k,
                embed_model=embed_model
            )
        elif retrieving_mode == DEBUG:
            self.retriever = util_retriever.RandomRetriever(
                nodes=self.food_on_embeddings,
                k=similarity_top_k,
                score_method="1"
            )
            print("Warning: Selecting random similar ingredients because of DEBUG mode.")
        elif retrieving_mode == DOC_2_VEC:
            self.retriever = util_retriever.Doc2VecRetriever(
                nodes=self.food_on_embeddings,
                similarity_top_k=similarity_top_k,
            )
        elif retrieving_mode == BOW_COSINE_RETRIEVER:
            self.retriever = util_retriever.BoWCosineRetriever(
                nodes=self.food_on_embeddings,
                similarity_top_k=similarity_top_k,
            )
        else:
            raise ValueError(f"Invalid retrieving mode: {retrieving_mode}. Must be one of {RETRIEVING_MODES}.")

    def get_ingredient_hierarchy(self, ingredients: List[str] =
    Field(description="The original ingredients for whose the hierarchy is generated.")) -> str:
        """
        Find ingredients in the hierarchy.
        """
        if isinstance(ingredients, FieldInfo):
            raise ValueError("The 'ingredients' parameter must be provided as a list of strings.")

        ingredients_dict = {}
        for ingredient in ingredients:
            # Get the most similar ingredients
            similar_ingredients = self.retriever.retrieve(ingredient)

            # Filter by similarity score
            similar_ingredients = [ingredient for ingredient in similar_ingredients if
                                   ingredient.score >= self.min_similarity_score]

            # Split ingredients into different categories
            food_on_ingredients_df = pd.DataFrame([ingredient.metadata for ingredient in similar_ingredients if
                                                   ingredient.metadata.get("type") == "food_on_product"])

            # Convert all ingredients to food_on
            food_on_urls = _food_on_to_urls(food_on_ingredients_df)

            # Explore the food_on ingredients
            food_on_ingredients: Dict[str, Dict[str, List[str]]] = _food_on_hierarchy(food_on_urls, self.food_on,
                                                                                      self.parent_depth,
                                                                                      self.child_depth)

            # Add the ingredient texts to the dict
            ingredients_dict[f"Hierarchy for ingredient '{ingredient}'"] = food_on_ingredients

        # Remove all empty entries recursively
        cleaned_data = _remove_empty_entries(ingredients_dict)

        return json.dumps(cleaned_data, indent=4, sort_keys=True)


def _recipe_ingredients_to_food_on(ingredients_df, ingredients_mapping, food_on_mappings):
    if len(ingredients_df) == 0:
        return []
    else:
        # ingredient id --> food.com url
        food_com_urls = set([url for ingredient_id in ingredients_df['ingredient_id']
                             if (url := ingredients_mapping[ingredient_id]) is not None])

        # food.com url --> food_on url
        food_on_urls = set([mapped_url for url in food_com_urls if
                            (mapped_url := food_on_mappings.get(url)) is not None])

        return list(food_on_urls)


def _usda_to_food_on(usda_df, usda_id_dict, usda_mappings_inv, food_on_mappings):
    if len(usda_df) > 0:
        if 'usda_id' not in usda_df.columns:
            # Map the usda ids to the food product ids
            usda_df['usda_id'] = usda_df['food_product_id'].map(usda_id_dict)

        # usda id --> food.com url
        food_com_urls = set(
            url for product_id in usda_df['usda_id']
            if (url_list := usda_mappings_inv.get(product_id)) is not None
            for url in url_list
        )

        # food.com url --> food_on url
        food_on_urls = set([mapped_url for url in food_com_urls
                            if (mapped_url := food_on_mappings.get(url)) is not None])

        return list(food_on_urls)
    else:
        return []


def _food_on_to_urls(food_on_df):
    return food_on_df.get('node', pd.Series()).tolist()


def _food_on_explore(food_on_urls, food_on, parent_depth, child_depth):
    parent_nodes = [
        parent_node for food_on_url in food_on_urls
        for parent_list in [food_on.get_parent_nodes(food_on_url, as_url=True, depth=parent_depth)]
        for parent_path in parent_list
        for parent_node in parent_path
    ]

    child_nodes = [
        child_node for food_on_url in food_on_urls
        for child_list in [food_on.get_child_nodes(food_on_url, as_url=True, depth=child_depth)]
        for child_node in child_list
    ]

    combined_nodes = set(parent_nodes + child_nodes)
    combined_urls = [node.toPython() for node in combined_nodes]

    # Original urls are already included in the parent nodes, thus no need to add them again, just for readability
    all_urls = list(set(list(food_on_urls) + combined_urls))

    return all_urls


def _food_on_to_usda(food_on_ingredients, food_on_mappings_inv, usda_mappings, usda_df):
    # food_on url --> food.com url
    food_com_urls = [mapped_url for url in food_on_ingredients
                     if (mapped_url := food_on_mappings_inv.get(url)) is not None]
    food_com_urls = set([url for url_list in food_com_urls for url in url_list])

    # food.com url --> usda id
    usda_ids = [usda_id for url in food_com_urls
                if (usda_id := usda_mappings.get(url)) is not None]

    # get the usda ingredients by filtering the usda dataframe
    usda_ingredients = usda_df[usda_df['usda_id'].isin(usda_ids)]

    return usda_ingredients


def _food_on_hierarchy(food_on_urls: Set[str], food_on, parent_depth: int = 5, child_depth: int = 5) \
        -> Dict[str, Dict[str, List[str]]]:
    hierarchy_dict: Dict[str, Dict[str, List[str]]] = {}

    for url in food_on_urls:
        # init dict for current url
        url_dict: Dict[str, List[str]] = {}

        # Get current url name
        url_names = food_on.get_node_names([url])
        url_name = url_names[0] if url_names else url

        # Get the hierarchy by getting parent and child nodes
        parent_paths = food_on.get_parent_nodes(url, as_url=False, depth=parent_depth)
        parent_paths = [node_name for path in parent_paths for node_name in path if node_name != url_name]
        child_paths = food_on.get_child_nodes(url, as_url=False, depth=child_depth)

        # Add node names to the url_dict
        url_dict['Parents (super nodes)'] = list(set(parent_paths))
        url_dict['Children (sub nodes)'] = list(set(child_paths))

        # Add the url_dict to the hierarchy_dict
        hierarchy_dict[url_name] = url_dict

    return hierarchy_dict

def _remove_empty_entries(data: Any) -> Any:
    if isinstance(data, dict):
        cleaned = {
            k: _remove_empty_entries(v)
            for k, v in data.items()
            if v not in ("", [], {}) and _remove_empty_entries(v) not in ("", [], {}, None)
        }
        return cleaned if cleaned else None

    elif isinstance(data, list):
        cleaned_list = [_remove_empty_entries(item) for item in data]
        cleaned_list = [item for item in cleaned_list if item not in ("", [], {}, None)]
        return cleaned_list if cleaned_list else None

    return data

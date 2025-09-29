import pandas as pd
from llama_index.core import Document, QueryBundle, PromptTemplate
from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.base.response.schema import RESPONSE_TYPE, Response
from llama_index.core.callbacks import CallbackManager
from llama_index.core.prompts.mixin import PromptMixinType
from llama_index.core.schema import QueryType
from typing import List, Optional, Dict, Any

from src.food_rag.retriever.df_retriever_base import DataFrameRetrieverBase
from src.food_rag.utils import hummus_docify
from src.utils.hummus_util import try_parse

DOCUMENT_DELIMITER = ".\n\n"
DEFAULT_TEMPLATE = PromptTemplate("""
You are a recipe recommender. Follow these rules:

1) Pick the best matching recipe for the user query.
2) Pick only one recipe.
3) Return only the title of the recipe in json format, like this:```json{"title": "<recipe_title>"}```
4) If there are no available recipes, return an empty string.
5) DO NOT ANSWER THE USER QUERY DIRECTLY BUT FOLLOW THESE INSTRUCTIONS.

User Query: {query}

Available Recipes:
{context}

Recommendation as json:
""")

MEAL_PLAN_TEMPLATE = PromptTemplate("""
You are a meal plan recommender. Follow these rules:

1) For each meal type (breakfast, lunch, dinner, snack) pick the best matching recipe for the user query.
2) Pick only one recipe per meal type, such that in total 4 recipes are returned.
3) Return only the title of the recipes in json format, like this:```json{"breakfast":{"title": "<recipe_title>"], "lunch": {"title": "<recipe_title>"}, "dinner": {"title": "<recipe_title>"}, "snack": {"title": "<recipe_title>"}}```
4) If there are no available recipes, return an empty string.
5) DO NOT ANSWER THE USER QUERY DIRECTLY BUT FOLLOW THESE INSTRUCTIONS.

User Query: {query}

Available Recipes:
{context}

Recommendation as json:
""")

LLM_DF_INPUT_KEY = 'llm_input_df'
RETRIEVED_DF_KEY = 'full_retrieved_df'


class RecipeRAGQueryEngine(BaseQueryEngine, DataFrameRetrieverBase):

    def __init__(self, retriever: DataFrameRetrieverBase, llm, callback_manager: Optional[CallbackManager] = None,
                 ranking_strategy: str = None, k: int = 5, prompt_template: PromptTemplate = None, llm_calls: int = 1,
                 return_full_recipes: bool = False, return_meal_plan: bool = False, debug: bool = False):
        """
        Initializes the RAG (Retrieval-Augmented Generation) pipeline.

        :param DataFrameRetrieverBase retriever: A custom retriever that returns a DataFrame.
        :param llm: An initialized language model instance.
        :param Optional[CallbackManager] callback_manager: A callback manager instance.
        :param str ranking_strategy: A ranking strategy to be used for the retrieved recipes. If None, no ranking is
        applied. If 'rating', recipes are ranked by their rating.
        :param int k: The number of recipes to retrieve before passed to the llm. Default is 5. If no ranking is
        applied, recipes are just filtered by occurrence in the retrieved DataFrame. If None, all recipes are passed
        to the llm.
        :param PromptTemplate prompt_template: A prompt template to be used for the llm. If None it is the
        DEFAULT_TEMPLATE.
        :param int llm_calls: The number of times the llm is called for one query call. Default is 1. If > 1, the llm is
         also called by the underlying retriever. In that case the counter must be set here correctly.
        :param bool return_full_recipes: If True, the full recipes are returned in the response if available. Else, only
         recipe title are returned if default template is used. Default is False.
        :param bool return_meal_plan: If True, a meal plan for all available meal types is returned. Default is False.
        :param bool debug: A flag to enable debug mode. If True, debug information is appended to the llms response in
        the metadata. Default is False.
        """

        super().__init__(callback_manager)
        self.retriever = retriever
        self.llm = llm
        self.ranking_strategy = ranking_strategy
        self.k = k
        self.debug = debug
        self.llm_calls = llm_calls
        self.return_full_recipes = return_full_recipes
        self.return_meal_plan = return_meal_plan
        self.prompt_template = prompt_template or (MEAL_PLAN_TEMPLATE if return_meal_plan else DEFAULT_TEMPLATE)

    @staticmethod
    def prepare_df_for_llm(recipes_df: pd.DataFrame) -> str:
        """
        Prepares the retrieved DataFrame as input for the LLM.

        :param pd.DataFrame recipes_df: Retrieved DataFrame of recipes.
        :returns: Formatted text to be used as input for the LLM.
        :rtype: str
        """
        documents: List[Document] = hummus_docify.recipes_to_docs(recipes_df, verbose=False)

        if len(documents) == 0:
            return "No fitting recipes could be found."
        else:
            return DOCUMENT_DELIMITER.join([document.text for document in documents])

    def _query(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        """
        Handles synchronous queries from the QueryEngine interface.

        :param QueryBundle query_bundle: Query bundle from the LlamaIndex framework.
        :returns: Response object containing the answer and any relevant metadata.
        """
        query = query_bundle.query_str
        metadata = {'query': query}

        # Change "recipe" to "meal plan" in the query if meal plan is requested
        if self.return_meal_plan and 'recipes' in query.lower():
            query = query.replace('recipes', 'meal plan')
        elif self.return_meal_plan and 'recipe' in query.lower():
            query = query.replace('recipe', 'meal plan')
        elif self.return_meal_plan and 'meal' not in query.lower():
            query = f"Create a meal plan for: {query}"

        # Retrieve recipes
        retrieved_df = self.retriever.retrieve(query)

        # Update metadata
        if self.debug:
            metadata[RETRIEVED_DF_KEY] = retrieved_df

        # Rank recipes
        if self.ranking_strategy == 'rating':
            retrieved_df = retrieved_df.sort_values(by='average_rating', ascending=False)

        # Filter recipes (if applicable)
        if self.return_meal_plan and 'meal_types' in retrieved_df.columns and not retrieved_df.empty:
            # get the first k recipes for each meal type
            if self.k is not None:
                retrieved_df = self._get_first_k_per_meal_type(
                    retrieved_df, meal_priority=['breakfast', 'lunch', 'dinner', 'snack'])
        else:  # return recipes only
            if self.k is not None:
                retrieved_df = retrieved_df.head(self.k)

        # Update metadata
        metadata[LLM_DF_INPUT_KEY] = retrieved_df

        # Prepare input for the LLM
        context = self.prepare_df_for_llm(retrieved_df)

        # Generate response
        answer = self.llm.predict(self.prompt_template, query=query, context=context)

        # Get full recipes if required
        if self.return_full_recipes:
            answer = self._get_full_recipe_str(response=answer, retrieved_df=retrieved_df)

        return Response(response=answer, metadata=metadata)

    async def _aquery(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        """
        Handles asynchronous queries from the QueryEngine interface.

        :param QueryBundle query_bundle: Query bundle from the LlamaIndex framework.
        :returns: Response object containing the answer and any relevant metadata.
        """
        query = query_bundle.query_str
        metadata = {'query': query}

        # Retrieve recipes asynchronously
        retrieved_df = await self.retriever.aretrieve(query)

        # Update metadata
        if self.debug:
            metadata[RETRIEVED_DF_KEY] = retrieved_df

        # Rank recipes
        if self.ranking_strategy == 'rating':
            retrieved_df = retrieved_df.sort_values(by='average_rating', ascending=False)

        # Filter recipes (if applicable)
        if self.return_meal_plan and 'meal_types' in retrieved_df.columns and not retrieved_df.empty:
            # get the first k recipes for each meal type
            if self.k is not None:
                retrieved_df = self._get_first_k_per_meal_type(
                    retrieved_df, meal_priority=['breakfast', 'lunch', 'dinner', 'snack'])
        else:  # return recipes only
            if self.k is not None:
                retrieved_df = retrieved_df.head(self.k)

        # Update metadata
        metadata[LLM_DF_INPUT_KEY] = retrieved_df

        # Prepare input for the LLM
        context = self.prepare_df_for_llm(retrieved_df)

        # Generate response asynchronously
        answer = await self.llm.apredict(self.prompt_template, query=query, context=context)

        # Get full recipes if required
        if self.return_full_recipes:
            answer = self._get_full_recipe_str(response=answer, retrieved_df=retrieved_df)

        return Response(response=answer, metadata=metadata)

    def _get_first_k_per_meal_type(self, retrieved_df, meal_priority=['breakfast', 'lunch', 'dinner', 'snack']):
        # Convert meal_types to a list if it's not already
        def normalize_meal_types(x):
            if isinstance(x, str):
                return [m.strip() for m in x.split(',') if m.strip()]
            elif isinstance(x, list):
                return [m.strip() for m in x if isinstance(m, str) and m.strip()]
            else:
                return []
        retrieved_df['meal_types'] = retrieved_df['meal_types'].apply(normalize_meal_types)

        # Classify rows
        retrieved_df['known_meal_types'] = retrieved_df['meal_types'].apply(
            lambda ms: [m for m in ms if m in meal_priority]
        )

        # Separate known and unknown meal types
        known_mask = retrieved_df['known_meal_types'].astype(bool)
        known = retrieved_df[known_mask].copy()
        unknown = retrieved_df[~known_mask].copy()

        # Explode the meal_type list into separate rows
        known = known.explode('known_meal_types')

        # Meal type order of priority
        selected = []
        used_ids = set()

        for meal in meal_priority:
            group = known[(known['known_meal_types'] == meal) & (~known['recipe_id'].isin(used_ids))]
            top_k = group.head(self.k).copy() # Take top k
            top_k['chosen_meal_type'] = meal

            # Fill with unknowns if needed
            shortfall = self.k - len(top_k)
            if shortfall > 0:
                filler = unknown[~unknown['recipe_id'].isin(used_ids)].head(shortfall).copy()
                filler['chosen_meal_type'] = meal
                top_k = pd.concat([top_k, filler])

            # Add to selected
            selected.append(top_k)
            used_ids.update(top_k['recipe_id'])

        # Concatenate all selected recipes
        retrieved_df = pd.concat(selected, ignore_index=True)
        retrieved_df['meal_types'] = retrieved_df['chosen_meal_type']
        retrieved_df.drop(columns=['known_meal_types', 'chosen_meal_type'], inplace=True)
        return retrieved_df

    def _get_prompt_modules(self) -> PromptMixinType:
        """
        Returns prompt modules for the QueryEngine interface.
        No additional prompt modules are used here.

        :returns: An empty PromptMixinType object.
        """
        return {}

    def _get_prompts(self) -> Dict[str, Any]:
        """Get prompts."""
        return {
            "recipe_query_engine_prompt": self.prompt_template,
        }

    def retrieve(self, str_or_query_bundle: QueryType) -> pd.DataFrame:
        return self.retriever.retrieve(str_or_query_bundle)

    async def aretrieve(self, str_or_query_bundle: QueryType) -> pd.DataFrame:
        return await self.retriever.aretrieve(str_or_query_bundle)

    def _get_full_recipe_str(self, response: str, retrieved_df) -> str:
        """
        Extracts the full recipe or meal_plan from the response.

        :param str response: Response from the LLM.
        :returns: str of all found information or the original response text if nothing new was found (i.e. responses
         title could not be matched).
        """

        def _try_get_title(obj): # it is ugly, but it works most of the time
            if isinstance(obj, dict):
                new_obj = obj.get('title', obj)
            else:
                new_obj = obj

            if isinstance(new_obj, dict):
                if not new_obj:
                    new_obj = obj
                else:
                    new_obj = str(new_obj)

            return str(new_obj)

        # Try to extract the json from the response (output between  ```json and ```).
        try:
            response_json_str = response.split('```json')[1].split('```')[0].strip()

            try:
                before_str = response.split('```json')[0].strip()
            except Exception:
                before_str = ""
            try:
                after_str = response.split('```json')[1].split('```', 1)[1].strip()
            except Exception:
                after_str = ""
        except Exception:
            try:
                response_json_str = response.split('```', 1)[1].split('```')[0].strip()
                try:
                    before_str = response.split('```')[0].strip()
                except Exception:
                    before_str = ""
                try:
                    after_str = response.split('```, 1')[1].split('```', 1)[1].strip()
                except Exception:
                    after_str = ""
            except Exception:
                response_json_str = response
                before_str = ""
                after_str = ""

        #  Try to parse the meal type JSON
        try:
            response_json = try_parse(response_json_str)

            if response_json:
                if isinstance(response_json, list):
                    recipes = response_json
                elif isinstance(response_json, str):
                    recipes = [response_json]
                elif isinstance(response_json, dict):
                    recipes = response_json
                else:
                    recipes = []
            else:
                recipes = []
        except SyntaxError:
            recipes = []

        # Get a list of recipe titles from the response
        if isinstance(recipes, list):
            recipe_titles = {'recipes': [_try_get_title(recipe) for recipe in recipes]}
        elif isinstance(recipes, dict):
            # Check for single recipe dict or meal plan dict
            meal_plan_indication = [key for key in recipes.keys() if key.lower() in
                                    ['breakfast', 'lunch', 'dinner', 'snack']]
            if meal_plan_indication:
                # If it's a meal plan, extract titles for each meal type
                recipe_titles = {
                    meal_type: [_try_get_title(r) for r in recipes] if isinstance(recipes, list)
                    else [_try_get_title(recipes)]
                    for meal_type, recipes in recipes.items()
                }
            else:
                # If it's a single recipe, extract the title
                recipe_titles = {'recipes': [_try_get_title(recipes)]}
        else:
            recipe_titles = {}

        # Get the recipes from the retrieved DataFrame if available, else keep the original response text
        if retrieved_df is not None and len(retrieved_df) > 0:
            # Filter the retrieved DataFrame to get the recipes that match the titles
            all_titles = set([title for titles in recipe_titles.values() for title in titles])
            matching_recipes = retrieved_df[retrieved_df['title'].isin(all_titles)]
            un_resolved_recipes = all_titles - set(matching_recipes['title'].tolist())

            if len(matching_recipes) > 0:
                # Serialize the matching recipes to a string
                recipe_docs = [recipe_doc.text for recipe_doc in
                               hummus_docify.recipes_to_docs(matching_recipes, verbose=False)]

                # Create dict recipe title --> Document
                recipe_doc_dict = {recipe['title']: {'recipe_doc': recipe_doc.replace('\nType: recipe', ''),
                                                     'meal_type': recipe.get('meal_types', '')}
                                   for recipe, recipe_doc in zip(matching_recipes.to_dict('records'), recipe_docs)}

                # Create a new response with the full recipes
                full_recipes_str = "Recommendation:\n"
                for recipe_title, recipe_data in recipe_doc_dict.items():
                    meal_type = recipe_data['meal_type']
                    recipe_doc = recipe_data['recipe_doc']

                    # Update the result
                    if meal_type:
                        full_recipes_str+= f"\nFor {meal_type}:\n{recipe_doc}\n"
                    else:
                        full_recipes_str += f"\n{recipe_doc}\n"

                # If there are unresolved recipes, add them to the response
                if un_resolved_recipes:
                    full_recipes_str += f"\n\nMore recommended recipes:\n{', '.join(un_resolved_recipes)}"
            else:
                # If no matching recipes were found, return the original response text
                full_recipes_str = response
        else:
            # If no matching recipes were found, return the original response text
            full_recipes_str = response

        # Add the before and after strings to the full recipes string
        if before_str:
            full_recipes_str = f"{before_str}\n\n{full_recipes_str}"
        if after_str:
            full_recipes_str = f"{full_recipes_str}\n\n{after_str}"

        return full_recipes_str


def get_llm_df_input(response: Response) -> pd.DataFrame | str | Dict[str, pd.DataFrame]:
    print('The following recipes were used as input for the llm:')
    return response.metadata.get(LLM_DF_INPUT_KEY, 'None.')


def get_retrieved_df(response: Response) -> pd.DataFrame | str | Dict[str, pd.DataFrame]:
    print('The following recipes were retrieved:')
    return response.metadata.get(RETRIEVED_DF_KEY, 'None or debug mode not enabled.')

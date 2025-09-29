import pandas as pd
from llama_index.core import Response
from llama_index.core.workflow import Context
from pydantic import Field
from pydantic.fields import FieldInfo
from typing import Dict, Any

from src.food_rag.query_engines.rag import RecipeRAGQueryEngine, LLM_DF_INPUT_KEY
from src.food_rag.retriever.df_retriever_base import DataFrameRetrieverBase
from src.food_rag.tools.tool_base import RecipeToolBase
from src.food_rag.tools.util_tools import JsonExtractorTool
from src.food_rag.utils.recipe_context import RecipeState
from src.utils import hummus_util


class RecipeRecommendationTool(RecipeToolBase):
    """
    A tool that generates recipe recommendations.
    """

    TOOL_NAME = "recipe_recommendation_tool"

    def __init__(self, recipe_query_engine: RecipeRAGQueryEngine, verbose=True):
        super().__init__(fn=self.recommend_recipe, tool_name=self.TOOL_NAME, verbose=verbose)

        self.recipe_query_engine = recipe_query_engine
        self.json_extractor_tool = JsonExtractorTool(llm=recipe_query_engine.llm, verbose=verbose)

    def recommend_recipe(self, ctx: Context,
                         question: str = Field(description="The user question asking for a recommendation.")) \
            -> Dict[str, Any]:
        """Recommends a single recipe based on the user question."""
        if isinstance(question, FieldInfo):
            raise ValueError("The 'question' parameter must be provided as a string.")

        current_state = RecipeState.from_context(ctx)

        recipe_response: Response = self.recipe_query_engine.query(question)
        retrieved_recipes = recipe_response.metadata.get(LLM_DF_INPUT_KEY, pd.DataFrame())

        # Update the llm usage count
        c = self.recipe_query_engine.llm_calls
        current_state.increment_llm_counter(parsing_inc=0, sequential_inc=c, total_inc=c)

        # Try to extract the json from the response (output between  ```json and ```).
        try:
            response_json_str = recipe_response.response.split('```json')[1].split('```')[0].strip()
        except Exception:
            response_json_str = self.json_extractor_tool.extract_json(ctx=ctx, text=recipe_response.response)

            if response_json_str == "":
                # If no JSON was found, use the original response
                response_json_str = recipe_response.response

        # Try to parse the JSON
        try:
            response_json = hummus_util.try_parse(response_json_str)

            if 'title' in response_json and response_json["title"] in retrieved_recipes["title"].values:
                recommended_recipe_df = retrieved_recipes[retrieved_recipes["title"] == response_json["title"]]
            else:
                recommended_recipe_df = retrieved_recipes
        except SyntaxError:
            recommended_recipe_df = retrieved_recipes

        # Get recipe details
        recommended_recipe_dict = (recommended_recipe_df.head(1).to_dict(orient="records")[0]
                                   if not recommended_recipe_df.empty else {})

        # Update history
        if recommended_recipe_dict:
            current_state.update_history_item(f"Recommended the recipe '{recommended_recipe_dict['title']}' for the"
                                              f" question '{question}'.")
        else:
            current_state.update_history_item(f"Tried to recommend a recipe for the question '{question}', but none "
                                              f"fitting one was found.")

        current_state.save_to_context()

        return recommended_recipe_dict


class RecipeRetrievalTool(RecipeToolBase):
    """
    A tool that retrieves a recipe from a query.
    """

    TOOL_NAME = "recipe_retrieval_tool"

    def __init__(self, recipe_retriever: DataFrameRetrieverBase, llm_calls=0, verbose=True):
        super().__init__(fn=self.retrieve_recipe, tool_name=self.TOOL_NAME, verbose=verbose)
        self.llm_calls = llm_calls
        self.recipe_retriever = recipe_retriever

    def retrieve_recipe(self, ctx: Context,
                        title: str = Field(description="The title of the recipe to retrieve.")) \
            -> Dict[str, Any]:
        """Retrieves information for a recipe title. Use this tool to gather all available information about this
        recipe."""
        if isinstance(title, FieldInfo):
            raise ValueError("The 'title' parameter must be provided as a string.")

        current_state = RecipeState.from_context(ctx)

        query = f"title: {title}"
        recipe_df: pd.DataFrame = self.recipe_retriever.retrieve(query)

        # Update the llm usage count
        current_state.increment_llm_counter(parsing_inc=0, sequential_inc=self.llm_calls, total_inc=self.llm_calls)
        current_state.save_to_context()

        recommended_recipe_dict = (recipe_df.head(1).to_dict(orient="records")[0] if not recipe_df.empty else {})
        return recommended_recipe_dict

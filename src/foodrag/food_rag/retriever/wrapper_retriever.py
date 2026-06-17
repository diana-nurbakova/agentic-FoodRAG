import pandas as pd
from llama_index.core import QueryBundle
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.schema import NodeWithScore, QueryType
from typing import List

from foodrag.food_rag.retriever.df_retriever_base import DataFrameRetrieverBase


class WrapperNodeRecipeRetriever(BaseRetriever, DataFrameRetrieverBase):
    def __init__(self, base_retriever: BaseRetriever, recipes_df: pd.DataFrame, recipe_id_field: str = "recipe_id"):
        """
        Wrapper retriever returning a DataFrame instead of nodes.

        :param BaseRetriever base_retriever: The base retriever that returns a list of nodes.
        :param pd.DataFrame recipes_df: The recipes DataFrame.
        :param str recipe_id_field: The field in the node's metadata that contains the recipe ID.
        """
        super().__init__()
        self.base_retriever = base_retriever
        self.recipes_df = recipes_df
        self.recipe_id_field = recipe_id_field

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """
        Retrieves results.

        :param QueryBundle query_bundle: The query bundle.
        :returns: The retrieved nodes.
        :rtype: List[NodeWithScore]
        """
        return self.base_retriever._retrieve(query_bundle)

    def retrieve(self, str_or_query_bundle: QueryType) -> pd.DataFrame:
        """
        Retrieves results and converts to DataFrame.

        :param QueryType str_or_query_bundle: Query string or QueryBundle (which optionally contains the embedding).
        :returns: The filtered recipes DataFrame.
        :rtype: pd.DataFrame
        """
        nodes: List[NodeWithScore] = self.base_retriever.retrieve(str_or_query_bundle)
        recipe_ids = self._extract_recipe_ids(nodes)
        return self._get_df_from_ids(recipe_ids)

    async def aretrieve(self, str_or_query_bundle: QueryType) -> pd.DataFrame:
        """
        Asynchronously retrieves results and converts to DataFrame.

        :param QueryType str_or_query_bundle: Query string or QueryBundle (which optionally contains the embedding).
        :returns: The filtered recipes DataFrame.
        :rtype: pd.DataFrame
        """
        nodes: List[NodeWithScore] = await self.base_retriever.aretrieve(str_or_query_bundle)
        recipe_ids = self._extract_recipe_ids(nodes)
        return self._get_df_from_ids(recipe_ids)

    def _extract_recipe_ids(self, nodes: List[NodeWithScore]) -> List[str]:
        """
        Extracts unique recipe IDs from nodes.

        :param List[NodeWithScore] nodes: The retrieved nodes.
        :returns: The list of unique recipe IDs.
        :rtype: List[str]
        """
        recipe_ids = [node.metadata.get(self.recipe_id_field) for node in nodes if
                      self.recipe_id_field in node.metadata]
        return list(dict.fromkeys(recipe_ids))  # Remove duplicates while preserving order.

    def _get_df_from_ids(self, recipe_ids: List[str]) -> pd.DataFrame:
        """
        Filters the recipes DataFrame based on recipe IDs.

        :param List[str] recipe_ids: The list of recipe IDs.
        :returns: The filtered DataFrame.
        :rtype: pd.DataFrame
        """
        filtered_recipes = self.recipes_df[self.recipes_df['recipe_id'].isin(recipe_ids)]
        return filtered_recipes.set_index('recipe_id').loc[recipe_ids].reset_index()  # Preserve order.

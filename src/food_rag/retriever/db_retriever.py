import pandas as pd
from llama_index.core.indices.vector_store import VectorIndexRetriever
from llama_index.core.schema import NodeWithScore, QueryType
from typing import List

from src.food_rag.retriever.df_retriever_base import DataFrameRetrieverBase


class RecipeRetriever(VectorIndexRetriever, DataFrameRetrieverBase):
    def __init__(self, index, recipes_df, vector_store_query_mode="default", similarity_top_k=10, split_node_ratio=2):
        """
        Modified Retriever to return documents directly.

        :param index: The vector store index instance.
        :param pd.DataFrame recipes_df: The recipes dataframe.
        :param str vector_store_query_mode: Query mode, either 'default', 'sparse', 'hybrid', or 'text-search'.
        :param int similarity_top_k: Number of top results to retrieve.
        :param float split_node_ratio: Ratio of how many nodes were created (on average) after a document is split.
        """
        super().__init__(index=index,
                         vector_store_query_mode=vector_store_query_mode,
                         similarity_top_k=similarity_top_k * split_node_ratio)
        self.document_top_k = similarity_top_k
        self.recipes_df = recipes_df

    def retrieve(self, str_or_query_bundle: QueryType) -> pd.DataFrame:
        """
        Overrides retrieve to return documents.

        :param QueryType str_or_query_bundle: Query string or QueryBundle (which optionally contains the embedding).
        :returns: The retrieved documents.
        :rtype: pd.DataFrame
        """
        results: List[NodeWithScore] = super().retrieve(str_or_query_bundle)

        unique_recipe_ids = self._get_unique_recipe_ids(results)
        recipes = self._get_df_from_ids(unique_recipe_ids)

        return recipes

    async def aretrieve(self, str_or_query_bundle: QueryType) -> pd.DataFrame:
        """
        Overrides aretrieve to return documents.

        :param QueryType str_or_query_bundle: Query string or QueryBundle (which optionally contains the embedding).
        :returns: The retrieved documents.
        :rtype: pd.DataFrame
        """
        results: List[NodeWithScore] = await super().aretrieve(str_or_query_bundle)

        unique_recipe_ids = self._get_unique_recipe_ids(results)
        recipes = self._get_df_from_ids(unique_recipe_ids)

        return recipes

    def _get_unique_recipe_ids(self, results):
        # Extract and resolve documents
        recipe_ids = [result.metadata['recipe_id'] for result in results]

        # Make sure that the documents are unique and are less than or equal to similarity_top_k
        unique_recipe_ids = list(dict.fromkeys(recipe_ids))
        unique_recipe_ids = unique_recipe_ids[:self.document_top_k]

        return unique_recipe_ids

    def _get_df_from_ids(self, unique_recipe_ids) -> pd.DataFrame:
        recipes = self.recipes_df[self.recipes_df['recipe_id'].isin(unique_recipe_ids)]
        recipes = recipes.set_index('recipe_id').loc[unique_recipe_ids].reset_index()  # Preserve order

        return recipes

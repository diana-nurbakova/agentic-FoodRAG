import asyncio
import pandas as pd
from llama_index.core import QueryBundle
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.schema import NodeWithScore, QueryType
from tenacity import RetryError
from typing import Dict, List, Tuple, Optional

from src.food_rag.retriever.df_retriever_base import DataFrameRetrieverBase


class DataFrameFusionRetriever(QueryFusionRetriever, DataFrameRetrieverBase):
    """A retriever that fuses the results of multiple retrievers to return a DataFrame instead of a List of nodes.
    Important Note: Only works with retrievers which _retrieve method returns a List of NodeWithScore objects."""

    def __init__(self, recipes_df: pd.DataFrame, recipe_id_field: str = "recipe_id", **kwargs):
        """
        Fusion retriever returning a DataFrame instead of nodes.

        :param pd.DataFrame recipes_df: The recipes DataFrame.
        :param str recipe_id_field: The field in the node's metadata that contains the recipe ID.
        :param kwargs: Additional arguments for initialization.
        """
        self.recipes_df = recipes_df
        self.recipe_id_field = recipe_id_field
        super().__init__(**kwargs)

    def _run_sync_queries(self, queries: List[QueryBundle]) -> Dict[Tuple[str, int], List[NodeWithScore]]:
        results = {}
        for query in queries:
            for i, retriever in enumerate(self._retrievers):
                try:
                    results[(query.query_str, i)] = retriever._retrieve(query)
                except (Exception, RetryError):
                    results[(query.query_str, i)] = []
        return results

    async def _run_async_queries(self, queries: List[QueryBundle]) -> Dict[Tuple[str, int], List[NodeWithScore]]:
        tasks, task_queries = [], []

        def try_aretrieve(query):
            try:
                return retriever._aretrieve(query)
            except (Exception, RetryError):
                return []

        for query in queries:
            for i, retriever in enumerate(self._retrievers):
                tasks.append(try_aretrieve(query))
                task_queries.append((query.query_str, i))

        task_results = await asyncio.gather(*tasks, return_exceptions=True)

        results = {}
        for query_tuple, query_result in zip(task_queries, task_results):
            if isinstance(query_result, Exception):
                if self._verbose:
                    print(f"Error in task {query_tuple}: {query_result}")
                results[query_tuple] = []
            else:
                results[query_tuple] = query_result

        return results

    def retrieve(self, str_or_query_bundle: QueryType) -> pd.DataFrame:
        """
        Retrieves results and converts to DataFrame.

        :param QueryType str_or_query_bundle: Query string or QueryBundle (which optionally contains the embedding).
        :returns: The filtered recipes DataFrame.
        :rtype: pd.DataFrame
        """
        nodes: List[NodeWithScore] = super().retrieve(str_or_query_bundle)
        recipe_ids = self._extract_recipe_ids(nodes)
        return self._get_df_from_ids(recipe_ids)

    async def aretrieve(self, str_or_query_bundle: QueryType) -> pd.DataFrame:
        """
        Asynchronously retrieves results and converts to DataFrame.

        :param QueryType str_or_query_bundle: Query string or QueryBundle (which optionally contains the embedding).
        :returns: The filtered recipes DataFrame.
        :rtype: pd.DataFrame
        """
        nodes: List[NodeWithScore] = await super().aretrieve(str_or_query_bundle)
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


class IntersectionRetriever(BaseRetriever, DataFrameRetrieverBase):
    def __init__(self, retriever1: DataFrameRetrieverBase, retriever2: DataFrameRetrieverBase, verbose: bool = False):
        """
        Initializes the retriever with two dataframe-based retrievers.

        :param DataFrameRetrieverBase retriever1: The first retriever.
        :param DataFrameRetrieverBase retriever2: The second retriever.
        :param bool verbose: Flag to enable verbose output (default is False).
        """
        self.retriever1 = retriever1
        self.retriever2 = retriever2
        self.verbose = verbose
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        raise NotImplementedError

    def retrieve(self, str_or_query_bundle: QueryType) -> pd.DataFrame:
        """
        Performs the retrieval and returns only rows with common `recipe_id`s in both results.

        :param QueryType str_or_query_bundle: Query string or QueryBundle (which optionally contains the embedding).
        :returns: The retrieved DataFrame with common `recipe_id`s.
        :rtype: pd.DataFrame
        """
        # Retrieve results from both retrievers
        df1 = self.retriever1.retrieve(str_or_query_bundle)
        df2 = self.retriever2.retrieve(str_or_query_bundle)

        if self.verbose:
            print(f"Retriever 1 returned {len(df1)} rows")
            print(f"Retriever 2 returned {len(df2)} rows")

        # Find intersection of rows based on `recipe_id`
        common_ids = set(df1["recipe_id"]).intersection(df2["recipe_id"])
        intersected_df = df1[df1["recipe_id"].isin(common_ids)]

        if self.verbose:
            print(f"Number of common rows: {len(intersected_df)}")

        return intersected_df

    async def aretrieve(self, str_or_query_bundle: QueryType) -> pd.DataFrame:
        """
        Asynchronous version of `_retrieve`.

        :param QueryType str_or_query_bundle: Query string or QueryBundle (which optionally contains the embedding).
        :returns: The retrieved DataFrame with common `recipe_id`s.
        :rtype: pd.DataFrame
        """
        # Retrieve results from both retrievers asynchronously
        df1 = await self.retriever1.aretrieve(str_or_query_bundle)
        df2 = await self.retriever2.aretrieve(str_or_query_bundle)

        if self.verbose:
            print(f"Retriever 1 returned {len(df1)} rows")
            print(f"Retriever 2 returned {len(df2)} rows")

        # Find intersection of rows based on `recipe_id`
        common_ids = set(df1["recipe_id"]).intersection(df2["recipe_id"])
        intersected_df = df1[df1["recipe_id"].isin(common_ids)]

        if self.verbose:
            print(f"Number of common rows: {len(intersected_df)}")

        return intersected_df


class MajorityIntersectionRetriever(BaseRetriever, DataFrameRetrieverBase):
    def __init__(self, retrievers: List[DataFrameRetrieverBase], vote_weights: Optional[Dict[int, float]] = None,
                 verbose: bool = False):
        """
        Initializes the retriever with multiple retrievers and optional vote weights.
        :param List[DataFrameRetrieverBase] retrievers: List of BaseRetriever instances.
        :param vote_weights: A dictionary mapping retriever indices to vote weights (0 to 1).
        :param verbose: Whether to print debug information.
        """
        self.retrievers = retrievers
        self.vote_weights = vote_weights or {i: 1.0 for i in range(len(retrievers))}
        self.verbose = verbose
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        raise NotImplementedError

    def retrieve(self, str_or_query_bundle: QueryType) -> pd.DataFrame:
        """
        Performs the retrieval and returns rows that appear in a majority of retrievers.

        :param QueryType str_or_query_bundle: Query string or QueryBundle (which optionally contains the embedding).
        :returns: The filtered DataFrame with rows that appear in a majority of retrievers.
        :rtype: pd.DataFrame
        """
        # Retrieve results from all retrievers
        dfs = [retriever.retrieve(str_or_query_bundle) for retriever in self.retrievers]

        if self.verbose:
            print(f"Retrieved dataframes from {len(dfs)} retrievers")

        # Combine results and count occurrences of each `recipe_id`
        all_ids = []
        weights = []

        for i, df in enumerate(dfs):
            weight = self.vote_weights.get(i, 1.0)
            all_ids.extend(df["recipe_id"])
            weights.extend([weight] * len(df))

        id_counts = pd.DataFrame({"recipe_id": all_ids, "weight": weights})
        id_counts = id_counts.groupby("recipe_id")["weight"].sum().reset_index()

        # Determine the threshold for majority
        total_weight = sum(self.vote_weights.values())
        threshold = total_weight / 2

        if self.verbose:
            print(f"Majority threshold: {threshold}")

        # Filter `recipe_id`s that meet the majority threshold
        majority_ids = id_counts[id_counts["weight"] >= threshold]["recipe_id"]

        # Merge rows corresponding to the majority `recipe_id`s from the first retriever
        result_df = dfs[0][dfs[0]["recipe_id"].isin(majority_ids)]

        if self.verbose:
            print(f"Number of rows in majority result: {len(result_df)}")

        return result_df

    async def aretrieve(self, str_or_query_bundle: QueryType) -> pd.DataFrame:
        """
        Asynchronous version of `retrieve`.

        :param QueryType str_or_query_bundle: Query string or QueryBundle (which optionally contains the embedding).
        :returns: The filtered DataFrame with rows that appear in a majority of retrievers.
        :rtype: pd.DataFrame
        """
        # Retrieve results asynchronously from all retrievers
        dfs = await asyncio.gather(
            *(retriever.aretrieve(str_or_query_bundle) for retriever in self.retrievers)
        )

        if self.verbose:
            print(f"Retrieved dataframes from {len(dfs)} retrievers")

        # Combine results and count occurrences of each `recipe_id`
        all_ids = []
        weights = []

        for i, df in enumerate(dfs):
            weight = self.vote_weights.get(i, 1.0)
            all_ids.extend(df["recipe_id"])
            weights.extend([weight] * len(df))

        id_counts = pd.DataFrame({"recipe_id": all_ids, "weight": weights})
        id_counts = id_counts.groupby("recipe_id")["weight"].sum().reset_index()

        # Determine the threshold for majority
        total_weight = sum(self.vote_weights.values())
        threshold = total_weight / 2

        if self.verbose:
            print(f"Majority threshold: {threshold}")

        # Filter `recipe_id`s that meet the majority threshold
        majority_ids = id_counts[id_counts["weight"] >= threshold]["recipe_id"]

        # Merge rows corresponding to the majority `recipe_id`s from the first retriever
        result_df = dfs[0][dfs[0]["recipe_id"].isin(majority_ids)]

        if self.verbose:
            print(f"Number of rows in majority result: {len(result_df)}")

        return result_df

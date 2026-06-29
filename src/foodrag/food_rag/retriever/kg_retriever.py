import logging
import pandas as pd
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.llms.llm import LLM
from llama_index.core.prompts.base import (
    BasePromptTemplate,
    PromptTemplate,
    PromptType,
)
from llama_index.core import QueryBundle
from llama_index.core.query_engine import KnowledgeGraphQueryEngine
from llama_index.core.response_synthesizers import (
    BaseSynthesizer,
)
from llama_index.core.schema import NodeWithScore, QueryType
from llama_index.core.storage.storage_context import StorageContext
from typing import Any, List, Optional

from foodrag.food_rag.retriever.df_retriever_base import DataFrameRetrieverBase

logger = logging.getLogger(__name__)


DEFAULT_GRAPH_QUERY_SYNTHESIS_PROMPT_TEXT = """Task: Generate a Cypher query from natural language input.

Instructions:
- Use only the provided relationship types, tags, and properties from the schema.
- Do not alter any relationship types, tags, or properties, including capitalization.
- Return only the Cypher query string (that can be directly executed on the graph).
- Do not add explanations, notes, apologies, or anything but the query in your responses.
- Do not enclose the result by ```cypher or ```.
- Always return the list of recipe_urls as answer.

Schema:
- Nodes: {node_types},
- Relations: {relation_types},
- Node Property Names: {node_features},
- Relation Property Names: {relation_features},

Notes:
- Scores (who score, fsa score, nutri score) are normalized between 0 and 1, with 1 being the highest.
- Default score is the nutri score.
- Recipes are unhealthy with a score < 0.3, neutral with a score ≥ 0.3, and healthy with a score ≥ 0.5.
- If there are nutrition requirements (e.g. amount of sugar), always use the properties given for the recipes themselves.
- If nutrition requirements are given as a range, use two conditions. For example, if sugar should be between 10 and 100, use r.Recipe.sugars > 10 AND r.Recipe.sugars < 100.

Question: {query_str}

Answer (Cypher Query):"""

DEFAULT_GRAPH_QUERY_SYNTHESIS_PROMPT = PromptTemplate(
    DEFAULT_GRAPH_QUERY_SYNTHESIS_PROMPT_TEXT,
    prompt_type=PromptType.TEXT_TO_GRAPH_QUERY,
)


class KGQueryEngine(KnowledgeGraphQueryEngine, BaseRetriever, DataFrameRetrieverBase):
    def __init__(
            self,
            recipes_df,
            llm: Optional[LLM] = None,
            similarity_top_k=10,
            storage_context: Optional[StorageContext] = None,
            graph_query_synthesis_prompt: Optional[BasePromptTemplate] = None,
            graph_response_answer_prompt: Optional[BasePromptTemplate] = None,
            refresh_schema: bool = False,
            verbose: bool = False,
            response_synthesizer: Optional[BaseSynthesizer] = None,
            **kwargs: Any,
    ):
        """
        Knowledge graph query engine.

        Query engine to call a knowledge graph.

        :param pd.DataFrame recipes_df: The recipes dataframe.
        :param Optional[LLM] llm: An LLM object (optional).
        :param int similarity_top_k: The number of similar recipes to retrieve (default: 10).
        :param Optional[StorageContext] storage_context: A storage context to use (optional).
        :param Optional[BasePromptTemplate] graph_query_synthesis_prompt: A prompt template for query synthesis (optional).
        :param Optional[BasePromptTemplate] graph_response_answer_prompt: A prompt template for response synthesis (optional).
        :param bool refresh_schema: Whether to refresh the schema.
        :param bool verbose: Whether to print intermediate results.
        :param Optional[BaseSynthesizer] response_synthesizer: A BaseSynthesizer object (optional).
        :param kwargs: Additional keyword arguments.
        """
        # Get graph store query synthesis prompt or use default --> Must define by ourselves as llama index has a bug
        # here: https://github.com/run-llama/llama_index/issues/11034
        _graph_query_synthesis_prompt = (
                graph_query_synthesis_prompt or DEFAULT_GRAPH_QUERY_SYNTHESIS_PROMPT
        )

        temp_kwargs = {'node_types': kwargs['node_types'], 'node_features': kwargs['node_features'],
                       'relation_types': kwargs['relation_types'], 'relation_features': kwargs['relation_features']}
        _graph_query_synthesis_prompt = _graph_query_synthesis_prompt.partial_format(**temp_kwargs)

        # Call the original __init__ method with the modified verbose parameter
        super().__init__(llm=llm, storage_context=storage_context,
                         graph_query_synthesis_prompt=_graph_query_synthesis_prompt,
                         graph_response_answer_prompt=graph_response_answer_prompt, refresh_schema=refresh_schema,
                         verbose=verbose, response_synthesizer=response_synthesizer, **kwargs)

        self.recipes_df = recipes_df
        self.similarity_top_k = similarity_top_k

    def retrieve(self, str_or_query_bundle: QueryType) -> pd.DataFrame:
        """
        Overrides retrieve to return a DataFrame.

        :param QueryType str_or_query_bundle: Query string or QueryBundle (which optionally contains the embedding).
        :returns: Retrieved results as a DataFrame.
        :rtype: pd.DataFrame
        """
        try:
            # KnowledgeGraphQueryEngine is a *query* engine: its public retrieve()
            # raises NotImplementedError ("use query directly"). Call the graph
            # retrieval step directly — _retrieve runs the Text-to-Cypher query and
            # returns a node whose metadata carries 'graph_store_response'.
            query_bundle = (QueryBundle(str_or_query_bundle)
                            if isinstance(str_or_query_bundle, str) else str_or_query_bundle)
            results: List[NodeWithScore] = self._retrieve(query_bundle)
        except Exception as e:
            if self._verbose:
                logger.error(f"Error in retrieving query: {str(str_or_query_bundle)}\n{e}")
            return self.recipes_df.iloc[0:0]  # Return an empty recipe df

        result_df = self._get_recipe_df(results)

        if self.similarity_top_k is not None:
            result_df = result_df.head(self.similarity_top_k)

        return result_df

    async def aretrieve(self, str_or_query_bundle: QueryType) -> pd.DataFrame:
        """
        Overrides aretrieve to return a DataFrame.

        :param QueryType str_or_query_bundle: Query string or QueryBundle (which optionally contains the embedding).
        :returns: Retrieved results as a DataFrame.
        :rtype: pd.DataFrame
        """
        try:
            query_bundle = (QueryBundle(str_or_query_bundle)
                            if isinstance(str_or_query_bundle, str) else str_or_query_bundle)
            results: List[NodeWithScore] = await self._aretrieve(query_bundle)
        except Exception as e:
            if self._verbose:
                logger.error(f"Error in retrieving query: {str(str_or_query_bundle)}\n{e}")
            return self.recipes_df.iloc[0:0]  # Return an empty recipe df

        result_df = self._get_recipe_df(results)

        if self.similarity_top_k is not None:
            result_df = result_df.head(self.similarity_top_k)

        return result_df

    def _get_recipe_df(self, results) -> pd.DataFrame:
        # Extract the graph-store response. Depending on the graph store / driver
        # version it can be a dict, a list of records, or a list of values, so
        # flatten the whole structure to scalar values rather than assuming a dict.
        try:
            graph_response = results[0].metadata['graph_store_response']
        except (KeyError, IndexError, TypeError):
            return self.recipes_df.iloc[0:0]

        # The generated Cypher may RETURN recipe_urls (str) or recipe_ids (int/str,
        # e.g. ``RETURN r.recipe_id``); resolve either form to a recipe_url.
        flat = self._flatten_and_deduplicate(graph_response)
        url_set = set(self.recipes_df['recipe_url'])
        id_to_url = dict(zip(self.recipes_df['recipe_id'].astype(str),
                             self.recipes_df['recipe_url']))
        ordered_urls = []
        for v in flat:
            if v in url_set:
                ordered_urls.append(v)
            else:
                u = id_to_url.get(str(v))
                if u is not None:
                    ordered_urls.append(u)
        if not ordered_urls:
            return self.recipes_df.iloc[0:0]

        # Map URLs → recipe rows, preserving the graph's result order (matters for MRR).
        indexed = self.recipes_df.set_index('recipe_url')
        indexed = indexed[~indexed.index.duplicated(keep='first')]
        keep = [u for u in ordered_urls if u in indexed.index]
        if not keep:
            return self.recipes_df.iloc[0:0]
        return indexed.loc[keep].reset_index()

    def _flatten_and_deduplicate(self, nested_structure):
        seen = set()
        result = []

        def flatten(structure):
            if isinstance(structure, dict):
                # If it's a dict, iterate through its values
                for value in structure.values():
                    flatten(value)
            elif isinstance(structure, list):
                # If it's a list, iterate through its elements
                for item in structure:
                    flatten(item)
            else:
                # If it's neither a list nor a dict, treat it as a value
                if structure not in seen:
                    seen.add(structure)
                    result.append(structure)

        flatten(nested_structure)
        return result

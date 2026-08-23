"""Naive RAG — ladder rung 3.

Standard single-shot retrieve-then-generate with *no* agentic
validation/correction/enhancement. Isolates what FoodRAG's Correction +
Enhancement agents add over vanilla RAG.

For the deterministic ladder metrics the recommendation list is the retriever's
top-k (the grounded candidate set the generator sees); the LLM-judged metrics on
the generated plan are scored by the separate judge pass, not here. Any
``DataFrameRetrieverBase`` (dense, sparse, hybrid, fusion) can be injected, so
this rung also doubles as the EXP-1 retriever wrapper expressed on the ladder.
"""
from __future__ import annotations

import pandas as pd

from foodrag.baselines.base import Baseline
from foodrag.food_rag.retriever.df_retriever_base import DataFrameRetrieverBase


class NaiveRAG(Baseline):
    """Rung 3: retrieve top-k with ``retriever``; recommend that grounded set."""

    rung = "3"
    config_id = "naive_rag"

    def __init__(self, retriever: DataFrameRetrieverBase, recipes_df: pd.DataFrame,
                 *, query_field: str = "question", recipe_id_field: str = "recipe_id"):
        self._retriever = retriever
        self._query_field = query_field
        self._recipe_id_field = recipe_id_field
        # Retrievers key results by recipe_id; the ladder/gold key by recipe_url.
        self._id_to_url = dict(zip(recipes_df[recipe_id_field], recipes_df["recipe_url"]))

    def recommend(self, qa_item: dict, k: int) -> list[str]:
        result = self._retriever.retrieve(qa_item[self._query_field])
        if result is None or len(result) == 0:
            return []
        result = result.head(k)
        if "recipe_url" in result.columns:
            return result["recipe_url"].tolist()
        return [self._id_to_url.get(rid) for rid in result[self._recipe_id_field]
                if self._id_to_url.get(rid) is not None]

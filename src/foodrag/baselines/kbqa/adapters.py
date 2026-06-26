"""Dual-use adapters that expose a :class:`KBQASystem` two ways (§5.7).

The whole point of the external-baseline design is that each KBQA system is used
in two roles, answering two different questions:

* **Use 1 — retrieval (EXP-1):** run the system natively as a retriever and score
  its ranked list on the retrieval + health + safety metrics. Here that is just a
  ladder :class:`~foodrag.baselines.base.Baseline` tagged ``EXP-1`` so it flows
  through the same :func:`~foodrag.baselines.runner.run_baseline` harness as the
  popularity/CF rungs and emits §13 records.
* **Use 2 — full-pipeline (EXP-7 rung 4):** attach FoodRAG's meal-plan assembly
  layer on top of the system's retrieved recipes. The assembly layer consumes a
  ``DataFrameRetrieverBase``, so :class:`KBQARetriever` adapts the system to that
  interface (binding the current persona, which a bare query string lacks).

Both wrap the *same* ``system.retrieve(qa_item, k)``, so Use-1 and Use-2 numbers
are attributable to the same retrieval mechanism (the §5.7 decomposition).
"""
from __future__ import annotations

from typing import Optional

import pandas as pd
from llama_index.core.schema import QueryBundle, QueryType

from foodrag.baselines.base import Baseline
from foodrag.baselines.kbqa.base import KBQASystem
from foodrag.food_rag.retriever.df_retriever_base import DataFrameRetrieverBase

ID_COL = "recipe_url"


class KBQABaseline(Baseline):
    """A KBQA system as a ladder rung (Use 1 EXP-1, or Use 2 EXP-7 rung 4).

    ``rung`` / ``config_id`` are instance attributes (not class constants) because
    the same system is the EXP-1 retriever *and* the EXP-7 rung-4 full-pipeline
    config; only the tags differ.
    """

    def __init__(self, system: KBQASystem, *, rung: str = "4",
                 config_id: Optional[str] = None):
        self.system = system
        self.rung = rung
        self.config_id = config_id or system.name

    def recommend(self, qa_item: dict, k: int) -> list[str]:
        return self.system.retrieve(qa_item, k)


class KBQARetriever(DataFrameRetrieverBase):
    """A KBQA system as a ``DataFrameRetrieverBase`` for the assembly layer (Use 2).

    The assembly layer retrieves with a query string, but the KBQA systems need
    the persona's constraints; :meth:`bind` attaches the current QA item so a
    subsequent :meth:`retrieve` walk uses the right constraints. ``retrieve`` also
    accepts a QA-item dict directly, for callers that have one.
    """

    def __init__(self, system: KBQASystem, recipes_df: pd.DataFrame, *, k: int = 10):
        self.system = system
        self.recipes_df = recipes_df.set_index(ID_COL, drop=False)
        self.k = k
        self._bound: Optional[dict] = None

    def bind(self, qa_item: dict) -> "KBQARetriever":
        self._bound = qa_item
        return self

    def _item_for(self, query: QueryType) -> dict:
        if isinstance(query, dict):
            return query
        if self._bound is not None:
            return self._bound
        text = query.query_str if isinstance(query, QueryBundle) else str(query)
        return {"question": text, "persona": {}}

    def _frame(self, urls: list[str]) -> pd.DataFrame:
        present = [u for u in urls if u in self.recipes_df.index]
        if not present:
            return self.recipes_df.iloc[0:0].reset_index(drop=True)
        return self.recipes_df.loc[present].reset_index(drop=True)

    def retrieve(self, str_or_query_bundle: QueryType) -> pd.DataFrame:
        item = self._item_for(str_or_query_bundle)
        return self._frame(self.system.retrieve(item, self.k))

    async def aretrieve(self, str_or_query_bundle: QueryType) -> pd.DataFrame:
        return self.retrieve(str_or_query_bundle)

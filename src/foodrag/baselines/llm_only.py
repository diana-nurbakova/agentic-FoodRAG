"""LLM-only (no retrieval) — ladder rung 2.

Llama 3.3 70B generates recipe recommendations directly from the persona's
question, with health/constraint instructions in the prompt but *no* HUMMUS
retrieval. This isolates the core RAG justification ("retrieval/grounding
matters"): generated titles are matched back to real recipes, and titles with no
match are hallucinations — surfaced by ``hallucination_rate`` and counted as
retrieval misses + unverifiable safety, which is exactly the failure mode this
rung is meant to expose.
"""
from __future__ import annotations

import pandas as pd
from llama_index.core import PromptTemplate

from foodrag.baselines.base import Baseline
from foodrag.utils.hummus_util import try_parse

DEFAULT_PROMPT = PromptTemplate("""\
You are a recipe recommender. Given the user's requirements, recommend the {k} \
best real recipes you know that satisfy ALL of them (dietary tags, included and \
excluded ingredients, and nutrition limits). Prefer healthier recipes.

Return ONLY a JSON list of recipe titles, e.g. ["Title One", "Title Two"].

User requirements:
{query}

Recommendations as a JSON list of titles:
""")


def _normalize(title: str) -> str:
    return " ".join(str(title).lower().split())


class LLMOnly(Baseline):
    """Rung 2: generate recipe titles from the LLM with no retrieval grounding."""

    rung = "2"
    config_id = "llm_only"

    def __init__(self, llm, recipes_df: pd.DataFrame, *, query_field: str = "question",
                 prompt: PromptTemplate = DEFAULT_PROMPT):
        self._llm = llm
        self._query_field = query_field
        self._prompt = prompt
        # Title → recipe_url, so generated titles can be grounded back to the
        # corpus; unmatched titles stay unresolved (hallucinations).
        self._title_to_url = {
            _normalize(t): u for t, u in zip(recipes_df["title"], recipes_df["recipe_url"])
        }

    def recommend(self, qa_item: dict, k: int) -> list[str]:
        prompt = self._prompt.format(k=k, query=qa_item[self._query_field])
        raw = self._llm.complete(prompt).text
        try:
            titles = try_parse(raw)
        except Exception:
            return []
        if not isinstance(titles, list):
            return []
        # Resolve to recipe_url where possible; keep the raw title otherwise so it
        # counts as an unresolved (hallucinated) recommendation downstream.
        return [self._title_to_url.get(_normalize(t), str(t)) for t in titles[:k]]

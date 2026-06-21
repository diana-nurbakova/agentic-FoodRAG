"""Baseline ladder (Evaluation Protocol §5.7).

Each rung turns a QA item into a ranked recipe list scored by the same
deterministic metrics (§4.1–4.3) and strata (§9) as FoodRAG:

* 0a ``PopularityRaw`` / 0b ``PopularityHealthAware`` — deterministic, no LLM
* 2  ``LLMOnly`` — generation without retrieval (needs an LLM)
* 3  ``NaiveRAG`` — retrieve-then-generate without agentic correction (needs a retriever)

The runner produces §13 Results-Schema records consumed by the Report tab.
Rungs 1 (item-based CF) and 4 (pFoodReq) are follow-up scope (§12.2).
"""
from foodrag.baselines.base import Baseline, answer_set_stratum, requirement_count, requirement_types
from foodrag.baselines.llm_only import LLMOnly
from foodrag.baselines.naive_rag import NaiveRAG
from foodrag.baselines.popularity import PopularityHealthAware, PopularityRaw
from foodrag.baselines.runner import aggregate, load_recipes_df, run_baseline, write_results

__all__ = [
    "Baseline",
    "PopularityRaw",
    "PopularityHealthAware",
    "LLMOnly",
    "NaiveRAG",
    "run_baseline",
    "aggregate",
    "write_results",
    "load_recipes_df",
    "answer_set_stratum",
    "requirement_count",
    "requirement_types",
]

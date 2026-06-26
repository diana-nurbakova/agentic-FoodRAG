"""Construct KBQA systems and their dual-use adapters from runtime resources.

One place that knows how to assemble a :class:`KBQASystem` from a graph client, an
optional generator LLM and optional fine-tuned checkpoints — so the runner, the
CLI and the eval notebook all build the baselines identically.
"""
from __future__ import annotations

from typing import Any, Optional

import pandas as pd

from foodrag.baselines.kbqa.adapters import KBQABaseline, KBQARetriever
from foodrag.baselines.kbqa.base import KBQAOptions, KBQASystem
from foodrag.baselines.kbqa.kg import HummusKG, KGClient, Neo4jKGClient
from foodrag.baselines.kbqa.rog import RoG, RoGPlanner
from foodrag.baselines.kbqa.tfs_kbqa import TFSKBQA, TFSParser
from foodrag.baselines.kbqa.tog import ToG

KBQA_NAMES = ("tog", "rog", "tfs_kbqa")


def build_kg(client: Optional[KGClient] = None, **kg_kwargs) -> HummusKG:
    """Wrap a :class:`KGClient` (default: Neo4j from env) in a :class:`HummusKG`."""
    return HummusKG(client or Neo4jKGClient.from_env(), **kg_kwargs)


def build_system(name: str, kg: HummusKG, *, llm: Optional[Any] = None,
                 options: Optional[KBQAOptions] = None,
                 rog_checkpoint: Optional[str] = None,
                 tfs_checkpoint: Optional[str] = None) -> KBQASystem:
    """Build one KBQA system by name, loading checkpoints where supplied."""
    options = options or KBQAOptions()
    if name == "tog":
        return ToG(kg, llm=llm, options=options)
    if name == "rog":
        planner = RoGPlanner(llm=llm, max_plans=options.beam_width,
                             checkpoint=RoGPlanner.load_checkpoint(rog_checkpoint) if rog_checkpoint else None)
        return RoG(kg, llm=llm, planner=planner, options=options)
    if name == "tfs_kbqa":
        parser = TFSParser(llm=llm,
                           checkpoint=TFSParser.load_checkpoint(tfs_checkpoint) if tfs_checkpoint else None)
        return TFSKBQA(kg, llm=llm, parser=parser, options=options)
    raise ValueError(f"Unknown KBQA system {name!r}; choose from {KBQA_NAMES}")


def build_baseline(name: str, kg: HummusKG, *, rung: str = "4",
                   config_id: Optional[str] = None, **kw) -> KBQABaseline:
    """A KBQA system as a ladder rung (Use 1 EXP-1 retriever / Use 2 EXP-7 rung 4)."""
    return KBQABaseline(build_system(name, kg, **kw), rung=rung, config_id=config_id)


def build_retriever(name: str, kg: HummusKG, recipes_df: pd.DataFrame, *, k: int = 10,
                    **kw) -> KBQARetriever:
    """A KBQA system as a ``DataFrameRetrieverBase`` for the assembly layer (Use 2)."""
    return KBQARetriever(build_system(name, kg, **kw), recipes_df, k=k)


def build_full_pipeline(name: str, kg: HummusKG, recipes_df: pd.DataFrame, *,
                        llm: Optional[Any] = None, options: Optional[KBQAOptions] = None,
                        config_id: Optional[str] = None, retrieve_k: int = 10,
                        retrieve_only: bool = True, require_explanation: bool = False,
                        rog_checkpoint: Optional[str] = None, tfs_checkpoint: Optional[str] = None,
                        **assembler_kw):
    """A KBQA system wired through FoodRAG's meal-plan assembly layer (Use 2, EXP-7
    rung 4): returns a ready ``RagMealPlanAssembler`` for
    :func:`~foodrag.baselines.full_pipeline.run_full_pipeline`.
    """
    from foodrag.baselines.full_pipeline import RagMealPlanAssembler

    retriever = build_retriever(name, kg, recipes_df, k=retrieve_k, llm=llm, options=options,
                                rog_checkpoint=rog_checkpoint, tfs_checkpoint=tfs_checkpoint)
    return RagMealPlanAssembler(retriever, llm, config_id=config_id or name, recipes_df=recipes_df,
                                retrieve_only=retrieve_only, require_explanation=require_explanation,
                                **assembler_kw)


def build_agentic_pipeline(name: str, kg: HummusKG, recipes_df: pd.DataFrame, *,
                           workflow_builder=None, hummus_data: Optional[Any] = None,
                           llm: Optional[Any] = None, options: Optional[KBQAOptions] = None,
                           config_id: Optional[str] = None, retrieve_k: int = 10,
                           query_engine_kwargs: Optional[dict] = None,
                           rog_checkpoint: Optional[str] = None, tfs_checkpoint: Optional[str] = None,
                           **assembler_kw):
    """A KBQA system under the full agentic pipeline (Heuristic 1, Use 2 §5.7).

    By default assembles the standard five-agent chain in code (turnkey) — pass
    ``hummus_data`` and leave ``workflow_builder=None``. Supply a custom
    ``workflow_builder(query_engine, retriever) -> AgentWorkflow`` to override the
    assembly (e.g. a different agent set).

    Either way, this owns the parts that must wrap the KBQA retriever — the
    retriever and the shared ``RecipeRAGQueryEngine`` — and binds the retriever per
    item, so the whole agent chain runs on this system's retrieval. Returns a
    :class:`~foodrag.baselines.full_pipeline.WorkflowMealPlanAssembler`.
    """
    from foodrag.baselines.full_pipeline import WorkflowMealPlanAssembler
    from foodrag.food_rag.query_engines.rag import RecipeRAGQueryEngine

    if workflow_builder is None:
        if hummus_data is None:
            raise ValueError(
                "Pass hummus_data to use the built-in Heuristic-1 workflow, "
                "or supply a custom workflow_builder.")
        from foodrag.baselines.kbqa.agentic import build_heuristic_workflow_builder

        workflow_builder = build_heuristic_workflow_builder(
            llm=llm, hummus_data=hummus_data, recipes_df=recipes_df)

    retriever = build_retriever(name, kg, recipes_df, k=retrieve_k, llm=llm, options=options,
                                rog_checkpoint=rog_checkpoint, tfs_checkpoint=tfs_checkpoint)
    query_engine = RecipeRAGQueryEngine(retriever=retriever, llm=llm, return_meal_plan=True,
                                        **(query_engine_kwargs or {}))
    agent_workflow = workflow_builder(query_engine, retriever)
    return WorkflowMealPlanAssembler(agent_workflow, retriever, config_id=config_id or name,
                                     **assembler_kw)

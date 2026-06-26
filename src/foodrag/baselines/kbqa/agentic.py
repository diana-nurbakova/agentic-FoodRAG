"""Turnkey Heuristic-1 agent-workflow builder for the KBQA full pipeline (§5.7).

The Use-2 agentic config (Recommend → Correct → Enhance → Explain → Finalize) is
assembled here in code, mirroring ``notebooks/food_RAG_agents.ipynb`` (cells 47–48)
so the caller no longer has to hand-build the workflow. The five-agent chain needs
only the generator ``llm``, the ``hummus_data`` object (for the explanation tool)
and the cleaned ``recipes_df`` (for the correction/validation tool) — *not* the
ingredient embeddings, which only the single-agent ReAct variant uses.

The KBQA retrieval enters through the shared ``RecipeRAGQueryEngine`` that
:func:`~foodrag.baselines.kbqa.factory.build_agentic_pipeline` constructs around
the KBQA retriever and hands to :func:`build_heuristic_workflow_builder`'s closure;
every retrieval the agents do then runs on that system.

Faithfulness note: two redundant kwargs from the notebook (``hummus_data`` on the
SubstitutionAgent, ``mode`` on the FinalizationAgent) are dropped — neither is a
declared parameter nor used by the agent's default tools — so the assembly does
not forward unknown kwargs to the underlying ReActAgent.
"""
from __future__ import annotations

from typing import Any, Callable


def build_heuristic_workflow_builder(*, llm: Any, hummus_data: Any, recipes_df: Any) -> Callable:
    """Return a ``workflow_builder(query_engine, retriever) -> AgentWorkflow`` that
    assembles the standard Heuristic-1 five-agent chain.

    ``llm`` / ``hummus_data`` / ``recipes_df`` are captured in the closure; the
    per-system ``query_engine`` (wrapping the bound KBQA retriever) is supplied by
    the factory at build time, so the whole chain retrieves through that system.
    """

    def _builder(query_engine, retriever):
        from llama_index.core.agent.workflow import AgentWorkflow

        from foodrag.food_rag.agents.heuristic_agents import (ExplanationAgent, FinalizationAgent,
                                                              RecommendationAgent, SubstitutionAgent,
                                                              ValidationAgent)
        from foodrag.food_rag.utils.recipe_context import RecipeState

        recommendation_agent = RecommendationAgent(query_engine=query_engine, llm=llm)
        validation_agent = ValidationAgent(llm=llm, query_engine=query_engine, recipe_df=recipes_df)
        substitution_agent = SubstitutionAgent(llm=llm)
        explanation_agent = ExplanationAgent(llm=llm, hummus_data=hummus_data)
        finalization_agent = FinalizationAgent(llm=llm)

        return AgentWorkflow(
            agents=[recommendation_agent, validation_agent, substitution_agent,
                    explanation_agent, finalization_agent],
            root_agent=recommendation_agent.name,
            initial_state=RecipeState.get_init_state(),
        )

    return _builder

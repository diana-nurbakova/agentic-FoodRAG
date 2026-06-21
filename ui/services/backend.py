"""Heavy backend singletons + retriever factory + agentic-workflow run
(spec §2.2/§2.3, §11.5 services/).

Mirrors the canonical assembly in ``notebooks/food_RAG_agents.ipynb`` (data prep)
and ``notebooks/eval_1.ipynb`` (the 5 retrievers). The engine (LLM + cleaned
HUMMUS dataframe + lazily-built pgvector index) is cached at module scope — the
~300 MB HUMMUS pickle load and the vector index are expensive, so they happen
lazily on first use and are reused across reruns.

Retrievers (spec §2.3):
  - dense / sparse / hybrid → ``RecipeRetriever`` over the pgvector index
  - text-to-pandas          → ``PandasQueryRetriever`` (dataframe only)
  - fusion                  → RRF over [dense, sparse, (KG if available)]
KG (text-to-Cypher over Neo4j) is built best-effort and only contributes to
fusion when it constructs cleanly — the protocol (§5.5) treats KG retrieval as
bounded/constraint-first, not a standalone competitor.
"""

from __future__ import annotations

import asyncio
import inspect
import os
from dataclasses import dataclass, field
from typing import Any, Optional

_ENGINE: Optional["Engine"] = None

# GUI retriever id → pgvector query mode for RecipeRetriever
_VECTOR_MODES = {"dense": "default", "sparse": "sparse", "hybrid": "hybrid"}
# Extra synthesis LLM calls beyond 1 (Text→Pandas runs an LLM to compile the filter)
_LLM_CALLS = {"text-to-pandas": 2, "dense": 1, "sparse": 1, "hybrid": 1, "fusion": 1}


@dataclass
class Engine:
    llm: Any
    embed_model: Any
    hummus_data: dict
    recipes_df: Any
    vector_index: Any = None                       # lazy pgvector VectorStoreIndex
    retrievers: dict = field(default_factory=dict)  # cache: name → DataFrameRetriever
    query_engines: dict = field(default_factory=dict)
    kg_attempted: bool = False
    kg_retriever: Any = None


def is_available() -> tuple[bool, str]:
    """Cheap precondition check (no heavy load). Returns (ok, reason)."""
    try:
        import foodrag  # noqa: F401
        from foodrag.config.settings import load_env, repo_root
    except Exception as exc:  # noqa: BLE001
        return False, f"foodrag not importable: {exc}"
    load_env()
    if not (repo_root() / "data" / "hummus").exists():
        return False, "HUMMUS data not found (data/hummus). Run the data pipeline first."
    if not (os.environ.get("OPENROUTER_API_KEY") or os.environ.get("DEEPINFRA_API_KEY")):
        return False, "No LLM API key in environment (OPENROUTER_API_KEY / DEEPINFRA_API_KEY)."
    return True, "ok"


# ── engine (data prep) ────────────────────────────────────────────────────
def _build_engine() -> Engine:
    from llama_index.core import Settings

    from foodrag.config import settings as foodrag_settings
    from foodrag.food_rag.utils import embedding_helper, hummus_docify
    from foodrag.utils import hummus_util, load_util, rdf_util

    foodrag_settings.load_env()
    data_path = str(foodrag_settings.repo_root() / "data")

    llm = foodrag_settings.build_llm()
    embed_model = foodrag_settings.build_embedding()
    Settings.llm = llm
    Settings.embed_model = embed_model
    try:
        splitter, _, _ = embedding_helper.infer_splitter(embed_model)
        Settings.chunk_size = splitter.chunk_size
    except Exception:  # noqa: BLE001
        pass

    hummus_data = load_util.get_current_hummus(data_path)
    df = hummus_data["recipes_df"].copy()
    df = hummus_docify.stringify_nested_lists(df)
    df = hummus_docify.stringify_nested_dicts(df)
    df = df.rename(columns={
        "duration": "duration [min]", "ingredients": "ingredient_quantities",
        "ingredient_names": "ingredients", "direction_size": "number_of_directions",
        "ingredients_sizes": "number_of_ingredients",
    })
    df = df.drop(columns=["last_changed_date", "food_kg_locator", "recipe_url",
                          "normalization_comment"], errors="ignore")
    df = hummus_util.clean_serves_to_int(df)
    try:
        hummus_data["food_on"] = rdf_util.PreProcessedFoodOn(hummus_data)
    except Exception:  # noqa: BLE001
        pass

    return Engine(llm=llm, embed_model=embed_model, hummus_data=hummus_data, recipes_df=df)


def get_engine() -> Engine:
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = _build_engine()
    return _ENGINE


# ── retriever factory (spec §2.3) ─────────────────────────────────────────
def _vector_index(engine: Engine):
    if engine.vector_index is None:
        from llama_index.core import VectorStoreIndex

        from foodrag.config.settings import embedding_dim, load_models_config
        from foodrag.stores import get_vector_store_provider

        dim = embedding_dim(config=load_models_config())
        provider = get_vector_store_provider("pgvector", reset=False)  # never drop loaded data
        store = provider.connect(table_name="hummus_recipes", embed_dim=dim)
        engine.vector_index = VectorStoreIndex.from_vector_store(store, embed_model=engine.embed_model)
    return engine.vector_index


def _recipe_retriever(engine: Engine, mode: str, top_k: int = 10):
    from foodrag.food_rag.retriever.db_retriever import RecipeRetriever

    return RecipeRetriever(index=_vector_index(engine), recipes_df=engine.recipes_df,
                           vector_store_query_mode=mode, similarity_top_k=top_k, split_node_ratio=2)


def _build_kg_retriever(engine: Engine, top_k: int = 10):
    """Best-effort Text-to-Cypher retriever over Neo4j (spec §5.5). None on failure."""
    if engine.kg_attempted:
        return engine.kg_retriever
    engine.kg_attempted = True
    try:
        from llama_index.core import StorageContext
        from llama_index.graph_stores.neo4j import Neo4jGraphStore

        from foodrag.config.settings import repo_root
        from foodrag.config.stores import neo4j_config
        from foodrag.food_rag.retriever.kg_retriever import KGQueryEngine
        from foodrag.hummus_transformation import transform_kg

        cfg = neo4j_config()
        graph_store = Neo4jGraphStore(username=cfg["user"], password=cfg["password"],
                                      url=cfg["uri"], database=cfg["database"])
        schema = transform_kg.Neo4JTransformer.load(
            data_path=str(repo_root() / "data"), mode=transform_kg.MODE_ALL).get_schema()
        engine.kg_retriever = KGQueryEngine(
            recipes_df=engine.recipes_df, similarity_top_k=top_k, llm=engine.llm,
            storage_context=StorageContext.from_defaults(graph_store=graph_store),
            verbose=False, **schema,
        )
    except Exception:  # noqa: BLE001 — KG is optional; fusion degrades to dense+sparse
        engine.kg_retriever = None
    return engine.kg_retriever


def build_retriever(engine: Engine, name: str):
    """Return a cached DataFrameRetriever for the GUI retriever id."""
    if name in engine.retrievers:
        return engine.retrievers[name]

    if name in _VECTOR_MODES:
        retriever = _recipe_retriever(engine, _VECTOR_MODES[name])
    elif name == "text-to-pandas":
        from foodrag.food_rag.retriever import pandas_retriever

        retriever = pandas_retriever.PandasQueryRetriever(df=engine.recipes_df, verbose=False)
    elif name == "fusion":
        from foodrag.food_rag.retriever.advanced_retriever import DataFrameFusionRetriever

        subs = [_recipe_retriever(engine, "default"), _recipe_retriever(engine, "sparse")]
        kg = _build_kg_retriever(engine)
        if kg is not None:
            subs.append(kg)
        retriever = DataFrameFusionRetriever(
            retrievers=subs, recipes_df=engine.recipes_df, recipe_id_field="recipe_id",
            mode="reciprocal_rerank", num_queries=1, use_async=False, similarity_top_k=10,
        )
    else:
        raise ValueError(f"Unknown retriever {name!r}")

    engine.retrievers[name] = retriever
    return retriever


def build_query_engine(engine: Engine, name: str):
    if name not in engine.query_engines:
        from foodrag.food_rag.query_engines.rag import RecipeRAGQueryEngine

        engine.query_engines[name] = RecipeRAGQueryEngine(
            retriever=build_retriever(engine, name), llm=engine.llm,
            ranking_strategy="rating", k=5, llm_calls=_LLM_CALLS.get(name, 1), debug=False,
        )
    return engine.query_engines[name]


# ── workflow assembly + run ───────────────────────────────────────────────
def build_workflow(engine: Engine, retriever: str = "text-to-pandas"):
    from llama_index.core.agent.workflow import AgentWorkflow

    from foodrag.food_rag.agents.heuristic_agents import (
        ExplanationAgent,
        FinalizationAgent,
        RecommendationAgent,
        SubstitutionAgent,
        ValidationAgent,
    )
    from foodrag.food_rag.tools import finalization_tool
    from foodrag.food_rag.utils.recipe_context import RecipeState

    qe = build_query_engine(engine, retriever)
    recommendation = RecommendationAgent(query_engine=qe)
    validation = ValidationAgent(llm=engine.llm, query_engine=qe, recipe_df=engine.recipes_df)
    substitution = SubstitutionAgent(hummus_data=engine.hummus_data, llm=engine.llm)
    explanation = ExplanationAgent(llm=engine.llm, hummus_data=engine.hummus_data)
    finalization = FinalizationAgent(llm=engine.llm, mode=finalization_tool.MODE_SIMPLE)
    return AgentWorkflow(
        agents=[recommendation, validation, substitution, explanation, finalization],
        root_agent=recommendation.name,
        initial_state=RecipeState.get_init_state(),
    )


def run_pipeline(question: str, retriever: str = "text-to-pandas", timeout: float = 300.0) -> dict:
    """Run the full agentic workflow for one question; return the RecipeState dict."""
    engine = get_engine()
    workflow = build_workflow(engine, retriever)

    async def _arun() -> Any:
        handler = workflow.run(user_msg=question)
        await handler
        state = handler.ctx.get("state")
        if inspect.iscoroutine(state):
            state = await state
        return state

    try:
        state = asyncio.run(asyncio.wait_for(_arun(), timeout=timeout))
    except RuntimeError:
        loop = asyncio.get_event_loop()
        state = loop.run_until_complete(asyncio.wait_for(_arun(), timeout=timeout))

    if isinstance(state, (list, tuple)):
        return state[0] if state else {}
    return state or {}

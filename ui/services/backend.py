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
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

# llama_index PGVectorStore stores rows in ``data_<table_name>``.
_PGV_TABLE = "data_hummus_recipes"

_ENGINE: Optional["Engine"] = None
_ENGINE_LOCK = threading.Lock()
_WARMING = False

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
    neo4j: Any = None                               # lazy Neo4jManager (path traversal)
    _recipe_url_cache: Any = None                   # lazy recipe_id → food.com url map


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
# Files the GUI/agent runtime never reads (~630 MB, mostly the 399 MB reviews CSV):
# reviews_df/users_df are offline-only (KG build / persona gen), usda_json.pkl is
# never keyed. Skipping them cuts cold-start data load ~30%.
_HUMMUS_EXCLUDE = {"reviews_df", "users_df", "usda_json"}
# Bump to invalidate the processed-data cache after a change to the prep below.
_ENGINE_CACHE_VERSION = 1


def _engine_cache_path() -> Path:
    from foodrag.config.settings import repo_root

    return Path(repo_root()) / "data" / "hummus" / "_ui_engine_cache.pkl"


def _source_signature(data_path: str) -> tuple:
    """(size, mtime) of the primary source file, to auto-invalidate a stale cache
    if the dataset is ever rebuilt."""
    src = Path(data_path) / "hummus" / "recipes_df.csv"
    try:
        st = src.stat()
        return (st.st_size, int(st.st_mtime))
    except OSError:
        return ()


def _prep_dataframe(hummus_data: dict):
    from foodrag.food_rag.utils import hummus_docify
    from foodrag.utils import hummus_util

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
    return hummus_util.clean_serves_to_int(df)


def _load_engine_data(data_path: str):
    """Return ``(df, hummus_data)`` for the engine.

    Fast path: a pickled snapshot of the cleaned dataframe + slim hummus_data, so
    subsequent launches skip parsing the 507 MB recipes CSV (and the rest). Slow path:
    parse from source, then write the snapshot. The FoodOn wrapper is rebuilt on load
    (cheap) rather than pickled, avoiding rdflib-graph serialisation issues."""
    import pickle

    from foodrag.utils import load_util, rdf_util

    cache = _engine_cache_path()
    sig = _source_signature(data_path)
    if cache.exists():
        try:
            with open(cache, "rb") as fh:
                blob = pickle.load(fh)
            if (isinstance(blob, dict) and blob.get("version") == _ENGINE_CACHE_VERSION
                    and tuple(blob.get("source") or ()) == sig):
                hummus_data, df = blob["hummus_data"], blob["df"]
                try:
                    hummus_data["food_on"] = rdf_util.PreProcessedFoodOn(hummus_data)
                except Exception:  # noqa: BLE001
                    pass
                return df, hummus_data
        except Exception:  # noqa: BLE001 — corrupt/incompatible cache → rebuild
            pass

    hummus_data = load_util.get_current_hummus(data_path, exclude=_HUMMUS_EXCLUDE)
    df = _prep_dataframe(hummus_data)

    # Persist the snapshot (without the FoodOn wrapper — rebuilt cheaply on load).
    try:
        cache.parent.mkdir(parents=True, exist_ok=True)
        slim = {k: v for k, v in hummus_data.items() if k != "food_on"}
        tmp = cache.with_suffix(".pkl.tmp")
        with open(tmp, "wb") as fh:
            pickle.dump({"version": _ENGINE_CACHE_VERSION, "source": sig, "df": df,
                         "hummus_data": slim}, fh, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(cache)  # atomic swap so a crash mid-write can't leave a partial cache
    except Exception:  # noqa: BLE001 — caching is best-effort
        pass

    try:
        hummus_data["food_on"] = rdf_util.PreProcessedFoodOn(hummus_data)
    except Exception:  # noqa: BLE001
        pass
    return df, hummus_data


def _quiet_engine_noise() -> None:
    """Silence known-benign load-time noise: rdflib's literal-conversion warning for a
    malformed dateTime in the FoodOn ontology, and pydantic's validate_default notice."""
    import logging
    import warnings

    logging.getLogger("rdflib").setLevel(logging.ERROR)      # bad dateTime literal in FoodOn
    logging.getLogger("rdflib.term").setLevel(logging.ERROR)
    warnings.filterwarnings("ignore", message="The 'validate_default' attribute")


def _build_engine() -> Engine:
    from llama_index.core import Settings

    from foodrag.config import settings as foodrag_settings
    from foodrag.food_rag.utils import embedding_helper

    _quiet_engine_noise()
    foodrag_settings.load_env()
    data_path = str(foodrag_settings.repo_root() / "data")

    # The data load and the (import-heavy, ~40s) LLM/embedding init are independent —
    # run the data load on a worker thread so it overlaps the model init instead of
    # adding to it. Pickle/pandas I/O releases the GIL, so this really overlaps.
    import concurrent.futures as _cf

    with _cf.ThreadPoolExecutor(max_workers=1) as _ex:
        data_future = _ex.submit(_load_engine_data, data_path)
        llm = foodrag_settings.build_llm()
        embed_model = foodrag_settings.build_embedding()
        df, hummus_data = data_future.result()

    Settings.llm = llm
    Settings.embed_model = embed_model
    try:
        splitter, _, _ = embedding_helper.infer_splitter(embed_model)
        Settings.chunk_size = splitter.chunk_size
    except Exception:  # noqa: BLE001
        pass

    return Engine(llm=llm, embed_model=embed_model, hummus_data=hummus_data, recipes_df=df)


def get_engine() -> Engine:
    """Return the cached engine, building it (HUMMUS + models) on first use.

    Double-checked locking so a background warm-up (``warm_engine_in_background``)
    and a concurrent live run never build the ~300 MB engine twice."""
    global _ENGINE
    if _ENGINE is None:
        with _ENGINE_LOCK:
            if _ENGINE is None:
                _ENGINE = _build_engine()
    return _ENGINE


def engine_ready() -> bool:
    """True once the engine is built (HUMMUS + models loaded)."""
    return _ENGINE is not None


def warm_engine_in_background() -> None:
    """Preload the engine in a daemon thread so the first live run doesn't pay the
    HUMMUS/model load. Idempotent and non-blocking — safe to call on every launch."""
    global _WARMING
    # Never build the heavy engine during a test run or when explicitly disabled.
    if os.environ.get("FOODRAG_SKIP_STORES_BOOTSTRAP") or "PYTEST_CURRENT_TEST" in os.environ:
        return
    if _ENGINE is not None or _WARMING:
        return
    _WARMING = True

    def _run() -> None:
        global _WARMING
        try:
            get_engine()
        except Exception:  # noqa: BLE001 — warm-up failures surface on the real run
            pass
        finally:
            _WARMING = False

    threading.Thread(target=_run, name="engine-warmup", daemon=True).start()


# ── retriever factory (spec §2.3) ─────────────────────────────────────────
def _pg_table_count(table: str) -> int:
    """Row count of a pgvector table (0 if the DB/table is unreachable or absent)."""
    import psycopg2

    from foodrag.config.stores import postgres_credentials

    creds = postgres_credentials()
    conn = None
    try:
        conn = psycopg2.connect(dbname=creds["db_name"], host=creds["host"],
                                password=creds["password"], port=creds["port"], user=creds["user"])
        with conn.cursor() as cur:
            cur.execute("SELECT to_regclass(%s)", (f"public.{table}",))
            if cur.fetchone()[0] is None:
                return 0
            cur.execute(f"SELECT count(*) FROM {table}")  # table is a fixed constant, not user input
            return int(cur.fetchone()[0])
    except Exception:  # noqa: BLE001 — treat any failure as "empty / not ready"
        return 0
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:  # noqa: BLE001
                pass


def _recipe_embeddings_dir() -> Optional[Path]:
    """Directory of precomputed recipe-embedding batches (data/embeddings/*/recipe_documents)."""
    from foodrag.config.settings import repo_root

    base = Path(repo_root()) / "data" / "embeddings"
    if not base.exists():
        return None
    for cand in sorted(base.glob("*/recipe_documents")):
        if any(cand.glob("batch_*.pkl")):
            return cand
    return None


def _ensure_recipe_index(engine: Engine, store) -> None:
    """Populate the pgvector table once from the precomputed on-disk embeddings.

    No-op when the table already holds rows — the pgdata volume persists them across
    restarts, so embeddings are ingested a single time and never recomputed."""
    if _pg_table_count(_PGV_TABLE) > 0:
        return
    emb_dir = _recipe_embeddings_dir()
    if emb_dir is None:
        raise RuntimeError(
            "The pgvector table is empty and no precomputed recipe embeddings were found under "
            "data/embeddings/*/recipe_documents — build them once (notebooks/init_vector_store.ipynb).")

    from llama_index.core import VectorStoreIndex

    from foodrag.food_rag.utils import embedding_helper

    doc_nodes_dict, _ = embedding_helper.load_existing_batches(str(emb_dir))
    nodes = [n for lst in doc_nodes_dict.values() for n in lst]
    if not nodes:
        raise RuntimeError(f"No embedding batches could be loaded from {emb_dir}.")
    # Nodes already carry their embeddings, so this inserts without recomputing them.
    idx = VectorStoreIndex.from_vector_store(store, embed_model=engine.embed_model)
    idx.build_index_from_nodes(nodes=nodes)


def _vector_index(engine: Engine):
    if engine.vector_index is None:
        from llama_index.core import VectorStoreIndex

        from foodrag.config.settings import embedding_dim, load_models_config
        from foodrag.stores import get_vector_store_provider

        dim = embedding_dim(config=load_models_config())
        provider = get_vector_store_provider("pgvector", reset=False)  # never drop loaded data
        store = provider.connect(table_name="hummus_recipes", embed_dim=dim)
        _ensure_recipe_index(engine, store)  # one-time ingest from disk if the table is empty
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
        # KGQueryEngine maps graph results back via 'recipe_url', which the cleaned
        # engine df drops — re-attach it from the original df.
        kg_df = engine.recipes_df
        if "recipe_url" not in kg_df.columns:
            orig = engine.hummus_data["recipes_df"][["recipe_id", "recipe_url"]]
            kg_df = engine.recipes_df.merge(orig, on="recipe_id", how="left")
        engine.kg_retriever = KGQueryEngine(
            recipes_df=kg_df, similarity_top_k=top_k, llm=engine.llm,
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
# Study/eval "hypotheses" (docs/ARCHITECTURE.md §4): H1 = full five-agent
# Recommend+Correct+Enhance chain; H2 = recommend-only (the recommendation tool
# alone, no validation/substitution/explanation).
HYP_H1 = "H1"
HYP_H2 = "H2"


def build_workflow(engine: Engine, retriever: str = "text-to-pandas",
                   hypothesis: str = HYP_H1, full_day: bool = False,
                   validation: bool = True, substitution: bool = True,
                   explanation: bool = True, time: str = "now",
                   max_correction_rounds: Optional[int] = None,
                   single_meal: Optional[str] = None):
    """Assemble the agentic workflow.

    ``hypothesis`` selects the chain: ``"H1"`` (default) = the Recommend → [Validate]
    → [Substitute] → [Explain] → Finalize pipeline; ``"H2"`` = a single recommend-only
    ``RecommendationAgent``.

    For H1 the three optional stages are individually toggleable (``validation`` /
    ``substitution`` / ``explanation``): each disabled stage is dropped from the graph
    and the hand-off chain + prompts are rewired so control flows to the next enabled
    stage. Recommend and Finalize are always present (the GUI shows them locked-on).

    ``full_day`` asks the recommendation tool for all four meal slots; ``time`` pins the
    recommendation time-of-day (e.g. ``"evening"`` to target dinner) or ``"now"``."""
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

    if str(hypothesis).upper() == HYP_H2:
        # Recommend-only: one agent, no hand-offs (H2 ablation).
        recommendation = RecommendationAgent(query_engine=qe, full_day=full_day,
                                             time=time, single_meal=single_meal, can_handoff_to=[])
        return AgentWorkflow(
            agents=[recommendation],
            root_agent=recommendation.name,
            initial_state=RecipeState.get_init_state(),
        )

    # H1 granular: the enabled optional stages between Recommend and Finalize, in order.
    chain = [name for name, on in (
        ("ValidationAgent", validation),
        ("SubstitutionAgent", substitution),
        ("ExplanationAgent", explanation),
    ) if on] + ["FinalizationAgent"]

    def _after(name: str) -> str:
        """The enabled stage that follows ``name`` in the chain (Finalization ends it)."""
        return chain[chain.index(name) + 1] if name in chain[:-1] else "FinalizationAgent"

    recommendation = RecommendationAgent(
        query_engine=qe, full_day=full_day, time=time, single_meal=single_meal,
        can_handoff_to=[chain[0]],
        system_prompt=(
            "You are the RecommendationAgent that can recommend meal plans. You MUST call "
            "the meal-plan recommendation tool to produce a plan; do not answer from your own "
            "knowledge. You should have at least some recommendations before handing off. "
            f"Once your recommendation is satisfied, hand off control to the {chain[0]}."),
    )
    agents = [recommendation]

    if validation:
        valid_target = _after("SubstitutionAgent") if substitution else _after("ValidationAgent")
        targets = (["SubstitutionAgent"] if substitution else []) + [valid_target]
        branch = (f"if the plan is still INVALID hand off to the SubstitutionAgent; otherwise "
                  f"hand off to the {valid_target}") if substitution else \
                 f"hand off to the {valid_target}"
        agents.append(ValidationAgent(
            llm=engine.llm, query_engine=qe, recipe_df=engine.recipes_df,
            can_handoff_to=targets, max_correction_rounds=max_correction_rounds,
            system_prompt=(
                "You are the ValidationAgent that validates meal plans. You MUST call the "
                "recipe correction tool to validate and, if needed, correct the current meal "
                f"plan before handing off. After the tool runs, {branch}."),
        ))

    if substitution:
        nxt = _after("SubstitutionAgent")
        agents.append(SubstitutionAgent(
            hummus_data=engine.hummus_data, llm=engine.llm, can_handoff_to=[nxt],
            system_prompt=(
                "You are the SubstitutionAgent that enhances or improves meal plans. Call the "
                f"substitution tool to improve the current meal plan, then hand off control to the {nxt}."),
        ))

    if explanation:
        agents.append(ExplanationAgent(
            llm=engine.llm, hummus_data=engine.hummus_data,
            can_handoff_to=["FinalizationAgent"],
        ))

    agents.append(FinalizationAgent(llm=engine.llm, mode=finalization_tool.MODE_SIMPLE))
    return AgentWorkflow(
        agents=agents,
        root_agent=recommendation.name,
        initial_state=RecipeState.get_init_state(),
    )


def recipe_url_for(recipe_id) -> str:
    """food.com recipe URL for a HUMMUS ``recipe_id`` (or "" when unknown).

    The working engine df drops ``recipe_url``; re-read it from the untouched
    original ``hummus_data['recipes_df']`` and cache a recipe_id → url map."""
    if recipe_id is None:
        return ""
    engine = get_engine()
    cache = getattr(engine, "_recipe_url_cache", None)
    if cache is None:
        cache = {}
        try:
            df = engine.hummus_data["recipes_df"]
            cols = [c for c in ("recipe_id", "recipe_url", "food_kg_locator") if c in df.columns]
            for row in df[cols].itertuples(index=False):
                rid = getattr(row, "recipe_id", None)
                url = getattr(row, "recipe_url", None) or ""
                if not url:
                    loc = getattr(row, "food_kg_locator", None) or ""
                    url = str(loc).replace("http://idea.rpi.edu/heals/kb/recipe/",
                                           "https://www.food.com/recipe/") if loc else ""
                if rid is not None and url:
                    cache[rid] = str(url)
        except Exception:  # noqa: BLE001 — best-effort; empty map just yields ""
            cache = {}
        engine._recipe_url_cache = cache
    url = cache.get(recipe_id)
    if url:
        return url
    try:  # tolerate int/str/float id mismatches
        return cache.get(int(recipe_id), "") or cache.get(str(recipe_id), "")
    except (TypeError, ValueError):
        return ""


def run_pipeline(question: str, retriever: str = "text-to-pandas",
                 hypothesis: str = HYP_H1, full_day: bool = False,
                 validation: bool = True, substitution: bool = True,
                 explanation: bool = True, time: str = "now",
                 max_correction_rounds: Optional[int] = None, single_meal: Optional[str] = None,
                 on_event=None, meta: Optional[dict] = None, timeout: float = 300.0) -> dict:
    """Run the agentic workflow for one question; return the RecipeState dict.

    ``on_event`` (optional) is called synchronously for every workflow event
    (AgentInput/AgentOutput/ToolCall/…) as the run progresses, so the GUI can show
    live per-stage progress. It runs on the calling thread; keep it cheap and
    exception-safe (raised errors are swallowed).

    ``meta`` (optional dict) receives run metadata: ``meta["partial"]`` is set True if
    the run hit ``timeout`` — the recommendation is committed to the context early, so we
    salvage and return that partial state instead of losing everything to a hard error."""
    meta = meta if meta is not None else {}
    engine = get_engine()
    workflow = build_workflow(engine, retriever, hypothesis=hypothesis, full_day=full_day,
                              validation=validation, substitution=substitution,
                              explanation=explanation, time=time,
                              max_correction_rounds=max_correction_rounds,
                              single_meal=single_meal)
    holder: dict = {}

    async def _drive() -> Any:
        handler = workflow.run(user_msg=question)
        holder["handler"] = handler
        if on_event is not None:
            async for ev in handler.stream_events():
                try:
                    on_event(ev)
                except Exception:  # noqa: BLE001 — progress display must never break the run
                    pass
        await handler                          # ensure completion / propagate errors
        state = handler.ctx.get("state")
        if inspect.iscoroutine(state):
            state = await state
        return state

    async def _guarded() -> Any:
        try:
            return await asyncio.wait_for(_drive(), timeout=timeout)
        except (asyncio.TimeoutError, TimeoutError):
            meta["partial"] = True
            handler = holder.get("handler")
            if handler is not None:  # salvage whatever was committed (usually the plan)
                try:
                    state = handler.ctx.get("state")
                    if inspect.iscoroutine(state):
                        state = await state
                    return state
                except Exception:  # noqa: BLE001
                    return {}
            return {}

    # Each run gets its own event loop via asyncio.run (works on the caller's — here a
    # background — thread). NOTE: a shared persistent loop was tried to silence litellm's
    # async-logging-worker warnings, but it hung the LLM/retrieval calls; the warnings are
    # harmless, so per-run loops are the correct trade-off.
    try:
        state = asyncio.run(_guarded())
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            state = loop.run_until_complete(_guarded())
        finally:
            loop.close()

    if isinstance(state, (list, tuple)):
        return state[0] if state else {}
    return state or {}


# ── bounded KG path traversal (eval protocol §5.5/§8.1, richer paths) ──────
def ingredient_class_cypher(recipe_id: int, max_ingredients: int = 8, depth: int = 2) -> str:
    """Bounded Cypher: recipe → HAS_INGREDIENT → ingredient → IS_EQUIVALENT_TO →
    Food_on → SUB_CLASS_OF*1..depth → Food_on. Depth/ingredient caps keep the
    traversal bounded (single already-recommended recipe → no hub fan-out, §5.5).
    Pure (string only) so it is unit-testable without a live database."""
    rid = int(recipe_id)
    n = max(1, min(int(max_ingredients), 25))
    d = max(1, min(int(depth), 4))
    return (
        f"MATCH (r:Recipe {{recipe_id: {rid}}})-[:HAS_INGREDIENT]->(i:Ingredient)"
        f"-[:IS_EQUIVALENT_TO]->(f:Food_on)\n"
        f"OPTIONAL MATCH (f)-[:SUB_CLASS_OF*1..{d}]->(a:Food_on)\n"
        f"WITH i.name AS ingredient, f.label AS food_class, collect(DISTINCT a.label) AS ancestors\n"
        f"WHERE food_class IS NOT NULL\n"
        f"RETURN ingredient, food_class, ancestors\n"
        f"LIMIT {n}"
    )


def _neo4j(engine: Engine):
    if engine.neo4j is None:
        from foodrag.config.stores import neo4j_config
        from foodrag.food_rag.utils.neo4J_helper import Neo4jManager

        cfg = neo4j_config()
        engine.neo4j = Neo4jManager(cfg["uri"], cfg["user"], cfg["password"], cfg["database"])
    return engine.neo4j


def recipe_ingredient_class_paths(recipe_id, max_ingredients: int = 8, depth: int = 2) -> list[dict]:
    """Live recipe→ingredient→FoodOn-class paths. Returns
    ``[{ingredient, food_class, ancestors}]``; [] on any failure (best-effort)."""
    if recipe_id is None:
        return []
    try:
        rows = _neo4j(get_engine()).query(ingredient_class_cypher(recipe_id, max_ingredients, depth))
    except Exception:  # noqa: BLE001
        return []
    out = []
    for r in rows or []:
        fc = r.get("food_class")
        if not fc:
            continue
        ancestors = [a for a in (r.get("ancestors") or []) if a and a != fc][:depth]
        out.append({"ingredient": r.get("ingredient") or "?", "food_class": fc, "ancestors": ancestors})
    return out

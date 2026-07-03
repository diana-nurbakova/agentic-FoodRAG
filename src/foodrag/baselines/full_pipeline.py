"""Use-2 full-pipeline driver: assemble a meal plan, judge it, emit §13 records.

This is the "Use 2" half of the dual-use design (§5.7): instead of scoring a
baseline's ranked list directly (Use 1, :mod:`runner`), it attaches **FoodRAG's
meal-plan assembly layer** on top of the baseline's retrieval and evaluates the
resulting plan end-to-end. Holding the assembly layer constant across baselines
makes end-to-end differences attributable to retrieval quality (the §5.7
decomposition), and is the generous adaptation that lets a full-pipeline FoodRAG
win mean something.

Per QA item it produces one §13.1 record combining three metric families on the
*assembled plan*:

* deterministic safety/retrieval (§4.1–4.3) — reusing :mod:`metrics` on the plan's
  recipes, so "valid recipe" matches the gold definition exactly;
* deterministic meal-plan quality (§4.4–4.5) — :mod:`meal_plan_metrics`;
* LLM-judged quality (§4.4, §7) — the dual-judge :mod:`judge` pass.

The assembly itself is behind the :class:`MealPlanAssembler` protocol, so the
orchestration is fully testable with a fake assembler; :class:`RagMealPlanAssembler`
is the live implementation that drives the real recommendation tool with a KBQA
(or any ``DataFrameRetrieverBase``) retriever bound per item.
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional, Sequence

import pandas as pd

from foodrag.baselines import constraints, guideline_adherence, meal_plan_metrics, metrics
from foodrag.baselines import judge as judge_mod
from foodrag.baselines.base import answer_set_stratum, requirement_count, requirement_types
from foodrag.baselines.runner import _resolve_rows

ID_COL = "recipe_url"
GOLD_TITLES_CAP = 20  # gold sets can be huge; show the judge a representative slice


@dataclass
class AssembledPlan:
    """The output of the assembly layer for one QA item."""

    question: str
    answer_text: str                          # response string shown to the judges
    plan: dict                                # {meal_type_str: recipe_row | None}
    recipe_urls: list[str] = field(default_factory=list)
    contexts: list[str] = field(default_factory=list)  # retrieved recipes (for faithfulness)
    has_explanation: bool = False
    # The raw NL explanation string (§8.2 claim-grounding input; the text
    # ``has_explanation`` is derived from). "" when the config emits no explanation.
    explanation_text: str = ""
    # Any structured "why this recipe" evidence the assembly layer carries. The current
    # pipeline emits pure NL, so this holds at most the per-slot validation verdicts
    # ({meal_type: verdict_str}); no KG paths are produced. {} when none.
    explanation_structured: dict = field(default_factory=dict)
    llm_usage: float = float("nan")
    llm_seq_usage: float = float("nan")
    llm_parsing: float = float("nan")
    latency_s: float = float("nan")


class MealPlanAssembler(ABC):
    """Turns a QA item into an :class:`AssembledPlan`.

    ``config_id`` is the result-file key (e.g. ``tog``); ``require_explanation``
    tells the judge/structural-validity layer whether an explanation is expected
    (Heuristic-1-style configs) or not (plain recommendation).
    """

    config_id: str = ""
    require_explanation: bool = False

    @abstractmethod
    def assemble(self, qa_item: dict) -> AssembledPlan:
        raise NotImplementedError


def run_full_pipeline(assembler: MealPlanAssembler, qa_set: Iterable[dict],
                      recipes_df: pd.DataFrame, *, judges: Sequence[judge_mod.Judge] = (),
                      k: int = 10, experiment_id: str = "EXP-7", rung: str = "4", run: int = 1,
                      expand: constraints.IngredientExpander = constraints.identity_expand,
                      id_prefix: str = "q") -> list[dict]:
    """Assemble + score every QA item; return §13.1 full-pipeline records."""
    indexed = recipes_df.set_index(ID_COL, drop=False)
    needed = [c for c in indexed.columns if c != "tags_str"]
    records: list[dict] = []

    for i, qa_item in enumerate(qa_set):
        gold = set(qa_item.get("answer") or [])
        persona = qa_item["persona"]

        assembled = assembler.assemble(qa_item)
        rows = _resolve_rows(assembled.recipe_urls, indexed, needed)

        block = metrics.deterministic_metrics(assembled.recipe_urls, rows, gold, persona, k, expand)
        if not gold:  # retrieval metrics undefined without a gold set
            for m in (f"hit_at_{k}", f"mrr_at_{k}", f"ndcg_at_{k}"):
                block[m] = float("nan")
        block.update(meal_plan_metrics.plan_quality_metrics(
            assembled.plan, has_explanation=assembled.has_explanation,
            require_explanation=assembler.require_explanation))
        # §4.5.1 plan-level guideline adherence + hard-ceiling → hard_constraint_violation.
        guide = guideline_adherence.guideline_adherence(assembled.plan, persona)
        block["guideline_adherence_overall"] = guide["guideline_adherence_overall"]
        block["guideline_adherence_per_nutrient"] = guide["guideline_adherence_per_nutrient"]
        block["guideline_class"] = guide["guideline_class"]
        # A plan exceeding a nutrient hard_ceiling (UL) is a safety flag (§4.3): raise
        # hard_constraint_violation to 1.0 for the plan (max with the rec-list value).
        if guide["guideline_hard_ceiling_breach"]:
            block["hard_constraint_violation"] = max(
                block.get("hard_constraint_violation", 0.0) or 0.0, 1.0)
        block.update(judge_mod.judge_plan(
            judges, question=assembled.question, answer=assembled.answer_text,
            contexts=assembled.contexts, gold=_gold_titles(gold, indexed),
            require_explanation=assembler.require_explanation))
        block.update({
            "llm_usage": assembled.llm_usage, "llm_seq_usage": assembled.llm_seq_usage,
            "llm_parsing": assembled.llm_parsing, "latency_s": assembled.latency_s,
        })

        records.append({
            "experiment_id": experiment_id,
            "config_id": assembler.config_id,
            "rung": rung,
            "run": run,
            "question_id": f"{id_prefix}_{i:05d}",
            "answer_set_size": len(gold),
            "stratum": answer_set_stratum(len(gold)),
            "num_requirements": requirement_count(persona),
            "requirement_types": requirement_types(persona),
            "k": k,
            "metrics": block,
            "retrieved_ids": assembled.recipe_urls,
            # Persist the assembled plan (per-slot recipe_url) so plan-level metrics are
            # reproducible/backfillable offline, plus the explanation inputs for the §8 task.
            "plan": _plan_urls(assembled.plan),
            "explanation_text": assembled.explanation_text,
            "explanation_structured": assembled.explanation_structured,
        })

    return records


def _plan_urls(plan: dict) -> dict:
    """``{meal_type: recipe_url | None}`` — the persisted, backfillable form of a plan."""
    return {str(mt): (r.get("recipe_url") if r else None) for mt, r in (plan or {}).items()}


def _gold_titles(gold: set[str], indexed: pd.DataFrame) -> list[str]:
    """A capped sample of gold recipe titles for the correctness judge prompt."""
    titles = []
    for url in list(gold)[:GOLD_TITLES_CAP]:
        if url in indexed.index:
            row = indexed.loc[url]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            title = row.get("title")
            if title:
                titles.append(str(title))
    return titles


# ── live assembly over the real recommendation tool ──────────────────────────

class RagMealPlanAssembler(MealPlanAssembler):
    """Drive FoodRAG's meal-plan recommendation tool with an injected retriever.

    The retriever (a :class:`~foodrag.baselines.kbqa.adapters.KBQARetriever` or any
    ``DataFrameRetrieverBase``) is bound to the QA item before each run so the KBQA
    systems see the persona's constraints. ``retrieve_only`` selects the assembly
    mode: ``True`` = recommend from the retrieved set (Heuristic-2-style), ``False``
    = the query engine's LLM picks per slot (Plain-RAG-style).
    """

    def __init__(self, retriever, llm, *, config_id: str, recipes_df: Optional[pd.DataFrame] = None,
                 retrieve_only: bool = True, k: int = 5, llm_calls: int = 1,
                 ranking_strategy: Optional[str] = "rating", time: Optional[str] = "now",
                 date: Optional[str] = "today", require_explanation: bool = False,
                 verbose: bool = False):
        self.retriever = retriever
        self.llm = llm
        self.config_id = config_id
        self.retrieve_only = retrieve_only
        self.k = k
        self.llm_calls = llm_calls
        self.ranking_strategy = ranking_strategy
        self.time = time
        self.date = date
        self.require_explanation = require_explanation
        self.verbose = verbose

    def assemble(self, qa_item: dict) -> AssembledPlan:
        from foodrag.food_rag.agents.workflow_runner import get_dummy_state
        from foodrag.food_rag.query_engines.rag import RecipeRAGQueryEngine
        from foodrag.food_rag.tools.meal_plan_tools import MealPlanRecommendationTool
        from foodrag.food_rag.utils.recipe_context import RecipeState

        if hasattr(self.retriever, "bind"):
            self.retriever.bind(qa_item)
        question = qa_item.get("question", "")

        query_engine = RecipeRAGQueryEngine(
            retriever=self.retriever, llm=self.llm, ranking_strategy=self.ranking_strategy,
            k=self.k, llm_calls=self.llm_calls, return_meal_plan=True)
        tool = MealPlanRecommendationTool(
            llm=self.llm, recipe_query_engine=query_engine, retrieve_only=self.retrieve_only,
            time=self.time, date=self.date, verbose=self.verbose)

        ctx = get_dummy_state().get_context()
        start = time.perf_counter()
        meal_plan = tool.recommend_meal_plan(ctx=ctx, question=question)
        latency = time.perf_counter() - start

        plan = self._normalise_plan(meal_plan)
        urls = _dedupe([r.get("recipe_url") for r in plan.values() if r and r.get("recipe_url")])
        usage = self._usage(RecipeState, ctx)
        return AssembledPlan(
            question=question, answer_text=meal_plan.get_recommendation_str(), plan=plan,
            recipe_urls=urls, contexts=self._contexts(question), has_explanation=self.require_explanation,
            latency_s=latency, **usage)

    @staticmethod
    def _normalise_plan(meal_plan) -> dict:
        return {str(mt): (recipe or None) for mt, recipe in meal_plan.to_dict().items()}

    def _contexts(self, question: str) -> list[str]:
        try:
            df = self.retriever.retrieve(question)
        except Exception:
            return []
        return df["title"].astype(str).tolist() if "title" in getattr(df, "columns", []) else []

    @staticmethod
    def _usage(recipe_state_cls, ctx) -> dict:
        try:
            state = recipe_state_cls.from_context(ctx)
            return {"llm_usage": state.get_llm_usage_counter(),
                    "llm_seq_usage": state.get_llm_sequential_usage_counter(),
                    "llm_parsing": state.get_llm_parsing_usage_counter()}
        except Exception:
            return {"llm_usage": float("nan"), "llm_seq_usage": float("nan"),
                    "llm_parsing": float("nan")}


# ── live assembly over the full agentic workflow (Heuristic 1) ───────────────

# RecipeState keys (mirrored so this module stays free of the heavy agent imports).
_PLAN_KEY = "meal_plan_recommendation"
_EXPLANATION_KEY = "meal_plan_explanation"
_VALIDATION_KEY = "meal_plan_validation"
_FINAL_KEY = "final_recommendation"
_LLM_USAGE_KEY = "llm_usage_counter"
_LLM_SEQ_KEY = "llm_sequential_usage_counter"
_LLM_PARSING_KEY = "llm_parsing_usage_counter"
_EXPLANATION_DEFAULTS = {"", "not written yet."}
_FINAL_DEFAULTS = {"", "final recommendation required."}


def _validation_evidence(state: dict) -> dict:
    """The only semi-structured "why this recipe" evidence the pipeline carries: the
    per-slot validation verdicts (``{meal_type: verdict_str}``). Pure NL otherwise — no
    KG paths. Returns ``{}`` when absent. Persisted for the future §8 explainability task.
    """
    val = state.get(_VALIDATION_KEY)
    if val is None:
        return {}
    try:
        return {str(mt): str(v) for mt, v in (val.to_dict().items() if hasattr(val, "to_dict")
                                              else dict(val).items())}
    except Exception:
        return {}


class WorkflowMealPlanAssembler(MealPlanAssembler):
    """Drive the full agentic pipeline (Recommend → Correct → Enhance → Explain)
    on top of a KBQA retriever — the Heuristic-1 Use-2 config.

    The agent workflow is assembled by the caller (it needs hummus_data,
    embeddings, etc. — the notebook's job) and shares its retrieval through the
    same ``DataFrameRetrieverBase`` we bind here per item, so the whole agent chain
    runs on the KBQA system's retrieval. The final :class:`RecipeState` yields the
    recommended plan, the explanation and the LLM-usage counters.

    ``run_workflow`` is injectable — ``(agent_workflow, question, memory) ->
    (state_dict, latency_s)`` — so the extraction logic is unit-testable without a
    live LLM; the default runs the real async workflow.
    """

    require_explanation = True

    def __init__(self, agent_workflow, retriever, *, config_id: str,
                 memory: Any = None, query_field: str = "question",
                 run_workflow=None):
        self.agent_workflow = agent_workflow
        self.retriever = retriever
        self.config_id = config_id
        self.memory = memory
        self.query_field = query_field
        self._run_workflow = run_workflow or _default_workflow_run

    def assemble(self, qa_item: dict) -> AssembledPlan:
        if hasattr(self.retriever, "bind"):
            self.retriever.bind(qa_item)
        question = qa_item.get(self.query_field) or qa_item.get("question", "")
        state, latency = self._run_workflow(self.agent_workflow, question, self.memory)
        return self._from_state(state or {}, question, latency)

    def _from_state(self, state: dict, question: str, latency: float) -> AssembledPlan:
        plan = _normalise_any_plan(state.get(_PLAN_KEY))
        explanation = str(state.get(_EXPLANATION_KEY, "") or "")
        has_explanation = explanation.strip().lower() not in _EXPLANATION_DEFAULTS
        urls = _dedupe([r.get("recipe_url") for r in plan.values() if r and r.get("recipe_url")])
        contexts = [str(r.get("title")) for r in plan.values() if r and r.get("title")]
        explanation_text = explanation if has_explanation else ""
        return AssembledPlan(
            question=question, answer_text=_answer_text(state, plan, explanation, has_explanation),
            plan=plan, recipe_urls=urls, contexts=contexts, has_explanation=has_explanation,
            explanation_text=explanation_text, explanation_structured=_validation_evidence(state),
            llm_usage=state.get(_LLM_USAGE_KEY, float("nan")),
            llm_seq_usage=state.get(_LLM_SEQ_KEY, float("nan")),
            llm_parsing=state.get(_LLM_PARSING_KEY, float("nan")), latency_s=latency)


def _normalise_any_plan(plan) -> dict:
    """A MealPlan object or a meal_type→recipe dict → ``{slot_str: recipe|None}``."""
    if plan is None:
        return {}
    items = plan.to_dict().items() if hasattr(plan, "to_dict") else dict(plan).items()
    return {str(mt): (recipe or None) for mt, recipe in items}


def _answer_text(state: dict, plan: dict, explanation: str, has_explanation: bool) -> str:
    parts: list[str] = []
    final = str(state.get(_FINAL_KEY, "") or "")
    if final.strip().lower() not in _FINAL_DEFAULTS:
        parts.append(final)
    else:
        titles = [f"{slot}: {r.get('title')}" for slot, r in plan.items() if r and r.get("title")]
        if titles:
            parts.append("Meal plan:\n" + "\n".join(titles))
    if has_explanation:
        parts.append(explanation)
    return "\n\n".join(parts) if parts else "No recommendation produced."


def _default_workflow_run(agent_workflow, question: str, memory):
    """Run the real async AgentWorkflow once; return ``(state_dict, latency_s)``."""
    import asyncio
    import time as _time

    try:  # allow running inside an existing loop (Jupyter)
        import nest_asyncio
        nest_asyncio.apply()
    except Exception:
        pass

    async def _go():
        if memory is not None:
            try:
                memory.reset()
            except Exception:
                pass
            handler = agent_workflow.run(user_msg=question, memory=memory)
        else:
            handler = agent_workflow.run(user_msg=question)
        await handler
        state = await handler.ctx.get("state")
        return state[0] if isinstance(state, (list, tuple)) else state

    start = _time.perf_counter()
    data = asyncio.run(_go())
    return data, _time.perf_counter() - start


def _dedupe(values: Iterable) -> list:
    seen, out = set(), []
    for v in values:
        if v and v not in seen:
            seen.add(v)
            out.append(v)
    return out

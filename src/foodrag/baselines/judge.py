"""LLM-as-judge pass for the full-pipeline experiments.

The Use-2 quality metrics on an assembled meal plan are LLM-judged, under the
dual-judge protocol: every metric is scored by **Judge A** (a cross-lineage
model, e.g. Gemma) and **Judge B** (same family as the generator, e.g. Llama),
emitted as ``<metric>_judgeA`` / ``<metric>_judgeB`` so inter-judge agreement and
the same-family-bias delta can be computed downstream.

Four metrics, prompts adapted from the LlamaIndex evaluator templates the
thesis used, kept frozen here:

* **Faithfulness** {0,1} — does the recommendation stay within the retrieved recipes?
* **Relevancy** {0,1} — does it fit the question + retrieved recipes?
* **Correctness** [0,1] — correct w.r.t. the question + gold answers (1–5 → /4).
* **Explanation Quality** [0,1] — coherent user-level rationale, judged only when
  the config emits an explanation (Heuristic 1); otherwise omitted.

The judge LLM is injected; with no LLM every metric is ``NaN`` (so the aggregate
skips it) — which keeps the orchestrator runnable/testable without a judge.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

from foodrag.baselines.kbqa import llm_io

FAITHFULNESS_PROMPT = """\
You are evaluating a meal-plan recommendation for faithfulness.

Retrieved recipes available to the system:
{contexts}

Recommendation:
{answer}

Does the recommendation use ONLY recipes from the retrieved set above (no invented
recipes)? Answer ONLY {{"passing": true}} or {{"passing": false}}.
"""

RELEVANCY_PROMPT = """\
You are evaluating a meal-plan recommendation for relevancy.

User question: {question}
Retrieved recipes:
{contexts}

Recommendation:
{answer}

Does the recommendation fit the user's question given the retrieved recipes?
Answer ONLY {{"passing": true}} or {{"passing": false}}.
"""

CORRECTNESS_PROMPT = """\
You are grading a meal-plan recommendation for correctness.

User question: {question}
Reference (gold) answer recipes: {gold}

Recommendation:
{answer}

Rate how correct the recommendation is with respect to the question and the gold
answers, from 1 (wrong; e.g. violates a stated allergy) to 5 (fully correct).
Penalise allergen/constraint violations more than minor tag mismatches.
Answer ONLY {{"score": <1-5>}}.
"""

EXPLANATION_PROMPT = """\
You are grading the explanation accompanying a meal-plan recommendation.

User question: {question}
Recommendation (with explanation):
{answer}

Rate the explanation as a coherent, user-level rationale (not meta-text, not
chain-of-thought, not raw data), from 1 (poor) to 5 (excellent).
Answer ONLY {{"score": <1-5>}}.
"""


@dataclass
class Judge:
    """A named judge: a logical label (``judgeA``/``judgeB``) + its LLM."""

    name: str
    llm: Optional[Any] = None


def build_judges(judge_names: Sequence[tuple[str, Optional[str]]] = (("judgeA", "gemma"), ("judgeB", None)),
                 **overrides) -> list[Judge]:
    """Build judges from the project's judge config.

    ``judge_names`` pairs a logical label with a ``models.yml`` judge key. The
    default makes **Judge A** the cross-lineage Gemma (the headline judge)
    and **Judge B** the configured default judge (``None``); both differ from the
    Llama generator, so the same-family-bias delta is measurable. Imported lazily
    so this module has no config dependency when judges are injected (e.g. tests).
    """
    from foodrag.config.settings import build_judge

    return [Judge(label, build_judge(name, **overrides)) for label, name in judge_names]


def judge_plan(judges: Sequence[Judge], *, question: str, answer: str,
               contexts: Sequence[str], gold: Optional[Sequence[str]] = None,
               require_explanation: bool = False) -> dict:
    """Score one assembled plan with every judge; return the judge-metric block."""
    ctx_text = "\n".join(contexts) if contexts else "No recipes were retrieved."
    gold_text = ", ".join(gold) if gold else "Not provided."
    metrics: dict = {}
    for judge in judges:
        metrics[f"faithfulness_{judge.name}"] = _binary(
            judge, FAITHFULNESS_PROMPT.format(contexts=ctx_text, answer=answer))
        metrics[f"relevancy_{judge.name}"] = _binary(
            judge, RELEVANCY_PROMPT.format(question=question, contexts=ctx_text, answer=answer))
        metrics[f"correctness_{judge.name}"] = _scaled(
            judge, CORRECTNESS_PROMPT.format(question=question, gold=gold_text, answer=answer))
        if require_explanation:
            metrics[f"explanation_quality_{judge.name}"] = _scaled(
                judge, EXPLANATION_PROMPT.format(question=question, answer=answer))
    return metrics


def _binary(judge: Judge, prompt: str) -> float:
    if judge.llm is None:
        return float("nan")
    value = llm_io.parse_json(llm_io.complete(judge.llm, prompt), {})
    if isinstance(value, dict) and "passing" in value:
        return 1.0 if value["passing"] else 0.0
    return float("nan")


def _scaled(judge: Judge, prompt: str) -> float:
    """Parse a 1–5 score and normalise to [0,1] as ``(score-1)/4``."""
    if judge.llm is None:
        return float("nan")
    value = llm_io.parse_json(llm_io.complete(judge.llm, prompt), {})
    if isinstance(value, dict) and "score" in value:
        try:
            score = float(value["score"])
        except (TypeError, ValueError):
            return float("nan")
        return max(0.0, min(1.0, (score - 1.0) / 4.0))
    return float("nan")

"""Deterministic per-question metrics over a ranked recommendation list.

Every baseline rung is scored the same way: it produces a ranked list of recipe
ids (``recipe_url``), and these functions turn that list + the persona + the gold
answer set into the metric block of a §13.1 per-question record.

Three families, all rule-based (no LLM):

* retrieval — :func:`hit_at_k`, :func:`mrr_at_k`, :func:`ndcg_at_k` (§4.1)
* health    — :func:`avg_score_at_k` over ``nutri_score`` / ``fsa_score`` (§4.2)
* safety    — :func:`violation_rate` / :func:`satisfaction_rate` (§4.3)

Recommendations that do not resolve to a real recipe (LLM hallucinations) are
passed as ``None`` rows: they count as retrieval misses and are reported via
``hallucination_rate`` rather than silently dropped.
"""
from __future__ import annotations

from math import log2
from typing import Callable, Optional, Sequence

from foodrag.baselines import constraints

# A resolved recipe row is a Mapping (dict or pd.Series); None = unresolved id.
Row = Optional[dict]


# ── retrieval (§4.1) ─────────────────────────────────────────────────────────

def hit_at_k(retrieved_ids: Sequence[str], gold: set[str], k: int) -> float:
    return 1.0 if any(i in gold for i in retrieved_ids[:k]) else 0.0


def mrr_at_k(retrieved_ids: Sequence[str], gold: set[str], k: int) -> float:
    for rank, rid in enumerate(retrieved_ids[:k], start=1):
        if rid in gold:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved_ids: Sequence[str], gold: set[str], k: int) -> float:
    """Binary-relevance nDCG@k (gain 1 for a gold recipe, 0 otherwise)."""
    dcg = sum(1.0 / log2(rank + 1)
              for rank, rid in enumerate(retrieved_ids[:k], start=1) if rid in gold)
    ideal_hits = min(k, len(gold))
    if ideal_hits == 0:
        return 0.0
    idcg = sum(1.0 / log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg


# ── health (§4.2) ────────────────────────────────────────────────────────────

def avg_score_at_k(rows: Sequence[Row], col: str, k: int) -> float:
    """Mean of ``col`` over the resolved recipes in the top-k (NaN if none)."""
    vals = [r[col] for r in rows[:k] if r is not None and r.get(col) is not None]
    return sum(vals) / len(vals) if vals else float("nan")


# ── safety (§4.3) ────────────────────────────────────────────────────────────

def violation_rate(rows: Sequence[Row], predicate: Callable[[dict], bool], k: int) -> float:
    """Fraction of the top-k recommendations for which ``predicate`` is True.

    Denominator is the number of recommendations actually returned (capped at k),
    so an unresolved (hallucinated) recipe is counted as a recommendation that
    could not be cleared — i.e. it does not get a free pass.
    """
    top = rows[:k]
    if not top:
        return 0.0
    flagged = sum(1 for r in top if r is None or predicate(r))
    return flagged / len(top)


def satisfaction_rate(rows: Sequence[Row], persona: dict,
                      expand: constraints.IngredientExpander, k: int) -> float:
    """Fraction of the top-k recommendations satisfying ALL requirements."""
    top = rows[:k]
    if not top:
        return 0.0
    ok = sum(1 for r in top if r is not None and constraints.satisfies_all(r, persona, expand))
    return ok / len(top)


def hallucination_rate(rows: Sequence[Row], k: int) -> float:
    """Fraction of the top-k recommendations that don't resolve to a real recipe."""
    top = rows[:k]
    if not top:
        return 0.0
    return sum(1 for r in top if r is None) / len(top)


# ── one-shot block for a §13.1 record ────────────────────────────────────────

def deterministic_metrics(retrieved_ids: Sequence[str], rows: Sequence[Row],
                          gold: set[str], persona: dict, k: int,
                          expand: constraints.IngredientExpander = constraints.identity_expand) -> dict:
    """Compute the full deterministic metric block for one question.

    ``rows`` must be aligned 1:1 with ``retrieved_ids`` (``None`` where an id did
    not resolve to a corpus recipe).
    """
    return {
        f"hit_at_{k}": hit_at_k(retrieved_ids, gold, k),
        f"mrr_at_{k}": mrr_at_k(retrieved_ids, gold, k),
        f"ndcg_at_{k}": ndcg_at_k(retrieved_ids, gold, k),
        f"avg_nutri_at_{k}": avg_score_at_k(rows, "nutri_score", k),
        f"avg_fsa_at_{k}": avg_score_at_k(rows, "fsa_score", k),
        "allergen_violation": violation_rate(
            rows, lambda r: constraints.violates_allergen(r, persona, expand), k),
        "hard_constraint_violation": violation_rate(
            rows, lambda r: constraints.violates_hard_nutrient(r, persona), k),
        "constraint_satisfaction": satisfaction_rate(rows, persona, expand, k),
        "hallucination_rate": hallucination_rate(rows, k),
    }

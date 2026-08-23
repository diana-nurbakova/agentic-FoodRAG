"""Deterministic per-question metrics over a ranked recommendation list.

Every baseline rung is scored the same way: it produces a ranked list of recipe
ids (``recipe_url``), and these functions turn that list + the persona + the gold
answer set into the metric block of a per-question record.

Three families, all rule-based (no LLM):

* retrieval — :func:`hit_at_k`, :func:`mrr_at_k`, :func:`ndcg_at_k`
* health    — :func:`avg_score_at_k` over ``nutri_score`` / ``fsa_score``
* safety    — :func:`violation_rate` / :func:`satisfaction_rate`

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


# ── retrieval ────────────────────────────────────────────────────────────────

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


# ── health ───────────────────────────────────────────────────────────────────

def avg_score_at_k(rows: Sequence[Row], col: str, k: int) -> float:
    """Mean of ``col`` over the resolved recipes in the top-k (NaN if none)."""
    vals = [r[col] for r in rows[:k] if r is not None and r.get(col) is not None]
    return sum(vals) / len(vals) if vals else float("nan")


# ── safety ───────────────────────────────────────────────────────────────────

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


# ── hard/soft per-constraint satisfaction ────────────────────────────────────

def hard_constraint_sat_perconstraint(rows: Sequence[Row], persona: dict,
                                      expand: constraints.IngredientExpander, k: int) -> float:
    """Mean over the top-k recs of (satisfied hard constraints / total hard constraints).

    An unresolved (``None``) rec satisfies zero constraints. Undefined (NaN) when the
    persona states no hard constraints — consistent with :func:`avg_score_at_k`.
    """
    preds = constraints.hard_constraints(persona, expand)
    if not preds:
        return float("nan")
    top = rows[:k]
    if not top:
        return float("nan")
    fracs = []
    for r in top:
        if r is None:
            fracs.append(0.0)
        else:
            fracs.append(sum(1 for p in preds if p(r)) / len(preds))
    return sum(fracs) / len(fracs)


def hard_constraint_sat_allornothing(rows: Sequence[Row], persona: dict,
                                     expand: constraints.IngredientExpander, k: int) -> float:
    """Fraction of the top-k recs satisfying ALL hard constraints.

    An unresolved (``None``) rec fails. Undefined (NaN) when the persona states no
    hard constraints.
    """
    preds = constraints.hard_constraints(persona, expand)
    if not preds:
        return float("nan")
    top = rows[:k]
    if not top:
        return float("nan")
    ok = sum(1 for r in top if r is not None and all(p(r) for p in preds))
    return ok / len(top)


def soft_constraint_sat(rows: Sequence[Row], persona: dict,
                        expand: constraints.IngredientExpander, k: int) -> float:
    """Mean over the top-k recs of (satisfied soft constraints / total soft constraints).

    An unresolved (``None``) rec satisfies zero. Undefined (NaN) when the persona
    states no soft constraints.
    """
    preds = constraints.soft_constraints(persona, expand)
    if not preds:
        return float("nan")
    top = rows[:k]
    if not top:
        return float("nan")
    fracs = []
    for r in top:
        if r is None:
            fracs.append(0.0)
        else:
            fracs.append(sum(1 for p in preds if p(r)) / len(preds))
    return sum(fracs) / len(fracs)


# ── one-shot block for a per-question record ──────────────────────────────────

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
        # relevance axis for the Health-vs-Relevance Pareto frontier (NaN if none resolve).
        f"avg_rating_at_{k}": avg_score_at_k(rows, "average_rating", k),
        "allergen_violation": violation_rate(
            rows, lambda r: constraints.violates_allergen(r, persona, expand), k),
        "hard_constraint_violation": violation_rate(
            rows, lambda r: constraints.violates_hard_nutrient(r, persona), k),
        "constraint_satisfaction": satisfaction_rate(rows, persona, expand, k),
        "hallucination_rate": hallucination_rate(rows, k),
        # hard/soft split (per-constraint & all-or-nothing; NaN if no such constraints)
        "hard_constraint_sat_perconstraint": hard_constraint_sat_perconstraint(rows, persona, expand, k),
        "hard_constraint_sat_allornothing": hard_constraint_sat_allornothing(rows, persona, expand, k),
        "soft_constraint_sat": soft_constraint_sat(rows, persona, expand, k),
    }

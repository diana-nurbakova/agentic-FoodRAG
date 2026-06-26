"""Common contract + health-aware ranking for the external KBQA baselines.

All three systems (ToG, RoG, TFS-KBQA) reduce to one operation — turn a QA item
into a ranked list of ``recipe_url`` ids — so they share a single
:class:`KBQASystem` interface. The dual-use adapters (§5.7) wrap *that* interface
twice: once as a ladder :class:`~foodrag.baselines.base.Baseline` (Use 1
retrieval, EXP-1; Use 2 full-pipeline, EXP-7) and once as a
``DataFrameRetrieverBase`` for FoodRAG's meal-plan assembly layer.

Ranking is factored out here because the fairness principle (§5.7) requires every
baseline to receive the *same* health information FoodRAG gets: whatever evidence
score a system assigns to a candidate, the final tie-break is health-aware
(NutriScore, then FSA, then popularity), so a baseline is never silently
under-equipped on healthiness.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

from foodrag.baselines.kbqa.kg import HummusKG


class KBQASystem(ABC):
    """A KBQA baseline: QA item → ranked ``recipe_url`` list.

    ``name`` is the stable config key used in result files (``tog`` / ``rog`` /
    ``tfs_kbqa``); the dual-use adapters read it.
    """

    name: str = ""

    @abstractmethod
    def retrieve(self, qa_item: dict, k: int) -> list[str]:
        """Return up to ``k`` ranked ``recipe_url`` ids for the QA item."""
        raise NotImplementedError


@dataclass
class KBQAOptions:
    """Shared knobs for the KBQA traversal/tractability behaviour (§5.5)."""

    max_depth: int = 3          # capped traversal depth (recipe↔ingredient↔class)
    beam_width: int = 5         # ToG beam / RoG plan count kept per step
    fanout: int = 50            # tail entities kept per (entity, relation) hop
    constraint_first: bool = False  # opt-in: traverse only within the valid set
    health_alpha: float = 0.25  # weight of health score in the final blend
    query_field: str = "question"


# ── health-aware ranking (§5.7 fairness, shared by all systems) ───────────────

def health_sort_key(attrs: dict) -> tuple:
    """Descending NutriScore, then FSA, then rating, then #ratings (NaN-safe)."""
    def num(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return float("-inf")
    return (num(attrs.get("nutri_score")), num(attrs.get("fsa_score")),
            num(attrs.get("average_rating")), num(attrs.get("number_of_ratings")))


def blend_order(scored: dict[object, float], kg: HummusKG, *, alpha: float = 0.25,
                key_to_url: Optional[dict] = None) -> list:
    """Order recipe keys by ``(1-alpha)·evidence + alpha·health`` (both min-max
    normalised across the candidate set), tie-broken by the health sort key.

    ``scored`` maps a recipe key to the system's evidence score (path score,
    reasoning confidence, …). Returns keys (or urls, if ``key_to_url`` given) best
    first. This is the single place the §5.7 "same health info" promise is kept.
    """
    if not scored:
        return []
    keys = list(scored)
    attrs = kg.recipe_attributes(keys)

    ev = _minmax({k: scored[k] for k in keys})
    he = _minmax({k: _health_scalar(attrs.get(k, {})) for k in keys})
    blended = {k: (1 - alpha) * ev[k] + alpha * he[k] for k in keys}

    ordered = sorted(keys, key=lambda k: (blended[k], health_sort_key(attrs.get(k, {}))),
                     reverse=True)
    if key_to_url is not None:
        out, seen = [], set()
        for k in ordered:
            url = key_to_url.get(k)
            if url and url not in seen:
                seen.add(url)
                out.append(url)
        return out
    return ordered


def _health_scalar(attrs: dict) -> float:
    try:
        return float(attrs.get("nutri_score"))
    except (TypeError, ValueError):
        return 0.0


def _minmax(values: dict[object, float]) -> dict[object, float]:
    nums = [v for v in values.values()]
    lo, hi = min(nums), max(nums)
    if hi <= lo:
        return {k: 1.0 for k in values}
    return {k: (v - lo) / (hi - lo) for k, v in values.items()}

"""Popularity baselines — ladder rungs 0a and 0b (§5.7).

* **0a — Popularity (raw):** top-rated recipes, *no* constraint filtering. The
  deliberately health-unaware control; it should score badly on the safety
  metrics (high allergen-violation, low constraint-satisfaction), quantifying
  what a naive "most popular recipe" app gets wrong.
* **0b — Popularity (health-aware):** the same rating-based ranker applied only
  to recipes that pass the persona's hard constraints — the baseline real recipe
  apps actually field.

Both rank by ``average_rating`` with ``number_of_ratings`` as the tie-break, so a
single 5-star vote cannot outrank a well-reviewed 4.9.
"""
from __future__ import annotations

import pandas as pd

from foodrag.baselines import constraints
from foodrag.baselines.base import Baseline

ID_COL = "recipe_url"
RANK_COLS = ["average_rating", "number_of_ratings"]


def _ranked_ids(df: pd.DataFrame, k: int) -> list[str]:
    if df.empty:
        return []
    ranked = df.sort_values(RANK_COLS, ascending=False)
    return ranked[ID_COL].head(k).tolist()


class PopularityRaw(Baseline):
    """Rung 0a: globally top-rated recipes, ignoring the persona entirely."""

    rung = "0a"
    config_id = "popularity_raw"

    def __init__(self, recipes_df: pd.DataFrame):
        # Persona-independent: the ranking is identical for every question, so
        # precompute a generous prefix once and slice per call.
        self._ranked = recipes_df.sort_values(RANK_COLS, ascending=False)[ID_COL].tolist()

    def recommend(self, qa_item: dict, k: int) -> list[str]:
        return self._ranked[:k]


class PopularityHealthAware(Baseline):
    """Rung 0b: top-rated recipes among those satisfying the persona's hard
    constraints."""

    rung = "0b"
    config_id = "popularity_health_aware"

    def __init__(self, recipes_df: pd.DataFrame,
                 expand: constraints.IngredientExpander = constraints.identity_expand):
        self._recipes_df = recipes_df
        self._expand = expand

    def recommend(self, qa_item: dict, k: int) -> list[str]:
        valid = constraints.filter_valid(qa_item["persona"], self._recipes_df, self._expand)
        return _ranked_ids(valid, k)

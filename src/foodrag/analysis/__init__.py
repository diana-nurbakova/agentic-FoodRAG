"""Offline analysis over pre-computed FoodRAG evaluation results.

Memory-light, read-only helpers that operate on the aggregate long-format result
tables, NOT on HUMMUS or any live pipeline. The
Pareto trade-off analysis lives here.
"""

from .pareto import (
    ParetoResult,
    pareto_frontier,
    health_vs_relevance,
    quality_vs_cost,
    safety_vs_coverage,
)

__all__ = [
    "ParetoResult",
    "pareto_frontier",
    "health_vs_relevance",
    "quality_vs_cost",
    "safety_vs_coverage",
]

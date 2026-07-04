"""Offline analysis over pre-computed FoodRAG evaluation results.

Memory-light, read-only helpers that operate on the aggregate long-format result
tables (Evaluation Protocol §13.2), NOT on HUMMUS or any live pipeline. The
Pareto trade-off analysis (§9.4) lives here.
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

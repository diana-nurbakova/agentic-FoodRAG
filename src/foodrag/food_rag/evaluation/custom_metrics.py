from llama_index.core.evaluation.retrieval.metrics_base import BaseRetrievalMetric
from typing import Dict, Type

from foodrag.food_rag.evaluation import retrieval_metrics
from foodrag.food_rag.evaluation import recipe_scores

METRIC_REGISTRY: Dict[str, Type[BaseRetrievalMetric]] = {
    "precision": retrieval_metrics.PrecisionMetric,
    "recall": retrieval_metrics.RecallMetric,
    "f1": retrieval_metrics.F1Metric,
    "who_score": recipe_scores.WHOScoreMetric,
    "fsa_score": recipe_scores.FSAScoreMetric,
    "nutri_score": recipe_scores.NUTRIScoreMetric,
    "who_delta_score": recipe_scores.WHODeltaScoreMetric,
    "fsa_delta_score": recipe_scores.FSADeltaScoreMetric,
    "nutri_delta_score": recipe_scores.NUTRIDeltaScoreMetric,
}

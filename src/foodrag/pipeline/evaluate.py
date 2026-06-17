"""Stage 7: evaluate a retriever against the QA set.

Builds a DataFrameRetrieverEvaluator for the given retriever and runs it over a
loaded QA set, writing batched results under ``<data_path>/runs/<run>/``. The
full 10-retriever sweep (and the legacy Nebula text-to-Cypher retriever) is
assembled in ``notebooks/eval_1.ipynb``; here a single retriever is evaluated
(pass it in), which is the reusable unit of that sweep.

    evaluate.run(qa_set, retriever, recipes_df, data_path="data", run_name="eval_1")
"""
from __future__ import annotations

from typing import Any, Optional

DEFAULT_METRICS = ["hit_rate", "mrr", "precision", "recall", "f1"]


def build_evaluator(retriever: Any, recipes_df: Any = None, metric_names: Optional[list[str]] = None) -> Any:
    """Construct a DataFrameRetrieverEvaluator from a retriever + metric names."""
    from foodrag.food_rag.evaluation.retriever_evaluator import DataFrameRetrieverEvaluator

    evaluator = DataFrameRetrieverEvaluator.from_metric_names(
        metric_names or DEFAULT_METRICS, retriever=retriever
    )
    if recipes_df is not None:
        evaluator.recipes_df = recipes_df
    return evaluator


def run(
    qa_set: dict,
    retriever: Any = None,
    *,
    recipes_df: Any = None,
    data_path: str = "data",
    run_name: str = "eval_1",
    batch_size: int = 10,
    metric_names: Optional[list[str]] = None,
    evaluator: Any = None,
    show_progress: bool = True,
) -> Any:
    """Evaluate ``retriever`` on ``qa_set``; return the evaluation results.

    Either pass a prebuilt ``evaluator`` or a ``retriever`` (+ optional
    ``recipes_df``) to build one. Delegates to ``evaluate_dataset`` which
    checkpoints batches under runs/<run_name>/.
    """
    if evaluator is None:
        if retriever is None:
            raise ValueError("Provide either an `evaluator` or a `retriever`.")
        evaluator = build_evaluator(retriever, recipes_df, metric_names)
    return evaluator.evaluate_dataset(
        qa_set, data_path, run_name, batch_size=batch_size, show_progress=show_progress
    )

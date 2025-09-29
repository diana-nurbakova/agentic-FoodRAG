import numpy as np
from llama_index.core.evaluation.retrieval.metrics_base import BaseRetrievalMetric, RetrievalMetricResult
from sklearn.metrics import recall_score, f1_score, precision_score
from typing import Any, List, Optional


class PrecisionMetric(BaseRetrievalMetric):
    """Compute the Precision (P) for retrieval evaluation."""

    metric_name = "precision"

    def compute(
            self,
            query: Optional[str] = None,
            expected_ids: Optional[List[str]] = None,
            retrieved_ids: Optional[List[str]] = None,
            expected_texts: Optional[List[str]] = None,
            retrieved_texts: Optional[List[str]] = None,
            **kwargs: Any,
    ) -> RetrievalMetricResult:
        if expected_ids and retrieved_ids:
            y_true = [1 if doc in expected_ids else 0 for doc in retrieved_ids]
            p = precision_score(y_true, [1] * len(y_true), zero_division=np.nan)

            return RetrievalMetricResult(
                score=p,
                metadata={"metric_name": self.metric_name, "query": query, "expected_ids": expected_ids,
                          "retrieved_ids": retrieved_ids},
            )
        else:
            return RetrievalMetricResult(
                score=float("nan"),
                metadata={"metric_name": self.metric_name, "query": query, "expected_ids": expected_ids,
                          "retrieved_ids": retrieved_ids, "details": "No expected documents."},
            )


class RecallMetric(BaseRetrievalMetric):
    """Compute the Recall for retrieval evaluation."""

    metric_name = "recall"

    def compute(
            self,
            query: Optional[str] = None,
            expected_ids: Optional[List[str]] = None,
            retrieved_ids: Optional[List[str]] = None,
            expected_texts: Optional[List[str]] = None,
            retrieved_texts: Optional[List[str]] = None,
            **kwargs: Any,
    ) -> RetrievalMetricResult:
        if expected_ids:
            y_true = [1 if doc in expected_ids else 0 for doc in retrieved_ids]
            recall = recall_score(y_true, [1] * len(retrieved_ids), zero_division=np.nan)

            return RetrievalMetricResult(
                score=recall,
                metadata={"metric_name": self.metric_name, "query": query, "expected_ids": expected_ids,
                          "retrieved_ids": retrieved_ids},
            )
        else:
            return RetrievalMetricResult(
                score=float("nan"),
                metadata={"metric_name": self.metric_name, "query": query, "expected_ids": expected_ids,
                          "retrieved_ids": retrieved_ids, "details": "No expected documents."},
            )


class F1Metric(BaseRetrievalMetric):
    """Compute the F1 Score for retrieval evaluation."""

    metric_name = "f1"

    def compute(
            self,
            query: Optional[str] = None,
            expected_ids: Optional[List[str]] = None,
            retrieved_ids: Optional[List[str]] = None,
            expected_texts: Optional[List[str]] = None,
            retrieved_texts: Optional[List[str]] = None,
            **kwargs: Any,
    ) -> RetrievalMetricResult:
        if expected_ids:
            y_true = [1 if doc in expected_ids else 0 for doc in retrieved_ids]
            f1 = f1_score(y_true, [1] * len(retrieved_ids), zero_division=np.nan)

            return RetrievalMetricResult(
                score=f1,
                metadata={"metric_name": self.metric_name, "query": query, "expected_ids": expected_ids,
                          "retrieved_ids": retrieved_ids},
            )
        else:
            return RetrievalMetricResult(
                score=float("nan"),
                metadata={"metric_name": self.metric_name, "query": query, "expected_ids": expected_ids,
                          "retrieved_ids": retrieved_ids, "details": "No expected documents."},
            )

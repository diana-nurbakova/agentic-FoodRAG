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
            # recall@k = (relevant items retrieved) / (total relevant items).
            # NOTE: the original used recall_score(y_true, [1]*len(retrieved)),
            # which makes FN=0 and recall always 1.0 -- fixed here.
            expected_set = set(expected_ids)
            relevant_retrieved = len(expected_set & set(retrieved_ids or []))
            recall = relevant_retrieved / len(expected_set) if expected_set else float("nan")

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
        if expected_ids and retrieved_ids:
            # f1 = harmonic mean of precision@k and recall@k (both fixed as above).
            expected_set = set(expected_ids)
            retrieved = list(retrieved_ids)
            relevant_in_retrieved = sum(1 for d in retrieved if d in expected_set)
            precision = relevant_in_retrieved / len(retrieved) if retrieved else 0.0
            recall = len(expected_set & set(retrieved)) / len(expected_set) if expected_set else 0.0
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

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

"""Unit tests for retrieval metrics (precision@k, recall@k, f1) + registry."""
from __future__ import annotations

import math

import pytest

from foodrag.food_rag.evaluation.retrieval_metrics import F1Metric, PrecisionMetric, RecallMetric

pytestmark = pytest.mark.unit


def _score(metric, expected, retrieved):
    return metric().compute(expected_ids=expected, retrieved_ids=retrieved).score


def test_precision_at_k():
    assert _score(PrecisionMetric, ["a", "b", "c"], ["a", "x", "b", "y"]) == 0.5
    assert _score(PrecisionMetric, ["a", "b"], ["a", "b", "c"]) == pytest.approx(2 / 3)
    assert _score(PrecisionMetric, ["a"], ["x", "y"]) == 0.0


def test_recall_at_k_not_always_one():
    # the original bug returned 1.0 here; correct recall@k is 2/3
    assert _score(RecallMetric, ["a", "b", "c"], ["a", "x", "b", "y"]) == pytest.approx(2 / 3)
    assert _score(RecallMetric, ["a", "b"], ["a", "b", "c"]) == 1.0  # all relevant retrieved
    assert _score(RecallMetric, ["a"], ["x", "y"]) == 0.0


def test_f1_is_harmonic_mean():
    # p=0.5, r=2/3 -> f1 = 2pr/(p+r) = 0.5714...
    assert _score(F1Metric, ["a", "b", "c"], ["a", "x", "b", "y"]) == pytest.approx(0.5714285, abs=1e-5)
    # p=2/3, r=1 -> f1 = 0.8
    assert _score(F1Metric, ["a", "b"], ["a", "b", "c"]) == pytest.approx(0.8)
    assert _score(F1Metric, ["a"], ["x", "y"]) == 0.0


def test_no_expected_returns_nan():
    assert math.isnan(_score(PrecisionMetric, [], ["a"]))
    assert math.isnan(_score(RecallMetric, [], ["a"]))


def test_metric_registry_complete():
    from foodrag.food_rag.evaluation.custom_metrics import METRIC_REGISTRY

    assert {"precision", "recall", "f1"} <= set(METRIC_REGISTRY)
    assert {"who_score", "fsa_score", "nutri_score"} <= set(METRIC_REGISTRY)
    # every registered metric is a class with a compute method
    for name, cls in METRIC_REGISTRY.items():
        assert hasattr(cls, "compute"), name

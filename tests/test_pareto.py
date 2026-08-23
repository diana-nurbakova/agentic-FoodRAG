"""Tests for foodrag.analysis.pareto.

Covers the generic ``pareto_frontier`` (hand-checked 2D sets, single point,
all-dominated-by-one, ties, a min-axis case, NaN exclusion) and the three
frontier builders against small synthetic aggregate frames.

Run ONLY this file:  pytest tests/test_pareto.py -q
"""

import math

import pandas as pd
import pytest

from foodrag.analysis.pareto import (
    ParetoResult,
    health_vs_relevance,
    pareto_frontier,
    quality_vs_cost,
    safety_vs_coverage,
)

pytestmark = [pytest.mark.unit]


# ── generic pareto_frontier ────────────────────────────────────────────────
def test_known_2d_both_max():
    # A(0.9,0.3) B(0.5,0.4) C(0.6,0.8): C dominates B; A and C non-dominated.
    pts = {"A": {"x": 0.9, "y": 0.3}, "B": {"x": 0.5, "y": 0.4}, "C": {"x": 0.6, "y": 0.8}}
    r = pareto_frontier(pts, {"x": "max", "y": "max"})
    assert r.frontier == ["A", "C"]
    assert r.dominated_by["B"] == ["C"]
    assert "B" in r.dominates["C"]
    assert r.dominates["A"] == []  # A beats no one (B has higher y)
    assert r.excluded == {}


def test_single_point_is_frontier():
    r = pareto_frontier({"only": {"x": 0.1, "y": 0.9}}, {"x": "max", "y": "max"})
    assert r.frontier == ["only"]
    assert r.dominated_by["only"] == []
    assert r.dominates["only"] == []


def test_all_dominated_by_one():
    # BEST dominates every other point on both axes.
    pts = {
        "BEST": {"x": 1.0, "y": 1.0},
        "p1": {"x": 0.4, "y": 0.4},
        "p2": {"x": 0.5, "y": 0.2},
        "p3": {"x": 0.1, "y": 0.9},
    }
    r = pareto_frontier(pts, {"x": "max", "y": "max"})
    assert r.frontier == ["BEST"]
    assert sorted(r.dominates["BEST"]) == ["p1", "p2", "p3"]
    for p in ("p1", "p2", "p3"):
        assert r.dominated_by[p] == ["BEST"]


def test_ties_all_equal_non_dominated():
    # Identical points never strictly dominate each other → all on the frontier.
    pts = {"a": {"x": 0.5, "y": 0.5}, "b": {"x": 0.5, "y": 0.5}, "c": {"x": 0.5, "y": 0.5}}
    r = pareto_frontier(pts, {"x": "max", "y": "max"})
    assert r.frontier == ["a", "b", "c"]
    assert all(r.dominates[k] == [] for k in pts)
    assert all(r.dominated_by[k] == [] for k in pts)


def test_tie_on_one_axis_strict_on_other():
    # equal x, but a has strictly higher y → a dominates b.
    pts = {"a": {"x": 0.5, "y": 0.8}, "b": {"x": 0.5, "y": 0.5}}
    r = pareto_frontier(pts, {"x": "max", "y": "max"})
    assert r.frontier == ["a"]
    assert r.dominates["a"] == ["b"]
    assert r.dominated_by["b"] == ["a"]


def test_min_axis_case():
    # x = cost (min better), y = quality (max better).
    pts = {
        "cheap_good": {"cost": 2, "quality": 0.8},
        "pricey_good": {"cost": 10, "quality": 0.85},
        "cheap_bad": {"cost": 2, "quality": 0.5},
    }
    r = pareto_frontier(pts, {"cost": "min", "quality": "max"})
    # cheap_good non-dominated; pricey_good non-dominated (best quality);
    # cheap_bad dominated by cheap_good (same cost, worse quality).
    assert r.frontier == ["cheap_good", "pricey_good"]
    assert r.dominated_by["cheap_bad"] == ["cheap_good"]
    assert "cheap_bad" in r.dominates["cheap_good"]


def test_min_axis_domination_direction():
    # Lower cost + equal quality should dominate on a min axis.
    pts = {"lo": {"cost": 1, "q": 0.5}, "hi": {"cost": 5, "q": 0.5}}
    r = pareto_frontier(pts, {"cost": "min", "q": "max"})
    assert r.frontier == ["lo"]
    assert r.dominates["lo"] == ["hi"]


def test_nan_exclusion_recorded():
    pts = {
        "ok": {"x": 0.5, "y": 0.5},
        "nan_y": {"x": 0.9, "y": float("nan")},
        "missing_x": {"y": 0.9},
        "none_x": {"x": None, "y": 0.2},
    }
    r = pareto_frontier(pts, {"x": "max", "y": "max"})
    assert set(r.points) == {"ok"}
    assert r.frontier == ["ok"]
    assert set(r.excluded) == {"nan_y", "missing_x", "none_x"}
    assert "y" in r.excluded["nan_y"]
    assert "x" in r.excluded["missing_x"]
    assert "x" in r.excluded["none_x"]


def test_bad_direction_raises():
    with pytest.raises(ValueError):
        pareto_frontier({"a": {"x": 1}}, {"x": "up"})


def test_empty_objectives_raises():
    with pytest.raises(ValueError):
        pareto_frontier({"a": {"x": 1}}, {})


def test_as_records_shape():
    pts = {"A": {"x": 0.9, "y": 0.3}, "C": {"x": 0.6, "y": 0.8}}
    r = pareto_frontier(pts, {"x": "max", "y": "max"})
    recs = {rec["config_id"]: rec for rec in r.as_records()}
    assert recs["A"]["frontier"] is True and recs["C"]["frontier"] is True
    assert recs["A"]["x"] == 0.9 and recs["C"]["y"] == 0.8
    assert "dominated_by" in recs["A"] and "dominates" in recs["A"]


# ── synthetic aggregate frames for the builders ────────────────────────────
def _row(exp, cfg, stratum, metric, mean):
    return {"experiment_id": exp, "config_id": cfg, "stratum": stratum,
            "metric": metric, "mean": mean}


def _agg(rows):
    return pd.DataFrame(rows)


def test_health_vs_relevance_ndcg_proxy():
    # No avg_rating_at_10 → falls back to ndcg_at_10 as the relevance proxy.
    rows = []
    data = {  # config: (ndcg, nutri)
        "text-to-pandas": (0.64, 0.60),
        "fusion": (0.58, 0.63),
        "sparse": (0.44, 0.58),
    }
    for cfg, (ndcg, nutri) in data.items():
        rows += [_row("EXP-1", cfg, "aggregate", "ndcg_at_10", ndcg),
                 _row("EXP-1", cfg, "aggregate", "avg_nutri_at_10", nutri)]
    r = health_vs_relevance(_agg(rows), experiment_id="EXP-1")
    assert r.axis_labels["relevance"] == "ndcg_at_10"
    # fusion (best nutri) and text-to-pandas (best ndcg) non-dominated; sparse dominated.
    assert set(r.frontier) == {"fusion", "text-to-pandas"}
    assert r.dominated_by["sparse"]  # someone dominates sparse


def test_health_vs_relevance_prefers_rating_when_present():
    rows = []
    for cfg, rating, nutri in [("a", 0.9, 0.3), ("b", 0.2, 0.9)]:
        rows += [_row("EXP-1", cfg, "aggregate", "avg_rating_at_10", rating),
                 _row("EXP-1", cfg, "aggregate", "relevancy_judgeA", 0.5),
                 _row("EXP-1", cfg, "aggregate", "avg_nutri_at_10", nutri)]
    r = health_vs_relevance(_agg(rows), experiment_id="EXP-1")
    assert r.axis_labels["relevance"] == "avg_rating_at_10"
    assert set(r.frontier) == {"a", "b"}  # classic health/relevance trade-off


def test_health_vs_relevance_fsa_variant():
    rows = [
        _row("EXP-1", "x", "aggregate", "ndcg_at_10", 0.5),
        _row("EXP-1", "x", "aggregate", "avg_fsa_at_10", 0.7),
    ]
    r = health_vs_relevance(_agg(rows), experiment_id="EXP-1", health_metric="avg_fsa_at_10")
    assert r.axis_labels["health"] == "avg_fsa_at_10"
    assert r.frontier == ["x"]


def test_quality_vs_cost_exp3():
    rows = []
    data = {  # config: (correctness, llm_usage)
        "rag": (0.476, 4),
        "heuristic_2": (0.512, 6),
        "heuristic_1": (0.572, 12),
    }
    for cfg, (corr, cost) in data.items():
        rows += [_row("EXP-3", cfg, "aggregate", "correctness", corr),
                 _row("EXP-3", cfg, "aggregate", "llm_usage", cost)]
    r = quality_vs_cost(_agg(rows), experiment_id="EXP-3")
    assert r.axis_labels == {"cost": "llm_usage", "quality": "correctness"}
    # Each is cheapest at its own correctness level → all three non-dominated.
    assert set(r.frontier) == {"rag", "heuristic_2", "heuristic_1"}


def test_quality_vs_cost_dominated_when_pricier_and_worse():
    rows = [
        _row("EXP-3", "good", "aggregate", "correctness", 0.6),
        _row("EXP-3", "good", "aggregate", "llm_usage", 4),
        _row("EXP-3", "bad", "aggregate", "correctness", 0.5),   # worse quality
        _row("EXP-3", "bad", "aggregate", "llm_usage", 8),        # and pricier
    ]
    r = quality_vs_cost(_agg(rows), experiment_id="EXP-3")
    assert r.frontier == ["good"]
    assert r.dominated_by["bad"] == ["good"]


def test_quality_vs_cost_scopes_metric_to_experiment():
    # EXP-7 stores correctness_judgeA; a global scan would mispick it for EXP-3.
    rows = [
        _row("EXP-3", "rag", "aggregate", "correctness", 0.5),
        _row("EXP-3", "rag", "aggregate", "llm_usage", 4),
        _row("EXP-7", "foodrag_h2", "aggregate", "correctness_judgeA", 0.25),
        _row("EXP-7", "foodrag_h2", "aggregate", "llm_usage", 3),
    ]
    agg = _agg(rows)
    r3 = quality_vs_cost(agg, experiment_id="EXP-3")
    assert r3.axis_labels["quality"] == "correctness"
    assert r3.frontier == ["rag"]
    r7 = quality_vs_cost(agg, experiment_id="EXP-7")
    assert r7.axis_labels["quality"] == "correctness_judgeA"


def test_quality_vs_cost_missing_returns_empty():
    rows = [_row("EXP-3", "rag", "aggregate", "llm_usage", 4)]  # no quality metric
    r = quality_vs_cost(_agg(rows), experiment_id="EXP-3")
    assert r.points == {} and r.frontier == []


def test_safety_vs_coverage_stratum_proxy():
    # Coverage rises as constraints relax (broad > tight) → clean trade-off,
    # all three strata non-dominated (strictness vs coverage trade).
    rows = []
    cov = {"tight": 0.50, "medium": 0.55, "broad": 0.62}
    for stratum, c in cov.items():
        rows.append(_row("EXP-10k", "pop", stratum, "constraint_satisfaction", c))
    r = safety_vs_coverage(_agg(rows), experiment_id="EXP-10k")
    assert set(r.points) == {"tight", "medium", "broad"}
    assert r.points["tight"]["strictness"] > r.points["broad"]["strictness"]
    assert set(r.frontier) == {"tight", "medium", "broad"}
    assert r.axis_labels["coverage"] == "constraint_satisfaction"


def test_safety_vs_coverage_flat_coverage_ties():
    # Equal coverage across strata → strictest (tight) dominates the equal-coverage
    # lower-strictness strata (tie on coverage, strictly higher strictness).
    rows = [_row("EXP-3", "h1", s, "structural_validity", 0.857)
            for s in ("tight", "medium", "broad")]
    r = safety_vs_coverage(_agg(rows), experiment_id="EXP-3")
    assert r.frontier == ["tight"]
    assert set(r.dominates["tight"]) == {"medium", "broad"}


def test_safety_vs_coverage_no_coverage_metric_empty():
    rows = [_row("EXP-1", "dense", "tight", "ndcg_at_10", 0.5)]
    r = safety_vs_coverage(_agg(rows), experiment_id="EXP-1")
    assert r.points == {} and r.frontier == []

"""Pareto trade-off analysis over evaluation results (Evaluation Protocol §9.4).

Several FoodRAG objectives compete (health vs relevance, quality vs cost, safety
vs coverage). This module reports them as **Pareto frontiers** over configurations
and baselines rather than isolated scalars, and exposes the *domination relation*
explicitly so the UI/paper can state "config X dominates baseline Y".

Design rules honoured from §9.4:

* **Domination:** config *A* dominates *B* iff *A* is at least as good as *B* on
  **every** axis (per that axis's ``"max"``/``"min"`` direction) AND strictly
  better on at least one. The non-dominated set is the frontier.
* **Safety is NEVER a Pareto axis.** Hard constraints gate the *feasible* set;
  frontiers are computed only over feasible outputs. ``safety_vs_coverage`` puts
  *coverage* (fraction of personas with a full feasible plan) on an axis against
  *strictness*, never a raw safety-violation rate.
* **NaN handling:** a config missing (NaN) any required axis is dropped from the
  frontier computation and recorded in ``excluded`` — never silently treated as
  zero (which would fabricate a dominated/dominating point).

The core :func:`pareto_frontier` is generic (points = ``{config_id: {axis: val}}``);
the three frontier builders pull the right metrics from an aggregate long-format
DataFrame (§13.2: columns ``experiment_id, config_id, stratum, metric, mean``) and
delegate to it. This module is pure/stdlib+pandas — no Streamlit, no backend, no
HUMMUS — so it is unit-testable in isolation and memory-light.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Tuple

try:  # pandas is only needed by the frontier builders, not the core algorithm.
    import pandas as pd
except Exception:  # pragma: no cover - pandas always present in this project
    pd = None  # type: ignore


Point = Mapping[str, float]
Objectives = Mapping[str, str]  # axis -> "max" | "min"


@dataclass
class ParetoResult:
    """Outcome of a Pareto analysis.

    Attributes:
        objectives:   axis -> "max"/"min" direction used.
        points:       config_id -> {axis: value} that were actually compared
                      (NaN-excluded configs removed).
        frontier:     sorted list of non-dominated config_ids.
        dominated_by: config_id -> list of config_ids that dominate it (empty
                      for frontier members).
        dominates:    config_id -> list of config_ids it dominates.
        excluded:     config_id -> reason (e.g. "NaN on avg_nutri_at_10") for
                      configs dropped before the comparison.
        axis_labels:  optional axis -> human label for the UI.
    """

    objectives: Objectives
    points: Dict[str, Dict[str, float]]
    frontier: List[str]
    dominated_by: Dict[str, List[str]]
    dominates: Dict[str, List[str]]
    excluded: Dict[str, str] = field(default_factory=dict)
    axis_labels: Dict[str, str] = field(default_factory=dict)

    def is_frontier(self, config_id: str) -> bool:
        return config_id in self.frontier

    def as_records(self) -> List[dict]:
        """Flat rows for plotting: one per compared config with its axis values,
        a ``frontier`` flag and who-dominates-whom lists."""
        rows = []
        frontier = set(self.frontier)
        for cfg, vals in self.points.items():
            row = {"config_id": cfg, "frontier": cfg in frontier}
            row.update(vals)
            row["dominated_by"] = list(self.dominated_by.get(cfg, []))
            row["dominates"] = list(self.dominates.get(cfg, []))
            rows.append(row)
        return rows


def _is_nan(x) -> bool:
    try:
        return x is None or (isinstance(x, float) and math.isnan(x))
    except TypeError:  # pragma: no cover
        return True


def _at_least_as_good(a: float, b: float, direction: str) -> bool:
    return a >= b if direction == "max" else a <= b


def _strictly_better(a: float, b: float, direction: str) -> bool:
    return a > b if direction == "max" else a < b


def _dominates(a: Point, b: Point, objectives: Objectives) -> bool:
    """True iff point ``a`` dominates point ``b`` (§9.4).

    ``a`` must be at least as good as ``b`` on every axis and strictly better on
    at least one. Two identical points do NOT dominate each other (ties).
    """
    better_somewhere = False
    for axis, direction in objectives.items():
        av, bv = a[axis], b[axis]
        if not _at_least_as_good(av, bv, direction):
            return False
        if _strictly_better(av, bv, direction):
            better_somewhere = True
    return better_somewhere


def pareto_frontier(
    points: Mapping[str, Point],
    objectives: Objectives,
    axis_labels: Optional[Mapping[str, str]] = None,
) -> ParetoResult:
    """Compute the non-dominated set + full domination relation.

    Args:
        points:     ``{config_id: {axis: value}}``. Values may be NaN/None; any
                    config missing a *required* axis (an axis in ``objectives``)
                    is excluded and recorded, never coerced to 0.
        objectives: ``{axis: "max" | "min"}`` — the optimisation direction per
                    axis. Determines domination.
        axis_labels: optional ``{axis: label}`` carried through for the UI.

    Returns:
        :class:`ParetoResult`. Handles the single-point case (that point is the
        frontier), all-equal (every point non-dominated — none strictly beats
        another), and ties (equal points never dominate each other).
    """
    if not objectives:
        raise ValueError("objectives must be non-empty")
    for axis, direction in objectives.items():
        if direction not in ("max", "min"):
            raise ValueError(f"objective {axis!r} direction must be 'max'/'min', got {direction!r}")

    kept: Dict[str, Dict[str, float]] = {}
    excluded: Dict[str, str] = {}
    for cfg, vals in points.items():
        missing = [ax for ax in objectives if ax not in vals or _is_nan(vals.get(ax))]
        if missing:
            excluded[cfg] = "NaN on " + ", ".join(sorted(missing))
            continue
        kept[cfg] = {ax: float(vals[ax]) for ax in objectives}

    dominated_by: Dict[str, List[str]] = {c: [] for c in kept}
    dominates: Dict[str, List[str]] = {c: [] for c in kept}
    for a in kept:
        for b in kept:
            if a == b:
                continue
            if _dominates(kept[a], kept[b], objectives):
                dominates[a].append(b)
                dominated_by[b].append(a)

    frontier = sorted(c for c in kept if not dominated_by[c])
    for c in dominates:
        dominates[c].sort()
    for c in dominated_by:
        dominated_by[c].sort()

    return ParetoResult(
        objectives=dict(objectives),
        points=kept,
        frontier=frontier,
        dominated_by=dominated_by,
        dominates=dominates,
        excluded=excluded,
        axis_labels=dict(axis_labels or {}),
    )


# ── aggregate-frame helpers ────────────────────────────────────────────────
def _first_present(available: set, candidates: Tuple[str, ...]) -> Optional[str]:
    for c in candidates:
        if c in available:
            return c
    return None


def _metrics_in_scope(agg, stratum: Optional[str], experiment_id: Optional[str]) -> set:
    """Metrics available within the chosen experiment/stratum slice.

    Candidate-metric selection (e.g. relevance proxy, quality metric) MUST be
    scoped to the slice being plotted — a global scan would pick, say,
    ``correctness_judgeA`` (only in EXP-7) even when the user asked for EXP-3
    (which stores ``correctness``), yielding an all-excluded empty frontier.
    """
    df = agg
    if experiment_id is not None:
        df = df[df["experiment_id"] == experiment_id]
    if stratum is not None:
        df = df[df["stratum"] == stratum]
    return set(df["metric"].unique())


def _collect_axis_values(
    agg,
    metric: str,
    stratum: str = "aggregate",
    experiment_id: Optional[str] = None,
    value_col: str = "mean",
) -> Dict[str, float]:
    """``{config_id: value}`` for one metric at one stratum.

    Aggregates across experiments unless ``experiment_id`` is given. If a config
    appears in multiple experiments for the same metric+stratum, the mean is
    taken (keeps the helper deterministic; real tables rarely collide).
    """
    df = agg[(agg["metric"] == metric) & (agg["stratum"] == stratum)]
    if experiment_id is not None:
        df = df[df["experiment_id"] == experiment_id]
    if df.empty:
        return {}
    grouped = df.groupby("config_id")[value_col].mean()
    return {str(k): float(v) for k, v in grouped.items()}


def _build_points(
    x_vals: Dict[str, float], y_vals: Dict[str, float], x_axis: str, y_axis: str
) -> Dict[str, Dict[str, float]]:
    """Union of configs across the two axes → points; a config present on only
    one axis gets NaN on the other (so :func:`pareto_frontier` excludes it)."""
    configs = set(x_vals) | set(y_vals)
    points: Dict[str, Dict[str, float]] = {}
    for cfg in configs:
        points[cfg] = {
            x_axis: x_vals.get(cfg, float("nan")),
            y_axis: y_vals.get(cfg, float("nan")),
        }
    return points


# ── §9.4 Frontier builders ─────────────────────────────────────────────────
# Relevance axis: the protocol lists ``avg_rating_at_10`` first, but that column
# is not in the current results; fall back to a relevance proxy in priority order.
_RELEVANCE_CANDIDATES = ("avg_rating_at_10", "relevancy_judgeA", "relevancy", "ndcg_at_10")
_QUALITY_CANDIDATES = ("correctness_judgeA", "correctness")
_COVERAGE_CANDIDATES = ("meal_slot_completeness", "structural_validity", "constraint_satisfaction")


def health_vs_relevance(
    agg,
    stratum: str = "aggregate",
    experiment_id: Optional[str] = None,
    health_metric: str = "avg_nutri_at_10",
) -> ParetoResult:
    """Health vs Relevance frontier (§9.4, §11.3).

    * x = relevance ("max"): ``avg_rating_at_10`` if present, else the best
      available proxy (``relevancy_judgeA`` → ``relevancy`` → ``ndcg_at_10``).
    * y = health ("max"): ``avg_nutri_at_10`` (pass ``health_metric='avg_fsa_at_10'``
      for the FSA variant).

    Shows the popularity-bias story: the rating-sorted / popularity baseline sits
    at high-relevance/low-health; health-aware configs trace toward high health.
    A FoodRAG config dominating a popularity baseline is stronger than any scalar.
    """
    metrics = _metrics_in_scope(agg, stratum, experiment_id)
    rel_metric = _first_present(metrics, _RELEVANCE_CANDIDATES)
    if rel_metric is None or health_metric not in metrics:
        return ParetoResult(
            objectives={"relevance": "max", "health": "max"},
            points={}, frontier=[], dominated_by={}, dominates={},
            excluded={}, axis_labels={"relevance": "relevance (n/a)", "health": health_metric},
        )
    x = _collect_axis_values(agg, rel_metric, stratum, experiment_id)
    y = _collect_axis_values(agg, health_metric, stratum, experiment_id)
    points = _build_points(x, y, "relevance", "health")
    return pareto_frontier(
        points,
        {"relevance": "max", "health": "max"},
        axis_labels={"relevance": rel_metric, "health": health_metric},
    )


def quality_vs_cost(
    agg,
    stratum: str = "aggregate",
    experiment_id: Optional[str] = None,
    cost_metric: str = "llm_usage",
) -> ParetoResult:
    """Quality vs Cost frontier (§9.4, §11.3).

    * x = cost ("min"): ``llm_usage`` (pass ``cost_metric='latency_s'`` for the
      latency variant).
    * y = quality ("max"): ``correctness_judgeA`` if present, else ``correctness``.

    Answers whether the agentic pipeline is Pareto-justified: is Heuristic 1 on
    the frontier, or is a cheaper config equally correct? Dominated configs are
    greyed in the UI.
    """
    metrics = _metrics_in_scope(agg, stratum, experiment_id)
    qual_metric = _first_present(metrics, _QUALITY_CANDIDATES)
    if qual_metric is None or cost_metric not in metrics:
        return ParetoResult(
            objectives={"cost": "min", "quality": "max"},
            points={}, frontier=[], dominated_by={}, dominates={},
            excluded={}, axis_labels={"cost": cost_metric, "quality": "quality (n/a)"},
        )
    x = _collect_axis_values(agg, cost_metric, stratum, experiment_id)
    y = _collect_axis_values(agg, qual_metric, stratum, experiment_id)
    points = _build_points(x, y, "cost", "quality")
    return pareto_frontier(
        points,
        {"cost": "min", "quality": "max"},
        axis_labels={"cost": cost_metric, "quality": qual_metric},
    )


# Answer-set-size strata used as the strictness PROXY for safety-vs-coverage.
# Tight (1–50 valid answers) ≈ high strictness; broad (>5000) ≈ low strictness.
_STRICTNESS_ORDER = ("tight", "medium", "broad")
_STRICTNESS_LEVEL = {"tight": 2.0, "medium": 1.0, "broad": 0.0}


def safety_vs_coverage(
    agg,
    experiment_id: Optional[str] = None,
    coverage_metric: Optional[str] = None,
    config_id: Optional[str] = None,
) -> ParetoResult:
    """Safety vs Coverage frontier (§9.4, §11.3).

    §9.4 axes are *relaxation level (strictness)* vs *fraction of personas with a
    full feasible plan*. An explicit per-config "relaxation level" axis is NOT in
    the current results, so — per the task's documented-proxy option — we use the
    **answer-set-size stratum as a strictness proxy**: tight (1–50 valid answers)
    → high strictness, medium, broad → low strictness (§9.1). Coverage is a
    config-level feasibility proxy: mean ``meal_slot_completeness`` (→
    ``structural_validity`` → ``constraint_satisfaction`` fallback) per stratum.

    Each *stratum* becomes one point (``strictness`` axis "max" = stricter,
    ``coverage`` axis "max"). The frontier then shows the trade-off between never
    relaxing safety (high strictness → fewer feasible plans) and coverage. Safety
    itself is never an axis — hard constraints gate feasibility upstream (§9.4);
    here strictness is a stratum proxy and coverage is the feasible-plan fraction.

    Note: this is a documented PROXY (strata, not a genuine per-config relaxation
    sweep). When the eval grows an explicit relaxation-level axis, swap the
    ``strictness`` values for it without touching :func:`pareto_frontier`.
    """
    metrics = _metrics_in_scope(agg, None, experiment_id)
    cov_metric = coverage_metric or _first_present(metrics, _COVERAGE_CANDIDATES)
    empty = ParetoResult(
        objectives={"strictness": "max", "coverage": "max"},
        points={}, frontier=[], dominated_by={}, dominates={},
        excluded={}, axis_labels={"strictness": "answer-set stratum (proxy)",
                                  "coverage": cov_metric or "coverage (n/a)"},
    )
    if cov_metric is None:
        return empty

    df = agg[agg["metric"] == cov_metric]
    if experiment_id is not None:
        df = df[df["experiment_id"] == experiment_id]
    if config_id is not None:
        df = df[df["config_id"] == config_id]
    df = df[df["stratum"].isin(_STRICTNESS_ORDER)]
    if df.empty:
        return empty

    points: Dict[str, Dict[str, float]] = {}
    for stratum in _STRICTNESS_ORDER:
        sub = df[df["stratum"] == stratum]
        if sub.empty:
            continue
        points[stratum] = {
            "strictness": _STRICTNESS_LEVEL[stratum],
            "coverage": float(sub["mean"].mean()),
        }
    if not points:
        return empty
    return pareto_frontier(
        points,
        {"strictness": "max", "coverage": "max"},
        axis_labels={"strictness": "answer-set stratum (strictness proxy)", "coverage": cov_metric},
    )

"""Pareto frontier + chart tests (spec Flow C)."""
import math

import pytest

from ui.components.pareto_chart import pareto_frontier

pytestmark = [pytest.mark.unit, pytest.mark.gui]


def test_frontier_both_maximise():
    # C dominates B; A and C are non-dominated (A better x, C better y)
    xs = [0.9, 0.5, 0.6]  # A, B, C
    ys = [0.3, 0.4, 0.8]
    flags = pareto_frontier(xs, ys, "max", "max")
    assert flags == [True, False, True]


def test_frontier_min_cost_max_quality():
    # x = cost (min better), y = quality (max better)
    # points: cheap-good (2,0.8), pricey-good (10,0.85), cheap-bad (2,0.5)
    xs = [2, 10, 2]
    ys = [0.8, 0.85, 0.5]
    flags = pareto_frontier(xs, ys, "min", "max")
    # (2,0.8) non-dominated; (10,0.85) non-dominated (best quality); (2,0.5) dominated by (2,0.8)
    assert flags == [True, True, False]


def test_single_point_is_frontier():
    assert pareto_frontier([1.0], [1.0], "max", "max") == [True]


def test_nan_never_on_frontier():
    flags = pareto_frontier([1.0, float("nan")], [1.0, 1.0], "max", "max")
    assert flags[0] is True and flags[1] is False


def test_duplicate_points_both_kept():
    # identical points don't strictly dominate each other → both non-dominated
    flags = pareto_frontier([0.5, 0.5], [0.5, 0.5], "max", "max")
    assert flags == [True, True]


def test_render_pareto_smoke():
    # render must not raise with a real DataFrame (Plotly build)
    pytest.importorskip("pandas")
    pytest.importorskip("plotly")
    from streamlit.testing.v1 import AppTest

    def script():
        import pandas as pd
        from ui.components.pareto_chart import render_pareto
        df = pd.DataFrame({"config": ["a", "b", "c"], "x": [0.9, 0.5, 0.6], "y": [0.3, 0.4, 0.8]})
        render_pareto(df, "x", "y", "X", "Y", x_dir="max", y_dir="max", ideal="top-right")

    at = AppTest.from_function(script)
    at.run()
    assert not at.exception, [e.value for e in at.exception]

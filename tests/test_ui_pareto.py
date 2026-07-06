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


def test_report_pareto_tab_renders():
    """The Report tab's Pareto panels render 3 interactive charts by default."""
    from pathlib import Path

    from streamlit.testing.v1 import AppTest

    app = str(Path(__file__).resolve().parents[1] / "app.py")
    at = AppTest.from_file(app, default_timeout=40)
    at.query_params["report"] = "true"
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    md = " ".join(m.value for m in at.markdown)
    caps = " ".join(c.value for c in at.caption)
    assert all(k in md for k in ("Health vs Relevance", "Quality vs Cost", "Safety vs Coverage"))
    assert "never" in caps and "Pareto axis" in caps  # safety-not-an-axis caveat
    assert len(at.get("plotly_chart")) >= 3


def test_render_pareto_honours_flag_col(monkeypatch):
    """render_pareto must use precomputed `flag_col` flags, NOT recompute them —
    so the Report tab's frontier comes from foodrag.analysis.pareto, not the
    component's local fallback."""
    import pandas as pd

    import ui.components.pareto_chart as pc

    captured = {}
    monkeypatch.setattr(pc.st, "plotly_chart", lambda fig, **k: captured.__setitem__("fig", fig))
    monkeypatch.setattr(pc.st, "caption", lambda *a, **k: None)

    # flag_col marks ONLY 'a' as frontier; a local recompute (max/max) would also
    # mark 'c' (0.6, 0.8) as non-dominated. So {'a'} proves flag_col wins.
    df = pd.DataFrame({"config": ["a", "b", "c"], "x": [0.9, 0.5, 0.6],
                       "y": [0.3, 0.4, 0.8], "frontier": [True, False, False]})
    assert set(c for c, f in zip(df["config"], pareto_frontier(df["x"].tolist(), df["y"].tolist(),
                                                                "max", "max")) if f) == {"a", "c"}
    pc.render_pareto(df, "x", "y", "X", "Y", x_dir="max", y_dir="max", flag_col="frontier")

    frontier_pts = set()
    for tr in captured["fig"].data:
        if tr.name == "Pareto-optimal":
            frontier_pts = set(tr.text)
    assert frontier_pts == {"a"}, f"flag_col ignored — recomputed instead ({frontier_pts})"


def test_report_panels_pass_flag_col(monkeypatch):
    """Every Report Pareto panel must feed render_pareto the module-computed flags
    via flag_col='frontier' (guards against a silent fallback to local compute)."""
    from pathlib import Path

    from streamlit.testing.v1 import AppTest

    import ui.screens.report as report

    calls = []
    monkeypatch.setattr(report, "render_pareto", lambda *a, **k: calls.append(k))

    app = str(Path(__file__).resolve().parents[1] / "app.py")
    at = AppTest.from_file(app, default_timeout=40)
    at.query_params["report"] = "true"
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert len(calls) >= 3, f"expected ≥3 Pareto panels, got {len(calls)}"
    assert all(k.get("flag_col") == "frontier" for k in calls), \
        "a panel omitted flag_col='frontier' → would silently recompute locally"

"""Reusable Pareto scatter with non-dominated frontier highlight (Flow C).

An *analysis lens only* — it visualises how configurations trade off two competing
objectives and highlights the Pareto-optimal (non-dominated) set. It does NOT
change how Flow A recommends (weighted-sum ranking). Safety is never
a true Pareto axis (it's a hard feasibility filter) — the Safety-vs-Coverage panel
uses this component only as a lens, with a caveat.

``pareto_frontier`` is pure (unit-testable); ``render_pareto`` draws an interactive
Plotly scatter (points + frontier line + hover for config details).
"""

from __future__ import annotations

import math

import streamlit as st

from ..theme import C


def pareto_frontier(xs: list[float], ys: list[float],
                    x_dir: str = "max", y_dir: str = "max") -> list[bool]:
    """Non-dominated flags for each (x, y) point.

    ``x_dir``/``y_dir`` ∈ {"max", "min"} give each objective's better direction.
    Point i is dominated if some j is ≥ on both objectives and strictly better on
    at least one (after orienting both to maximisation). NaN points are never on
    the frontier.
    """
    sx = 1.0 if x_dir == "max" else -1.0
    sy = 1.0 if y_dir == "max" else -1.0
    n = len(xs)
    flags = [True] * n
    for i in range(n):
        xi, yi = xs[i], ys[i]
        if xi is None or yi is None or (isinstance(xi, float) and math.isnan(xi)) or \
                (isinstance(yi, float) and math.isnan(yi)):
            flags[i] = False
            continue
        oxi, oyi = sx * xi, sy * yi
        for j in range(n):
            if i == j:
                continue
            xj, yj = xs[j], ys[j]
            if xj is None or yj is None or (isinstance(xj, float) and math.isnan(xj)) or \
                    (isinstance(yj, float) and math.isnan(yj)):
                continue
            oxj, oyj = sx * xj, sy * yj
            if oxj >= oxi and oyj >= oyi and (oxj > oxi or oyj > oyi):
                flags[i] = False
                break
    return flags


def render_pareto(df, x_col: str, y_col: str, x_label: str, y_label: str,
                  x_dir: str = "max", y_dir: str = "max", config_col: str = "config",
                  ideal: str = "", height: int = 380, flag_col: str | None = None) -> None:
    """Interactive Plotly Pareto scatter: all configs as points, the non-dominated
    set highlighted and connected by a frontier line, hover shows config details.

    If ``flag_col`` names a boolean column in ``df``, those precomputed
    non-dominated flags are used verbatim (e.g. from ``foodrag.analysis.pareto``,
    which also exposes the full domination relation) instead of recomputing them
    here. Otherwise the local :func:`pareto_frontier` is used."""
    import plotly.graph_objects as go

    d = df.dropna(subset=[x_col, y_col]).copy()
    if d.empty:
        st.info("No data for this panel.")
        return
    if flag_col is not None and flag_col in d.columns:
        flags = [bool(f) for f in d[flag_col].tolist()]
    else:
        flags = pareto_frontier(d[x_col].tolist(), d[y_col].tolist(), x_dir, y_dir)
    d["role"] = ["Pareto-optimal" if f else "dominated" for f in flags]
    hover = (f"<b>%{{customdata}}</b><br>{x_label}: %{{x:.3f}}<br>{y_label}: %{{y:.3f}}<extra></extra>")

    fig = go.Figure()
    front = d[d["role"] == "Pareto-optimal"].sort_values(x_col)
    if len(front) >= 2:
        fig.add_trace(go.Scatter(x=front[x_col], y=front[y_col], mode="lines", name="frontier",
                                 line=dict(color=C["primary"], dash="dash", width=2),
                                 hoverinfo="skip", showlegend=False))
    for role, col, size in (("dominated", C["off"], 12), ("Pareto-optimal", C["primary"], 16)):
        r = d[d["role"] == role]
        if r.empty:
            continue
        fig.add_trace(go.Scatter(
            x=r[x_col], y=r[y_col], mode="markers+text", name=role,
            text=r[config_col], textposition="top center", textfont=dict(size=10, color=C["text"]),
            customdata=r[config_col], hovertemplate=hover,
            marker=dict(color=col, size=size, line=dict(width=1, color="white")),
        ))
    fig.update_layout(
        height=height, margin=dict(l=10, r=10, t=30, b=10), plot_bgcolor="white",
        xaxis_title=x_label, yaxis_title=y_label,
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0),
    )
    grid = dict(showgrid=True, gridcolor=C["border"], zeroline=False)
    fig.update_xaxes(**grid)
    fig.update_yaxes(**grid)
    st.plotly_chart(fig, use_container_width=True)
    if ideal:
        st.caption(ideal)

"""Knowledge-graph path visualisation (spec §4.4 "KG path visualization").

Renders the user → condition/skill → constraint/category → recipe BFS path as
inline SVG. A textual node/edge list is also emitted as a text alternative
(WCAG 1.1.1 / spec §8.2). Ported from the mockup's KGPath component.
"""

from __future__ import annotations

from ..mock_data import KG_EDGES, KG_NODES
from ..theme import C, F, html

_NODE_COLORS = {
    "user": C["primary"], "condition": C["accent"], "skill": C["nutB"],
    "constraint": C["nutD"], "category": C["nutC"], "recipe": C["primary"],
}


def render_kg_path() -> None:
    nodes = {n["id"]: n for n in KG_NODES}

    edges_svg = ""
    for e in KG_EDGES:
        f, t = nodes[e["from"]], nodes[e["to"]]
        mx, my = (f["x"] + t["x"]) / 2, (f["y"] + t["y"]) / 2
        edges_svg += (
            f'<line x1="{f["x"]}" y1="{f["y"]}" x2="{t["x"]}" y2="{t["y"]}" '
            f'stroke="{C["primary"]}" stroke-width="2" stroke-opacity="0.2" />'
            f'<text x="{mx}" y="{my - 5}" text-anchor="middle" fill="{C["textMuted"]}" '
            f'font-size="7" font-family="{F}">{e["label"]}</text>'
        )

    nodes_svg = ""
    for n in KG_NODES:
        col = _NODE_COLORS[n["type"]]
        nodes_svg += (
            f'<circle cx="{n["x"]}" cy="{n["y"]}" r="20" fill="{col}" fill-opacity="0.12" '
            f'stroke="{col}" stroke-width="2" />'
            f'<text x="{n["x"]}" y="{n["y"] + 1}" text-anchor="middle" dominant-baseline="middle" '
            f'fill="{col}" font-size="8" font-weight="700" font-family="{F}">{n["label"]}</text>'
            f'<text x="{n["x"]}" y="{n["y"] + 32}" text-anchor="middle" fill="{C["textMuted"]}" '
            f'font-size="6" font-style="italic" font-family="{F}">{n["type"]}</text>'
        )

    html(
        f"<div style=\"font-size:9px;color:{C['textMuted']};margin-bottom:4px;font-family:{F}\">"
        f"HUMMUS KG path (BFS, max 3 hops). Skill node: Bölz et al. (2025).</div>"
        f"<div style=\"background:#fafaf7;border-radius:6px;border:1px solid {C['border']}\">"
        f"<svg width=\"100%\" height=\"240\" viewBox=\"0 0 460 240\">{edges_svg}{nodes_svg}</svg></div>"
    )

    # Text alternative for screen readers / non-visual contexts
    import streamlit as st

    with st.expander("Path as text (accessible)"):
        for e in KG_EDGES:
            st.write(f"**{nodes[e['from']]['label']}** —{e['label']}→ **{nodes[e['to']]['label']}**")

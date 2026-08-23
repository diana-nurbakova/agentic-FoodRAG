"""KG-path explanation viz.

Renders the grounded path(s) connecting a recommended recipe to each stated
persona requirement (e.g. ``recipe → has → sugars = 4g → satisfies → sugar ≤ 10``,
``recipe → excludes → Gluten``). Paths are real (built from the recipe's
attributes/ingredients/tags via ``services.path_finding``), so this replaces the
old hardcoded graph. Path Coverage / Path Validity are shown, and a text
alternative is provided (WCAG 1.1.1).
"""

from __future__ import annotations

from typing import Callable, Optional

import streamlit as st

from ..services import path_finding
from ..theme import C, F, html

_NODE_COLORS = {
    "recipe": C["primary"], "attribute": C["primaryLight"], "constraint": C["nutD"],
    "allergen": C["accent"], "tag": C["nutC"], "ingredient": C["nutB"], "class": C["sub"],
}


def _chip(label: str, ntype: str) -> str:
    col = _NODE_COLORS.get(ntype, C["textMuted"])
    return (f"<span style=\"display:inline-block;background:{col}1f;color:{col};border:1px solid {col};"
            f"border-radius:12px;padding:2px 9px;font-family:{F};font-size:10px;font-weight:600;"
            f"white-space:nowrap\">{label}</span>")


def _edge(label: str, ok: bool) -> str:
    col = C["primaryLight"] if ok else C["accent"]
    return (f"<span style=\"font-family:{F};font-size:8px;color:{col};padding:0 4px;white-space:nowrap\">"
            f"—{label}→</span>")


def _path_row(path: dict) -> str:
    parts = []
    nodes, edges = path["nodes"], path["edges"]
    for i, (label, ntype) in enumerate(nodes):
        parts.append(_chip(label, ntype))
        if i < len(edges):
            parts.append(_edge(edges[i], path["satisfied"]))
    mark = "✓" if path["satisfied"] else "✗"
    mark_col = C["nutA"] if path["satisfied"] else C["accent"]
    chain = "".join(parts)
    return (f"<div style=\"display:flex;align-items:center;gap:4px;flex-wrap:wrap;padding:5px 0;"
            f"border-top:1px solid {C['border']}\">"
            f"<span style=\"color:{mark_col};font-weight:700;width:12px\">{mark}</span>{chain}</div>")


def render_recipe_paths(card: dict, persona: dict,
                        expand: Optional[Callable] = None) -> dict:
    """Render grounded requirement-paths for one recipe card. Returns the report."""
    report = path_finding.build_paths(card, persona, expand)

    if report["n"] == 0:
        html(f"<div style=\"font-family:{F};font-size:11px;color:{C['textMuted']}\">"
             f"No explicit dietary requirements for this persona — nothing to ground.</div>")
        return report

    cov, val = report["coverage"], report["validity"]
    cov_col = C["nutA"] if cov >= 0.999 else (C["nutC"] if cov >= 0.5 else C["accent"])
    html(
        f"<div style=\"font-family:{F};font-size:9px;color:{C['textMuted']};margin-bottom:4px\">"
        f"Grounded KG paths from <b>{report['recipe']}</b> to each requirement "
        f"(constraint-first; every edge is real).</div>"
        f"<div style=\"display:flex;gap:14px;font-family:{F};font-size:10px;margin-bottom:2px\">"
        f"<span>Path coverage: <b style=\"color:{cov_col}\">{cov:.0%}</b></span>"
        f"<span>Path validity: <b style=\"color:{C['nutA']}\">{val:.0%}</b></span></div>"
    )
    html("".join(_path_row(p) for p in report["paths"]))

    _render_text_alternative(report)
    return report


def render_kg_ingredient_paths(recipe_name: str, rows: list[dict]) -> None:
    """Render real recipe→ingredient→FoodOn-class paths from Neo4j.

    ``rows`` = ``[{ingredient, food_class, ancestors}]`` from
    ``backend.recipe_ingredient_class_paths``."""
    if not rows:
        html(f"<div style=\"font-family:{F};font-size:10px;color:{C['textMuted']}\">"
             f"No FoodOn-linked ingredients found in the knowledge graph for this recipe.</div>")
        return
    html(f"<div style=\"font-family:{F};font-size:9px;color:{C['textMuted']};margin:6px 0 4px\">"
         f"Ingredient → FoodOn class chains for <b>{recipe_name}</b>, traversed live over Neo4j "
         f"(bounded depth).</div>")
    for row in rows:
        nodes = [(recipe_name, "recipe"), (row["ingredient"], "ingredient"), (row["food_class"], "class")]
        nodes += [(a, "class") for a in row.get("ancestors", [])]
        edges = ["contains", "is a"] + ["is a"] * len(row.get("ancestors", []))
        parts = []
        for i, (label, ntype) in enumerate(nodes):
            parts.append(_chip(label, ntype))
            if i < len(edges):
                parts.append(_edge(edges[i], ok=True))
        html(f"<div style=\"display:flex;align-items:center;gap:4px;flex-wrap:wrap;padding:5px 0;"
             f"border-top:1px solid {C['border']}\">{''.join(parts)}</div>")


def _render_text_alternative(report: dict) -> None:
    # Text alternative (accessible)
    with st.expander("Paths as text (accessible)"):
        for p in report["paths"]:
            chain = " → ".join(
                lbl + (f" —{p['edges'][i]}→" if i < len(p["edges"]) else "")
                for i, (lbl, _t) in enumerate(p["nodes"])
            ).replace(" —", " —").rstrip("→ ").strip()
            st.write(("✓" if p["satisfied"] else "✗") + " " + chain)
    return report

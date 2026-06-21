"""Recipe detail modal (spec §4.4 "recipe detail modal with ingredient
substitutions"). Uses ``st.dialog`` for the overlay.
"""

from __future__ import annotations

import streamlit as st

from ..theme import C, F, FP, html
from .nutrition import bars_html, donut_html, skill_html, triple_score_html


def _summary_html(recipe: dict) -> str:
    cells = ""
    for s in (
        {"l": "Cal", "v": recipe["cal"], "i": "🔥"},
        {"l": "Protein", "v": f"{recipe['protein']}g", "i": "💪"},
        {"l": "Fiber", "v": f"{recipe['fiber']}g", "i": "🌿"},
        {"l": "Sugar", "v": f"{recipe['sugar']}g", "i": "🍬"},
    ):
        cells += (
            f"<div style=\"text-align:center\"><div style=\"font-size:12px\">{s['i']}</div>"
            f"<div style=\"font-family:{F};font-weight:700;font-size:11px;color:{C['primary']}\">{s['v']}</div>"
            f"<div style=\"font-family:{F};font-size:7px;color:{C['textMuted']}\">{s['l']}</div></div>"
        )
    return (
        f"<div style=\"background:{C['primaryPale']};border-radius:6px;padding:8px;margin-top:8px;"
        f"display:flex;justify-content:space-around\">{cells}</div>"
    )


def _ingredients_html(recipe: dict) -> str:
    subs = {s["orig"]: s for s in (recipe.get("substitutions") or [])}
    rows = ""
    for ing in recipe["ingredients"]:
        sub = subs.get(ing)
        deco = "line-through" if sub else "none"
        col = C["textMuted"] if sub else C["text"]
        rows += (
            f"<div style=\"margin-bottom:3px\">"
            f"<div style=\"font-family:{F};font-size:11px;color:{col};text-decoration:{deco}\">• {ing}</div>"
        )
        if sub:
            rows += (
                f"<div style=\"margin-left:14px;font-family:{F};font-size:10px;color:{C['sub']};"
                f"background:#f3e8ff;padding:2px 6px;border-radius:4px;margin-top:1px\">"
                f"→ <b>{sub['sub']}</b> <span style=\"font-size:8px;color:{C['textMuted']}\">({sub['reason']})</span></div>"
            )
        rows += "</div>"
    return rows


def _directions_html(recipe: dict) -> str:
    rows = ""
    for i, step in enumerate(recipe["steps"], 1):
        rows += (
            f"<div style=\"display:flex;gap:6px;font-family:{F};font-size:11px;color:{C['text']};"
            f"line-height:1.5;margin-bottom:4px\">"
            f"<span style=\"width:18px;height:18px;border-radius:50%;background:{C['primaryPale']};"
            f"color:{C['primary']};display:flex;align-items:center;justify-content:center;font-size:9px;"
            f"font-weight:700;flex-shrink:0\">{i}</span><span>{step}</span></div>"
        )
    return rows


@st.dialog("Recipe details", width="large")
def recipe_detail_dialog(recipe: dict) -> None:
    """Modal overlay with full recipe information."""
    html(
        f"<div style=\"display:flex;align-items:center;gap:6px;margin-bottom:6px\">"
        f"<span style=\"font-size:24px\">{recipe['img']}</span>"
        f"<span style=\"font-family:{FP};font-size:18px;color:{C['primary']}\">{recipe['name']}</span></div>"
        f"<div style=\"display:flex;gap:10px;flex-wrap:wrap;margin-bottom:4px;font-family:{F};"
        f"font-size:10px;color:{C['textMuted']};align-items:center\">"
        f"<span>⏱ {recipe['time']}</span><span>🍽 {recipe['servings']}p</span>{skill_html(recipe['skill'])}</div>"
        f"{triple_score_html(recipe['who'], recipe['fsa'], recipe['nutriscore'])}"
        f"{_summary_html(recipe)}"
    )

    with st.expander("Full nutrition breakdown", expanded=False):
        html(
            f"<div style=\"display:flex;gap:10px;align-items:center\">"
            f"{donut_html(recipe['protein'], recipe['carbs'], recipe['fat'], recipe['cal'], 56)}"
            f"<div style=\"flex:1\">{bars_html(recipe, expanded=True)}</div></div>"
        )

    html(f"<h3 style=\"font-family:{F};font-size:13px;font-weight:700;color:{C['text']};margin:12px 0 5px\">Ingredients</h3>")
    html(_ingredients_html(recipe))
    html(f"<h3 style=\"font-family:{F};font-size:13px;font-weight:700;color:{C['text']};margin:12px 0 5px\">Directions</h3>")
    html(_directions_html(recipe))

    if st.button("Close", key="close_detail", use_container_width=True):
        st.rerun()

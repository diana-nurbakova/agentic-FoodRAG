"""Recipe card: emoji, title, Nutri-Score, meta, macro donut + bars, triple
score, substitution badge, plus a Details link and 👍/👎 feedback row.

The visual block is HTML (``recipe_card_html``); the interactive footer
(Details button, feedback) uses native Streamlit widgets so clicks are
captured server-side (``render_recipe_card``).
"""

from __future__ import annotations

from typing import Callable

import streamlit as st

from ..state import log_event
from ..theme import C, F, MACRO, html
from .nutrition import bars_html, donut_html, nutriscore_html, skill_html, triple_score_html


def recipe_card_html(recipe: dict, compact: bool = False) -> str:
    """Static visual block for a recipe (no interactive footer)."""
    pad = 8 if compact else 10
    name_sz = 11 if compact else 12
    emoji_sz = 24 if compact else 28
    donut_sz = 38 if compact else 44

    subs = recipe.get("substitutions") or []
    sub_badge = ""
    if subs:
        n = len(subs)
        sub_badge = (
            f"<div style=\"margin-top:4px;padding:3px 6px;background:#f3e8ff;border-radius:4px;"
            f"font-family:{F};font-size:8px;color:{C['sub']}\">🔄 {n} ingredient substitution{'s' if n > 1 else ''} applied</div>"
        )

    macros = (
        f"<div style=\"display:flex;gap:6px;font-family:{F};font-size:8px;color:{C['textMuted']};margin-bottom:2px\">"
        f"<span><b style=\"color:{MACRO['protein']}\">●</b>P {recipe['protein']}g</span>"
        f"<span><b style=\"color:{MACRO['carbs']}\">●</b>C {recipe['carbs']}g</span>"
        f"<span><b style=\"color:{MACRO['fat']}\">●</b>F {recipe['fat']}g</span></div>"
    )

    return (
        f"<div style=\"background:{C['card']};border:1px solid {C['border']};border-radius:8px;padding:{pad}px\">"
        f"<div style=\"display:flex;gap:8px\">"
        f"<div style=\"font-size:{emoji_sz}px;line-height:1\">{recipe['img']}</div>"
        f"<div style=\"flex:1;min-width:0\">"
        f"<div style=\"display:flex;justify-content:space-between;align-items:flex-start;gap:4px\">"
        f"<div style=\"font-family:{F};font-weight:700;font-size:{name_sz}px;color:{C['text']}\">{recipe['name']}</div>"
        f"{nutriscore_html(recipe['nutriscore'], 'small')}</div>"
        f"<div style=\"display:flex;gap:6px;margin-top:2px;font-family:{F};font-size:8px;color:{C['textMuted']};align-items:center\">"
        f"<span>⏱{recipe['time']}</span><span>🍽{recipe['servings']}p</span>{skill_html(recipe['skill'])}</div>"
        f"</div></div>"
        f"<div style=\"display:flex;gap:8px;margin-top:6px;align-items:center\">"
        f"{donut_html(recipe['protein'], recipe['carbs'], recipe['fat'], recipe['cal'], donut_sz)}"
        f"<div style=\"flex:1\">{macros}{bars_html(recipe, expanded=False)}</div></div>"
        f"{triple_score_html(recipe['who'], recipe['fsa'], recipe['nutriscore'])}"
        f"{sub_badge}</div>"
    )


def render_recipe_card(recipe: dict, key: str, on_expand: Callable[[dict], None], compact: bool = False) -> None:
    """Render the card plus its interactive footer."""
    html(recipe_card_html(recipe, compact))
    c1, c2, c3 = st.columns([3, 1, 1])
    with c1:
        if st.button("📖 Details", key=f"details_{key}", use_container_width=True):
            on_expand(recipe)
    fb_key = f"fb_{key}"
    current = st.session_state.get(fb_key)
    with c2:
        if st.button("👍", key=f"up_{key}", type="primary" if current == "up" else "secondary",
                     use_container_width=True):
            st.session_state[fb_key] = "up"
            log_event("recipe_feedback", {"recipe_name": recipe["name"], "vote": "up"})
            st.rerun()
    with c3:
        if st.button("👎", key=f"down_{key}", type="primary" if current == "down" else "secondary",
                     use_container_width=True):
            st.session_state[fb_key] = "down"
            log_event("recipe_feedback", {"recipe_name": recipe["name"], "vote": "down"})
            st.rerun()

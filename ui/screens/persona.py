"""Screen 3: Persona selection + custom builder.

Predefined personas (admin-curated from HUMMUS) shown as cards with badges.
The "No Profile" card opens a custom persona builder mapping to the
``persona_config.yml`` structure.
"""

from __future__ import annotations

import streamlit as st

from ..components.study_common import render_step_nav
from ..state import goto, log_event
from ..study_data import flow_a_personas
from ..theme import C, F, FP, html


def _persona_card_html(p: dict) -> str:
    badges = ""
    for a in p["allergies"]:
        badges += (f"<span style=\"background:#fef3c7;color:#92400e;font-size:8px;padding:1px 6px;"
                   f"border-radius:6px;font-family:{F};font-weight:600;margin:1px\">⚠{a}</span>")
    for t in p["likedTags"]:
        badges += (f"<span style=\"background:{C['primaryPale']};color:{C['primary']};font-size:8px;"
                   f"padding:1px 6px;border-radius:6px;font-family:{F};font-weight:600;margin:1px\">♥{t}</span>")
    if p["nutrientConstraint"] != "None":
        badges += (f"<span style=\"background:{C['tagBg']};color:{C['textMuted']};font-size:8px;"
                   f"padding:1px 6px;border-radius:6px;font-family:{F};font-weight:600;margin:1px\">"
                   f"{p['nutrientConstraint']}</span>")
    badges += (f"<span style=\"background:{C['primaryPale']};color:{C['primary']};font-size:8px;"
               f"padding:1px 6px;border-radius:6px;font-family:{F};font-weight:600;margin:1px\">CS:{p['cs']}</span>")

    age = (f"<span style=\"font-weight:400;color:{C['textMuted']};font-size:10px;margin-left:3px\">{p['age']}y</span>"
           if p["age"] else "")
    # Real evaluation profiles carry a full constraint description; show it as the subtitle.
    subtitle = p.get("desc") or (p["condition"] if p["condition"] != "None" else "No health conditions")
    for d in (p.get("dislikedIng", []) + p.get("dislikedTags", [])):
        badges += (f"<span style=\"background:#fee2e2;color:#991b1b;font-size:8px;padding:1px 6px;"
                   f"border-radius:6px;font-family:{F};font-weight:600;margin:1px\">✕{d}</span>")
    return (
        f"<div style=\"background:{C['card']};border:1px solid {C['border']};border-radius:8px;"
        f"padding:12px;min-height:128px\">"
        f"<div style=\"font-size:22px;margin-bottom:3px\">{p['emoji']}</div>"
        f"<div style=\"font-family:{F};font-weight:700;font-size:13px;color:{C['text']}\">{p['name']}{age}</div>"
        f"<div style=\"font-family:{F};font-size:9px;color:{C['textMuted']};margin-top:2px;line-height:1.4\">{subtitle}</div>"
        f"<div style=\"margin-top:5px;display:flex;flex-wrap:wrap;gap:2px\">{badges}</div></div>"
    )


def _builder() -> None:
    html(f"<div style=\"font-family:{F};font-size:14px;font-weight:700;color:{C['primary']};"
         f"margin:8px 0\">⚙️ Build Your Persona</div>")
    condition = st.text_input("Health condition", placeholder="e.g. Type 2 Diabetes, High cholesterol, None")
    allergies = st.text_input("Allergies / Blacklisted ingredients", placeholder="e.g. Gluten, Nuts, Shellfish")
    liked = st.text_input("Liked tags", placeholder="e.g. Mediterranean, High-protein")
    disliked = st.text_input("Disliked ingredients", placeholder="e.g. Tofu, White bread")
    nutrient = st.text_input("Nutrient constraint", placeholder="e.g. Sugar ≤ 10g/meal, Protein ≥ 30g/meal")

    def _split(s: str) -> list[str]:
        return [x.strip() for x in s.split(",") if x.strip()]

    if st.button("Use this persona →", type="primary", use_container_width=True):
        persona = {
            "id": 99, "name": "Custom", "age": None, "emoji": "⚙️",
            "condition": condition or "None", "allergies": _split(allergies),
            "likedTags": _split(liked), "dislikedTags": [], "likedIng": [],
            "dislikedIng": _split(disliked), "nutrientConstraint": nutrient or "None",
            "cs": "—", "fs": "—",
        }
        _select(persona)


def _select(persona: dict) -> None:
    st.session_state["persona"] = persona
    # Real evaluation profiles seed the query with their exact study prompt so the live
    # pipeline reproduces evaluated behaviour (user can still edit or clear it).
    if persona.get("prompt"):
        st.session_state["query_text"] = persona["prompt"]
    log_event("persona_selected", {"persona_id": persona["id"], "persona_data": persona})
    goto("main")


def render() -> None:
    # Flow A is consent → persona → main: Back returns to consent, Home to welcome.
    render_step_nav(back_screen="consent", home_screen="welcome")
    html(
        f"<h2 style=\"font-family:{FP};font-size:20px;color:{C['primary']};margin:0 0 3px\">Select a Persona</h2>"
        f"<p style=\"font-family:{F};font-size:10px;color:{C['textMuted']};margin:0 0 14px\">"
        f"Structured profiles from the test set. CS/FS scores: Bölz et al. (2025).</p>"
    )

    cols = st.columns(2)
    for idx, p in enumerate(flow_a_personas()):
        with cols[idx % 2]:
            html(_persona_card_html(p))
            is_no_profile = p["id"] == 4
            label = "🛠 Build custom…" if is_no_profile else f"Select {p['name']}"
            if st.button(label, key=f"persona_{p['id']}", use_container_width=True):
                if is_no_profile:
                    st.session_state["show_builder"] = True
                    st.rerun()
                else:
                    _select(p)
            st.write("")

    if st.session_state.get("show_builder"):
        st.divider()
        _builder()

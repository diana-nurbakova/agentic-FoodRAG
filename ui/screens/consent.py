"""Screen 1: Consent (spec §4.1).

GDPR: no sensitive personal data; random session ID; consent required before
continuing. Continue is disabled until the box is checked.
"""

from __future__ import annotations

import streamlit as st

from ..state import goto, log_event
from ..theme import C, F, FP, html


def render() -> None:
    html(
        f"<div style=\"text-align:center;margin-bottom:16px\">"
        f"<div style=\"font-size:32px\">🍽️</div>"
        f"<h1 style=\"font-family:{FP};font-size:24px;color:{C['primary']};margin:2px 0 0\">FoodRAG</h1>"
        f"<p style=\"font-family:{F};color:{C['textMuted']};font-size:11px;margin:0\">"
        f"Health-Aware RAG-Based Food Recommendation</p></div>"
    )

    html(f"<h2 style=\"font-family:{F};font-size:16px;margin:0 0 8px;color:{C['text']}\">Research Consent</h2>")
    for label, body in (
        ("Purpose", "Evaluation of a food recommendation system — LIRIS (UMR 5205 CNRS), INSA Lyon."),
        ("Data", "Interactions, HEXAD scores, pipeline preferences, timing. Fictional personas — no real health data."),
        ("Anonymity", "Random session IDs. Free-text queries are logged — please do not enter real health data."),
        ("Withdrawal", "Stop anytime. Data deleted on request. Retention: 24 months."),
    ):
        html(
            f"<p style=\"font-family:{F};font-size:11px;line-height:1.7;color:{C['textMuted']};margin:0 0 5px\">"
            f"<b style=\"color:{C['text']}\">{label}:</b> {body}</p>"
        )

    st.caption(f"Session ID: `{st.session_state['session_id']}`")

    ok = st.checkbox("I consent to anonymous interaction logging.", key="consent_checkbox")
    if st.button("Continue", type="primary", disabled=not ok, use_container_width=True):
        st.session_state["consent_given"] = True
        log_event("consent_given")
        # Branch to the selected flow (spec §3). Defaults to exploration.
        goto("demographics" if st.session_state.get("flow") == "evaluation" else "hexad")

"""FoodRAG demo — Streamlit entry point (spec §11.5).

Routing + session bootstrap. Flow A (Interactive Exploration) and Flow B
(Structured Evaluation) are implemented and share the consent screen; a welcome
screen selects the flow. The Admin view (``?admin=true``) is a later phase.
Run with: ``streamlit run app.py``.
"""

from __future__ import annotations

import streamlit as st

from ui.screens import consent, hexad, main, persona, report, welcome
from ui.screens.admin import view as admin_view
from ui.screens.evaluation import (
    completion,
    demographics,
    plan_evaluation,
    post_study,
    profile_assignment,
    ranking,
)
from ui.state import init_state
from ui.theme import inject_global_css

st.set_page_config(page_title="FoodRAG Demo", page_icon="🍽️", layout="centered")

inject_global_css()
init_state()

# Admin view (spec §6.1): ?admin=true reveals the password-gated admin panel.
if st.query_params.get("admin") == "true":
    admin_view.render()
    st.stop()

# Report tab (Flow C, Evaluation Protocol §11): ?report=true opens it directly.
if st.query_params.get("report") == "true":
    report.render()
    st.stop()

_ROUTES = {
    "welcome": welcome.render,
    "report": report.render,
    "consent": consent.render,
    # Flow A
    "hexad": hexad.render,
    "persona": persona.render,
    "main": main.render,
    # Flow B
    "demographics": demographics.render,
    "profile_assignment": profile_assignment.render,
    "plan_evaluation": plan_evaluation.render,
    "post_study": post_study.render,
    "ranking": ranking.render,
    "completion": completion.render,
}

screen = st.session_state.get("screen", "welcome")
_ROUTES.get(screen, welcome.render)()

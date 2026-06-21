"""Admin view shell: authentication gate + tabbed sections (spec §6.1).

Auth: ``?admin=true`` query param reveals a password prompt compared against the
``ADMIN_PASSWORD`` env var (spec §6.1, §11.3). If unset, a dev fallback is used
with a visible warning. Authenticated admin sessions are tagged ``is_admin`` so
they're excluded from participant analytics (spec §6.5).
"""

from __future__ import annotations

import os

import streamlit as st

from ...theme import C, F, FP, html
from . import dashboard, data_export, persona_manager, pipeline_defaults, questionnaire_config

_DEV_FALLBACK = "admin"  # used only when ADMIN_PASSWORD is not configured


def _authenticate() -> bool:
    if st.session_state.get("admin_authenticated"):
        return True

    html(
        f"<div style=\"text-align:center;margin-bottom:14px\">"
        f"<div style=\"font-size:30px\">🔐</div>"
        f"<h1 style=\"font-family:{FP};font-size:22px;color:{C['primary']};margin:4px 0 0\">FoodRAG Admin</h1>"
        f"<p style=\"font-family:{F};font-size:12px;color:{C['textMuted']};margin:2px 0 0\">"
        f"Study configuration & analytics</p></div>"
    )

    expected = os.environ.get("ADMIN_PASSWORD")
    if not expected:
        st.warning("`ADMIN_PASSWORD` is not set — using the dev fallback. Set the secret before deployment.")
        expected = _DEV_FALLBACK

    pwd = st.text_input("Admin password", type="password", key="admin_pwd")
    if st.button("Sign in", type="primary"):
        if pwd == expected:
            st.session_state["admin_authenticated"] = True
            st.session_state["is_admin"] = True
            st.rerun()
        else:
            st.error("Incorrect password.")
    return False


def render() -> None:
    if not _authenticate():
        return

    top_l, top_r = st.columns([4, 1])
    with top_l:
        html(f"<span class=\"foodrag-brand\" style=\"font-size:20px\">FoodRAG</span>"
             f"<span style=\"font-family:{F};font-size:11px;color:{C['accent']};font-weight:600;"
             f"margin-left:8px;background:{C['accentPale']};padding:2px 8px;border-radius:6px\">ADMIN</span>")
    with top_r:
        if st.button("Exit admin", use_container_width=True):
            st.session_state["admin_authenticated"] = False
            st.session_state["is_admin"] = False
            st.query_params.clear()
            st.session_state["screen"] = "welcome"
            st.rerun()

    tabs = st.tabs([
        "📊 Analytics", "👤 Personas", "📋 Questionnaire", "⚙️ Pipeline", "💾 Data export",
    ])
    with tabs[0]:
        dashboard.render()
    with tabs[1]:
        persona_manager.render()
    with tabs[2]:
        questionnaire_config.render()
    with tabs[3]:
        pipeline_defaults.render()
    with tabs[4]:
        data_export.render()

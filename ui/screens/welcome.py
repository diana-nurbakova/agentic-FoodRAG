"""Entry screen: choose a flow (spec §3).

Flow A (Interactive Exploration) for demo/reviewers; Flow B (Structured
Evaluation) for the user study. When the admin pins a single active flow
(``active_flow`` != "both"), this screen auto-routes past the chooser.
"""

from __future__ import annotations

import streamlit as st

from ..admin_config import get_admin_config
from ..state import goto
from ..theme import C, F, FP, html


def _start(flow: str) -> None:
    st.session_state["flow"] = flow
    goto("consent")


def render() -> None:
    active = get_admin_config()["active_flow"]
    if active == "exploration":
        _start("exploration")
    if active == "evaluation":
        _start("evaluation")

    html(
        f"<div style=\"text-align:center;margin-bottom:18px\">"
        f"<div style=\"font-size:36px\">🍽️</div>"
        f"<h1 style=\"font-family:{FP};font-size:26px;color:{C['primary']};margin:4px 0 0\">FoodRAG</h1>"
        f"<p style=\"font-family:{F};color:{C['textMuted']};font-size:12px;margin:2px 0 0\">"
        f"Health-Aware, Explainable Food Recommendation · LIRIS · INSA Lyon</p></div>"
    )
    html(f"<p style=\"font-family:{F};font-size:13px;color:{C['textMuted']};text-align:center;margin-bottom:16px\">"
         f"Choose how you'd like to continue.</p>")

    col_a, col_b = st.columns(2)
    with col_a:
        html(
            f"<div style=\"background:{C['card']};border:1px solid {C['border']};border-radius:12px;"
            f"padding:18px;min-height:170px\">"
            f"<div style=\"font-size:26px\">🧭</div>"
            f"<div style=\"font-family:{F};font-size:15px;font-weight:700;color:{C['text']};margin-top:6px\">Recommender System</div>"
            f"<div style=\"font-family:{F};font-size:12px;color:{C['textMuted']};line-height:1.6;margin-top:4px\">"
            f"Open-ended demo: pick a persona, run the agent pipeline, switch retrievers, compare configs in arena mode.</div></div>"
        )
        if st.button("Start exploring →", key="flow_a", type="primary", use_container_width=True):
            _start("exploration")
    with col_b:
        html(
            f"<div style=\"background:{C['card']};border:1px solid {C['border']};border-radius:12px;"
            f"padding:18px;min-height:170px\">"
            f"<div style=\"font-size:26px\">📋</div>"
            f"<div style=\"font-family:{F};font-size:15px;font-weight:700;color:{C['text']};margin-top:6px\">Take part in the study</div>"
            f"<div style=\"font-family:{F};font-size:12px;color:{C['textMuted']};line-height:1.6;margin-top:4px\">"
            f"Structured evaluation: a few questions, then rate and rank meal plans for an assigned profile (~12–15 min).</div></div>"
        )
        if st.button("Start the study →", key="flow_b", use_container_width=True):
            _start("evaluation")

    st.write("")
    if st.button("📊 Explore the evaluation results (Report)", key="flow_c", use_container_width=True):
        goto("report")

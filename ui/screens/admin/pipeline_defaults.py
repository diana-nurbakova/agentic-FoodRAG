"""Pipeline defaults (spec §6.4).

Defaults the Flow A pipeline starts from, and whether participants may change it
or use arena mode (lock for a controlled between-subjects study, unlock for the
open demo). These take effect for the current session's Flow A main screen.
"""

from __future__ import annotations

import streamlit as st

from ...admin_config import pipeline_defaults
from ...mock_data import RETRIEVER_LABELS

_RETRIEVERS = list(RETRIEVER_LABELS.keys())


def render() -> None:
    pd_cfg = pipeline_defaults()

    st.subheader("Pipeline defaults")
    st.caption("Starting configuration for Flow A, and what participants are allowed to change.")

    c1, c2 = st.columns(2)
    with c1:
        pd_cfg["default_retriever"] = st.selectbox(
            "Default retriever", _RETRIEVERS,
            index=_RETRIEVERS.index(pd_cfg["default_retriever"]),
            format_func=lambda k: RETRIEVER_LABELS[k],
        )
        pd_cfg["validation_default"] = st.toggle("Validation agent on by default",
                                                value=pd_cfg["validation_default"])
        pd_cfg["substitution_default"] = st.toggle("Substitution agent on by default",
                                                  value=pd_cfg.get("substitution_default", True))
        pd_cfg["explanation_default"] = st.toggle("Explanation agent on by default",
                                                 value=pd_cfg["explanation_default"])
    with c2:
        pd_cfg["arena_available"] = st.toggle("Arena mode available", value=pd_cfg["arena_available"],
                                             help="Disable for focused single-condition evaluation.")
        pd_cfg["user_can_change_pipeline"] = st.toggle(
            "Participants can change the pipeline", value=pd_cfg["user_can_change_pipeline"],
            help="Lock for a controlled study; unlock for the open demo.",
        )

    st.divider()
    st.subheader("Live backend")
    st.caption("When on, Flow A's **Run pipeline** calls the real agentic RAG backend "
               "(Llama 3.3 70B via OpenRouter + HUMMUS). Real runs cost money and take "
               "tens of seconds; off uses mock data. `FOODRAG_REAL_BACKEND=1` also forces it on.")
    pd_cfg["use_real_backend"] = st.toggle(
        "Use real backend for Flow A", value=pd_cfg.get("use_real_backend", False),
    )
    if pd_cfg["use_real_backend"]:
        from ...services import backend as _backend

        ok, reason = _backend.is_available()
        (st.success if ok else st.warning)(f"Backend availability: {reason}")
        st.caption("Live retrievers: Dense, Sparse (BM25), Hybrid, Text→Pandas, and Fusion "
                   "(RRF over dense+sparse+KG). Fusion includes the Neo4j KG retriever when it constructs.")

    st.divider()
    if st.button("Apply defaults to a fresh Flow A config now"):
        st.session_state["config_a"] = {
            "retriever": pd_cfg["default_retriever"],
            "validation": pd_cfg["validation_default"],
            "substitution": pd_cfg.get("substitution_default", True),
            "explanation": pd_cfg["explanation_default"],
        }
        # reset any completed run so the new config is used
        for k in ("done_A", "times_A", "ran_sig_A"):
            st.session_state.pop(k, None)
        st.success("Flow A configuration reset to the new defaults.")

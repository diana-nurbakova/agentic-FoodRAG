"""Session-state initialisation and small routing helpers.

The app uses single-page routing via ``st.session_state['screen']`` (a wizard),
deliberately *not* Streamlit's ``pages/`` auto-discovery — the flows are linear
and we want full control over navigation and a clean URL.

Two flows share the consent screen (spec §3): Flow A (exploration) and Flow B
(structured evaluation). A welcome screen selects the flow when both are active.
"""

from __future__ import annotations

import uuid

import streamlit as st

from .admin_config import default_admin_config

# Flow A screens (spec §4) and Flow B screens (spec §5)
FLOW_A_SCREENS = ["consent", "hexad", "persona", "main"]
FLOW_B_SCREENS = ["consent", "demographics", "profile_assignment", "plan_evaluation",
                  "post_study", "ranking", "completion"]


def init_state() -> None:
    """Seed session state on first load. Idempotent."""
    ss = st.session_state
    ss.setdefault("session_id", uuid.uuid4().hex[:8])
    # Stable per-session participant number → counterbalancing/assignment (spec §5.3).
    # Production uses a shared counter from the config service for global balance.
    ss.setdefault("participant_number", int(ss["session_id"], 16) % 997)
    ss.setdefault("screen", "welcome")
    ss.setdefault("flow", None)                 # "exploration" | "evaluation"
    ss.setdefault("consent_given", False)

    # Admin (spec §6): config + auth flags
    ss.setdefault("admin_config", default_admin_config())
    ss.setdefault("admin_authenticated", False)
    ss.setdefault("is_admin", False)

    # Flow A — seed the pipeline config from the admin pipeline defaults (spec §6.4)
    ss.setdefault("hexad_scores", None)
    ss.setdefault("hexad_type", None)
    ss.setdefault("persona", None)
    _pd = ss["admin_config"]["pipeline_defaults"]
    ss.setdefault("config_a", {"retriever": _pd["default_retriever"],
                               "validation": _pd["validation_default"],
                               "explanation": _pd["explanation_default"]})
    ss.setdefault("config_b", {"retriever": "sparse", "validation": True, "explanation": False})

    # Flow B
    ss.setdefault("demographics", None)
    ss.setdefault("assigned_profile_id", None)
    ss.setdefault("plan_order", None)
    ss.setdefault("plan_index", 0)
    ss.setdefault("plan_answers", [])
    ss.setdefault("attention_results", [])
    ss.setdefault("post_study_answers", None)
    ss.setdefault("ranking_result", None)

    # Lightweight in-memory interaction log (replace with logging_service later)
    ss.setdefault("event_log", [])


def goto(screen: str) -> None:
    """Navigate to a screen and rerun."""
    st.session_state["screen"] = screen
    st.rerun()


def log_event(event_type: str, payload: dict | None = None) -> None:
    """Log an interaction event (spec §9).

    Delegates to the logging service, which enriches to the §9.4 schema, buffers
    in memory, and pushes to the HF Dataset repo / external DB when configured.
    """
    from .services.logging_service import log_event as _log_event

    _log_event(event_type, payload)

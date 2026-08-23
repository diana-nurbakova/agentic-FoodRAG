"""Session-state initialisation and small routing helpers.

The app uses single-page routing via ``st.session_state['screen']`` (a wizard),
deliberately *not* Streamlit's ``pages/`` auto-discovery — the flows are linear
and we want full control over navigation and a clean URL.

Two flows share the consent screen: Flow A (exploration) and Flow B
(structured evaluation). A welcome screen selects the flow when both are active.
"""

from __future__ import annotations

import uuid

import streamlit as st

from .admin_config import default_admin_config
from .services.config_service import effective_config

# Flow A screens and Flow B screens
# HEXAD questionnaire hidden for now — kept out of the Flow A screen sequence.
FLOW_A_SCREENS = ["consent", "persona", "main"]
FLOW_B_SCREENS = ["consent", "demographics", "profile_assignment", "plan_evaluation",
                  "post_study", "ranking", "completion"]


def init_state() -> None:
    """Seed session state on first load. Idempotent."""
    ss = st.session_state
    ss.setdefault("session_id", uuid.uuid4().hex[:8])
    # Stable per-session participant number → counterbalancing/assignment.
    # Production uses a shared counter from the config service for global balance.
    ss.setdefault("participant_number", int(ss["session_id"], 16) % 997)
    ss.setdefault("screen", "welcome")
    ss.setdefault("flow", None)                 # "exploration" | "evaluation"
    ss.setdefault("consent_given", False)

    # Admin: config persisted to the dataset repo, deep-merged
    # over defaults so older saved configs still get new keys.
    if "admin_config" not in ss:
        try:
            ss["admin_config"] = effective_config()
        except Exception:  # noqa: BLE001 — never block startup on config load
            ss["admin_config"] = default_admin_config()
    ss.setdefault("admin_authenticated", False)
    ss.setdefault("is_admin", False)

    # Flow A — seed the pipeline config from the admin pipeline defaults
    ss.setdefault("hexad_scores", None)
    ss.setdefault("hexad_type", None)
    ss.setdefault("persona", None)
    _pd = ss["admin_config"]["pipeline_defaults"]
    ss.setdefault("config_a", {"retriever": _pd["default_retriever"],
                               "validation": _pd["validation_default"],
                               "substitution": _pd.get("substitution_default", True),
                               "explanation": _pd["explanation_default"]})
    ss.setdefault("config_b", {"retriever": "sparse", "validation": True,
                               "substitution": True, "explanation": False})

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
    """Navigate to a screen and rerun. Use imperatively (inside the script body),
    never from a widget callback — ``st.rerun()`` is forbidden there."""
    st.session_state["screen"] = screen
    st.rerun()


def nav_to(screen: str) -> None:
    """Callback-safe navigation for ``st.button(on_click=...)``: set the target
    screen only. No ``st.rerun()`` — the widget interaction already reruns the app,
    and a callback that sets state (rather than the inline ``if st.button(): goto()``
    pattern) can't be dropped by a concurrent ``run_every`` fragment rerun."""
    st.session_state["screen"] = screen


def log_event(event_type: str, payload: dict | None = None) -> None:
    """Log an interaction event.

    Delegates to the logging service, which enriches to the schema, buffers
    in memory, and pushes to the HF Dataset repo / external DB when configured.
    """
    from .services.logging_service import log_event as _log_event

    _log_event(event_type, payload)

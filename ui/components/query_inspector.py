"""Query inspector (spec §4.4): collapsible panel showing the per-meal query
split, the generated retriever query, and the extracted constraints as chips.

Removable constraints can be deleted (logs ``constraint_removed``); the allergy/
health "Diabetes-friendly" constraint is non-removable (safety, spec §10.2).
"""

from __future__ import annotations

import copy

import streamlit as st

from ..mock_data import CONSTRAINTS, MOCK_QUERY, MOCK_SPLITS
from ..state import log_event
from ..theme import C, F, html

_CHIP_COLORS = {
    "allergy": ("#fef3c7", "#92400e"),
    "nutriscore": (C["primaryPale"], C["primary"]),
    "health": (C["accentPale"], C["accent"]),
    "nutrition": (C["tagBg"], C["textMuted"]),
}


def render_query_inspector(state_key: str) -> None:
    """Render the inspector. ``state_key`` scopes the editable constraint list."""
    ck = f"constraints_{state_key}"
    if ck not in st.session_state:
        st.session_state[ck] = copy.deepcopy(CONSTRAINTS)
    constraints = st.session_state[ck]

    with st.expander("🔍 Query Inspector", expanded=False):
        splits = ""
        for i, s in enumerate(MOCK_SPLITS):
            border = f"border-top:1px solid {C['border']};" if i > 0 else ""
            splits += (
                f"<div style=\"font-family:{F};font-size:8px;color:{C['text']};padding:2px 0;{border}\">"
                f"<b style=\"color:{C['primary']}\">{s['meal']}:</b> {s['q']}</div>"
            )
        html(
            f"<div style=\"font-family:{F};font-size:8px;font-weight:600;color:{C['textMuted']};"
            f"margin-bottom:3px\">Per-meal query split:</div>{splits}"
        )

        html(
            f"<div style=\"font-family:{F};font-size:8px;font-weight:600;color:{C['textMuted']};"
            f"margin:6px 0 3px\">Generated retriever query:</div>"
        )
        st.code(MOCK_QUERY, language="python")

        # Constraint chips (display) + removal controls
        chips = ""
        for c in constraints:
            bg, fg = _CHIP_COLORS.get(c["type"], _CHIP_COLORS["nutrition"])
            chips += (
                f"<span style=\"background:{bg};color:{fg};font-size:9px;padding:2px 7px;border-radius:6px;"
                f"font-family:{F};font-weight:600;margin:2px 3px 2px 0;display:inline-block\">{c['label']}</span>"
            )
        html(
            f"<div style=\"font-family:{F};font-size:8px;font-weight:600;color:{C['textMuted']};"
            f"margin:6px 0 3px\">Constraints:</div><div>{chips}</div>"
        )

        removable = [c for c in constraints if c["removable"]]
        if removable:
            st.caption("Remove a constraint to relax retrieval:")
            cols = st.columns(len(removable))
            for col, c in zip(cols, removable):
                with col:
                    if st.button(f"✕ {c['label']}", key=f"rm_{state_key}_{c['label']}",
                                 use_container_width=True):
                        st.session_state[ck] = [x for x in constraints if x["label"] != c["label"]]
                        log_event("constraint_removed", {"constraint_label": c["label"]})
                        st.rerun()

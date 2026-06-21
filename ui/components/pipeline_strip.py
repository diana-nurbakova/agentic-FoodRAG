"""Pipeline strip: click-to-toggle agents + retriever selector (spec §4.4).

Recommendation (retriever) is always on; Validation and Explanation toggle.
Rendered with native widgets so toggles/selection are captured server-side,
styled to echo the mockup's node strip. Per-agent timings appear after a run.
"""

from __future__ import annotations

import streamlit as st

from ..mock_data import RETRIEVER_LABELS
from ..state import log_event
from ..theme import C, F, html


def _node_html(icon: str, label: str, active: bool, time: float | None, dashed: bool) -> str:
    border = f"2px solid {C['primary']}" if active else f"2px dashed {C['off']}"
    bg = C["primaryPale"] if active else "transparent"
    opacity = 1 if active else 0.5
    lbl_col = C["primary"] if active else C["textMuted"]
    time_html = (
        f"<div style=\"font-family:{F};font-size:8px;color:{C['primaryLight']}\">{time}s</div>" if time else ""
    )
    return (
        f"<div style=\"display:flex;flex-direction:column;align-items:center;gap:1px;padding:4px 8px;"
        f"border-radius:6px;background:{bg};border:{border};opacity:{opacity};min-width:70px\">"
        f"<span style=\"font-size:14px\">{icon}</span>"
        f"<span style=\"font-family:{F};font-size:9px;font-weight:700;color:{lbl_col}\">{label}</span>"
        f"{time_html}</div>"
    )


def render_pipeline_strip(config: dict, key: str, times: dict | None = None,
                          disabled: bool = False) -> dict:
    """Render the strip; returns the (possibly mutated) config.

    Layout: [retriever select] → [Validation toggle] → [Explanation toggle].
    When ``disabled`` (admin locked the pipeline, spec §6.4) the controls are
    read-only.
    """
    arrow = f"<div style=\"display:flex;align-items:center;justify-content:center;height:48px;color:{C['primary']};opacity:0.5\">→</div>"

    col_r, col_a1, col_v, col_a2, col_e = st.columns([3, 0.4, 2, 0.4, 2])

    with col_r:
        html(_node_html("🔍", "Recommend", True, times.get("retriever") if times else None, dashed=False))
        new_retriever = st.selectbox(
            "Retriever",
            options=list(RETRIEVER_LABELS.keys()),
            format_func=lambda k: RETRIEVER_LABELS[k],
            index=list(RETRIEVER_LABELS).index(config["retriever"]),
            key=f"retriever_{key}",
            label_visibility="collapsed",
            disabled=disabled,
        )
        if new_retriever != config["retriever"]:
            config["retriever"] = new_retriever
            log_event("agent_toggled", {"agent": "retriever", "state": new_retriever})

    with col_a1:
        html(arrow)
    with col_a2:
        html(arrow)

    with col_v:
        html(_node_html("✅", "Validation", config["validation"],
                        times.get("validation") if times else None, dashed=not config["validation"]))
        config["validation"] = st.toggle("Validation", value=config["validation"],
                                          key=f"val_{key}", label_visibility="collapsed",
                                          disabled=disabled)

    with col_e:
        html(_node_html("💡", "Explanation", config["explanation"],
                        times.get("explanation") if times else None, dashed=not config["explanation"]))
        config["explanation"] = st.toggle("Explanation", value=config["explanation"],
                                          key=f"exp_{key}", label_visibility="collapsed",
                                          disabled=disabled)

    return config

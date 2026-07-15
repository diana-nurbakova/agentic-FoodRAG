"""Pipeline strip: click-to-toggle agents + retriever selector (spec §4.4).

Recommendation (retriever) and Finalization are always on and shown locked;
Validation, Substitution and Explanation toggle and are wired straight to the
agent graph (ui/services/backend.build_workflow). Rendered with native widgets
so toggles/selection are captured server-side. Per-agent timings appear after a run.
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

    Layout: [Recommend + retriever] → [Validation] → [Substitution] → [Explanation]
    → [Finalize]. Recommend and Finalize are always-on (shown locked); the three
    middle stages toggle and drive ``build_workflow``. When ``disabled`` (admin locked
    the pipeline, spec §6.4) every control is read-only.
    """
    config.setdefault("substitution", True)
    arrow = (f"<div style=\"display:flex;align-items:center;justify-content:center;height:48px;"
             f"color:{C['primary']};opacity:0.5\">→</div>")

    col_r, a1, col_v, a2, col_s, a3, col_e, a4, col_f = st.columns(
        [2.6, 0.3, 1.9, 0.3, 1.9, 0.3, 1.9, 0.3, 1.7])

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

    for col in (a1, a2, a3, a4):
        with col:
            html(arrow)

    for col, icon, label, cfg_key in (
        (col_v, "✅", "Validation", "validation"),
        (col_s, "🔧", "Substitution", "substitution"),
        (col_e, "💡", "Explanation", "explanation"),
    ):
        with col:
            html(_node_html(icon, label, config[cfg_key],
                            times.get(cfg_key) if times else None, dashed=not config[cfg_key]))
            new_val = st.toggle(label, value=config[cfg_key], key=f"{cfg_key}_{key}",
                                label_visibility="collapsed", disabled=disabled)
            if new_val != config[cfg_key]:
                config[cfg_key] = new_val
                log_event("agent_toggled", {"agent": cfg_key, "state": new_val})

    with col_f:
        html(_node_html("📦", "Finalize", True, None, dashed=False))
        st.caption("always on")

    return config

"""Correction Agent log (spec §4.4 / §2.2 Agent 2).

Shows per-meal validation rounds with INVALID/OK status and the relaxation
action taken. Only rendered when the Validation agent is enabled.
"""

from __future__ import annotations

from ..mock_data import CORRECTION_LOG
from ..theme import C, F, html


def render_correction_log() -> None:
    rows = ""
    for i, c in enumerate(CORRECTION_LOG):
        status_col = C["nutA"] if c["status"] == "OK" else C["warn"]
        border = f"border-top:1px solid {C['border']};" if i > 0 else ""
        action = (
            f"<div style=\"color:{C['sub']};font-size:8px;margin-top:1px\">{c['action']}</div>"
            if c.get("action")
            else ""
        )
        rows += (
            f"<div style=\"font-family:{F};font-size:9px;line-height:1.6;padding:4px 0;{border}color:{C['text']}\">"
            f"<span style=\"font-weight:700\">{c['meal']}</span> · Round {c['round']} · "
            f"<span style=\"color:{status_col};font-weight:700;margin-left:4px\">{c['status']}</span>"
            f"<div style=\"color:{C['textMuted']};font-size:8px;margin-top:1px\">{c['reason']}</div>"
            f"{action}</div>"
        )
    html(
        f"<div style=\"background:{C['card']};border:1px solid {C['border']};border-radius:8px;"
        f"padding:10px;margin-bottom:8px\">"
        f"<div style=\"font-family:{F};font-weight:700;font-size:11px;color:{C['text']};margin-bottom:6px\">"
        f"🔄 Correction Agent Log</div>{rows}</div>"
    )

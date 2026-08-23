"""5-point labelled Likert item (ResQue-compatible).

All five points are labelled (1 = Strongly disagree … 5 = Strongly agree),
matching ResQue's validated format. Returns the selected value (1–5) or None.
"""

from __future__ import annotations

import streamlit as st

from ..study_data import LIKERT_5
from ..theme import C, F, html


def render_likert(text: str, key: str, scale: list[str] | None = None) -> int | None:
    """Render one Likert item. Returns 1..len(scale) or None if unanswered."""
    scale = scale or LIKERT_5
    html(
        f"<div style=\"font-family:{F};font-size:13px;color:{C['text']};margin:10px 0 2px;line-height:1.5\">{text}</div>"
    )
    value = st.radio(
        text,
        options=list(range(1, len(scale) + 1)),
        format_func=lambda v: f"{v} · {scale[v - 1]}",
        index=None,
        horizontal=True,
        key=key,
        label_visibility="collapsed",
    )
    return value

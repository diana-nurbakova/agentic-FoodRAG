"""Drag-to-rank widget with up/down arrow fallback.

Streamlit can't capture native HTML5 drag-and-drop server-side, so this uses
an accessible fallback: ▲/▼ buttons to reorder. Position 1 = most
preferred. The order is kept in ``st.session_state[key]`` as a list of item
indices into ``items``.
"""

from __future__ import annotations

import streamlit as st

from ..theme import C, F, html


def render_ranking(items: list[dict], key: str) -> list[int]:
    """Render a reorderable list. ``items`` = [{label, desc}]; returns the order."""
    if key not in st.session_state:
        st.session_state[key] = list(range(len(items)))
    order = st.session_state[key]

    def _swap(pos: int, target: int) -> None:
        order[pos], order[target] = order[target], order[pos]
        st.session_state[key] = order
        st.rerun()

    html(f"<div style=\"font-family:{F};font-size:11px;color:{C['textMuted']};margin-bottom:8px\">"
         f"Use ▲ / ▼ to reorder. Position 1 = most preferred.</div>")

    for pos, item_idx in enumerate(order):
        item = items[item_idx]
        c_num, c_body, c_up, c_down = st.columns([0.6, 6, 0.7, 0.7])
        with c_num:
            html(
                f"<div style=\"width:28px;height:28px;border-radius:50%;background:{C['primaryPale']};"
                f"color:{C['primary']};font-weight:700;font-size:13px;display:flex;align-items:center;"
                f"justify-content:center;margin-top:6px\">{pos + 1}</div>"
            )
        with c_body:
            html(
                f"<div style=\"font-family:{F};font-size:13px;font-weight:600;color:{C['text']}\">{item['label']}</div>"
                f"<div style=\"font-family:{F};font-size:11px;color:{C['textMuted']}\">{item.get('desc', '')}</div>"
            )
        with c_up:
            if st.button("▲", key=f"{key}_up_{pos}", disabled=pos == 0, use_container_width=True):
                _swap(pos, pos - 1)
        with c_down:
            if st.button("▼", key=f"{key}_down_{pos}", disabled=pos == len(order) - 1,
                         use_container_width=True):
                _swap(pos, pos + 1)

    return order

"""Visit analytics dashboard.

Funnel, engagement, temporal and quality-flag metrics for participants
(admin sessions excluded). Mock aggregates + a live overlay from the current
session's event buffer.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from ...admin_analytics import get_analytics
from ...theme import C, F, html


def render() -> None:
    a = get_analytics(st.session_state.get("event_log", []))

    st.subheader("Funnel")
    st.caption("Participant progression (admin sessions excluded).")
    for label, count, pct in a["funnel"]:
        html(
            f"<div style=\"margin-bottom:8px\">"
            f"<div style=\"display:flex;justify-content:space-between;font-family:{F};font-size:12px;"
            f"color:{C['text']};margin-bottom:3px\"><span>{label}</span>"
            f"<span style=\"color:{C['textMuted']}\">{count} · {pct}%</span></div>"
            f"<div style=\"height:8px;background:{C['border']};border-radius:5px;overflow:hidden\">"
            f"<div style=\"height:100%;width:{pct}%;background:{C['primary']};border-radius:5px\"></div></div></div>"
        )

    st.divider()
    st.subheader("Engagement")
    cols = st.columns(len(a["engagement"]))
    for col, (k, v) in zip(cols, a["engagement"].items()):
        col.metric(k, v)

    c1, c2 = st.columns(2)
    with c1:
        st.caption("Retriever usage")
        st.bar_chart(pd.Series(a["retriever_usage"], name="sessions"))
    with c2:
        st.caption("Sessions per day")
        st.bar_chart(pd.Series(a["sessions_per_day"], name="sessions"))

    st.divider()
    st.subheader("Quality flags")
    st.caption("Careless-responding & data-quality indicators.")
    qcols = st.columns(3)
    for i, (k, v) in enumerate(a["quality"].items()):
        qcols[i % 3].metric(k, v)

    st.divider()
    st.subheader("Live session (this browser)")
    st.caption(f"{a['live_total_events']} events in the current in-memory buffer.")
    if a["live_counts"]:
        df = pd.DataFrame(sorted(a["live_counts"].items(), key=lambda x: -x[1]),
                          columns=["event_type", "count"])
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("No events yet — walk through a flow in another tab to populate this.")

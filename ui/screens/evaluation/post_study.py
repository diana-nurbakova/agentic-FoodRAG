"""Step 7: Post-study questionnaire (asked once).

Constructs that apply to the system as a whole, not per plan: explanation
quality, constraint compliance, behavioural intention, usability. Same 5-point
Likert. Grouped by construct.
"""

from __future__ import annotations

import streamlit as st

from ...components.likert_scale import render_likert
from ...components.study_common import render_progress, render_step_nav
from ...state import goto, log_event
from ...study_data import POST_STUDY_ITEMS
from ...theme import C, F, html


def render() -> None:
    render_step_nav("plan_evaluation")
    render_progress(7, 9, "Step 7 of 8 — A few questions about the system")
    html(f"<h2>About the system overall</h2>"
         f"<p style=\"font-family:{F};font-size:13px;color:{C['textMuted']};line-height:1.7;margin:0 0 12px\">"
         f"These questions are about the system as a whole, not any single plan.</p>")

    answers: dict[str, int | None] = {}
    last_construct = None
    for item in POST_STUDY_ITEMS:
        if item["construct"] != last_construct:
            html(f"<div style=\"font-family:{F};font-size:12px;font-weight:600;color:{C['text']};"
                 f"border-left:3px solid {C['primary']};padding-left:8px;margin:18px 0 4px\">{item['construct']}</div>")
            last_construct = item["construct"]
        answers[item["id"]] = render_likert(item["text"], key=f"ps_{item['id']}")

    complete = all(v is not None for v in answers.values())
    if not complete:
        st.caption("Please answer all questions before continuing.")

    if st.button("Continue to final ranking →", type="primary", disabled=not complete,
                 use_container_width=True):
        st.session_state["post_study_answers"] = answers
        # Group by construct for the structured event payload
        by_construct: dict[str, dict[str, int]] = {}
        for item in POST_STUDY_ITEMS:
            by_construct.setdefault(item["construct"], {})[item["id"]] = answers[item["id"]]
        log_event("post_study_completed", by_construct)
        goto("ranking")

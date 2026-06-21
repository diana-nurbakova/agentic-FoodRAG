"""Step 8: Final ranking (spec §5.6).

Rank all plans most→least preferred (▲/▼ fallback for drag), pick the one you'd
actually follow, and leave open-text feedback. Plans are labelled by the order
the participant saw them.
"""

from __future__ import annotations

import streamlit as st

from ...components.ranking_widget import render_ranking
from ...components.study_common import render_progress
from ...state import goto, log_event
from ...study_data import STUDY_CONFIG, STUDY_PLANS
from ...theme import C, F, html


def _seen_plans() -> list[dict]:
    ss = st.session_state
    n_plans = STUDY_CONFIG["plans_per_profile"]
    plans = STUDY_PLANS[ss["assigned_profile_id"]][:n_plans]
    return [plans[ss["plan_order"][i]] for i in range(n_plans)]


def render() -> None:
    render_progress(8, 9, "Step 8 of 8 — Compare all plans")
    seen = _seen_plans()

    html(f"<h2>Compare all the plans</h2>"
         f"<p style=\"font-family:{F};font-size:13px;color:{C['textMuted']};line-height:1.7;margin:0 0 14px\">"
         f"Rank the plans from best to worst, then answer a few final questions.</p>")

    items = [{"label": f"Meal Plan {i + 1}", "desc": " · ".join(m["name"] for m in p["meals"])}
             for i, p in enumerate(seen)]
    order = render_ranking(items, key="rank_order")

    st.divider()
    preferred = st.radio("Which plan would you actually follow?",
                         [f"Meal Plan {i + 1}" for i in range(len(seen))],
                         index=None, key="rank_preferred")

    feedback = st.text_area("Why did you choose your top-ranked plan?",
                            placeholder="What influenced your choice? What made it stand out?",
                            key="rank_feedback")
    other = st.text_area("Anything confusing, unrealistic, or missing? (optional)",
                         placeholder="e.g. Plan 2 had ingredients I can't find easily…",
                         key="rank_other")

    can_submit = preferred is not None and len(feedback.strip()) >= 10
    if not can_submit:
        st.caption("Please pick the plan you'd follow and write at least a short reason for your top choice.")

    if st.button("Submit & see your summary →", type="primary", disabled=not can_submit,
                 use_container_width=True):
        pref_pos = int(preferred.split()[-1]) - 1
        result = {
            "ranking_positions": order,           # seen-position order, best→worst
            "ranking_conds": [seen[i]["cond"] for i in order],
            "preferred_position": pref_pos,
            "preferred_cond": seen[pref_pos]["cond"],
            "feedback_text": feedback.strip(),
            "other_text": other.strip(),
        }
        st.session_state["ranking_result"] = result
        log_event("plans_ranked", {
            "profile_id": st.session_state["assigned_profile_id"],
            "ranking": result["ranking_conds"],
            "preferred_plan": result["preferred_cond"],
            "feedback_text": result["feedback_text"],
        })
        goto("completion")

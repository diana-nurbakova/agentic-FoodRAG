"""Step 3: Profile assignment (between-subjects, spec §5.3).

The profile is **randomly assigned** (round-robin), not chosen — this removes
self-selection bias. The participant sees the assigned profile and its prompt
but cannot pick it. The plan viewing order is set here via the Latin square.
"""

from __future__ import annotations

import streamlit as st

from ...components.study_common import render_progress, render_step_nav
from ...counterbalancing import assign_profile_index, get_plan_order, latin_square_group
from ...state import goto, log_event
from ...study_data import STUDY_CONFIG, get_profile, get_study_profiles
from ...theme import C, F, html


def _ensure_assignment() -> dict:
    """Assign a profile + plan order once per session (idempotent)."""
    ss = st.session_state
    if ss.get("assigned_profile_id") is None:
        profiles = get_study_profiles()
        pnum = ss["participant_number"]
        idx = assign_profile_index(pnum, len(profiles))
        ss["assigned_profile_id"] = profiles[idx]["id"]
        n_plans = STUDY_CONFIG["plans_per_profile"]
        ss["plan_order"] = get_plan_order(pnum, n_plans)
        ss["latin_square_group"] = latin_square_group(pnum, n_plans)
    return get_profile(ss["assigned_profile_id"])


def render() -> None:
    profile = _ensure_assignment()
    render_step_nav("demographics")
    render_progress(3, 9, "Step 3 of 8 — Your assigned profile")

    html(f"<h2>You've been assigned a profile</h2>"
         f"<p style=\"font-family:{F};font-size:13px;color:{C['textMuted']};line-height:1.7;margin:0 0 14px\">"
         f"To keep the study fair, each participant is <b>randomly assigned</b> one profile to evaluate "
         f"(you don't choose it). Please role-play this profile when rating the meal plans.</p>")

    tags = "".join(
        f"<span style=\"background:{C['primaryPale']};color:{C['primary']};font-size:11px;padding:2px 8px;"
        f"border-radius:20px;font-weight:500;margin:2px\">{t}</span>" for t in profile["tags"]
    )
    html(
        f"<div style=\"background:{C['card']};border:1.5px solid {C['primary']};border-radius:16px;padding:20px\">"
        f"<div style=\"font-size:30px;margin-bottom:8px\">{profile['icon']}</div>"
        f"<div style=\"font-family:{F};font-size:18px;font-weight:700;color:{C['text']}\">{profile['name']}</div>"
        f"<div style=\"font-family:{F};font-size:13px;color:{C['primary']};font-weight:600;margin:2px 0 8px\">{profile['goal']}</div>"
        f"<div style=\"font-family:{F};font-size:13px;color:{C['textMuted']};line-height:1.6\">{profile['desc']}</div>"
        f"<div style=\"margin-top:10px;display:flex;flex-wrap:wrap;gap:4px\">{tags}</div></div>"
    )
    html(
        f"<div style=\"background:{C['primaryPale']};border:1px solid {C['primary']};border-radius:10px;"
        f"padding:14px;margin-top:14px\">"
        f"<div style=\"font-family:{F};font-size:12px;color:{C['primary']};font-weight:600;margin-bottom:4px\">"
        f"Profile prompt sent to the system</div>"
        f"<div style=\"font-family:{F};font-size:13px;color:{C['textMuted']};line-height:1.65;font-style:italic\">"
        f"\"{profile['prompt']}\"</div></div>"
    )

    if st.button("Start the evaluation →", type="primary", use_container_width=True):
        st.session_state["plan_index"] = 0
        st.session_state["plan_answers"] = []
        st.session_state["attention_results"] = []
        log_event("profile_assigned", {
            "profile_id": profile["id"],
            "profile_prompt": profile["prompt"],
            "assignment_group": st.session_state["latin_square_group"],
        })
        goto("plan_evaluation")

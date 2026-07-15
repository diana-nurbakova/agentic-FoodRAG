"""Step 9: Completion (spec §5.9).

Summary of the session: profile assigned (not chosen), plans evaluated,
attention-check pass/fail, average rating per plan, final ranking, and a
toggleable researcher data view. Logs ``study_completed`` once.
"""

from __future__ import annotations

import json

import streamlit as st

from ...components.study_common import render_progress
from ...state import log_event
from ...study_data import STUDY_CONFIG, get_profile, load_study_plans
from ...theme import C, F, html


def _avg(values: list[int]) -> float:
    return round(sum(values) / len(values), 1) if values else 0.0


def render() -> None:
    ss = st.session_state
    profile = get_profile(ss["assigned_profile_id"])
    n_plans = STUDY_CONFIG["plans_per_profile"]
    plans = load_study_plans(ss["assigned_profile_id"], n_plans)
    seen = [plans[ss["plan_order"][i]] for i in range(n_plans)]
    plan_answers = ss.get("plan_answers", [])
    passed = sum(1 for x in ss.get("attention_results", []) if x)
    failed = len(ss.get("attention_results", [])) - passed

    if not ss.get("study_completed_logged"):
        log_event("study_completed", {
            "plans_seen": n_plans,
            "completion_status": "complete",
            "attention_checks_passed": passed,
            "attention_checks_failed": failed,
        })
        ss["study_completed_logged"] = True
        # Push the full session log on completion (spec §9.1 "on session end")
        from ...services.logging_service import flush_now

        flush_now(ss["session_id"])

    render_progress(9, 9, "Study complete")
    html(
        f"<div style=\"text-align:center;padding:8px 0 20px\">"
        f"<div style=\"font-size:48px\">🌿</div>"
        f"<h1 style=\"margin:6px 0\">Thank you!</h1>"
        f"<p style=\"font-family:{F};font-size:14px;color:{C['textMuted']};line-height:1.7;max-width:460px;margin:0 auto\">"
        f"Your responses help us understand how different system designs affect perceived quality and trust "
        f"in food recommendations.</p></div>"
    )

    # Session recap
    html(
        f"<div style=\"background:{C['primaryPale']};border:1px solid {C['primary']};border-radius:12px;"
        f"padding:16px;display:flex;gap:14px;align-items:center;margin-bottom:16px\">"
        f"<div style=\"font-size:30px\">{profile['icon']}</div><div>"
        f"<div style=\"font-family:{F};font-size:11px;font-weight:600;color:{C['primary']};text-transform:uppercase;"
        f"letter-spacing:0.5px\">Assigned profile (not chosen)</div>"
        f"<div style=\"font-family:{F};font-size:15px;font-weight:600;color:{C['text']}\">{profile['name']} — {profile['goal']}</div>"
        f"<div style=\"font-family:{F};font-size:12px;color:{C['textMuted']};margin-top:2px\">"
        f"Session {ss['session_id']} · {n_plans} plans evaluated · "
        f"attention checks: {passed} passed, {failed} failed</div></div></div>"
    )

    # Average rating per plan (seen order)
    html(f"<div style=\"font-family:{F};font-size:11px;font-weight:600;color:{C['textMuted']};"
         f"text-transform:uppercase;letter-spacing:0.8px;margin-bottom:8px\">Your average rating per plan</div>")
    avgs = []
    for i in range(len(seen)):
        vals = list(plan_answers[i]["likert_responses"].values()) if i < len(plan_answers) else []
        avgs.append(_avg(vals))
    max_avg = max(avgs) if avgs else 0
    ranking_positions = ss.get("ranking_result", {}).get("ranking_positions", [])
    top = ranking_positions[0] if ranking_positions else None
    pref = ss.get("ranking_result", {}).get("preferred_position")
    for i, avg in enumerate(avgs):
        pct = avg / 5 * 100
        bar = C["primary"] if avg == max_avg else C["primaryLight"]
        badges = ""
        if i == top:
            badges += f"<span style=\"background:{C['primaryPale']};color:{C['primary']};font-size:10px;padding:1px 7px;border-radius:20px;margin-left:6px\">Top ranked</span>"
        if i == pref:
            badges += f"<span style=\"background:{C['accentPale']};color:{C['accent']};font-size:10px;padding:1px 7px;border-radius:20px;margin-left:6px\">Would follow</span>"
        html(
            f"<div style=\"margin-bottom:10px\">"
            f"<div style=\"display:flex;justify-content:space-between;align-items:center;margin-bottom:4px\">"
            f"<span style=\"font-family:{F};font-size:13px;font-weight:600;color:{C['text']}\">Meal Plan {i + 1}{badges}</span>"
            f"<span style=\"font-family:{F};font-size:13px;font-weight:700;color:{bar}\">{avg} / 5</span></div>"
            f"<div style=\"height:10px;background:{C['border']};border-radius:6px;overflow:hidden\">"
            f"<div style=\"height:100%;width:{pct:.0f}%;background:{bar};border-radius:6px\"></div></div></div>"
        )

    # Researcher data view (toggleable, spec §5.9)
    with st.expander("🔬 Recorded response data (researcher view)"):
        record = {
            "session_id": ss["session_id"],
            "flow": "evaluation",
            "demographics": ss.get("demographics"),
            "profile_id": ss["assigned_profile_id"],
            "latin_square_group": ss.get("latin_square_group"),
            "plan_order_conds": [s["cond"] for s in seen],
            "plan_order_configs": [s.get("config") for s in seen],
            "plan_answers": plan_answers,
            "post_study": ss.get("post_study_answers"),
            "ranking": ss.get("ranking_result"),
            "attention": {"passed": passed, "failed": failed},
        }
        st.code(json.dumps(record, indent=2, ensure_ascii=False), language="json")

    if st.button("Start over", use_container_width=True):
        for k in list(ss.keys()):
            del ss[k]
        st.rerun()

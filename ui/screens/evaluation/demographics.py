"""Step 2: Demographics (GDPR-compliant, spec §5.2).

Only non-sensitive, grouped fields. No gender, ethnicity, religion, health
conditions, real dietary restrictions, or exact age (spec §7.2).
"""

from __future__ import annotations

import streamlit as st

from ...components.study_common import render_progress
from ...state import goto, log_event
from ...study_data import DEMO_AGE, DEMO_COOK_FREQ, DEMO_COOK_SKILL, DEMO_COUNTRIES, DEMO_EMPLOYMENT
from ...theme import C, F, html


def render() -> None:
    render_progress(2, 9, "Step 2 of 8 — About you")
    html(f"<h2>A few quick questions</h2>"
         f"<p style=\"font-family:{F};font-size:13px;color:{C['textMuted']};line-height:1.7;margin:0 0 14px\">"
         f"These help us check whether results differ across participant groups. All answers are anonymous.</p>")

    html(
        f"<div style=\"background:#fdf3e3;border:1px solid #f0d098;border-radius:8px;padding:12px 14px;"
        f"margin-bottom:16px;display:flex;gap:10px;align-items:flex-start\">"
        f"<span style=\"font-size:16px;flex-shrink:0\">🔒</span>"
        f"<span style=\"font-family:{F};font-size:12px;color:#92510a;line-height:1.6\">"
        f"No sensitive personal data is collected (no health information, religion, or precise age). "
        f"Responses are stored anonymously and processed in aggregate only.</span></div>"
    )

    age = st.radio("What is your age range?", DEMO_AGE, index=None, horizontal=True, key="demo_age")
    country = st.selectbox("Country of residence", DEMO_COUNTRIES, index=None,
                           placeholder="Select a country…", key="demo_country")
    employment = st.radio("What best describes your current situation?", DEMO_EMPLOYMENT,
                          index=None, horizontal=True, key="demo_employment")
    cook_freq = st.radio("How often do you cook at home?", DEMO_COOK_FREQ,
                         index=None, horizontal=True, key="demo_cook_freq")
    cook_skill = st.radio("How would you rate your cooking skills?", DEMO_COOK_SKILL,
                          index=None, horizontal=True, key="demo_cook_skill")

    complete = all([age, country, employment, cook_freq, cook_skill])
    if not complete:
        st.caption("Please answer all questions to continue.")

    if st.button("Continue →", type="primary", disabled=not complete, use_container_width=True):
        demographics = {"age_range": age, "country": country, "employment": employment,
                        "cooking_frequency": cook_freq, "cooking_skill": cook_skill}
        st.session_state["demographics"] = demographics
        log_event("demographics_completed", demographics)
        goto("profile_assignment")

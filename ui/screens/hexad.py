"""Screen 2: HEXAD-12 questionnaire (spec §4.2).

Tondello et al. (2016) 12-item short scale, 7-point Likert, 2 items per type.
Items are ordered so that item ``i`` scores HEXAD type ``i % 6``. Scoring:
sum per type (max 14); dominant type stored for personalisation/logging.
"""

from __future__ import annotations

import streamlit as st

from ..components.study_common import render_step_nav
from ..mock_data import HEXAD_Q
from ..state import goto, log_event
from ..theme import C, F, FP, html

HEXAD_TYPES = ["Player", "Philanthropist", "Free Spirit", "Socialiser", "Achiever", "Disruptor"]


def _score(answers: list[int]) -> tuple[dict[str, int], str]:
    scores = {t: 0 for t in HEXAD_TYPES}
    for i, val in enumerate(answers):
        scores[HEXAD_TYPES[i % 6]] += val
    dominant = max(scores, key=scores.get)
    return scores, dominant


def render() -> None:
    render_step_nav(back_screen="consent", home_screen="welcome")
    html(
        f"<h2 style=\"font-family:{FP};font-size:20px;color:{C['primary']};margin:0 0 3px\">Player Profile (HEXAD)</h2>"
        f"<p style=\"font-family:{F};font-size:10px;color:{C['textMuted']};margin:0 0 12px\">"
        f"Rate each statement 1 (strongly disagree) → 7 (strongly agree). Tondello et al. 12-item scale.</p>"
    )

    if "hexad_answers" not in st.session_state:
        st.session_state["hexad_answers"] = [0] * 12
    answers = st.session_state["hexad_answers"]

    answered = sum(1 for a in answers if a > 0)
    st.progress(answered / 12, text=f"{answered} / 12 answered")

    options = [1, 2, 3, 4, 5, 6, 7]
    for i, q in enumerate(HEXAD_Q):
        html(
            f"<div style=\"font-family:{F};font-size:12px;color:{C['text']};margin:8px 0 2px\">"
            f"<span style=\"color:{C['textMuted']};font-size:10px;margin-right:4px\">{i + 1}.</span>{q}</div>"
        )
        current = answers[i] if answers[i] > 0 else None
        choice = st.radio(
            q, options, index=options.index(current) if current else None,
            key=f"hexad_{i}", horizontal=True, label_visibility="collapsed",
        )
        answers[i] = choice or 0

    done = all(a > 0 for a in answers)
    if st.button("Continue →", type="primary", disabled=not done, use_container_width=True):
        scores, dominant = _score(answers)
        st.session_state["hexad_scores"] = scores
        st.session_state["hexad_type"] = dominant
        log_event("hexad_completed", {"scores": scores, "dominant_type": dominant})
        goto("persona")

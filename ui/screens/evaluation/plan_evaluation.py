"""Steps 4–6: Plan evaluation (spec §5.4).

Each participant evaluates N plans (default 3) for the assigned profile, in
Latin-square order. Per plan: view the 4-meal plan + explanation, then rate 7
ResQue-based items + 1 attention check (5-point Likert) at a rotated position.
"""

from __future__ import annotations

import streamlit as st

from ...components.attention_check import render_attention_check
from ...components.likert_scale import render_likert
from ...components.recipe_detail import recipe_detail_dialog
from ...components.study_common import (
    meal_plan_html,
    render_explanation,
    render_progress,
    render_step_nav,
)
from ...state import goto, log_event
from ...study_data import ATTENTION_ITEM, STUDY_CONFIG, load_study_plans, plan_eval_items_for
from ...theme import C, F, html


def _render_recipe_details(plan: dict, plan_index: int) -> None:
    """Per-recipe 'View details' buttons that open the ingredients/steps/food.com
    modal — participants need ingredients + steps to judge whether a recipe is easy
    or hard. Only shown for recipes that actually carry that detail (real plans)."""
    recipes = [r for r in plan.get("recipes", []) if r.get("ingredients") or r.get("steps")]
    if not recipes:
        return
    html(f"<div style=\"font-family:{F};font-size:11px;font-weight:600;color:{C['textMuted']};"
         f"text-transform:uppercase;letter-spacing:0.8px;margin:14px 0 4px\">"
         f"See ingredients &amp; steps</div>")
    cols = st.columns(len(recipes))
    for col, recipe in zip(cols, recipes):
        with col:
            label = f"{recipe.get('img', '📖')} {recipe.get('meal', 'Recipe')}"
            if st.button(label, key=f"detail_{plan_index}_{recipe.get('meal', '')}",
                         use_container_width=True):
                recipe_detail_dialog(recipe)


def render() -> None:
    ss = st.session_state
    n_plans = STUDY_CONFIG["plans_per_profile"]
    plan_index = ss["plan_index"]
    profile_id = ss["assigned_profile_id"]

    plans = load_study_plans(profile_id, n_plans)
    cond_index = ss["plan_order"][plan_index]
    if not plans or cond_index >= len(plans):
        render_step_nav(back_screen="profile_assignment")
        st.error(f"No meal plans are available for profile `{profile_id}`. "
                 "An admin needs to run `scripts/build_study_plans.py` to generate them.")
        return
    plan = plans[cond_index]

    def _back() -> None:
        # Step back to the previous meal plan, or out to the profile screen.
        # Callback-safe (render_step_nav on_click): set state only, no st.rerun().
        if plan_index > 0:
            ss["plan_index"] = plan_index - 1
            ss["screen"] = "plan_evaluation"
        else:
            ss["screen"] = "profile_assignment"

    render_step_nav(on_back=_back)
    render_progress(4 + plan_index, 9,
                    f"Step {4 + plan_index} of 8 — Meal plan {plan_index + 1} of {n_plans}")
    html(f"<h2>Meal Plan {plan_index + 1}</h2>"
         f"<p style=\"font-family:{F};font-size:13px;color:{C['textMuted']};line-height:1.7;margin:0 0 12px\">"
         f"Review this plan and its explanation, then rate each statement below.</p>")

    html(meal_plan_html(plan))
    _render_recipe_details(plan, plan_index)
    render_explanation(plan)

    html(f"<div style=\"font-family:{F};font-size:11px;font-weight:600;color:{C['textMuted']};"
         f"text-transform:uppercase;letter-spacing:0.8px;margin:20px 0 4px\">Rate this meal plan</div>")

    # 7 ResQue items + attention check at this plan's rotated position
    items = plan_eval_items_for(plan_index)
    answers: dict[str, int | None] = {}
    for item in items:
        wkey = f"pe_{plan_index}_{item['id']}"
        if item["id"] == ATTENTION_ITEM["id"]:
            val, _ = render_attention_check(wkey)
        else:
            val = render_likert(item["text"], key=wkey)
        answers[item["id"]] = val

    complete = all(v is not None for v in answers.values())
    if not complete:
        st.caption("Please answer all questions before continuing.")

    last = plan_index + 1 >= n_plans
    label = "Continue to post-study questions →" if last else f"Continue to plan {plan_index + 2} →"
    if st.button(label, type="primary", disabled=not complete, use_container_width=True):
        attention_passed = answers.get(ATTENTION_ITEM["id"]) == ATTENTION_ITEM["expected"]
        likert_responses = {k: v for k, v in answers.items() if k != ATTENTION_ITEM["id"]}

        # Record by position (not append) so navigating Back and re-submitting a
        # plan overwrites its entry rather than duplicating it.
        record = {
            "plan_position_seen": plan_index,
            "plan_cond": plan["cond"],
            "plan_config": plan.get("config"),  # hidden in GUI, recorded for analysis
            "likert_responses": likert_responses,
            "attention_check_passed": attention_passed,
        }
        for store, value in ((ss["plan_answers"], record), (ss["attention_results"], attention_passed)):
            while len(store) <= plan_index:
                store.append(None)
            store[plan_index] = value

        log_event("plan_evaluated", {
            "profile_id": profile_id,
            "plan_cond": plan["cond"],
            "plan_config": plan.get("config"),
            "plan_position_seen": plan_index,
            "latin_square_group": ss["latin_square_group"],
            "likert_responses": likert_responses,
            "attention_check_passed": attention_passed,
        })
        if not attention_passed:
            log_event("attention_check_failed", {
                "plan_index": plan_index,
                "expected_response": ATTENTION_ITEM["expected"],
                "actual_response": answers.get(ATTENTION_ITEM["id"]),
            })

        if last:
            goto("post_study")
        else:
            ss["plan_index"] = plan_index + 1
            goto("plan_evaluation")

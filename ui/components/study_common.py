"""Shared Flow B visuals: step progress bar, meal-plan card, explanation panel."""

from __future__ import annotations

from typing import Callable, Optional

import streamlit as st

from ..state import nav_to
from ..theme import C, F, NUTRI_C, html

# Flow B step labels (spec §5.1). Plan-evaluation steps are expanded by caller.
FLOW_B_STEPS = ["Consent", "About you", "Your profile", "Plan 1", "Plan 2", "Plan 3",
                "Post-study", "Ranking", "Done"]


def render_step_nav(back_screen: Optional[str] = None, *,
                    on_back: Optional[Callable[[], None]] = None,
                    home_screen: str = "welcome") -> None:
    """Top-of-screen navigation: a Home button and an optional Back button.

    Shared by every wizard screen (Flow A and Flow B). Uses ``on_click`` callbacks
    rather than the inline ``if st.button(): goto(...)`` pattern, so a concurrent
    ``run_every`` fragment rerun (e.g. the live-backend warm poller) can't land
    between the click and its handling and drop the button's value.

    Widget state persists by key across reruns, so moving Back then forward keeps
    the participant's answers. Pass ``back_screen`` for a plain previous step, or
    ``on_back`` for custom logic (e.g. stepping between meal plans) — the latter
    MUST be callback-safe (mutate ``st.session_state`` only, never call
    ``st.rerun()``/``goto``; the click already triggers the rerun)."""
    left, right, _ = st.columns([1.1, 1.1, 3.8])
    with left:
        st.button("🏠 Home", key="nav_home", use_container_width=True,
                  on_click=nav_to, args=(home_screen,))
    with right:
        if on_back is not None:
            st.button("← Back", key="nav_back", use_container_width=True, on_click=on_back)
        elif back_screen is not None:
            st.button("← Back", key="nav_back", use_container_width=True,
                      on_click=nav_to, args=(back_screen,))


def render_progress(step: int, total: int, label: str) -> None:
    """Thin progress bar + step dots (1-indexed ``step``)."""
    pct = min(step / total * 100, 100)
    dots = ""
    for i in range(total):
        if i < step - 1:
            bg, size = C["primary"], 7
        elif i == step - 1:
            bg, size = C["primaryLight"], 9
        else:
            bg, size = C["border"], 7
        dots += f"<div style=\"width:{size}px;height:{size}px;border-radius:50%;background:{bg}\"></div>"
    html(
        f"<div style=\"font-family:{F};font-size:11px;color:{C['textMuted']};margin-bottom:6px;"
        f"letter-spacing:0.3px\">{label}</div>"
        f"<div style=\"height:3px;background:{C['border']};border-radius:4px\">"
        f"<div style=\"height:3px;width:{pct:.0f}%;background:{C['primary']};border-radius:4px\"></div></div>"
        f"<div style=\"display:flex;gap:6px;margin-top:8px;align-items:center\">{dots}</div>"
    )


def _nutri_pill(score: str) -> str:
    fg = "#333" if score == "C" else "#fff"
    return (f"<span style=\"font-size:10px;font-weight:700;color:{fg};background:{NUTRI_C[score]};"
            f"padding:1px 6px;border-radius:4px\">{score}</span>")


def _plan_recipes(plan: dict) -> list[dict]:
    """Normalise a plan to a list of meal rows (name/img/time/cal/nutri).

    Supports the unified real-plan shape (``recipes``: full cards with per-serving
    nutrition) and the legacy mock shape (``meals``: type/icon/name/time/kcal/nutri)."""
    if plan.get("recipes"):
        return [{"type": r.get("meal", ""), "icon": r.get("img", "🍽"), "name": r.get("name", ""),
                 "time": r.get("time", "—"), "kcal": r.get("cal", 0),
                 "nutri": r.get("nutriscore", "C"), "serving_size_g": r.get("serving_size_g")}
                for r in plan["recipes"]]
    return plan.get("meals", [])


def meal_plan_html(plan: dict) -> str:
    """Render the 4-meal plan card with name, total time, per-serving kcal, Nutri-Score."""
    meals = _plan_recipes(plan)
    rows = ""
    for i, m in enumerate(meals):
        border = f"border-top:1px solid {C['border']};" if i > 0 else ""
        serving_g = m.get("serving_size_g")
        serving_note = (f"<span style=\"color:{C['textMuted']};opacity:0.8\">/ serving"
                        f"{f' (~{serving_g} g)' if serving_g else ''}</span>")
        rows += (
            f"<div style=\"display:flex;gap:12px;padding:12px 0;{border}align-items:flex-start\">"
            f"<div style=\"font-size:24px;width:40px;text-align:center;flex-shrink:0\">{m['icon']}</div>"
            f"<div style=\"flex:1\">"
            f"<div style=\"font-size:10px;font-weight:600;color:{C['textMuted']};text-transform:uppercase;"
            f"letter-spacing:0.8px;margin-bottom:2px\">{m['type']}</div>"
            f"<div style=\"font-family:{F};font-size:14px;font-weight:600;color:{C['text']};margin-bottom:3px\">{m['name']}</div>"
            f"<div style=\"font-family:{F};font-size:12px;color:{C['textMuted']};display:flex;gap:10px;flex-wrap:wrap;align-items:center\">"
            f"<span>⏱ {m['time']}</span><span>🔥 {m['kcal']} kcal {serving_note}</span>"
            f"<span>Nutri-Score {_nutri_pill(m['nutri'])}</span></div></div></div>"
        )
    total_kcal = sum(m["kcal"] for m in meals)
    return (
        f"<div style=\"background:{C['card']};border:1px solid {C['border']};border-radius:12px;padding:6px 18px\">{rows}</div>"
        f"<div style=\"text-align:right;font-family:{F};font-size:12px;color:{C['textMuted']};font-style:italic;"
        f"margin-top:6px\">{total_kcal} kcal/day (per serving)</div>"
    )


def render_explanation(plan: dict) -> None:
    """Collapsible 'why was this recommended' panel (spec §5.4)."""
    expl = plan.get("explanation")
    if not expl:
        return
    with st.expander("💡 Why was this plan recommended?", expanded=False):
        html(f"<p style=\"font-family:{F};font-size:13px;line-height:1.7;color:{C['text']};margin:0 0 8px\">{expl['summary']}</p>")
        for r in expl.get("reasons", []):
            html(
                f"<div style=\"display:flex;gap:8px;margin-top:8px;align-items:flex-start\">"
                f"<span style=\"font-size:14px;flex-shrink:0\">{r['icon']}</span>"
                f"<span style=\"font-family:{F};font-size:12px;color:{C['textMuted']};line-height:1.6\">{r['text']}</span></div>"
            )

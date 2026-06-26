"""Screen 4: Main interface (spec §4.4).

Pipeline strip, query input, simulated processing, query inspector, correction
log, three alternative plans × four meals, recipe cards with triple scoring,
recipe detail modal, explanation tab, KG path, and arena mode (compare two
pipeline configs). Data is mocked (spec §12 Phase 2 wires the real backend).
"""

from __future__ import annotations

import time

import streamlit as st

from ..admin_config import pipeline_defaults
from ..components.correction_log import render_correction_log
from ..components.kg_viz import render_recipe_paths
from ..components.pipeline_strip import render_pipeline_strip
from ..components.query_inspector import render_query_inspector
from ..components.recipe_card import render_recipe_card
from ..components.recipe_detail import recipe_detail_dialog
from ..mock_data import (
    MEALS,
    PLAN_META,
    RECIPES_SETS,
    RETRIEVER_LABELS,
    RETRIEVER_LATENCY,
)
from ..services import rag_service
from ..state import log_event
from ..theme import C, F, html


# ── helpers ───────────────────────────────────────────────────────────────
def _config_sig(cfg: dict) -> str:
    return f"{cfg['retriever']}|{cfg['validation']}|{cfg['explanation']}"


def _simulate_run(config: dict) -> dict:
    """Fake the agent pipeline latency breakdown (spec §2.3 estimates)."""
    base = RETRIEVER_LATENCY.get(config["retriever"], 3.5)
    val = 1.2 if config["validation"] else 0.0
    exp = 2.0 if config["explanation"] else 0.0
    total = base + val + exp

    # Brief animated timer (scaled down so the demo stays responsive)
    placeholder = st.empty()
    steps = 12
    for i in range(steps + 1):
        shown = total * i / steps
        placeholder.markdown(
            f"<div style=\"background:{C['card']};border:1px solid {C['border']};border-radius:6px;"
            f"padding:10px;text-align:center\">"
            f"<div style=\"font-family:{F};font-size:18px;color:{C['primary']};font-weight:700\">{shown:.1f}s</div>"
            f"<div style=\"font-family:{F};font-size:8px;color:{C['textMuted']}\">Processing…</div></div>",
            unsafe_allow_html=True,
        )
        time.sleep(1.0 / steps)
    placeholder.empty()

    return {
        "retriever": round(base, 1),
        "validation": round(val, 1) if val else None,
        "explanation": round(exp, 1) if exp else None,
        "total": round(total, 1),
    }


def _stats_line(config: dict, times: dict) -> None:
    calls = 9 if config["validation"] and config["explanation"] else 6 if config["validation"] else 4
    tokens = "4,820" if config["validation"] and config["explanation"] else "3,100" if config["validation"] else "1,950"
    extra = ""
    if times.get("validation"):
        extra += f" · Val: {times['validation']}s"
    if times.get("explanation"):
        extra += f" · Expl: {times['explanation']}s"
    no_val = "" if config["validation"] else f"<span style=\"color:{C['warn']}\"> · ⚠ No validation</span>"
    html(
        f"<div style=\"font-family:{F};font-size:9px;color:{C['textMuted']};margin-bottom:6px;line-height:1.6\">"
        f"✅ <b>{times['total']}s</b> · Ret: {times['retriever']}s{extra}{no_val}<br>"
        f"📊 LLM calls: <b>{calls}</b> · Tokens: <b>~{tokens}</b></div>"
    )


def _summary_panel_html(data: dict) -> str:
    cells = ""
    for s in (
        {"l": "Total Cal", "v": data["cal"], "i": "🔥"},
        {"l": "Avg NS", "v": data["ns"], "i": "⭐"},
        {"l": "Protein", "v": data["prot"], "i": "💪"},
        {"l": "Fiber", "v": data["fiber"], "i": "🌿"},
        {"l": "Skill", "v": data["skill"], "i": "👨‍🍳"},
    ):
        cells += (
            f"<div style=\"text-align:center\"><div style=\"font-size:12px\">{s['i']}</div>"
            f"<div style=\"font-family:{F};font-weight:700;font-size:11px;color:{C['primary']}\">{s['v']}</div>"
            f"<div style=\"font-family:{F};font-size:7px;color:{C['textMuted']}\">{s['l']}</div></div>"
        )
    return (
        f"<div style=\"background:{C['primaryPale']};border-radius:6px;padding:8px;margin:6px 0;"
        f"display:flex;justify-content:space-around\">{cells}</div>"
    )


def _open_detail(recipe: dict) -> None:
    log_event("recipe_expanded", {"recipe_name": recipe["name"]})
    recipe_detail_dialog(recipe)


# ── pipeline results panel (one per side) ──────────────────────────────────
def _live_expander():
    """FoodOn descendant-expander from the live engine for ontology-grounded
    allergen paths; None (substring) if the backend isn't built."""
    try:
        from ..services import backend, eval_metrics

        if backend._ENGINE is None:  # don't trigger the heavy load just for paths
            return None
        return eval_metrics.foodon_expander(backend.get_engine().hummus_data)
    except Exception:  # noqa: BLE001
        return None


def _render_safety_panel(safety: dict | None, side: str) -> None:
    """Deterministic safety summary for the live plan (protocol §4.3)."""
    if not safety or not safety.get("n"):
        return
    # Log once per result (keyed by side + plan size)
    sig = f"{side}:{safety['n']}:{safety['allergen_rate']:.2f}:{safety['hard_rate']:.2f}"
    if st.session_state.get(f"safety_logged_{side}") != sig:
        log_event("safety_check", {
            "allergen_violation_rate": round(safety["allergen_rate"], 3),
            "hard_constraint_violation_rate": round(safety["hard_rate"], 3),
            "constraint_satisfaction_rate": round(safety["satisfaction_rate"], 3),
            "foodon": safety.get("foodon", False),
        })
        st.session_state[f"safety_logged_{side}"] = sig

    bl = safety.get("blacklist") or []
    cons = safety.get("constraints") or []
    if not bl and not cons:
        html(f"<div style=\"font-family:{F};font-size:9px;color:{C['textMuted']};margin-bottom:6px\">"
             f"🛡 No hard allergens or nutrient constraints set for this persona.</div>")
        return

    allergen_ct = sum(1 for p in safety["per_recipe"] if p["allergens"])
    a_ok = allergen_ct == 0
    a_col = C["nutA"] if a_ok else C["nutE"]
    sat = safety["satisfaction_rate"]
    foodon = " · FoodOn" if safety.get("foodon") else ""
    checks = ", ".join(bl + cons)
    html(
        f"<div style=\"background:{'#eaf7ef' if a_ok and sat>=0.999 else C['accentPale']};"
        f"border:1px solid {a_col};border-radius:8px;padding:8px 10px;margin-bottom:8px\">"
        f"<div style=\"font-family:{F};font-weight:700;font-size:11px;color:{C['text']};margin-bottom:4px\">"
        f"🛡 Safety check (deterministic{foodon})</div>"
        f"<div style=\"display:flex;gap:14px;flex-wrap:wrap;font-family:{F};font-size:10px\">"
        f"<span>Allergen violations: <b style=\"color:{a_col}\">{allergen_ct}/{safety['n']}</b> (target 0)</span>"
        f"<span>Constraint satisfaction: <b style=\"color:{C['nutA'] if sat>=0.999 else C['warn']}\">{sat:.0%}</b></span>"
        f"</div>"
        f"<div style=\"font-family:{F};font-size:8px;color:{C['textMuted']};margin-top:3px\">Checked: {checks}</div></div>"
    )


def _render_recipe_safety_badge(safety: dict | None) -> None:
    if not safety or safety.get("ok", True):
        return
    parts = []
    if safety.get("allergens"):
        parts.append("⚠ contains " + ", ".join(safety["allergens"]))
    if safety.get("constraints"):
        parts.append("exceeds " + ", ".join(safety["constraints"]))
    html(
        f"<div style=\"background:{C['accentPale']};border:1px solid {C['accent']};border-radius:5px;"
        f"padding:3px 7px;margin-bottom:3px;font-family:{F};font-size:9px;color:{C['accent']};font-weight:600\">"
        f"🚨 Safety: {' · '.join(parts)}</div>"
    )


def _render_real_results(side: str, arena_mode: bool) -> None:
    """Live-backend variant: one generated plan from the agentic RAG pipeline."""
    cfg_key = f"config_{side.lower()}"
    config = st.session_state[cfg_key]
    locked = not pipeline_defaults()["user_can_change_pipeline"]
    config = render_pipeline_strip(config, key=side, disabled=locked)
    st.session_state[cfg_key] = config

    if config["retriever"] == "fusion":
        st.caption("ℹ Fusion runs reciprocal-rank fusion over dense + sparse "
                   "(+ Neo4j KG when available, per protocol §5.5).")

    res_key, err_key = f"real_res_{side}", f"real_err_{side}"

    if st.button("▶ Run live pipeline", key=f"runreal_{side}", type="primary",
                 use_container_width=True):
        persona = st.session_state["persona"]
        query = st.session_state.get("query_text", "")
        log_event("pipeline_started", {"config": dict(config), "backend": "real"})
        st.session_state.pop(err_key, None)
        try:
            with st.spinner("Running the agentic pipeline (retrieve → validate → explain)… "
                            "first run also loads HUMMUS, so it can take a while."):
                result = rag_service.generate_meal_plan(persona, query, config)
            st.session_state[res_key] = result
            log_event("pipeline_completed",
                      {"backend": "real", "llm_calls": result["llm_calls"],
                       "elapsed": result["elapsed"]})
        except Exception as exc:  # noqa: BLE001
            st.session_state[err_key] = f"{type(exc).__name__}: {exc}"
        st.rerun()

    if st.session_state.get(err_key):
        st.error(f"Live pipeline failed: {st.session_state[err_key]}")
        st.caption("Check the live-backend status in the admin panel, or turn the live "
                   "backend off to use mock data.")
        return

    result = st.session_state.get(res_key)
    if not result:
        st.info("Configure the pipeline and click **Run live pipeline** to generate a plan.")
        return

    # Stats
    html(f"<div style=\"font-family:{F};font-size:9px;color:{C['textMuted']};margin:6px 0;line-height:1.6\">"
         f"✅ <b>{result['elapsed']}s</b> · 📊 LLM calls: <b>{result['llm_calls']}</b> · "
         f"retriever: {RETRIEVER_LABELS.get(config['retriever'], config['retriever'])} · "
         f"<i>live backend</i></div>")

    _render_safety_panel(result.get("safety"), side)

    # Per-meal query split (real)
    if result.get("splits"):
        with st.expander("🔍 Query Inspector (per-meal split)", expanded=False):
            for s in result["splits"]:
                html(f"<div style=\"font-family:{F};font-size:9px;color:{C['text']};padding:2px 0\">"
                     f"<b style=\"color:{C['primary']}\">{s['meal']}:</b> {s['q']}</div>")

    # Correction log (real) when validation produced rows
    if config["validation"] and result.get("correction_log"):
        rows = ""
        for i, c in enumerate(result["correction_log"]):
            scol = C["nutA"] if c["status"] == "OK" else C["warn"]
            border = f"border-top:1px solid {C['border']};" if i > 0 else ""
            action = (f"<div style=\"color:{C['sub']};font-size:8px;margin-top:1px\">{c['action']}</div>"
                      if c.get("action") else "")
            rows += (f"<div style=\"font-family:{F};font-size:9px;line-height:1.6;padding:4px 0;{border}\">"
                     f"<b>{c['meal']}</b> · Round {c['round']} · "
                     f"<span style=\"color:{scol};font-weight:700\">{c['status']}</span>"
                     f"<div style=\"color:{C['textMuted']};font-size:8px\">{c['reason']}</div>{action}</div>")
        html(f"<div style=\"background:{C['card']};border:1px solid {C['border']};border-radius:8px;"
             f"padding:10px;margin-bottom:8px\"><div style=\"font-family:{F};font-weight:700;font-size:11px;"
             f"color:{C['text']};margin-bottom:6px\">🔄 Correction Agent Log</div>{rows}</div>")

    recipes = result["recipes"]
    persona = st.session_state["persona"]
    tab_labels = ["🍽 Recipes"] + (["💡 Explanation"] if result.get("explanation") else []) + ["🔗 KG paths"]
    tabs = st.tabs(tab_labels)
    with tabs[0]:
        if not recipes:
            st.warning("The pipeline returned no recipes for this request. Try rephrasing the query.")
        for i, r in enumerate(recipes):
            html(f"<div style=\"font-family:{F};font-size:8px;font-weight:700;color:{C['textMuted']};"
                 f"text-transform:uppercase;letter-spacing:1px;margin:6px 0 3px\">{r['meal']}</div>")
            _render_recipe_safety_badge(r.get("safety"))
            render_recipe_card(r, key=f"real_{side}_{i}", on_expand=_open_detail, compact=arena_mode)
    idx = 1
    if result.get("explanation"):
        with tabs[idx]:
            html(f"<p style=\"font-family:{F};font-size:11px;line-height:1.7;color:{C['text']};margin:0\">"
                 f"{result['explanation']}</p>")
        idx += 1
    with tabs[idx]:
        if recipes:
            choice = st.selectbox("Explain which meal?", range(len(recipes)),
                                  format_func=lambda i: f"{recipes[i]['meal']}: {recipes[i]['name']}",
                                  key=f"kgreal_{side}")
            chosen = recipes[choice]
            render_recipe_paths(chosen, persona, expand=_live_expander())
            # Richer paths: real recipe→ingredient→FoodOn-class chains from Neo4j
            if chosen.get("recipe_id") is not None:
                from ..components.kg_viz import render_kg_ingredient_paths
                from ..services import backend

                st.divider()
                rows = backend.recipe_ingredient_class_paths(chosen["recipe_id"])
                render_kg_ingredient_paths(chosen["name"], rows)
        else:
            st.info("No recipes to ground.")


def render_pipeline_results(side: str, arena_mode: bool) -> None:
    if rag_service.real_enabled():
        _render_real_results(side, arena_mode)
        return
    cfg_key = f"config_{side.lower()}"
    config = st.session_state[cfg_key]

    times_key, done_key, plan_key, ran_key = (
        f"times_{side}", f"done_{side}", f"plan_{side}", f"ran_sig_{side}",
    )
    st.session_state.setdefault(plan_key, "balanced")

    # Pipeline strip (mutates config in place); locked if admin disallows changes
    locked = not pipeline_defaults()["user_can_change_pipeline"]
    config = render_pipeline_strip(config, key=side, times=st.session_state.get(times_key),
                                   disabled=locked)
    st.session_state[cfg_key] = config

    # Invalidate previous run if the config changed
    if st.session_state.get(done_key) and st.session_state.get(ran_key) != _config_sig(config):
        st.session_state[done_key] = False
        st.session_state[times_key] = None

    if not st.session_state.get(done_key):
        if st.button("▶ Run pipeline", key=f"run_{side}", type="primary", use_container_width=True):
            log_event("pipeline_started", {"config": dict(config)})
            times = _simulate_run(config)
            st.session_state[times_key] = times
            st.session_state[done_key] = True
            st.session_state[ran_key] = _config_sig(config)
            log_event("pipeline_completed", {"config": dict(config), "times": times})
            st.rerun()
        return

    times = st.session_state[times_key]
    _stats_line(config, times)

    render_query_inspector(state_key=side)
    if config["validation"]:
        render_correction_log()

    # Plan selector
    plan = st.session_state[plan_key]
    pcols = st.columns(len(PLAN_META))
    for col, (k, meta) in zip(pcols, PLAN_META.items()):
        with col:
            if st.button(f"{meta['icon']} {meta['label']}", key=f"plan_{side}_{k}",
                         type="primary" if plan == k else "secondary", use_container_width=True):
                st.session_state[plan_key] = k
                log_event("plan_switched", {"to_plan": k})
                st.rerun()
    plan = st.session_state[plan_key]
    recipes = RECIPES_SETS[plan]
    meta = PLAN_META[plan]

    # View toggle
    view = st.radio("View", ["🗓 Full Day", "🍽 Single"], key=f"view_{side}",
                    horizontal=True, label_visibility="collapsed")
    single = view == "🍽 Single"

    if single:
        r = recipes[0]
        summary = {"cal": str(r["cal"]), "ns": r["nutriscore"], "prot": f"{r['protein']}g",
                   "fiber": f"{r['fiber']}g", "skill": r["skill"]}
    else:
        summary = meta["summary"]
    html(_summary_panel_html(summary))

    # Tabs
    tab_labels = ["🍽 Recipes"] + (["💡 Explanation"] if config["explanation"] else []) + ["🔗 KG"]
    tabs = st.tabs(tab_labels)

    with tabs[0]:
        if single:
            render_recipe_card(recipes[0], key=f"{side}_single", on_expand=_open_detail, compact=arena_mode)
        else:
            for i, meal in enumerate(MEALS):
                html(f"<div style=\"font-family:{F};font-size:8px;font-weight:700;color:{C['textMuted']};"
                     f"text-transform:uppercase;letter-spacing:1px;margin:6px 0 3px\">{meal}</div>")
                render_recipe_card(recipes[i], key=f"{side}_{i}", on_expand=_open_detail, compact=arena_mode)

    idx = 1
    if config["explanation"]:
        with tabs[idx]:
            html(
                f"<div style=\"font-family:{F};font-weight:700;font-size:12px;color:{C['text']};"
                f"margin-bottom:4px\">Why this meal plan?</div>"
                f"<p style=\"font-family:{F};font-size:11px;line-height:1.7;color:{C['text']};margin:0\">"
                f"This plan prioritises low-glycemic ingredients for stable blood sugar. All recipes are "
                f"Nutri-Score A–B with &lt;10g sugar/meal. The salmon provides omega-3 for insulin "
                f"sensitivity. Recipes match your cooking-skill profile (Bölz et al., 2025). One ingredient "
                f"substitution was applied: honey → stevia drops in the snack to reduce glycemic impact.</p>"
            )
            fb = st.radio("Was this explanation helpful?", ["👍 Helpful", "👎 Not helpful", "🤔 Unclear"],
                          key=f"expl_fb_{side}", index=None, horizontal=True)
            if fb:
                log_event("explanation_feedback", {"vote": fb})
        idx += 1

    with tabs[idx]:
        persona = st.session_state["persona"]
        meal_recipes = [recipes[0]] if single else recipes
        choice = st.selectbox("Explain which meal?", range(len(meal_recipes)),
                              format_func=lambda i: f"{('' if single else MEALS[i] + ': ')}{meal_recipes[i]['name']}",
                              key=f"kg_meal_{side}")
        render_recipe_paths(meal_recipes[choice], persona)

    # Arena vote
    if arena_mode:
        if st.button(f"🏆 Pipeline {side} is better", key=f"vote_{side}", use_container_width=True):
            st.session_state["arena_winner"] = side
            other = "B" if side == "A" else "A"
            log_event("arena_vote", {
                "winner": side,
                "config_A": st.session_state["config_a"],
                "config_B": st.session_state["config_b"],
            })
            st.rerun()


# ── screen entry ───────────────────────────────────────────────────────────
def _cfg_summary(cfg: dict) -> str:
    parts = [RETRIEVER_LABELS[cfg["retriever"]].split(" ")[0]]
    if cfg["validation"]:
        parts.append("✅")
    if cfg["explanation"]:
        parts.append("💡")
    return "+".join(parts)


def render() -> None:
    persona = st.session_state["persona"]
    hexad_type = st.session_state.get("hexad_type") or "—"
    st.session_state.setdefault("arena_mode", False)
    arena_available = pipeline_defaults()["arena_available"]
    if not arena_available:
        st.session_state["arena_mode"] = False

    # Header
    head_l, head_r = st.columns([4, 1])
    with head_l:
        html(
            f"<span class=\"foodrag-brand\" style=\"font-size:18px\">FoodRAG</span>"
            f"<span style=\"font-family:{F};font-size:9px;color:{C['textMuted']};margin-left:6px\">"
            f"{persona['emoji']} {persona['name']} · CS:{persona['cs']} FS:{persona['fs']}</span>"
            f"<span style=\"background:{C['primaryPale']};color:{C['primary']};font-size:8px;padding:1px 6px;"
            f"border-radius:6px;font-family:{F};font-weight:600;margin-left:6px\">HEXAD: {hexad_type}</span>"
        )
    with head_r:
        if arena_available and st.button(
            "⚔ Arena", type="primary" if st.session_state["arena_mode"] else "secondary",
            use_container_width=True,
        ):
            st.session_state["arena_mode"] = not st.session_state["arena_mode"]
            st.session_state["arena_winner"] = None
            st.rerun()

    st.text_input("Query", key="query_text",
                  placeholder="e.g. Suggest a low-sugar dinner for diabetes…",
                  label_visibility="collapsed")

    arena = st.session_state["arena_mode"]
    winner = st.session_state.get("arena_winner")
    if winner:
        st.success(f"🏆 Pipeline {winner} preferred! Vote logged.")

    if not arena:
        render_pipeline_results("A", arena_mode=False)
        return

    # Arena mode
    view = st.radio("Arena view", ["📑 Tabs", "⬜⬜ Side-by-side"], key="arena_view",
                    horizontal=True, label_visibility="collapsed")
    cfg_a, cfg_b = st.session_state["config_a"], st.session_state["config_b"]

    if view == "📑 Tabs":
        ta, tb = st.tabs([f"Pipeline A · {_cfg_summary(cfg_a)}", f"Pipeline B · {_cfg_summary(cfg_b)}"])
        with ta:
            render_pipeline_results("A", arena_mode=True)
        with tb:
            render_pipeline_results("B", arena_mode=True)
    else:
        ca, cb = st.columns(2)
        with ca:
            html(f"<div style=\"text-align:center;font-family:{F};font-weight:700;font-size:11px;"
                 f"color:{C['primary']}\">Pipeline A · {_cfg_summary(cfg_a)}</div>")
            render_pipeline_results("A", arena_mode=True)
        with cb:
            html(f"<div style=\"text-align:center;font-family:{F};font-weight:700;font-size:11px;"
                 f"color:{C['primary']}\">Pipeline B · {_cfg_summary(cfg_b)}</div>")
            render_pipeline_results("B", arena_mode=True)

"""Screen 4: Main interface (spec §4.4).

Pipeline strip, query input, simulated processing, query inspector, correction
log, three alternative plans × four meals, recipe cards with triple scoring,
recipe detail modal, explanation tab, KG path, and arena mode (compare two
pipeline configs). Data is mocked (spec §12 Phase 2 wires the real backend).
"""

from __future__ import annotations

import os
import threading
import time
from datetime import datetime

import streamlit as st

# In-flight live runs, keyed by session+side. The pipeline runs on a background thread
# and appends trace lines here; a polling fragment renders them (Streamlit buffers
# element updates during a blocking run, so the work must NOT block the script thread).
_LIVE_RUNS: dict = {}

from ..admin_config import pipeline_defaults
from ..components.correction_log import render_correction_log
from ..components.kg_viz import render_recipe_paths
from ..components.pipeline_strip import render_pipeline_strip
from ..components.query_inspector import render_query_inspector
from ..components.recipe_card import render_recipe_card
from ..components.recipe_detail import recipe_detail_dialog
from ..components.study_common import render_step_nav
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
    return (f"{cfg['retriever']}|{cfg['validation']}|"
            f"{cfg.get('substitution', True)}|{cfg['explanation']}")


def _full_day() -> bool:
    """Whether the user asked for a full-day plan (vs. a single meal)."""
    return bool(st.session_state.get("full_day", False))


def _current_meal() -> str:
    """Meal slot matching the local time of day — the default single-meal target
    when the query names no explicit meal (spec §4.4)."""
    h = datetime.now().hour
    if 5 <= h < 11:
        return "Breakfast"
    if 11 <= h < 16:
        return "Lunch"
    if 16 <= h < 21:
        return "Dinner"
    return "Snack"


def _query_meal_type(query: str) -> str | None:
    """The explicit meal slot named in the query (lower-case), or None. Word-boundary
    match so 'dinner' hits but 'breakfast-y' style substrings still resolve."""
    import re

    q = (query or "").lower()
    for meal in MEALS:
        if re.search(rf"\b{meal.lower()}\b", q):
            return meal.lower()
    return None


def _target_meal(query: str) -> str:
    """The single meal to recommend: the query's explicit meal, else the current slot."""
    mt = _query_meal_type(query)
    return mt.capitalize() if mt else _current_meal()


# Live-pipeline progress: agent name → (icon, human label) for the status panel.
_AGENT_STEPS = {
    "RecommendationAgent": ("🔍", "Retrieving & recommending recipes"),
    "ValidationAgent": ("✅", "Validating dietary constraints"),
    "SubstitutionAgent": ("🔧", "Substituting to fix violations"),
    "ExplanationAgent": ("💡", "Writing the explanation"),
    "FinalizationAgent": ("📦", "Finalizing the plan"),
}


def _friendly_tool(name: str) -> str:
    return (name or "").replace("_tool", "").replace("_", " ").strip().capitalize()


def _recommended_titles(ev) -> list[str]:
    """Extract 'Meal — Recipe title' strings from a recommendation tool's result, so the
    recommended recipes are visible immediately (before validation runs)."""
    raw = getattr(getattr(ev, "tool_output", None), "raw_output", None)
    try:
        if hasattr(raw, "to_dict"):
            raw = raw.to_dict()
        items = list(raw.items()) if hasattr(raw, "items") else []
    except Exception:  # noqa: BLE001
        return []
    out = []
    for meal, rec in items:
        meal_name = getattr(meal, "value", None) or str(meal)
        title = None
        if hasattr(rec, "get"):
            title = rec.get("title")
        elif hasattr(rec, "to_dict"):
            try:
                title = rec.to_dict().get("title")
            except Exception:  # noqa: BLE001
                title = None
        if title:
            out.append(f"{str(meal_name).capitalize()} — {title}")
    return out


def _output_preview(text: str, limit: int = 260) -> str:
    """One-line, truncated preview of an agent/tool output for the live status panel.
    Leads with a verdict chip when the text announces VALID/INVALID."""
    t = " ".join((text or "").split())
    if len(t) < 8:
        return ""
    up = t.upper()
    chip = "🔴 INVALID · " if "INVALID" in up else ("🟢 VALID · " if "VALID" in up else "")
    body = t if len(t) <= limit else t[:limit].rstrip() + "…"
    return chip + body


def _clean_md(t: str) -> str:
    t = " ".join((t or "").split())
    for ch in ("`", "*", "_"):  # avoid breaking markdown emphasis
        t = t.replace(ch, "")
    return t


def _make_trace_appender(buf: dict):
    """on_event callback that appends a readable trace to ``buf`` (a plain dict shared
    with the polling fragment). Runs on the pipeline's background thread, so it must NOT
    call any ``st.*`` — it only mutates ``buf['lines']`` / ``buf['stream']`` / ``buf['stage']``."""
    seen = {"agent": None}

    def _push(line: str) -> None:
        if line and (not buf["lines"] or buf["lines"][-1] != line):
            buf["lines"].append(line)

    def _on_event(ev) -> None:
        etype = type(ev).__name__
        agent = getattr(ev, "current_agent_name", None)
        if agent and agent != seen["agent"] and agent in _AGENT_STEPS:
            seen["agent"] = agent
            icon, label = _AGENT_STEPS[agent]
            buf["stage"] = f"{icon} {label}"
            _push(f"{icon} **{label}**")
            buf["stream"] = ""
        elif etype == "AgentStream":                      # LLM writing — show it live
            buf["stream"] = getattr(ev, "response", "") or ""
        elif etype == "ToolCall":
            tname = getattr(ev, "tool_name", "") or ""
            if tname and "handoff" not in tname.lower():
                _push(f"&nbsp;&nbsp;↳ {_friendly_tool(tname)}…")
        elif etype == "ToolCallResult":
            tname = (getattr(ev, "tool_name", "") or "").lower()
            titles = _recommended_titles(ev) if ("recommend" in tname or "alternativ" in tname) else []
            if titles:
                _push("🍽 **Recommended:** " + " · ".join(titles))
            else:
                try:
                    out = ev.tool_output
                    preview = _output_preview(str(getattr(out, "content", out) or ""))
                except Exception:  # noqa: BLE001
                    preview = ""
                if preview:
                    _push(f"&nbsp;&nbsp;{preview}")
            buf["stream"] = ""
        elif etype == "AgentOutput":
            try:
                preview = _output_preview(ev.response.content or "")
            except Exception:  # noqa: BLE001
                preview = ""
            if preview:
                _push(f"&nbsp;&nbsp;{preview}")
            buf["stream"] = ""

    return _on_event


def _start_live_run(buf: dict, persona: dict, query: str, run_config: dict) -> None:
    """Run the live pipeline on a daemon thread, streaming a trace into ``buf`` and
    parking the result/error there for the polling fragment to pick up."""
    from ..services import rag_service

    on_event = _make_trace_appender(buf)

    def _bg() -> None:
        try:
            buf["result"] = rag_service.generate_meal_plan(persona, query, run_config, on_event=on_event)
        except Exception as exc:  # noqa: BLE001 — surfaced in the UI via buf["error"]
            buf["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            buf["done"] = True

    threading.Thread(target=_bg, name="foodrag-live-run", daemon=True).start()


def _render_trace(buf: dict) -> None:
    """Render the current live trace (called every second from the polling fragment)."""
    from ..services import backend

    elapsed = int(time.time() - buf.get("t0", time.time()))
    stage = buf.get("stage") or ("📚 Loading HUMMUS dataset & models (first run)…"
                                 if not backend.engine_ready() else "Starting…")
    st.markdown(f"**{stage}**  ·  ⏱ {elapsed}s")
    lines = buf.get("lines", [])[-10:]
    body = "  \n".join(lines)
    if buf.get("stream"):
        body += f"  \n&nbsp;&nbsp;💭 _{_clean_md(buf['stream'][-320:])}…_"
    if body:
        st.markdown(body)


def _poll_live_run(run_key: str, side: str, res_key: str, err_key: str, run_flag: str) -> None:
    """Render the in-flight run's trace, then either promote the finished result or
    sleep ~1s and rerun the whole app to poll again.

    A full-app rerun loop (not a ``run_every`` fragment, which proved unreliable, nor
    ``st.rerun(scope='fragment')``, which is illegal on a fragment's first execution).
    The page renders fully *before* the sleep, so there's no mid-render blocking/garble."""
    buf = _LIVE_RUNS.get(run_key)
    if buf is None:
        st.session_state[run_flag] = False
        st.rerun()
    with st.container(border=True):
        _render_trace(buf)
    if buf.get("done"):
        result = buf.get("result")
        if result is not None:
            st.session_state[res_key] = result
            log_event("pipeline_completed",
                      {"backend": "real", "llm_calls": result.get("llm_calls"),
                       "elapsed": result.get("elapsed"), "partial": result.get("partial", False)})
        if buf.get("error"):
            st.session_state[err_key] = buf["error"]
        st.session_state[run_flag] = False
        _LIVE_RUNS.pop(run_key, None)
        st.rerun()
    else:
        time.sleep(1.0)
        st.rerun()


def _hummus_warm_indicator() -> None:
    """Show a background-load status while the engine warms, and notify once when done.
    Only relevant when the live backend is on.

    Deliberately NOT an auto-refreshing fragment: a ``run_every`` fragment forces
    frontend re-renders that ghost/duplicate the st.columns pipeline strip while a live
    run blocks the script. A static caption is stable; the ready toast fires on the next
    natural rerun (a live run ends with one, so it appears right after)."""
    from ..services import backend

    if not st.session_state.get("backend_live", True):
        return
    if backend.engine_ready():
        if not st.session_state.get("_hummus_notified"):
            st.session_state["_hummus_notified"] = True
            st.toast("✅ HUMMUS dataset loaded — live backend ready.", icon="✅")
        return
    # Self-heal: (re)start the warm-up if it isn't running — e.g. after a module
    # reload reset the engine singleton. Idempotent (no-op while already warming).
    backend.warm_engine_in_background()
    st.caption("⏳ Loading HUMMUS dataset & models in the background — "
               "live runs will be fast once this finishes. Reload the page to refresh status.")


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
    res_key, err_key = f"real_res_{side}", f"real_err_{side}"
    run_flag = f"live_running_{side}"
    run_key = f"live_{st.session_state.get('session_id', '')}_{side}"
    running = bool(st.session_state.get(run_flag))

    locked = not pipeline_defaults()["user_can_change_pipeline"]
    config = render_pipeline_strip(config, key=side, disabled=locked or running)
    st.session_state[cfg_key] = config

    if config["retriever"] == "fusion":
        st.caption("ℹ Fusion runs reciprocal-rank fusion over dense + sparse "
                   "(+ Neo4j KG when available, per protocol §5.5).")

    # A run is in flight for this side → show the live trace and poll (background thread
    # does the work so the script never blocks and updates flush live).
    if running:
        st.markdown("**▶ Running the live pipeline…**")
        _poll_live_run(run_key, side, res_key, err_key, run_flag)
        return

    if st.button("▶ Run live pipeline", key=f"runreal_{side}", type="primary",
                 use_container_width=True):
        from ..services.stores_bootstrap import ensure_stores_up

        persona = st.session_state["persona"]
        query = st.session_state.get("query_text", "")
        single = bool(st.session_state.get("single_meal_mode", False))
        # Three modes: Single → one meal (overrides Full day); Full day → all 4 meals;
        # neither → the meals remaining in the current day (time-aware). Only Single sets
        # a target meal (the query's explicit meal, else the current-time slot).
        full_day = _full_day() and not single
        run_config = {**config, "full_day": full_day,
                      "meal_type": _target_meal(query).lower() if single else None}
        log_event("pipeline_started", {"config": run_config, "backend": "real"})
        st.session_state.pop(err_key, None)
        st.session_state.pop(res_key, None)
        ensure_stores_up()  # self-heal (fast no-op when Postgres is reachable)
        buf = {"lines": [], "stream": "", "stage": "", "done": False,
               "result": None, "error": None, "t0": time.time()}
        _LIVE_RUNS[run_key] = buf
        st.session_state[run_flag] = True
        _start_live_run(buf, persona, query, run_config)
        st.rerun()

    if st.session_state.get(err_key):
        err = st.session_state[err_key]
        st.error(f"Live pipeline failed: {err}")
        # The vector-store retrievers (Dense/Sparse/Hybrid/Fusion) need pgvector on :5432.
        if any(tok in err for tok in ("5432", "OperationalError", "Connection refused",
                                      "connection to server")):
            st.warning(
                "The vector database (pgvector, port 5432) isn't running, so this retriever "
                "can't fetch recipes. Either **switch the retriever to Text→Pandas** above "
                "(runs with no database), or start the backing stores with `docker compose up -d`.")
        else:
            st.caption("Check the live-backend status in the admin panel, or switch the "
                       "Live backend toggle off to use mock data.")
        return

    result = st.session_state.get(res_key)
    if not result:
        st.info("Configure the pipeline and click **Run live pipeline** to generate a plan.")
        return

    if result.get("partial"):
        st.warning(
            "⏱ The pipeline hit its time limit before validation/explanation finished — "
            "showing the **recommendation as-is** (not fully validated). This usually happens on "
            "hard constraints that trigger many correction rounds. To get a full run faster, try a "
            "simpler query, or turn off **Validation** in the pipeline strip for a recommend-only run.")

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

    # Mock preview: the Single toggle shows one meal (query's explicit meal, else the
    # current slot); otherwise the full-day view (the live backend distinguishes
    # remaining-day vs full-day, but the mock set only carries the 4 canonical meals).
    single = bool(st.session_state.get("single_meal_mode", False))
    query = st.session_state.get("query_text", "")
    cur_idx = MEALS.index(_target_meal(query)) if single else 0

    if single:
        r = recipes[cur_idx]
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
            html(f"<div style=\"font-family:{F};font-size:8px;font-weight:700;color:{C['textMuted']};"
                 f"text-transform:uppercase;letter-spacing:1px;margin:6px 0 3px\">{MEALS[cur_idx]}</div>")
            render_recipe_card(recipes[cur_idx], key=f"{side}_single", on_expand=_open_detail, compact=arena_mode)
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
        meal_recipes = [recipes[cur_idx]] if single else recipes
        meal_names = [MEALS[cur_idx]] if single else MEALS
        choice = st.selectbox("Explain which meal?", range(len(meal_recipes)),
                              format_func=lambda i: f"{meal_names[i]}: {meal_recipes[i]['name']}",
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
    hexad_type = st.session_state.get("hexad_type")  # None while HEXAD is hidden
    st.session_state.setdefault("arena_mode", False)
    arena_available = pipeline_defaults()["arena_available"]
    if not arena_available:
        st.session_state["arena_mode"] = False

    # Navigation: Flow A is consent → persona → main, so Back returns to the
    # persona chooser and Home to the flow chooser (spec §4.4). Shared helper uses
    # on_click callbacks, robust against the live-backend warm-poll fragment.
    render_step_nav(back_screen="persona", home_screen="welcome")

    # Header
    head_l, head_r = st.columns([4, 1])
    with head_l:
        hexad_badge = (
            f"<span style=\"background:{C['primaryPale']};color:{C['primary']};font-size:8px;padding:1px 6px;"
            f"border-radius:6px;font-family:{F};font-weight:600;margin-left:6px\">HEXAD: {hexad_type}</span>"
            if hexad_type else ""
        )
        html(
            f"<span class=\"foodrag-brand\" style=\"font-size:18px\">FoodRAG</span>"
            f"<span style=\"font-family:{F};font-size:9px;color:{C['textMuted']};margin-left:6px\">"
            f"{persona['emoji']} {persona['name']} · CS:{persona['cs']} FS:{persona['fs']}</span>"
            f"{hexad_badge}"
        )
    with head_r:
        if arena_available and st.button(
            "⚔ Arena", type="primary" if st.session_state["arena_mode"] else "secondary",
            use_container_width=True,
        ):
            st.session_state["arena_mode"] = not st.session_state["arena_mode"]
            st.session_state["arena_winner"] = None
            st.rerun()

    st.text_area("Query", key="query_text",
                 placeholder="e.g. Suggest a low-sugar dinner for diabetes…",
                 label_visibility="collapsed", height=80)

    st.session_state.setdefault("full_day", False)
    st.session_state.setdefault("single_meal_mode", False)
    st.session_state.setdefault("backend_live", True)
    _q = st.session_state.get("query_text", "")
    _explicit = _query_meal_type(_q)
    _single_target = (_explicit.capitalize() if _explicit else _current_meal())
    _reason = "named in your query" if _explicit else "based on the time of day"
    _single_on = st.session_state.get("single_meal_mode", False)
    tog_l, tog_m, tog_r = st.columns(3)
    with tog_l:
        # Single overrides Full day (a single meal can't also be a full day).
        st.toggle(
            "🗓 Full day", key="full_day", disabled=_single_on,
            help=("Off: the meals remaining in the current day (time-aware). "
                  "On: a full-day plan — breakfast, lunch, dinner and a snack."),
        )
    with tog_m:
        st.toggle(
            "🍽 Single", key="single_meal_mode",
            help=(f"On: just one meal — {_single_target} ({_reason}). Overrides Full day. "
                  "Fastest, and the most likely to finish a fully-validated plan."),
        )
    with tog_r:
        forced = os.environ.get("FOODRAG_REAL_BACKEND", "").lower() in ("1", "true", "yes", "on")
        st.toggle(
            "🟢 Live backend", key="backend_live", disabled=forced,
            help=("On: run the real agentic pipeline over HUMMUS (slower, hits the LLM). "
                  "Off: fast mock preview." + (" Forced on by env." if forced else "")),
        )

    _hummus_warm_indicator()

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

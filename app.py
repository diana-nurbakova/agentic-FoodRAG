"""FoodRAG demo — Streamlit entry point (spec §11.5).

Routing + session bootstrap. Flow A (Interactive Exploration) and Flow B
(Structured Evaluation) are implemented and share the consent screen; a welcome
screen selects the flow. The Admin view (``?admin=true``) is a later phase.
Run with: ``streamlit run app.py``.
"""

from __future__ import annotations

import streamlit as st

# Load .env before anything reads os.environ (ADMIN_PASSWORD, FOODRAG_REAL_BACKEND,
# store creds). Real deployments set these as real env vars; load_env does not override.
try:
    from foodrag.config.settings import load_env

    load_env()
except Exception:  # noqa: BLE001 — foodrag not importable: env vars must come from the environment
    pass

from ui.screens import consent, hexad, main, persona, report, welcome
from ui.screens.admin import view as admin_view
from ui.screens.evaluation import (
    completion,
    demographics,
    plan_evaluation,
    post_study,
    profile_assignment,
    ranking,
)
from ui.services.stores_bootstrap import ensure_stores_up, stores_reachable
from ui.state import init_state
from ui.theme import inject_global_css

st.set_page_config(page_title="FoodRAG Demo", page_icon="🍽️", layout="centered")


@st.cache_resource(show_spinner="Starting backing stores (Docker: Postgres + Neo4j)…")
def _bootstrap_stores() -> dict:
    """Bring the pgvector Postgres (and Neo4j) up once per server process (spec §2.2)."""
    return ensure_stores_up()


@st.cache_resource(show_spinner=False)
def _warm_backend() -> bool:
    """Start loading HUMMUS + models in the background so the first live run is fast.
    Skipped in tests / when store bootstrap is disabled."""
    import os as _os

    if _os.environ.get("FOODRAG_SKIP_STORES_BOOTSTRAP") or "PYTEST_CURRENT_TEST" in _os.environ:
        return False
    from ui.services import backend

    backend.warm_engine_in_background()
    return True


inject_global_css()
_bootstrap_stores()  # best-effort one-time start (cached per server)
# Warn based on LIVE reachability, not the cached start attempt — and only once per
# outage, so a transient/at-launch failure doesn't nag on every rerun.
if stores_reachable():
    st.session_state.pop("_stores_warned", None)
elif not st.session_state.get("_stores_warned"):
    st.session_state["_stores_warned"] = True
    st.toast("⚠ Live backend: Postgres (pgvector, :5432) isn't reachable. Start Docker "
             "Desktop then `docker compose up -d`, or use the Text→Pandas retriever.", icon="⚠️")
_warm_backend()
init_state()

# Admin view (spec §6.1): ?admin=true reveals the password-gated admin panel.
if st.query_params.get("admin") == "true":
    admin_view.render()
    st.stop()

# Report tab (Flow C, Evaluation Protocol §11): ?report=true opens it directly.
if st.query_params.get("report") == "true":
    report.render()
    st.stop()

_ROUTES = {
    "welcome": welcome.render,
    "report": report.render,
    "consent": consent.render,
    # Flow A
    "hexad": hexad.render,
    "persona": persona.render,
    "main": main.render,
    # Flow B
    "demographics": demographics.render,
    "profile_assignment": profile_assignment.render,
    "plan_evaluation": plan_evaluation.render,
    "post_study": post_study.render,
    "ranking": ranking.render,
    "completion": completion.render,
}

screen = st.session_state.get("screen", "welcome")
_ROUTES.get(screen, welcome.render)()

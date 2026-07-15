"""Admin-editable study configuration (spec §6.3).

The canonical config lives in ``st.session_state['admin_config']`` and mirrors
the JSON the admin would store in the HF Dataset repo (so it can be edited
without redeployment). Accessors here let the flows read effective settings;
``default_admin_config()`` seeds the defaults.

Live propagation wired so far: ``active_flow`` (welcome), ``pipeline_defaults``
(Flow A main), and active/ordered personas (Flow A persona screen). Other
settings are edited, persisted in session, and exportable; deeper propagation
(e.g. ``plans_per_profile`` into Flow B) lands when the backing data supports it.
"""

from __future__ import annotations

import copy

import streamlit as st

from .mock_data import PERSONAS
from .study_data import DEFAULT_STUDY_CONDITIONS

# Questionnaire block catalogue (spec §6.3 table) — selectable when adding blocks
BLOCK_CATALOG = {
    "consent": "Consent (fixed text)",
    "demographics": "Demographics (GDPR fields)",
    "hexad12": "HEXAD-12 (Tondello 2016)",
    "cooking_skills": "Cooking skills (14-technique, Bölz 2025)",
    "rebs": "REBS (24 items, Pelletier 2004)",
    "plan_evaluation": "Plan evaluation (ResQue + attention check)",
    "post_study": "Post-study (SCS + ResQue D + FoodRAG)",
    "resque15": "ResQue-15 (full short form)",
    "scs": "System Causability Scale (Holzinger 2020)",
    "custom": "Custom (admin-defined items)",
}

_DEFAULT_BLOCKS = [
    {"type": "consent", "required": True, "position": "pre", "active": True},
    {"type": "demographics", "required": True, "position": "pre", "active": True,
     "fields": {"age_range": True, "country": True, "employment": True,
                "cooking_frequency": True, "cooking_skill": True}},
    {"type": "hexad12", "required": False, "position": "pre", "active": True},
    {"type": "cooking_skills", "required": False, "position": "pre", "active": False},
    {"type": "plan_evaluation", "required": True, "position": "during", "active": True,
     "items_source": "resque_adapted", "attention_check_per_block": True},
    {"type": "post_study", "required": True, "position": "post", "active": True,
     "constructs": ["explanation_quality", "constraint_compliance",
                    "behavioural_intention", "usability"]},
]


def default_admin_config() -> dict:
    """Fresh default config (spec §6.3 JSON shape)."""
    return {
        "study_id": "recsys2026-review",
        "active_flow": "both",                       # both | exploration | evaluation
        "profile_assignment": "random_round_robin",
        "plans_per_profile": 3,
        "counterbalancing": "latin_square",
        "attention_checks": True,
        "likert_scale": 5,
        "blocks": copy.deepcopy(_DEFAULT_BLOCKS),
        "pipeline_defaults": {
            "default_retriever": "hybrid",
            "validation_default": True,
            "substitution_default": True,
            "explanation_default": True,
            "arena_available": True,
            "user_can_change_pipeline": True,
            # Live backend (spec §2.2): off by default — real runs hit the LLM
            # provider (cost + latency). Env FOODRAG_REAL_BACKEND also forces it on.
            "use_real_backend": False,
        },
        # Persona management (spec §6.2): display order + active flag per persona
        "personas": [{"id": p["id"], "name": p["name"], "active": True} for p in PERSONAS],
        "imported_personas": [],
        # Flow A recommender: which 3 evaluation profiles (profiles.json ids) seed the
        # persona cards; the 4th card is always the user-defined custom builder.
        "flow_a_profiles": ["P1", "P3", "P4"],
        # Flow B study design (admin-editable): the within-subjects CONDITIONS (each
        # mapped to an already-run eval result file) and the PROFILES (each an eval
        # question). Empty ``study_profiles`` falls through to the generated
        # profiles.json default (see ``study_data.get_study_profiles``).
        "study_conditions": copy.deepcopy(DEFAULT_STUDY_CONDITIONS),
        "study_profiles": [],
    }


def get_admin_config() -> dict:
    return st.session_state["admin_config"]


def pipeline_defaults() -> dict:
    return get_admin_config()["pipeline_defaults"]


def study_conditions() -> list[dict]:
    """The admin-configured Flow B conditions (falls back to the default axis)."""
    return get_admin_config().get("study_conditions") or copy.deepcopy(DEFAULT_STUDY_CONDITIONS)


def study_profiles_cfg() -> list[dict]:
    """The admin-configured Flow B profiles (may be empty → generated default is used)."""
    return get_admin_config().get("study_profiles") or []


def flow_a_profile_ids() -> list[str]:
    """The evaluation-profile ids seeding the Flow A recommender persona cards."""
    try:
        return get_admin_config().get("flow_a_profiles") or ["P1", "P3", "P4"]
    except (KeyError, AttributeError):  # session not yet initialised
        return ["P1", "P3", "P4"]


def _persona_index() -> dict:
    idx = {p["id"]: p for p in PERSONAS}
    for p in get_admin_config().get("imported_personas", []):
        idx[p["id"]] = p
    return idx


def active_personas() -> list[dict]:
    """Predefined + imported personas that are active, in admin display order."""
    idx = _persona_index()
    out = [idx[e["id"]] for e in get_admin_config()["personas"] if e["active"] and e["id"] in idx]
    return out or PERSONAS  # never leave the selection empty

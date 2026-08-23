"""Persona management.

Browse personas (in the demo, the predefined set standing in for a live HUMMUS
query), toggle them active/inactive, reorder them, and import custom edge-case
personas from JSON. The active, ordered list feeds the Flow A persona screen
(``ui.admin_config.active_personas``).
"""

from __future__ import annotations

import json

import streamlit as st

from ...admin_config import _persona_index, get_admin_config
from ...study_data import load_all_profiles
from ...theme import C, F, html


def _move(personas: list[dict], pos: int, delta: int) -> None:
    target = pos + delta
    if 0 <= target < len(personas):
        personas[pos], personas[target] = personas[target], personas[pos]
        st.rerun()


def _flow_a_profile_picker(cfg: dict) -> None:
    """Choose which evaluation profiles seed the Flow A recommender persona cards."""
    st.subheader("Flow A recommender profiles")
    st.caption("Pick the real evaluation profiles (from profiles.json) shown as persona cards "
               "in the interactive recommender. A 4th user-defined (custom) card is always added.")
    all_profiles = load_all_profiles()
    if not all_profiles:
        st.info("No profiles.json found (run scripts/build_study_plans.py --select).")
        return
    labels = {p["id"]: f"{p['id']} — {p.get('desc', p.get('name', ''))[:80]}" for p in all_profiles}
    current = [pid for pid in (cfg.get("flow_a_profiles") or []) if pid in labels]
    chosen = st.multiselect(
        "Profiles shown in Flow A", options=list(labels), default=current,
        format_func=lambda pid: labels[pid], key="flow_a_profiles_pick",
    )
    if chosen != current:
        cfg["flow_a_profiles"] = chosen
    if len(chosen) != 3:
        st.caption(f"⚠ {len(chosen)} selected — 3 recommended (the 4th slot is the custom builder).")
    st.divider()


def render() -> None:
    cfg = get_admin_config()
    personas = cfg["personas"]
    idx = _persona_index()

    _flow_a_profile_picker(cfg)

    st.subheader("Active study personas")
    st.caption("Browse profiles (demo: predefined set ≈ live HUMMUS query), toggle and reorder. "
               "The active, ordered list is shown to participants in Flow A.")

    for pos, entry in enumerate(personas):
        p = idx.get(entry["id"], {})
        cond = p.get("condition", "—")
        allergies = ", ".join(p.get("allergies", [])) or "none"
        c_info, c_act, c_up, c_down = st.columns([6, 1.4, 0.7, 0.7])
        with c_info:
            html(
                f"<div style=\"font-family:{F};font-size:13px;font-weight:600;color:{C['text']}\">"
                f"{p.get('emoji', '👤')} {entry['name']}</div>"
                f"<div style=\"font-family:{F};font-size:11px;color:{C['textMuted']}\">"
                f"{cond} · allergies: {allergies} · {p.get('nutrientConstraint', 'None')}</div>"
            )
        with c_act:
            entry["active"] = st.toggle("Active", value=entry["active"], key=f"pa_{entry['id']}")
        with c_up:
            if st.button("▲", key=f"pup_{entry['id']}", disabled=pos == 0, use_container_width=True):
                _move(personas, pos, -1)
        with c_down:
            if st.button("▼", key=f"pdn_{entry['id']}", disabled=pos == len(personas) - 1,
                         use_container_width=True):
                _move(personas, pos, 1)

    active_n = sum(1 for e in personas if e["active"])
    st.caption(f"{active_n} of {len(personas)} personas active.")

    st.divider()
    st.subheader("Import custom personas (JSON)")
    st.caption("Upload edge-case profiles not in the dataset. Expected: a JSON list of persona objects "
               "with at least `id` and `name`.")
    up = st.file_uploader("Persona JSON", type=["json"], key="persona_json")
    if up is not None and st.button("Import", type="primary", key="persona_import"):
        try:
            data = json.load(up)
            items = data if isinstance(data, list) else [data]
            added = 0
            for item in items:
                if "id" not in item or "name" not in item:
                    continue
                cfg["imported_personas"].append(item)
                cfg["personas"].append({"id": item["id"], "name": item["name"], "active": True})
                added += 1
            st.success(f"Imported {added} persona(s).")
            st.rerun()
        except (json.JSONDecodeError, ValueError) as e:
            st.error(f"Could not parse JSON: {e}")

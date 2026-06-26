"""Questionnaire configuration (spec §6.3).

Study-level settings + an ordered list of questionnaire blocks (active,
required, position). Mirrors the JSON stored in the HF Dataset repo. ``active_flow``
takes effect live (welcome screen); the rest are persisted and exported, with
deeper propagation following as the backing data supports it.
"""

from __future__ import annotations

import streamlit as st

from ...admin_config import BLOCK_CATALOG, default_admin_config, get_admin_config
from ...services import config_service
from ...theme import C, F, html

_FLOW_OPTS = ["both", "exploration", "evaluation"]
_ASSIGN_OPTS = ["random_round_robin", "participant_choice"]
_CB_OPTS = ["latin_square", "random_shuffle"]
_POSITIONS = ["pre", "during", "post"]


def _move(blocks: list[dict], pos: int, delta: int) -> None:
    target = pos + delta
    if 0 <= target < len(blocks):
        blocks[pos], blocks[target] = blocks[target], blocks[pos]
        st.rerun()


def _persistence_controls(cfg: dict) -> None:
    st.subheader("Configuration persistence")
    status = config_service.status()
    dest = (f"HF repo `{status['repo_id']}` + local" if status["remote_enabled"]
            else "local file only (set `FOODRAG_DATASET_REPO` + `HF_TOKEN` for the dataset repo)")
    st.caption(f"Persists to {dest}. Changes apply to new sessions without redeployment (spec §6.3).")
    c1, c2, c3 = st.columns([1, 1, 2])
    if c1.button("💾 Save configuration", type="primary"):
        ok, msg = config_service.save_config(cfg)
        (st.success if ok else st.error)(msg)
    if c2.button("↻ Reload from store"):
        persisted = config_service.load_persisted()
        if persisted is None:
            st.info("No persisted config found; keeping current settings.")
        else:
            st.session_state["admin_config"] = config_service.deep_merge(
                default_admin_config(), persisted)
            st.success("Reloaded persisted configuration.")
            st.rerun()
    c3.caption(f"Local: `{status['local_path']}` "
               f"({'exists' if status['local_exists'] else 'not saved yet'})")
    st.divider()


def render() -> None:
    cfg = get_admin_config()

    _persistence_controls(cfg)

    st.subheader("Study settings")
    c1, c2, c3 = st.columns(3)
    with c1:
        cfg["active_flow"] = st.selectbox("Active flow", _FLOW_OPTS,
                                          index=_FLOW_OPTS.index(cfg["active_flow"]),
                                          help="Takes effect immediately on the welcome screen.")
        cfg["profile_assignment"] = st.selectbox("Profile assignment", _ASSIGN_OPTS,
                                                 index=_ASSIGN_OPTS.index(cfg["profile_assignment"]))
    with c2:
        cfg["plans_per_profile"] = st.number_input("Plans per profile", min_value=2, max_value=4,
                                                   value=cfg["plans_per_profile"], step=1,
                                                   help="Demo data currently provides 3 plans per profile.")
        cfg["counterbalancing"] = st.selectbox("Counterbalancing", _CB_OPTS,
                                               index=_CB_OPTS.index(cfg["counterbalancing"]))
    with c3:
        cfg["likert_scale"] = st.selectbox("Likert scale", [5, 7],
                                           index=[5, 7].index(cfg["likert_scale"]))
        cfg["attention_checks"] = st.toggle("Attention checks", value=cfg["attention_checks"])

    st.divider()
    st.subheader("Questionnaire blocks")
    st.caption("Order, activate and place each block (pre- / during- / post-task).")
    blocks = cfg["blocks"]
    for pos, b in enumerate(blocks):
        with st.container(border=True):
            html(f"<div style=\"font-family:{F};font-size:13px;font-weight:700;color:{C['text']}\">"
                 f"{BLOCK_CATALOG.get(b['type'], b['type'])}</div>"
                 f"<div style=\"font-family:{F};font-size:10px;color:{C['textMuted']};font-family:monospace\">{b['type']}</div>")
            cc1, cc2, cc3, cc4, cc5 = st.columns([1.3, 1.3, 2, 0.6, 0.6])
            b["active"] = cc1.toggle("Active", value=b["active"], key=f"b_act_{pos}")
            b["required"] = cc2.toggle("Required", value=b["required"], key=f"b_req_{pos}")
            b["position"] = cc3.selectbox("Position", _POSITIONS, index=_POSITIONS.index(b["position"]),
                                          key=f"b_pos_{pos}", label_visibility="collapsed")
            if cc4.button("▲", key=f"b_up_{pos}", disabled=pos == 0, use_container_width=True):
                _move(blocks, pos, -1)
            if cc5.button("▼", key=f"b_dn_{pos}", disabled=pos == len(blocks) - 1, use_container_width=True):
                _move(blocks, pos, 1)
            if not b.get("required") and cc1.button("Remove", key=f"b_rm_{pos}"):
                blocks.pop(pos)
                st.rerun()

    st.divider()
    add_l, add_r = st.columns([4, 1])
    with add_l:
        new_type = st.selectbox("Add a block", list(BLOCK_CATALOG.keys()),
                                format_func=lambda k: BLOCK_CATALOG[k], key="add_block_type")
    with add_r:
        st.write("")
        if st.button("Add", type="primary", use_container_width=True):
            blocks.append({"type": new_type, "required": False, "position": "pre", "active": True})
            st.rerun()

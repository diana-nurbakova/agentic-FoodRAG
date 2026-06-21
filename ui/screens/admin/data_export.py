"""Data export (spec §6.6).

Download the in-memory interaction buffer as JSONL, view auto-push status,
force a flush, link to the dataset repo, and export the current study config
JSON (spec §6.3).
"""

from __future__ import annotations

import json

import streamlit as st

from ...admin_config import get_admin_config
from ...services import logging_service


def _to_jsonl(events: list[dict]) -> str:
    return "\n".join(json.dumps(e, ensure_ascii=False) for e in events)


def render() -> None:
    events = st.session_state.get("event_log", [])
    session_id = st.session_state["session_id"]

    st.subheader("Interaction logs")
    st.caption(f"{len(events)} events in the current in-memory buffer (spec §9.1).")
    st.download_button(
        "⬇ Download buffer as JSONL",
        data=_to_jsonl(events) or "",
        file_name=f"foodrag_{session_id}.jsonl",
        mime="application/x-ndjson",
        disabled=not events,
    )

    st.divider()
    st.subheader("Auto-push status")
    status = logging_service.status()
    st.caption("Logs are pushed to the private HF Dataset repo every "
               f"{status['interval_s'] // 60} minutes and on session end (spec §9.2).")

    if status["remote_enabled"]:
        st.success(f"Remote logging ON → `{status['repo_id']}`"
                   + (" · Supabase secondary ON" if status["supabase_enabled"] else ""))
    else:
        st.warning("Remote logging OFF — events are in-memory only. "
                   "Set `FOODRAG_DATASET_REPO` (+ `HF_TOKEN`) or enter a repo below.")

    repo = st.text_input("HF Dataset repo id (e.g. your-username/foodrag-evaluation-data)",
                         value=status["repo_id"] or "", key="dataset_repo_input")
    if st.button("Save repo"):
        logging_service.set_dataset_repo(repo)
        st.rerun()

    last = logging_service.last_push(session_id)
    c1, c2, c3 = st.columns(3)
    if last:
        c1.metric("Last push", last["timestamp"], help=("OK" if last["ok"] else last["error"]))
        c3.metric("Last push", "✅ OK" if last["ok"] else "❌ failed")
    else:
        c1.metric("Last push", "—")
        c3.metric("Last push", "never")
    c2.metric("Pending events", len(events))

    if st.button("⤴ Flush all logs now", type="primary"):
        ok, msg = logging_service.flush_now(session_id)
        (st.success if ok else st.error)(f"Flush: {msg}")

    if status["repo_id"]:
        st.link_button("Open HF Dataset repo",
                       f"https://huggingface.co/datasets/{status['repo_id']}")

    st.divider()
    st.subheader("Study configuration (JSON)")
    st.caption("This is what would be persisted to the HF Dataset repo so changes apply without redeployment (spec §6.3).")
    cfg_json = json.dumps(get_admin_config(), indent=2, ensure_ascii=False)
    st.code(cfg_json, language="json")
    st.download_button("⬇ Download study config", data=cfg_json,
                       file_name="study_config.json", mime="application/json")

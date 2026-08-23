"""Interaction logging service.

Architecture:

    Streamlit app (in-memory buffer)
       ├── every 5 min ──────→ push JSONL to HF Dataset repo (primary)
       ├── on session end ───→ push session log to HF Dataset repo
       ├── on admin request ─→ manual download / force flush
       └── on each event ────→ insert to external DB over HTTPS (secondary)

Every event is enriched to the record schema and appended to the session's
in-memory buffer (``st.session_state['event_log']``). If a HF Dataset repo is
configured (``FOODRAG_DATASET_REPO`` + ``HF_TOKEN``), buffers are pushed to
``logs/<session_id>.jsonl`` by a background thread and on demand. Everything
degrades to in-memory-only when no repo/token is set — never raising into the UI.
"""

from __future__ import annotations

import datetime
import json
import os
import tempfile
import threading
from typing import Any, Optional

import streamlit as st

FLUSH_INTERVAL_SECONDS = 300  # every 5 minutes

# Process-wide registry: session_id -> the live event_log list (shared by ref so
# the background thread sees appends made on the main thread).
_registry: dict[str, list] = {}
_registry_lock = threading.Lock()
_last_push: dict[str, dict] = {}
_thread_started = False


# ── configuration ─────────────────────────────────────────────────────────
def _repo_id() -> Optional[str]:
    return os.environ.get("FOODRAG_DATASET_REPO") or None


def _hf_token() -> Optional[str]:
    return os.environ.get("HF_TOKEN") or None


def remote_enabled() -> bool:
    """True when both a dataset repo and an HF token are configured."""
    return bool(_repo_id() and _hf_token())


def set_dataset_repo(repo_id: str) -> None:
    """Set the target dataset repo at runtime (admin). Visible to the thread."""
    if repo_id:
        os.environ["FOODRAG_DATASET_REPO"] = repo_id.strip()
    else:
        os.environ.pop("FOODRAG_DATASET_REPO", None)


# ── record schema ─────────────────────────────────────────────────────────
def _now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def enrich(event_type: str, payload: dict | None) -> dict:
    """Build a full log record from the current session state."""
    ss = st.session_state
    persona = ss.get("persona") or {}
    return {
        "session_id": ss.get("session_id"),
        "timestamp": _now_iso(),
        "event_type": event_type,
        "is_admin": bool(ss.get("is_admin")),
        "flow": ss.get("flow") or "exploration",
        "persona_id": persona.get("id"),
        "hexad_type": ss.get("hexad_type"),
        "hexad_scores": ss.get("hexad_scores"),
        "pipeline_config": ss.get("config_a"),
        "payload": payload or {},
    }


# ── public API ────────────────────────────────────────────────────────────
def log_event(event_type: str, payload: dict | None = None) -> None:
    """Enrich, buffer in-memory, mirror to the secondary DB, and ensure the
    periodic flush is running. Never raises into the UI."""
    ss = st.session_state
    record = enrich(event_type, payload)
    buffer = ss.setdefault("event_log", [])
    buffer.append(record)

    sid = record["session_id"]
    if sid:
        with _registry_lock:
            _registry[sid] = buffer  # keep a reference to the live list
    _supabase_insert(record)
    _ensure_thread()


# ── primary sink: HF Dataset repo ─────────────────────────────────────────
def flush_session(session_id: str, events: list[dict]) -> tuple[bool, str]:
    """Push one session's JSONL to ``logs/<session_id>.jsonl``. Returns (ok, msg)."""
    if not remote_enabled():
        return False, "remote logging disabled (set FOODRAG_DATASET_REPO + HF_TOKEN)"
    if not events:
        return False, "nothing to flush"
    try:
        from huggingface_hub import HfApi

        api = HfApi(token=_hf_token())
        repo_id = _repo_id()
        try:
            api.create_repo(repo_id=repo_id, repo_type="dataset", private=True, exist_ok=True)
        except Exception:  # noqa: BLE001 — repo may already exist / perms; upload will report
            pass

        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False,
                                             encoding="utf-8") as fh:
                for event in events:
                    fh.write(json.dumps(event, ensure_ascii=False) + "\n")
                tmp_path = fh.name
            api.upload_file(
                path_or_fileobj=tmp_path,
                path_in_repo=f"logs/{session_id}.jsonl",
                repo_id=repo_id,
                repo_type="dataset",
            )
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

        _last_push[session_id] = {"timestamp": _now_iso(), "count": len(events), "ok": True, "error": None}
        return True, "ok"
    except Exception as exc:  # noqa: BLE001
        _last_push[session_id] = {"timestamp": _now_iso(), "count": len(events), "ok": False,
                                  "error": f"{type(exc).__name__}: {exc}"}
        return False, str(exc)


def flush_now(session_id: Optional[str] = None) -> tuple[bool, str]:
    """Force a flush of one session (or all registered sessions)."""
    if session_id is not None:
        with _registry_lock:
            events = list(_registry.get(session_id, []))
        return flush_session(session_id, events)
    return flush_all()


def flush_all() -> tuple[bool, str]:
    with _registry_lock:
        items = [(sid, list(buf)) for sid, buf in _registry.items()]
    if not items:
        return False, "no active sessions"
    ok_all = True
    for sid, events in items:
        ok, _ = flush_session(sid, events)
        ok_all = ok_all and ok
    return ok_all, f"flushed {len(items)} session(s)"


def _periodic_flush() -> None:
    import time

    while True:
        time.sleep(FLUSH_INTERVAL_SECONDS)
        try:
            flush_all()
        except Exception:  # noqa: BLE001 — background thread must never die
            pass


def _ensure_thread() -> None:
    global _thread_started
    if _thread_started or not remote_enabled():
        return
    threading.Thread(target=_periodic_flush, daemon=True, name="foodrag-log-flush").start()
    _thread_started = True


# ── status (admin data-export panel) ──────────────────────────────────────
def last_push(session_id: str) -> Optional[dict]:
    return _last_push.get(session_id)


def status() -> dict:
    return {
        "remote_enabled": remote_enabled(),
        "repo_id": _repo_id(),
        "interval_s": FLUSH_INTERVAL_SECONDS,
        "supabase_enabled": _supabase_enabled(),
    }


# ── secondary sink: external DB over HTTPS (optional) ──────────────────────
def _supabase_enabled() -> bool:
    return bool(os.environ.get("SUPABASE_URL") and os.environ.get("SUPABASE_KEY"))


def _supabase_insert(record: dict) -> None:
    """Best-effort per-event insert to Supabase. No-op if unconfigured/uninstalled."""
    if not _supabase_enabled():
        return
    try:
        from supabase import create_client

        client = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])
        client.table("interaction_logs").insert(record).execute()
    except Exception:  # noqa: BLE001 — secondary sink must never break the app
        pass

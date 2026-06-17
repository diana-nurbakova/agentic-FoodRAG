"""Structured JSONL logging of every LLM/embedding call.

Each call appends one JSON line with timestamp, model, request id, prompt,
response, latency, token usage, cost, attempt and error. Logs go to
``data/logs/llm_calls-<date>.jsonl`` by default (override via env
``FOODRAG_LLM_LOG`` or :func:`configure`).
"""
from __future__ import annotations

import json
import os
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

_LOCK = threading.Lock()
_LOG_PATH: Optional[Path] = None


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _default_log_path() -> Path:
    env = os.environ.get("FOODRAG_LLM_LOG")
    if env:
        return Path(env)
    day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return _repo_root() / "data" / "logs" / f"llm_calls-{day}.jsonl"


def configure(path: Optional[Path] = None) -> Path:
    """Set the JSONL log path; returns the resolved path."""
    global _LOG_PATH
    _LOG_PATH = Path(path) if path else _default_log_path()
    return _LOG_PATH


def log_path() -> Path:
    return _LOG_PATH or configure()


def _truncate(text: Any, limit: int = 8000) -> Any:
    if isinstance(text, str) and len(text) > limit:
        return text[:limit] + f"...<truncated {len(text) - limit} chars>"
    return text


@dataclass
class CallRecord:
    kind: str  # "complete" | "chat" | "embedding"
    model: Optional[str] = None
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    prompt: Any = None
    response: Any = None
    latency_s: Optional[float] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    cost_usd: Optional[float] = None
    attempt: Optional[int] = None
    error: Optional[str] = None
    extra: dict[str, Any] = field(default_factory=dict)
    ts: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


def write(record: CallRecord) -> None:
    path = log_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(record)
    payload["prompt"] = _truncate(payload.get("prompt"))
    payload["response"] = _truncate(payload.get("response"))
    line = json.dumps(payload, ensure_ascii=False, default=str)
    with _LOCK:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")


def extract_usage(response: Any) -> dict[str, Any]:
    """Pull token usage / cost out of a LiteLLM raw response if present."""
    out: dict[str, Any] = {}
    raw = getattr(response, "raw", None) or {}
    usage = raw.get("usage") if isinstance(raw, dict) else getattr(raw, "usage", None)
    if usage is not None:
        get = usage.get if isinstance(usage, dict) else lambda k, d=None: getattr(usage, k, d)
        out["prompt_tokens"] = get("prompt_tokens")
        out["completion_tokens"] = get("completion_tokens")
        out["total_tokens"] = get("total_tokens")
    if isinstance(raw, dict):
        hidden = raw.get("_hidden_params") or {}
        if "response_cost" in hidden:
            out["cost_usd"] = hidden["response_cost"]
    return out


@contextmanager
def record_call(kind: str, model: Optional[str] = None, prompt: Any = None, **extra):
    """Time a call and emit a :class:`CallRecord` (success or error)."""
    rec = CallRecord(kind=kind, model=model, prompt=prompt, extra=extra)
    start = time.perf_counter()
    box: dict[str, Any] = {"record": rec}
    try:
        yield box
    except Exception as exc:  # noqa: BLE001 - we log then re-raise
        rec.latency_s = round(time.perf_counter() - start, 4)
        rec.error = f"{type(exc).__name__}: {exc}"
        write(rec)
        raise
    else:
        rec.latency_s = round(time.perf_counter() - start, 4)
        resp = box.get("response")
        if resp is not None:
            rec.response = str(resp)
            usage = extract_usage(resp)
            rec.prompt_tokens = usage.get("prompt_tokens")
            rec.completion_tokens = usage.get("completion_tokens")
            rec.total_tokens = usage.get("total_tokens")
            rec.cost_usd = usage.get("cost_usd")
        write(rec)

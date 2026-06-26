"""Robust LLM call + structured-output parsing shared by the KBQA systems.

The KBQA prompts ask the model for small JSON payloads (scored relations,
scored entities, a Cypher string). Two concerns are centralised here:

* **Tolerant parsing** — models wrap JSON in prose or code fences; we reuse the
  project's :func:`hummus_util.try_parse` (the same recovery the LLM-only rung
  uses) and degrade to a safe default rather than raising mid-traversal.
* **No-LLM fallback** — when ``llm`` is ``None`` (deterministic tests, or a run
  configured without a generator) every call returns ``""`` so callers fall back
  to uniform scoring and the traversal still proceeds and stays testable.
"""
from __future__ import annotations

from typing import Any, Optional

from foodrag.utils.hummus_util import try_parse


def complete(llm: Optional[Any], prompt: str) -> str:
    """Call ``llm.complete(prompt).text``; return ``""`` on no-LLM or error."""
    if llm is None:
        return ""
    try:
        return (llm.complete(prompt).text or "").strip()
    except Exception:
        return ""


def parse_json(text: str, default: Any) -> Any:
    """Best-effort parse of a JSON value from ``text``; ``default`` on failure."""
    if not text:
        return default
    try:
        value = try_parse(text)
    except Exception:
        return default
    return value if value is not None else default


def parse_scores(text: str, n: int) -> dict[int, float]:
    """Parse ``[{"id": i, "score": s}, ...]`` into ``{0-based-index: score}``.

    Tolerates 1-based ids and out-of-range ids (dropped). Empty/garbled output
    yields ``{}`` so the caller can fall back to uniform scores.
    """
    value = parse_json(text, [])
    out: dict[int, float] = {}
    if isinstance(value, list):
        for item in value:
            if not isinstance(item, dict) or "id" not in item:
                continue
            try:
                idx, score = int(item["id"]), float(item.get("score", 0.0))
            except (TypeError, ValueError):
                continue
            if idx >= n and (idx - 1) < n:  # tolerate 1-based ids
                idx -= 1
            if 0 <= idx < n:
                out[idx] = score
    return out

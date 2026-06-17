"""Robust recovery of JSON from messy LLM output.

LLMs wrap JSON in prose, ```json fences, single quotes, Python literals
(``True``/``None``), and trailing commas. :func:`recover_json` tries a ladder
of increasingly forgiving strategies and (optionally) validates the result
against a Pydantic model. Generalises ``hummus_util.try_parse``.
"""
from __future__ import annotations

import ast
import json
import re
from typing import Any, Optional, Type

_SENTINEL = object()


class JSONRecoveryError(ValueError):
    """Raised when no strategy could recover JSON from the text."""


_FENCE_RE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.DOTALL)


def _candidate_strings(text: str) -> list[str]:
    """Ordered candidate substrings that might contain the JSON payload."""
    text = (text or "").strip()
    candidates: list[str] = []

    def add(s: Optional[str]) -> None:
        if s:
            s = s.strip()
            if s and s not in candidates:
                candidates.append(s)

    add(text)
    for m in _FENCE_RE.findall(text):
        add(m)
    # Largest balanced object / array slice. Prefer the one whose opening
    # bracket appears first (the outermost container), so an object nested
    # inside an array doesn't shadow the array.
    slices = []
    for open_ch, close_ch in (("{", "}"), ("[", "]")):
        i, j = text.find(open_ch), text.rfind(close_ch)
        if 0 <= i < j:
            slices.append((i, text[i : j + 1]))
    for _, sl in sorted(slices, key=lambda t: t[0]):
        add(sl)
    return candidates


def _repair(s: str) -> str:
    """Light, conservative repairs for near-JSON strings."""
    # Python literals -> JSON literals (word-boundary only).
    s = re.sub(r"\bTrue\b", "true", s)
    s = re.sub(r"\bFalse\b", "false", s)
    s = re.sub(r"\b(None|NaN)\b", "null", s)
    # Remove trailing commas before } or ].
    s = re.sub(r",(\s*[}\]])", r"\1", s)
    return s


def _try_load(s: str) -> Any:
    """Try json -> ast.literal_eval -> repaired json. Returns _SENTINEL on failure."""
    try:
        return json.loads(s)
    except (json.JSONDecodeError, ValueError):
        pass
    try:
        return ast.literal_eval(s)
    except (ValueError, SyntaxError):
        pass
    try:
        return json.loads(_repair(s))
    except (json.JSONDecodeError, ValueError):
        return _SENTINEL


def recover_json(
    text: str,
    schema: Optional[Type] = None,
    default: Any = _SENTINEL,
) -> Any:
    """Recover a JSON value from ``text``.

    :param schema: optional Pydantic model class; if given, the recovered value
        is validated/coerced into it (``model_validate``).
    :param default: value to return when recovery fails; if omitted, raises
        :class:`JSONRecoveryError`.
    """
    parsed = _SENTINEL
    for cand in _candidate_strings(text):
        result = _try_load(cand)
        if result is not _SENTINEL:
            parsed = result
            break

    if parsed is _SENTINEL:
        if default is not _SENTINEL:
            return default
        snippet = (text or "")[:200].replace("\n", "\\n")
        raise JSONRecoveryError(f"Could not recover JSON from: {snippet!r}")

    if schema is not None:
        try:
            return schema.model_validate(parsed)  # pydantic v2
        except AttributeError:  # pragma: no cover - pydantic v1 fallback
            return schema.parse_obj(parsed)

    return parsed


def try_parse(string_object: str) -> Any:
    """Backwards-compatible shim for the old hummus_util.try_parse."""
    return recover_json(string_object)

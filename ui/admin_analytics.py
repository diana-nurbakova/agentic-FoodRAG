"""Visit analytics.

Combines mock historical aggregates (so the dashboard looks populated in the
demo) with live counts derived from the current session's in-memory event log.
Admin sessions are excluded from participant analytics via the ``is_admin`` flag
; here the live overlay simply reflects the current event buffer.

Replace ``_MOCK_*`` with reads from the analytics service / HF Dataset repo.
"""

from __future__ import annotations

from collections import Counter

# ── Mock historical aggregates (demo only) ────────────────────────────────
_MOCK_FUNNEL = [
    ("Sessions started", 412, 100),
    ("Consent given", 388, 94),
    ("HEXAD / Demographics completed", 351, 85),
    ("Persona selected / assigned", 333, 81),
    ("First pipeline run / plan viewed", 318, 77),
    ("Session completed", 274, 66),
]

_MOCK_RETRIEVER_USAGE = {"Hybrid": 141, "Dense": 88, "BM25": 52, "Fusion": 41, "Text→Pandas": 23}
_MOCK_SESSIONS_PER_DAY = {
    "Jun 14": 38, "Jun 15": 54, "Jun 16": 61, "Jun 17": 49,
    "Jun 18": 72, "Jun 19": 66, "Jun 20": 72,
}
_MOCK_ENGAGEMENT = {
    "Avg session duration": "8m 42s",
    "Queries / session": "3.1",
    "Arena A vs B win rate": "A 54% · B 46%",
    "Recipe 👍/👎 ratio": "78% 👍",
}
_MOCK_QUALITY = {
    "Suspicious sessions (<60s)": 11,
    "Incomplete sessions": 138,
    "Attention: 1 failure": 24,
    "Attention: 2+ failures": 9,
    "Straightlining (>80% identical)": 7,
    "Response speed flag (<2s/item)": 5,
}


def live_counts(event_log: list[dict]) -> Counter:
    """Event-type counts from the current session buffer."""
    return Counter(e["event_type"] for e in event_log)


def get_analytics(event_log: list[dict]) -> dict:
    counts = live_counts(event_log)
    return {
        "funnel": _MOCK_FUNNEL,
        "retriever_usage": _MOCK_RETRIEVER_USAGE,
        "sessions_per_day": _MOCK_SESSIONS_PER_DAY,
        "engagement": _MOCK_ENGAGEMENT,
        "quality": _MOCK_QUALITY,
        "live_counts": counts,
        "live_total_events": sum(counts.values()),
    }

"""Unit tests for the degenerate-meal-plan protection (error-analysis follow-up).

The t2p full-day run produced degenerate plans (the SAME recipe in multiple slots):
foodrag_h2_t2p_fullday was 100% "same recipe in all 4 slots", h1_t2p_fullday 17.7%.
Two defences are covered here:

GUARD — ``constraints.enforce_distinct_slots`` (applies to ALL configs)
    Deterministically drops any LATER slot whose recipe shares a ``recipe_url`` with
    an earlier slot, keeping the recipe in its FIRST (stable-order) slot. This
    guarantees no plan has a duplicate recipe across slots regardless of retriever.

RETRIEVER FIX — ``NonRepeatingReplayRetriever`` (driver, full-day t2p only)
    Hands out a forward slice of the ranked pool per ``retrieve`` call and advances a
    per-instance cursor, so each per-slot ``.head(1)`` gets a DISTINCT rank-ordered
    recipe and slots beyond the pool size get an empty frame (unfilled) not a dup.

Memory-light: no hummus load, no LLM.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

from foodrag.baselines import constraints

pytestmark = pytest.mark.unit


# ── helpers ──────────────────────────────────────────────────────────────────

def _recipe(url):
    return {"recipe_url": url, "title": url}


# ── GUARD: enforce_distinct_slots semantics ──────────────────────────────────

def test_same_recipe_in_four_slots_keeps_only_first():
    dup = _recipe("same")
    plan = {"breakfast": dict(dup), "lunch": dict(dup),
            "dinner": dict(dup), "snack": dict(dup)}
    out = constraints.enforce_distinct_slots(plan)
    assert list(out.keys()) == ["breakfast"]      # only the FIRST slot kept
    assert out["breakfast"]["recipe_url"] == "same"


def test_four_distinct_recipes_unchanged():
    plan = {"breakfast": _recipe("b"), "lunch": _recipe("l"),
            "dinner": _recipe("d"), "snack": _recipe("s")}
    out = constraints.enforce_distinct_slots(plan)
    assert out == plan
    assert out is not plan                         # a NEW dict is returned


def test_none_and_empty_slots_preserved():
    plan = {"breakfast": None, "lunch": {}, "dinner": _recipe("d")}
    out = constraints.enforce_distinct_slots(plan)
    assert out["breakfast"] is None
    assert out["lunch"] == {}
    assert out["dinner"]["recipe_url"] == "d"


def test_input_not_mutated():
    dup = _recipe("same")
    plan = {"breakfast": dict(dup), "dinner": dict(dup)}
    original_keys = set(plan.keys())
    constraints.enforce_distinct_slots(plan)
    assert set(plan.keys()) == original_keys       # original untouched


def test_ordering_keeps_breakfast_over_dinner():
    """A duplicate across breakfast + dinner keeps BREAKFAST (stable STANDARD_SLOTS
    order: breakfast, lunch, dinner, snack)."""
    dup = _recipe("same")
    # Insert dinner FIRST to prove order is by STANDARD_SLOTS, not insertion order.
    plan = {"dinner": dict(dup), "breakfast": dict(dup)}
    out = constraints.enforce_distinct_slots(plan)
    assert "breakfast" in out
    assert "dinner" not in out


def test_recipe_without_url_always_kept():
    # Recipes lacking a recipe_url cannot be identified as duplicates → all kept.
    plan = {"breakfast": {"title": "x"}, "lunch": {"title": "y"}}
    out = constraints.enforce_distinct_slots(plan)
    assert set(out.keys()) == {"breakfast", "lunch"}


# ── RETRIEVER FIX: NonRepeatingReplayRetriever ───────────────────────────────

def _load_driver_module():
    """Load the out-of-repo driver (c:/tmp/run_foodrag_pipeline.py) by file path.

    Module-level code only does dotenv/chdir/path setup (no hummus/LLM), so importing
    is cheap; skip gracefully if the driver is not present on this machine."""
    driver_path = Path(r"c:/tmp/run_foodrag_pipeline.py")
    if not driver_path.exists():
        pytest.skip("driver run_foodrag_pipeline.py not present")
    spec = importlib.util.spec_from_file_location("run_foodrag_pipeline_test", driver_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _head1_url(frame):
    head = frame.head(1)
    return None if head.empty else head.iloc[0]["recipe_url"]


def test_non_repeating_replay_hands_out_distinct_then_empty():
    mod = _load_driver_module()
    recipes_df = pd.DataFrame([_recipe("u0"), _recipe("u1"), _recipe("u2")])
    r = mod.NonRepeatingReplayRetriever(recipes_df, ["u0", "u1"])
    # 4 per-slot retrieve() calls: u0, u1, then exhausted -> empty, empty.
    assert _head1_url(r.retrieve("q")) == "u0"
    assert _head1_url(r.retrieve("q")) == "u1"
    assert _head1_url(r.retrieve("q")) is None
    assert _head1_url(r.retrieve("q")) is None

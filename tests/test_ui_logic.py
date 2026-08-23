"""GUI logic unit tests: HEXAD scoring, counterbalancing, attention placement,
and Flow B study-data integrity. No Streamlit runtime required.
"""
import pytest

pytestmark = [pytest.mark.unit, pytest.mark.gui]


# ── HEXAD-12 scoring ──────────────────────────────────────────────────────
def test_hexad_score_dominant_and_sums():
    from ui.screens.hexad import HEXAD_TYPES, _score

    # item i scores type i % 6; max all "Player" items (indices 0, 6)
    answers = [7 if i % 6 == 0 else 1 for i in range(12)]
    scores, dominant = _score(answers)
    assert dominant == "Player"
    assert scores["Player"] == 14            # 7 + 7
    assert set(scores) == set(HEXAD_TYPES)
    assert all(v == 2 for t, v in scores.items() if t != "Player")  # 1 + 1


# ── Counterbalancing ──────────────────────────────────────────────────────
def test_latin_square_3_balanced_each_position():
    from ui.counterbalancing import get_plan_order

    seen = {0: set(), 1: set(), 2: set()}
    for pn in range(3):
        order = get_plan_order(pn, 3)
        assert sorted(order) == [0, 1, 2]
        for pos, plan in enumerate(order):
            seen[pos].add(plan)
    assert all(s == {0, 1, 2} for s in seen.values())


def test_latin_square_4_williams_balanced():
    from ui.counterbalancing import get_plan_order

    seen = {p: set() for p in range(4)}
    for pn in range(4):
        order = get_plan_order(pn, 4)
        assert sorted(order) == [0, 1, 2, 3]
        for pos, plan in enumerate(order):
            seen[pos].add(plan)
    assert all(s == {0, 1, 2, 3} for s in seen.values())


def test_round_robin_profile_assignment():
    from ui.counterbalancing import assign_profile_index

    assigned = [assign_profile_index(pn, 5) for pn in range(10)]
    assert assigned == [0, 1, 2, 3, 4, 0, 1, 2, 3, 4]


# ── Attention-check placement ─────────────────────────────────────────────
def test_attention_check_positions_rotate():
    from ui.study_data import ATTENTION_ITEM, ATTENTION_POSITIONS, plan_eval_items_for

    for plan_index in range(3):
        items = plan_eval_items_for(plan_index)
        assert len(items) == 8  # 7 ResQue + 1 attention
        pos = [i for i, it in enumerate(items) if it["id"] == ATTENTION_ITEM["id"]][0] + 1
        assert pos == ATTENTION_POSITIONS[plan_index]


# ── Flow B study-data integrity ───────────────────────────────────────────
def test_study_data_integrity():
    from ui.study_data import (
        PLAN_EVAL_ITEMS,
        POST_STUDY_ITEMS,
        STUDY_CONFIG,
        STUDY_PLANS,
        STUDY_PROFILES,
    )

    assert len(PLAN_EVAL_ITEMS) == 7 and len(POST_STUDY_ITEMS) == 7
    ids = [p["id"] for p in STUDY_PROFILES]
    assert "baseline" in ids and len(STUDY_PROFILES) == 5
    n = STUDY_CONFIG["plans_per_profile"]
    for pid in ids:
        plans = STUDY_PLANS[pid]
        assert len(plans) >= n
        for plan in plans[:n]:
            assert len(plan["meals"]) == 4  # breakfast/lunch/dinner/snack
            assert plan["explanation"]["summary"]

"""GUI flow tests (Streamlit AppTest): every screen renders without exception.

Screens are reached by jumping via ``session_state`` rather than full click-through
— AppTest mishandles unselected (``index=None``) radios across navigations, which
the real browser does not. These tests assert render-correctness, not the harness's
widget bookkeeping.
"""
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

pytestmark = [pytest.mark.unit, pytest.mark.gui]

APP = str(Path(__file__).resolve().parents[1] / "app.py")

PERSONA = {"id": 1, "name": "Marie", "emoji": "👩‍⚕️", "condition": "Type 2 Diabetes",
           "allergies": ["Gluten"], "likedTags": ["Mediterranean"], "dislikedTags": [],
           "likedIng": [], "dislikedIng": [], "nutrientConstraint": "Sugar ≤ 10g/meal",
           "cs": "High", "fs": "Medium"}


@pytest.fixture(autouse=True)
def isolate(tmp_path, monkeypatch):
    monkeypatch.setenv("FOODRAG_STUDY_CONFIG", str(tmp_path / "study_config.json"))
    monkeypatch.delenv("FOODRAG_DATASET_REPO", raising=False)
    monkeypatch.delenv("FOODRAG_REAL_BACKEND", raising=False)


def _render(setup=None, query=None):
    at = AppTest.from_file(APP, default_timeout=30)
    if query:
        for k, v in query.items():
            at.query_params[k] = v
    at.run()
    if setup:
        setup(at)
        at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def test_welcome_offers_flows_and_report():
    at = _render()
    labels = " ".join(b.label for b in at.button)
    assert "exploring" in labels and "study" in labels and "Report" in labels


@pytest.mark.parametrize("screen", ["consent", "hexad", "persona"])
def test_flow_a_onboarding_screens(screen):
    _render(lambda at: at.session_state.__setitem__("screen", screen))


def test_flow_a_main_prerun_mock():
    def setup(at):
        at.session_state["persona"] = PERSONA
        at.session_state["hexad_type"] = "Achiever"
        at.session_state["backend_live"] = False   # mock preview path
        at.session_state["screen"] = "main"
    at = _render(setup)
    assert any("Run pipeline" in b.label for b in at.button)


def test_flow_a_main_prerun_live_default():
    # The recommender defaults to the LIVE backend switch → the live run button renders
    # (the engine is only built on click, so rendering stays cheap).
    def setup(at):
        at.session_state["persona"] = PERSONA
        at.session_state["screen"] = "main"
    at = _render(setup)
    assert at.session_state["backend_live"] is True
    assert any("Run live pipeline" in b.label for b in at.button)


def test_flow_b_demographics_and_assignment():
    for screen in ("demographics", "profile_assignment"):
        _render(lambda at, s=screen: (at.session_state.__setitem__("flow", "evaluation"),
                                       at.session_state.__setitem__("screen", s)))


def _first_profile_id() -> str:
    from ui.study_data import get_study_profiles

    return get_study_profiles()[0]["id"]


def test_flow_b_plan_evaluation():
    pid = _first_profile_id()

    def setup(at):
        ss = at.session_state
        ss["flow"] = "evaluation"; ss["screen"] = "plan_evaluation"
        ss["assigned_profile_id"] = pid; ss["plan_order"] = [0, 1, 2]
        ss["latin_square_group"] = 0
        ss["plan_index"] = 0; ss["plan_answers"] = []; ss["attention_results"] = []
    at = _render(setup)
    assert len(at.radio) == 8  # 7 ResQue + 1 attention


def test_flow_b_post_study_ranking_completion():
    from ui.study_data import PLAN_EVAL_ITEMS, POST_STUDY_ITEMS

    _render(lambda at: (at.session_state.__setitem__("flow", "evaluation"),
                        at.session_state.__setitem__("screen", "post_study")))

    pid = _first_profile_id()

    def rank(at):
        ss = at.session_state
        ss["flow"] = "evaluation"; ss["screen"] = "ranking"
        ss["assigned_profile_id"] = pid; ss["plan_order"] = [1, 2, 0]
    _render(rank)

    def done(at):
        ss = at.session_state
        ss["flow"] = "evaluation"; ss["screen"] = "completion"
        ss["assigned_profile_id"] = pid; ss["plan_order"] = [1, 2, 0]; ss["latin_square_group"] = 1
        ss["demographics"] = {"age_range": "25–34", "country": "France", "employment": "Student",
                              "cooking_frequency": "Daily", "cooking_skill": "Intermediate"}
        ss["plan_answers"] = [{"plan_position_seen": i, "plan_cond": f"C{i+1}",
                               "likert_responses": {it["id"]: 4 for it in PLAN_EVAL_ITEMS},
                               "attention_check_passed": i != 0} for i in range(3)]
        ss["attention_results"] = [False, True, True]
        ss["post_study_answers"] = {it["id"]: 4 for it in POST_STUDY_ITEMS}
        ss["ranking_result"] = {"ranking_positions": [2, 0, 1], "ranking_conds": ["C3", "C1", "C2"],
                                "preferred_position": 2, "preferred_cond": "C3",
                                "feedback_text": "best fit", "other_text": ""}
    _render(done)


def test_admin_auth_and_tabs(monkeypatch):
    monkeypatch.setenv("ADMIN_PASSWORD", "secret123")
    at = AppTest.from_file(APP, default_timeout=30)
    at.query_params["admin"] = "true"
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    at.text_input[0].set_value("secret123")
    [b for b in at.button if b.label == "Sign in"][0].click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.session_state["admin_authenticated"] is True
    assert len(at.tabs) >= 5


def test_report_tab_renders():
    at = _render(query={"report": "true"})
    assert len(at.tabs) == 6

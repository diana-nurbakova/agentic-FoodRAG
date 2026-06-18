"""Unit tests for the resumable meal-type annotation runner (no network)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from foodrag.pipeline import annotate

pytestmark = pytest.mark.unit


class _FakeResp:
    def __init__(self, text):
        self.text = text


class _FakeLLM:
    def __init__(self, text='```json\n["breakfast"]\n```'):
        self.text = text
        self.calls = 0

    def complete(self, prompt):
        self.calls += 1
        return _FakeResp(self.text)


def test_annotate_one_parses_json_block():
    assert annotate.annotate_one({"title": "x"}, _FakeLLM('```json\n["breakfast", "snack"]\n```')) == [
        "breakfast", "snack"
    ]


def test_annotate_one_handles_garbage():
    assert annotate.annotate_one({"title": "x"}, _FakeLLM("not json at all")) == []


def _hummus(ids):
    return {"recipes_df": pd.DataFrame({"recipe_id": ids, "title": [f"r{i}" for i in ids]})}


def test_annotate_checkpoints_and_resumes(tmp_path: Path):
    (tmp_path / "hummus").mkdir()
    hummus = _hummus([1, 2, 3])

    llm1 = _FakeLLM()
    out1 = annotate.run(data_path=str(tmp_path), hummus_data=hummus, llm=llm1,
                        workers=1, save=False, verbose=False)
    assert out1["annotated"] == 3
    assert llm1.calls == 3
    assert annotate.store_path_for(str(tmp_path)).exists()

    # resume: everything already done -> no new LLM calls
    llm2 = _FakeLLM()
    out2 = annotate.run(data_path=str(tmp_path), hummus_data=hummus, llm=llm2,
                        workers=1, save=False, verbose=False)
    assert llm2.calls == 0
    assert out2["annotated"] == 3


def test_annotate_errors_are_recorded_not_fatal(tmp_path: Path):
    (tmp_path / "hummus").mkdir()

    class BoomLLM:
        calls = 0

        def complete(self, prompt):
            BoomLLM.calls += 1
            raise RuntimeError("provider down")

    out = annotate.run(data_path=str(tmp_path), hummus_data=_hummus([1, 2]),
                       llm=BoomLLM(), workers=1, save=False, verbose=False)
    # errors recorded, run completes; nothing counted as ok
    assert out["annotated"] == 0
    from foodrag.llm.runstore import RunStore
    store = RunStore(annotate.store_path_for(str(tmp_path)))
    assert len(store.errors()) == 2


def test_annotate_saves_meal_types_into_recipes_df(tmp_path: Path):
    (tmp_path / "hummus").mkdir()
    hummus = _hummus([1, 2])
    annotate.run(data_path=str(tmp_path), hummus_data=hummus, llm=_FakeLLM(),
                 workers=1, save=True, verbose=False)
    # recipes_df.csv written with the meal_types column populated
    saved = pd.read_csv(tmp_path / "hummus" / "recipes_df.csv")
    assert "meal_types" in saved.columns
    assert "breakfast" in str(saved["meal_types"].iloc[0])

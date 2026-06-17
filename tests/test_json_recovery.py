"""Unit tests for robust JSON recovery."""
from __future__ import annotations

import pytest

from foodrag.llm.json_recovery import JSONRecoveryError, recover_json

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "text,expected",
    [
        ('{"a": 1}', {"a": 1}),
        ('  {"a": 1}  ', {"a": 1}),
        ("```json\n{\"a\": 1}\n```", {"a": 1}),
        ("```\n{\"a\": 1}\n```", {"a": 1}),
        ("Here is the plan:\n{\"a\": 1}\nThanks!", {"a": 1}),
        ("{'a': 1, 'b': [1, 2]}", {"a": 1, "b": [1, 2]}),  # single quotes
        ('{"a": true, "b": None, "c": False}', {"a": True, "b": None, "c": False}),
        ('{"a": 1, "b": 2,}', {"a": 1, "b": 2}),  # trailing comma
        ("[1, 2, 3]", [1, 2, 3]),
        ('prefix [ {"x": 1} ] suffix', [{"x": 1}]),
    ],
)
def test_recover_variants(text, expected):
    assert recover_json(text) == expected


def test_default_on_failure():
    assert recover_json("not json at all", default=None) is None
    assert recover_json("", default={}) == {}


def test_raises_without_default():
    with pytest.raises(JSONRecoveryError):
        recover_json("totally not json {oops")


def test_schema_validation():
    pydantic = pytest.importorskip("pydantic")

    class Plan(pydantic.BaseModel):
        recipe: str
        servings: int

    out = recover_json('```json\n{"recipe": "soup", "servings": "4"}\n```', schema=Plan)
    assert out.recipe == "soup"
    assert out.servings == 4  # coerced str -> int

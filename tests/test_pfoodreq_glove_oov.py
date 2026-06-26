"""Unit tests for the GloVe OOV analysis (faithful-reproduction honesty check)."""
from __future__ import annotations

import pandas as pd
import pytest

from foodrag.baselines.pfoodreq.glove_oov import (compute_oov, extract_food_vocab,
                                                  load_glove_vocab, tokenize)

pytestmark = pytest.mark.unit


def test_tokenize_splits_hyphen_slash_and_lowercases():
    assert tokenize("Lamb-Sheep") == ["lamb", "sheep"]
    assert tokenize("low-fat/skim milk") == ["low", "fat", "skim", "milk"]


def test_extract_vocab_respects_min_freq_and_sources():
    df = pd.DataFrame([
        {"ingredient_names": ["spinach", "rice"], "tags": ["vegan"], "title": "Spinach Bowl"},
        {"ingredient_names": ["spinach"], "tags": ["vegan"], "title": "Plain Rice"},
    ])
    counters = extract_food_vocab(df, min_freq=2)
    # spinach: 2x ingredient + 1x title = appears; rice 1 ingredient + 1 title = 2 overall
    assert counters["ingredient_names"]["spinach"] == 2
    assert "rice" not in counters["ingredient_names"]  # freq 1 in that source, dropped
    assert counters["tags"]["vegan"] == 2


def test_compute_oov_type_and_token_rates():
    counters = {"ingredient_names": __import__("collections").Counter(
        {"spinach": 8, "rice": 2, "galangal": 1})}
    glove = {"spinach", "rice"}  # 'galangal' is OOV
    rep = compute_oov(counters, glove)
    assert rep["n_types"] == 3 and rep["n_tokens"] == 11
    assert rep["type_oov_rate"] == pytest.approx(1 / 3)
    assert rep["token_oov_rate"] == pytest.approx(1 / 11)
    assert rep["type_hit_ratio"] == pytest.approx(2 / 3)
    assert rep["top_oov"][0] == ("galangal", 1)


def test_load_glove_vocab(tmp_path):
    p = tmp_path / "glove.txt"
    p.write_text("Spinach 0.1 0.2\nrice 0.3 0.4\n", encoding="utf-8")
    vocab = load_glove_vocab(p)
    assert vocab == {"spinach", "rice"}  # lowercased

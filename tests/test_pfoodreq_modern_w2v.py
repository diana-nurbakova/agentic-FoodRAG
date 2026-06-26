"""Unit tests for the modern-embedding ablation writer (§5.8).

Checks the emitted file is GloVe-format (the exact shape pFoodReq's
build_pretrained_w2v.py parses) and that reserved tokens are skipped — using a
fake deterministic embedder, no model download.
"""
from __future__ import annotations

import pytest

from foodrag.baselines.pfoodreq.build_modern_w2v import write_glove_format
from foodrag.baselines.pfoodreq.validate import load_recipe_kg  # unrelated; not used

pytestmark = pytest.mark.unit


def _fake_embedder(dim=4):
    # deterministic: vector = [len(word), index-in-batch, 0.5, -0.5]
    def _embed(batch):
        return [[float(len(w)), float(i), 0.5, -0.5] for i, w in enumerate(batch)]
    return _embed


def test_writes_glove_format_and_dim(tmp_path):
    out = tmp_path / "modern_emb.txt"
    stats = write_glove_format(["spinach", "rice", "galangal"], _fake_embedder(), out,
                               skip_tokens={"PAD", "UNK"})
    assert stats["n_words"] == 3 and stats["dim"] == 4

    lines = out.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3
    first = lines[0].split()
    assert first[0] == "spinach"
    assert len(first) == 1 + 4  # word + 4 dims
    # values are parseable floats (what load_embeddings does)
    assert [float(x) for x in first[1:]] == [7.0, 0.0, 0.5, -0.5]


def test_skips_reserved_tokens(tmp_path):
    out = tmp_path / "emb.txt"
    stats = write_glove_format(["PAD", "UNK", "rice"], _fake_embedder(), out,
                               skip_tokens={"PAD", "UNK"})
    assert stats["n_words"] == 1
    assert out.read_text(encoding="utf-8").split()[0] == "rice"

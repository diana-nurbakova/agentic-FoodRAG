"""Unit tests for the modern-embedding ablation writer.

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


def test_resume_only_embeds_remaining(tmp_path):
    out = tmp_path / "emb.txt"
    calls = []

    def counting_embedder(batch):
        calls.extend(batch)
        return [[1.0, 2.0, 3.0, 4.0] for _ in batch]

    # first run: 2 of 4 words (simulate an interrupted session)
    write_glove_format(["a", "b"], counting_embedder, out)
    assert calls == ["a", "b"]

    # resume with the full vocab: only the 2 new words are embedded
    calls.clear()
    stats = write_glove_format(["a", "b", "c", "d"], counting_embedder, out, resume=True)
    assert calls == ["c", "d"]
    assert stats == {**stats, "n_new": 2, "n_resumed": 2, "n_words": 4}
    words = [ln.split()[0] for ln in out.read_text(encoding="utf-8").splitlines()]
    assert words == ["a", "b", "c", "d"]


def test_resume_repairs_truncated_tail(tmp_path):
    out = tmp_path / "emb.txt"
    # a complete line + a truncated final line (interrupted mid-write, no newline)
    out.write_text("rice 1.0 2.0 3.0 4.0\nspinach 0.5 0.6", encoding="utf-8")
    stats = write_glove_format(["rice", "spinach"], _fake_embedder(), out, resume=True)
    # 'spinach' was truncated -> dropped and re-embedded; 'rice' kept
    assert stats["n_resumed"] == 1 and stats["n_new"] == 1
    lines = out.read_text(encoding="utf-8").splitlines()
    assert [ln.split()[0] for ln in lines] == ["rice", "spinach"]
    assert all(len(ln.split()) == 1 + 4 for ln in lines)  # no corrupt vector survived


def test_resume_false_overwrites(tmp_path):
    out = tmp_path / "emb.txt"
    write_glove_format(["a", "b"], _fake_embedder(), out)
    stats = write_glove_format(["c"], _fake_embedder(), out, resume=False)
    assert stats["n_words"] == 1
    assert [ln.split()[0] for ln in out.read_text(encoding="utf-8").splitlines()] == ["c"]

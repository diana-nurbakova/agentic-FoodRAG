"""Wiring tests for the ported pFoodReq/BAMnet pipeline (§5.8 sub-task a).

Verifies the vendored tree is present and the driver's config plumbing works,
without running the (heavy, CPU) training — that is exercised manually.
"""
from __future__ import annotations

import pytest

from foodrag.baselines.pfoodreq.build_modern_w2v import write_glove_format

pytestmark = pytest.mark.unit


def test_vendored_bamnet_present():
    from foodrag.baselines.pfoodreq.train_bamnet import _bamnet_src
    src = _bamnet_src()
    assert (src / "build_all_data.py").exists()
    assert (src / "core" / "bamnet" / "bamnet.py").exists()
    assert (src / "config" / "pfoodreq.yml").exists()


def test_base_config_loads_required_keys():
    from foodrag.baselines.pfoodreq.train_bamnet import _base_config
    opt = _base_config()
    for key in ("query_size", "ans_path_bow_size", "test_margin", "batch_size",
                "learning_rate", "mem_size", "num_hops"):
        assert key in opt, key
    assert isinstance(opt["test_margin"], list)


def test_write_glove_format_skips_whitespace_tokens(tmp_path):
    # multi-word vocab tokens (e.g. 'saturated fat') can't be GloVe-encoded and
    # must be skipped, else build_pretrained_w2v's float parse breaks.
    out = tmp_path / "emb.txt"
    stats = write_glove_format(["rice", "saturated fat", "spinach", "PAD"],
                               lambda b: [[1.0, 2.0] for _ in b], out, skip_tokens={"PAD"})
    assert stats["n_words"] == 2  # 'saturated fat' (space) + 'PAD' (skip) dropped
    words = [line.split()[0] for line in out.read_text(encoding="utf-8").splitlines()]
    assert set(words) == {"rice", "spinach"}


def test_run_pfoodreq_importable():
    # the orchestrator wires generate -> build_all_data -> train -> evaluate
    from foodrag.baselines.pfoodreq import run_pfoodreq
    assert hasattr(run_pfoodreq, "run")
    assert hasattr(run_pfoodreq, "build_embeddings")
    assert hasattr(run_pfoodreq, "build_all_data")

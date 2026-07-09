"""Unit tests for the paired significance-test module (Protocol §6 / §13.3).

Covers the core statistics on known vectors (identical → p=1, clearly-different
→ p<0.05, all-zero handling), Holm–Bonferroni ordering/monotonicity, NaN-aware
run-averaging + drop-on-NaN pairing alignment, and the deterministic bootstrap.
No I/O, no HUMMUS, no LLM.
"""
from __future__ import annotations

import numpy as np
import pytest

from foodrag.analysis.stats import (
    build_paired_vectors,
    compute_pairwise,
    holm_bonferroni,
    median_diff,
    paired_bootstrap,
    paired_wilcoxon,
    per_question_means,
)


# ── paired_wilcoxon ────────────────────────────────────────────────────────
def test_wilcoxon_identical_vectors_p_is_one():
    a = [0.1, 0.5, 0.9, 0.3, 0.7, 0.2, 0.8]
    stat, p = paired_wilcoxon(a, a)
    assert p == 1.0
    assert stat == 0.0


def test_wilcoxon_clearly_different_is_significant():
    rng = np.random.default_rng(0)
    base = rng.random(50)
    a = base + 0.4          # a is consistently higher by 0.4
    b = base
    stat, p = paired_wilcoxon(a, b)
    assert p < 0.05


def test_wilcoxon_small_sample_does_not_raise():
    # 3 pairs: too few for signed-rank significance, must return gracefully.
    stat, p = paired_wilcoxon([1.0, 2.0, 3.0], [0.9, 1.8, 2.7])
    assert 0.0 <= p <= 1.0


def test_wilcoxon_empty_is_nan_p_one():
    stat, p = paired_wilcoxon([], [])
    assert np.isnan(stat)
    assert p == 1.0


# ── median_diff (effect size) ──────────────────────────────────────────────
def test_median_diff():
    a = [1.0, 2.0, 3.0, 4.0]
    b = [0.5, 1.0, 1.5, 2.0]
    # diffs = [0.5, 1.0, 1.5, 2.0] -> median 1.25
    assert median_diff(a, b) == pytest.approx(1.25)


# ── Holm–Bonferroni ────────────────────────────────────────────────────────
def test_holm_ordering_and_values():
    # Classic worked example: p = [0.01, 0.04, 0.03], m = 3.
    # sorted: 0.01(*3)=0.03, 0.03(*2)=0.06, 0.04(*1)=0.04 -> monotone: 0.06.
    p = [0.01, 0.04, 0.03]
    corr = holm_bonferroni(p)
    assert corr[0] == pytest.approx(0.03)
    assert corr[2] == pytest.approx(0.06)      # 0.03 raw, scaled by 2
    assert corr[1] == pytest.approx(0.06)      # 0.04 scaled by 1 = 0.04, lifted to 0.06 by monotonicity
    # corrected p is always >= raw p and <= 1.
    assert all(c >= r for c, r in zip(corr, p))
    assert all(c <= 1.0 for c in corr)


def test_holm_preserves_input_order():
    p = [0.2, 0.001, 0.05]
    corr = holm_bonferroni(p)
    # smallest raw p (index 1) stays smallest after correction.
    assert np.argmin(corr) == 1


def test_holm_caps_at_one_and_handles_nan():
    corr = holm_bonferroni([0.9, float("nan"), 0.8])
    assert all(0.0 <= c <= 1.0 for c in corr)
    # NaN treated as 1.0
    assert corr[1] == 1.0


def test_holm_empty():
    assert holm_bonferroni([]) == []


# ── NaN-aware run averaging + pairing alignment ────────────────────────────
def _rec(qid, run, metric, val, **extra):
    r = {"question_id": qid, "run": run, "metrics": {metric: val}}
    r.update(extra)
    return r


def test_per_question_means_nan_aware():
    recs = [
        _rec("q1", 1, "hit_at_10", 1.0),
        _rec("q1", 2, "hit_at_10", None),   # NaN run ignored
        _rec("q1", 3, "hit_at_10", 0.0),
        _rec("q2", 1, "hit_at_10", None),   # all-NaN question -> NaN
        _rec("q2", 2, "hit_at_10", None),
    ]
    means = per_question_means(recs, "hit_at_10")
    assert means["q1"] == pytest.approx(0.5)   # mean of [1.0, 0.0], None dropped
    assert np.isnan(means["q2"])


def test_build_paired_vectors_drops_nan_and_aligns():
    a_recs = [
        _rec("q1", 1, "m", 1.0),
        _rec("q2", 1, "m", 0.5),
        _rec("q3", 1, "m", None),   # NaN on a-side -> pair dropped
        _rec("q4", 1, "m", 0.2),    # q4 absent on b-side -> dropped
    ]
    b_recs = [
        _rec("q1", 1, "m", 0.0),
        _rec("q2", 1, "m", 0.5),
        _rec("q3", 1, "m", 0.9),
    ]
    a, b, qids = build_paired_vectors(a_recs, b_recs, "m")
    assert qids == ["q1", "q2"]      # q3 (NaN) and q4 (unaligned) dropped
    assert list(a) == [1.0, 0.5]
    assert list(b) == [0.0, 0.5]


def test_build_paired_vectors_stratum_filter():
    a_recs = [_rec("q1", 1, "m", 1.0, num_requirements=3),
              _rec("q2", 1, "m", 0.4, num_requirements=5)]
    b_recs = [_rec("q1", 1, "m", 0.0, num_requirements=3),
              _rec("q2", 1, "m", 0.4, num_requirements=5)]
    a, b, qids = build_paired_vectors(a_recs, b_recs, "m", {"num_requirements": 3})
    assert qids == ["q1"]


# ── paired_bootstrap ───────────────────────────────────────────────────────
def test_bootstrap_deterministic_and_ci_brackets_mean():
    rng = np.random.default_rng(1)
    a = rng.random(200) + 0.2
    b = rng.random(200)
    r1 = paired_bootstrap(a, b, n=2000, seed=42)
    r2 = paired_bootstrap(a, b, n=2000, seed=42)
    assert r1.mean_diff == pytest.approx(r2.mean_diff)
    assert r1.ci_low == pytest.approx(r2.ci_low)      # reproducible with seed
    assert r1.ci_low <= r1.mean_diff <= r1.ci_high
    assert r1.p_value < 0.05                            # a clearly > b


def test_bootstrap_all_zero_diff():
    r = paired_bootstrap([0.3, 0.3, 0.3], [0.3, 0.3, 0.3], n=500)
    assert r.mean_diff == 0.0
    assert r.p_value == 1.0


# ── end-to-end record generation on a tiny synthetic dataset ───────────────
def test_compute_pairwise_holm_within_metric_family():
    rng = np.random.default_rng(7)
    qids = [f"q{i}" for i in range(40)]
    base = {q: rng.random() for q in qids}

    def mk(offset):
        return [_rec(q, 1, "hit_at_10", min(1.0, base[q] + offset)) for q in qids]

    records = {
        "foodrag_h1_t2p": mk(0.3),
        "foodrag_h2_t2p": mk(0.0),
        "foodrag_h1_dense": mk(0.05),
    }
    pairs = [("foodrag_h1_t2p", "foodrag_h2_t2p"),
             ("foodrag_h1_t2p", "foodrag_h1_dense")]
    rows = compute_pairwise(records, config_pairs=pairs)
    # every row carries the §13.3 fields incl. the correction.
    for r in rows:
        assert r["experiment_id"] == "EXP-7"
        assert "p_value_corrected" in r and "significant" in r
        assert r["p_value_corrected"] >= r["p_value"] - 1e-12
        assert r["n_pairs"] == 40
    assert {r["metric"] for r in rows} == {"hit_at_10"}

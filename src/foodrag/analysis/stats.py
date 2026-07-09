"""Paired per-question significance tests between FoodRAG configs (Protocol §6).

The thesis (and the current results pipeline) report means ± SD and bootstrap
95% CIs only. Protocol §6 ("Statistical Testing Protocol") additionally requires
**paired** comparison tests over per-question scores, and §13.3 defines the
Pairwise-Comparison Record this module emits.

What §6 asks for and where it lives here:

* **Paired Wilcoxon signed-rank** over per-question scores averaged across runs
  → :func:`paired_wilcoxon`.
* **Paired bootstrap** (10,000 resamples) for the difference in means, an allowed
  alternative → :func:`paired_bootstrap`.
* **Effect size** = median per-question difference (a − b) → :func:`median_diff`.
* **Holm–Bonferroni** correction across the family of comparisons sharing a metric
  → :func:`holm_bonferroni`.

Everything here is **offline and memory-light**: it reads the per-question JSON
files under ``eval_results/json/`` (numpy + scipy only — no HUMMUS, no LLM, no
Streamlit) and is unit-testable in isolation. The per-question files may be
rewritten by a concurrent regeneration job, so :func:`load_per_question` reads
each file once and retries a few times on a partial-write parse failure.

Pairing rule (§6.2–6.3): for a metric and a config, each question's score is the
NaN-aware mean of that metric across the config's runs; two configs are then
aligned on ``question_id`` and pairs where **either** side is NaN are dropped.

Run ``python -m foodrag.analysis.stats`` to (re)generate
``eval_results/pairwise.csv`` and ``eval_results/json/pairwise.json`` (EXP-7).
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy import stats

# ── metric polarity ────────────────────────────────────────────────────────
# Every reported metric here is "higher is better" EXCEPT violation/error rates.
# Used only to phrase the "a beats b" direction in the report; the numeric
# records (mean_diff, median_diff, p_value) are polarity-agnostic.
LOWER_IS_BETTER = frozenset({"allergen_violation", "hard_constraint_violation",
                             "unsupported_claim_rate", "hallucination_rate"})


def higher_is_better(metric: str) -> bool:
    return metric not in LOWER_IS_BETTER


# ── core paired statistics ─────────────────────────────────────────────────
def paired_wilcoxon(a: Sequence[float], b: Sequence[float]) -> Tuple[float, float]:
    """Paired Wilcoxon signed-rank test on aligned vectors ``a`` and ``b``.

    Returns ``(statistic, p_value)``. Edge cases handled per §6:

    * all per-question differences are (numerically) zero → ``(0.0, 1.0)``
      (scipy raises for this; there is no signal to test).
    * fewer than ~6 non-zero-difference pairs → the signed-rank test cannot reach
      significance; we still return scipy's value when it computes, else
      ``(nan, 1.0)`` rather than raising.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"paired vectors must be same length, got {a.shape} vs {b.shape}")
    d = a - b
    if d.size == 0:
        return float("nan"), 1.0
    if np.allclose(d, 0.0, atol=1e-12):
        return 0.0, 1.0
    n_nonzero = int(np.count_nonzero(~np.isclose(d, 0.0, atol=1e-12)))
    try:
        res = stats.wilcoxon(a, b)
    except ValueError:
        # e.g. all-zero after scipy's own zero handling, or too few samples.
        return float("nan"), 1.0
    p = float(res.pvalue)
    if n_nonzero < 6 and np.isnan(p):
        return float(res.statistic), 1.0
    return float(res.statistic), p


def median_diff(a: Sequence[float], b: Sequence[float]) -> float:
    """Effect size (§6): the median per-question difference ``a − b``."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size == 0:
        return float("nan")
    return float(np.median(a - b))


@dataclass
class BootstrapResult:
    mean_diff: float
    ci_low: float
    ci_high: float
    p_value: float


def paired_bootstrap(
    a: Sequence[float],
    b: Sequence[float],
    n: int = 10000,
    seed: int = 0,
    alpha: float = 0.05,
) -> BootstrapResult:
    """Paired bootstrap for the difference in means (§6, allowed alternative).

    Resamples the paired differences ``d = a − b`` (``n`` resamples with a fixed
    ``seed``), returning the observed mean difference, its percentile CI
    (``[alpha/2, 1-alpha/2]``), and a two-sided bootstrap p-value defined as the
    fraction of resample means on the far side of 0 from the observed mean,
    doubled (§6 "fraction of resamples with diff crossing 0"). p is clipped to
    ``[1/n, 1]`` — a zero count means "smaller than the resolution", not exactly 0.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    d = a - b
    m = d.size
    if m == 0:
        return BootstrapResult(float("nan"), float("nan"), float("nan"), 1.0)
    obs = float(np.mean(d))
    if np.allclose(d, 0.0, atol=1e-12):
        return BootstrapResult(0.0, 0.0, 0.0, 1.0)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, m, size=(n, m))
    boot_means = d[idx].mean(axis=1)
    lo = float(np.percentile(boot_means, 100 * (alpha / 2)))
    hi = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    # fraction of resamples on the opposite side of 0 from the observed mean.
    if obs >= 0:
        tail = float(np.mean(boot_means <= 0.0))
    else:
        tail = float(np.mean(boot_means >= 0.0))
    p = min(1.0, max(1.0 / n, 2.0 * tail))
    return BootstrapResult(obs, lo, hi, p)


def holm_bonferroni(pvalues: Sequence[float]) -> List[float]:
    """Holm–Bonferroni step-down correction (§6.5), preserving input order.

    For ``m`` p-values sorted ascending, the ``k``-th smallest is scaled by
    ``m − k`` and the running maximum is enforced (monotone, capped at 1.0).
    NaN p-values (a test that could not be computed) are treated as 1.0 for the
    correction and returned as 1.0.
    """
    p = np.asarray(pvalues, dtype=float)
    m = p.size
    if m == 0:
        return []
    clean = np.where(np.isnan(p), 1.0, p)
    order = np.argsort(clean, kind="mergesort")  # stable
    corrected = np.empty(m, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * float(clean[idx]))
        corrected[idx] = min(running, 1.0)
    return [float(x) for x in corrected]


# ── per-question loading & pairing ─────────────────────────────────────────
def load_per_question(
    path: Path | str,
    retries: int = 5,
    wait_s: float = 2.0,
) -> List[dict]:
    """Read one ``<config>_per_question.json`` once, retrying on a partial write.

    A concurrent regeneration job may be mid-write; a ``JSONDecodeError`` (or a
    transient read error) triggers up to ``retries`` attempts spaced ``wait_s``
    apart. Any other error is raised immediately.
    """
    path = Path(path)
    last: Optional[Exception] = None
    for attempt in range(retries):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as exc:  # partial / locked write
            last = exc
            if attempt < retries - 1:
                time.sleep(wait_s)
    raise RuntimeError(f"could not read {path} after {retries} attempts") from last


def _nanmean(values: Sequence[float]) -> float:
    """NaN-aware mean of a question's metric across runs (None → NaN)."""
    arr = np.array([np.nan if v is None else v for v in values], dtype=float)
    if arr.size == 0 or np.all(np.isnan(arr)):
        return float("nan")
    return float(np.nanmean(arr))


def per_question_means(
    records: Sequence[dict],
    metric: str,
    stratum_filter: Optional[Mapping[str, object]] = None,
) -> Dict[str, float]:
    """``{question_id: mean-across-runs}`` for one metric.

    Records missing the metric contribute NaN for that run. ``stratum_filter`` is
    an optional ``{record_field: value}`` restriction (e.g.
    ``{"num_requirements": 4}``) applied before pairing; ``None`` = all questions
    (the "aggregate" stratum).
    """
    buckets: Dict[str, List[float]] = {}
    for r in records:
        if stratum_filter and any(r.get(k) != v for k, v in stratum_filter.items()):
            continue
        qid = r.get("question_id")
        if qid is None:
            continue
        val = (r.get("metrics") or {}).get(metric)
        buckets.setdefault(str(qid), []).append(np.nan if val is None else val)
    return {qid: _nanmean(vals) for qid, vals in buckets.items()}


def build_paired_vectors(
    records_a: Sequence[dict],
    records_b: Sequence[dict],
    metric: str,
    stratum_filter: Optional[Mapping[str, object]] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Aligned per-question paired vectors for ``metric`` across two configs.

    Averages each question's metric across runs (NaN-aware), aligns on the shared
    ``question_id`` set, and drops pairs where either side is NaN. Returns
    ``(a, b, question_ids)`` with ``question_ids`` the retained, sorted keys.
    """
    ma = per_question_means(records_a, metric, stratum_filter)
    mb = per_question_means(records_b, metric, stratum_filter)
    common = sorted(set(ma) & set(mb))
    a_vals, b_vals, qids = [], [], []
    for qid in common:
        av, bv = ma[qid], mb[qid]
        if np.isnan(av) or np.isnan(bv):
            continue
        a_vals.append(av)
        b_vals.append(bv)
        qids.append(qid)
    return np.asarray(a_vals, dtype=float), np.asarray(b_vals, dtype=float), qids


def metric_available(records: Sequence[dict], metric: str) -> bool:
    """True iff at least one record carries a non-null value for ``metric``."""
    for r in records:
        v = (r.get("metrics") or {}).get(metric)
        if v is not None and not (isinstance(v, float) and np.isnan(v)):
            return True
    return False


# ── §13.3 pairwise record generation ───────────────────────────────────────
# The interpretable comparisons (task spec). Each entry: (config_a, config_b).
CONFIG_PAIRS: List[Tuple[str, str]] = [
    # Agentic (H1) vs recommend-only (H2), per retriever.
    ("foodrag_h1_dense", "foodrag_h2"),
    ("foodrag_h1_t2p", "foodrag_h2_t2p"),
    # Retriever sensitivity (t2p vs dense), per system.
    ("foodrag_h1_t2p", "foodrag_h1_dense"),
    ("foodrag_h2_t2p", "foodrag_h2"),
    # FoodRAG-best vs baselines (shared metrics only).
    ("foodrag_h1_t2p", "naive_rag"),
    ("foodrag_h1_t2p", "llm_only"),
]

# Metric family (task spec). allergen_violation is added for the baseline pairs
# (it is the safety metric the baselines DO carry); it is skipped automatically
# for any config that lacks it.
FOODRAG_METRICS: List[str] = [
    "hit_at_10", "mrr_at_10",
    "correctness_judgeA", "faithfulness_judgeA", "relevancy_judgeA",
    "hard_constraint_sat_perconstraint", "guideline_adherence_overall",
    "path_coverage", "claim_grounding_rate",
]
BASELINE_METRICS: List[str] = ["hit_at_10", "mrr_at_10", "allergen_violation"]
BASELINE_CONFIGS = frozenset({"naive_rag", "llm_only"})


def _metrics_for_pair(config_a: str, config_b: str) -> List[str]:
    if config_a in BASELINE_CONFIGS or config_b in BASELINE_CONFIGS:
        return BASELINE_METRICS
    return FOODRAG_METRICS


def compute_pairwise(
    records_by_config: Mapping[str, Sequence[dict]],
    config_pairs: Sequence[Tuple[str, str]] = tuple(CONFIG_PAIRS),
    stratum: str = "aggregate",
    stratum_filter: Optional[Mapping[str, object]] = None,
    experiment_id: str = "EXP-7",
    bootstrap_n: int = 10000,
    seed: int = 0,
) -> List[dict]:
    """Build §13.3 Pairwise-Comparison Records for every (metric, config pair).

    Holm–Bonferroni is applied **within each metric family** (all config pairs
    tested for that metric at this stratum), per §6.5. Returns a list of dict
    records ready for CSV/JSON serialisation.
    """
    # Collect raw rows first, grouped by metric so we can correct within-family.
    rows: List[dict] = []
    for config_a, config_b in config_pairs:
        ra = records_by_config.get(config_a)
        rb = records_by_config.get(config_b)
        if ra is None or rb is None:
            continue
        for metric in _metrics_for_pair(config_a, config_b):
            if not (metric_available(ra, metric) and metric_available(rb, metric)):
                continue
            a, b, qids = build_paired_vectors(ra, rb, metric, stratum_filter)
            if a.size == 0:
                continue
            stat, p = paired_wilcoxon(a, b)
            boot = paired_bootstrap(a, b, n=bootstrap_n, seed=seed)
            rows.append({
                "experiment_id": experiment_id,
                "metric": metric,
                "stratum": stratum,
                "config_a": config_a,
                "config_b": config_b,
                "mean_a": float(np.mean(a)),
                "mean_b": float(np.mean(b)),
                "mean_diff": float(np.mean(a) - np.mean(b)),
                "median_diff": median_diff(a, b),
                "test": "wilcoxon_paired",
                "statistic": stat,
                "p_value": p,
                "bootstrap_mean_diff": boot.mean_diff,
                "bootstrap_ci_low": boot.ci_low,
                "bootstrap_ci_high": boot.ci_high,
                "bootstrap_p_value": boot.p_value,
                "n_pairs": int(a.size),
                "higher_is_better": higher_is_better(metric),
            })

    # Holm–Bonferroni within each metric family (same stratum).
    by_metric: Dict[str, List[dict]] = {}
    for row in rows:
        by_metric.setdefault(row["metric"], []).append(row)
    for family in by_metric.values():
        corrected = holm_bonferroni([r["p_value"] for r in family])
        for r, pc in zip(family, corrected):
            r["p_value_corrected"] = pc
            r["significant"] = bool(pc < 0.05)
    return rows


# ── CLI entry point ────────────────────────────────────────────────────────
_JSON_COLUMNS = [
    "experiment_id", "metric", "stratum", "config_a", "config_b",
    "mean_a", "mean_b", "mean_diff", "median_diff",
    "test", "statistic", "p_value", "p_value_corrected", "significant",
    "n_pairs", "higher_is_better",
    "bootstrap_mean_diff", "bootstrap_ci_low", "bootstrap_ci_high", "bootstrap_p_value",
]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def main() -> None:
    import csv

    root = _repo_root()
    json_dir = root / "eval_results" / "json"
    needed = {c for pair in CONFIG_PAIRS for c in pair}
    records_by_config = {
        cid: load_per_question(json_dir / f"{cid}_per_question.json") for cid in sorted(needed)
    }
    rows = compute_pairwise(records_by_config)
    rows.sort(key=lambda r: (r["metric"], r["config_a"], r["config_b"]))

    out_csv = root / "eval_results" / "pairwise.csv"
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_JSON_COLUMNS)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k) for k in _JSON_COLUMNS})

    out_json = json_dir / "pairwise.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump([{k: r.get(k) for k in _JSON_COLUMNS} for r in rows], f,
                  ensure_ascii=False, indent=1)

    n_sig = sum(1 for r in rows if r.get("significant"))
    print(f"wrote {len(rows)} pairwise records ({n_sig} significant after Holm) -> "
          f"{out_csv} and {out_json}")


if __name__ == "__main__":
    main()

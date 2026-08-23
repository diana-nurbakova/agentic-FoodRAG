"""Run a baseline over a QA set and emit Results-Schema records.

Produces both halves of the schema:

* :func:`run_baseline` →.1 per-question records (one per question × run),
  each tagged with its strata.
* :func:`aggregate` →.2 per ``config × stratum × metric`` rows with mean,
  bootstrap 95% CI, per-question (run-to-run) variance and counts.

:func:`write_results` lands a per-question Parquet plus an ``aggregate.csv`` that
matches the columns the Report tab already reads from ``eval_results/``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd

from foodrag.baselines import base, constraints, metrics
from foodrag.baselines.base import Baseline

ID_COL = "recipe_url"
DEFAULT_K = 10

# aggregate column order (matches eval_results/aggregate.csv).
AGG_COLUMNS = ["experiment_id", "config_id", "stratum", "metric", "mean",
               "ci_low", "ci_high", "per_question_variance", "n_questions", "n_runs", "judge"]


def load_recipes_df(path: str | Path) -> pd.DataFrame:
    """Read ``recipes_df.csv`` and deserialise the list columns the metrics need.

    Mirrors ``load_util._load_hummus`` for ``tags`` / ``ingredient_names`` so the
    constraint predicates see real lists, not their string reprs.
    """
    from foodrag.utils.load_util import serialize

    df = pd.read_csv(path)
    for col in ("tags", "ingredient_names"):
        if col in df.columns:
            df[col] = df[col].apply(serialize)
    return df


def run_baseline(baseline: Baseline, qa_set: Iterable[dict], recipes_df: pd.DataFrame,
                 *, k: int = DEFAULT_K, experiment_id: str = "EXP-7", run: int = 1,
                 expand: constraints.IngredientExpander = constraints.identity_expand,
                 id_prefix: str = "q", max_workers: int = 1) -> list[dict]:
    """Score ``baseline`` on every QA item; return.1 per-question records.

    ``max_workers > 1`` runs the per-question ``baseline.recommend`` calls (the
    slow LLM step for rungs 2/ToG) concurrently in a thread pool, preserving
    order; metric computation stays sequential and deterministic. ``=1`` is the
    original sequential behaviour.
    """
    indexed = recipes_df.set_index(ID_COL, drop=False)
    needed = [c for c in indexed.columns if c not in ("tags_str",)]
    qa_list = list(qa_set)

    # Phase 1 — recommendations (parallelisable I/O-bound LLM calls). Each call is
    # wall-clock timed so per-question latency is reported alongside the
    # quality metrics. Under threading the timing is the call's own duration.
    import time as _time

    def _timed_recommend(qa):
        _t0 = _time.perf_counter()
        rec = baseline.recommend(qa, k)
        return rec, _time.perf_counter() - _t0

    if max_workers and max_workers > 1:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            paired = list(pool.map(_timed_recommend, qa_list))
    else:
        paired = [_timed_recommend(qa) for qa in qa_list]
    recs = [r for r, _ in paired]
    latencies = [lat for _, lat in paired]

    # Phase 2 — metrics + records (fast, ordered, deterministic).
    records: list[dict] = []
    for i, (qa_item, retrieved_ids) in enumerate(zip(qa_list, recs)):
        gold = base.gold_urls(qa_item)
        persona = qa_item["persona"]

        rows = _resolve_rows(retrieved_ids, indexed, needed)

        metric_block = metrics.deterministic_metrics(
            retrieved_ids, rows, gold, persona, k, expand)
        metric_block["latency_s"] = latencies[i]  # per-question execution time
        if not gold:  # retrieval metrics are undefined without a gold set
            for m in (f"hit_at_{k}", f"mrr_at_{k}", f"ndcg_at_{k}"):
                metric_block[m] = float("nan")

        records.append({
            "experiment_id": experiment_id,
            "config_id": baseline.config_id,
            "rung": baseline.rung,
            "run": run,
            "question_id": f"{id_prefix}_{i:05d}",
            "answer_set_size": len(gold),
            "stratum": base.answer_set_stratum(len(gold)),
            "num_requirements": base.requirement_count(persona),
            "requirement_types": base.requirement_types(persona),
            "k": k,
            "metrics": metric_block,
            "retrieved_ids": retrieved_ids,
        })

    return records


def _resolve_rows(retrieved_ids: Sequence[str], indexed: pd.DataFrame,
                  needed: list[str]) -> list[Optional[dict]]:
    """Map each id to its recipe row dict, or ``None`` if it isn't a real recipe."""
    rows: list[Optional[dict]] = []
    for rid in retrieved_ids:
        if rid in indexed.index:
            row = indexed.loc[rid]
            if isinstance(row, pd.DataFrame):  # duplicate id → take the first
                row = row.iloc[0]
            rows.append(row.to_dict())
        else:
            rows.append(None)
    return rows


# ── aggregation ──────────────────────────────────────────────────────────────

def _bootstrap_ci(values: np.ndarray, n_boot: int, seed: int,
                  block: int = 500) -> tuple[float, float]:
    """95% percentile bootstrap CI of the mean; degenerate inputs return the mean.

    The resample matrix is drawn in row blocks rather than as one
    ``(n_boot, n)`` array. At n_boot = 10,000 the one-shot draw is 125 MB for a
    1,646-question config and 763 MB for a 10,000-question one, which raises
    ``MemoryError`` on a constrained box — it is why ``build_claim_registry.py``
    and ``reaggregate_corrected.py`` could not run to completion.

    ``Generator.choice`` fills row-major, so k blocks of ``(rows, n)`` consume
    the stream in exactly the order one ``(n_boot, n)`` draw would: the
    resamples, and therefore the interval, are **bit-identical**. Verified over
    several (n_boot, n) shapes before this replaced the one-shot draw.
    """
    if values.size == 0:
        return float("nan"), float("nan")
    if values.size == 1:
        return float(values[0]), float(values[0])
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=float)
    done = 0
    while done < n_boot:
        rows = min(block, n_boot - done)
        while True:
            try:
                means[done:done + rows] = rng.choice(
                    values, size=(rows, values.size), replace=True).mean(axis=1)
                break
            except MemoryError:
                if rows == 1:
                    raise
                rows = max(1, rows // 4)   # same stream either way, so still exact
        done += rows
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def aggregate(records: Sequence[dict], *, experiment_id: Optional[str] = None,
              config_id: Optional[str] = None, judge: str = "",
              n_boot: int = 10_000, seed: int = 0) -> pd.DataFrame:
    """Roll per-question records up to.2 ``config × stratum × metric`` rows.

    ``per_question_variance`` is the mean over questions of each question's
    variance across runs — 0 for single-run deterministic baselines.
    """
    if not records:
        return pd.DataFrame(columns=AGG_COLUMNS)
    experiment_id = experiment_id or records[0]["experiment_id"]
    config_id = config_id or records[0]["config_id"]
    n_runs = len({r["run"] for r in records})
    metric_names = list(records[0]["metrics"].keys())

    rows: list[dict] = []
    for s_idx, stratum in enumerate(["aggregate"] + _strata(records)):
        subset = records if stratum == "aggregate" else [r for r in records if r["stratum"] == stratum]
        for m_idx, metric in enumerate(metric_names):
            per_q = _per_question_means(subset, metric)
            values = np.array([v for v in per_q.values() if not np.isnan(v)], dtype=float)
            if values.size == 0:
                continue
            ci_low, ci_high = _bootstrap_ci(values, n_boot, seed + 1000 * s_idx + m_idx)
            rows.append({
                "experiment_id": experiment_id,
                "config_id": config_id,
                "stratum": stratum,
                "metric": metric,
                "mean": float(values.mean()),
                "ci_low": ci_low,
                "ci_high": ci_high,
                "per_question_variance": _run_variance(subset, metric),
                "n_questions": int(values.size),
                "n_runs": n_runs,
                "judge": judge,
            })
    return pd.DataFrame(rows, columns=AGG_COLUMNS)


def _strata(records: Sequence[dict]) -> list[str]:
    order = {"tight": 0, "medium": 1, "broad": 2, "empty": 3}
    present = {r["stratum"] for r in records}
    return sorted(present, key=lambda s: order.get(s, 99))


def _asfloat(v) -> float:
    """Coerce a metric value to float; non-numeric (dict/str/None) → NaN.

    Some.1 metric fields are non-scalar (e.g. ``guideline_adherence_per_nutrient``
    is a dict, ``guideline_class`` a string). Those are kept in the per-question
    record for fidelity but cannot be averaged — coercing them to NaN makes
    :func:`aggregate` skip them instead of raising, while their flattened scalar
    counterparts (``guideline_adherence_<nutrient>``) aggregate normally.
    """
    if isinstance(v, bool):
        return float(v)
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def _per_question_means(records: Sequence[dict], metric: str) -> dict[str, float]:
    """For each question, the mean of ``metric`` across its runs (NaN-aware)."""
    by_q: dict[str, list[float]] = {}
    for r in records:
        by_q.setdefault(r["question_id"], []).append(_asfloat(r["metrics"].get(metric, float("nan"))))
    return {q: float(np.nanmean(v)) if not _all_nan(v) else float("nan") for q, v in by_q.items()}


def _run_variance(records: Sequence[dict], metric: str) -> float:
    """Mean over questions of the across-run variance of ``metric``."""
    by_q: dict[str, list[float]] = {}
    for r in records:
        by_q.setdefault(r["question_id"], []).append(_asfloat(r["metrics"].get(metric, float("nan"))))
    variances = [float(np.nanvar(v)) for v in by_q.values() if not _all_nan(v)]
    return float(np.mean(variances)) if variances else 0.0


def _all_nan(values: list[float]) -> bool:
    return all(np.isnan(v) for v in values)


# ── persistence ──────────────────────────────────────────────────────────────

def write_results(records: Sequence[dict], aggregates: pd.DataFrame,
                  out_dir: str | Path = "eval_results", *, config_id: Optional[str] = None) -> dict[str, Path]:
    """Write per-question Parquet + merge aggregate rows into ``aggregate.csv``.

    Aggregate rows for the same ``(experiment_id, config_id, stratum, metric,
    judge)`` are replaced, so re-running a baseline updates its rows in place
    without disturbing other configs already in the file.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    config_id = config_id or (records[0]["config_id"] if records else "baseline")

    per_q = pd.DataFrame(records)
    pq_path = out_dir / f"{config_id}_per_question.parquet"
    try:
        per_q.to_parquet(pq_path, index=False)
    except Exception:  # parquet engine missing → fall back to pickle
        pq_path = pq_path.with_suffix(".pkl")
        per_q.to_pickle(pq_path)

    agg_path = out_dir / "aggregate.csv"
    keys = ["experiment_id", "config_id", "stratum", "metric", "judge"]
    if agg_path.exists():
        existing = pd.read_csv(agg_path, keep_default_na=False, na_values=[""])
        merged = pd.concat([existing, aggregates], ignore_index=True)
        merged = merged.drop_duplicates(subset=keys, keep="last")
    else:
        merged = aggregates
    merged.to_csv(agg_path, index=False)

    return {"per_question": pq_path, "aggregate": agg_path}

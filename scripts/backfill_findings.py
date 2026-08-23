"""Backfill machine-readable backing for the ``specs/*-findings.md`` reports.

Five findings reports predate the audit tooling and quote numbers that exist
only as markdown tables: `checks-T-Q-H`, `A1-corpus-size`, `D2-agentic-value`,
`E3-E4-P3-P4` and `P1-safety-fix-contamination`. Their analysis scripts lived in
session scratchpads and are gone. This module regenerates those numbers from the
authoritative artefacts and writes them to ``audit/`` as CSV/JSON.

    foodrag-backfill                      # every section that can run here
    foodrag-backfill --only qh,p1         # a subset
    foodrag-backfill --list               # what each section produces

Each section also **re-checks its own report**: headline values are declared in
:data:`EXPECTED` with the figure the report prints, and the manifest records
``match`` / ``MISMATCH`` per value. A backfill that silently disagrees with the
document it backs is worse than no backfill, so a mismatch is a non-zero exit.

Two sections need inputs that are not reproducible off this machine:

* ``t`` and ``a1`` read ``data/``, which is gitignored (``.gitignore:149``).
* ``p1`` and ``d2`` read per-question parquets out of git history at
  ``74176e8^`` — reproducible anywhere the repo is cloned, but not from the
  working tree alone.

Sections whose inputs are absent are recorded as ``skipped`` in the manifest
rather than failing the run.
"""
from __future__ import annotations

# scipy must be imported before any parquet is loaded: with a large resident
# working set Windows fails its import with "paging file is too small", and the
# pairwise bootstrap allocates 10,000 x n floats per metric-pair. See
# specs/D2-agentic-value-findings.md, the environment note.
import scipy.stats  # noqa: F401  (import-order side effect, deliberate)

import argparse
import ast
import io
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "audit"

# The commit that applied the safety fixes. Its parent holds the pre-fix records.
FIX_COMMIT = "74176e8"
PRE = f"{FIX_COMMIT}^"

EMPTY_PLAN_MESSAGE = (
    "No meal plan could be recommended that satisfies the given requirements."
)

# Headline values each report prints, checked against what this module derives.
# Keyed section -> label -> (expected, tolerance). Tolerance 0 means exact.
EXPECTED: dict[str, dict[str, tuple[float, float]]] = {
    "qh": {
        "llama70b_empty_question_runs": (694, 0),
        "gpt4o_empty_question_runs": (4528, 0),
        "intersection_empty_question_runs": (1587, 0),
        "llama70b_hit_at_10": (0.853827460510, 1e-11),
        "gpt4o_hit_at_10": (0.427095990279, 1e-11),
        "llama70b_conditional_hit_at_10": (0.932589, 5e-7),
        "gpt4o_conditional_hit_at_10": (0.947059, 5e-7),
        "h1_t2p_conditional_hit_at_10": (0.846196, 5e-7),
        "h1_dense_conditional_hit_at_10": (0.027440, 5e-7),
        "B2_ratio_llama70b": (15.4440, 5e-4),
        "B2_ratio_gpt4o": (7.7253, 5e-4),
    },
    "p1": {
        "fix1_total": (72, 0),
        "fix2_total": (421, 0),
        "stamped_total": (493, 0),
        "h1_t2p_prefix_allergen": (0.000310, 5e-7),
        "h1_dense_prefix_allergen": (0.001969, 5e-7),
        "h2_prefix_allergen": (0.003427, 5e-7),
        "h2_t2p_prefix_allergen": (0.000608, 5e-7),
        "fullday_hallucinated_total": (643, 0),
        # P1 §4 — the re-judge table, config-wide on h1_t2p
        "h1_t2p_all_faithfulness_judgeB_changed": (395, 0),
        "h1_t2p_all_faithfulness_judgeB_prefix": (0.913412, 5e-7),
        "h1_t2p_all_faithfulness_judgeA_changed": (371, 0),
        "h1_t2p_all_faithfulness_judgeA_prefix": (0.920175, 5e-7),
        "h1_t2p_all_relevancy_judgeB_changed": (407, 0),
        "h1_t2p_all_relevancy_judgeA_changed": (398, 0),
        "h1_t2p_all_correctness_judgeA_changed": (388, 0),
        "h1_t2p_all_correctness_judgeA_prefix": (0.687396, 5e-7),
        "h1_t2p_all_claim_grounding_rate_changed": (406, 0),
        "h1_t2p_all_explanation_quality_judgeB_changed": (421, 0),
    },
    "d2": {
        "strong_correctness_prefix_a": (0.688502, 5e-7),
        "strong_correctness_published_a": (0.765216, 5e-7),
        "strong_correctness_prefix_b": (0.586072, 5e-7),
        "strong_correctness_published_b": (0.586022, 5e-7),
        "fix2_records": (421, 0),
        "correctness_judgeA_changed": (385, 0),
        "correctness_judgeB_changed": (203, 0),
        # D2.2 quotes these to 4 dp; they also verify the chunked bootstrap
        # reproduces the stock one-shot draw exactly.
        "strong_correctness_prefix_ci_low": (0.0906, 5e-5),
        "strong_correctness_prefix_ci_high": (0.1141, 5e-5),
        "strong_correctness_prefix_median_diff": (0.0833, 5e-5),
        "strong_correctness_published_median_diff": (0.1667, 5e-5),
    },
    "e34": {
        "rows_changed": (13, 0),
        "coerced_nans_total": (930, 0),
        "tog_hit_n_published": (300, 0),
        "tog_hit_n_corrected": (100, 0),
        "tog_hit_var_published": (0.0, 0),
        "tog_hit_var_corrected": (0.031111, 5e-6),
        "h1_t2p_hit_published": (0.772875, 5e-7),
        "h1_t2p_hit_harmonised": (0.756784, 5e-7),
        "h1_t2p_fullday_hit_harmonised": (0.765087, 5e-7),
        "D3_ratio_harmonised": (28.10, 5e-3),
    },
    "a1": {
        "ret_configs_outside_corpus": (0, 0),
        "ids_ret_dense": (9601, 0),
        "ids_ret_sparse": (5132, 0),
        "ids_ret_fulltext": (5132, 0),
        "ids_ret_hybrid": (9600, 0),
        "ids_ret_random": (15961, 0),
        "ids_ret_intersection": (56, 0),
        "ids_ret_text2pandas@gpt4o-mini": (7694, 0),
        "ids_ret_text2pandas@llama70b": (7799, 0),
    },
    "t": {
        "corpus_rows": (270284, 0),
        "tags_non_empty": (232777, 0),
        "meal_types_non_empty": (266726, 0),
        # T.6 — the appended/native split, on the correct recipe_url join
        "appended_tag_instances": (20667, 0),
        "appended_recipes": (19058, 0),
        "appended_breakfast": (2412, 0),
        "appended_lunch": (2865, 0),
        "appended_dinner": (11012, 0),
        "appended_snack": (4378, 0),
        "native_breakfast": (13520, 0),
        "native_lunch": (23660, 0),
        "native_dinner": (0, 0),
        "native_snack": (0, 0),
    },
}


# ── helpers ──────────────────────────────────────────────────────────────────

def _metric(m: Any, key: str) -> float:
    if not isinstance(m, dict):
        return float("nan")
    v = m.get(key)
    if v is None:
        return float("nan")
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def _is_empty(a: Any) -> bool:
    return a is None or len(a) == 0


def _safe_nanmean(s: pd.Series) -> Optional[float]:
    """``nanmean`` that returns None instead of warning on an all-NaN slice.

    An all-NaN column is a real observation here — ``claim_grounding_rate`` is
    undefined post-fix on empty-plan records, because there is no claim left to
    ground — so it is recorded as None rather than coerced to 0.
    """
    return None if s.isna().all() else float(np.nanmean(s))


def _per_question_mean(df: pd.DataFrame, metric: str) -> float:
    """The published aggregation: mean across runs per question, then mean.

    Mirrors ``runner._per_question_means`` (``runner.py:161-175``, ``205-210``).
    """
    v = df["metrics"].apply(lambda m: _metric(m, metric))
    per_q = v.groupby(df["question_id"]).apply(
        lambda s: float(np.nanmean(s)) if s.notna().any() else float("nan"))
    return float(np.nanmean(per_q))


def read_parquet_at(rev: str, path: str) -> Optional[pd.DataFrame]:
    """One per-question parquet as it stood at ``rev``."""
    try:
        blob = subprocess.run(["git", "show", f"{rev}:{path}"], cwd=ROOT,
                              capture_output=True, check=True).stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return pd.read_parquet(io.BytesIO(blob))


def _write(name: str, obj: pd.DataFrame | dict | list) -> str:
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / name
    if isinstance(obj, pd.DataFrame):
        obj.to_csv(p, index=False)
        n = len(obj)
    else:
        p.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
        n = len(obj)
    print(f"    -> audit/{name}  ({n} rows)")
    return name


# ── section: Q and H, from checks-T-Q-H-findings.md ──────────────────────────

RET_SOURCES = {
    "ret_text2pandas@llama70b": "eval_results_llama",
    "ret_text2pandas@gpt4o-mini": "eval_results",
    "ret_intersection": "eval_results",
    "ret_dense": "eval_results",
    "ret_sparse": "eval_results",
    "ret_hybrid": "eval_results",
    "ret_fulltext": "eval_results",
    "ret_random": "eval_results",
}


def section_qh(_: argparse.Namespace) -> dict:
    """Q.1-Q.6 and H.3 — empties, conditional hit@10, per-run split, B2 ratios."""
    got: dict[str, float] = {}
    outputs: list[str] = []

    # ── Q.1 / Q.3 — per config ──
    per_config, per_run, stability = [], [], []
    for cfg, d in RET_SOURCES.items():
        p = ROOT / d / f"{cfg}_per_question.parquet"
        if not p.exists():
            continue
        df = pd.read_parquet(p)
        df["_empty"] = df["retrieved_ids"].apply(_is_empty)
        df["_hit"] = df["metrics"].apply(lambda m: _metric(m, "hit_at_10"))
        df["_lat"] = df["metrics"].apply(lambda m: _metric(m, "latency_s"))
        gold_nz = (df["answer_set_size"] > 0 if "answer_set_size" in df.columns
                   else pd.Series(True, index=df.index))
        e = df["_empty"]

        per_config.append({
            "config_id": cfg, "source_dir": d,
            "question_runs": len(df), "runs": int(df["run"].nunique()),
            "questions": int(df["question_id"].nunique()),
            "empty_question_runs": int(e.sum()),
            "empty_rate": round(float(e.mean()), 6),
            # Q.1: an empty result is scored 0.0, never excluded
            "empty_scored_zero": int((e & (df["_hit"] == 0.0)).sum()),
            "empty_scored_nan": int((e & df["_hit"].isna()).sum()),
            "gold_empty_rows": int((~gold_nz).sum()),
            "answer_set_size_min": (int(df["answer_set_size"].min())
                                    if "answer_set_size" in df.columns else None),
            "hit_at_10_all": _per_question_mean(df, "hit_at_10"),
            "hit_at_10_nonempty_gold": _per_question_mean(df[gold_nz], "hit_at_10"),
            "hit_at_10_nonempty_result": _per_question_mean(df[~e], "hit_at_10"),
        })

        for run, g in df.groupby("run"):
            lat = g["_lat"].dropna()
            per_run.append({
                "config_id": cfg, "run": int(run), "questions": len(g),
                "empty": int(g["_empty"].sum()),
                "empty_rate": round(float(g["_empty"].mean()), 6),
                "hit_at_10": float(np.nanmean(g["_hit"])),
                "lat_min": float(lat.min()) if len(lat) else None,
                "lat_mean": float(lat.mean()) if len(lat) else None,
                "lat_max": float(lat.max()) if len(lat) else None,
            })

        # Q.5 — how many of a question's runs came back empty
        if df["run"].nunique() > 1:
            dist = e.groupby(df["question_id"]).sum().astype(int).value_counts()
            for k, v in sorted(dist.items()):
                stability.append({"config_id": cfg, "empty_runs_for_question": int(k),
                                  "n_questions": int(v)})

    pc = pd.DataFrame(per_config)
    outputs.append(_write("TQH_Q_per_config.csv", pc))
    outputs.append(_write("TQH_Q_per_run.csv", pd.DataFrame(per_run)))
    outputs.append(_write("TQH_Q5_stability.csv", pd.DataFrame(stability)))

    idx = pc.set_index("config_id")
    got["llama70b_empty_question_runs"] = idx.loc["ret_text2pandas@llama70b", "empty_question_runs"]
    got["gpt4o_empty_question_runs"] = idx.loc["ret_text2pandas@gpt4o-mini", "empty_question_runs"]
    got["intersection_empty_question_runs"] = idx.loc["ret_intersection", "empty_question_runs"]
    got["llama70b_hit_at_10"] = idx.loc["ret_text2pandas@llama70b", "hit_at_10_all"]
    got["gpt4o_hit_at_10"] = idx.loc["ret_text2pandas@gpt4o-mini", "hit_at_10_all"]
    got["llama70b_conditional_hit_at_10"] = idx.loc[
        "ret_text2pandas@llama70b", "hit_at_10_nonempty_result"]
    got["gpt4o_conditional_hit_at_10"] = idx.loc[
        "ret_text2pandas@gpt4o-mini", "hit_at_10_nonempty_result"]

    # ── Q.6 — does the empty-rate story carry into D1/D3? ──
    q6 = []
    for cfg in ("foodrag_h1_t2p", "foodrag_h1_dense"):
        p = ROOT / "eval_results" / f"{cfg}_per_question.parquet"
        if not p.exists():
            continue
        df = pd.read_parquet(p)
        df["_empty"] = df["retrieved_ids"].apply(_is_empty)
        df["_hit"] = df["metrics"].apply(lambda m: _metric(m, "hit_at_10"))
        gold_nz = df["answer_set_size"] > 0
        uncond = _per_question_mean(df, "hit_at_10")
        cond = _per_question_mean(df[~df["_empty"]], "hit_at_10")
        q6.append({
            "config_id": cfg, "runs": int(df["run"].nunique()),
            "empty_rate": round(float(df["_empty"].mean()), 6),
            "hit_at_10": uncond, "hit_at_10_nonempty_result": cond,
            # the published n: questions with at least one non-NaN run
            "n_questions_published": int(df["_hit"].groupby(
                df["question_id"]).apply(lambda s: s.notna().any()).sum()),
            "n_questions_total": int(df["question_id"].nunique()),
            "hit_at_10_nan_rows": int(df["_hit"].isna().sum()),
            "nan_on_nonempty_gold": int((df["_hit"].isna() & gold_nz).sum()),
            "pooled_mean": float(np.nanmean(df["_hit"])),
        })
    q6df = pd.DataFrame(q6)
    if len(q6df) == 2:
        a = q6df.set_index("config_id")
        q6df.attrs["ratio_unconditional"] = (a.loc["foodrag_h1_t2p", "hit_at_10"]
                                             / a.loc["foodrag_h1_dense", "hit_at_10"])
        got["h1_t2p_conditional_hit_at_10"] = a.loc[
            "foodrag_h1_t2p", "hit_at_10_nonempty_result"]
        got["h1_dense_conditional_hit_at_10"] = a.loc[
            "foodrag_h1_dense", "hit_at_10_nonempty_result"]
        q6df["ratio_vs_dense_unconditional"] = (
            q6df["hit_at_10"] / a.loc["foodrag_h1_dense", "hit_at_10"])
        q6df["ratio_vs_dense_conditional"] = (
            q6df["hit_at_10_nonempty_result"]
            / a.loc["foodrag_h1_dense", "hit_at_10_nonempty_result"])
    outputs.append(_write("TQH_Q6_d1_d3.csv", q6df))

    # ── H.3 — the B2 ratios, from the aggregates rather than the records ──
    def agg_value(d: str, cfg: str, metric: str = "hit_at_10") -> Optional[float]:
        p = ROOT / d / "aggregate.csv"
        if not p.exists():
            return None
        a = pd.read_csv(p)
        r = a[(a.config_id == cfg) & (a.stratum == "aggregate") & (a.metric == metric)]
        return float(r["mean"].iloc[0]) if len(r) else None

    dense = agg_value("eval_results", "ret_dense")
    rows = []
    for cfg, d in (("ret_text2pandas@llama70b", "eval_results_llama"),
                   ("ret_text2pandas@gpt4o-mini", "eval_results")):
        v = agg_value(d, cfg)
        if v is None or dense is None:
            continue
        rows.append({"numerator_config": cfg, "numerator_source": d,
                     "numerator_hit_at_10": v,
                     "denominator_config": "ret_dense",
                     "denominator_hit_at_10": dense, "ratio": v / dense})
    h = pd.DataFrame(rows)
    outputs.append(_write("TQH_H3_b2_ratios.csv", h))
    if len(h) == 2:
        got["B2_ratio_llama70b"] = float(h.iloc[0]["ratio"])
        got["B2_ratio_gpt4o"] = float(h.iloc[1]["ratio"])

    return {"outputs": outputs, "values": got}


# ── section: P1 safety-fix contamination ─────────────────────────────────────

P1_CONFIGS = ["foodrag_h1_t2p", "foodrag_h1_dense", "foodrag_h2", "foodrag_h2_t2p"]
FULLDAY_CONFIGS = ["foodrag_h1_t2p", "foodrag_h1_t2p_fullday",
                   "foodrag_h1_dense_fullday"]


def section_p1(_: argparse.Namespace) -> dict:
    """The Fix-1/Fix-2 split, pre-fix allergen values, and the re-judge deltas."""
    got: dict[str, float] = {}
    outputs: list[str] = []

    split, allergen, deltas = [], [], []
    fix1_total = fix2_total = stamped_total = 0

    for cfg in P1_CONFIGS:
        rel = f"eval_results/{cfg}_per_question.parquet"
        pre = read_parquet_at(PRE, rel)
        post = read_parquet_at(FIX_COMMIT, rel)
        if pre is None or post is None:
            return {"status": "skipped",
                    "reason": f"cannot read {rel} at {PRE}/{FIX_COMMIT}"}

        key = ["question_id", "run"]
        pre = pre.set_index(key).sort_index()
        post = post.set_index(key).sort_index()
        common = pre.index.intersection(post.index)
        pre, post = pre.loc[common], post.loc[common]

        av = pre["metrics"].apply(lambda m: _metric(m, "allergen_violation"))
        # Trigger predicates, apply_safety_fixes.py:277-281
        fix1 = av > 0
        fix2 = (pre["retrieved_ids"].apply(_is_empty)
                & pre.get("explanation_text", pd.Series("", index=pre.index))
                     .fillna("").astype(str).str.len().gt(0))
        stamped = (post["_safety_fixed"].notna()
                   if "_safety_fixed" in post.columns
                   else pd.Series(False, index=post.index))

        fix1_total += int(fix1.sum())
        fix2_total += int(fix2.sum())
        stamped_total += int(stamped.sum())
        split.append({"config_id": cfg, "records": len(common),
                      "fix1_allergen": int(fix1.sum()),
                      "fix2_explanation": int(fix2.sum()),
                      "stamped": int(stamped.sum())})

        post_av = post["metrics"].apply(lambda m: _metric(m, "allergen_violation"))
        vals = av[av > 0].round(6).value_counts().sort_index()
        allergen.append({
            "config_id": cfg,
            "prefix_mean": float(np.nanmean(av)),
            "postfix_mean": float(np.nanmean(post_av)),
            "records_gt_zero": int((av > 0).sum()),
            "value_distribution": "; ".join(f"{int(c)} x {v}" for v, c in vals.items()),
            "max_value": float(av.max()),
            "records_at_1.0": int((av == 1.0).sum()),
        })

        # Re-judge deltas. Two record classes, because the two reports quote
        # different ones and conflating them loses the point of each:
        #   "all"  — every record of the config. P1 §4's table: config-wide means
        #            with a changed count (correctness_judgeA 388 = 385 Fix-2
        #            records plus the 3 Fix-1 records, which were re-judged too).
        #   "fix2" — the 421 empty-plan records only. D2.5's table, where the
        #            means are the ones that make the explanation-sensitivity
        #            point (0.045 -> 0.943).
        keys = sorted({k for m in pre["metrics"] if isinstance(m, dict) for k in m})
        for label, mask in (("all", pd.Series(True, index=pre.index)),
                            ("fix2", fix2)):
            if not mask.any():
                continue
            for metric in keys:
                a = pre.loc[mask, "metrics"].apply(lambda m: _metric(m, metric))
                b = post.loc[mask, "metrics"].apply(lambda m: _metric(m, metric))
                changed = int((~np.isclose(a, b, equal_nan=True)).sum())
                if not changed:
                    continue
                deltas.append({
                    "config_id": cfg, "record_class": label, "metric": metric,
                    "n_records": int(mask.sum()), "n_changed": changed,
                    "prefix_mean": _safe_nanmean(a),
                    "postfix_mean": _safe_nanmean(b),
                    "delta": (None if _safe_nanmean(a) is None or _safe_nanmean(b) is None
                              else _safe_nanmean(b) - _safe_nanmean(a)),
                    "postfix_all_nan": int(b.isna().all()),
                })

    outputs.append(_write("P1_fix_split.csv", pd.DataFrame(split)))
    outputs.append(_write("P1_allergen_prefix.csv", pd.DataFrame(allergen)))
    outputs.append(_write("P1_rejudge_deltas.csv",
                          pd.DataFrame(deltas).sort_values(
                              ["config_id", "metric"]) if deltas else pd.DataFrame()))

    got |= {"fix1_total": fix1_total, "fix2_total": fix2_total,
            "stamped_total": stamped_total}
    for r in allergen:
        got[f"{r['config_id'].replace('foodrag_', '')}_prefix_allergen"] = r["prefix_mean"]
    # P1 §4's re-judge table, config-wide on h1_t2p
    for d in deltas:
        if d["config_id"] == "foodrag_h1_t2p" and d["record_class"] == "all":
            got[f"h1_t2p_all_{d['metric']}_changed"] = d["n_changed"]
            if d["prefix_mean"] is not None:
                got[f"h1_t2p_all_{d['metric']}_prefix"] = d["prefix_mean"]

    # ── §5 — empty plans still carrying a hallucinated explanation ──
    hall = []
    total_h = 0
    for cfg in FULLDAY_CONFIGS:
        p = ROOT / "eval_results" / f"{cfg}_per_question.parquet"
        if not p.exists():
            p = p.with_suffix(".pkl")
            if not p.exists():
                continue
        df = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_pickle(p)
        if "explanation_text" not in df.columns:
            continue
        txt = df["explanation_text"].fillna("").astype(str).str.strip()
        empty_plan = df["retrieved_ids"].apply(_is_empty) & txt.str.len().gt(0)
        canned = txt.eq(EMPTY_PLAN_MESSAGE)
        n_h = int((empty_plan & ~canned).sum())
        total_h += n_h if cfg != "foodrag_h1_t2p" else 0
        hall.append({"config_id": cfg,
                     "empty_plan_with_explanation": int(empty_plan.sum()),
                     "is_fixed_message": int((empty_plan & canned).sum()),
                     "still_hallucinated": n_h})
    outputs.append(_write("P1_fullday_hallucinations.csv", pd.DataFrame(hall)))
    got["fullday_hallucinated_total"] = total_h

    return {"outputs": outputs, "values": got}


# ── section: D2 agentic value ────────────────────────────────────────────────

def _install_chunked_bootstrap(stats: Any) -> None:
    """Draw the paired bootstrap in row blocks instead of one (n x m) array.

    ``paired_bootstrap`` allocates ``rng.integers(0, m, size=(n, m))`` in one go
    — 125 MB per metric-pair at n = 10,000, m = 1,643 — which fails outright on
    this box (``MemoryError``), exactly as the D2 report's environment note
    predicts. ``Generator.integers`` fills row-major, so k blocks of ``(rows, m)``
    consume the same stream in the same order as one ``(n, m)`` draw: the
    resamples, and therefore the CI and p-value, are **bit-identical**. Same trick
    ``build_claim_registry.py:118-148`` uses on ``runner._bootstrap_ci``.
    """
    original = stats.paired_bootstrap

    def chunked(a, b, n: int = 10000, seed: int = 0, alpha: float = 0.05,
                block: int = 500):
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        d = a - b
        m = d.size
        if m == 0:
            return stats.BootstrapResult(float("nan"), float("nan"), float("nan"), 1.0)
        obs = float(np.mean(d))
        if np.allclose(d, 0.0, atol=1e-12):
            return stats.BootstrapResult(0.0, 0.0, 0.0, 1.0)
        rng = np.random.default_rng(seed)
        means = np.empty(n, dtype=float)
        done = 0
        while done < n:
            rows = min(block, n - done)
            while True:
                try:
                    idx = rng.integers(0, m, size=(rows, m))
                    break
                except (MemoryError, ValueError):
                    if rows == 1:
                        raise
                    rows = max(1, rows // 4)
            means[done:done + rows] = d[idx].mean(axis=1)
            done += rows
        lo = float(np.percentile(means, 100 * (alpha / 2)))
        hi = float(np.percentile(means, 100 * (1 - alpha / 2)))
        tail = (float(np.mean(means <= 0.0)) if obs >= 0
                else float(np.mean(means >= 0.0)))
        return stats.BootstrapResult(obs, lo, hi, min(1.0, max(1.0 / n, 2.0 * tail)))

    chunked.__wrapped__ = original
    stats.paired_bootstrap = chunked

def section_d2(args: argparse.Namespace) -> dict:
    """Pre-fix vs published pairwise, and D2.5's explanation-sensitivity table."""
    from foodrag.analysis import stats

    _install_chunked_bootstrap(stats)

    got: dict[str, float] = {}
    outputs: list[str] = []
    needed = sorted({c for pair in stats.CONFIG_PAIRS for c in pair})

    def records_at(rev: str) -> Optional[dict[str, list[dict]]]:
        out: dict[str, list[dict]] = {}
        for cfg in needed:
            for ext in ("parquet", "pkl"):
                df = (read_parquet_at(rev, f"eval_results/{cfg}_per_question.{ext}")
                      if ext == "parquet" else None)
                if df is not None:
                    out[cfg] = df.to_dict("records")
                    break
            else:
                return None
            if cfg not in out:
                return None
        return out

    rows = []
    for label, rev in (("prefix", PRE), ("published", FIX_COMMIT)):
        recs = records_at(rev)
        if recs is None:
            return {"status": "skipped", "reason": f"per-question records absent at {rev}"}
        for r in stats.compute_pairwise(recs, bootstrap_n=args.bootstrap, seed=0):
            rows.append({"variant": label, **r})
    pw = pd.DataFrame(rows)
    outputs.append(_write("D2_pairwise_prefix_vs_published.csv", pw))

    sel = pw[(pw.metric == "correctness_judgeA")
             & (pw.config_a == "foodrag_h1_t2p") & (pw.config_b == "foodrag_h2_t2p")]
    for _, r in sel.iterrows():
        if r["variant"] == "prefix":
            got["strong_correctness_prefix_a"] = float(r["mean_a"])
            got["strong_correctness_prefix_b"] = float(r["mean_b"])
            # These two exercise the chunked bootstrap: if the block draw
            # diverged from the stock one-shot draw, the CI moves and this fails.
            got["strong_correctness_prefix_ci_low"] = float(r["bootstrap_ci_low"])
            got["strong_correctness_prefix_ci_high"] = float(r["bootstrap_ci_high"])
            got["strong_correctness_prefix_median_diff"] = float(r["median_diff"])
        else:
            got["strong_correctness_published_a"] = float(r["mean_a"])
            got["strong_correctness_published_b"] = float(r["mean_b"])
            got["strong_correctness_published_median_diff"] = float(r["median_diff"])

    # ── D2.5 — the Fix-2 record class: retrieval unchanged, judges move ──
    rel = "eval_results/foodrag_h1_t2p_per_question.parquet"
    pre, post = read_parquet_at(PRE, rel), read_parquet_at(FIX_COMMIT, rel)
    if pre is None or post is None:
        return {"outputs": outputs, "values": got,
                "note": "D2.5 skipped: h1_t2p records absent in history"}

    key = ["question_id", "run"]
    pre, post = pre.set_index(key).sort_index(), post.set_index(key).sort_index()
    common = pre.index.intersection(post.index)
    pre, post = pre.loc[common], post.loc[common]
    fix2 = (pre["retrieved_ids"].apply(_is_empty)
            & pre.get("explanation_text", pd.Series("", index=pre.index))
                 .fillna("").astype(str).str.len().gt(0))
    got["fix2_records"] = int(fix2.sum())

    def ids(s: Any) -> tuple:
        return tuple() if s is None else tuple(s)

    same_ids = int((pre.loc[fix2, "retrieved_ids"].apply(ids)
                    == post.loc[fix2, "retrieved_ids"].apply(ids)).sum())
    both_empty = int((pre.loc[fix2, "retrieved_ids"].apply(_is_empty)
                      & post.loc[fix2, "retrieved_ids"].apply(_is_empty)).sum())

    d25 = [{"metric": "_retrieved_ids_identical", "n_records": int(fix2.sum()),
            "n_changed": int(fix2.sum()) - same_ids,
            "prefix_mean": None, "postfix_mean": None, "delta": None,
            "note": f"{both_empty} empty on both sides"}]
    for metric in ("hit_at_10", "correctness_judgeA", "faithfulness_judgeA",
                   "relevancy_judgeA", "correctness_judgeB"):
        a = pre.loc[fix2, "metrics"].apply(lambda m: _metric(m, metric))
        b = post.loc[fix2, "metrics"].apply(lambda m: _metric(m, metric))
        d25.append({"metric": metric, "n_records": int(fix2.sum()),
                    "n_changed": int((~np.isclose(a, b, equal_nan=True)).sum()),
                    "prefix_mean": float(np.nanmean(a)),
                    "postfix_mean": float(np.nanmean(b)),
                    "delta": float(np.nanmean(b) - np.nanmean(a)), "note": ""})
    d = pd.DataFrame(d25)
    outputs.append(_write("D2_5_explanation_sensitivity.csv", d))
    for m, k in (("correctness_judgeA", "correctness_judgeA_changed"),
                 ("correctness_judgeB", "correctness_judgeB_changed")):
        got[k] = int(d[d.metric == m]["n_changed"].iloc[0])

    return {"outputs": outputs, "values": got}


# ── section: E-3 / E-4 re-aggregation ────────────────────────────────────────

def section_e34(_: argparse.Namespace) -> dict:
    """Delegates to the existing tool, which already knows how to do this."""
    csv = OUT / "E3_E4_reaggregated.csv"
    OUT.mkdir(parents=True, exist_ok=True)
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "reaggregate_corrected.py"),
                        "--root", str(ROOT), "--csv", str(csv)],
                       cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        return {"status": "skipped", "reason": r.stderr.strip().splitlines()[-1:] or "failed"}
    if not csv.exists():
        return {"status": "skipped", "reason": "tool produced no CSV"}
    d = pd.read_csv(csv)
    print(f"    -> audit/{csv.name}  ({len(d)} rows)")

    got: dict[str, float] = {"rows_compared": len(d),
                             "rows_changed": int((d["delta"].abs() > 1e-9).sum()),
                             "coerced_nans_total": int(d.groupby("config_id")
                                                       ["coerced_nans"].max().sum())}

    def pick(cfg: str, metric: str, col: str) -> Optional[float]:
        row = d[(d.config_id == cfg) & (d.metric == metric)]
        return float(row.iloc[0][col]) if len(row) else None

    for label, (cfg, metric, col) in {
        # E-3 — ToG's denominator was inflated 3x and its variance was an artefact
        "tog_hit_n_published": ("tog", "hit_at_10", "n_pub"),
        "tog_hit_n_corrected": ("tog", "hit_at_10", "n_harm"),
        "tog_hit_var_published": ("tog", "hit_at_10", "var_pub"),
        "tog_hit_var_corrected": ("tog", "hit_at_10", "var_harm"),
        # E-4 — the harmonisation deltas on D1's headline
        "h1_t2p_hit_published": ("foodrag_h1_t2p", "hit_at_10", "published"),
        "h1_t2p_hit_harmonised": ("foodrag_h1_t2p", "hit_at_10", "harmonised"),
        "h1_t2p_fullday_hit_harmonised": ("foodrag_h1_t2p_fullday", "hit_at_10",
                                          "harmonised"),
        "h1_dense_hit_harmonised": ("foodrag_h1_dense", "hit_at_10", "harmonised"),
    }.items():
        v = pick(cfg, metric, col)
        if v is not None:
            got[label] = v
    if {"h1_t2p_hit_harmonised", "h1_dense_hit_harmonised"} <= got.keys():
        got["D3_ratio_harmonised"] = (got["h1_t2p_hit_harmonised"]
                                      / got["h1_dense_hit_harmonised"])
    return {"outputs": [csv.name], "values": got}


# ── section: A1 corpus size ──────────────────────────────────────────────────

def section_a1(_: argparse.Namespace) -> dict:
    """§5.4's retriever-coverage table, plus the population cardinalities."""
    got: dict[str, float] = {}
    outputs: list[str] = []

    recipes = ROOT / "data" / "hummus" / "recipes_df.csv"
    corpus_ids: Optional[set] = None
    pops = []
    if recipes.exists():
        urls: set[str] = set()
        for chunk in pd.read_csv(recipes, usecols=["recipe_url"], chunksize=100_000):
            urls.update(chunk["recipe_url"].astype(str))
        corpus_ids = urls
        pops.append({"population": "R_file (recipes_df.csv)", "cardinality": len(urls),
                     "evidence": "data/hummus/recipes_df.csv unique recipe_url"})
    raw = ROOT / "data" / "hummus" / "hummus_data" / "raw" / "recipes.csv"
    if raw.exists():
        u: set[str] = set()
        for chunk in pd.read_csv(raw, usecols=["recipe_url"], chunksize=100_000):
            u.update(chunk["recipe_url"].astype(str))
        pops.append({"population": "R_HUMMUS (raw release)", "cardinality": len(u),
                     "evidence": "data/hummus/hummus_data/raw/recipes.csv unique recipe_url"})
    node = ROOT / "data" / "neo4j_kg" / "all" / "node_Recipe.csv"
    if node.exists():
        n = sum(1 for _ in node.open(encoding="utf-8", errors="ignore")) - 1
        pops.append({"population": "Neo4j Recipe nodes", "cardinality": n,
                     "evidence": "data/neo4j_kg/all/node_Recipe.csv rows"})
    if pops:
        outputs.append(_write("A1_populations.csv", pd.DataFrame(pops)))

    # §5.4 — every id any retriever returned, against the corpus
    cov = []
    outside_total = 0
    for cfg, d in RET_SOURCES.items():
        p = ROOT / d / f"{cfg}_per_question.parquet"
        if not p.exists():
            continue
        df = pd.read_parquet(p)
        seen: set = set()
        for a in df["retrieved_ids"]:
            if a is not None:
                seen.update(a)
        outside = (len(seen - corpus_ids) if corpus_ids is not None else None)
        outside_total += outside or 0
        got[f"ids_{cfg}"] = len(seen)
        cov.append({"config_id": cfg, "question_runs": len(df),
                    "empty_results": int(df["retrieved_ids"].apply(_is_empty).sum()),
                    "distinct_ids_returned": len(seen),
                    "outside_corpus": outside})
    outputs.append(_write("A1_retriever_coverage.csv", pd.DataFrame(cov)))
    if corpus_ids is not None:
        got["ret_configs_outside_corpus"] = outside_total
        got["corpus_rows"] = len(corpus_ids)

    return {"outputs": outputs, "values": got}


# ── section: T tag provenance ────────────────────────────────────────────────

HEURISTIC_LABELS = ("breakfast", "lunch", "dinner", "snack")


def _as_list(v: Any) -> list[str]:
    """Parse a ``tags`` / ``meal_types`` cell.

    Both columns are stored as Python reprs (``"['low-carb', 'shrimp']"``), so
    ``ast.literal_eval`` is the correct reader. Quote-swapping into ``json.loads``
    silently mangles any tag containing an apostrophe — ``shepherd's-pie`` and
    friends — which would undercount the very tags T.6 is about.
    """
    if isinstance(v, list):
        return [str(x) for x in v]
    if not isinstance(v, str) or not v.strip():
        return []
    try:
        parsed = ast.literal_eval(v)
    except (ValueError, SyntaxError):
        return [t.strip(" '\"") for t in v.strip("[]").split(",") if t.strip(" '\"")]
    return [str(x) for x in parsed] if isinstance(parsed, (list, tuple)) else []


_FOODCOM_ID = re.compile(r"-(\d+)/?$")


def _native_meal_tags(raw: Path) -> dict[str, set[str]]:
    """``{food.com id: which of the four labels food.com itself carries}``.

    T.6's split needs the *correct* join — the food.com id embedded in
    ``recipe_url`` — not ``hummus_util.py:642``'s broken ``recipe_id``-to-``id``
    join, which matches 145,805 rows and is wrong on every one of them.
    """
    out: dict[str, set[str]] = {}
    for chunk in pd.read_csv(raw, usecols=["id", "tags"], chunksize=100_000):
        for rid, tags in zip(chunk["id"].astype(str), chunk["tags"]):
            present = {t for t in _as_list(tags) if t in HEURISTIC_LABELS}
            if present:
                out[rid] = present
    return out


def section_t(_: argparse.Namespace) -> dict:
    """T.2/T.6 — which column holds what, and how the heuristic labels got there."""
    recipes = ROOT / "data" / "hummus" / "recipes_df.csv"
    if not recipes.exists():
        return {"status": "skipped", "reason": "data/hummus/recipes_df.csv absent"}

    got: dict[str, float] = {}
    outputs: list[str] = []

    raw = ROOT / "data" / "hummus" / "hummus_data" / "additional" / "RAW_recipes.csv"
    native = _native_meal_tags(raw) if raw.exists() else None

    rows = 0
    tags_ne = mt_ne = 0
    per_label = {t: 0 for t in HEURISTIC_LABELS}
    native_label = {t: 0 for t in HEURISTIC_LABELS}
    appended_recipes: set[str] = set()
    field_fired: dict[str, int] = {}
    pattern_hits: dict[str, int] = {}
    # ⚠ RECONSTRUCTED, NOT REPRODUCED. T.6 quotes shares for these pattern
    # classes but does not publish the regexes that produced them, and the
    # original analysis script is gone (it lived in a session scratchpad). These
    # are re-derivations under the patterns written here; they are counted per
    # appended tag *instance*, matched against that instance's own label, which
    # is the unit T.6's denominator (20,667) uses. They do not land on T.6's
    # numbers and are recorded as an independent measurement, not a check.
    def _pat(body: str) -> Callable[[str, str], bool]:
        return lambda text, label: bool(
            re.search(body.replace("<meal>", re.escape(label)), text, re.I))

    PATTERNS: dict[str, Callable[[str, str], bool]] = {
        "for (a) <meal>": _pat(r"\bfor\s+(?:a\s+|an\s+|the\s+)?<meal>\b"),
        "<meal> leftovers/party/guests/table":
            _pat(r"\b<meal>\s+(leftovers|party|parties|guests|table)\b"),
        "serve(d/s) ... <meal>": _pat(r"\bserve[ds]?\b[^.]{0,40}\b<meal>\b"),
        "dinner party": lambda text, label: (
            label == "dinner" and bool(re.search(r"\bdinner\s+part(y|ies)\b", text, re.I))),
        "great/perfect/ideal for <meal>":
            _pat(r"\b(great|perfect|ideal)\s+for\s+(?:a\s+|an\s+)?<meal>\b"),
        "kraft dinner": lambda text, label: (
            label == "dinner" and bool(re.search(r"\bkraft\s+dinner\b", text, re.I))),
    }

    # The three fields _append_additional_tags matches over (hummus_util.py:673-675)
    # plus the two tag columns. Column names are the shipped ones, not guesses.
    usecols = ["title", "description", "directions", "tags", "meal_types",
               "recipe_url"]
    head = pd.read_csv(recipes, nrows=1)
    missing = [c for c in usecols if c not in head.columns]
    if missing:
        return {"status": "skipped",
                "reason": f"recipes_df.csv lacks {missing}; schema changed"}
    for chunk in pd.read_csv(recipes, usecols=usecols, chunksize=50_000):
        rows += len(chunk)
        if "tags" in chunk:
            t = chunk["tags"].apply(_as_list)
            tags_ne += int(t.apply(bool).sum())
        else:
            t = pd.Series([[]] * len(chunk), index=chunk.index)
        if "meal_types" in chunk:
            mt_ne += int(chunk["meal_types"].apply(lambda v: bool(_as_list(v))).sum())

        title = chunk["title"].fillna("").astype(str)
        desc = chunk["description"].fillna("").astype(str)
        steps = chunk["directions"].fillna("").astype(str)
        url = chunk["recipe_url"].fillna("").astype(str)
        for i in chunk.index:
            present = [x for x in t.loc[i] if x in HEURISTIC_LABELS]
            if not present:
                continue
            m = _FOODCOM_ID.search(url.loc[i])
            own_native = native.get(m.group(1), set()) if (native and m) else set()
            blob = f"{title.loc[i]} {desc.loc[i]} {steps.loc[i]}"
            appended_here = False
            for label in present:
                if label in own_native:
                    native_label[label] += 1
                    continue                      # food.com's own tag, not appended
                per_label[label] += 1
                appended_here = True
                fields = [n for n, s in (("title", title.loc[i]),
                                         ("description", desc.loc[i]),
                                         ("directions", steps.loc[i]))
                          if label in s.lower()]
                key = "+".join(fields) or "none"
                field_fired[key] = field_fired.get(key, 0) + 1
                # per appended instance, against that instance's own label
                for name, hit in PATTERNS.items():
                    if hit(blob, label):
                        pattern_hits[name] = pattern_hits.get(name, 0) + 1
                if not fields or fields == ["description"] or "title" not in fields:
                    if any(hit(blob, label) for hit in PATTERNS.values()):
                        pattern_hits["_no_title_support_and_incidental"] = \
                            pattern_hits.get("_no_title_support_and_incidental", 0) + 1
            if not appended_here:
                continue
            appended_recipes.add(url.loc[i])

    total_labels = sum(per_label.values())
    outputs.append(_write("T_column_census.csv", pd.DataFrame([
        {"column": "tags", "rows": rows, "non_empty": tags_ne,
         "share": round(tags_ne / rows, 6) if rows else None,
         "writer": "hummus_util.py:664-678 (substring heuristic, appended)"},
        {"column": "meal_types", "rows": rows, "non_empty": mt_ne,
         "share": round(mt_ne / rows, 6) if rows else None,
         "writer": "pipeline/annotate.py:158-161 (LLM classification)"},
    ])))
    outputs.append(_write("T_heuristic_labels.csv", pd.DataFrame(
        [{"label": k, "heuristic_appended": v,
          "native_food_com": native_label[k] if native else None}
         for k, v in per_label.items()])))
    got["appended_tag_instances"] = total_labels
    got["appended_recipes"] = len(appended_recipes)
    for k, v in per_label.items():
        got[f"appended_{k}"] = v
    if native:
        for k, v in native_label.items():
            got[f"native_{k}"] = v
    outputs.append(_write("T_field_support.csv", pd.DataFrame(
        [{"fields_matched": k, "instances": v,
          "share": round(v / total_labels, 6) if total_labels else None}
         for k, v in sorted(field_fired.items(), key=lambda kv: -kv[1])])))
    outputs.append(_write("T_incidental_patterns.csv", pd.DataFrame(
        [{"pattern": k, "appended_instances": v,
          "share_of_appended": round(v / total_labels, 6) if total_labels else None,
          "reproduces_report": "no — regexes not published, see note"}
         for k, v in sorted(pattern_hits.items(), key=lambda kv: -kv[1])])))

    got |= {"corpus_rows": rows, "tags_non_empty": tags_ne,
            "meal_types_non_empty": mt_ne}
    return {"outputs": outputs, "values": got,
            "note": ("T_incidental_patterns.csv is a re-derivation, not a "
                     "reproduction: T.6 does not publish the regexes behind its "
                     "pattern shares and the original script is gone. Every other "
                     "T output reproduces the report exactly.")}


# ── driver ───────────────────────────────────────────────────────────────────

SECTIONS: dict[str, tuple[str, Callable[[argparse.Namespace], dict]]] = {
    "qh": ("checks-T-Q-H-findings.md — Q.1-Q.6, H.3", section_qh),
    "t": ("checks-T-Q-H-findings.md — T.2, T.6 (needs data/)", section_t),
    "a1": ("A1-corpus-size-findings.md — populations, §5.4 coverage (needs data/)", section_a1),
    "p1": ("P1-safety-fix-contamination-findings.md (needs git history)", section_p1),
    "d2": ("D2-agentic-value-findings.md (needs git history)", section_d2),
    "e34": ("E3-E4-P3-P4-findings.md — E-3/E-4 re-aggregation", section_e34),
}


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--only", help="comma-separated section names")
    p.add_argument("--list", action="store_true", help="list sections and exit")
    p.add_argument("--bootstrap", type=int, default=10000,
                   help="pairwise bootstrap resamples (D2); the published run used 10000")
    args = p.parse_args(argv)

    if args.list:
        for name, (desc, _) in SECTIONS.items():
            print(f"  {name:5s} {desc}")
        return 0

    names = ([s.strip() for s in args.only.split(",")] if args.only
             else list(SECTIONS))
    unknown = [n for n in names if n not in SECTIONS]
    if unknown:
        p.error(f"unknown section(s): {unknown}; choose from {list(SECTIONS)}")

    print("# Backfill — machine-readable backing for specs/*-findings.md\n")
    manifest: dict[str, Any] = {"sections": {}}
    mismatches = 0

    for name in names:
        desc, fn = SECTIONS[name]
        print(f"## {name} — {desc}")
        try:
            res = fn(args)
        except Exception as exc:                      # noqa: BLE001 — reported, not raised
            print(f"    !! failed: {type(exc).__name__}: {exc}")
            manifest["sections"][name] = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
            mismatches += 1
            continue

        if res.get("status") == "skipped":
            print(f"    -- skipped: {res.get('reason')}")
            manifest["sections"][name] = res
            continue

        checks = []
        for label, (want, tol) in EXPECTED.get(name, {}).items():
            if label not in res.get("values", {}):
                continue
            have = float(res["values"][label])
            ok = (have == want) if tol == 0 else abs(have - want) <= tol
            checks.append({"value": label, "expected": want, "derived": have,
                           "status": "match" if ok else "MISMATCH"})
            if not ok:
                mismatches += 1
                print(f"    !! MISMATCH {label}: report says {want}, derived {have}")
        if checks and all(c["status"] == "match" for c in checks):
            print(f"    ok: {len(checks)}/{len(checks)} headline values reproduce the report")
        manifest["sections"][name] = {"status": "ok", "outputs": res.get("outputs", []),
                                      "checks": checks, "values": res.get("values", {})}
        if res.get("note"):
            manifest["sections"][name]["note"] = res["note"]

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "backfill_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    print(f"\nmanifest -> audit/backfill_manifest.json")
    print("MISMATCHES PRESENT" if mismatches else "all reproduced values match their report")
    return 1 if mismatches else 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Data layer for the Report tab (Flow C).

Loads the *real* pre-computed evaluation outputs (``eval_results/`` — aggregate.csv,
pairwise.csv, per-question parquets) and derives the small frames the new
result-driven panels need (error taxonomy, cross-judge agreement, time-of-day
meal selection, volatility, illustrative cases). Read-only: nothing here re-runs
the pipeline; the parquets are the source of truth (spec
``FoodRAG_Report_Tab_Modifications`` §2).

All loaders are memoised (``lru_cache``) so the parquets are read/expanded once
per process, not once per Streamlit rerun. Every function degrades gracefully
(returns ``None``/empty) when a file is absent, so the report renders on a
machine that only ships a subset of ``eval_results/``.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .. import eval_results

# ── copy / labels ──────────────────────────────────────────────────────────
_COPY_PATH = Path(__file__).resolve().parents[1] / "report_copy.json"


@lru_cache(maxsize=1)
def copy() -> dict:
    """Editable Report-tab copy (titles, descriptions, label maps, marquee)."""
    try:
        return json.loads(_COPY_PATH.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}


def config_label(cid: str) -> str:
    return copy().get("config_labels", {}).get(cid, cid)


def metric_label(m: str) -> str:
    return copy().get("metric_labels", {}).get(m, m)


def exp_meta(exp: str) -> dict:
    """Descriptive {title, sub, desc} for an experiment id (falls back to id)."""
    return copy().get("experiments", {}).get(exp, {"title": exp, "sub": exp, "desc": ""})


def ceiling_note(cid: str) -> Optional[str]:
    """Reference-ceiling / cost caveat for a config, if any (spec §7)."""
    return copy().get("reference_ceilings", {}).get(cid)


# ── aggregate / pairwise ───────────────────────────────────────────────────
def load_aggregate() -> Optional[pd.DataFrame]:
    return eval_results.load_aggregate()


def load_pairwise() -> Optional[pd.DataFrame]:
    return eval_results.load_pairwise()


def cell(agg: pd.DataFrame, experiment: str, config: str, metric: str,
         stratum: str = "aggregate") -> Optional[float]:
    """Single ``mean`` value for one (experiment, config, metric, stratum) cell."""
    if agg is None:
        return None
    sub = agg[(agg["experiment_id"] == experiment) & (agg["config_id"] == config)
              & (agg["metric"] == metric) & (agg["stratum"] == stratum)]
    return float(sub["mean"].iloc[0]) if len(sub) else None


def marquee_values(agg: pd.DataFrame) -> list[dict]:
    """Resolve the marquee spec (report_copy.json) against the real aggregate.

    Returns the copy dict for each card with a live ``value`` (or ``None`` if the
    cell is missing) so the headline cards never drift from the data."""
    out = []
    for card in copy().get("marquee", []):
        v = cell(agg, card["experiment"], card["config"], card["metric"],
                 card.get("stratum", "aggregate"))
        out.append({**card, "value": v})
    return out


# ── per-question parquet access ────────────────────────────────────────────
# The four FoodRAG system configs carry per-question plan + judge fields; the
# baselines/retrievers carry retrieval metrics only.
SYS_CONFIGS = {
    "foodrag_h1_t2p": "foodrag_h1_t2p_per_question.parquet",
    "foodrag_h1_dense": "foodrag_h1_dense_per_question.parquet",
    "foodrag_h2_t2p": "foodrag_h2_t2p_per_question.parquet",
    "foodrag_h2": "foodrag_h2_per_question.parquet",
}


@lru_cache(maxsize=16)
def _per_question(fname: str) -> Optional[pd.DataFrame]:
    """Load one per-question parquet with the ``metrics`` dict expanded to columns."""
    path = eval_results.results_dir() / fname
    if not path.exists() or path.stat().st_size == 0:
        return None
    try:
        df = pd.read_parquet(path)
    except Exception:  # noqa: BLE001
        return None
    if "metrics" in df.columns:
        m = pd.json_normalize(df["metrics"]).reset_index(drop=True)
        df = pd.concat([df.drop(columns=["metrics"]).reset_index(drop=True), m], axis=1)
    return df


def _plan_empty(plan) -> bool:
    """True if a stored plan has no filled meal slot."""
    if isinstance(plan, dict):
        return not any(v for v in plan.values())
    return True


# ── derived frame: error taxonomy (spec §4 Tier 2, NEW) ────────────────────
@lru_cache(maxsize=1)
def error_taxonomy() -> Optional[pd.DataFrame]:
    """Per-config failure-rate table over the four FoodRAG system configs.

    Buckets (each a fraction of per-question records, higher = worse):
      retrieval_miss     hit@10 == 0 (gold never retrieved)
      empty_plan         planner returned no filled slot (honest no-recommendation)
      hard_violation     any hard-constraint violation
      low_correctness    correctness (Judge A) < 0.5
      unfaithful         faithfulness (Judge A) < 0.5
      structural_invalid full-day structural validity == 0  (ARTIFACT — partial-day, see §5)
    """
    rows = []
    for cid, fname in SYS_CONFIGS.items():
        df = _per_question(fname)
        if df is None or df.empty:
            continue
        n = len(df)

        def frac(mask) -> float:
            return float(np.mean(mask)) if n else float("nan")

        def col(name):
            return df[name] if name in df.columns else pd.Series([np.nan] * n)

        rec = {
            "config_id": cid,
            "n": n,
            "retrieval_miss": frac(col("hit_at_10").fillna(0) == 0),
            "empty_plan": frac(df["plan"].map(_plan_empty)) if "plan" in df.columns else float("nan"),
            "hard_violation": frac(col("hard_constraint_violation").fillna(0) > 0),
            "low_correctness": frac(col("correctness_judgeA") < 0.5),
            "unfaithful": frac(col("faithfulness_judgeA") < 0.5),
            "structural_invalid": frac(col("structural_validity").fillna(0) == 0),
        }
        rows.append(rec)
    if not rows:
        return None
    return pd.DataFrame(rows).set_index("config_id")


# ── derived frame: cross-judge agreement (spec §4 Tier 3) ──────────────────
_JUDGE_METRICS = ["correctness", "faithfulness", "relevancy", "explanation_quality"]


@lru_cache(maxsize=1)
def judge_agreement() -> Optional[pd.DataFrame]:
    """Cross-judge (Gemma-3-27B vs DeepSeek-V3) agreement per config × metric.

    Returns long rows: config_id, metric, pearson, mean_a, mean_b, mean_abs_diff, n.
    Pearson is NaN when a judge column is constant (undefined correlation)."""
    rows = []
    for cid, fname in SYS_CONFIGS.items():
        df = _per_question(fname)
        if df is None or df.empty:
            continue
        for met in _JUDGE_METRICS:
            ca, cb = f"{met}_judgeA", f"{met}_judgeB"
            if ca not in df.columns or cb not in df.columns:
                continue
            pair = df[[ca, cb]].dropna()
            if len(pair) < 3:
                continue
            a, b = pair[ca].to_numpy(float), pair[cb].to_numpy(float)
            pear = float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else float("nan")
            rows.append({
                "config_id": cid, "metric": met, "pearson": pear,
                "mean_a": float(a.mean()), "mean_b": float(b.mean()),
                "mean_abs_diff": float(np.mean(np.abs(a - b))), "n": len(pair),
            })
    if not rows:
        return None
    return pd.DataFrame(rows)


# ── derived frame: time-of-day meal selection (spec §4 Tier 3, NEW) ────────
@lru_cache(maxsize=1)
def time_of_day() -> Optional[pd.DataFrame]:
    """Per-hour meal-type selection (eval_results/mealplan_time_of_day.csv)."""
    path = eval_results.results_dir() / "mealplan_time_of_day.csv"
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:  # noqa: BLE001
        return None


# ── derived frame: volatility (per-question variance) ──────────────────────
def volatility(agg: pd.DataFrame, experiment: str = "EXP-7",
               stratum: str = "aggregate") -> Optional[pd.DataFrame]:
    """Per-config × metric per-question variance for an experiment (from aggregate)."""
    if agg is None:
        return None
    sub = agg[(agg["experiment_id"] == experiment) & (agg["stratum"] == stratum)
              & (agg["per_question_variance"].fillna(0) > 0)]
    if sub.empty:
        return None
    return sub[["config_id", "metric", "per_question_variance", "n_questions"]].copy()


# ── derived: illustrative cases (spec §4 Tier 3, Case browser) ─────────────
def _excerpt(text, n: int = 240) -> str:
    s = "" if text is None else str(text)
    return (s[:n] + "…") if len(s) > n else s


@lru_cache(maxsize=1)
def cases() -> list[dict]:
    """A few illustrative per-question cases from the H1-t2p run.

    Best-effort: allergen-fixed, judge-disagreement, honest-empty-plan and a clean
    success. Each dict carries the plan, an explanation excerpt and key metrics so
    the UI can render a card without re-reading parquet."""
    df = _per_question(SYS_CONFIGS["foodrag_h1_t2p"])
    if df is None or df.empty:
        return []
    out: list[dict] = []

    def pack(row, kind, why):
        return {
            "kind": kind, "why": why,
            "question_id": str(row.get("question_id", "")),
            "plan": {k: v for k, v in (row.get("plan") or {}).items() if v}
                    if isinstance(row.get("plan"), dict) else {},
            "explanation": _excerpt(row.get("explanation_text")),
            "hit_at_10": row.get("hit_at_10"),
            "correctness_judgeA": row.get("correctness_judgeA"),
            "correctness_judgeB": row.get("correctness_judgeB"),
            "allergen_violation": row.get("allergen_violation"),
        }

    # Judge disagreement — largest |A − B| on correctness.
    if {"correctness_judgeA", "correctness_judgeB"}.issubset(df.columns):
        d = df.dropna(subset=["correctness_judgeA", "correctness_judgeB"]).copy()
        if len(d):
            d["_gap"] = (d["correctness_judgeA"] - d["correctness_judgeB"]).abs()
            r = d.sort_values("_gap", ascending=False).iloc[0]
            out.append(pack(r, "Judge disagreement",
                            f"Judges split: Gemma {r['correctness_judgeA']:.2f} vs "
                            f"DeepSeek {r['correctness_judgeB']:.2f}."))

    # Allergen-fixed — a record the deterministic post-filter touched.
    if "_safety_fixed" in df.columns:
        d = df[df["_safety_fixed"] == True]  # noqa: E712
        if len(d):
            out.append(pack(d.iloc[0], "Allergen post-filter",
                            "A blacklisted recipe was dropped by the deterministic safety filter."))

    # Honest empty plan — no slot filled (no-recommendation gate).
    if "plan" in df.columns:
        d = df[df["plan"].map(_plan_empty)]
        if len(d):
            out.append(pack(d.iloc[0], "Honest empty plan",
                            "No constraint-valid recipe found — the planner returns an honest "
                            "no-recommendation rather than hallucinating."))

    # Clean success — hit and high correctness.
    if {"hit_at_10", "correctness_judgeA"}.issubset(df.columns):
        d = df[(df["hit_at_10"] == 1) & (df["correctness_judgeA"] >= 0.9)]
        if len(d):
            out.append(pack(d.iloc[0], "Clean success",
                            "Gold retrieved and the plan judged correct."))
    return out

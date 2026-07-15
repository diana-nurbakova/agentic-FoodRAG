"""Read already-run evaluation results + QA metadata for the user study.

The study does **not** re-run the pipeline: its meal plans are selected from
pre-computed full-day evaluation result files (one ``{meal: food.com URL}`` plan
per question per config). This module reads those result files and the QA question
metadata, renders a question's persona as a human-readable study profile, and
scores candidate questions so the admin can pick a diverse, not-"too-easy" default
set of profiles.

Pure data access: no Streamlit, no HUMMUS load. Recipe-detail resolution (URL →
ingredients/steps/nutrition) happens in ``scripts/build_study_plans.py`` where the
HUMMUS dataframe is available.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

import pandas as pd

# Tight stratum QA set — matches the EXP-7 full-day experiments (n=1,646).
DEFAULT_QA_SET = "data/qa/main_10k_tight/recipe_question_answer_set.pkl"
_MEALS = ["breakfast", "lunch", "dinner", "snack"]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else _repo_root() / p


# ── result files (per-question plans) ─────────────────────────────────────
@lru_cache(maxsize=8)
def load_results(result_file: str) -> pd.DataFrame:
    """Load a per-question result file (``.parquet`` or ``.pkl``), indexed by question_id."""
    p = _resolve(result_file)
    df = pd.read_pickle(p) if p.suffix == ".pkl" else pd.read_parquet(p)
    if "question_id" in df.columns:
        df = df.set_index("question_id", drop=False)
    return df


def result_row(result_file: str, question_id: str) -> dict:
    """The full per-question result record a config produced ({} if missing)."""
    df = load_results(result_file)
    if question_id not in df.index:
        return {}
    row = df.loc[question_id]
    if hasattr(row, "iloc") and getattr(row, "ndim", 1) > 1:
        row = row.iloc[0]
    return row.to_dict() if hasattr(row, "to_dict") else dict(row)


def plan_for(result_file: str, question_id: str) -> dict:
    """The ``{meal: food.com URL}`` plan a config produced for a question ({} if missing)."""
    plan = result_row(result_file, question_id).get("plan")
    return dict(plan) if isinstance(plan, dict) else {}


def explanation_for(result_file: str, question_id: str) -> str:
    """The config's free-text explanation for a question ("" if none)."""
    row = result_row(result_file, question_id)
    text = row.get("explanation_text")
    return str(text).strip() if text and str(text) != "nan" else ""


def plan_is_complete(plan: dict) -> bool:
    """True when the plan fills all four meal slots with a URL."""
    return bool(plan) and all(plan.get(m) for m in _MEALS)


def plan_is_distinct(plan: dict, min_distinct: int = 4) -> bool:
    """True when the plan's meals use at least ``min_distinct`` different recipes.

    Weaker configs sometimes return the *same* recipe for every meal (a degenerate
    full-day plan). Such plans are poor study stimuli, so selection excludes them."""
    urls = [plan.get(m) for m in _MEALS if plan.get(m)]
    return len(set(urls)) >= min_distinct


def available_fullday_configs() -> list[dict]:
    """Discover usable full-day result files under ``eval_results/`` for the admin picker.

    Returns one entry per config_id (``{config_id, result_file, hypothesis, retriever}``),
    preferring the largest non-empty file when both ``.parquet`` and ``.pkl`` exist."""
    root = _repo_root() / "eval_results"
    best: dict[str, dict] = {}
    for p in sorted(root.glob("*fullday*per_question.*")):
        if p.suffix not in (".parquet", ".pkl") or p.stat().st_size == 0:
            continue
        config_id = p.stem.replace("_per_question", "")
        entry = {
            "config_id": config_id,
            "result_file": str(p.relative_to(root.parent)).replace("\\", "/"),
            "hypothesis": "H1" if "_h1_" in config_id else "H2" if "_h2_" in config_id else "",
            "retriever": ("text-to-pandas" if "_t2p_" in config_id
                          else "dense" if "_dense_" in config_id
                          else "sparse" if "_sparse_" in config_id
                          else "hybrid" if "_hybrid_" in config_id else ""),
            "_size": p.stat().st_size,
        }
        if config_id not in best or entry["_size"] > best[config_id]["_size"]:
            best[config_id] = entry
    return [{k: v for k, v in e.items() if k != "_size"} for e in best.values()]


# ── QA metadata (persona → profile) ───────────────────────────────────────
@lru_cache(maxsize=4)
def load_qa_set(qa_path: str = DEFAULT_QA_SET) -> list:
    """Load the QA item list (index i ↔ question_id ``q_{i:05d}``)."""
    return pd.read_pickle(_resolve(qa_path))


def qid_to_index(question_id: str) -> int:
    return int(str(question_id).split("_")[-1])


def qa_item(question_id: str, qa_path: str = DEFAULT_QA_SET) -> Optional[dict]:
    items = load_qa_set(qa_path)
    idx = qid_to_index(question_id)
    return items[idx] if 0 <= idx < len(items) else None


def _clean(term: str) -> str:
    """Tidy a raw persona term (e.g. ``'low - fat butter'`` → ``'low-fat butter'``)."""
    return " ".join(str(term).split()).replace(" - ", "-")


def _constraint_lines(persona: dict) -> list[str]:
    """Human-readable constraint bullets from a QA persona dict."""
    lines: list[str] = []
    tags = persona.get("tags", {}) or {}
    if tags.get("likes"):
        lines.append("Likes cuisine/tags: " + ", ".join(_clean(t) for t in tags["likes"]))
    if tags.get("dislikes"):
        lines.append("Avoids tags: " + ", ".join(_clean(t) for t in tags["dislikes"]))
    ings = persona.get("ingredients", {}) or {}
    if ings.get("likes"):
        lines.append("Wants ingredients: " + ", ".join(_clean(t) for t in ings["likes"]))
    if ings.get("dislikes"):
        lines.append("Avoids ingredients: " + ", ".join(_clean(t) for t in ings["dislikes"]))
    for nutrient, spec in (persona.get("relative_nutrients", {}) or {}).items():
        lim = (spec or {}).get("limits", {})
        pref = (spec or {}).get("preference", "")
        lo, hi = lim.get("lower"), lim.get("upper")
        lines.append(f"{nutrient}: {pref} ({lo}–{hi} per 100 g)")
    for nutrient, cap in (persona.get("total_nutrients", {}) or {}).items():
        lines.append(f"Daily {nutrient} ≤ {round(float(cap))}")
    for score, floor in (persona.get("nutrition_scores", {}) or {}).items():
        lines.append(f"{score} ≥ {round(float(floor), 2)}")
    return lines


def _constraint_tags(persona: dict) -> list[str]:
    """Short chips summarising which requirement families are present."""
    tags = []
    if (persona.get("tags") or {}).get("likes") or (persona.get("tags") or {}).get("dislikes"):
        tags.append("Tags")
    if (persona.get("ingredients") or {}).get("likes") or (persona.get("ingredients") or {}).get("dislikes"):
        tags.append("Ingredients")
    if persona.get("relative_nutrients"):
        tags.append("Nutrient limits")
    if persona.get("total_nutrients"):
        tags.append("Daily budget")
    if persona.get("nutrition_scores"):
        tags.append("Health score")
    return tags


def constraint_count(persona: dict) -> int:
    """Total number of individual constraints in a persona (difficulty proxy)."""
    tags = persona.get("tags", {}) or {}
    ings = persona.get("ingredients", {}) or {}
    return (len(tags.get("likes", [])) + len(tags.get("dislikes", []))
            + len(ings.get("likes", [])) + len(ings.get("dislikes", []))
            + len(persona.get("relative_nutrients", {}) or {})
            + len(persona.get("total_nutrients", {}) or {})
            + len(persona.get("nutrition_scores", {}) or {}))


def profile_from_question(question_id: str, *, index: int | None = None,
                          qa_path: str = DEFAULT_QA_SET) -> dict:
    """Display metadata for a study profile derived from a QA question."""
    item = qa_item(question_id, qa_path) or {}
    persona = item.get("persona", {}) or {}
    llm_qs = item.get("llm_questions") or [{}]
    prompt = (llm_qs[0].get("question") if llm_qs else "") or item.get("question", "")
    tag_chips = _constraint_tags(persona)
    answer_size = len(item.get("answer") or [])
    ordinal = index if index is not None else 0
    return {
        "id": f"P{ordinal + 1}",
        "question_id": question_id,
        "icon": "🍽",
        "name": f"Profile {ordinal + 1}",
        "goal": " · ".join(tag_chips) or "General",
        "desc": " ".join(_constraint_lines(persona)) or "No specific dietary constraints.",
        "tags": tag_chips,
        "prompt": prompt.strip(),
        "specificity": "specific" if constraint_count(persona) >= 3 else "general",
        "constraints": persona,
        "constraint_lines": _constraint_lines(persona),
        "constraint_count": constraint_count(persona),
        "answer_set_size": answer_size,
    }


# ── candidate selection (diverse, not "too easy") ─────────────────────────
def _requirement_families(persona: dict) -> frozenset[str]:
    return frozenset(_constraint_tags(persona))


def rank_candidates(condition_files: list[str], *, qa_path: str = DEFAULT_QA_SET,
                    min_constraints: int = 2, max_answer_set: int = 40,
                    min_distinct: int = 3, plan_ok=None) -> list[dict]:
    """Score candidate questions for use as study profiles.

    "Not too easy": keep only questions with ≥ ``min_constraints`` constraints and a
    gold answer set no larger than ``max_answer_set`` (fewer valid recipes ⇒ harder).
    Requires every condition's result file to contain a **complete** 4-meal plan with
    ≥ ``min_distinct`` different recipes (so all plans are real and non-degenerate).
    ``plan_ok(plan)`` (optional) is an extra per-plan predicate — e.g. a calorie
    plausibility check needing HUMMUS — applied to every condition's plan. Sorted
    hardest-first."""
    qa = load_qa_set(qa_path)
    scored: list[dict] = []
    for idx, item in enumerate(qa):
        persona = item.get("persona", {}) or {}
        n = constraint_count(persona)
        size = len(item.get("answer") or [])
        if n < min_constraints or size == 0 or size > max_answer_set:
            continue
        qid = f"q_{idx:05d}"
        plans = [plan_for(f, qid) for f in condition_files]
        # Every condition must have a complete, non-degenerate plan, so the participant
        # compares real full-day plans, not repeats of one recipe.
        if not all(plan_is_complete(p) and plan_is_distinct(p, min_distinct) for p in plans):
            continue
        if plan_ok is not None and not all(plan_ok(p) for p in plans):
            continue
        # Harder = more constraints, fewer valid answers. Diversity handled at pick time.
        difficulty = n + max(0.0, (max_answer_set - size) / max_answer_set)
        scored.append({"question_id": qid, "index": idx, "difficulty": round(difficulty, 3),
                       "constraint_count": n, "answer_set_size": size,
                       "families": sorted(_requirement_families(persona))})
    scored.sort(key=lambda r: r["difficulty"], reverse=True)
    return scored


def select_default_profiles(condition_files: list[str], n: int = 4, *,
                            qa_path: str = DEFAULT_QA_SET, min_distinct: int = 3,
                            plan_ok=None) -> list[dict]:
    """Greedily pick ``n`` hard-but-diverse questions (varying requirement families)."""
    candidates = rank_candidates(condition_files, qa_path=qa_path, min_distinct=min_distinct,
                                 plan_ok=plan_ok)
    picked: list[dict] = []
    seen_families: set[frozenset] = set()
    # First pass: prefer a fresh requirement-family signature for diversity.
    for c in candidates:
        sig = frozenset(c["families"])
        if sig not in seen_families:
            picked.append(c)
            seen_families.add(sig)
        if len(picked) >= n:
            break
    # Top up (if too few distinct signatures) with the next hardest remaining.
    if len(picked) < n:
        picked_ids = {c["question_id"] for c in picked}
        for c in candidates:
            if c["question_id"] not in picked_ids:
                picked.append(c)
            if len(picked) >= n:
                break
    return [profile_from_question(c["question_id"], index=i, qa_path=qa_path)
            for i, c in enumerate(picked[:n])]

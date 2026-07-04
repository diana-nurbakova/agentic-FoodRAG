"""Offline backfill of the new deterministic metrics onto already-computed records.

The Evaluation-Protocol metrics added after the runs (hard/soft constraint split
§4.3, avg_rating, slot_appropriateness §4.5, guideline adherence §4.5.1) are all
deterministic. Every full-pipeline record already stores ``retrieved_ids`` = the
final plan's recipe_urls, so the plan reconstructs offline (``urls_to_plan``) and
the new metrics are computed with ZERO LLM calls — no re-run needed.

The backfill is ADD-ONLY and idempotent: it never overwrites the existing
retrieval / plan / judge / latency metrics (which used the real assembled plan
and real LLM judges); it only inserts the new keys, folds a guideline
hard-ceiling (UL) breach into ``hard_constraint_violation`` (max), and persists
the reconstructed ``plan`` when absent.

Usage:  python -m scripts.backfill_metrics [config_id ...]
        (default: foodrag_h2 foodrag_h2_t2p foodrag_h1_dense)
Only runs with a complete 1646-line checkpoint are processed; a config's
aggregate/parquet are (re)written only when ALL its runs are complete.
"""
from __future__ import annotations

import json
import os
import pickle
import sys
import tempfile
from pathlib import Path

os.chdir(Path(__file__).resolve().parents[1])

from foodrag.baselines import constraints, explainability, guideline_adherence, meal_plan_metrics, metrics
from foodrag.baselines.runner import aggregate, load_recipes_df, write_results

K = 10
N_EXPECTED = 1646
CKPT = Path("data/qa/main_10k_tight/foodrag_ckpt")
QA_PATH = "data/qa/main_10k_tight/recipe_question_answer_set.pkl"
MEAL_PRIORITY = ("breakfast", "lunch", "dinner", "snack")

# config_id -> its per-run checkpoint file names
CONFIG_RUNS = {
    "foodrag_h2":       [f"foodrag_h2_run{r}.jsonl" for r in (1, 2, 3)],
    "foodrag_h2_t2p":   [f"foodrag_h2_t2p_run{r}.jsonl" for r in (1, 2, 3)],
    "foodrag_h1_dense": [f"foodrag_h1_dense_run{r}.jsonl" for r in (1, 2, 3)],
    "foodrag_h1_t2p":   [f"foodrag_h1_t2p_run{r}.jsonl" for r in (1, 2, 3)],
}

NEW_DET_KEYS = ("hard_constraint_sat_perconstraint", "hard_constraint_sat_allornothing",
                "soft_constraint_sat", "avg_rating_at_10")


def urls_to_plan(urls, indexed) -> dict:
    plan, used, leftovers = {}, set(), []
    for u in urls:
        if u not in indexed.index:
            continue
        row = indexed.loc[u]
        if getattr(row, "ndim", 1) == 2:
            row = row.iloc[0]
        rec = row.to_dict()
        mts = rec.get("meal_types")
        cand = []
        if isinstance(mts, str):
            cand = [m.strip() for m in mts.split(",") if m.strip() in MEAL_PRIORITY]
        elif isinstance(mts, (list, tuple)):
            cand = [str(m).strip() for m in mts if str(m).strip() in MEAL_PRIORITY]
        placed = False
        for slot in cand:
            if slot not in used:
                plan[slot] = rec; used.add(slot); placed = True; break
        if not placed:
            leftovers.append(rec)
    for rec in leftovers:
        for slot in MEAL_PRIORITY:
            if slot not in used:
                plan[slot] = rec; used.add(slot); break
    return plan


def _resolve(urls, indexed):
    rows = []
    for u in urls:
        if u in indexed.index:
            r = indexed.loc[u]
            rows.append((r.iloc[0] if getattr(r, "ndim", 1) == 2 else r).to_dict())
        else:
            rows.append(None)
    return rows


def backfill_record(rec: dict, qa, indexed) -> dict:
    """Insert the new deterministic keys into ``rec`` in place (add-only)."""
    i = int(rec["question_id"].split("_")[1])
    persona = qa[i]["persona"]
    gold = set(qa[i].get("answer") or [])
    urls = rec.get("retrieved_ids") or []
    rows = _resolve(urls, indexed)
    plan = urls_to_plan(urls, indexed)
    m = rec["metrics"]

    det = metrics.deterministic_metrics(urls, rows, gold, persona, K, constraints.identity_expand)
    for k in NEW_DET_KEYS:
        m[k] = det[k]

    m["slot_appropriateness"] = meal_plan_metrics.slot_appropriateness(plan)

    g = guideline_adherence.guideline_adherence(plan, persona)
    m["guideline_adherence_overall"] = g["guideline_adherence_overall"]
    for nut, val in g["guideline_adherence_per_nutrient"].items():
        m[f"guideline_adherence_{nut}"] = val  # flattened scalars → aggregatable
    if g.get("guideline_hard_ceiling_breach"):
        m["hard_constraint_violation"] = max(m.get("hard_constraint_violation", 0.0) or 0.0, 1.0)

    # §8.1 graph-path coverage/validity — from the reconstructed recipes (no explanation
    # text needed, so populated on EVERY full-pipeline config). §8.2 claim grounding needs
    # record["explanation_text"] → only present on new-code runs (foodrag_h1_t2p); NaN otherwise.
    m.update(explainability.explainability_metrics(
        persona, rows, explanation_text=rec.get("explanation_text", ""),
        expand=constraints.identity_expand))

    # non-scalar §13.1 fields live at top level (kept for fidelity; aggregate skips them)
    rec["guideline_adherence_per_nutrient"] = g["guideline_adherence_per_nutrient"]
    rec["guideline_class"] = g["guideline_class"]
    if "plan" not in rec or not rec.get("plan"):
        rec["plan"] = {mt: (row.get("recipe_url") if row else None) for mt, row in plan.items()}
    return rec


def _complete_runs(files):
    ok = []
    for f in files:
        p = CKPT / f
        if p.exists() and sum(1 for _ in p.open(encoding="utf-8")) == N_EXPECTED:
            ok.append(p)
    return ok


def _rewrite(path: Path, records):
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")
    os.replace(tmp, path)


def main(argv):
    configs = argv or ["foodrag_h2", "foodrag_h2_t2p", "foodrag_h1_dense"]
    recipes_df = load_recipes_df("data/hummus/recipes_df.csv")
    indexed = recipes_df.set_index("recipe_url", drop=False)
    qa = pickle.load(open(QA_PATH, "rb"))

    for cfg in configs:
        files = CONFIG_RUNS.get(cfg)
        if not files:
            print(f"[{cfg}] unknown config, skip"); continue
        complete = _complete_runs(files)
        if not complete:
            print(f"[{cfg}] no complete runs yet, skip"); continue
        all_records = []
        for p in complete:
            recs = [json.loads(l) for l in p.open(encoding="utf-8")]
            for rec in recs:
                backfill_record(rec, qa, indexed)
            _rewrite(p, recs)
            all_records.extend(recs)
            print(f"[{cfg}] backfilled {p.name} ({len(recs)} records)")
        if len(complete) == len(files):
            agg = aggregate(all_records)
            write_results(all_records, agg, out_dir="eval_results", config_id=cfg)
            av = agg[agg.stratum == "aggregate"].set_index("metric")["mean"]
            show = {k: round(float(av[k]), 3) for k in
                    ("guideline_adherence_overall", "hard_constraint_sat_perconstraint",
                     "slot_appropriateness", "avg_rating_at_10",
                     "path_coverage", "path_validity",
                     "claim_grounding_rate", "unsupported_claim_rate") if k in av.index}
            print(f"[{cfg}] AGGREGATED {len(complete)}/{len(files)} runs -> {show}")
        else:
            print(f"[{cfg}] {len(complete)}/{len(files)} runs complete — jsonl backfilled, "
                  f"deferring aggregate until all runs finish")


if __name__ == "__main__":
    main(sys.argv[1:])

"""Offline recompute of the slot-normalized plan-level metrics over stored records.

Implements the metric definitions in :mod:`foodrag.baselines.slot_normalized` as a
POST-PROCESSING pass: for each FoodRAG full-pipeline checkpoint record it adds the
``*_norm`` metrics (scored against the meals the plan was *requested* to produce, with
nutrient bands energy-share scaled) while preserving the existing full-day values as
``*_full``. NO pipeline re-run, no LLM, no 7GB hummus — only ``recipes_df`` (nutrients /
meal_types) + the stored records + ``config/guideline_reference_tables.json`` (bands).

Q (requested slots) source, per record:
- ``requested_meal_types`` present  → ``requested_slots_source = "persisted"``
  (only ``foodrag_h1_t2p`` ran post-persist).
- absent (h2 / h2_t2p / h1_dense)   → ``requested_slots_source = "approx_from_plan"``:
  Q = the plan's filled slots. Completeness/structural are then ~trivial for those 3
  (flagged by the source field); the energy-share-scaled guideline/calorie metrics stay
  meaningful.

The pass is ADD-ONLY and idempotent (keys are recomputed/overwritten; the full-day metric
values are read but never changed). Files are rewritten atomically (temp + os.replace).
Each config is re-aggregated (aggregate + write_results) only when all 3 runs are complete,
then ``eval_results/json`` is refreshed via ``scripts.export_results_json``.

Usage:  python -m scripts.recompute_slot_normalized [config_id ...]
        (default: foodrag_h1_t2p foodrag_h2 foodrag_h2_t2p foodrag_h1_dense)
"""
from __future__ import annotations

import json
import os
import runpy
import sys
import tempfile
from pathlib import Path

os.chdir(Path(__file__).resolve().parents[1])

from foodrag.baselines import guideline_adherence, slot_normalized
from foodrag.baselines.runner import aggregate, load_recipes_df, write_results

N_EXPECTED = 1646
CKPT = Path("data/qa/main_10k_tight/foodrag_ckpt")
RECIPES_PATH = "data/hummus/recipes_df.csv"

CONFIG_RUNS = {
    "foodrag_h1_t2p":   [f"foodrag_h1_t2p_run{r}.jsonl" for r in (1, 2, 3)],
    "foodrag_h2":       [f"foodrag_h2_run{r}.jsonl" for r in (1, 2, 3)],
    "foodrag_h2_t2p":   [f"foodrag_h2_t2p_run{r}.jsonl" for r in (1, 2, 3)],
    "foodrag_h1_dense": [f"foodrag_h1_dense_run{r}.jsonl" for r in (1, 2, 3)],
    # full-day variants (all 4 meal slots requested; requested_meal_types persisted for all).
    "foodrag_h1_t2p_fullday":   [f"foodrag_h1_t2p_fullday_run{r}.jsonl" for r in (1, 2, 3)],
    "foodrag_h2_fullday":       [f"foodrag_h2_fullday_run{r}.jsonl" for r in (1, 2, 3)],
    "foodrag_h2_t2p_fullday":   [f"foodrag_h2_t2p_fullday_run{r}.jsonl" for r in (1, 2, 3)],
    "foodrag_h1_dense_fullday": [f"foodrag_h1_dense_fullday_run{r}.jsonl" for r in (1, 2, 3)],
}

# full-day metric key -> the backward-compat *_full key it is preserved under.
FULL_MAP = {
    "meal_slot_completeness": "meal_slot_completeness_full",
    "structural_validity": "structural_validity_full",
    "guideline_adherence_overall": "guideline_adherence_full",
    "calorie_adherence": "calorie_adherence_full",
}

# scalar norm keys folded into rec["metrics"] (aggregatable).
_SCALAR_NORM_KEYS = (
    "meal_slot_completeness_norm", "structural_validity_norm", "guideline_adherence_norm",
    "calorie_adherence_norm", "intra_plan_diversity_norm", "requested_day_energy_share",
    "n_requested_slots",
)


def apply_norm(rec: dict, indexed, ref: dict) -> None:
    """Add the slot-normalized keys + preserve full-day values as ``*_full`` (in place)."""
    m = rec.setdefault("metrics", {})

    # preserve the current full-day headline values (add-only; idempotent — they never change).
    for src, dst in FULL_MAP.items():
        if src in m:
            m[dst] = m[src]

    block = slot_normalized.compute_record_norm(rec, indexed, ref)
    for k in _SCALAR_NORM_KEYS:
        m[k] = block[k]
    # flattened per-nutrient scalars (aggregatable) — spec: report the vector always.
    for short, val in block["guideline_adherence_norm_per_nutrient"].items():
        m[f"guideline_adherence_norm_{short}"] = val
    # non-scalar / string fidelity fields live at top level (aggregate skips them).
    rec["guideline_adherence_norm_per_nutrient"] = block["guideline_adherence_norm_per_nutrient"]
    rec["requested_slots_source"] = block["requested_slots_source"]


def _rewrite(path: Path, records) -> None:
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")
    os.replace(tmp, path)


def _complete_runs(files):
    ok = []
    for f in files:
        p = CKPT / f
        if p.exists() and sum(1 for _ in p.open(encoding="utf-8")) == N_EXPECTED:
            ok.append(p)
    return ok


def main(argv):
    configs = argv or list(CONFIG_RUNS)
    recipes_df = load_recipes_df(RECIPES_PATH)
    indexed = recipes_df.set_index("recipe_url", drop=False)
    ref = guideline_adherence.load_reference_tables()

    aggregated_any = False
    for cfg in configs:
        files = CONFIG_RUNS.get(cfg)
        if not files:
            print(f"[{cfg}] unknown config, skip"); continue
        complete = _complete_runs(files)
        if not complete:
            print(f"[{cfg}] no complete runs yet, skip"); continue
        all_records, src_counts = [], {"persisted": 0, "approx_from_plan": 0}
        for p in complete:
            recs = [json.loads(l) for l in p.open(encoding="utf-8")]
            for rec in recs:
                apply_norm(rec, indexed, ref)
                src_counts[rec.get("requested_slots_source", "approx_from_plan")] += 1
            _rewrite(p, recs)
            all_records.extend(recs)
            print(f"[{cfg}] recomputed {p.name} ({len(recs)} records)")

        if len(complete) == len(files):
            agg = aggregate(all_records, config_id=cfg)
            write_results(all_records, agg, out_dir="eval_results", config_id=cfg)
            aggregated_any = True
            av = agg[agg.stratum == "aggregate"].set_index("metric")["mean"]
            show = {k: round(float(av[k]), 4) for k in (
                "meal_slot_completeness_full", "meal_slot_completeness_norm",
                "structural_validity_full", "structural_validity_norm",
                "guideline_adherence_full", "guideline_adherence_norm",
                "calorie_adherence_full", "calorie_adherence_norm",
                "requested_day_energy_share",
            ) if k in av.index}
            print(f"[{cfg}] AGGREGATED {len(complete)}/{len(files)} runs  src={src_counts}\n"
                  f"        {show}")
        else:
            print(f"[{cfg}] {len(complete)}/{len(files)} runs complete — jsonl recomputed, "
                  f"deferring aggregate until all runs finish")

    if aggregated_any:
        print("\n[export] refreshing eval_results/json ...")
        runpy.run_module("scripts.export_results_json", run_name="__main__")


if __name__ == "__main__":
    main(sys.argv[1:])

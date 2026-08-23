"""Day-level budget case studies: popularity individual recs vs the FoodRAG meal plan.

Companion to ``docs/BUDGET_CASE_STUDIES.md``. Produces, for tight-stratum personas that
state a ``total_nutrients`` daily budget:

* per-recipe nutrition + health scores for the popularity-raw top-10 (rung 0a), the
  popularity health-aware top-10 (rung 0b) and the H1-t2p meal plan (rung 4);
* day-level budget accounting (sum of the budgeted nutrient across the day vs the cap);
* an aggregate table over every such persona.

Budget semantics. ``config/nutrition_config.yml:TOTAL_NUTRIENTS`` states the reference
intakes **per day** (2000 kcal, 2300 mg sodium, ...); the persona generator perturbs them
by +/-20%. But ``foodrag.baselines.constraints`` enforces each cap **per recipe**
(``row[n]/100 * servingSize <= cap``) -- no component sums the day. Day totals here use
that same whole-serving quantity, so the accounting matches the codebase exactly and is
identical across all three systems.

QA-set alignment. The popularity rungs were scored on ``data/qa/main_10k`` (10k personas)
while the full-day FoodRAG runs use ``data/qa/main_10k_tight`` (the 1,646 tight personas,
re-indexed). ``question_id`` therefore collides across the two files without referring to
the same persona; :func:`build_qa_map` maps tight -> 10k by exact question text.

Usage
-----
    python scripts/budget_case_study.py --case q_00417 q_01087
    python scripts/budget_case_study.py --aggregate
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from foodrag.baselines import constraints  # noqa: E402
from foodrag.baselines.runner import load_recipes_df  # noqa: E402

SERVING = "servingSize [g]"
HEALTH = ["nutri_score", "fsa_score", "who_score"]
SHOWN_NUTRIENTS = ["calories [kcal]", "saturatedFat [g]", "sodium [mg]", "sugars [g]",
                   "dietaryFiber [g]", "protein [g]", "totalFat [g]",
                   "totalCarbohydrate [g]", "cholesterol [mg]", "caloriesFromFat [kcal]"]
SLOT_ORDER = ["breakfast", "lunch", "dinner", "snack"]
DAY_K = 4  # top-4 individual recs treated as a day, to match the 4 plan slots


def load_all():
    recipes = load_recipes_df(ROOT / "data/hummus/recipes_df.csv")
    idx = recipes.set_index("recipe_url", drop=False)
    idx = idx[~idx.index.duplicated(keep="first")]
    tight = pd.read_pickle(ROOT / "data/qa/main_10k_tight/recipe_question_answer_set.pkl")
    full = pd.read_pickle(ROOT / "data/qa/main_10k/recipe_question_answer_set.pkl")
    fd = pd.read_parquet(ROOT / "eval_results/foodrag_h1_t2p_fullday_per_question.parquet")
    fd = fd[fd["run"] == 1].set_index("question_id")
    praw = pd.read_parquet(ROOT / "eval_results/popularity_raw_per_question.parquet").set_index("question_id")
    pha = pd.read_parquet(ROOT / "eval_results/popularity_health_aware_per_question.parquet").set_index("question_id")
    return idx, tight, full, fd, praw, pha


def build_qa_map(tight, full) -> dict[str, str]:
    """Map ``main_10k_tight`` question ids to ``main_10k`` ids by exact question text."""
    by_q = defaultdict(list)
    for i, r in enumerate(full):
        by_q[r["question"]].append(i)
    mapping = {}
    for j, r in enumerate(tight):
        hits = by_q.get(r["question"], [])
        if len(hits) > 1:  # disambiguate identical question strings by answer-set size
            hits = [i for i in hits if len(full[i]["answer"]) == len(r["answer"])] or hits
        if hits:
            mapping[f"q_{j:05d}"] = f"q_{hits[0]:05d}"
    return mapping


def whole(row, nutrient: str) -> float:
    """Whole-serving amount of ``nutrient`` (HUMMUS stores per-100 g)."""
    return row[nutrient] / 100 * row[SERVING]


def rinfo(idx, url: str, persona: dict) -> dict:
    r = idx.loc[url]
    d = {"title": r["title"], "url": url, "serving_g": float(r[SERVING]),
         "servings_per_recipe": r["servingsPerRecipe"],
         "rating": float(r["average_rating"]), "n_ratings": int(r["number_of_ratings"])}
    d.update({n: round(float(whole(r, n)), 1) for n in SHOWN_NUTRIENTS})
    d.update({h: round(float(r[h]), 3) for h in HEALTH})
    d["hard_ok"] = not constraints.violates_hard_nutrient(r, persona)
    d["allergen_ok"] = not constraints.violates_allergen(r, persona)
    return d


def _show(items, title, slots=None):
    print(f"\n--- {title} ---")
    print(f"{'#':<4} {'title':<44} {'kcal':>6} {'satfat':>6} {'sodium':>7} {'sugar':>6} "
          f"{'fibre':>5} {'nutri':>5} {'fsa':>5} {'who':>5} {'rate':>5}  valid")
    for i, it in enumerate(items):
        tag = slots[i] if slots else str(i + 1)
        ok = "OK" if (it["hard_ok"] and it["allergen_ok"]) else "FAIL"
        print(f"{tag:<4} {it['title'][:44]:<44} {it['calories [kcal]']:>6.0f} "
              f"{it['saturatedFat [g]']:>6.1f} {it['sodium [mg]']:>7.0f} {it['sugars [g]']:>6.1f} "
              f"{it['dietaryFiber [g]']:>5.1f} {it['nutri_score']:>5.2f} {it['fsa_score']:>5.2f} "
              f"{it['who_score']:>5.2f} {it['rating']:>5.2f}  {ok}")


def case(qid: str, data, qmap):
    idx, tight, full, fd, praw, pha = data
    rec = fd.loc[qid]
    qa = tight[int(qid.split("_")[1])]
    persona = qa["persona"]
    fq = qmap[qid]
    plan = rec["plan"]

    print("=" * 104)
    print(f"CASE {qid}  (= {fq} in main_10k)   stratum={rec['stratum']}   "
          f"gold answer set = {rec['answer_set_size']} recipes")
    print("=" * 104)
    print("\n--- PERSONA (structured query) ---")
    print(qa["question"])

    plan_items = [(s, plan[s]) for s in SLOT_ORDER if plan.get(s)]
    sys_items = [rinfo(idx, u, persona) for _, u in plan_items]
    raw_items = [rinfo(idx, u, persona) for u in praw.loc[fq, "retrieved_ids"]]
    ha_items = [rinfo(idx, u, persona) for u in pha.loc[fq, "retrieved_ids"]]

    _show(raw_items, "A. POPULARITY-RAW top-10 (rung 0a; individual recs, persona-independent)")
    _show(ha_items, "B. POPULARITY HEALTH-AWARE top-10 (rung 0b; individual recs, valid-set pre-filtered)")
    _show(sys_items, "C. FOODRAG H1-t2p MEAL PLAN (rung 4)", slots=[s[:4] for s, _ in plan_items])

    print("\n--- BUDGET STATUS (persona's stated daily budget) ---")
    rows = []
    for label, items in [(f"A. popularity-raw top-{DAY_K} as a day", raw_items[:DAY_K]),
                         (f"B. pop health-aware top-{DAY_K} as a day", ha_items[:DAY_K]),
                         ("C. FoodRAG meal plan (4 slots)", sys_items)]:
        for n, cap in (persona.get("total_nutrients") or {}).items():
            tot = sum(i[n] for i in items)
            rows.append({"scope": label, "nutrient": n, "cap": round(cap, 1),
                         "day_total": round(tot, 1), "pct_of_cap": round(100 * tot / cap, 1),
                         "day_within_budget": tot <= cap,
                         "recipes_over_cap_alone": sum(1 for i in items if i[n] > cap),
                         "n_recipes": len(items)})
    print(pd.DataFrame(rows).to_string(index=False))

    print("\n--- DAY NUTRITION TOTALS ---")
    tot_rows = []
    for label, items in [(f"A. pop-raw top-{DAY_K}", raw_items[:DAY_K]),
                         (f"B. pop-health top-{DAY_K}", ha_items[:DAY_K]),
                         ("C. FoodRAG plan", sys_items)]:
        d = {"day": label, "n_recipes": len(items)}
        d.update({n: round(sum(i[n] for i in items), 1) for n in
                  ["calories [kcal]", "saturatedFat [g]", "sodium [mg]", "sugars [g]",
                   "dietaryFiber [g]", "protein [g]"]})
        d.update({"mean " + h: round(float(np.mean([i[h] for i in items])), 3) for h in HEALTH})
        tot_rows.append(d)
    print(pd.DataFrame(tot_rows).to_string(index=False))

    print("\n--- VALIDITY ---")
    for label, items in [("A. pop-raw top-10", raw_items), ("B. pop-health top-10", ha_items),
                         ("C. FoodRAG plan", sys_items)]:
        nv = sum(1 for i in items if i["hard_ok"] and i["allergen_ok"])
        na = sum(1 for i in items if not i["allergen_ok"])
        uniq = len({i["url"] for i in items})
        print(f"{label:<22} valid {nv}/{len(items)}   allergen violations {na}   unique recipes {uniq}")

    m = rec["metrics"]
    print(f"\nrecorded metrics — FoodRAG: hit@10={m.get('hit_at_10')} "
          f"hard_sat={m.get('hard_constraint_sat_perconstraint'):.3f} "
          f"guideline={m.get('guideline_adherence_overall')} "
          f"calorie_adherence={m.get('calorie_adherence'):.3f}")
    print(f"recorded metrics — pop-raw: hit@10={praw.loc[fq,'metrics'].get('hit_at_10')} "
          f"constraint_sat={praw.loc[fq,'metrics'].get('constraint_satisfaction')}")
    print(f"recorded metrics — pop-health: hit@10={pha.loc[fq,'metrics'].get('hit_at_10')} "
          f"constraint_sat={pha.loc[fq,'metrics'].get('constraint_satisfaction')}")
    print()


def aggregate(data, qmap, out_csv: Path | None = None):
    idx, tight, full, fd, praw, pha = data

    def day_total(urls, n):
        rows = idx.reindex([u for u in urls if u in idx.index])
        return float((rows[n] / 100 * rows[SERVING]).sum()), len(rows)

    rows = []
    for j, qa in enumerate(tight):
        qid = f"q_{j:05d}"
        if qid not in fd.index or qid not in qmap:
            continue
        tn = qa["persona"].get("total_nutrients") or {}
        if not tn:
            continue
        rec = fd.loc[qid]
        plan = rec["plan"]
        if not isinstance(plan, dict):
            continue
        plan_urls = [plan[s] for s in SLOT_ORDER if plan.get(s)]
        if len(plan_urls) != 4:
            continue
        fq = qmap[qid]
        n, cap = next(iter(tn.items()))
        ha_all = list(pha.loc[fq, "retrieved_ids"])
        raw_tot, _ = day_total(list(praw.loc[fq, "retrieved_ids"])[:DAY_K], n)
        ha_tot, ha_k = day_total(ha_all[:DAY_K], n)
        sys_tot, _ = day_total(plan_urls, n)
        rows.append({"qid": qid, "qid_10k": fq, "nutrient": n, "cap": cap,
                     "gold_set": rec["answer_set_size"],
                     "raw_pct": 100 * raw_tot / cap,
                     "ha_pct": 100 * ha_tot / cap if ha_k else np.nan,
                     "sys_pct": 100 * sys_tot / cap,
                     "ha_valid_available": len(ha_all),
                     "ha_can_fill_day": len(ha_all) >= DAY_K,
                     "sys_unique_recipes": len(set(plan_urls))})

    d = pd.DataFrame(rows)
    print(f"Tight personas stating a daily budget AND given a full 4-slot plan: {len(d)}\n")
    print("=== DAY TOTAL AS % OF THE PERSONA'S STATED DAILY CAP (100% = exactly at budget) ===")
    print(pd.DataFrame({
        f"A. popularity-raw top-{DAY_K}": d["raw_pct"].describe(percentiles=[.25, .5, .75]),
        f"B. pop health-aware top-{DAY_K}": d["ha_pct"].describe(percentiles=[.25, .5, .75]),
        "C. FoodRAG meal plan": d["sys_pct"].describe(percentiles=[.25, .5, .75]),
    }).round(1).to_string())

    print("\n=== SHARE OF PERSONAS WHOSE DAY STAYS WITHIN THE STATED DAILY BUDGET ===")
    for label, col in [(f"A. popularity-raw top-{DAY_K}", "raw_pct"),
                       (f"B. pop health-aware top-{DAY_K}", "ha_pct"),
                       ("C. FoodRAG meal plan", "sys_pct")]:
        s = d[col].dropna()
        print(f"{label:<30} within budget {100*(s<=100).mean():5.1f}%   "
              f"median {s.median():7.1f}%   mean {s.mean():7.1f}%")

    print("\n=== CAN AN INDIVIDUAL RECOMMENDER EVEN FILL A DAY? (health-aware valid set) ===")
    print(f"personas whose valid set has >= {DAY_K} recipes: {100*d['ha_can_fill_day'].mean():.1f}%")
    fillable = d[d["ha_can_fill_day"]]
    print(f"of those, day exceeds the budget anyway: "
          f"{100*(fillable['ha_pct']>100).mean():.1f}% ({(fillable['ha_pct']>100).sum()}/{len(fillable)})")
    print("\nvalid-set size distribution:")
    print(d["ha_valid_available"].value_counts().sort_index().to_string())

    print("\n=== FoodRAG plan diversity (unique recipes across the 4 slots) ===")
    print(d["sys_unique_recipes"].value_counts().sort_index().to_string())

    print("\n=== MEDIAN % OF CAP BY BUDGETED NUTRIENT ===")
    piv = d.groupby("nutrient")[["raw_pct", "ha_pct", "sys_pct"]].median().round(1)
    piv["n"] = d.groupby("nutrient").size()
    print(piv.to_string())

    if out_csv:
        d.to_csv(out_csv, index=False)
        print(f"\nwrote {out_csv}")
    return d


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--case", nargs="*", default=[], help="tight question ids, e.g. q_00417")
    ap.add_argument("--aggregate", action="store_true", help="run the aggregate budget table")
    ap.add_argument("--out-csv", type=Path, default=None)
    args = ap.parse_args()
    if not args.case and not args.aggregate:
        ap.error("pass --case <qid ...> and/or --aggregate")

    data = load_all()
    qmap = build_qa_map(data[1], data[2])
    for q in args.case:
        case(q, data, qmap)
    if args.aggregate:
        aggregate(data, qmap, args.out_csv)


if __name__ == "__main__":
    main()

"""Apply the two committed safety fixes (commit 54a62ae) to the AFFECTED §13.1
evaluation records, re-judge the fixed output, and re-aggregate — so the stored
results reflect the fixed system for exactly the records the fixes change.

Both fixes are POST-HOC / deterministic, so re-running the whole pipeline would
only change the final plan + explanation. Instead we apply the fix to the stored
record and re-judge (LLM) that changed output.

Fix 1 — constraints.enforce_allergen_safety(plan, persona): drop any meal slot
whose recipe contains a persona-blacklisted (disliked) ingredient. Affects any
record with metrics.allergen_violation > 0.
Fix 2 — ExplanationTool.EMPTY_PLAN_MESSAGE: an empty plan carries the fixed safe
message, not a hallucinated explanation. Affects h1_t2p records with no
retrieved_ids but a non-empty explanation_text (and any allergen record whose
filtered plan becomes empty).

Memory-light: recipes_df + the judge LLMs only (NO 7GB hummus, NO full re-run).
Idempotent + resumable: each processed record is stamped ``_safety_fixed`` with a
version tag; already-stamped records are skipped. Files are rewritten atomically
and re-aggregation only runs when all 3 runs of a config are complete.

Usage:
    python -m scripts.apply_safety_fixes --verify         # 10 allergen + 10 empty, no write
    python -m scripts.apply_safety_fixes --limit 10       # process <=10 affected/file, write+checkpoint
    python -m scripts.apply_safety_fixes                  # process all 493, write, re-aggregate
    python -m scripts.apply_safety_fixes --aggregate-only # just re-aggregate affected configs
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import tempfile
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

os.chdir(Path(__file__).resolve().parents[1])

import pandas as pd

from foodrag.baselines import (constraints, explainability, guideline_adherence,
                               meal_plan_metrics, metrics)
from foodrag.baselines import judge as judge_mod
from foodrag.baselines.runner import aggregate, load_recipes_df, write_results
from foodrag.food_rag.tools.explanation_tools import ExplanationTool

CKPT = Path("data/qa/main_10k_tight/foodrag_ckpt")
QA_PATH = "data/qa/main_10k_tight/recipe_question_answer_set.pkl"
RECIPES_PATH = "data/hummus/recipes_df.csv"
N_EXPECTED = 1646
K = 10
MEAL_PRIORITY = ("breakfast", "lunch", "dinner", "snack")
GOLD_TITLES_CAP = 20                      # mirror full_pipeline.GOLD_TITLES_CAP
FIX_VERSION = "54a62ae-v1"                # bump to force a re-process
EMPTY_MSG = ExplanationTool.EMPTY_PLAN_MESSAGE

# config_id -> emits an explanation (Heuristic 1) => judge explanation_quality
REQUIRE_EXPLANATION = {
    "foodrag_h2": False, "foodrag_h2_t2p": False,
    "foodrag_h1_dense": True, "foodrag_h1_t2p": True,
}
RUN_FILES = {cfg: [f"{cfg}_run{r}.jsonl" for r in (1, 2, 3)] for cfg in REQUIRE_EXPLANATION}
ALLERGEN_CONFIGS = ("foodrag_h2", "foodrag_h2_t2p", "foodrag_h1_dense", "foodrag_h1_t2p")
EMPTY_CONFIGS = ("foodrag_h1_t2p",)


# ── plan reconstruction (verbatim from scripts/backfill_metrics.py) ───────────
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


def _dedupe(values):
    seen, out = set(), []
    for v in values:
        if v and v not in seen:
            seen.add(v); out.append(v)
    return out


# ── judge-input reconstruction (mirror full_pipeline / the H1 assembler) ──────
def _plan_titles(plan: dict) -> list[str]:
    """Recipe titles of the FILLED slots — the H1 assembler's ``contexts``."""
    return [str(r.get("title")) for r in plan.values() if r and r.get("title")]


def _answer_text(plan: dict, explanation: str, has_explanation: bool) -> str:
    """Mirror full_pipeline._answer_text's fallback branch (no persisted final_recommendation):
    ``"Meal plan:\\n{slot}: {title}"`` + the explanation when the config emits one.
    """
    parts: list[str] = []
    titles = [f"{slot}: {r.get('title')}" for slot, r in plan.items() if r and r.get("title")]
    if titles:
        parts.append("Meal plan:\n" + "\n".join(titles))
    if has_explanation and explanation:
        parts.append(explanation)
    return "\n\n".join(parts) if parts else "No recommendation produced."


def _gold_titles(gold: set[str], indexed: pd.DataFrame) -> list[str]:
    titles = []
    for url in list(gold)[:GOLD_TITLES_CAP]:
        if url in indexed.index:
            row = indexed.loc[url]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            title = row.get("title")
            if title:
                titles.append(str(title))
    return titles


# ── deterministic + judge recompute for a fixed plan (mirror run_full_pipeline) ─
def recompute_deterministic(rec, persona, gold, plan, urls, rows, explanation_text,
                            require_explanation) -> None:
    """Recompute + merge (add-only, overwriting changed deterministic keys) the whole
    deterministic block from the filtered plan, exactly as full_pipeline.run_full_pipeline
    builds it. Mutates rec['metrics'] in place."""
    m = rec["metrics"]

    block = metrics.deterministic_metrics(urls, rows, gold, persona, K, constraints.identity_expand)
    if not gold:  # retrieval metrics undefined without a gold set
        for name in (f"hit_at_{K}", f"mrr_at_{K}", f"ndcg_at_{K}"):
            block[name] = float("nan")

    has_explanation = bool(explanation_text and explanation_text.strip())
    block.update(meal_plan_metrics.plan_quality_metrics(
        plan, has_explanation=has_explanation, require_explanation=require_explanation))

    guide = guideline_adherence.guideline_adherence(plan, persona)
    block["guideline_adherence_overall"] = guide["guideline_adherence_overall"]
    if guide.get("guideline_hard_ceiling_breach"):
        block["hard_constraint_violation"] = max(
            block.get("hard_constraint_violation", 0.0) or 0.0, 1.0)
    # flattened per-nutrient scalars (aggregatable) + non-scalar top-level fidelity fields
    for nut, val in guide["guideline_adherence_per_nutrient"].items():
        block[f"guideline_adherence_{nut}"] = val
    rec["guideline_adherence_per_nutrient"] = guide["guideline_adherence_per_nutrient"]
    rec["guideline_class"] = guide["guideline_class"]

    block.update(explainability.explainability_metrics(
        persona, list(plan.values()), explanation_text=explanation_text,
        expand=constraints.identity_expand))

    m.update(block)


def rejudge(rec, judges, persona, gold, plan, explanation_text,
            require_explanation, indexed) -> None:
    """Re-judge the fixed plan and overwrite the judged metric keys."""
    has_explanation = bool(explanation_text and explanation_text.strip())
    answer = _answer_text(plan, explanation_text, has_explanation)
    contexts = _plan_titles(plan)
    question = rec.get("question", "")
    judged = judge_mod.judge_plan(
        judges, question=question, answer=answer, contexts=contexts,
        gold=_gold_titles(gold, indexed), require_explanation=require_explanation)
    rec["metrics"].update(judged)


# ── per-record processing ─────────────────────────────────────────────────────
def is_allergen(rec) -> bool:
    v = rec["metrics"].get("allergen_violation")
    return isinstance(v, (int, float)) and v == v and v > 0


def is_empty_expl(rec) -> bool:
    urls = rec.get("retrieved_ids") or []
    et = rec.get("explanation_text") or ""
    return len(urls) == 0 and et.strip() != ""


def already_fixed(rec) -> bool:
    return rec.get("_safety_fixed") == FIX_VERSION


def process_allergen(rec, qa, indexed, judges, cfg) -> dict:
    """Apply fix 1 (+fix 2 if plan empties), recompute deterministic, re-judge."""
    i = int(rec["question_id"].split("_")[1])
    persona = qa[i]["persona"]
    gold = set(qa[i].get("answer") or [])
    require_expl = REQUIRE_EXPLANATION[cfg]
    before_av = rec["metrics"].get("allergen_violation")

    urls0 = rec.get("retrieved_ids") or []
    plan0 = urls_to_plan(urls0, indexed)
    plan = constraints.enforce_allergen_safety(plan0, persona)      # fix 1
    new_urls = _dedupe([r.get("recipe_url") for r in plan.values() if r and r.get("recipe_url")])
    rows = _resolve(new_urls, indexed)

    # fix 2: if the filtered plan is now empty AND this config emits an explanation,
    # the explanation becomes the fixed safe message.
    explanation_text = rec.get("explanation_text", "")
    plan_empty = not any(r for r in plan.values())
    if plan_empty and require_expl:
        explanation_text = EMPTY_MSG
        rec["explanation_text"] = EMPTY_MSG

    recompute_deterministic(rec, persona, gold, plan, new_urls, rows,
                            explanation_text, require_expl)
    rejudge(rec, judges, persona, gold, plan, explanation_text, require_expl, indexed)

    rec["retrieved_ids"] = new_urls
    rec["plan"] = {str(mt): (r.get("recipe_url") if r else None) for mt, r in plan.items()}
    rec["_safety_fixed"] = FIX_VERSION
    after_av = rec["metrics"].get("allergen_violation")
    return {"qid": rec["question_id"], "before_av": before_av, "after_av": after_av,
            "empty_after": plan_empty, "n_slots": len(new_urls)}


def process_empty_expl(rec, qa, indexed, judges, cfg) -> dict:
    """Apply fix 2 to an empty-plan record: fixed message + refresh §8 + re-judge."""
    i = int(rec["question_id"].split("_")[1])
    persona = qa[i]["persona"]
    gold = set(qa[i].get("answer") or [])
    require_expl = REQUIRE_EXPLANATION[cfg]           # True for h1_t2p
    before_expl = (rec.get("explanation_text") or "")[:80]
    before_cg = rec["metrics"].get("claim_grounding_rate")

    urls = rec.get("retrieved_ids") or []            # empty
    plan = urls_to_plan(urls, indexed)               # {}
    rows = _resolve(urls, indexed)                    # []

    rec["explanation_text"] = EMPTY_MSG               # fix 2
    # §8: fixed message has no extractable factual claims -> claim_grounding NaN
    rec["metrics"].update(explainability.explainability_metrics(
        persona, rows, explanation_text=EMPTY_MSG, expand=constraints.identity_expand))
    # re-judge the fixed message (answer = fixed message; empty plan -> empty contexts)
    rejudge(rec, judges, persona, gold, plan, EMPTY_MSG, require_expl, indexed)

    rec["_safety_fixed"] = FIX_VERSION
    after_cg = rec["metrics"].get("claim_grounding_rate")
    return {"qid": rec["question_id"], "before_expl": before_expl,
            "after_expl": rec["explanation_text"][:80],
            "before_cg": before_cg, "after_cg": after_cg}


def process_record(rec, qa, indexed, judges, cfg) -> tuple[str, dict]:
    if is_allergen(rec):
        return "allergen", process_allergen(rec, qa, indexed, judges, cfg)
    if cfg in EMPTY_CONFIGS and is_empty_expl(rec):
        return "empty_expl", process_empty_expl(rec, qa, indexed, judges, cfg)
    return "skip", {}


# ── file processing ───────────────────────────────────────────────────────────
def _affected_indices(records, cfg):
    idxs = []
    for idx, rec in enumerate(records):
        hit = is_allergen(rec) or (cfg in EMPTY_CONFIGS and is_empty_expl(rec))
        if hit and not already_fixed(rec):
            idxs.append(idx)
    return idxs


def _rewrite(path: Path, records) -> None:
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")
    os.replace(tmp, path)


def process_file(path: Path, cfg, qa, indexed, judges, workers, limit) -> None:
    records = [json.loads(l) for l in path.open(encoding="utf-8")]
    todo = _affected_indices(records, cfg)
    if limit is not None:
        todo = todo[:limit]
    if not todo:
        print(f"[{path.name}] nothing to do (0 affected/unfixed)")
        return
    print(f"[{path.name}] {len(todo)} affected records to fix with {workers} workers")

    lock = threading.Lock()
    done = {"n": 0, "allergen": 0, "empty_expl": 0, "err": 0}
    last_ckpt = {"n": 0}

    def _work(idx):
        rec = records[idx]
        try:
            kind, info = process_record(rec, qa, indexed, judges, cfg)
            return idx, kind, info, None
        except Exception as e:  # pragma: no cover
            return idx, "error", {}, f"{type(e).__name__}: {e}\n{traceback.format_exc()}"

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_work, idx) for idx in todo]
        for fut in as_completed(futures):
            idx, kind, info, err = fut.result()
            with lock:
                done["n"] += 1
                n = done["n"]
                if kind in done:
                    done[kind] += 1
                if kind == "error":
                    done["err"] += 1
                    print(f"[{path.name}] ERROR record {idx}: {err.splitlines()[0]}")
                elif n <= 5 or n % 100 == 0:
                    print(f"[{path.name}] {n}/{len(todo)} {kind} :: {info}")
                if n - last_ckpt["n"] >= 100:
                    _rewrite(path, records)
                    last_ckpt["n"] = n
                    print(f"[{path.name}] checkpoint @ {n}/{len(todo)}")

    _rewrite(path, records)
    print(f"[{path.name}] DONE: {done['n']} processed "
          f"({done['allergen']} allergen, {done['empty_expl']} empty-expl, {done['err']} err)")


# ── verify mode (no writes) ───────────────────────────────────────────────────
def verify(qa, indexed, judges) -> None:
    import copy
    print("=== VERIFY: 10 allergen + 10 empty-plan-explanation (no writes) ===\n")
    # 10 allergen from h2_run1
    allergen_seen = 0
    for line in (CKPT / "foodrag_h2_run1.jsonl").open(encoding="utf-8"):
        rec = json.loads(line)
        if is_allergen(rec) and allergen_seen < 10:
            r = copy.deepcopy(rec)
            info = process_allergen(r, qa, indexed, judges, "foodrag_h2")
            print(f"ALLERGEN {info['qid']}: allergen_violation {info['before_av']} -> "
                  f"{info['after_av']}  (slots left={info['n_slots']}, empty_after={info['empty_after']})")
            allergen_seen += 1
        if allergen_seen >= 10:
            break
    print()
    # 10 empty-plan-explanation from h1_t2p_run1
    empty_seen = 0
    for line in (CKPT / "foodrag_h1_t2p_run1.jsonl").open(encoding="utf-8"):
        rec = json.loads(line)
        if is_empty_expl(rec) and empty_seen < 10:
            r = copy.deepcopy(rec)
            info = process_empty_expl(r, qa, indexed, judges, "foodrag_h1_t2p")
            print(f"EMPTY {info['qid']}: cg {info['before_cg']} -> {info['after_cg']}")
            print(f"   expl before: {info['before_expl']!r}")
            print(f"   expl after : {info['after_expl']!r}")
            empty_seen += 1
        if empty_seen >= 10:
            break
    print("\n=== verify complete (nothing written) ===")


# ── re-aggregation ────────────────────────────────────────────────────────────
def reaggregate(cfg) -> None:
    files = [CKPT / f for f in RUN_FILES[cfg]]
    complete = []
    for p in files:
        n = sum(1 for _ in p.open(encoding="utf-8")) if p.exists() else 0
        if n == N_EXPECTED:
            complete.append(p)
        else:
            print(f"[aggregate {cfg}] {p.name}: {n}/{N_EXPECTED} — incomplete")
    if len(complete) != len(files):
        print(f"[aggregate {cfg}] {len(complete)}/{len(files)} runs complete — skip")
        return
    all_records = []
    for p in complete:
        all_records.extend(json.loads(l) for l in p.open(encoding="utf-8"))
    agg = aggregate(all_records, config_id=cfg)
    write_results(all_records, agg, out_dir="eval_results", config_id=cfg)
    av = agg[agg.stratum == "aggregate"].set_index("metric")["mean"]
    show = {k: (round(float(av[k]), 4) if k in av.index else "n/a") for k in
            ("allergen_violation", "hard_constraint_violation", "constraint_satisfaction",
             "claim_grounding_rate", "unsupported_claim_rate",
             "faithfulness_judgeA", "correctness_judgeA")}
    print(f"[aggregate {cfg}] AGGREGATED {len(complete)}/{len(files)} runs -> {show}")


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--aggregate-only", action="store_true")
    ap.add_argument("--configs", default=None, help="comma-separated subset")
    args = ap.parse_args(argv)

    recipes_df = load_recipes_df(RECIPES_PATH)
    indexed = recipes_df.set_index("recipe_url", drop=False)
    qa = pickle.load(open(QA_PATH, "rb"))

    configs = args.configs.split(",") if args.configs else list(REQUIRE_EXPLANATION)

    if args.aggregate_only:
        for cfg in configs:
            reaggregate(cfg)
        return

    judges = judge_mod.build_judges()
    print("judges:", [(j.name, getattr(j.llm, "model", None)) for j in judges])

    if args.verify:
        verify(qa, indexed, judges)
        return

    for cfg in configs:
        for f in RUN_FILES[cfg]:
            p = CKPT / f
            if not p.exists():
                print(f"[{f}] MISSING — skip"); continue
            process_file(p, cfg, qa, indexed, judges, args.workers, args.limit)

    if args.limit is None:
        for cfg in configs:
            reaggregate(cfg)
    else:
        print(f"[sanity] --limit {args.limit} run; skipping aggregate.")


if __name__ == "__main__":
    main(sys.argv[1:])

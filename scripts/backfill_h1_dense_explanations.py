"""Targeted backfill of NL explanations onto the foodrag_h1_dense records so §8.2
claim-grounding (claim_grounding_rate / unsupported_claim_rate) can be computed for
that config (currently only foodrag_h1_t2p has it).

Why this exists
---------------
foodrag_h1_dense ran on pre-persist code, so its §13.1 records have no
``explanation_text`` — hence claim_grounding_rate / unsupported_claim_rate are NaN.
The final plan's recipes ARE recoverable: ``record["retrieved_ids"]`` is the final
plan's recipe_urls, so we reconstruct the plan (same ``urls_to_plan`` logic used in
``scripts/backfill_metrics.py`` and the driver's SeededH1Assembler) and generate a
faithful explanation of that final plan with a SINGLE LLM call per item — the
ExplanationTool only. No CorrectionTool / SubstitutionTool / FinalizationTool, and no
7GB HUMMUS KG load (ExplanationTool.explain never touches ``hummus_data``).

CAVEAT: the explanation is freshly generated on the reconstructed final plan
(retrieved_ids), NOT the exact as-run explanation (which was never persisted on the
pre-persist code). It is a faithful approximation for measuring the Explanation
Agent's grounding behaviour on the same final plan.

Idempotent + resumable: records that already carry a non-empty ``explanation_text``
are skipped, so a restart resumes. Legitimately-empty seeds (no retrieved_ids) get
``explanation_text = ""`` and their claim-grounding stays NaN (correct — nothing to
ground). Each per-run jsonl is rewritten atomically (temp file + os.replace).

Usage
-----
    python -m scripts.backfill_h1_dense_explanations                 # all 3 runs
    python -m scripts.backfill_h1_dense_explanations --limit 20      # sanity batch
    python -m scripts.backfill_h1_dense_explanations --workers 10
    python -m scripts.backfill_h1_dense_explanations --aggregate-only

The final step re-aggregates foodrag_h1_dense and (re)writes eval_results/aggregate.csv
+ the per-question parquet so they pick up the §8.2 keys.
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

from foodrag.baselines import constraints, explainability
from foodrag.baselines.runner import aggregate, load_recipes_df, write_results
from foodrag.config.settings import build_llm, load_models_config

CKPT = Path("data/qa/main_10k_tight/foodrag_ckpt")
QA_PATH = "data/qa/main_10k_tight/recipe_question_answer_set.pkl"
RECIPES_PATH = "data/hummus/recipes_df.csv"
CONFIG_ID = "foodrag_h1_dense"
RUN_FILES = [f"{CONFIG_ID}_run{r}.jsonl" for r in (1, 2, 3)]
N_EXPECTED = 1646
MEAL_PRIORITY = ("breakfast", "lunch", "dinner", "snack")


# ── plan reconstruction (copied verbatim from scripts/backfill_metrics.py) ────
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


# ── single-item explanation generation (1 LLM call) ──────────────────────────
def generate_explanation(question: str, plan_dict: dict, llm) -> str:
    """Seed a RecipeState with the reconstructed final plan and run ONLY the
    ExplanationTool. Same seeding as the driver's SeededH1Assembler, minus the
    correct/finalize chain. Returns the generated NL explanation string.

    ExplanationTool.explain uses only the seeded plan + original question + LLM
    (verified: it never reads ``hummus_data``), so we pass hummus_data=None — no
    HUMMUS KG load.
    """
    from foodrag.food_rag.meal_plan.meal_plan_structures import MealPlan, MealPlanQuestions
    from foodrag.food_rag.utils.recipe_context import RecipeState
    from foodrag.food_rag.agents.workflow_runner import get_dummy_state
    from foodrag.food_rag.tools.explanation_tools import ExplanationTool

    state = get_dummy_state()
    state[RecipeState.ORIGINAL_USER_QUESTION_KEY] = question
    state[RecipeState.MEAL_PLAN_RECOMMENDATION_KEY] = MealPlan.from_dict(plan_dict)
    state[RecipeState.MEAL_PLAN_QUESTIONS_KEY] = MealPlanQuestions.from_dict(
        {slot: question for slot in plan_dict})
    state.save_to_context()
    ctx = state.get_context()

    explanation = ExplanationTool(llm=llm, hummus_data=None, verbose=False).explain(ctx=ctx)
    return str(explanation or "")


def refresh_section8(rec: dict, persona: dict, rows: list, explanation_text: str) -> None:
    """Recompute the §8 explainability block in place. path_coverage / path_validity
    are unchanged (recipes are the same); claim_grounding_rate / unsupported_claim_rate
    now populate from the fresh explanation_text. Everything else in the record is
    left intact (add-only for explanation_text)."""
    block = rec["metrics"]
    block.update(explainability.explainability_metrics(
        persona, rows, explanation_text=explanation_text,
        expand=constraints.identity_expand))


def _has_explanation(rec: dict) -> bool:
    txt = rec.get("explanation_text")
    return isinstance(txt, str) and txt.strip() != ""


def process_record(rec: dict, qa, indexed, llm) -> tuple[str, str]:
    """Fill rec["explanation_text"] + refresh §8. Returns (status, sample) for logging.
    Never raises: a failed explanation leaves explanation_text unset (retried next run).
    """
    i = int(rec["question_id"].split("_")[1])
    persona = qa[i]["persona"]
    urls = rec.get("retrieved_ids") or []
    rows = _resolve(urls, indexed)
    plan_dict = urls_to_plan(urls, indexed)

    if not plan_dict:
        # legit-empty seed → empty explanation; claim-grounding stays NaN (correct)
        rec["explanation_text"] = ""
        refresh_section8(rec, persona, rows, "")
        return ("empty", "")

    question = qa[i].get("question", "")
    explanation = generate_explanation(question, plan_dict, llm)
    rec["explanation_text"] = explanation
    refresh_section8(rec, persona, rows, explanation)
    cg = rec["metrics"].get("claim_grounding_rate")
    return ("ok", f"cg={cg} | {explanation[:120]!r}")


def _rewrite(path: Path, records) -> None:
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")
    os.replace(tmp, path)


def process_file(path: Path, qa, indexed, llm, workers: int, limit: int | None) -> None:
    records = [json.loads(l) for l in path.open(encoding="utf-8")]
    todo = [(idx, r) for idx, r in enumerate(records) if not _has_explanation(r)]
    if limit is not None:
        todo = todo[:limit]
    if not todo:
        print(f"[{path.name}] nothing to do (all {len(records)} records already have explanation_text)")
        return
    print(f"[{path.name}] {len(todo)} records to process "
          f"({len(records) - len(todo)} already done) with {workers} workers")

    lock = threading.Lock()
    done = {"n": 0, "empty": 0, "err": 0}
    last_checkpoint = {"n": 0}

    def _work(item):
        idx, rec = item
        try:
            status, sample = process_record(rec, qa, indexed, llm)
            return (idx, status, sample, None)
        except Exception as e:  # pragma: no cover — leave explanation unset, continue
            return (idx, "error", "", f"{type(e).__name__}: {e}\n{traceback.format_exc()}")

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_work, it) for it in todo]
        for fut in as_completed(futures):
            idx, status, sample, err = fut.result()
            with lock:
                done["n"] += 1
                n = done["n"]
                if status == "empty":
                    done["empty"] += 1
                if status == "error":
                    done["err"] += 1
                    print(f"[{path.name}] ERROR on record {idx}: {err.splitlines()[0]}")
                elif n <= 5 or n % 200 == 0:
                    print(f"[{path.name}] {n}/{len(todo)} {status} :: {sample}")
                # checkpoint every 250 completed items (atomic rewrite of the whole file)
                if n - last_checkpoint["n"] >= 250:
                    _rewrite(path, records)
                    last_checkpoint["n"] = n
                    print(f"[{path.name}] checkpoint @ {n}/{len(todo)}")

    _rewrite(path, records)
    print(f"[{path.name}] DONE: {done['n']} processed "
          f"({done['empty']} legit-empty, {done['err']} errors, rewritten)")


def reaggregate(indexed) -> None:
    """Re-aggregate foodrag_h1_dense across all 3 runs and (re)write eval_results."""
    complete = []
    for f in RUN_FILES:
        p = CKPT / f
        n = sum(1 for _ in p.open(encoding="utf-8")) if p.exists() else 0
        if n == N_EXPECTED:
            complete.append(p)
        else:
            print(f"[aggregate] {f}: {n}/{N_EXPECTED} lines — not complete")
    if len(complete) != len(RUN_FILES):
        print(f"[aggregate] only {len(complete)}/{len(RUN_FILES)} runs complete — skipping aggregate")
        return
    all_records = []
    for p in complete:
        all_records.extend(json.loads(l) for l in p.open(encoding="utf-8"))
    agg = aggregate(all_records)
    write_results(all_records, agg, out_dir="eval_results", config_id=CONFIG_ID)
    av = agg[agg.stratum == "aggregate"].set_index("metric")["mean"]
    show = {k: (round(float(av[k]), 4) if k in av.index else "n/a") for k in
            ("path_coverage", "path_validity", "claim_grounding_rate", "unsupported_claim_rate")}
    # count how many records actually contributed a claim-grounding value
    n_expl = sum(1 for r in all_records if _has_explanation(r))
    n_cg = sum(1 for r in all_records
               if isinstance(r["metrics"].get("claim_grounding_rate"), (int, float))
               and r["metrics"]["claim_grounding_rate"] == r["metrics"]["claim_grounding_rate"])
    print(f"[aggregate] {CONFIG_ID} AGGREGATED {len(complete)}/{len(RUN_FILES)} runs -> {show}")
    print(f"[aggregate] {n_expl}/{len(all_records)} records have explanation_text; "
          f"{n_cg}/{len(all_records)} contribute a (non-NaN) claim_grounding_rate")


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--limit", type=int, default=None,
                    help="process at most N records per file (sanity batch)")
    ap.add_argument("--aggregate-only", action="store_true",
                    help="skip generation, just re-aggregate from existing jsonl")
    args = ap.parse_args(argv)

    recipes_df = load_recipes_df(RECIPES_PATH)
    indexed = recipes_df.set_index("recipe_url", drop=False)
    qa = pickle.load(open(QA_PATH, "rb"))

    if args.aggregate_only:
        reaggregate(indexed)
        return

    llm = build_llm(config=load_models_config())
    for f in RUN_FILES:
        p = CKPT / f
        if not p.exists():
            print(f"[{f}] MISSING — skip")
            continue
        process_file(p, qa, indexed, llm, args.workers, args.limit)

    # Only aggregate on a full pass (a --limit run is a sanity batch, not complete)
    if args.limit is None:
        reaggregate(indexed)
    else:
        print(f"[sanity] --limit {args.limit} run complete; skipping aggregate. "
              f"Re-run without --limit to finish + aggregate.")


if __name__ == "__main__":
    main(sys.argv[1:])

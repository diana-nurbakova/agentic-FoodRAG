"""EXP-5 (Realistic tier): KG-retrieval evaluation on the TIGHT stratum.

Per the Evaluation Protocol §5.5, KG retrieval is only evaluated where the valid
set is small (1-50 answers) so the Text-to-Cypher traversal is bounded and
performant; broad-stratum KG retrieval remains infeasible without precomputation.
This runner reports Hit@10 / MRR@10 / latency for a sample of the tight stratum
and appends the aggregates to ``eval_results/aggregate.csv`` (real, not synthetic).

    python scripts/eval_kg_tight.py --qa small_1 --sample 12 --depth 10

Heavy: loads HUMMUS (~200s) and makes one Text-to-Cypher LLM call per question.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

AGG_COLS = ["experiment_id", "config_id", "stratum", "metric", "mean", "ci_low", "ci_high",
            "per_question_variance", "n_questions", "n_runs", "judge"]


def _mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def _variance(xs):
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    return sum((x - m) ** 2 for x in xs) / (len(xs) - 1)


def _ci(xs):
    """Normal-approx 95% CI on the mean (sample is small; documented as such)."""
    n = len(xs)
    if n < 2:
        m = _mean(xs)
        return m, m
    sem = math.sqrt(_variance(xs) / n)
    m = _mean(xs)
    return max(0.0, m - 1.96 * sem), m + 1.96 * sem


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--qa", default="small_1", help="QA sub-dir under data/qa")
    ap.add_argument("--sample", type=int, default=12, help="number of tight-stratum questions")
    ap.add_argument("--depth", type=int, default=10, help="KG retriever similarity_top_k")
    ap.add_argument("--out", default=str(REPO / "eval_results" / "aggregate.csv"))
    args = ap.parse_args(argv)

    from ui.services import backend
    from foodrag.food_rag.utils import qa_helper
    from foodrag.persona import qa_generator

    print(f"[1/4] building engine (loads HUMMUS)…", flush=True)
    t0 = time.time()
    engine = backend.get_engine()
    data_path = str(REPO / "data")
    print(f"      engine ready in {time.time() - t0:.0f}s; df={engine.recipes_df.shape}", flush=True)

    print(f"[2/4] loading QA set '{args.qa}' + selecting tight stratum (1-50 answers)…", flush=True)
    # load_qa_set maps answers via 'recipe_url' → use the ORIGINAL df (the cleaned
    # engine df drops recipe_url). recipe_id is present on both.
    orig_df = engine.hummus_data["recipes_df"]
    qas = qa_helper.get_best_questions(
        qa_generator.load_qa_set(data_path, sub_dir_name=args.qa, recipes_df=orig_df)
        ["recipe_question_answer_set"], discard_empty_qas=True)
    # get_best_questions returns a dict keyed by index → iterate values
    all_qas = list(qas.values()) if isinstance(qas, dict) else list(qas)
    tight = [q for q in all_qas if 1 <= len(q["answer"]) <= 50][: args.sample]
    print(f"      {len(tight)} tight questions (of {len(all_qas)} total)", flush=True)
    if not tight:
        print("SMOKE_FAIL: no tight-stratum questions", flush=True)
        return 1

    kg = backend._build_kg_retriever(engine, top_k=args.depth)
    if kg is None:
        print("SMOKE_FAIL: KG retriever unavailable", flush=True)
        return 1

    print(f"[3/4] running KG (Text-to-Cypher) over {len(tight)} questions…", flush=True)
    hits, rrs, lats = [], [], []
    for i, q in enumerate(tight):
        valid = set(q["answer"])
        t = time.time()
        try:
            df = kg.retrieve(q["question"])
            n_ret = 0 if df is None else len(df)
            ids = df["recipe_id"].tolist()[:10] if n_ret else []
            if i == 0:
                titles = list(df["title"])[:3] if n_ret and "title" in df.columns else []
                print(f"      [debug q0] retrieved {n_ret} rows; sample titles={titles}", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"      q{i}: retrieve error {type(exc).__name__}: {exc}", flush=True)
            ids = []
        lat = time.time() - t
        rank = next((j + 1 for j, rid in enumerate(ids) if rid in valid), None)
        hits.append(1.0 if rank else 0.0)
        rrs.append(1.0 / rank if rank else 0.0)
        lats.append(lat)
        print(f"      q{i}: |valid|={len(valid)} hit={hits[-1]:.0f} rr={rrs[-1]:.2f} {lat:.1f}s", flush=True)

    n = len(tight)
    metrics = {"hit_at_10": hits, "mrr_at_10": rrs, "latency_s": lats}
    print(f"[4/4] tight-stratum: Hit@10={_mean(hits):.3f} MRR@10={_mean(rrs):.3f} "
          f"latency={_mean(lats):.1f}s (n={n})", flush=True)

    # Append/replace EXP-5 rows in aggregate.csv
    out = Path(args.out)
    rows = []
    if out.exists():
        with open(out, encoding="utf-8") as fh:
            rows = [r for r in csv.DictReader(fh) if r.get("experiment_id") != "EXP-5"]
    for metric, xs in metrics.items():
        lo, hi = _ci(xs)
        rows.append({"experiment_id": "EXP-5", "config_id": "kg_text_to_cypher", "stratum": "tight",
                     "metric": metric, "mean": round(_mean(xs), 3), "ci_low": round(lo, 3),
                     "ci_high": round(hi, 3), "per_question_variance": round(_variance(xs), 4),
                     "n_questions": n, "n_runs": 1, "judge": ""})
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=AGG_COLS)
        w.writeheader()
        w.writerows(rows)
    print(f"      wrote {sum(r['experiment_id']=='EXP-5' for r in rows)} EXP-5 rows → {out}", flush=True)

    # Mark EXP-5 as real in the manifest
    man_path = out.parent / "manifest.json"
    man = {}
    if man_path.exists():
        man = json.loads(man_path.read_text(encoding="utf-8"))
    real = set(man.get("real_experiments", []))
    real.add("EXP-5")
    man["real_experiments"] = sorted(real)
    man["exp5_note"] = (f"EXP-5 KG retrieval: real run, tight stratum only, n={n} "
                        f"({args.qa}). Broad-stratum KG retrieval is infeasible without "
                        f"precomputation (protocol §5.5).")
    man_path.write_text(json.dumps(man, indent=2), encoding="utf-8")
    print("KG_EVAL_OK", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

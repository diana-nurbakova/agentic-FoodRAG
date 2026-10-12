"""Audit the 421 re-judged empty-plan records (74176e8): what the judges saw, before vs after.

The §13 records never stored the judge input. The LLM call logs did: every judge
prompt is logged verbatim. For each ``foodrag_h1_t2p`` record that
``apply_safety_fixes.py`` re-judged as an empty-plan record (pre-fix: no
``retrieved_ids`` but a non-empty ``explanation_text``), this recovers

  * the ORIGINAL judge input — ``data/logs/llm_calls-2026-07-04.jsonl`` (the
    H1-t2p run, 4 Jul 20:24Z – 5 Jul 22:45Z), matched on the QA question text and
    the record's pre-fix explanation; and
  * the RE-JUDGE input — ``data/logs/llm_calls-2026-07-06.jsonl``, matched on the
    gold-title set (the re-judge prompts carry no question; see below),

then compares question, gold, contexts and answer field by field, and checks which
logged responses reproduce the stored pre- and post-fix scores.

    python scripts/audit_rejudge_inputs.py            # -> audit/rejudge_inputs_421.json (+ .csv)
"""
from __future__ import annotations

import argparse
import io
import json
import pickle
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
QA = REPO / "data/qa/main_10k_tight/recipe_question_answer_set.pkl"
LOG_ORIG = REPO / "data/logs/llm_calls-2026-07-04.jsonl"
LOG_REJ = REPO / "data/logs/llm_calls-2026-07-06.jsonl"
PARQUET = "eval_results/foodrag_h1_t2p_per_question.parquet"
EMPTY_MSG = "No meal plan could be recommended that satisfies the given requirements."

PATTERNS = {
    "correctness": re.compile(r"User question: (?P<q>.*?)\nReference \(gold\) answer recipes: (?P<gold>.*?)\n\n"
                              r"Recommendation:\n(?P<a>.*)\n\nRate how correct", re.S),
    "relevancy": re.compile(r"User question: (?P<q>.*?)\nRetrieved recipes:\n(?P<ctx>.*?)\n\n"
                            r"Recommendation:\n(?P<a>.*)\n\nDoes the recommendation fit", re.S),
    "faithfulness": re.compile(r"Retrieved recipes available to the system:\n(?P<ctx>.*?)\n\n"
                               r"Recommendation:\n(?P<a>.*)\n\nDoes the recommendation use ONLY", re.S),
    "explanation_quality": re.compile(r"User question: (?P<q>.*?)\nRecommendation \(with explanation\):\n"
                                      r"(?P<a>.*)\n\nRate the explanation", re.S),
}
JUDGE = {"gemma-3-27b-it": "judgeA", "DeepSeek-V3": "judgeB"}


def parquet_at(rev: str) -> pd.DataFrame:
    raw = subprocess.run(["git", "show", f"{rev}:{PARQUET}"], cwd=REPO, capture_output=True, check=True).stdout
    return pd.read_parquet(io.BytesIO(raw))


def response_text(resp) -> str:
    if isinstance(resp, dict):
        for k in ("text", "content", "message"):
            if k in resp:
                return str(resp[k])
    return str(resp or "")


def score(metric: str, text: str):
    m = re.search(r'"passing"\s*:\s*(true|false)', text, re.I)
    if metric in ("faithfulness", "relevancy"):
        return None if not m else (1.0 if m.group(1).lower() == "true" else 0.0)
    m = re.search(r'"score"\s*:\s*"?([0-9.]+)', text)
    return None if not m else max(0.0, min(1.0, (float(m.group(1)) - 1) / 4))


def gold_set(text: str) -> frozenset:
    return frozenset(t.strip() for t in text.split(", ") if t.strip()) if text.strip() != "Not provided." else frozenset()


def scan(path: Path, keep) -> list[dict]:
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if "meal-plan recommendation" not in line:
                continue
            try:
                r = json.loads(line)
            except ValueError:
                continue
            judge = JUDGE.get(str(r.get("model", "")).split("/")[-1])
            if judge is None:
                continue
            p = r.get("prompt") or ""
            for metric, pat in PATTERNS.items():
                m = pat.search(p)
                if m:
                    rec = {"metric": metric, "judge": judge, "ts": r.get("ts"), **m.groupdict(), "prompt_full": p,
                           "response": response_text(r.get("response"))}
                    if keep(rec):
                        out.append(rec)
                    break
    return out


def classify_answer(a: str) -> dict:
    return {"chars": len(a), "is_empty_msg": a.strip() == EMPTY_MSG,
            "has_meal_plan_block": "Your meal plan:" in a or "Meal plan:" in a,
            "names_recipe_fields": bool(re.search(r"\b(title|ingredient_quantities|directions)\b", a))}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "audit" / "rejudge_inputs_421.json"))
    args = ap.parse_args(argv)

    with open(QA, "rb") as fh:
        qa = pickle.load(fh)
    pre = parquet_at("74176e8~1")
    post = parquet_at("74176e8")
    target = pre[pre.retrieved_ids.apply(lambda x: len(x) == 0)
                 & pre.explanation_text.fillna("").str.strip().ne("")].copy()
    print(f"pre-fix empty-plan records with an explanation: {len(target)}", file=sys.stderr)
    post_idx = post.set_index(["run", "question_id"])

    questions = {qa[int(q.split('_')[1])]["question"] for q in target.question_id}
    orig = scan(LOG_ORIG, lambda r: r.get("q", "").strip() in questions or r["metric"] == "faithfulness")
    rej = scan(LOG_REJ, lambda r: True)
    print(f"logged judge prompts: original {len(orig)}, re-judge {len(rej)}", file=sys.stderr)

    by_q = defaultdict(list)
    for r in orig:
        if r["metric"] != "faithfulness":
            by_q[r["q"].strip()].append(r)

    rows, agg = [], Counter()
    for t in target.itertuples():
        i = int(t.question_id.split("_")[1])
        item = qa[i]
        q = item["question"]
        expl = str(t.explanation_text).strip()
        probe = expl[:200]
        o = [r for r in by_q.get(q, []) if probe in r["a"]]
        o_corr = {r["judge"]: r for r in o if r["metric"] == "correctness"}
        row = {"run": int(t.run), "question_id": t.question_id, "question": q}
        oc = o_corr.get("judgeA")
        row["orig_found"] = oc is not None
        if oc is not None:
            row["orig_question_in_prompt"] = oc["q"].strip()
            row["orig_answer"] = oc["a"]
            row.update({f"orig_answer_{k}": v for k, v in classify_answer(oc["a"]).items()})
            row["orig_explanation_is_prefix_of_answer"] = oc["a"].strip().startswith(expl[:200])
            row["orig_gold_n"] = len(gold_set(oc["gold"]))
            for j in ("judgeA", "judgeB"):
                r = o_corr.get(j)
                row[f"orig_corr_{j}_logged"] = score("correctness", r["response"]) if r else None
                row[f"orig_corr_{j}_stored"] = t.metrics.get(f"correctness_{j}")
        pm = post_idx.loc[(t.run, t.question_id)].metrics
        for j in ("judgeA", "judgeB"):
            row[f"post_corr_{j}_stored"] = pm.get(f"correctness_{j}")
        rows.append(row)
        agg["records"] += 1
        agg["orig_found"] += row["orig_found"]

    # Re-judge side, aggregated: every fixed-message correctness prompt, by field.
    rej_msg = [r for r in rej if r["metric"] == "correctness" and r["a"].strip() == EMPTY_MSG]
    rej_summary = {
        "fixed_message_correctness_prompts": dict(Counter(r["judge"] for r in rej_msg)),
        "question_field_empty": dict(Counter(r["judge"] for r in rej_msg if r["q"].strip() == "")),
        "answer_exactly_fixed_message": len(rej_msg),
        "all_rejudge_prompts_by_metric_and_question": {
            f"{m}|{'empty' if not r_q else 'present'}": n
            for (m, r_q), n in Counter((r["metric"], bool(r.get("q", "x").strip())) for r in rej).items()},
        "rejudge_score_distribution_judgeA": dict(Counter(
            score("correctness", r["response"]) for r in rej_msg if r["judge"] == "judgeA")),
        "rejudge_score_distribution_judgeB": dict(Counter(
            score("correctness", r["response"]) for r in rej_msg if r["judge"] == "judgeB")),
        "ts_range": [min(r["ts"] for r in rej_msg), max(r["ts"] for r in rej_msg)] if rej_msg else None,
    }
    df = pd.DataFrame(rows)
    found = df[df.orig_found]
    summary = {
        "records": int(len(df)), "original_prompt_recovered": int(df.orig_found.sum()),
        "original_side": {
            "question_present": int((found.orig_question_in_prompt.str.len() > 0).sum()),
            "answer_is_fixed_message": int(found.orig_answer_is_empty_msg.sum()),
            "answer_starts_with_stored_explanation": int(found.orig_explanation_is_prefix_of_answer.sum()),
            "answer_has_meal_plan_block": int(found.orig_answer_has_meal_plan_block.sum()),
            "answer_chars_median": float(found.orig_answer_chars.median()),
            "logged_score_matches_stored_judgeA": int((found.orig_corr_judgeA_logged == found.orig_corr_judgeA_stored).sum()),
            "logged_score_matches_stored_judgeB": int((found.orig_corr_judgeB_logged == found.orig_corr_judgeB_stored).sum()),
            "stored_mean_correctness_judgeA": float(found.orig_corr_judgeA_stored.mean()),
        },
        "rejudge_side": rej_summary,
        "post_fix_stored_mean_correctness_judgeA": float(df.post_corr_judgeA_stored.mean()),
        "post_fix_stored_mean_correctness_judgeB": float(df.post_corr_judgeB_stored.mean()),
    }
    out = Path(args.out)
    out.write_text(json.dumps({"summary": summary}, indent=2, default=str), encoding="utf-8")
    df.drop(columns=["orig_answer"], errors="ignore").to_csv(out.with_suffix(".csv"), index=False)
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""E-7 — re-judge the 421 empty-plan items, question × answer (specs/E7-rejudge-2x2-spec.md).

    C1  question present · original final recommendation   (drift anchor vs stored 4–5 Jul)
    C2  question present · fixed message
    C3  question empty   · original final recommendation
    C4  question empty   · fixed message                    (drift anchor vs stored 6 Jul)

All cells in one session, Judges A and B, the frozen ``judge.py`` templates, and the
module's own parsing (``judge._scaled`` / ``judge._binary`` applied to the raw reply),
with "No recipes were retrieved." as the retrieved-recipe text. The original answer
and gold text come from the recovered 4–5 Jul prompt (scripts/audit_rejudge_inputs.py)
and are held fixed across cells; only question and answer vary. Every C1 prompt is
rebuilt from the template and must equal the recovered prompt byte for byte.

    python scripts/run_e7_rejudge.py --metric correctness            # required
    python scripts/run_e7_rejudge.py --metric relevancy --run-id <id> # optional, same session
    python scripts/run_e7_rejudge.py --analyse <run_id>
"""
from __future__ import annotations

import argparse
import json
import pickle
import re
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import audit_rejudge_inputs as rec  # noqa: E402

OUT_ROOT = REPO / "eval_results_e7_rejudge"
CELLS = {"C1": (True, "orig"), "C2": (True, "msg"), "C3": (False, "orig"), "C4": (False, "msg")}
NO_CTX = "No recipes were retrieved."
N_BOOT, SEED = 10_000, 0


def build_items() -> list[dict]:
    """The 421 records with their recovered original question, gold text and answer."""
    with open(rec.QA, "rb") as fh:
        qa = pickle.load(fh)
    pre = rec.parquet_at("74176e8~1")
    post = rec.parquet_at("74176e8").set_index(["run", "question_id"])
    target = pre[pre.retrieved_ids.apply(len).eq(0) & pre.explanation_text.fillna("").str.strip().ne("")]
    questions = {qa[int(q.split("_")[1])]["question"] for q in target.question_id}
    logged = rec.scan(rec.LOG_ORIG, lambda r: r.get("q", "").strip() in questions)
    by_q: dict[str, list] = {}
    for r in logged:
        by_q.setdefault(r["q"].strip(), []).append(r)
    items = []
    for t in target.itertuples():
        q = qa[int(t.question_id.split("_")[1])]["question"]
        probe = str(t.explanation_text).strip()[:200]
        hits = [r for r in by_q.get(q, []) if probe in r["a"]]
        corr = {r["judge"]: r for r in hits if r["metric"] == "correctness"}
        rel = {r["judge"]: r for r in hits if r["metric"] == "relevancy"}
        if "judgeA" not in corr:
            raise SystemExit(f"no recovered original prompt for {t.run}/{t.question_id}")
        base = corr["judgeA"]
        pm = post.loc[(t.run, t.question_id)].metrics
        items.append({
            "key": f"r{t.run}_{t.question_id}", "run": int(t.run), "question_id": t.question_id,
            "question": base["q"], "gold": base["gold"], "orig_answer": base["a"],
            "stored": {"C1": {j: t.metrics.get(f"correctness_{j}") for j in ("judgeA", "judgeB")},
                       "C4": {j: pm.get(f"correctness_{j}") for j in ("judgeA", "judgeB")},
                       "C1_relevancy": {j: t.metrics.get(f"relevancy_{j}") for j in ("judgeA", "judgeB")},
                       "C4_relevancy": {j: pm.get(f"relevancy_{j}") for j in ("judgeA", "judgeB")}},
            "recovered_prompts": {"correctness": {j: r["prompt_full"] for j, r in corr.items()},
                                  "relevancy": {j: r["prompt_full"] for j, r in rel.items()}},
        })
    return items


def prompt_for(metric: str, item: dict, cell: str) -> str:
    from foodrag.baselines import judge as J

    has_q, ans = CELLS[cell]
    question = item["question"] if has_q else ""
    answer = item["orig_answer"] if ans == "orig" else rec.EMPTY_MSG
    if metric == "correctness":
        return J.CORRECTNESS_PROMPT.format(question=question, gold=item["gold"], answer=answer)
    return J.RELEVANCY_PROMPT.format(question=question, contexts=NO_CTX, answer=answer)


class _Canned:
    def __init__(self, text):
        self.text = text

    def complete(self, prompt):
        return self


def parse(metric: str, text: str) -> float:
    """The pipeline's own parse of a judge reply (judge._scaled / judge._binary)."""
    from foodrag.baselines import judge as J

    fn = J._scaled if metric == "correctness" else J._binary
    return fn(J.Judge("x", _Canned(text)), "")


def call(judge, prompt: str) -> dict:
    t0 = time.perf_counter()
    try:
        resp = judge.llm.complete(prompt)
        text, err = (resp.text or "").strip(), None
        raw = getattr(resp, "raw", None) or {}
        model = raw.get("model") if isinstance(raw, dict) else getattr(raw, "model", None)
    except Exception as exc:  # recorded, not retried beyond the LLM's own policy
        text, err, model = "", f"{type(exc).__name__}: {str(exc)[:300]}", None
    return {"text": text, "error": err, "provider_model": model, "latency_s": time.perf_counter() - t0}


def run(metric: str, run_id: str | None, workers: int) -> str:
    from foodrag.baselines.judge import build_judges
    from foodrag.config import settings

    settings.load_env()
    items = build_items()
    # Template check: C1 rebuilt == the recovered 4–5 Jul prompt, byte for byte.
    mism = [(it["key"], j) for it in items for j, p in it["recovered_prompts"][metric].items()
            if prompt_for(metric, it, "C1").strip() != p.strip()]
    if mism:
        raise SystemExit(f"C1 template mismatch on {len(mism)} prompts, e.g. {mism[:3]}")
    judges = build_judges()
    run_id = run_id or f"e7_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    out = OUT_ROOT / run_id
    out.mkdir(parents=True, exist_ok=True)
    (out / "items.json").write_text(json.dumps([{k: v for k, v in it.items() if k != "recovered_prompts"}
                                                for it in items], indent=1), encoding="utf-8")
    meta_path = out / "run_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {
        "run_id": run_id, "spec": "specs/E7-rejudge-2x2-spec.md", "items": len(items),
        "commit": rec.subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True,
                                     text=True).stdout.strip(),
        "judges": {j.name: j.llm.model for j in judges},
        "decoding": {j.name: {"temperature": j.llm.temperature} for j in judges},
        "phases": {}}
    meta["phases"][metric] = {"started": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                              "template_check": f"C1 == recovered prompt for all "
                                                f"{sum(len(it['recovered_prompts'][metric]) for it in items)} "
                                                f"recovered {metric} prompts"}
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    ckpt = out / f"calls_{metric}.jsonl"
    done = set()
    if ckpt.exists():
        done = {(r["key"], r["cell"], r["judge"]) for r in map(json.loads, ckpt.read_text(encoding="utf-8").splitlines())}
    jobs = [(it, cell, j) for it in items for cell in CELLS for j in judges
            if (it["key"], cell, j.name) not in done]
    lock, n, t0 = threading.Lock(), 0, time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as pool, open(ckpt, "a", encoding="utf-8") as fh:
        futs = {pool.submit(call, j, prompt_for(metric, it, cell)): (it, cell, j) for it, cell, j in jobs}
        for fut in as_completed(futs):
            it, cell, j = futs[fut]
            r = fut.result()
            score = parse(metric, r["text"]) if r["error"] is None else float("nan")
            row = {"key": it["key"], "cell": cell, "judge": j.name, "metric": metric,
                   "score": None if score != score else score, **r,
                   "stray_token_nan": score != score and bool(re.search(
                       r'"(score|passing)"\s*:', r["text"])) and r["error"] is None,
                   "ts": datetime.now(timezone.utc).isoformat(timespec="seconds")}
            with lock:
                fh.write(json.dumps(row) + "\n")
                fh.flush()
            n += 1
            if n % 200 == 0 or n == len(jobs):
                print(f"[{metric}] {n}/{len(jobs)} {time.perf_counter() - t0:.0f}s", flush=True)
    meta["phases"][metric]["finished"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return run_id


def _ci(x) -> dict:
    x = np.asarray([v for v in x if v is not None and v == v], dtype=float)
    if not len(x):
        return {"mean": None, "lo": None, "hi": None, "n": 0}
    b = x[np.random.default_rng(SEED).integers(0, len(x), (N_BOOT, len(x)))].mean(axis=1)
    return {"mean": float(x.mean()), "lo": float(np.percentile(b, 2.5)), "hi": float(np.percentile(b, 97.5)),
            "n": int(len(x))}


def analyse(run_id: str) -> dict:
    out = OUT_ROOT / run_id
    items = {it["key"]: it for it in json.loads((out / "items.json").read_text(encoding="utf-8"))}
    res = {"run_id": run_id, "spec": "specs/E7-rejudge-2x2-spec.md",
           "naming": "a follow-up re-judging of the 421 items — never 'corrected scores'",
           "metrics": {}}
    for ckpt in sorted(out.glob("calls_*.jsonl")):
        metric = ckpt.stem.split("_", 1)[1]
        df = pd.DataFrame(map(json.loads, ckpt.read_text(encoding="utf-8").splitlines()))
        m = {"calls": len(df), "per_judge": {}}
        for j, g in df.groupby("judge"):
            wide = g.pivot_table(index="key", columns="cell", values="score", aggfunc="first")
            wide = wide.reindex(columns=list(CELLS))
            cells = {c: {**_ci(wide[c].tolist()),
                         "nan": int(g[g.cell == c].score.isna().sum()),
                         "api_errors": int(g[(g.cell == c) & g.error.notna()].shape[0]),
                         "stray_token_nan": int(g[(g.cell == c) & g.stray_token_nan].shape[0])} for c in CELLS}

            def eff(a, b):
                d = (wide[a] - wide[b]).dropna()
                return {**_ci(d.tolist()), "n_paired": int(len(d))}

            inter = ((wide.C2 - wide.C1) - (wide.C4 - wide.C3)).dropna()
            skey = "C1" if metric == "correctness" else "C1_relevancy"
            stored_c1 = pd.Series({k: items[k]["stored"][skey][j] for k in wide.index}, dtype=float)
            stored_c4 = pd.Series({k: items[k]["stored"]["C4" if metric == "correctness" else "C4_relevancy"][j]
                                   for k in wide.index}, dtype=float)
            drift = {"C1_today_minus_stored": float(wide.C1.mean() - stored_c1.mean()),
                     "C4_today_minus_stored": float(wide.C4.mean() - stored_c4.mean()),
                     "stored_C1_mean": float(stored_c1.mean()), "stored_C4_mean": float(stored_c4.mean())}
            drift["exceeds_0.10"] = any(abs(v) > 0.10 for k, v in drift.items() if k.endswith("_stored"))
            m["per_judge"][j] = {
                "cells": cells,
                "question_effect_C2_minus_C4": eff("C2", "C4"),
                "answer_effect_C2_minus_C1": eff("C2", "C1"),
                "answer_effect_without_question_C4_minus_C3": eff("C4", "C3"),
                "interaction_(C2-C1)-(C4-C3)": {**_ci(inter.tolist()), "n_paired": int(len(inter))},
                "drift_check": drift,
                "provider_models": dict(Counter(g.provider_model.fillna("unreported"))),
            }
        res["metrics"][metric] = m
    (out / "e7_results.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    (out / "e7_results.md").write_text(render(res), encoding="utf-8")
    print(render(res))
    return res


def render(res: dict) -> str:
    f = lambda x: "—" if x is None else f"{x:.3f}"  # noqa: E731
    c = lambda d: f"{f(d['mean'])} [{f(d['lo'])}, {f(d['hi'])}]"  # noqa: E731
    L = [f"# E-7 — follow-up re-judging of the 421 empty-plan items (`{res['run_id']}`)", "",
         "One session; Judges A and B; frozen templates; pipeline parser. Means with bootstrap 95% CIs over "
         "items (10,000 resamples, seed 0). Descriptive only; no significance claims across judges.", ""]
    for metric, m in res["metrics"].items():
        L += [f"## {metric}", "", "| Judge | C1 q+orig | C2 q+msg | C3 −q+orig | C4 −q+msg |", "|---|---|---|---|---|"]
        for j, v in m["per_judge"].items():
            L.append(f"| {j} | " + " | ".join(f"{c(v['cells'][k])} (NaN {v['cells'][k]['nan']}, stray "
                                               f"{v['cells'][k]['stray_token_nan']})" for k in CELLS) + " |")
        L += ["", "| Judge | Question effect C2−C4 | Answer effect C2−C1 | C4−C3 | Interaction | "
                  "Drift C1 / C4 (today − stored) |", "|---|---|---|---|---|---|"]
        for j, v in m["per_judge"].items():
            d = v["drift_check"]
            L.append(f"| {j} | {c(v['question_effect_C2_minus_C4'])} | {c(v['answer_effect_C2_minus_C1'])} | "
                     f"{c(v['answer_effect_without_question_C4_minus_C3'])} | {c(v['interaction_(C2-C1)-(C4-C3)'])} | "
                     f"{d['C1_today_minus_stored']:+.3f} / {d['C4_today_minus_stored']:+.3f}"
                     f"{' ⚠ >0.10: effects are session-specific' if d['exceeds_0.10'] else ''} |")
        L.append("")
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", choices=("correctness", "relevancy"))
    ap.add_argument("--run-id")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--analyse")
    args = ap.parse_args(argv)
    if args.analyse:
        analyse(args.analyse)
        return 0
    rid = run(args.metric, args.run_id, args.workers)
    print("RUN_ID", rid)
    analyse(rid)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

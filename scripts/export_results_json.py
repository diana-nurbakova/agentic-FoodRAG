"""Export evaluation results to JSON:
  - eval_results/json/<config>_per_question.json : every per-question.1 record
    for that config (each record tagged with its ``run``), for all configs.
  - eval_results/json/aggregate.json : the.2 aggregate records
    (config x stratum x metric) INCLUDING bootstrap 95% CIs (ci_low/ci_high),
    per-question variance, n_questions and n_runs.

Sources: the per-question Parquet files under eval_results/ and eval_results_llama/,
and the two aggregate.csv files. Idempotent — safe to re-run after a config is
updated (e.g. after the h1_dense explanation pass adds claim grounding).

Usage: python -m scripts.export_results_json
"""
from __future__ import annotations

import glob
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

os.chdir(Path(__file__).resolve().parents[1])
OUT = Path("eval_results/json")
OUT.mkdir(parents=True, exist_ok=True)


def _clean(o):
    """Recursively make a value JSON-serialisable (numpy scalars/arrays, NaN->None)."""
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, np.ndarray)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, float):
        return None if np.isnan(o) else o
    return o


def _safe_name(cid: str) -> str:
    return cid.replace("/", "_").replace("@", "_at_")


# ---- per-question records, one JSON per config ----
# Prefer Parquet; fall back to the .pkl a config was written as when its records
# have a column the Parquet engine rejects (write_results falls back to pickle).
parquets = [p for p in sorted(glob.glob("eval_results/*_per_question.parquet")) +
            sorted(glob.glob("eval_results_llama/*_per_question.parquet"))
            if os.path.getsize(p) > 0]  # skip 0-byte turds from a failed to_parquet
_covered = {Path(p).name[: -len("_per_question.parquet")] for p in parquets}
pkls = [p for p in sorted(glob.glob("eval_results/*_per_question.pkl")) +
        sorted(glob.glob("eval_results_llama/*_per_question.pkl"))
        if Path(p).name[: -len("_per_question.pkl")] not in _covered]
index = []
for pq in parquets + pkls:
    df = pd.read_pickle(pq) if pq.endswith(".pkl") else pd.read_parquet(pq)
    cid = str(df["config_id"].iloc[0]) if "config_id" in df.columns and len(df) else Path(pq).stem
    recs = [_clean(r) for r in df.to_dict(orient="records")]
    # sort by (run, question_id) so the file reads run-by-run
    recs.sort(key=lambda r: (r.get("run", 0), str(r.get("question_id", ""))))
    path = OUT / f"{_safe_name(cid)}_per_question.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(recs, f, ensure_ascii=False, indent=1)
    n_runs = len({r.get("run") for r in recs})
    index.append({"config_id": cid, "file": path.name, "n_records": len(recs), "n_runs": n_runs})
    print(f"{cid}: {len(recs)} records ({n_runs} runs) -> {path.name}")

# ---- aggregate records WITH confidence intervals ----
aggs = [pd.read_csv("eval_results/aggregate.csv")]
if os.path.exists("eval_results_llama/aggregate.csv"):
    aggs.append(pd.read_csv("eval_results_llama/aggregate.csv"))
agg = pd.concat(aggs, ignore_index=True)
agg_recs = [_clean(r) for r in agg.to_dict(orient="records")]
with open(OUT / "aggregate.json", "w", encoding="utf-8") as f:
    json.dump(agg_recs, f, ensure_ascii=False, indent=1)
print(f"\naggregate.json: {len(agg_recs)} (config x stratum x metric) rows with mean + ci_low/ci_high")

with open(OUT / "index.json", "w", encoding="utf-8") as f:
    json.dump(index, f, ensure_ascii=False, indent=1)
print(f"index.json: {len(index)} configs")

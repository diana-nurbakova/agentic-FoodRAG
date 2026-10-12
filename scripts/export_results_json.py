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


# Directories searched, in order. `eval_results_ret_rerun/` (E-1') holds configs
# whose (experiment_id, config_id) are IDENTICAL to the superseded ones in
# eval_results/ and eval_results_llama/ — same 'EXP-ret-tight',
# 'ret_text2pandas@gpt4o-mini', 8,230 rows. Exporting both into one namespace
# silently overwrote one with the other, which is the failure this project is
# auditing. Any config_id appearing in more than one directory is therefore
# suffixed with its source directory, and an unqualified collision is a hard
# error rather than a last-writer-wins.
#
# The published homes (`eval_results/`, `eval_results_llama/`) keep their
# canonical unsuffixed filenames so existing references stay valid — they never
# collide with each other. Only the newcomer moves.
SOURCE_DIRS = ("eval_results", "eval_results_llama", "eval_results_ret_rerun")
DIR_SUFFIX = {"eval_results": "", "eval_results_llama": "",
              "eval_results_ret_rerun": "__rerun"}


def _source_dir(path: str) -> str:
    return Path(path).parent.name


# ---- per-question records, one JSON per config ----
# Prefer Parquet; fall back to the .pkl a config was written as when its records
# have a column the Parquet engine rejects (write_results falls back to pickle).
parquets = [p for d in SOURCE_DIRS
            for p in sorted(glob.glob(f"{d}/*_per_question.parquet"))
            if os.path.getsize(p) > 0]  # skip 0-byte turds from a failed to_parquet
_covered = {(_source_dir(p), Path(p).name[: -len("_per_question.parquet")]) for p in parquets}
pkls = [p for d in SOURCE_DIRS
        for p in sorted(glob.glob(f"{d}/*_per_question.pkl"))
        if (_source_dir(p), Path(p).name[: -len("_per_question.pkl")]) not in _covered]

# Which config_ids appear in more than one source directory?
_seen: dict[str, set] = {}
for pq in parquets + pkls:
    stem = Path(pq).name.split("_per_question")[0]
    _seen.setdefault(stem, set()).add(_source_dir(pq))
_ambiguous = {c for c, dirs in _seen.items() if len(dirs) > 1}
if _ambiguous:
    print(f"config_ids present in multiple directories (suffixed to keep them distinct): "
          f"{sorted(_ambiguous)}\n")

index = []
_written: dict[str, str] = {}
for pq in parquets + pkls:
    df = pd.read_pickle(pq) if pq.endswith(".pkl") else pd.read_parquet(pq)
    cid = str(df["config_id"].iloc[0]) if "config_id" in df.columns and len(df) else Path(pq).stem
    src = _source_dir(pq)
    suffix = DIR_SUFFIX.get(src, f"__{src}") if cid in _ambiguous else ""
    recs = [_clean(r) for r in df.to_dict(orient="records")]
    # sort by (run, question_id) so the file reads run-by-run
    recs.sort(key=lambda r: (r.get("run", 0), str(r.get("question_id", ""))))
    path = OUT / f"{_safe_name(cid)}{suffix}_per_question.json"
    if path.name in _written:
        raise SystemExit(
            f"REFUSING to overwrite {path.name}: already written from "
            f"{_written[path.name]}, now again from {pq}. Two configs share an "
            f"output name; add a DIR_SUFFIX entry rather than losing one.")
    _written[path.name] = pq
    with open(path, "w", encoding="utf-8") as f:
        json.dump(recs, f, ensure_ascii=False, indent=1)
    n_runs = len({r.get("run") for r in recs})
    index.append({"config_id": cid, "source_dir": src, "file": path.name,
                  "n_records": len(recs), "n_runs": n_runs})
    print(f"{src}/{cid}: {len(recs)} records ({n_runs} runs) -> {path.name}")

# ---- aggregate records WITH confidence intervals ----
aggs = []
for d in SOURCE_DIRS:
    p = f"{d}/aggregate.csv"
    if os.path.exists(p):
        a = pd.read_csv(p)
        # Without this the E-1' rows are indistinguishable from the superseded
        # ones they supersede: same experiment_id, config_id, stratum and metric,
        # different value. Any consumer keyed on those four fields would pick one
        # arbitrarily.
        a["source_dir"] = d
        aggs.append(a)
agg = pd.concat(aggs, ignore_index=True)
_keys = ["experiment_id", "config_id", "stratum", "metric"]
_dupes = agg[agg.duplicated(_keys, keep=False)]
if len(_dupes):
    pairs = sorted({(r.config_id, r.source_dir) for r in _dupes.itertuples()})
    print(f"\n⚠ {len(_dupes)} aggregate rows share (experiment_id, config_id, stratum, "
          f"metric) across directories — disambiguate on `source_dir`: {pairs}")
agg_recs = [_clean(r) for r in agg.to_dict(orient="records")]
with open(OUT / "aggregate.json", "w", encoding="utf-8") as f:
    json.dump(agg_recs, f, ensure_ascii=False, indent=1)
print(f"\naggregate.json: {len(agg_recs)} (config x stratum x metric) rows with mean + ci_low/ci_high")

with open(OUT / "index.json", "w", encoding="utf-8") as f:
    json.dump(index, f, ensure_ascii=False, indent=1)
print(f"index.json: {len(index)} configs")

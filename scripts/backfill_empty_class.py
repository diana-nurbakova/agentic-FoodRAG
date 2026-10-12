"""Backfill ``empty_class`` into the E-1' per-question parquets (check SCH, route A).

The E-1' re-run persisted the four-way empty-result partition to the
``*_trace.jsonl`` sidecar only; the parquet it wrote carries the same twelve
columns as the void artefact it supersedes. A paper proposing ``empty_class``
as a first-class per-question field needs its own artefacts to carry it.

No re-run and no API calls: the trace and the parquet are two views of the same
``(config_id, run, question_id)`` population, so the value is recovered by a
join. The join is verified total and the class is checked against
``retrieved_ids`` on every row before anything is written.

``scripts/eval_retrievers_tight.py`` writes the column natively from here on;
this script exists for the artefacts produced before that change.

Usage::

    python scripts/backfill_empty_class.py --check     # verify only, write nothing
    python scripts/backfill_empty_class.py             # backfill in place
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Optional

import pandas as pd

KEY = ["config_id", "run", "question_id"]
OK = "ok"


def load_trace(path: Path) -> pd.DataFrame:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    return pd.DataFrame(rows)


def reconcile(per_q: pd.DataFrame, trace: pd.DataFrame) -> pd.Series:
    """Return the ``empty_class`` column aligned to ``per_q``, or raise."""
    for name, df in (("parquet", per_q), ("trace", trace)):
        dupes = int(df.duplicated(subset=KEY).sum())
        if dupes:
            raise SystemExit(f"{name}: {dupes} duplicate {KEY} rows — join is unsafe")

    pk = set(map(tuple, per_q[KEY].to_numpy()))
    tk = set(map(tuple, trace[KEY].to_numpy()))
    if pk != tk:
        raise SystemExit(f"key sets differ: {len(pk - tk)} parquet-only, "
                         f"{len(tk - pk)} trace-only rows")

    merged = per_q.merge(trace[KEY + ["empty_class", "retrieved_ids"]],
                         on=KEY, how="left", suffixes=("", "_trace"))
    if len(merged) != len(per_q):
        raise SystemExit(f"join changed row count: {len(per_q)} -> {len(merged)}")
    if merged["empty_class"].isna().any():
        raise SystemExit(f"{int(merged['empty_class'].isna().sum())} rows with no class")

    # The sidecar and the artefact must agree about what was retrieved, else the
    # class is being attached to a different measurement than the one it describes.
    ids_disagree = sum(1 for a, b in zip(merged["retrieved_ids"],
                                         merged["retrieved_ids_trace"])
                       if list(a) != list(b))
    if ids_disagree:
        raise SystemExit(f"{ids_disagree} rows where trace and parquet retrieved_ids differ")

    inconsistent = sum(1 for ids, k in zip(merged["retrieved_ids"], merged["empty_class"])
                       if bool(len(ids)) != (k == OK))
    if inconsistent:
        raise SystemExit(f"{inconsistent} rows where empty_class disagrees with retrieved_ids")

    return merged["empty_class"].astype("string")


def process(out_dir: Path, config_id: str, *, check_only: bool) -> dict:
    pq_path = out_dir / f"{config_id}_per_question.parquet"
    tr_path = out_dir / f"{config_id}_trace.jsonl"
    per_q = pd.read_parquet(pq_path)
    trace = load_trace(tr_path)

    already = "empty_class" in per_q.columns
    classes = reconcile(per_q, trace)
    hist = dict(Counter(classes.tolist()))

    status = "verified" if check_only else ("re-verified" if already else "backfilled")
    if not check_only:
        if already:
            # Idempotent: a second run must confirm, not overwrite blindly.
            differing = int((per_q["empty_class"].astype("string") != classes).sum())
            if differing:
                raise SystemExit(f"{config_id}: existing empty_class disagrees with the "
                                 f"trace on {differing} rows — refusing to overwrite")
        else:
            backup = pq_path.with_suffix(".parquet.pre-sch.bak")
            if not backup.exists():
                shutil.copy2(pq_path, backup)
            per_q["empty_class"] = classes
            per_q.to_parquet(pq_path, index=False)

    return {"config_id": config_id, "rows": len(per_q), "status": status,
            "classes": hist, "columns": len(per_q.columns)}


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", default="eval_results_ret_rerun", type=Path)
    p.add_argument("--config", action="append",
                   help="config_id (repeatable; default: both E-1' backbones)")
    p.add_argument("--check", action="store_true",
                   help="verify the join and the class/ids consistency, write nothing")
    args = p.parse_args(argv)

    configs = args.config or ["ret_text2pandas@gpt4o-mini", "ret_text2pandas@llama70b"]
    total = Counter()
    for cid in configs:
        r = process(args.out_dir, cid, check_only=args.check)
        total.update(r["classes"])
        print(f"[{r['status']}] {r['config_id']}: {r['rows']:,} rows, "
              f"{r['columns']} columns, {r['classes']}", flush=True)
    print(f"[total] {sum(total.values()):,} question-runs  {dict(total)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""E-3 + E-4 — corrected re-aggregation from the per-question records.

No LLM calls, no re-run. Both corrections are pure recomputation from the
authoritative artefacts (``experiment-specification.md`` §0).

**E-3 — ToG denominator.** ``tog`` question ids are unique per ``(run, question)``
(``r1c0_00000``, ``r3c9_00008``), so ``runner._per_question_means`` groups by
``question_id`` and sees 300 independent questions where there are 100 evaluated
three times. Consequences: ``n_questions`` inflated 3x, ``per_question_variance``
= 0.0 by construction rather than by measurement, bootstrap CI too narrow.
Fix: strip the ``r<run>`` prefix, regroup on the true id, recompute.

**E-4 — Empty-handling convention.** Three conventions coexist: zero-on-empty
(16 configs), NaN-on-empty (5 configs, and *intermittent within* ``foodrag_h2``),
and ToG's id scheme. Zero-on-empty is the declared default (protocol §1.5).
Harmonising means: where a record has a NON-EMPTY gold set but a NaN retrieval
metric, that NaN came from the empty *result*, not the empty *gold*, and becomes
0.0. Records with empty gold stay NaN under both conventions — that is the
legitimate ``runner.py:92-94`` guard, and it is left alone.

    foodrag-reaggregate                     # report published vs harmonised
    foodrag-reaggregate --csv out.csv       # also write the delta table
"""
from __future__ import annotations

import argparse
import glob
import re
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

# Only the retrieval metrics carry the convention split. Other NaNs (avg_nutri
# when nothing resolves, guideline_* when a nutrient is absent) are genuine
# "undefined", not "empty result", and must not be coerced to 0.
RETRIEVAL_METRICS = ("hit_at_10", "mrr_at_10", "ndcg_at_10")

TOG_ID = re.compile(r"^r\d+(?P<rest>.+)$")


def load_records(path: Path) -> list[dict]:
    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_pickle(path)
    return df.to_dict("records")


def fix_tog_ids(records: list[dict]) -> tuple[list[dict], int, int]:
    """E-3: strip the run prefix so the same question collapses across runs."""
    before = len({r["question_id"] for r in records})
    for r in records:
        m = TOG_ID.match(str(r["question_id"]))
        if m:
            r["question_id"] = m.group("rest").lstrip("_")
    after = len({r["question_id"] for r in records})
    return records, before, after


def harmonise_empties(records: list[dict]) -> int:
    """E-4: NaN retrieval metric on a NON-EMPTY gold set becomes 0.0."""
    n = 0
    for r in records:
        if (r.get("answer_set_size") or 0) <= 0:
            continue                      # empty gold -> NaN under both conventions
        m = r.get("metrics") or {}
        for k in RETRIEVAL_METRICS:
            v = m.get(k)
            if v is None or (isinstance(v, float) and np.isnan(v)):
                m[k] = 0.0
                n += 1
    return n


def published_table(root: Path) -> pd.DataFrame:
    frames = []
    for a in sorted(glob.glob(str(root / "eval_results*" / "aggregate.csv"))):
        df = pd.read_csv(a, keep_default_na=False, na_values=[""])
        df["aggregate_file"] = Path(a).parent.name
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def per_question_path(agg_dir: Path, config_id: str) -> Optional[Path]:
    for suf in (".parquet", ".pkl"):
        p = agg_dir / f"{config_id}_per_question{suf}"
        if p.exists():
            return p
    return None


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default=".")
    p.add_argument("--csv", help="write the (config, metric, published, harmonised, delta) table")
    args = p.parse_args(argv)
    root = Path(args.root).resolve()

    from foodrag.baselines import runner

    pub = published_table(root)
    configs = pub[["aggregate_file", "config_id", "experiment_id"]].drop_duplicates()

    rows: list[dict] = []
    notes: list[str] = []
    print("# E-3 + E-4 — corrected re-aggregation\n")

    for _, c in configs.iterrows():
        agg_dir = root / c["aggregate_file"]
        pq = per_question_path(agg_dir, c["config_id"])
        if pq is None:
            notes.append(f"{c['config_id']}: NO per-question artefact — cannot recompute")
            continue
        records = load_records(pq)
        for r in records:                       # parquet gives numpy types / arrays
            if isinstance(r.get("metrics"), dict):
                r["metrics"] = dict(r["metrics"])

        tog_before = tog_after = None
        if c["config_id"] == "tog":
            records, tog_before, tog_after = fix_tog_ids(records)
        n_coerced = harmonise_empties(records)

        agg = runner.aggregate(records, experiment_id=c["experiment_id"],
                               config_id=c["config_id"])
        agg = agg[agg.stratum == "aggregate"]

        pub_c = pub[(pub.config_id == c["config_id"]) &
                    (pub.aggregate_file == c["aggregate_file"]) &
                    (pub.stratum == "aggregate")]
        for _, h in agg.iterrows():
            pr = pub_c[pub_c.metric == h["metric"]]
            if not len(pr):
                continue
            pv = float(pr.iloc[0]["mean"])
            rows.append({
                "config_id": c["config_id"], "metric": h["metric"],
                "published": pv, "harmonised": float(h["mean"]),
                "delta": float(h["mean"]) - pv,
                "n_pub": int(pr.iloc[0]["n_questions"]),
                "n_harm": int(h["n_questions"]),
                "var_pub": float(pr.iloc[0]["per_question_variance"]),
                "var_harm": float(h["per_question_variance"]),
                "ci_pub": f"[{float(pr.iloc[0]['ci_low']):.4f}, {float(pr.iloc[0]['ci_high']):.4f}]",
                "ci_harm": f"[{h['ci_low']:.4f}, {h['ci_high']:.4f}]",
                "coerced_nans": n_coerced,
            })
        if tog_before is not None:
            notes.append(f"tog: distinct question_id {tog_before} -> {tog_after} "
                         f"after stripping the run prefix")
        if n_coerced:
            notes.append(f"{c['config_id']}: coerced {n_coerced} NaN retrieval values to 0.0")

    t = pd.DataFrame(rows)
    changed = t[t.delta.abs() > 1e-12].copy()

    print("## E-3 — ToG re-aggregation\n")
    tog = t[t.config_id == "tog"]
    if len(tog):
        with pd.option_context("display.width", 220, "display.float_format", lambda v: f"{v:.6f}"):
            print(tog[["metric", "published", "harmonised", "delta", "n_pub", "n_harm",
                       "var_pub", "var_harm", "ci_pub", "ci_harm"]].to_string(index=False))

    print("\n## E-4 — configs whose value changes under zero-on-empty\n")
    other = changed[changed.config_id != "tog"]
    if len(other):
        with pd.option_context("display.width", 220, "display.float_format", lambda v: f"{v:.6f}"):
            print(other[["config_id", "metric", "published", "harmonised", "delta",
                         "n_pub", "n_harm"]].to_string(index=False))
    else:
        print("  (none)")

    print(f"\n## summary\n\n  rows compared          : {len(t)}")
    print(f"  rows with |delta| > 0  : {len(changed)}")
    print(f"  configs affected       : {sorted(changed.config_id.unique())}")
    if notes:
        print("\n## notes\n")
        for n in notes:
            print(f"  - {n}")

    if args.csv:
        t.to_csv(args.csv, index=False)
        print(f"\ndelta table -> {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

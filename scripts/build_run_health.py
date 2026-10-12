"""WB.5 — a per-run health record for EVERY config, not only the two re-run ones.

``eval_results_ret_rerun/run_health.json`` is written by the E-1' driver and
covers the two backbones that driver ran. Every other config in the repository
has a health record only inside the census, and nothing carries the
``empty_class`` partition at all.

This emits one record per ``(directory, config_id, run)`` across every
``eval_results*/aggregate.csv``, merging:

  · the census fields (empty rate, latency shape, primary metric, verdict), and
  · the ``empty_class`` histogram where a ``*_trace.jsonl`` sidecar exists.

⚠ **The null is the point.** For the 21 configs with no trace, ``empty_classes``
is written as ``null`` with an explicit ``empty_classes_reason``, never omitted
and never zero-filled. An absent field and a measured-zero field are different
claims, and the whole argument for a health-aware schema is that the published
artefacts cannot tell them apart. Zero-filling here would reproduce the defect.

Thresholds are frozen (S.3). Verdicts match the frozen census exactly.

    python scripts/build_run_health.py                     # -> audit/run_health.json
    python scripts/build_run_health.py --out other.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import run_health_census as census  # noqa: E402


def trace_classes(agg_path: Path, config_id: str) -> tuple[Optional[dict], str]:
    """The per-run ``empty_class`` histogram from the sidecar, or why there is none."""
    trace = agg_path.parent / f"{config_id}_trace.jsonl"
    if not trace.exists():
        return None, ("no *_trace.jsonl sidecar: this config was produced by a driver "
                      "that did not classify empty results (the defect E-1'.4 and "
                      "check SCH exist to remove)")
    per_run: dict[int, Counter] = {}
    with open(trace, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            per_run.setdefault(int(r["run"]), Counter())[r.get("empty_class") or "?"] += 1
    return {k: dict(v) for k, v in per_run.items()}, ""


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default=".")
    p.add_argument("--out", default="audit/run_health.json")
    args = p.parse_args(argv)
    root = Path(args.root).resolve()

    rows: list[dict] = []
    classes_by_cfg: dict[tuple[str, str], tuple[Optional[dict], str]] = {}
    for agg in census.find_aggregates(root):
        adf = __import__("pandas").read_csv(agg, keep_default_na=False, na_values=[""])
        for cfg in sorted(set(adf.config_id)):
            df = census.load_per_question(agg, cfg)
            if df is None:
                continue
            rows.extend(census.run_rows(df, cfg, agg))
            classes_by_cfg[(agg.parent.name, cfg)] = trace_classes(agg, cfg)

    thr = census.fit_latency_thresholds(rows, frozen=True)
    by_bucket: dict[tuple, list[dict]] = {}
    for r in rows:
        by_bucket.setdefault(census._key(r)[:2], []).append(r)

    out = []
    for r in rows:
        d, cfg, run = census._key(r)
        verdict = "+".join(census.verdicts_for(r, by_bucket[(d, cfg)], thr))
        classes, reason = classes_by_cfg.get((d, cfg), (None, "no artefact"))
        rec = dict(r)
        rec["directory"] = d
        rec["verdict"] = verdict
        rec["empty_classes"] = (classes or {}).get(run) if classes else None
        rec["empty_classes_reason"] = "" if rec["empty_classes"] else reason
        out.append(rec)

    dest = Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(
        {"thresholds": thr,
         "note": ("empty_classes is null, not zero-filled, wherever the producing "
                  "driver did not classify empty results. See WB.5 / E-1'.4."),
         "runs": out}, indent=2), encoding="utf-8")

    have = sum(1 for r in out if r["empty_classes"])
    cfgs = {(r["directory"], r["config_id"]) for r in out}
    print(f"wrote {dest}: {len(out)} (config, run) pairs over {len(cfgs)} configs")
    print(f"  with an empty_class partition : {have}")
    print(f"  with empty_classes = null     : {len(out) - have}")
    print(f"  verdicts: {dict(Counter(r['verdict'] for r in out))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

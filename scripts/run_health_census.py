"""Check S — per-run health census over every config in every aggregate.

Motivated by ``specs/checks-T-Q-H-findings.md`` Q.2/Q.5: ``ret_text2pandas@gpt4o-mini``
runs 4 and 5 returned nothing for all 1,646 questions, and the aggregate records
``n_questions=1646, n_runs=5`` exactly as it does for a healthy config. The
aggregate schema has no run-health channel, so a void run is invisible.

This sweeps every ``(config, run)`` pair and applies the pre-registered rules in
``specs/checks-S-R-spec.md`` S.3. Thresholds for SUSPECT-LATENCY are *fitted* on
the known-good/known-bad runs and printed before any verdict, per S.3's note that
a threshold chosen after seeing the data is not a detection rule.

    foodrag-health                      # sweep + report
    foodrag-health --json out.json      # also persist the census

Reports every run's verdict, HEALTHY included (S.3): a sweep that reports only
failures cannot be distinguished from a sweep that stopped early.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

VOID = "VOID"
DEGRADED = "DEGRADED"
SUSPECT_LATENCY = "SUSPECT-LATENCY"
CONVENTION_SPLIT = "CONVENTION-SPLIT"
HEALTHY = "HEALTHY"

# Runs used to fit the latency thresholds. Known-bad from the T/Q/H findings.
KNOWN_BAD = {("ret_text2pandas@gpt4o-mini", 4), ("ret_text2pandas@gpt4o-mini", 5)}


# ── loading ──────────────────────────────────────────────────────────────────

def find_aggregates(root: Path) -> list[Path]:
    return sorted(Path(p) for p in glob.glob(str(root / "eval_results*" / "aggregate.csv")))


def load_per_question(agg_path: Path, config_id: str) -> Optional[pd.DataFrame]:
    """The per-question artefact backing one config, whatever format it is in."""
    d = agg_path.parent
    for suffix in (".parquet", ".pkl"):
        p = d / f"{config_id}_per_question{suffix}"
        if p.exists():
            try:
                return pd.read_parquet(p) if suffix == ".parquet" else pd.read_pickle(p)
            except Exception:
                continue
    p = d / "json" / f"{config_id}_per_question.json"
    if p.exists():
        try:
            return pd.DataFrame(json.loads(p.read_text(encoding="utf-8")))
        except Exception:
            return None
    return None


def _metric(m: Any, key: str) -> float:
    if not isinstance(m, dict):
        return float("nan")
    v = m.get(key)
    if v is None:
        return float("nan")
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def primary_metric_name(df: pd.DataFrame) -> Optional[str]:
    keys: list[str] = []
    for m in df["metrics"].head(50):
        if isinstance(m, dict):
            keys = list(m.keys())
            break
    for pref in ("hit_at_10", "hit_at_5", "mrr_at_10"):
        if pref in keys:
            return pref
    return next((k for k in keys if k.startswith("hit_at_")), None)


# ── per-run metrics (S.2) ────────────────────────────────────────────────────

def run_rows(df: pd.DataFrame, config_id: str, agg_path: Path) -> list[dict]:
    pm = primary_metric_name(df)
    has_ids = "retrieved_ids" in df.columns
    if has_ids:
        df = df.copy()
        df["_n"] = df["retrieved_ids"].apply(lambda a: 0 if a is None else len(a))
    if pm:
        df["_p"] = df["metrics"].apply(lambda m: _metric(m, pm))
    df["_lat"] = df["metrics"].apply(lambda m: _metric(m, "latency_s"))
    gold = df["answer_set_size"] if "answer_set_size" in df.columns else None

    out = []
    for run, g in df.groupby("run"):
        lat = g["_lat"].dropna().to_numpy()
        rec: dict[str, Any] = {
            "aggregate_file": str(agg_path).replace("\\", "/"),
            "config_id": config_id,
            "run": int(run),
            "questions": int(len(g)),
            "primary_metric": pm,
        }
        rec["empty"] = int((g["_n"] == 0).sum()) if has_ids else None
        rec["empty_rate"] = float((g["_n"] == 0).mean()) if has_ids else None
        if pm:
            p = g["_p"]
            rec["primary_all_zero_rate"] = float((p.fillna(0) == 0).mean())
            rec["primary_mean"] = float(np.nanmean(p)) if p.notna().any() else float("nan")
            rec["primary_nan"] = int(p.isna().sum())
            if gold is not None:
                nz = g["answer_set_size"] > 0
                rec["nan_on_nonempty_gold"] = int((p.isna() & nz).sum())
        if lat.size:
            q1, q3 = np.percentile(lat, [25, 75])
            rec |= {
                "lat_min": float(lat.min()), "lat_med": float(np.median(lat)),
                "lat_max": float(lat.max()), "lat_iqr": float(q3 - q1),
                "lat_cv": float(lat.std() / lat.mean()) if lat.mean() else float("nan"),
            }
        if has_ids:
            seen: set = set()
            for a in g["retrieved_ids"]:
                if a is not None:
                    seen.update(a)
            rec["distinct_recipes"] = len(seen)
        out.append(rec)
    return out


# ── pre-registered rules (S.3) ───────────────────────────────────────────────

def fit_latency_thresholds(rows: list[dict]) -> dict:
    """Fit SUSPECT-LATENCY thresholds on the known-bad runs' OWN config family.

    A global fit across all configs is meaningless: ``ret_random`` answers in
    ~0.03 s and ``tog`` in ~940 s, so the global minimum max-latency collapses to
    0.11 s and the rule can never fire. The mechanism the rule encodes — "a
    uniform per-call latency returning nothing is a retry budget expiring" — is
    only interpretable against the *same config's* other runs.

    So the rule is applied peer-relative within a config (see
    :func:`verdicts_for`), and the values fitted here are reported for
    transparency: they are what the text2pandas family looks like when healthy.
    """
    fam = [r for r in rows if r["config_id"].startswith("ret_text2pandas@")
           and r.get("lat_cv") is not None
           and not math.isnan(r.get("lat_cv", float("nan")))]
    healthy = [r for r in fam if (r["config_id"], r["run"]) not in KNOWN_BAD
               and (r.get("empty_rate") or 0) < 1.0]
    bad = [r for r in fam if (r["config_id"], r["run"]) in KNOWN_BAD]
    return {
        "family": "ret_text2pandas@*",
        "margin": 3.0,
        "healthy_runs_used": len(healthy),
        "cv_threshold": min(r["lat_cv"] for r in healthy) if healthy else None,
        "max_latency_threshold": min(r["lat_max"] for r in healthy) if healthy else None,
        "healthy_cv": sorted(round(r["lat_cv"], 4) for r in healthy),
        "known_bad_cv": [round(r["lat_cv"], 4) for r in bad],
        "known_bad_max": [round(r["lat_max"], 2) for r in bad],
    }


def verdicts_for(r: dict, peers: list[dict], thr: dict) -> list[str]:
    v: list[str] = []
    er = r.get("empty_rate")
    if er == 1.0 or r.get("primary_all_zero_rate") == 1.0:
        v.append(VOID)
    if er is not None and not v:
        others = [p["empty_rate"] for p in peers
                  if p["run"] != r["run"] and p.get("empty_rate") is not None]
        med = float(np.median(others)) if others else None
        if er >= 0.40 or (med is not None and med > 0 and er >= 3 * med):
            v.append(DEGRADED)
    # Peer-relative within the config, WITH a margin. A bare "lowest of its
    # peers" test is rank-based: it fires on exactly one run of every multi-run
    # config whether or not anything is wrong. The known-bad runs are not merely
    # lowest, they are an order of magnitude lower (CV 7.3x, max 5.3x below their
    # healthy peers), so the rule requires a factor-of-MARGIN gap.
    cv, mx = r.get("lat_cv"), r.get("lat_max")
    peer_cv = [p["lat_cv"] for p in peers if p["run"] != r["run"]
               and p.get("lat_cv") is not None and not math.isnan(p["lat_cv"])]
    peer_mx = [p["lat_max"] for p in peers if p["run"] != r["run"]
               and p.get("lat_max") is not None]
    margin = thr.get("margin", 3.0)
    if cv is not None and mx is not None and not math.isnan(cv):
        peer_hit = (peer_cv and peer_mx
                    and cv * margin < min(peer_cv) and mx * margin < min(peer_mx))
        # Family-fitted absolute test. Needed because the peer test MASKS the
        # known-bad case: gpt4o-mini runs 4 and 5 share the signature, so each is
        # the other's peer and neither stands out. A rule that cannot catch the
        # case it was derived from is not a rule.
        fam_hit = (r["config_id"].startswith("ret_text2pandas@")
                   and thr.get("cv_threshold") is not None
                   and cv < thr["cv_threshold"] and mx < thr["max_latency_threshold"])
        if peer_hit or fam_hit:
            v.append(SUSPECT_LATENCY)
    if r.get("nan_on_nonempty_gold"):
        v.append(CONVENTION_SPLIT)
    return v or [HEALTHY]


# ── main ─────────────────────────────────────────────────────────────────────

def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default=".")
    p.add_argument("--json", help="write the full census here")
    args = p.parse_args(argv)
    root = Path(args.root).resolve()

    aggs = find_aggregates(root)
    print(f"# Check S — per-run health census\n\naggregate files: {len(aggs)}")
    for a in aggs:
        print(f"  - {str(a).replace(chr(92), '/')}")

    # ── S.1.1 enumerate ──
    inventory: list[dict] = []
    for a in aggs:
        adf = pd.read_csv(a, keep_default_na=False, na_values=[""])
        for cfg, g in adf.groupby("config_id"):
            agg_rows = g[g.stratum == "aggregate"]
            inventory.append({
                "aggregate_file": str(a).replace("\\", "/"), "config_id": cfg,
                "experiment_id": sorted(set(g.experiment_id))[0],
                "n_questions": int(agg_rows.n_questions.max()) if len(agg_rows) else None,
                "runs_declared": int(agg_rows.n_runs.max()) if len(agg_rows) else None,
                "metrics_rows": int(len(g)),
            })
    print(f"\n## S.1.1 — inventory: {len(inventory)} configs\n")
    inv = pd.DataFrame(inventory)
    print(inv.to_string(index=False))

    # ── S.2 per-run metrics ──
    rows: list[dict] = []
    missing: list[str] = []
    for item in inventory:
        a = Path(item["aggregate_file"])
        df = load_per_question(a, item["config_id"])
        if df is None or "run" not in df.columns:
            missing.append(f"{item['config_id']} ({a.parent.name})")
            continue
        rows.extend(run_rows(df, item["config_id"], a))

    if missing:
        print(f"\n⚠ no per-question artefact for {len(missing)} config(s): {missing}")

    # ── S.3 fitted thresholds, printed BEFORE the verdicts ──
    thr = fit_latency_thresholds(rows)
    print("\n## S.3 — fitted detection thresholds (stated before verdicts)\n")
    print(f"  healthy runs used to fit : {thr['healthy_runs_used']}")
    print(f"  family used to fit       : {thr[chr(39)+chr(39)]}" if False else f"  family used to fit       : {thr['family']}")
    print(f"  latency CV threshold     : < {thr['cv_threshold']:.4f}"
          if thr["cv_threshold"] is not None else "  latency CV threshold     : n/a")
    print(f"  latency max threshold    : < {thr['max_latency_threshold']:.2f}s"
          if thr["max_latency_threshold"] is not None else "")
    print(f"  peer-relative margin     : {thr['margin']}x (cv AND max must be this far below the config's other runs)")
    print(f"  healthy family CV        : {thr['healthy_cv']}")
    print(f"  known-bad runs' CV       : {thr['known_bad_cv']}")
    print(f"  known-bad runs' max lat  : {thr['known_bad_max']}")
    print("  VOID      : empty rate = 100% OR primary metric 0.0 on every question")
    print("  DEGRADED  : empty rate >= 40% absolute OR >= 3x median of the config's other runs")
    print("  CONV-SPLIT: any NaN primary on a non-empty-gold row")

    by_cfg: dict[str, list[dict]] = {}
    for r in rows:
        by_cfg.setdefault(r["config_id"], []).append(r)
    for r in rows:
        r["verdict"] = "+".join(verdicts_for(r, by_cfg[r["config_id"]], thr))

    print(f"\n## S.2/S.3 — per-run verdicts ({len(rows)} (config, run) pairs)\n")
    disp = pd.DataFrame(rows)
    cols = ["config_id", "run", "questions", "empty", "empty_rate", "primary_mean",
            "primary_nan", "nan_on_nonempty_gold", "lat_med", "lat_max", "lat_cv",
            "distinct_recipes", "verdict"]
    cols = [c for c in cols if c in disp.columns]
    with pd.option_context("display.width", 250, "display.max_rows", 400,
                           "display.float_format", lambda v: f"{v:.4f}"):
        print(disp[cols].to_string(index=False))

    print("\n## verdict summary\n")
    vc: dict[str, int] = {}
    for r in rows:
        for v in r["verdict"].split("+"):
            vc[v] = vc.get(v, 0) + 1
    for v, c in sorted(vc.items(), key=lambda kv: -kv[1]):
        print(f"  {c:>4}  {v}")

    # ── S.4 cross-config consistency ──
    print("\n## S.4 — cross-config consistency\n")
    print("### S.4.3 — declared runs vs distinct run ids")
    bad = 0
    for item in inventory:
        obs = [r for r in rows if r["config_id"] == item["config_id"]
               and r["aggregate_file"] == item["aggregate_file"]]
        if obs and item["runs_declared"] and len(obs) != item["runs_declared"]:
            print(f"  ⚠ {item['config_id']}: declares {item['runs_declared']}, "
                  f"artefact has {len(obs)}")
            bad += 1
    print(f"  {'all configs consistent' if not bad else f'{bad} mismatch(es)'}")

    print("\n### S.4.2 — scoring-path partition (NaN primary on non-empty gold)")
    split = sorted({r["config_id"] for r in rows if r.get("nan_on_nonempty_gold")})
    clean = sorted({r["config_id"] for r in rows} - set(split))
    print(f"  NaN-on-empty-result writer ({len(split)}): {split}")
    print(f"  zero-on-empty-result writer ({len(clean)}): {clean}")

    if args.json:
        Path(args.json).write_text(json.dumps(
            {"inventory": inventory, "thresholds": thr, "runs": rows}, indent=2,
            default=str), encoding="utf-8")
        print(f"\ncensus -> {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

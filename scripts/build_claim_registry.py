"""REG — Generate the claim registry from the artefacts.

Per ``experiment-specification.md`` §0 and §6.5 the registry is **generated**, not
hand-maintained: `numerical-claims.md` and `docs/RESULTS_REPORT.md` are derived
from the result files, not authoritative over them. This script reads **only**
``eval_results*/*_per_question.{parquet,pkl}`` and recomputes every publishable
number from them, then attaches the provenance a bare value does not carry:

  config · metric · stratum · value · CI · n · runs
    · convention        (zero-on-empty | NaN-on-empty)
    · health verdict    (S'.1 census, computed from the same records)
    · patched?          (presence of `_safety_fixed`, and how many records)
    · source commit     (last commit touching the artefact)
    · guard             (why a value must not be quoted, if so)

REG.3 acceptance: every number in the draft resolves to exactly one row, and every
row carries a health verdict and a patched flag. Numbers that cannot resolve do
not enter the draft.

    foodrag-registry --csv registry.csv --md registry.md
"""
from __future__ import annotations

import argparse
import glob
import subprocess
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

# ── REG.2 seeded guards ──────────────────────────────────────────────────────
# (config, metric) -> guard text.  metric None = applies to every metric.
GUARDS: list[tuple[str, Optional[str], str]] = [
    ("ret_text2pandas@gpt4o-mini", None,
     "WITHDRAWN (B5): 2 of 5 runs 100% empty; the comparison is void. Superseded by E-1."),
    ("kg_text_to_cypher", None,
     "DO NOT QUOTE: n=12, one run, no per-question artefact, no aggregate stratum row."),
    ("ret_intersection", None,
     "Quote only with its 96.42% empty rate."),
    ("ret_random", None,
     "hit@10=0.000 is the intended behaviour of a random baseline, not a void run."),
    ("tog", None,
     "E-3: n=100 not 300; per_question_variance 0.0311 not 0; CIs ~60% wider than published."),
    (None, "calorie_adherence",
     "Scale-invariant and anti-correlated with in-band rate; rename to calorie_distribution_shape."),
    (None, "allergen_violation",
     "P1: 0.000 is POST-remediation. Quote pre-fix 0.000310 (h1_t2p) / 0.001969 (h1_dense) "
     "/ 0.003427 (h2) / 0.000608 (h2_t2p)."),
    ("foodrag_h1_t2p", "correctness_judgeA",
     "D2: quote pre-fix 0.689, never 0.765. p=0.089 belongs to hit_at_10 on the dense pair, "
     "not to correctness."),
    ("foodrag_h1_t2p", "faithfulness_judgeA",
     "D2: quote pre-fix 0.922, never 0.997 (corrected p falls 3.9e-56 -> 6.7e-03)."),
    ("foodrag_h1_t2p", "relevancy_judgeA",
     "D2: the H2 regression is -0.113 pre-fix, not -0.033."),
    ("foodrag_h1_t2p_fullday", None,
     "X: 614 live hallucinated empty-plan explanations. No explanation metric may be compared "
     "against a non-_fullday config."),
    ("foodrag_h1_dense_fullday", None,
     "X: 29 live hallucinated empty-plan explanations."),
    (None, "hit_at_10",
     "S: no aggregate value without its per-run health record."),
]

PLAN_NUTRITION = ("guideline_adherence_overall", "calorie_adherence")
RETRIEVAL = ("hit_at_10", "mrr_at_10", "ndcg_at_10")


def guards_for(config: str, metric: str) -> list[str]:
    out = []
    for c, m, text in GUARDS:
        if (c is None or c == config) and (m is None or m == metric):
            out.append(text)
    return out


def last_commit(path: Path) -> str:
    r = subprocess.run(["git", "log", "-1", "--format=%h %ad", "--date=short", "--", str(path)],
                       capture_output=True, text=True, check=False)
    return (r.stdout or "").strip() or "untracked"


def load(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_pickle(path)


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default=".")
    p.add_argument("--csv")
    p.add_argument("--md")
    args = p.parse_args(argv)
    root = Path(args.root).resolve()

    from foodrag.baselines import runner
    from scripts import run_health_census as census

    # The stock bootstrap allocates (n_boot x n_questions) floats in one go —
    # 763 MB for the 10,000-question popularity configs, which fails while a
    # concurrent job holds memory. Draw in chunks from the SAME rng stream:
    # Generator.choice fills row-major, so k chunks of (m, n) consume the stream
    # in the same order as one (k*m, n) draw and yield identical resamples.
    def _chunked_ci(values, n_boot, seed, chunk=500):
        if values.size == 0:
            return float("nan"), float("nan")
        if values.size == 1:
            return float(values[0]), float(values[0])
        rng = np.random.default_rng(seed)
        means = np.empty(n_boot, dtype=float)
        done = 0
        while done < n_boot:
            m = min(chunk, n_boot - done)
            means[done:done + m] = rng.choice(
                values, size=(m, values.size), replace=True).mean(axis=1)
            done += m
        return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))

    runner._bootstrap_ci = _chunked_ci

    # verdicts, recomputed from the same per-question records
    agg_files = census.find_aggregates(root)
    cen_rows: list[dict] = []
    for a in agg_files:
        adf = pd.read_csv(a, keep_default_na=False, na_values=[""])
        for cfg in sorted(adf.config_id.unique()):
            df = census.load_per_question(a, cfg)
            if df is not None and "run" in df.columns:
                cen_rows.extend(census.run_rows(df, cfg, a))
    thr = census.fit_latency_thresholds(cen_rows)
    by_cfg: dict[str, list[dict]] = {}
    for r in cen_rows:
        by_cfg.setdefault(r["config_id"], []).append(r)
    for r in cen_rows:
        r["verdict"] = "+".join(census.verdicts_for(r, by_cfg[r["config_id"]], thr))

    rows: list[dict] = []
    for a in agg_files:
        adf = pd.read_csv(a, keep_default_na=False, na_values=[""])
        for cfg in sorted(adf.config_id.unique()):
            pq = None
            for suf in (".parquet", ".pkl"):
                cand = a.parent / f"{cfg}_per_question{suf}"
                if cand.exists():
                    pq = cand
                    break
            if pq is None:
                rows.append({"config": cfg, "metric": "*", "stratum": "*", "value": None,
                             "ci": "", "n": None, "runs": None, "convention": "",
                             "verdict": "NO ARTEFACT", "patched": "", "source": "",
                             "guard": " | ".join(guards_for(cfg, "*")) or "UNVERIFIABLE"})
                continue

            df = load(pq)
            recs = df.to_dict("records")
            for r in recs:
                if isinstance(r.get("metrics"), dict):
                    r["metrics"] = dict(r["metrics"])
            exp = sorted(set(adf[adf.config_id == cfg].experiment_id))[0]
            agg = runner.aggregate(recs, experiment_id=exp, config_id=cfg)
            agg = agg[agg.stratum == "aggregate"]

            # Per-run verdicts are compound strings ("A+B"); flatten to distinct
            # atoms so a config with two compound runs does not repeat a label.
            atoms: set[str] = set()
            for x in by_cfg.get(cfg, []):
                atoms.update(x["verdict"].split("+"))
            if len(atoms) > 1:
                atoms.discard("HEALTHY")   # "HEALTHY+VOID" reads as if it were both
            verdicts = sorted(atoms) or ["?"]
            conv = ("NaN-on-empty" if any(x.get("nan_on_nonempty_gold")
                                          for x in by_cfg.get(cfg, [])) else "zero-on-empty")
            n_patched = int(df["_safety_fixed"].notna().sum()) if "_safety_fixed" in df.columns else 0
            src = last_commit(pq)

            for _, h in agg.iterrows():
                g = guards_for(cfg, h["metric"])
                rows.append({
                    "config": cfg, "metric": h["metric"], "stratum": "aggregate",
                    "value": round(float(h["mean"]), 6),
                    "ci": f"[{float(h['ci_low']):.4f}, {float(h['ci_high']):.4f}]",
                    "n": int(h["n_questions"]), "runs": int(h["n_runs"]),
                    "convention": conv, "verdict": "+".join(verdicts),
                    "patched": f"{n_patched} records" if n_patched else "no",
                    "source": src, "guard": " | ".join(g),
                })

    reg = pd.DataFrame(rows)
    print(f"# REG — claim registry\n\nrows: {len(reg)}   configs: {reg.config.nunique()}")
    print(f"rows carrying a guard: {int((reg.guard.astype(str) != '').sum())}")
    print(f"rows from patched artefacts: {int((reg.patched != 'no').sum())}")
    print(f"configs with a non-HEALTHY verdict: "
          f"{sorted(reg[~reg.verdict.str.fullmatch('HEALTHY', na=False)].config.unique())}\n")

    key = reg[reg.metric.isin(RETRIEVAL + ("correctness_judgeA", "faithfulness_judgeA",
                                           "allergen_violation"))]
    with pd.option_context("display.width", 250, "display.max_rows", 300):
        print(key[["config", "metric", "value", "n", "runs", "convention",
                   "verdict", "patched"]].to_string(index=False))

    if args.csv:
        reg.to_csv(args.csv, index=False)
        print(f"\nregistry -> {args.csv}")
    if args.md:
        with open(args.md, "w", encoding="utf-8") as fh:
            fh.write("# Claim registry (generated — do not hand-edit)\n\n")
            fh.write(f"Generated from per-question artefacts only. Rows: {len(reg)}.\n\n")
            fh.write("| config | metric | value | CI | n | runs | convention | health | patched | source | guard |\n")
            fh.write("|---|---|---|---|---|---|---|---|---|---|---|\n")
            for _, r in reg.iterrows():
                fh.write(f"| {r['config']} | {r['metric']} | {r['value']} | {r['ci']} | {r['n']} | "
                         f"{r['runs']} | {r['convention']} | {r['verdict']} | {r['patched']} | "
                         f"{r['source']} | {r['guard']} |\n")
        print(f"registry (md) -> {args.md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

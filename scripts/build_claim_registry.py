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
import gc
import glob
import subprocess
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

# ── REG.2 seeded guards ──────────────────────────────────────────────────────
# (config, metric, source_dir) -> guard text. None = wildcard on that axis.
# ⚠ source_dir matters: `ret_text2pandas@gpt4o-mini` exists twice under the same
# config_id — the withdrawn void run in eval_results/ and the healthy E-1'
# re-run in eval_results_ret_rerun/. A config-only guard would stamp the fresh
# run as withdrawn.
GUARDS: list[tuple[Optional[str], Optional[str], Optional[str], str]] = [
    ("ret_text2pandas@gpt4o-mini", None, "eval_results",
     "WITHDRAWN (B5): runs 4 and 5 returned nothing for all 1,646 questions; the "
     "comparison is void. Superseded by eval_results_ret_rerun/."),
    ("ret_text2pandas@gpt4o-mini", None, "eval_results_ret_rerun",
     "E-1' re-run, 5/5 runs healthy (2.19-2.55% empty). Fresh-vs-fresh only: do NOT "
     "compare against the published 0.853827, and never merge with the void run."),
    ("ret_text2pandas@llama70b", None, "eval_results_ret_rerun",
     "E-1' re-run, 5/5 runs healthy (3.77-4.86% empty). Diverges from the published "
     "0.853827; report both, do not reconcile (R.2: original decoding params unrecoverable)."),
    ("ret_text2pandas@llama70b", None, "eval_results_llama",
     "B1 stands: 5/5 healthy runs, clean under S.5. E-1' does not restate it."),
    ("kg_text_to_cypher", None, None,
     "DO NOT QUOTE: n=12, one run, no per-question artefact, no aggregate stratum row."),
    ("ret_intersection", None, None,
     "Quote only with its 96.42% empty rate."),
    ("ret_random", None, None,
     "hit@10=0.000 is the intended behaviour of a random baseline, not a void run."),
    ("tog", None, None,
     "E-3: n=100 not 300; per_question_variance 0.0311 not 0; CIs ~60% wider than published."),
    (None, "calorie_adherence", None,
     "Scale-invariant and anti-correlated with in-band rate; rename to calorie_distribution_shape."),
    (None, "allergen_violation", None,
     "P1: 0.000 is POST-remediation. Quote pre-fix 0.000310 (h1_t2p) / 0.001969 (h1_dense) "
     "/ 0.003427 (h2) / 0.000608 (h2_t2p)."),
    ("foodrag_h1_t2p", "correctness_judgeA", None,
     "D2: quote pre-fix 0.689, never 0.765. p=0.089 belongs to hit_at_10 on the dense pair, "
     "not to correctness."),
    ("foodrag_h1_t2p", "faithfulness_judgeA", None,
     "D2: quote pre-fix 0.922, never 0.997 (corrected p falls 3.9e-56 -> 6.7e-03)."),
    ("foodrag_h1_t2p", "relevancy_judgeA", None,
     "D2: the H2 regression is -0.113 pre-fix, not -0.033."),
    ("foodrag_h1_t2p_fullday", None, None,
     "X: 614 live hallucinated empty-plan explanations. No explanation metric may be compared "
     "against a non-_fullday config."),
    ("foodrag_h1_dense_fullday", None, None,
     "X: 29 live hallucinated empty-plan explanations."),
    (None, "hit_at_10", None,
     "S: no aggregate value without its per-run health record."),
]

# WB.6 / E-1'.5 — the FOURTH convention split, on identifiers. S.4.2's three
# denominator conventions are joined by three question-id schemes. This one is
# keyed by experiment, not by config, so it is applied separately below rather
# than through GUARDS.
#
#   shared            EXP-7, EXP-10k, EXP-10k-tight   q_00000        set identity
#   config-namespaced EXP-ret-tight                   ret_dense_00000  POSITIONAL only
#   per-run           EXP-10k-tog                     r1c0_00000     neither (S.4.4)
ID_SCHEME_GUARDS: dict[str, str] = {
    "EXP-ret-tight":
        "ID SCHEME (E-1'.5): question ids are config-namespaced and pairwise "
        "DISJOINT across configs; only the 1,646 numeric suffixes align. Any join "
        "on question_id across EXP-ret-tight configs returns zero rows silently. "
        "Pair positionally, on the stripped suffix.",
    "EXP-10k-tog":
        "ID SCHEME (S.4.4): question ids are unique per (run, question), so no id "
        "is shared across runs and the run dimension collapses. See the tog guard.",
}

# LAT.2 — wherever SUSPECT-LATENCY appears it must carry its false-positive rate.
# Attached from the verdict rather than per-config so it cannot go stale.
LATENCY_FPR_GUARD = (
    "SUSPECT-LATENCY IS NOT A WORKING DETECTOR (FI.2b, LAT.2): 30% out-of-sample "
    "false-positive rate; across the 61-pair census its every independent firing "
    "is a false positive and its every true firing is redundant with VOID or "
    "CONVENTION-SPLIT. This row's health status does not depend on it.")

PLAN_NUTRITION = ("guideline_adherence_overall", "calorie_adherence")
RETRIEVAL = ("hit_at_10", "mrr_at_10", "ndcg_at_10")


def guards_for(config: str, metric: str, source_dir: Optional[str] = None) -> list[str]:
    out = []
    for c, m, d, text in GUARDS:
        if ((c is None or c == config) and (m is None or m == metric)
                and (d is None or source_dir is None or d == source_dir)):
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
    # WB.3 / FI.5 — frozen is the DEFAULT here too, so the registry's health
    # column cannot silently change when new artefacts land.
    p.add_argument("--frozen", action="store_true", default=True,
                   help="use S.3's published latency thresholds (DEFAULT)")
    p.add_argument("--refit", dest="frozen", action="store_false",
                   help="refit the thresholds on the artefacts on disk (see FI.5)")
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
            while True:
                try:
                    means[done:done + m] = rng.choice(
                        values, size=(m, values.size), replace=True).mean(axis=1)
                    break
                except (MemoryError, np.core._exceptions._ArrayMemoryError):
                    # 8 GB box with ~600 MB free commit: shrink the draw rather
                    # than abort. The rng stream is consumed in the same order
                    # either way, so the resamples are unchanged.
                    if m == 1:
                        raise
                    m = max(1, m // 4)
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
            del df
            gc.collect()
    thr = census.fit_latency_thresholds(cen_rows, frozen=args.frozen)
    # ⚠ Bucket peers by (source directory, config_id), NOT config_id alone.
    # `eval_results_ret_rerun/` (E-1') carries a `ret_text2pandas@gpt4o-mini`
    # with the same experiment_id and config_id as the superseded void run in
    # `eval_results/`. Keying on config_id alone pools them, so the void run's
    # 100%-empty peers would be compared against the fresh run's healthy ones
    # and both verdicts would be wrong.
    def _bucket(r: dict) -> tuple:
        return (Path(r["aggregate_file"]).parent.name, r["config_id"])

    by_cfg: dict[tuple, list[dict]] = {}
    for r in cen_rows:
        by_cfg.setdefault(_bucket(r), []).append(r)
    for r in cen_rows:
        r["verdict"] = "+".join(census.verdicts_for(r, by_cfg[_bucket(r)], thr))

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
                rows.append({"config": cfg, "source_dir": a.parent.name,
                             "metric": "*", "stratum": "*", "value": None,
                             "ci": "", "n": None, "runs": None, "convention": "",
                             "verdict": "NO ARTEFACT", "empty_class": "NOT CARRIED",
                             "patched": "", "source": "",
                             "guard": " | ".join(guards_for(cfg, "*", a.parent.name)) or "UNVERIFIABLE"})
                continue

            df = load(pq)
            n_patched = int(df["_safety_fixed"].notna().sum()) if "_safety_fixed" in df.columns else 0
            # WB.7 / SCH — does the PERSISTED record carry the empty-result class?
            # This is the schema column the paper's deliverable 5 is about: a
            # config whose parquet has no empty_class cannot distinguish an
            # unanswerable question from a failed one, and says so here.
            if "empty_class" in df.columns:
                hist = df["empty_class"].value_counts().to_dict()
                empty_class = " / ".join(f"{k} {v}" for k, v in sorted(hist.items()))
            else:
                empty_class = "NOT CARRIED"
            recs = df.to_dict("records")
            del df
            gc.collect()
            for r in recs:
                if isinstance(r.get("metrics"), dict):
                    r["metrics"] = dict(r["metrics"])
            exp = sorted(set(adf[adf.config_id == cfg].experiment_id))[0]
            agg = runner.aggregate(recs, experiment_id=exp, config_id=cfg)
            agg = agg[agg.stratum == "aggregate"]

            # Per-run verdicts are compound strings ("A+B"); flatten to distinct
            # atoms so a config with two compound runs does not repeat a label.
            atoms: set[str] = set()
            for x in by_cfg.get((a.parent.name, cfg), []):
                atoms.update(x["verdict"].split("+"))
            if len(atoms) > 1:
                atoms.discard("HEALTHY")   # "HEALTHY+VOID" reads as if it were both
            verdicts = sorted(atoms) or ["?"]
            conv = ("NaN-on-empty" if any(x.get("nan_on_nonempty_gold")
                                          for x in by_cfg.get((a.parent.name, cfg), []))
                    else "zero-on-empty")
            src = last_commit(pq)

            verdict_str = "+".join(verdicts)
            for _, h in agg.iterrows():
                g = guards_for(cfg, h["metric"], a.parent.name)
                if exp in ID_SCHEME_GUARDS:
                    g.append(ID_SCHEME_GUARDS[exp])
                if "SUSPECT-LATENCY" in verdict_str:
                    g.append(LATENCY_FPR_GUARD)
                rows.append({
                    "config": cfg, "source_dir": a.parent.name,
                    "metric": h["metric"], "stratum": "aggregate",
                    "value": round(float(h["mean"]), 6),
                    "ci": f"[{float(h['ci_low']):.4f}, {float(h['ci_high']):.4f}]",
                    "n": int(h["n_questions"]), "runs": int(h["n_runs"]),
                    "convention": conv, "verdict": verdict_str,
                    "empty_class": empty_class,
                    "patched": f"{n_patched} records" if n_patched else "no",
                    "source": src, "guard": " | ".join(g),
                })
            del recs, agg
            gc.collect()

    reg = pd.DataFrame(rows)
    print(f"# REG — claim registry\n\nrows: {len(reg)}   configs: {reg.config.nunique()}")
    print(f"rows carrying a guard: {int((reg.guard.astype(str) != '').sum())}")
    print(f"rows from patched artefacts: {int((reg.patched != 'no').sum())}")
    print(f"configs with a non-HEALTHY verdict: "
          f"{sorted(reg[~reg.verdict.str.fullmatch('HEALTHY', na=False)].config.unique())}\n")

    key = reg[reg.metric.isin(RETRIEVAL + ("correctness_judgeA", "faithfulness_judgeA",
                                           "allergen_violation"))]
    with pd.option_context("display.width", 250, "display.max_rows", 300):
        print(key[["config", "source_dir", "metric", "value", "n", "runs",
                   "convention", "verdict", "patched"]].to_string(index=False))

    if args.csv:
        reg.to_csv(args.csv, index=False)
        print(f"\nregistry -> {args.csv}")
    if args.md:
        with open(args.md, "w", encoding="utf-8") as fh:
            fh.write("# Claim registry (generated — do not hand-edit)\n\n")
            fh.write(f"Generated from per-question artefacts only. Rows: {len(reg)}.\n\n")
            fh.write("| config | metric | value | CI | n | runs | convention | empty_class | health | patched | source | guard |\n")
            fh.write("|---|---|---|---|---|---|---|---|---|---|---|---|\n")
            for _, r in reg.iterrows():
                fh.write(f"| {r['config']} | {r['metric']} | {r['value']} | {r['ci']} | {r['n']} | "
                         f"{r['runs']} | {r['convention']} | {r['empty_class']} | {r['verdict']} | {r['patched']} | "
                         f"{r['source']} | {r['guard']} |\n")
        print(f"registry (md) -> {args.md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

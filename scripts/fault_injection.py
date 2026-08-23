"""Check FI — fault-injection validation of the S health-detection rules.

``specs/checks-PREV-FI-E1prime-spec.md`` FI.0: S's rules were *fitted on the
failures they found* — VOID at 100% empty, DEGRADED at >=40% or >=3x peer
median, SUSPECT-LATENCY on the observed 10.5-15.0 s band all derive from the two
void ``ret_text2pandas@gpt4o-mini`` runs. Rules fitted to their own positives
have unknown sensitivity. FI measures that sensitivity prospectively by
injecting synthetic faults into the 31 HEALTHY ``(config, run)`` pairs from S'.1
and asking which rules fire.

    foodrag-fi --out eval_results_fi/

Two disciplines, both from the spec and both load-bearing:

* Do not tune a rule in response to an FI result. The rules are imported from
  :mod:`scripts.run_health_census` unchanged and driven with the thresholds *as
  published in* ``specs/checks-S-R-findings.md`` S.3 (:data:`FROZEN_THR`).
  Retuning against FI output would rebuild exactly the circularity FI exists to
  remove.
* The threshold values are **frozen constants here, not refitted**. The shipped
  census refits ``cv_threshold`` from whatever runs are on disk
  (:func:`~scripts.run_health_census.fit_latency_thresholds`), so its rule moves
  when new artefacts land. FI pins S's published values.

Every ``(fault, magnitude, k)`` cell runs ``--trials`` seeded trials; the master
seed and each cell's derived seed are recorded in the output.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from run_health_census import (  # the rules under test, imported unchanged
    HEALTHY,
    load_per_question,
    primary_metric_name,
    verdicts_for,
)

# ── the frozen rule parameters (specs/checks-S-R-findings.md S.3) ─────────────
# Do not edit these in response to an FI result. They are S's published fit.
FROZEN_THR: dict[str, Any] = {
    "family": "ret_text2pandas@*",
    "margin": 3.0,
    "cv_threshold": 0.4034,
    "max_latency_threshold": 79.13,
}

MASTER_SEED = 20260823

# S'.1's substrate: the original two aggregate directories. The E-1' re-run in
# ``eval_results_ret_rerun/`` is deliberately NOT substrate — it is out-of-sample
# and is used only for the FI.2b false-positive check.
SUBSTRATE_DIRS = ("eval_results", "eval_results_llama")
RERUN_DIR = "eval_results_ret_rerun"

PARTIAL_P = (0.1, 0.2, 0.4, 0.6, 0.8)
LATENCY_W = (0.05, 0.10, 0.20)  # uniform band half-width as a fraction of the
                                # run's own median. w=0.10 reproduces the
                                # observed void band (CV = w/sqrt(3) = 0.058 vs
                                # the 0.056/0.058 actually seen).

RULES = ["VOID_empty", "VOID_allzero", "DEG_abs", "DEG_peer",
         "LAT_peer", "LAT_fam", "CONV"]


# ── run state ────────────────────────────────────────────────────────────────

@dataclass
class RunState:
    """One (config, run) reduced to the arrays the rules actually read."""
    config_id: str
    run: int
    empty: np.ndarray          # bool, per question: retrieved nothing
    prim: np.ndarray           # float, primary metric (NaN allowed)
    lat: np.ndarray            # float, latencies with NaN dropped (may be empty)
    gold_nonempty: np.ndarray  # bool, answer_set_size > 0

    def copy(self) -> "RunState":
        return RunState(self.config_id, self.run, self.empty.copy(),
                        self.prim.copy(), self.lat.copy(), self.gold_nonempty)

    def record(self) -> dict:
        """The dict shape :func:`verdicts_for` consumes."""
        r: dict[str, Any] = {
            "config_id": self.config_id,
            "run": self.run,
            "empty_rate": float(self.empty.mean()),
            "primary_all_zero_rate": float(
                (np.nan_to_num(self.prim, nan=0.0) == 0).mean()),
            "nan_on_nonempty_gold": int(
                (np.isnan(self.prim) & self.gold_nonempty).sum()),
        }
        if self.lat.size:
            m = float(self.lat.mean())
            r["lat_cv"] = float(self.lat.std() / m) if m else float("nan")
            r["lat_max"] = float(self.lat.max())
            r["lat_med"] = float(np.median(self.lat))
        return r


def build_states(root: Path, dirs: tuple[str, ...]) -> list[RunState]:
    out: list[RunState] = []
    for d in dirs:
        agg = root / d / "aggregate.csv"
        if not agg.exists():
            continue
        adf = pd.read_csv(agg, keep_default_na=False, na_values=[""])
        for cfg in sorted(set(adf.config_id)):
            df = load_per_question(agg, cfg)
            if df is None or "run" not in df.columns:
                continue
            pm = primary_metric_name(df)
            if pm is None or "retrieved_ids" not in df.columns:
                continue
            for run, g in df.groupby("run"):
                empty = g["retrieved_ids"].apply(
                    lambda a: a is None or len(a) == 0).to_numpy(dtype=bool)
                prim = g["metrics"].apply(
                    lambda m: float(m.get(pm)) if isinstance(m, dict)
                    and m.get(pm) is not None else float("nan")
                ).to_numpy(dtype=float)
                lat = g["metrics"].apply(
                    lambda m: float(m.get("latency_s")) if isinstance(m, dict)
                    and m.get("latency_s") is not None else float("nan")
                ).to_numpy(dtype=float)
                lat = lat[~np.isnan(lat)]
                gold = ((g["answer_set_size"].to_numpy(dtype=float) > 0)
                        if "answer_set_size" in g.columns
                        else np.ones(len(g), dtype=bool))
                out.append(RunState(cfg, int(run), empty, prim, lat, gold))
    return out


# ── the individual frozen rules, decomposed ──────────────────────────────────
# verdicts_for() returns a combined verdict and suppresses DEGRADED once VOID
# fires. FI.4 asks for a rule x fault x k matrix, so each rule is also evaluated
# independently here. Same predicates, same constants.

def rule_fires(r: dict, peers: list[dict]) -> dict[str, Optional[bool]]:
    er = r.get("empty_rate")
    others = [p["empty_rate"] for p in peers if p.get("empty_rate") is not None]
    med = float(np.median(others)) if others else None
    cv, mx = r.get("lat_cv"), r.get("lat_max")
    peer_cv = [p["lat_cv"] for p in peers
               if p.get("lat_cv") is not None and not math.isnan(p["lat_cv"])]
    peer_mx = [p["lat_max"] for p in peers if p.get("lat_max") is not None]
    has_lat = cv is not None and mx is not None and not math.isnan(cv)

    # None means inapplicable — no peer to compare against, no latency
    # persisted, wrong config family — and is counted separately from a miss.
    return {
        "VOID_empty": er == 1.0,
        "VOID_allzero": r.get("primary_all_zero_rate") == 1.0,
        "DEG_abs": er is not None and er >= 0.40,
        "DEG_peer": None if med is None else bool(med > 0 and er >= 3 * med),
        "LAT_peer": (None if not (has_lat and peer_cv and peer_mx)
                     else bool(cv * FROZEN_THR["margin"] < min(peer_cv)
                               and mx * FROZEN_THR["margin"] < min(peer_mx))),
        "LAT_fam": (None if not (has_lat and r["config_id"].startswith("ret_text2pandas@"))
                    else bool(cv < FROZEN_THR["cv_threshold"]
                              and mx < FROZEN_THR["max_latency_threshold"])),
        "CONV": bool(r.get("nan_on_nonempty_gold")),
    }


def composite(r: dict, peers: list[dict]) -> str:
    """The shipped verdict, from the shipped code, on the frozen thresholds."""
    return "+".join(verdicts_for(r, peers + [r], FROZEN_THR))


# ── injections ───────────────────────────────────────────────────────────────

def inject(s: RunState, fault: str, mag: Optional[float],
           rng: np.random.Generator) -> RunState:
    t = s.copy()
    if fault == "total_void":
        t.empty[:] = True
        t.prim[:] = 0.0
    elif fault == "partial_degrade":
        live = np.flatnonzero(~t.empty)
        n = int(round(mag * live.size))
        if n:
            pick = rng.choice(live, size=n, replace=False)
            t.empty[pick] = True
            t.prim[pick] = 0.0
    elif fault == "convention_flip":
        t.prim[t.empty] = np.nan          # empties written NaN instead of 0
    elif fault == "latency_collapse":
        if t.lat.size:
            med = float(np.median(t.lat))
            t.lat = rng.uniform(med * (1 - mag), med * (1 + mag), size=t.lat.size)
    else:
        raise ValueError(fault)
    return t


def applicable(s: RunState, fault: str) -> bool:
    """A fault that cannot alter a run is not a miss — it is not injected."""
    if fault == "convention_flip":
        return bool(s.empty.any())        # nothing to flip if never empty
    if fault == "latency_collapse":
        return s.lat.size > 0             # llm_only/naive_rag persist no latency
    if fault == "partial_degrade":
        return bool((~s.empty).any())
    return True


def _seed(*parts: Any) -> int:
    """Deterministic across processes — ``hash()`` is salted per interpreter."""
    h = hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).hexdigest()
    return (MASTER_SEED + int(h[:8], 16)) % (2 ** 31)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if not n:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


# ── main ─────────────────────────────────────────────────────────────────────

def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default=".")
    p.add_argument("--out", default="eval_results_fi")
    p.add_argument("--trials", type=int, default=100)
    args = p.parse_args(argv)
    root = Path(args.root).resolve()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print("# Check FI — fault-injection validation\n")
    print(f"master seed : {MASTER_SEED}")
    print(f"trials/cell : {args.trials}")
    print("frozen rule parameters (NOT refitted):")
    for k, v in FROZEN_THR.items():
        print(f"  {k:24s} {v}")

    # ── substrate: HEALTHY pairs only, under the frozen rules ──
    states = build_states(root, SUBSTRATE_DIRS)
    by_cfg: dict[str, list[RunState]] = {}
    for s in states:
        by_cfg.setdefault(s.config_id, []).append(s)

    healthy: dict[str, list[RunState]] = {}
    base_rows = []
    for cfg, runs in by_cfg.items():
        recs = [s.record() for s in runs]
        for s, r in zip(runs, recs):
            peers = [x for x in recs if x["run"] != r["run"]]
            v = composite(r, peers)
            base_rows.append({"config_id": cfg, "run": s.run, "verdict": v,
                              "empty_rate": r["empty_rate"],
                              "lat_cv": r.get("lat_cv"), "lat_max": r.get("lat_max"),
                              "n_questions": int(s.empty.size)})
            if v == HEALTHY:
                healthy.setdefault(cfg, []).append(s)

    n_healthy = sum(len(v) for v in healthy.values())
    print(f"\n## FI.1 — substrate\n\n{n_healthy} HEALTHY (config, run) pairs "
          f"across {len(healthy)} configs, from {list(SUBSTRATE_DIRS)}\n")
    sub = pd.DataFrame([{"config_id": c, "R_healthy": len(v),
                         "runs": ",".join(str(s.run) for s in v),
                         "n_questions": int(v[0].empty.size),
                         "has_latency": bool(v[0].lat.size),
                         "base_empty_rate": round(float(v[0].empty.mean()), 4)}
                        for c, v in sorted(healthy.items())])
    print(sub.to_string(index=False))
    sub.to_csv(out / "FI_substrate.csv", index=False)
    pd.DataFrame(base_rows).to_csv(out / "FI_substrate_verdicts.csv", index=False)

    # ── FI.2 false positives on uninjected runs ──
    fp_rows = []

    def _fp_block(source: str, groups: dict[str, list[RunState]]) -> None:
        for cfg, runs in sorted(groups.items()):
            recs = [s.record() for s in runs]
            for s, r in zip(runs, recs):
                peers = [x for x in recs if x["run"] != r["run"]]
                f = rule_fires(r, peers)
                fp_rows.append({"source": source, "config_id": cfg, "run": s.run,
                                "composite": composite(r, peers),
                                **{k: ("" if v is None else int(v))
                                   for k, v in f.items()}})

    _fp_block("substrate", healthy)

    # FI.2b — out-of-sample: the E-1' re-run, which S never saw
    rr_by_cfg: dict[str, list[RunState]] = {}
    for s in build_states(root, (RERUN_DIR,)):
        rr_by_cfg.setdefault(s.config_id, []).append(s)
    _fp_block("E-1'_rerun", rr_by_cfg)

    fp = pd.DataFrame(fp_rows)
    fp.to_csv(out / "FI_false_positives.csv", index=False)

    # ── the injection grid ──
    grid: list[tuple[str, Optional[float]]] = (
        [("total_void", None)]
        + [("partial_degrade", q) for q in PARTIAL_P]
        + [("convention_flip", None)]
        + [("latency_collapse", w) for w in LATENCY_W]
    )

    cells: list[dict] = []
    for cfg, runs in sorted(healthy.items()):
        R = len(runs)
        for fault, mag in grid:
            if not all(applicable(s, fault) for s in runs):
                cells.append({"config_id": cfg, "R": R, "fault": fault,
                              "magnitude": mag, "k": None, "k_over_R": None,
                              "applicable": 0, "trials": 0, "injected_runs": 0})
                continue
            for k in range(1, R + 1):
                seed_base = _seed(cfg, fault, mag, k)
                hits = {r: 0 for r in RULES}
                inapp = {r: 0 for r in RULES}
                any_hit_trials = collateral = collateral_n = 0
                n_inj = detected_composite = 0
                ach_empty: list[float] = []
                ach_cv: list[float] = []
                for t in range(args.trials):
                    rng = np.random.default_rng(seed_base + t)
                    idx = set(rng.choice(R, size=k, replace=False).tolist())
                    st = [inject(runs[i], fault, mag, rng) if i in idx else runs[i]
                          for i in range(R)]
                    recs = [s.record() for s in st]
                    trial_hit = False
                    for i in range(R):
                        peers = [recs[j] for j in range(R) if j != i]
                        f = rule_fires(recs[i], peers)
                        fired = any(v for v in f.values() if v)
                        if i in idx:
                            n_inj += 1
                            ach_empty.append(recs[i]["empty_rate"])
                            if recs[i].get("lat_cv") is not None:
                                ach_cv.append(recs[i]["lat_cv"])
                            for rn, v in f.items():
                                if v is None:
                                    inapp[rn] += 1
                                elif v:
                                    hits[rn] += 1
                            if composite(recs[i], peers) != HEALTHY:
                                detected_composite += 1
                            trial_hit = trial_hit or fired
                        else:
                            collateral_n += 1
                            collateral += int(fired)
                    any_hit_trials += int(trial_hit)
                lo, hi = wilson(detected_composite, n_inj)
                row = {"config_id": cfg, "R": R, "fault": fault, "magnitude": mag,
                       "k": k, "k_over_R": round(k / R, 4), "applicable": 1,
                       "trials": args.trials, "injected_runs": n_inj,
                       "seed_base": seed_base,
                       "det_composite": round(detected_composite / n_inj, 4),
                       "det_composite_lo": round(lo, 4),
                       "det_composite_hi": round(hi, 4),
                       "det_cell_any": round(any_hit_trials / args.trials, 4),
                       "achieved_empty_rate": round(float(np.mean(ach_empty)), 6),
                       "achieved_lat_cv": (round(float(np.mean(ach_cv)), 6)
                                           if ach_cv else None),
                       "collateral_fp": (round(collateral / collateral_n, 4)
                                         if collateral_n else None)}
                for rn in RULES:
                    n_app = n_inj - inapp[rn]
                    row[f"det_{rn}"] = round(hits[rn] / n_app, 4) if n_app else None
                    row[f"napp_{rn}"] = n_app
                cells.append(row)

    cdf = pd.DataFrame(cells)
    cdf.to_csv(out / "FI_cells.csv", index=False)

    # ── FI.4 headline matrix: rule x fault x k/R, pooled over configs ──
    live = cdf[cdf.applicable == 1].copy()
    recs_m = []
    for (fault, mag, R, k), g in live.groupby(["fault", "magnitude", "R", "k"],
                                              dropna=False):
        row = {"fault": fault, "magnitude": mag, "R": int(R), "k": int(k),
               "k_over_R": round(k / R, 4), "configs": int(len(g)),
               "injected_runs": int(g.injected_runs.sum()),
               "achieved_empty_rate": round(float(
                   (g.achieved_empty_rate * g.injected_runs).sum()
                   / g.injected_runs.sum()), 6)}
        for rn in RULES:
            n_app = int(g[f"napp_{rn}"].sum())
            row[f"det_{rn}"] = (round(float((g[f"det_{rn}"].fillna(0)
                                             * g[f"napp_{rn}"]).sum()) / n_app, 4)
                                if n_app else None)
            row[f"napp_{rn}"] = n_app
        row["det_composite"] = round(float((g.det_composite * g.injected_runs).sum()
                                           / g.injected_runs.sum()), 4)
        recs_m.append(row)
    mat = pd.DataFrame(recs_m).sort_values(["fault", "magnitude", "R", "k"])
    mat.to_csv(out / "FI_results.csv", index=False)

    print("\n## FI.4 — detection rate by fault x k/R (pooled over configs)\n")
    show = ["fault", "magnitude", "R", "k", "k_over_R", "configs",
            "injected_runs", "achieved_empty_rate"] + \
           [f"det_{r}" for r in RULES] + ["det_composite"]
    with pd.option_context("display.width", 250, "display.max_rows", 300):
        print(mat[show].to_string(index=False))

    # ── FI.2 masking threshold: where peer-relative detection crosses 0.5 ──
    mask_rows = []
    multi = mat[mat.R >= 2]
    for rn in ("DEG_peer", "LAT_peer"):
        for (fault, mag, R), g in multi.groupby(["fault", "magnitude", "R"],
                                                dropna=False):
            g = g[g[f"napp_{rn}"] > 0].dropna(subset=[f"det_{rn}"]).sort_values("k")
            if g.empty:
                continue
            xs = g.k_over_R.to_numpy(dtype=float)
            ys = g[f"det_{rn}"].to_numpy(dtype=float)
            # Three outcomes, and collapsing them would be misleading: a rule
            # blind to a fault is not the same as a rule masked at k/R = 0.
            thr: Optional[float] = None
            if ys[0] < 0.5:
                status = "never_detects"     # below 0.5 even at k = 1
            else:
                status = "never_masked"      # stays >= 0.5 through k = R
                for i in range(1, len(xs)):
                    if ys[i] < 0.5:
                        status = "masked"
                        thr = xs[i - 1] + (xs[i] - xs[i - 1]) * (
                            (ys[i - 1] - 0.5) / (ys[i - 1] - ys[i]))
                        break
            mask_rows.append({"rule": rn, "fault": fault, "magnitude": mag,
                              "R": int(R), "configs": int(g.configs.max()),
                              "det_at_min_k": round(float(ys[0]), 4),
                              "det_at_k_eq_R": round(float(ys[-1]), 4),
                              "status": status,
                              "masking_threshold_k_over_R":
                                  None if thr is None else round(float(thr), 4)})
    mdf = pd.DataFrame(mask_rows).sort_values(["rule", "fault", "magnitude", "R"])
    mdf.to_csv(out / "FI_masking.csv", index=False)
    print("\n## FI.2 — masking threshold (k/R at which peer-relative detection < 0.5)\n")
    print(mdf.to_string(index=False))

    print("\n## FI.2 — false positives on uninjected runs\n")
    print(fp.to_string(index=False))
    n_sub = int((fp.source == "substrate").sum())
    n_rr = int((fp.source == "E-1'_rerun").sum())
    sub_fp = fp[(fp.source == "substrate") & (fp.composite != HEALTHY)]
    rr_fp = fp[(fp.source == "E-1'_rerun") & (fp.composite != HEALTHY)]
    print(f"\n  substrate  : {len(sub_fp)} / {n_sub} flagged")
    print(f"  E-1' re-run: {len(rr_fp)} / {n_rr} flagged")

    (out / "FI_config.json").write_text(json.dumps({
        "master_seed": MASTER_SEED, "trials": args.trials,
        "frozen_thresholds": FROZEN_THR, "substrate_dirs": list(SUBSTRATE_DIRS),
        "partial_p": list(PARTIAL_P), "latency_w": list(LATENCY_W),
        "healthy_pairs": n_healthy, "rules": RULES,
    }, indent=2), encoding="utf-8")
    print(f"\nwrote -> {out}/FI_results.csv, FI_cells.csv, FI_masking.csv, "
          f"FI_false_positives.csv, FI_substrate.csv, FI_config.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# `audit/` — machine-readable backing for the `specs/*-findings.md` reports

Every number quoted in a findings report should be recoverable from a CSV or JSON
here, not only from a markdown table. Nothing in this directory is hand-edited;
each file names the command that regenerates it.

## Check tooling — reports that shipped with a generator

| file | backs | regenerate with |
|---|---|---|
| `run_health_census.json` | `checks-S-R-findings.md` S.1.1–S.4 — inventory, fitted thresholds, per-run verdicts for all 61 `(config, run)` pairs | `python scripts/run_health_census.py --frozen --json audit/run_health_census.json` |
| `artefact_provenance_ledger.csv` | `X-Sprime-REG-findings.md` S′.2 — per-artefact commit history, row counts, schema deltas | `python scripts/artefact_provenance_ledger.py --csv audit/artefact_provenance_ledger.csv` |
| `claim_registry.csv` + `.md` | `X-Sprime-REG-findings.md` REG — every reportable value with its health verdict, denominator convention, patch status and source commit | `python scripts/build_claim_registry.py --csv audit/claim_registry.csv --md audit/claim_registry.md` |

`--frozen` on the census applies check S's *published* SUSPECT-LATENCY thresholds
rather than refitting them on whatever artefacts are on disk. Use it. Refitting
moves the threshold onto the new data and cannot flag it — see
`specs/checks-PREV-FI-E1prime-findings.md` FI.5.

Check FI's own outputs live in `eval_results_fi/`, next to the artefacts they were
computed from, and are documented in that check's findings report.

## Backfilled reports — `scripts/backfill_findings.py`

Five reports predated the audit tooling; their analysis scripts lived in session
scratchpads and are gone. `backfill_findings.py` regenerates their numbers from
the authoritative artefacts.

    python scripts/backfill_findings.py            # every section
    python scripts/backfill_findings.py --only qh  # one section
    python scripts/backfill_findings.py --list     # what each section produces

| section | backs | outputs |
|---|---|---|
| `qh` | `checks-T-Q-H-findings.md` Q.1–Q.6, H.3 | `TQH_Q_per_config.csv`, `TQH_Q_per_run.csv`, `TQH_Q5_stability.csv`, `TQH_Q6_d1_d3.csv`, `TQH_H3_b2_ratios.csv` |
| `t` | `checks-T-Q-H-findings.md` T.2, T.6 | `T_column_census.csv`, `T_heuristic_labels.csv`, `T_field_support.csv`, `T_incidental_patterns.csv` |
| `a1` | `A1-corpus-size-findings.md` §2, §5.4 | `A1_populations.csv`, `A1_retriever_coverage.csv` |
| `p1` | `P1-safety-fix-contamination-findings.md` §2–§5 | `P1_fix_split.csv`, `P1_allergen_prefix.csv`, `P1_rejudge_deltas.csv`, `P1_fullday_hallucinations.csv` |
| `d2` | `D2-agentic-value-findings.md` D2.2–D2.5 | `D2_pairwise_prefix_vs_published.csv`, `D2_5_explanation_sensitivity.csv` |
| `e34` | `E3-E4-P3-P4-findings.md` E-3, E-4 | `E3_E4_reaggregated.csv` |

### It re-checks the reports, it does not just restate them

Each section declares the figures its report prints and compares them against
what it derives; `backfill_manifest.json` records `match` / `MISMATCH` per value
and a mismatch exits non-zero. **72 headline values across the six sections
currently reproduce their reports**, including `hit_at_10` to 12 dp, the
Holm-corrected p-values, and the paired-bootstrap CI.

That check has already earned its keep twice: it caught D2.2's `mean_b` being two
different numbers pre-fix (0.586072) and published (0.586022), and it is what
established that the chunked bootstrap below is exact rather than merely close.

### Inputs, and where each section can run

* `qh` needs only tracked artefacts — it runs anywhere the repo is cloned.
* `p1` and `d2` read per-question parquets out of git history at `74176e8^`
  (the safety-fix commit's parent). Reproducible from any clone, not from the
  working tree alone.
* `t` and `a1` read `data/`, which is gitignored (`.gitignore:149`). **They can
  only run on a machine that holds the corpus.** A section whose inputs are
  absent is recorded as `skipped`, not failed.

### ⚠ One output is a re-derivation, not a reproduction

`T_incidental_patterns.csv`. T.6 quotes shares for its incidental-mention pattern
classes but never publishes the regexes that produced them, and the script that
did is gone. The patterns in `backfill_findings.py` are written from the report's
prose and do not land on its numbers. Every other `t` output reproduces exactly —
including the appended-vs-native tag split, which is the load-bearing part of T.6.

The row is marked `reproduces_report = no` in the CSV so it cannot be mistaken for
a verified figure.

## A defect this work fixed

`runner._bootstrap_ci` drew its whole resample matrix in one allocation — 125 MB
for a 1,646-question config, 763 MB for a 10,000-question one — and raised
`MemoryError` on a constrained box. That is why `reaggregate_corrected.py` could
not complete, and why `build_claim_registry.py` carried a local workaround. It now
draws in row blocks. `Generator.choice` fills row-major, so the blocks consume the
rng stream in the same order as the one-shot draw and the interval is
**bit-identical** — verified across several `(n_boot, n)` shapes before the change,
and confirmed downstream by D2's CI reproducing `[0.0906, 0.1141]` exactly.

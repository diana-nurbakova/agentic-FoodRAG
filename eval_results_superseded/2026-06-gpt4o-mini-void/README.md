# Superseded: `ret_text2pandas@gpt4o-mini` (Jun–Jul 2026) — VOID

⛔ **These artefacts are evidence, not results. Do not compute a published number from them.**

Archived 2026-08-23 from repo commit `242439a`, per `specs/checks-S-R-spec.md` R.3.

## Why

Two of the five runs returned an empty result for **every one of the 1,646 tight
questions**. The published aggregate recorded this as `n_questions = 1646,
n_runs = 5` — exactly as it records a healthy config. The failure is invisible in
the aggregate; that invisibility is the finding.

| run | empty | empty rate | hit@10 | latency med / max / CV |
|---|---:|---:|---:|---|
| 1 | 118 | 7.17% | 0.880316 | 13.29 / 750.96 / 1.269 |
| 2 | 120 | 7.29% | 0.877278 | 13.48 / 93.16 / 0.514 |
| 3 | 998 | 60.63% | 0.377886 | 12.76 / 79.13 / 0.403 |
| 4 | **1,646** | **100.00%** | 0.000000 | 12.72 / 15.03 / 0.056 |
| 5 | **1,646** | **100.00%** | 0.000000 | 12.70 / 15.22 / 0.058 |

`118 + 120 + 998 + 1646 + 1646 = 4528` — the exact empty count in the parquet.

The published aggregate value **0.427096** is the mean over all five, i.e. the mean
of two healthy runs, one 60%-empty run, and two that produced nothing.

## Probable cause

Not provable from these artefacts, but every observable is consistent with
**OpenAI credit exhaustion partway through run 3**. On 2026-08-23 the account
returned `RateLimitError: You have no credits remaining`. `RateLimitError` is
retryable (`src/foodrag/llm/retry.py:20-21`), so once credits are gone each call
burns its retry budget and returns empty after a near-constant delay — which is
the uniform 10.5–15.0 s band in runs 4–5, against 2.4–751 s when healthy.

⚠ The original run left **no entry in any `data/logs/llm_calls-*.jsonl`**, and
`config/models.yml` had no `openai` provider when it ran, so **the model behind
the original `config_id` is unverifiable**. The label `gpt4o-mini` is not evidence.

## What this supports

- §7.4: a void run is invisible in the aggregate schema, which has no health channel.
- Check S: 51 `(config, run)` pairs swept; this is the only genuine VOID pair.
- The withdrawal of claim **B5** (`0.427 → 0.854` backbone swap). The comparison
  does not exist; it was not a weaker effect.

## ⛔ What must not be done with these files

- **Do not report 0.878797** (the mean of healthy runs 1–2). It is 2-of-5 selected
  after the fact — forbidden by `checks-S-R-spec.md` R.4 rules 3 and 4, and the
  reason the re-run (E-1) exists.
- **Do not merge these runs with the E-1 re-run.** Different snapshot, different
  date, and these were selected *because* they were healthy.
- Do not overwrite or delete. The replacement lands in `eval_results_ret_rerun/`.

## Contents

| file | what |
|---|---|
| `ret_text2pandas@gpt4o-mini_per_question.parquet` | 8,230 per-question records (1,646 × 5 runs) |
| `aggregate_rows.csv` | the 20 affected rows from `eval_results/aggregate.csv` + header |
| `t2p_openai_ckpt/run{1..5}.pkl` | per-run checkpoints; run4/run5 are 440,881 bytes each |

The originals remain in place at `eval_results/` and
`data/qa/main_10k_tight/t2p_openai_ckpt/`. This is a copy, not a move.

## See also

- `specs/checks-T-Q-H-findings.md` — check Q, where the void runs were found
- `specs/checks-S-R-findings.md` — check S's census and R's pre-registration
- `specs/experiment-specification.md` — E-1, the replacement run

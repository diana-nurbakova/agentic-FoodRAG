# E-6 — Guarded, instrumented rerun: findings

*Generated 2026-10-12T19:39:24+00:00 by `scripts/e6_report.py` from `E6-results.json`. One run per configuration; per-question bootstrap CIs only (10,000 resamples, seed 0); no run-to-run variance is claimed. Call these numbers "a guarded rerun", never "corrected results".*

## Status

| Tier | Configuration | Status |
|---|---|---|
| A | `foodrag_h1_t2p` | complete |
| B | `foodrag_h2` | complete |
| C | `foodrag_h2_t2p` | complete |

Attempts:

- Tier A `foodrag_h1_t2p_full_20261012T082328Z_02fc79e` (commit `02fc79e`): complete
- Tier B `foodrag_h2_full_20261012T144843Z_02fc79e` (commit `02fc79e`): aborted_by_operator
- Tier B `foodrag_h2_full_20261012T162653Z_1a66f4d` (commit `1a66f4d`): complete
- Tier C `foodrag_h2_t2p_full_20261012T180207Z_3e35e89` (commit `3e35e89`): complete

## Tier A — `foodrag_h1_t2p`

**Outcome (spec §9):** ≈0.689 — post-fix value NOT reproduced by the guarded system (spec §9 row 2).

### Correctness, Judge A — three numbers side by side (descriptive, cross-session)

| Record | Value |
|---|---|
| original, pre-fix (3 runs) | 0.689 |
| original, post-fix patch (3 runs) | 0.765 |
| **E-6 guarded rerun (1 run)** | **0.670 [0.655, 0.686] (n = 1635)** |

Caveat: descriptive; cross-session (~3 months), one E-6 run vs three original runs; provider snapshot may differ (generator served as Llama-3.3-70B-Instruct-Turbo).

### Pre-registered quantities (§8.2)

| Quantity | Value |
|---|---|
| Items / health verdict | 1646 / HEALTHY |
| Status | zero_rows 133, ok 1502, llm_error 8, judge_error 3 |
| Empty rate (final plan has no recipe) | 0.086 |
| Filter removals (items with ≥ 1) | 1 (1) |
| Removals per 1,000 items / per 1,000 candidate recipes | 0.608 / 0.265 |
| Pre-fix violations per run (original, for comparison) | 1.000 |
| Recipes the filter could not inspect | 0 |
| **Gate fires** | **0** (expected ≈ 140) |
| Correctness Judge A, overall | 0.670 [0.655, 0.686] (n = 1635) |
| Correctness Judge B, overall | 0.718 [0.701, 0.734] (n = 1638) |
| Correctness A, gate-fired / not gate-fired | — [—, —] (n = 0) / 0.670 [0.655, 0.686] (n = 1635) |
| Judge disagreement \|A − B\|, correctness | 0.146 [0.137, 0.156] (n = 1635) |
| hit_at_10 | 0.794 [0.774, 0.813] (n = 1638) |
| claim_grounding_rate | 0.683 [0.666, 0.700] (n = 1565) |
| allergen_violation | 0.000 [0.000, 0.000] (n = 1638) — **0 by construction (§7.2); not a result** |

### Why 0.765 is not reproduced — the gate never fires

The empty-plan gate fired on **0** of 1646 items. Of 141 empty final plans, **133** received an LLM-written explanation instead (the remainder are `llm_error` items with no explanation). `ExplanationTool.explain` gates the *substituted* plan whenever one is non-empty, and the Substitution agent produces one even when every slot is empty, so the gate never sees the empty plan. The retroactive patch (`74176e8`) replaced every empty-plan explanation with the fixed message; the guarded pipeline never does. The pre-registered gate-fired split (§8.2.3) is therefore empty; the closest substitute is below.

**EXPLORATORY — not pre-registered** — correctness on empty vs non-empty final plans:

| Judge | Empty plans | Non-empty plans |
|---|---|---|
| judgeA | 0.061 [0.028, 0.098] (n = 132) | 0.724 [0.709, 0.738] (n = 1503) |
| judgeB | 0.137 [0.102, 0.175] (n = 133) | 0.769 [0.754, 0.784] (n = 1505) |

### Filter interventions

- `q_01495`: removed https://www.food.com/recipe/lindas-lobster-rangoon-205992 (matched: green onions); 3 candidates → 1 returned.

### Provenance

- Run `foodrag_h1_t2p_full_20261012T082328Z_02fc79e`, commit `02fc79e`, started 2026-10-12T08:23:29+00:00, 12 workers, time pin `13.04.2025 morning`.
- Question set sha256 `700d5f6ec0e1e146…`, corpus `7f329bfc5e21a873…`, t2p replay `278e74bd441f8b0f…`.
- Provider-reported models: `openai/meta-llama/Llama-3.3-70B-Instruct` → `meta-llama/Llama-3.3-70B-Instruct-Turbo` ×32045; `openai/google/gemma-3-27b-it` → `google/gemma-3-27b-it` ×6552; `openai/deepseek-ai/DeepSeek-V3` → `deepseek-ai/DeepSeek-V3` ×6552.
- Item wall time: median 177.467 s, p95 443.010 s.

## Tier B — `foodrag_h2`

| Quantity | Value |
|---|---|
| Items / health verdict | 1646 / HEALTHY |
| Status | ok 1646 |
| Filter removals (items with ≥ 1) | 21 (20) |
| Removals per 1,000 candidate recipes | 5.337 |
| Pre-fix violations per run (original) | 13.000 |
| Correctness Judge A / B | 0.257 [0.246, 0.269] (n = 1646) / 0.223 [0.212, 0.235] (n = 1646) |

## Tier C — `foodrag_h2_t2p`

| Quantity | Value |
|---|---|
| Items / health verdict | 1646 / HEALTHY |
| Status | zero_rows 133, ok 1513 |
| Filter removals (items with ≥ 1) | 3 (1) |
| Removals per 1,000 candidate recipes | 0.826 |
| Pre-fix violations per run (original) | 1.000 |
| Correctness Judge A / B | 0.587 [0.572, 0.603] (n = 1646) / 0.598 [0.581, 0.615] (n = 1646) |

## Gain, Tier A − Tier C (Judge A correctness)

the only gain E-6 may report (fresh-vs-fresh, same session, same time pin): **0.083 [0.067, 0.099] (n = 1635)**, n_paired = 1635.

## Standing caveats

- One run per configuration; CIs are over questions only.
- `allergen_violation = 0.000` is by construction and is never a finding.
- Cross-session comparisons with the original records are descriptive (drift, provider snapshot).
- Deviations from the spec and post-registration events: `specs/E6-preregistration.md` §0 and §12 (untracked; its hash is committed in `_snapshot/preregistration.sha256`).

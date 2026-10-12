# Paired Significance Tests

> Values for `foodrag_h1_t2p`, `foodrag_h1_dense`, `foodrag_h2`, `foodrag_h2_t2p` are reported pre-remediation (records at `74176e8^`). Commit `74176e8` filtered excluded ingredients from 72 stored outputs and replaced 421 empty-plan explanations, then re-scored; post-fix values appear only where labelled.

The results pipeline reports means ± SD and **bootstrap 95% CIs** (stored for
**every** `config × stratum × metric` in `eval_results/json/aggregate.json`).
The evaluation protocol additionally requires **paired**
comparison tests over per-question scores. This document reports those tests for the
interpretable FoodRAG configuration comparisons (EXP-7), for both the partial-day
(deployed) and the full-day (all-four-slots) regimes.

## Data & ground truth (context for the pairing)

All pairs below are aligned on `question_id` over the **tight** evaluation stratum:
the 1,646 personas (of 10,000 generated) whose rule-based gold answer set has 1–50
recipes. Each persona is a set of hard preference constraints (tag/ingredient
likes-dislikes, per-100 g nutrient bands, per-serving daily caps, health-score
floors); the **ground truth** is the *exhaustive* set of HUMMUS corpus recipes
(270,284 recipes) satisfying all of them, computed deterministically by
`_get_valid_recipes_for_persona` (§1.1 of `docs/RESULTS_REPORT.md`). Because gold is
rule-derived and complete, the paired per-question scores are exact and the Wilcoxon
test is over genuine within-question differences, not annotation noise. The tight set
is constraint-heavy (mean 4.99 requirements/persona) which is why `n_pairs` for some
metrics is < 1,646 (metrics undefined on empty/over-filtered plans are dropped from
the pair). Separately, 3 questions errored in all three `foodrag_h1_t2p` runs (every
metric NaN), so every pair involving `foodrag_h1_t2p` has at most 1,643 pairs.

## Method

Section references (§6, §13.3) are to the evaluation protocol, `specs/FoodRAG_Evaluation_Protocol.md`.

* **Pairing (protocol §6, items 2–3).** For a metric and a config, each question's score is the
  NaN-aware mean of that metric across the config's runs (FoodRAG: 3 runs;
  `llm_only`: 5; `naive_rag`: 1). Two configs are aligned on `question_id` (all
  cover the same 1,646 tight questions) and pairs where *either* side is NaN are
  dropped — so `n_pairs` is below 1,646 for metrics that are undefined on some
  questions (e.g. `hard_constraint_sat_perconstraint` / `guideline_adherence` on
  empty t2p plans, `claim_grounding_rate` where no NL claims were emitted).
* **Test.** Paired **Wilcoxon signed-rank** over the per-question vectors
  (`scipy.stats.wilcoxon`). All-zero-difference → p = 1.0; < 6 non-zero pairs is
  handled gracefully. A paired **bootstrap** (10,000 resamples, seed 0) of the
  mean difference is reported alongside as the §6 alternative (mean-diff 95% CI).
* **Effect size (§6).** Median per-question difference (a − b).
* **Correction (protocol §6, item 5).** **Holm–Bonferroni** within each metric family (all
  config pairs tested for that metric at this stratum). `significant` =
  corrected p < 0.05.
* **Stratum.** `aggregate` (all 1,646 tight questions).

Records: `eval_results/pairwise.csv` and `eval_results/json/pairwise.json`
(one §13.3 row per metric × config-pair). Regenerate with
`python -m foodrag.analysis.stats`. Those published records are computed on the
**post-fix** artefacts. The Family 1–3 tables below are the same computation
(`foodrag.analysis.stats.compute_pairwise` over the full `CONFIG_PAIRS` set) run on the
pre-fix records of the four FoodRAG configs (`git show 74176e8^:eval_results/<cfg>_per_question.parquet`)
and the unchanged `naive_rag` / `llm_only` records. The Holm families are unchanged by
this substitution: every metric family has the same number of config pairs pre-fix as
in `pairwise.csv` (`hit_at_10` / `mrr_at_10` 6; correctness, faithfulness, relevancy,
guideline adherence, hard-constraint satisfaction, path coverage 4; `allergen_violation` 2;
`claim_grounding_rate` 1).

**Polarity.** All metrics below are higher-is-better **except**
`allergen_violation` (lower is better) — for that metric a *negative* mean_diff
means `config_a` is safer.

Columns: mean_a, mean_b = per-question means; **p** = Wilcoxon p; **p_corr** =
Holm-corrected; **median Δ** = median per-question (a − b); **95% CI** = bootstrap
CI on the mean difference; **sig** = corrected p < 0.05.

---

## Family 1 — Agentic (H1) vs recommend-only (H2), per retriever

*Does the agentic explanation/enhancement pipeline (H1) beat the recommend-only
system (H2) holding the retriever fixed?*

### foodrag_h1_dense vs foodrag_h2 (dense retriever)
| metric | mean_a | mean_b | p | p_corr | median Δ | 95% CI | sig | n |
|---|---|---|---|---|---|---|---|---|
| correctness_judgeA | 0.286 | 0.248 | <0.001 | <0.001 | 0.083 | [0.029, 0.047] | yes | 1646 |
| faithfulness_judgeA | 0.989 | 0.232 | <0.001 | <0.001 | 1.000 | [0.742, 0.774] | yes | 1646 |
| relevancy_judgeA | 0.452 | 0.308 | <0.001 | <0.001 | 0.000 | [0.125, 0.164] | yes | 1646 |
| guideline_adherence_overall | 0.299 | 0.280 | <0.001 | <0.001 | 0.000 | [0.013, 0.026] | yes | 1646 |
| hard_constraint_sat_perconstraint | 0.663 | 0.645 | <0.001 | <0.001 | 0.000 | [0.012, 0.025] | yes | 1646 |
| mrr_at_10 | 0.018 | 0.023 | 0.035 | 0.035 | 0.000 | [-0.010, 0.000] | yes (H2 wins) | 1646 |
| hit_at_10 | 0.027 | 0.032 | 0.089 | 0.089 | 0.000 | [-0.010, 0.001] | **no** | 1646 |
| path_coverage | 0.651 | 0.655 | 0.261 | 0.261 | 0.000 | [-0.010, 0.002] | **no** | 1646 |

### foodrag_h1_t2p vs foodrag_h2_t2p (text-to-pandas retriever)
| metric | mean_a | mean_b | p | p_corr | median Δ | 95% CI | sig | n |
|---|---|---|---|---|---|---|---|---|
| correctness_judgeA | 0.689 | 0.586 | <0.001 | <0.001 | 0.083 | [0.091, 0.114] | yes | 1643 |
| faithfulness_judgeA | 0.922 | 0.892 | 0.007 | 0.007 | 0.000 | [0.012, 0.047] | yes | 1643 |
| hit_at_10 | 0.773 | 0.708 | <0.001 | <0.001 | 0.000 | [0.052, 0.077] | yes | 1643 |
| mrr_at_10 | 0.738 | 0.708 | <0.001 | <0.001 | 0.000 | [0.021, 0.039] | yes | 1643 |
| path_coverage | 0.883 | 0.867 | <0.001 | <0.001 | 0.000 | [0.012, 0.019] | yes | 1643 |
| relevancy_judgeA | 0.788 | 0.901 | <0.001 | <0.001 | 0.000 | [-0.132, -0.094] | yes (H2 wins) | 1643 |
| guideline_adherence_overall | 0.266 | 0.276 | 0.017 | 0.034 | 0.000 | [-0.016, -0.003] | yes (H2 wins) | 1516 |
| hard_constraint_sat_perconstraint | 0.977 | 0.976 | 0.777 | 0.777 | 0.000 | [-0.003, 0.003] | **no** | 1516 |

*Post-fix (published `pairwise.csv`, after the 421 replaced explanations were re-scored; not the evaluated system):* correctness 0.765 vs 0.586 (corrected p = 8.5e-88, median Δ = 0.167); faithfulness 0.997 vs 0.892 (corrected p = 3.9e-56); relevancy 0.869 vs 0.902 (Δ −0.033).

**Read.** H1 raises **faithfulness** and **correctness** over H2 under both
retrievers (e.g. dense: faithfulness 0.989 vs 0.232, median Δ = 1.0, and correctness
0.286 vs 0.248, corrected p = 1.4e-19; t2p: correctness 0.689 vs 0.586, corrected
p = 1.9e-58, median Δ = 0.083, 95% CI [0.091, 0.114]) — the agentic layer's headline win.
On t2p the faithfulness lift is small (0.922 vs 0.892, corrected p = 6.7e-3). Retrieval
quality (hit@10 / mrr@10) is *not* improved by the agentic layer on the dense retriever
(hit@10 n.s., corrected p = 0.089; mrr@10 marginally *lower*, corrected p = 0.035),
consistent with the agentic layer acting on explanation/answer quality, not first-stage
retrieval. On t2p, H1 *loses* relevancy to H2 by **−0.113** (0.788 vs 0.901, corrected
p = 3.9e-33; the published post-fix figure, −0.033, understates it) and slightly loses
guideline adherence — a significant regression worth noting.

## Family 2 — Retriever sensitivity (text-to-pandas vs dense), per system

*How much does swapping the dense retriever for text-to-pandas (t2p) move each
system?*

### foodrag_h1_t2p vs foodrag_h1_dense (agentic system)
| metric | mean_a | mean_b | p | p_corr | median Δ | 95% CI | sig | n |
|---|---|---|---|---|---|---|---|---|
| hit_at_10 | 0.773 | 0.027 | <0.001 | <0.001 | 1.000 | [0.726, 0.765] | yes | 1643 |
| mrr_at_10 | 0.738 | 0.018 | <0.001 | <0.001 | 1.000 | [0.700, 0.740] | yes | 1643 |
| correctness_judgeA | 0.689 | 0.286 | <0.001 | <0.001 | 0.500 | [0.386, 0.420] | yes | 1643 |
| relevancy_judgeA | 0.788 | 0.452 | <0.001 | <0.001 | 0.333 | [0.312, 0.360] | yes | 1643 |
| hard_constraint_sat_perconstraint | 0.977 | 0.657 | <0.001 | <0.001 | 0.333 | [0.306, 0.333] | yes | 1516 |
| path_coverage | 0.883 | 0.651 | <0.001 | <0.001 | 0.267 | [0.216, 0.247] | yes | 1643 |
| claim_grounding_rate | 0.651 | 0.551 | <0.001 | <0.001 | 0.126 | [0.082, 0.118] | yes | 1626 |
| faithfulness_judgeA | 0.922 | 0.989 | <0.001 | <0.001 | 0.000 | [-0.080, -0.055] | yes (dense wins) | 1643 |
| guideline_adherence_overall | 0.266 | 0.297 | <0.001 | <0.001 | -0.042 | [-0.040, -0.023] | yes (dense wins) | 1516 |

*Post-fix (published `pairwise.csv`; not the evaluated system):* correctness 0.765 vs 0.285; relevancy 0.869 vs 0.455; claim_grounding_rate 0.712 vs 0.552 (n = 1,499); faithfulness 0.997 vs 0.989 (t2p higher).
Pre-fix, faithfulness runs the other way: H1-dense is higher.

### foodrag_h2_t2p vs foodrag_h2 (recommend-only system)
| metric | mean_a | mean_b | p | p_corr | median Δ | 95% CI | sig | n |
|---|---|---|---|---|---|---|---|---|
| hit_at_10 | 0.709 | 0.032 | <0.001 | <0.001 | 1.000 | [0.655, 0.699] | yes | 1646 |
| mrr_at_10 | 0.709 | 0.023 | <0.001 | <0.001 | 1.000 | [0.664, 0.708] | yes | 1646 |
| correctness_judgeA | 0.586 | 0.248 | <0.001 | <0.001 | 0.333 | [0.321, 0.356] | yes | 1646 |
| relevancy_judgeA | 0.901 | 0.308 | <0.001 | <0.001 | 0.667 | [0.572, 0.615] | yes | 1646 |
| faithfulness_judgeA | 0.893 | 0.232 | <0.001 | <0.001 | 0.667 | [0.642, 0.680] | yes | 1646 |
| hard_constraint_sat_perconstraint | 0.976 | 0.636 | <0.001 | <0.001 | 0.333 | [0.327, 0.354] | yes | 1519 |
| path_coverage | 0.868 | 0.655 | <0.001 | <0.001 | 0.250 | [0.197, 0.228] | yes | 1646 |
| guideline_adherence_overall | 0.276 | 0.279 | 0.078 | 0.078 | 0.000 | [-0.012, 0.005] | **no** | 1519 |

**Read.** The retriever swap is the **dominant** effect in the whole study.
Text-to-pandas lifts hit@10 from ~0.03 to ~0.77 (H1) / ~0.71 (H2) (median Δ = 1.0
for both; 95% CI [0.73, 0.77] H1, [0.66, 0.70] H2) — a categorical retrieval win that dense retrieval never
approaches on these tight, constraint-heavy questions. It also carries large
gains in correctness, relevancy, hard-constraint satisfaction and path coverage.
The exceptions: guideline adherence is *not* improved (H1: slightly worse with
t2p; H2: n.s.), and in the agentic system faithfulness is *lower* with t2p (0.922 vs
0.989, corrected p < 0.001), largely because of t2p's 421 empty-plan records, whose
explanations narrate plans that were never produced (Judge-A faithfulness 0.119 on those
records).

## Family 3 — FoodRAG-best (H1-t2p) vs baselines (shared metrics)

*Shared metrics only — the baselines carry no judge/explainability metrics.*

### foodrag_h1_t2p vs naive_rag
| metric | mean_a | mean_b | p | p_corr | median Δ | 95% CI | sig | n |
|---|---|---|---|---|---|---|---|---|
| hit_at_10 | 0.773 | 0.055 | <0.001 | <0.001 | 1.000 | [0.696, 0.739] | yes | 1643 |
| mrr_at_10 | 0.738 | 0.023 | <0.001 | <0.001 | 1.000 | [0.695, 0.735] | yes | 1643 |
| allergen_violation | 0.0003 | 0.0076 | <0.001 | <0.001 | 0.000 | [-0.010, -0.005] | yes (FoodRAG lower) | 1643 |

### foodrag_h1_t2p vs llm_only
| metric | mean_a | mean_b | p | p_corr | median Δ | 95% CI | sig | n |
|---|---|---|---|---|---|---|---|---|
| hit_at_10 | 0.773 | 0.006 | <0.001 | <0.001 | 1.000 | [0.748, 0.786] | yes | 1643 |
| mrr_at_10 | 0.738 | 0.003 | <0.001 | <0.001 | 1.000 | [0.716, 0.755] | yes | 1643 |
| allergen_violation | 0.0003 | 0.9048 | <0.001 | <0.001 | -0.980 | [-0.912, -0.897] | yes (FoodRAG lower) | 1643 |

**Read.** FoodRAG-best dominates both baselines on retrieval (hit@10 0.773 vs
0.055 / 0.006, median Δ = 1.0). The `llm_only` allergen row is **not a safety
comparison**: LLM-only returns non-existent recipes for 90.5% of items, so its
constraint compliance cannot be verified. `allergen_violation` counts unresolvable
recipes as violations and therefore equals the hallucination rate on this
configuration; it does not measure exposure to excluded ingredients. Against
`naive_rag`, whose recipes resolve, FoodRAG's pre-fix allergen rate is lower (0.0003
vs 0.0076, corrected p < 0.001). The published post-fix value for FoodRAG, 0.000,
results from a filter applied to stored outputs after evaluation (72 records across
the four FoodRAG configs) and holds by construction, because filter and metric are
the same function.

---

## Family 4 — Full-day variants (all-four-slots regime)

A separate Holm family (records: `eval_results/pairwise_fullday.csv` /
`eval_results/json/pairwise_fullday.json`; regenerate with the same
`python -m foodrag.analysis.stats`). These pairs force `requested_meal_types =
[breakfast, lunch, dinner, snack]` so the plan-structure metrics (`*_full` /
`*_norm`) become interpretable. The family also adds the *scope-effect* pairs
(same `question_id`, partial-day vs full-day) to separate slot-count artifacts from
genuine limitations. Selected rows (corrected p; median Δ = a − b):

### Agentic (H1) vs recommend-only (H2), full-day

| pair | metric | mean_a | mean_b | p_corr | median Δ | sig |
|---|---|---|---|---|---|---|
| h1_t2p_fullday vs h2_t2p_fullday | correctness_judgeA | 0.561 | 0.718 | 4.5e-115 | -0.083 | yes (H2 wins) |
| h1_t2p_fullday vs h2_t2p_fullday | meal_slot_completeness_norm | 0.782 | 0.731 | 2.0e-36 | 0.000 | yes (H1 wins) |
| h1_t2p_fullday vs h2_t2p_fullday | hit_at_10 | 0.786 | 0.828 | 8.7e-19 | 0.000 | yes (H2 wins) |
| h1_dense_fullday vs h2_fullday | correctness_judgeA | 0.281 | 0.262 | 1.3e-06 | 0.000 | yes (H1 wins) |
| h1_dense_fullday vs h2_fullday | hit_at_10 | 0.049 | 0.046 | 7.3e-01 | 0.000 | **no** |

### Retriever sensitivity (t2p vs dense), full-day

| pair | metric | mean_a | mean_b | p_corr | median Δ | sig |
|---|---|---|---|---|---|---|
| h1_t2p_fullday vs h1_dense_fullday | hit_at_10 | 0.786 | 0.049 | 1.2e-251 | +1.000 | yes (t2p) |
| h1_t2p_fullday vs h1_dense_fullday | correctness_judgeA | 0.561 | 0.281 | 2.3e-156 | +0.333 | yes (t2p) |
| h1_t2p_fullday vs h1_dense_fullday | structural_validity_norm | 0.022 | 0.392 | 1.5e-165 | -0.333 | yes (dense) |
| h1_t2p_fullday vs h1_dense_fullday | meal_slot_completeness_norm | 0.782 | 0.990 | 5.3e-75 | 0.000 | yes (dense) |
| h1_t2p_fullday vs h1_dense_fullday | guideline_adherence_norm | 0.279 | 0.215 | 7.3e-28 | +0.083 | yes (t2p) |
| h2_t2p_fullday vs h2_fullday | hit_at_10 | 0.828 | 0.046 | 1.2e-270 | +1.000 | yes (t2p) |
| h2_t2p_fullday vs h2_fullday | correctness_judgeA | 0.718 | 0.262 | 5.5e-227 | +0.500 | yes (t2p) |

### Scope effect (partial-day vs full-day, same questions)

| pair | metric | mean_a | mean_b | p_corr | median Δ | sig |
|---|---|---|---|---|---|---|
| h1_t2p vs h1_t2p_fullday | correctness_judgeA | 0.688 | 0.561 | 1.3e-78 | +0.083 | yes (partial higher) |
| h1_t2p vs h1_t2p_fullday | faithfulness_judgeA | 0.922 | 0.827 | 4.4e-41 | 0.000 | yes (partial higher) |
| h1_t2p vs h1_t2p_fullday | relevancy_judgeA | 0.788 | 0.584 | 2.9e-87 | 0.000 | yes (partial higher) |
| h1_t2p vs h1_t2p_fullday | meal_slot_completeness_norm † | 0.913 | 0.782 | 7.1e-48 | 0.000 | yes (partial higher) |
| h1_t2p vs h1_t2p_fullday | guideline_adherence_overall | 0.266 | 0.239 | 3.2e-13 | 0.000 | yes (partial higher) |
| h2 vs h2_fullday | guideline_adherence_overall | 0.279 | 0.209 | 2.0e-55 | +0.083 | yes (partial higher) |
| h2 vs h2_fullday | structural_validity_norm † | 0.189 | 0.362 | 1.4e-59 | 0.000 | yes (full higher) |
| h2 vs h2_fullday | hit_at_10 | 0.032 | 0.046 | 2.1e-04 | 0.000 | yes (full higher) |

The partial-day side of these rows is the pre-fix record set, recomputed with
`compute_pairwise` over the full-day `CONFIG_PAIRS_FULLDAY` family. † The slot-normalized
metrics were added after `74176e8` and exist only on the post-fix records; they are not
explanation-facing, and the remediation did not change the plans they score beyond the 72
filtered records, so those two rows are taken from `pairwise_fullday.csv`.

⛔ **Comparability guard (X.2).** No explanation-facing metric (`faithfulness_*`,
`relevancy_*`, `explanation_quality_*`, `claim_grounding_rate`, `unsupported_claim_rate`,
`correctness_*`) may be compared between a `*_fullday` config and a *post-fix* non-`_fullday`
config. The published `pairwise_fullday.csv` rows for `h1_t2p` vs `h1_t2p_fullday` do exactly
that: the partial-day side has its 421 empty-plan explanations replaced by a template, while
the full-day side keeps 614 hallucinated ones. Between 28% and 44% of each published gap is
the patch (published: correctness +0.204, faithfulness +0.170, relevancy +0.285; pre-fix:
+0.127, +0.094, +0.204). Only the pre-fix figures above are like-for-like.

**Read.** The full-day regime confirms two separate stories with paired tests:
(i) the four-slot **completeness/structural** metrics (`*_full`) rise going
partial→full for every config (all corrected p < 1e-97) — the low partial-day readings
were a **slot-count artifact** (the slot-normalized `*_norm` variants do not rise
uniformly: e.g. h1_t2p completeness_norm 0.913→0.782; only h2's structural_norm
rises, 0.189→0.362); (ii) **guideline
adherence** *falls* going partial→full (h2 0.279→0.209, p = 2.0e-55; h1_t2p
0.266→0.239, p = 3.2e-13) — a **genuine limitation** that a full day *worsens*, not
fixes. Within full-day, the retriever swap remains the dominant effect (hit@10 +0.74
to +0.78, p < 1.2e-251), but it **trades off** against slot coverage: dense fills
four distinct slots (completeness_norm 0.990, structural_norm 0.392) where t2p's
replay pool exhausts (0.782 / 0.022, p < 5.3e-75).

---

## Headline results

* **H1-t2p vs H2-t2p — correctness:** H1 beats H2 (0.689 vs 0.586; Wilcoxon
  corrected p = 1.9e-58, median Δ = 0.083, 95% CI [0.091, 0.114]). The agentic
  layer's clearest quality win. Under dense retrieval the same contrast is
  0.286 vs 0.248 (corrected p = 1.4e-19) — smaller but also significant.
* **H1-t2p vs H1-dense — hit@10:** t2p beats dense (0.773 vs 0.027; p < 0.001,
  median Δ = 1.000, 95% CI [0.726, 0.765]). The retriever swap, not the agentic
  layer, drives retrieval.
* **Faithfulness, agentic vs recommend-only (dense):** H1 beats H2 (0.989 vs
  0.232; p < 0.001, median Δ = 1.000, 95% CI [0.742, 0.774]).
* **Allergen violation vs `llm_only`:** not a safety comparison. LLM-only
  returns non-existent recipes for 90.5% of items, so its constraint compliance
  cannot be verified; its `allergen_violation` (0.905) equals its hallucination rate
  and does not measure exposure to excluded ingredients.

### Non-significant / reversed comparisons (called out)

* **hit@10, H1-dense vs H2 (dense):** *not* significant (0.027 vs 0.032,
  corrected p = 0.089) — the agentic layer does not improve dense retrieval; mrr@10
  is in fact marginally *lower* for H1-dense (0.018 vs 0.023, corrected p = 0.035).
* **path_coverage, H1-dense vs H2:** *not* significant (corrected p = 0.261).
* **hard_constraint_sat, H1-t2p vs H2-t2p:** *not* significant (0.977 vs 0.976,
  corrected p = 0.777) — both t2p systems already near-saturate this.
* **faithfulness, H1-t2p vs H1-dense:** *reversed* — H1-dense is higher (0.989 vs
  0.922, corrected p < 0.001); the published table shows the opposite direction
  (post-fix: 0.997 vs 0.989).
* **guideline_adherence, H2-t2p vs H2:** *not* significant (corrected p = 0.078);
  and H1-t2p is *worse* than H2-t2p on guideline adherence (0.266 vs 0.276,
  corrected p = 0.034) and relevancy (0.788 vs 0.901, **−0.113**, corrected
  p = 3.9e-33; published post-fix −0.033) — significant regressions from the agentic
  layer under t2p.

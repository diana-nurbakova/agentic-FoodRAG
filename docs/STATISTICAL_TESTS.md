# Paired Significance Tests (Protocol §6 / §13.3)

The results pipeline reports means ± SD and **bootstrap 95% CIs** (stored for
**every** `config × stratum × metric` in `eval_results/json/aggregate.json`).
Protocol §6 ("Statistical Testing Protocol") additionally requires **paired**
comparison tests over per-question scores. This document reports those tests for the
interpretable FoodRAG configuration comparisons (EXP-7), for both the partial-day
(deployed) and the full-day (all-four-slots) regimes.

## Data & ground truth (context for the pairing)

All pairs below are aligned on `question_id` over the **tight** evaluation stratum:
the 1,646 personas (of 10,000 generated) whose rule-based gold answer set has 1–50
recipes. Each persona is a set of hard preference constraints (tag/ingredient
likes-dislikes, per-100 g nutrient bands, per-serving daily caps, health-score
floors); the **ground truth** is the *exhaustive* set of HUMMUS corpus recipes
(310,228 recipes) satisfying all of them, computed deterministically by
`_get_valid_recipes_for_persona` (§1.1 of `docs/RESULTS_REPORT.md`). Because gold is
rule-derived and complete, the paired per-question scores are exact and the Wilcoxon
test is over genuine within-question differences, not annotation noise. The tight set
is constraint-heavy (mean 4.99 requirements/persona) which is why `n_pairs` for some
metrics is < 1,646 (metrics undefined on empty/over-filtered plans are dropped from
the pair).

## Method

* **Pairing (§6.2–6.3).** For a metric and a config, each question's score is the
  NaN-aware mean of that metric across the config's runs (FoodRAG: 3 runs;
  `llm_only`: 5; `naive_rag`: 1). Two configs are aligned on `question_id` (all
  cover the same 1,646 tight questions) and pairs where *either* side is NaN are
  dropped — so `n_pairs` is below 1,646 for metrics that are undefined on some
  questions (e.g. `path_coverage` / `guideline_adherence` on empty plans,
  `claim_grounding_rate` where no NL claims were emitted).
* **Test.** Paired **Wilcoxon signed-rank** over the per-question vectors
  (`scipy.stats.wilcoxon`). All-zero-difference → p = 1.0; < 6 non-zero pairs is
  handled gracefully. A paired **bootstrap** (10,000 resamples, seed 0) of the
  mean difference is reported alongside as the §6 alternative (mean-diff 95% CI).
* **Effect size (§6).** Median per-question difference (a − b).
* **Correction (§6.5).** **Holm–Bonferroni** within each metric family (all
  config pairs tested for that metric at this stratum). `significant` =
  corrected p < 0.05.
* **Stratum.** `aggregate` (all 1,646 tight questions).

Records: `eval_results/pairwise.csv` and `eval_results/json/pairwise.json`
(one §13.3 row per metric × config-pair). Regenerate with
`python -m foodrag.analysis.stats`.

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
| correctness_judgeA | 0.285 | 0.247 | <0.001 | <0.001 | 0.062 | [0.029, 0.047] | yes | 1646 |
| faithfulness_judgeA | 0.989 | 0.236 | <0.001 | <0.001 | 1.000 | [0.737, 0.769] | yes | 1646 |
| relevancy_judgeA | 0.454 | 0.311 | <0.001 | <0.001 | 0.000 | [0.125, 0.163] | yes | 1646 |
| guideline_adherence_overall | 0.299 | 0.280 | <0.001 | <0.001 | 0.000 | [0.013, 0.026] | yes | 1646 |
| hard_constraint_sat_perconstraint | 0.664 | 0.646 | <0.001 | <0.001 | 0.000 | [0.011, 0.024] | yes | 1646 |
| mrr_at_10 | 0.018 | 0.023 | 0.034 | 0.034 | 0.000 | [-0.010, -0.000] | yes | 1646 |
| hit_at_10 | 0.027 | 0.032 | 0.089 | 0.089 | 0.000 | [-0.010, 0.001] | **no** | 1646 |
| path_coverage | 0.652 | 0.656 | 0.268 | 0.268 | 0.000 | [-0.010, 0.003] | **no** | 1646 |

### foodrag_h1_t2p vs foodrag_h2_t2p (text-to-pandas retriever)
| metric | mean_a | mean_b | p | p_corr | median Δ | 95% CI | sig | n |
|---|---|---|---|---|---|---|---|---|
| correctness_judgeA | 0.765 | 0.586 | <0.001 | <0.001 | 0.167 | [0.163, 0.195] | yes | 1643 |
| faithfulness_judgeA | 0.997 | 0.892 | <0.001 | <0.001 | 0.000 | [0.093, 0.117] | yes | 1643 |
| hit_at_10 | 0.773 | 0.708 | <0.001 | <0.001 | 0.000 | [0.052, 0.077] | yes | 1643 |
| mrr_at_10 | 0.739 | 0.708 | <0.001 | <0.001 | 0.000 | [0.022, 0.039] | yes | 1643 |
| path_coverage | 0.883 | 0.867 | <0.001 | <0.001 | 0.000 | [0.013, 0.020] | yes | 1643 |
| relevancy_judgeA | 0.869 | 0.902 | <0.001 | <0.001 | 0.000 | [-0.047, -0.018] | yes (H2 wins) | 1643 |
| guideline_adherence_overall | 0.266 | 0.276 | 0.019 | 0.038 | 0.000 | [-0.016, -0.003] | yes (H2 wins) | 1515 |
| hard_constraint_sat_perconstraint | 0.977 | 0.976 | 0.711 | 0.711 | 0.000 | [-0.003, 0.003] | **no** | 1515 |

**Read.** H1 raises **faithfulness** and **correctness** over H2 under both
retrievers (e.g. dense: faithfulness 0.989 vs 0.236, median Δ = 1.0; t2p:
correctness 0.765 vs 0.586, Wilcoxon p < 0.001, median Δ = 0.167, 95% CI
[0.163, 0.195]) — the agentic layer's headline win. Retrieval quality (hit@10 /
mrr@10) is *not* improved by the agentic layer on the dense retriever (hit@10
n.s.), consistent with the agentic layer acting on explanation/answer quality,
not first-stage retrieval. On t2p, H1 slightly *loses* relevancy and guideline
adherence to H2 — a small but significant regression worth noting.

## Family 2 — Retriever sensitivity (text-to-pandas vs dense), per system

*How much does swapping the dense retriever for text-to-pandas (t2p) move each
system?*

### foodrag_h1_t2p vs foodrag_h1_dense (agentic system)
| metric | mean_a | mean_b | p | p_corr | median Δ | 95% CI | sig | n |
|---|---|---|---|---|---|---|---|---|
| hit_at_10 | 0.773 | 0.027 | <0.001 | <0.001 | 1.000 | [0.726, 0.765] | yes | 1643 |
| mrr_at_10 | 0.739 | 0.018 | <0.001 | <0.001 | 1.000 | [0.700, 0.740] | yes | 1643 |
| correctness_judgeA | 0.765 | 0.285 | <0.001 | <0.001 | 0.500 | [0.466, 0.494] | yes | 1643 |
| relevancy_judgeA | 0.869 | 0.455 | <0.001 | <0.001 | 0.333 | [0.393, 0.435] | yes | 1643 |
| hard_constraint_sat_perconstraint | 0.977 | 0.658 | <0.001 | <0.001 | 0.333 | [0.305, 0.333] | yes | 1516 |
| path_coverage | 0.883 | 0.652 | <0.001 | <0.001 | 0.267 | [0.215, 0.246] | yes | 1643 |
| claim_grounding_rate | 0.712 | 0.552 | <0.001 | <0.001 | 0.167 | [0.144, 0.177] | yes | 1499 |
| faithfulness_judgeA | 0.997 | 0.989 | <0.001 | <0.001 | 0.000 | [0.004, 0.012] | yes | 1643 |
| guideline_adherence_overall | 0.266 | 0.298 | <0.001 | <0.001 | -0.042 | [-0.040, -0.023] | yes (dense wins) | 1516 |

### foodrag_h2_t2p vs foodrag_h2 (recommend-only system)
| metric | mean_a | mean_b | p | p_corr | median Δ | 95% CI | sig | n |
|---|---|---|---|---|---|---|---|---|
| hit_at_10 | 0.709 | 0.032 | <0.001 | <0.001 | 1.000 | [0.655, 0.699] | yes | 1646 |
| mrr_at_10 | 0.709 | 0.023 | <0.001 | <0.001 | 1.000 | [0.664, 0.708] | yes | 1646 |
| correctness_judgeA | 0.586 | 0.247 | <0.001 | <0.001 | 0.333 | [0.322, 0.356] | yes | 1646 |
| relevancy_judgeA | 0.901 | 0.311 | <0.001 | <0.001 | 0.667 | [0.570, 0.612] | yes | 1646 |
| faithfulness_judgeA | 0.893 | 0.236 | <0.001 | <0.001 | 0.667 | [0.638, 0.676] | yes | 1646 |
| hard_constraint_sat_perconstraint | 0.977 | 0.638 | <0.001 | <0.001 | 0.333 | [0.325, 0.352] | yes | 1518 |
| path_coverage | 0.867 | 0.656 | <0.001 | <0.001 | 0.250 | [0.195, 0.227] | yes | 1646 |
| guideline_adherence_overall | 0.276 | 0.279 | 0.073 | 0.073 | 0.000 | [-0.012, 0.004] | **no** | 1518 |

**Read.** The retriever swap is the **dominant** effect in the whole study.
Text-to-pandas lifts hit@10 from ~0.03 to ~0.77 for both systems (median Δ = 1.0,
95% CI [0.73, 0.77]) — a categorical retrieval win that dense retrieval never
approaches on these tight, constraint-heavy questions. It also carries large
gains in correctness, relevancy, hard-constraint satisfaction and path coverage.
The one exception: guideline adherence is *not* improved (H1: slightly worse with
t2p; H2: n.s.).

## Family 3 — FoodRAG-best (H1-t2p) vs baselines (shared metrics)

*Shared metrics only — the baselines carry no judge/explainability metrics.*

### foodrag_h1_t2p vs naive_rag
| metric | mean_a | mean_b | p | p_corr | median Δ | 95% CI | sig | n |
|---|---|---|---|---|---|---|---|---|
| hit_at_10 | 0.773 | 0.055 | <0.001 | <0.001 | 1.000 | [0.696, 0.739] | yes | 1643 |
| mrr_at_10 | 0.739 | 0.023 | <0.001 | <0.001 | 1.000 | [0.695, 0.735] | yes | 1643 |
| allergen_violation | 0.000 | 0.008 | <0.001 | <0.001 | 0.000 | [-0.010, -0.006] | yes (FoodRAG safer) | 1643 |

### foodrag_h1_t2p vs llm_only
| metric | mean_a | mean_b | p | p_corr | median Δ | 95% CI | sig | n |
|---|---|---|---|---|---|---|---|---|
| hit_at_10 | 0.773 | 0.006 | <0.001 | <0.001 | 1.000 | [0.748, 0.786] | yes | 1643 |
| mrr_at_10 | 0.739 | 0.003 | <0.001 | <0.001 | 1.000 | [0.716, 0.755] | yes | 1643 |
| allergen_violation | 0.000 | 0.905 | <0.001 | <0.001 | -0.980 | [-0.912, -0.897] | yes (FoodRAG safer) | 1643 |

**Read.** FoodRAG-best dominates both baselines on retrieval (hit@10 0.773 vs
0.055 / 0.006, median Δ = 1.0). On **safety**, `llm_only` violates an allergen
constraint on ~90% of questions (allergen_violation 0.905) while FoodRAG is at
0.000 (median Δ = −0.98, 95% CI [−0.91, −0.90]); `naive_rag` is nearly safe
(0.008) but still significantly worse than FoodRAG's deterministic allergen
filter.

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
| h1_t2p vs h1_t2p_fullday | correctness_judgeA | 0.765 | 0.561 | 1.7e-111 | +0.167 | yes (partial higher) |
| h1_t2p vs h1_t2p_fullday | meal_slot_completeness_norm | 0.913 | 0.782 | 7.1e-48 | 0.000 | yes |
| h1_t2p vs h1_t2p_fullday | guideline_adherence_overall | 0.266 | 0.239 | 2.5e-13 | 0.000 | yes (partial higher) |
| h2 vs h2_fullday | guideline_adherence_overall | 0.279 | 0.209 | 2.3e-55 | +0.083 | yes (partial higher) |
| h2 vs h2_fullday | structural_validity_norm | 0.189 | 0.362 | 1.4e-59 | 0.000 | yes (full higher) |
| h2 vs h2_fullday | hit_at_10 | 0.032 | 0.046 | 2.1e-04 | 0.000 | yes (full higher) |

**Read.** The full-day regime confirms two separate stories with paired tests:
(i) **completeness/structural** metrics rise going partial→full (all p < 1e-14) —
the low partial-day readings were a **slot-count artifact**; (ii) **guideline
adherence** *falls* going partial→full (h2 0.279→0.209, p = 2.3e-55; h1_t2p
0.266→0.239, p = 2.5e-13) — a **genuine limitation** that a full day *worsens*, not
fixes. Within full-day, the retriever swap remains the dominant effect (hit@10 +0.74
to +0.78, p < 1.2e-251), but it **trades off** against slot coverage: dense fills
four distinct slots (completeness_norm 0.990, structural_norm 0.392) where t2p's
replay pool exhausts (0.782 / 0.022, p < 5.3e-75).

---

## Headline results

* **H1-t2p vs H2-t2p — correctness:** H1 beats H2 (0.765 vs 0.586; Wilcoxon
  p < 0.001, corrected p < 0.001, median Δ = 0.167, 95% CI [0.163, 0.195]). The
  agentic layer's clearest quality win.
* **H1-t2p vs H1-dense — hit@10:** t2p beats dense (0.773 vs 0.027; p < 0.001,
  median Δ = 1.000, 95% CI [0.726, 0.765]). The retriever swap, not the agentic
  layer, drives retrieval.
* **Faithfulness, agentic vs recommend-only (dense):** H1 beats H2 (0.989 vs
  0.236; p < 0.001, median Δ = 1.000, 95% CI [0.737, 0.769]).
* **Safety vs `llm_only`:** FoodRAG-best is far safer (allergen_violation 0.000
  vs 0.905; p < 0.001, median Δ = −0.980, 95% CI [−0.912, −0.897]).

### Non-significant / reversed comparisons (called out)

* **hit@10, H1-dense vs H2 (dense):** *not* significant (0.027 vs 0.032,
  corrected p = 0.089) — the agentic layer does not change dense retrieval.
* **path_coverage, H1-dense vs H2:** *not* significant (corrected p = 0.268).
* **hard_constraint_sat, H1-t2p vs H2-t2p:** *not* significant (0.977 vs 0.976,
  corrected p = 0.711) — both t2p systems already near-saturate this.
* **guideline_adherence, H2-t2p vs H2:** *not* significant (corrected p = 0.073);
  and H1-t2p is *worse* than H2-t2p on guideline adherence (0.266 vs 0.276,
  corrected p = 0.038) and relevancy (0.869 vs 0.902) — small but significant
  regressions from the agentic layer under t2p.

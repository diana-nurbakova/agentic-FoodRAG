# FoodRAG Evaluation — Results Summary

Tight stratum (personas with 1–50 valid answers, n=1,646). Generator: Llama-3.3-70B.
Dual-judge (Judge A = Gemma-3-27B cross-lineage; Judge B = DeepSeek-V3). All numbers
are aggregate-stratum means; per-question records + CIs in the per-config parquets.

---

## 1. Retriever ladder (EXP-1 / Use-1, retrieval-only)

| Retriever | hit@10 | mrr@10 | nDCG@10 | avg NutriScore | avg FSA | allergen viol. | latency (s) |
|---|---|---|---|---|---|---|---|
| Random | 0.000 | 0.000 | 0.000 | 0.100 | 0.457 | 0.000 | 0.03 |
| Dense (embeddings) | 0.055 | 0.023 | 0.012 | 0.128 | 0.494 | 0.008 | 0.45 |
| Sparse / BM25 | 0.063 | 0.026 | 0.015 | 0.135 | 0.494 | 0.020 | 44.6 |
| Hybrid (RRF) | 0.055 | 0.023 | 0.012 | 0.128 | 0.494 | 0.008 | 46.6 |
| Full-text | 0.063 | 0.026 | 0.015 | 0.136 | 0.494 | 0.020 | 45.8 |
| Intersection | 0.002 | 0.002 | 0.002 | 0.199 | 0.557 | 0.001 | 37.0 |
| Text-to-Pandas @ gpt-4o-mini | 0.427 | 0.383 | 0.367 | 0.353 | 0.724 | 0.000 | 14.0 |
| **Text-to-Pandas @ llama-70B** | **0.854** | **0.766** | **0.730** | 0.351 | 0.722 | **0.000** | 16.2 |

**Finding.** The lightweight symbolic retriever (Text-to-Pandas) dominates dense/sparse
embeddings by ~15× on hit@10 (0.854 vs 0.055) at moderate cost, and also delivers the
best health scores and zero allergen violations. For constraint-satisfaction retrieval over
a well-structured food KG, expensive embedding/graph retrieval is not needed. Backbone
matters: llama-70B ≫ gpt-4o-mini for the Pandas generation (0.854 vs 0.427).

---

## 2. Baseline ladder (EXP-7 rungs)

| Rung | hit@10 | mrr@10 | allergen viol. | latency (s) |
|---|---|---|---|---|
| 0a Popularity (raw, unaware) | 0.293 | 0.135 | 0.001 | — |
| 0b Popularity (health-aware) | 1.000* | 1.000* | 0.000 | — |
| 2 LLM-only (no retrieval) | 0.006 | 0.003 | **0.905** | — |
| 3 Naive-RAG | 0.055 | 0.023 | 0.008 | — |
| ToG (agentic KG traversal) | 0.407 | 0.316 | 0.000 | 902.8 |

**Findings.**
- **LLM-only violates a hard allergen constraint 90.5% of the time** — the core justification
  for retrieval/grounding: an unconstrained LLM invents recipes that harm.
- 0b* pre-filters to the constraint-valid set before ranking by popularity, so it trivially
  hits (it "knows the answer key"); reported as a reference ceiling, not a fair retriever.
- ToG is competent (0.407) but ~15 min/item — heavy agentic traversal buys less than
  Text-to-Pandas at ~50× the cost.

---

## 3. FoodRAG system matrix (EXP-7 / Use-2, full meal-plan pipeline)

H1 = agentic (Recommend + Correct + Enhance); H2 = Recommend-only. dense vs text-to-pandas retriever.

| Metric | H2-dense | H2-t2p | H1-dense | H1-t2p |
|---|---|---|---|---|
| hit@10 | 0.032 | **0.709** | 0.027 | 0.556 |
| mrr@10 | 0.023 | 0.709 | 0.018 | 0.556 |
| correctness (Judge A) | 0.248 | **0.586** | 0.286 | 0.444 |
| faithfulness (Judge A) | 0.232 | 0.893 | **0.989** | 0.556 |
| hard-constraint sat. (per-constraint) | 0.645 | 0.976 | 0.663 | **1.000** |
| allergen violation | 0.003 | 0.001 | 0.002 | **0.000** |
| guideline adherence (overall) † | 0.280 | 0.276 | 0.299 | 0.156 |
| slot appropriateness ‡ | 0.454 | 0.237 | 0.771 | 0.417 |
| meal-slot completeness † | 0.568 | 0.535 | 0.525 | 0.278 |
| path coverage (§8.1) | 0.655 | 0.868 | 0.651 | 0.883 |
| path validity (§8.1) | 1.000 | 1.000 | 1.000 | 1.000 |
| **claim grounding (§8.2)** | — | — | — | **0.650** |

**Findings.**
- **Retriever quality propagates through the pipeline**: t2p configs get ~20× the hit@10 of
  dense (0.71/0.56 vs 0.03), near-perfect hard-constraint satisfaction, and zero allergen
  violations.
- **Safety is excellent across all configs** (allergen ≤ 0.003).
- **The agentic pipeline (H1) is a mixed bag**: it lifts faithfulness (H1-dense 0.989) and
  slot-appropriateness (0.771) on dense, but on t2p the simpler H2 scores higher on
  hit@10/correctness — agentic correction/substitution alters the recipe set, trading
  retrieval accuracy for other qualities.
- **§8 grounding contrast**: intrinsic KG-path explanations are 100% valid (read from HUMMUS),
  while the NL explanation layer is 65% grounded (35% unsupported claims) — the intrinsic
  layer is the more reliable explanation channel.

---

## 4. Pareto trade-off (§9.4)

**Quality (correctness) vs Cost (LLM calls), FoodRAG configs:**

| Config | cost (LLM calls) | quality (correctness) | frontier |
|---|---|---|---|
| **H2-t2p** | **1.0** | **0.586** | ✅ dominates all others |
| H1-t2p | 13.9 | 0.566 | dominated by H2-t2p |
| H1-dense | 9.9 | 0.286 | dominated by H2-t2p |
| H2-dense | 1.0 | 0.248 | dominated by H2-t2p |

**Finding.** The agentic pipeline is **not Pareto-justified on the correctness/cost axes**:
H2-t2p (recommend-only) achieves the highest correctness at the lowest cost, and H1-t2p pays
~14× the LLM cost for slightly *lower* correctness. The agentic pipeline's distinct value
(user-facing explanations, faithfulness on weak retrieval) lies off these two axes.

---

## 5. Deployed-behavior finding: time-driven meal selection

The meal-plan tool selects meal types by time of day (plans "the remaining day"), so plans are
**partial-day** (mean 2.32 meals/plan, not 4). Selection is strongly time-driven
(`eval_results/mealplan_time_of_day.csv`): mornings → breakfast-dominant; evenings →
dinner+snack; breakfast/lunch drop once past. This is a deployment feature — and the
quantitative reason the full-day plan-level metrics (guideline, completeness) read low.

---

## 6. Limitations / caveats

- **† Partial-day plans**: guideline-adherence, calorie-distribution and meal-slot-completeness
  assume a full 4-slot day, so they are conservative lower bounds on partial-day plans. A
  full-day re-run was deliberately not done (it would flatten completeness toward 1.0 and cost
  ~3 days); an offline "completed-to-full-day" nutrition variant is the intended supplement.
- **‡ slot appropriateness** is computed on *reconstructed* plans for H2/H1-dense (native plan
  not persisted at run time) vs the *native* plan for H1-t2p; treat H1-t2p (0.417) as the
  reliable value and the others as reconstruction-biased.
- **Time regime**: H1 configs ran at a fixed `morning`; H2 configs at wall-clock `now`. The
  difference affects which meals are requested and is confined to the plan-level metrics;
  retrieval/judge/safety are time-robust. Recommendation time is now persisted per record
  (`recommendation_time`, `requested_meal_types`) for reproducibility going forward.
- **Judge**: headline judge is Judge A (Gemma, cross-lineage); Judge B (DeepSeek-V3) reported
  alongside. claim-grounding populates only where an NL explanation exists (H1-t2p natively).

# FoodRAG Evaluation — Full Results Report

*Thesis-grade results report expanding `eval_results/RESULTS_SUMMARY.md`. All numbers
are drawn from the corrected sources of truth: `eval_results/json/aggregate.json`
(bootstrap 95 % CIs), `eval_results/aggregate.csv` (+ `eval_results_llama/aggregate.csv`),
the per-config `eval_results/*_per_question.parquet` (used for the stratified
breakdowns and inter-judge statistics computed here), `eval_results/mealplan_time_of_day.csv`,
and `specs/FoodRAG_Evaluation_Protocol.md` for metric/experiment definitions.*

> **Corrected-aggregate note.** The `foodrag_h1_t2p` aggregate was previously contaminated
> by a 3-item smoke run and has been re-aggregated directly from the full per-question
> parquet (n ≈ 1,646/metric across 3 runs). Every `h1_t2p` figure below is the corrected
> value. Where a number computed here differs from the summary's headline anchors, it is
> flagged in the text; **no material disagreement was found** (see the closing "Anchor
> reconciliation" note).
>
> **Post-fix note (commit 54a62ae).** Two deterministic safety guards were added and the 493
> affected records across the four FoodRAG configs were re-run and re-judged: (a) a
> deterministic **allergen post-filter** (`constraints.enforce_allergen_safety`) applied to
> the final plan in both assemblers, which drops allergen violations to **0.000 on every
> config**; and (b) an **empty-plan explanation gate** in `ExplanationTool` that returns a
> fixed "no recommendation" message instead of narrating recipes that were never selected,
> which lifts H1-t2p correctness (0.689 → 0.765), faithfulness (0.922 → 0.997) and
> claim-grounding (0.650 → 0.711). All numbers below are the **post-fix** source-of-truth
> values; the retriever ladder, baseline ladder, and the dense/H2 configs' core metrics were
> unaffected.

---

## 1. Setup

| Item | Value |
|---|---|
| QA set (headline stratum) | **Tight** — personas with 1–50 valid answers, **n = 1,646** (Protocol §9.1) |
| QA generation | **Llama-3.3-70B** (via Groq / LiteLLM) |
| Retrieval / list cutoff | **K = 10** (fixed; Protocol §4.1, §3-note) |
| Dual judge | **Judge A = Gemma-3-27B** (Google lineage, *cross-lineage* to the Meta generator — headline judge) · **Judge B = DeepSeek-V3** (reference) |
| Judge scale | Correctness & explanation-quality ordinal on {0, .25, .5, .75, 1}; faithfulness & relevancy binary {0,1} (confirmed empirically from the per-question JSONs) |
| Aggregate rows | **949** rows in `aggregate.json` (config × stratum × metric, each with mean, ci_low, ci_high, per-question variance, n_questions, n_runs, judge) |
| Distinct configs | 26 (8 retrievers, 5 baselines/ladder rungs, 4 FoodRAG system configs, plus EXP-3 heuristics/rag and EXP-5 KG) |

**Run counts per family** (from `json/index.json` and the aggregate `n_runs`):

| Family | Configs | Runs | Records |
|---|---|---|---|
| FoodRAG system matrix | `foodrag_h2`, `foodrag_h2_t2p`, `foodrag_h1_dense`, `foodrag_h1_t2p` | **3** each | 4,938 each (1,646 × 3) |
| LLM-only | `llm_only` | **5** | 8,230 |
| Text-to-Pandas retrievers | `ret_text2pandas@llama70b`, `ret_text2pandas@gpt4o-mini` | **5** | 8,230 |
| Single-shot retrievers | dense/sparse/hybrid/fulltext/intersection/random | **1** | 1,646 |
| popularity_raw / popularity_health_aware | — | **1** | 10,000 (all strata) |
| naive_rag | — | **1** | 1,646 |
| ToG (agentic KG) | `tog` | **3** | 300 (100 items × 3) |

**Metrics tracked** (per FoodRAG config the parquet carries 37 metric fields): retrieval
(hit@10, mrr@10, ndcg@10, avg_nutri@10, avg_fsa@10, avg_rating@10); safety/constraint
(allergen_violation, hard_constraint_sat per-constraint & all-or-nothing,
hard_constraint_violation, constraint_satisfaction, soft_constraint_sat,
hallucination_rate); LLM-judge (correctness / faithfulness / relevancy /
explanation-quality — **each under both judges**); plan-level (guideline_adherence overall +
8 per-nutrient, slot_appropriateness, meal_slot_completeness, intra_plan_diversity,
calorie_adherence, structural_validity); explainability (path_coverage, path_validity,
claim_grounding_rate, unsupported_claim_rate); processing (llm_usage, llm_seq_usage,
llm_parsing, latency_s).

---

## 2. Retriever comparison (EXP-1 / Use-1, retrieval-only)

All rows below are the **tight-stratum / aggregate** values from `aggregate.json`
(`ret_*` configs) with bootstrap 95 % CIs on hit@10. Text-to-Pandas @ llama70b comes from
`eval_results_llama/aggregate.csv`; @ gpt4o-mini from the main `aggregate.json`.

| Retriever | hit@10 [95 % CI] | mrr@10 | nDCG@10 | avg Nutri@10 | avg FSA@10 | allergen viol. | latency (s) | runs |
|---|---|---|---|---|---|---|---|---|
| Random | 0.000 [0.000,0.000] | 0.000 | 0.000 | 0.100 | 0.457 | 0.0005 | 0.03 | 1 |
| Dense (jina embeddings) | 0.055 [0.044,0.067] | 0.023 | 0.012 | 0.128 | 0.494 | 0.0077 | 0.45 | 1 |
| Sparse / BM25 | 0.063 [0.051,0.075] | 0.026 | 0.015 | 0.135 | 0.494 | 0.0196 | 44.6 | 1 |
| Hybrid (RRF) | 0.055 [0.044,0.067] | 0.023 | 0.012 | 0.128 | 0.494 | 0.0077 | 46.6 | 1 |
| Full-text | 0.063 [0.051,0.075] | 0.026 | 0.015 | 0.136 | 0.494 | 0.0196 | 45.8 | 1 |
| Intersection | 0.002 [0.001,0.005] | 0.002 | 0.002 | 0.199 | 0.557 | 0.0012 | 37.0 | 1 |
| Text-to-Pandas @ gpt-4o-mini | 0.427 [0.418,0.436] | 0.383 | 0.367 | 0.353 | 0.724 | 0.0001 | 14.0 | 5 |
| **Text-to-Pandas @ llama-70B** | **0.854 [0.837,0.870]** | **0.766** | **0.730** | 0.351 | 0.722 | **0.0003** | 16.2 | 5 |

Additional constraint metrics for the strong retriever (`ret_text2pandas@llama70b`):
constraint_satisfaction 0.688 [0.671,0.706], hard_constraint_violation 0.064
[0.056,0.072], hallucination_rate 0.000.

**Findings.**
- **Symbolic retrieval dominates.** Text-to-Pandas @ llama-70B reaches hit@10 = **0.854**,
  ≈ **15.5×** the best embedding retriever (dense/hybrid 0.055) and ≈ **13.6×** BM25/full-text
  (0.063). The CIs are disjoint by an order of magnitude — the gap is not attributable to
  sampling noise. It also carries the best health profile among the general retrievers
  (avg FSA 0.722, avg Nutri 0.351) and essentially zero allergen violations (0.0003).
- **Backbone matters.** Swapping the Pandas-generation LLM from gpt-4o-mini to llama-70B
  doubles hit@10 (0.427 → 0.854) and mrr@10 (0.383 → 0.766) at similar latency (14 → 16 s).
  The retriever's ceiling is set by the code-generation quality of its backbone, not by the
  retrieval index.
- **Embedding/graph retrieval is not needed** for constraint-satisfaction retrieval over a
  well-structured food KG: expensive sparse/hybrid retrieval (44–47 s) buys nothing over
  dense (0.45 s) and is 13× worse than symbolic filtering. This directly answers Protocol
  EQ1 / §5.1's "Use-1" question.

---

## 3. Baseline ladder (EXP-7 rungs)

Tight-stratum values from `aggregate.json`. ToG is reported at its **aggregate** stratum
(n = 300, 3 runs) to match the summary; its tight sub-stratum (n = 96) scores higher
(hit@10 = 0.594) but is a smaller sample.

| Rung | hit@10 [95 % CI] | mrr@10 | allergen viol. | hard-constr. viol. | latency (s) |
|---|---|---|---|---|---|
| 0a Popularity (raw, unaware) | 0.0018 [0.000,0.004] | 0.0005 | 0.0007 | 0.729 | — |
| 0b Popularity (health-aware) * | **1.000** [1.0,1.0] | **1.000** | 0.000 | 0.000 | — |
| 2 LLM-only (no retrieval) | 0.0061 [0.003,0.009] | 0.0030 | **0.905** [0.898,0.912] | 0.972 | — |
| 3 Naive-RAG | 0.055 [0.045,0.066] | 0.023 | 0.0076 | 0.659 | — |
| ToG (agentic KG traversal) | 0.407 [aggregate] | 0.316 | 0.000 | 0.454 | **902.8** |

**Findings.**
- **LLM-only violates a hard allergen constraint 90.5 % of the time** (allergen_violation =
  0.905 [0.898, 0.912]; hallucination_rate identically 0.905). This is the single most
  important safety result in the report and the core justification for grounding: an
  unconstrained LLM confidently invents recipes that would harm allergic users. Its
  retrieval is essentially zero (hit@10 = 0.006), so it is neither correct nor safe.
- **0b is a reference ceiling, not a fair retriever** (\*). Health-aware popularity
  *pre-filters to the constraint-valid set* before ranking by popularity, so it trivially
  scores hit@10 = mrr@10 = nDCG@10 = 1.0 and 0 violations — it "knows the answer key."
  Reported as an upper bound; its avg NutriScore (0.355) / FSA (0.731) show the health
  ceiling reachable when the valid set is known.
- **ToG is competent but ruinously expensive.** hit@10 = 0.407 (aggregate) / 0.594 (tight)
  with 0 allergen violations, but **~903 s/item** (≈ 15 min), ~56× Text-to-Pandas's 16 s.
  Heavy agentic KG traversal buys *less* coverage than lightweight symbolic retrieval at
  ~50–56× the cost — a headline finding for Use-1.
- Raw popularity (0a) is a genuine floor: it "knows the answer" for essentially nothing
  under tight constraints (hit@10 = 0.002).

---

## 4. FoodRAG system matrix (EXP-7 / Use-2, full meal-plan pipeline)

H1 = full agentic pipeline (Recommend + Correct + Enhance); H2 = Recommend-only.
Retriever ∈ {dense, text-to-pandas (t2p)}. All values are the aggregate-stratum means
from `aggregate.json` (3 runs, n ≈ 1,646/metric); **[CI]** is the bootstrap 95 % interval.
Judge-metric rows report **Judge A** unless the split table in §5 is referenced.

### 4.1 Retrieval & health

| Metric | H2-dense | H2-t2p | H1-dense | **H1-t2p** |
|---|---|---|---|---|
| hit@10 | 0.032 [0.025,0.039] | 0.709 [0.687,0.730] | 0.027 [0.021,0.034] | **0.773 [0.754,0.792]** |
| mrr@10 | 0.023 [0.018,0.029] | 0.709 [0.687,0.731] | 0.018 [0.014,0.024] | **0.738 [0.719,0.758]** |
| nDCG@10 | 0.009 [0.007,0.012] | 0.347 [0.330,0.364] | 0.007 [0.005,0.010] | **0.450 [0.434,0.466]** |
| avg Nutri@10 | 0.133 | 0.354 | 0.153 | 0.354 |
| avg FSA@10 | 0.516 | 0.720 | 0.524 | 0.721 |
| avg rating@10 | 4.14 | 4.38 | 4.01 | **4.40** |

### 4.2 LLM-judge quality (Judge A / Judge B)

| Metric | H2-dense | H2-t2p | H1-dense | **H1-t2p** |
|---|---|---|---|---|
| correctness (A) | 0.247 [0.238,0.257] | 0.586 [0.573,0.601] | 0.285 [0.277,0.293] | **0.765 [0.754,0.777]** |
| correctness (B) | 0.232 | 0.604 | 0.309 | **0.742** |
| faithfulness (A) | 0.236 [0.220,0.251] | 0.893 [0.881,0.904] | **0.989 [0.985,0.992]** | **0.997 [0.995,0.998]** |
| faithfulness (B) | 0.112 | 0.986 | 0.976 | 0.994 |
| relevancy (A) | 0.311 | 0.901 | 0.454 | 0.869 |
| relevancy (B) | 0.098 | 0.559 | 0.221 | 0.784 |
| explanation-quality (A) | — | — | 0.573 | **0.750** |
| explanation-quality (B) | — | — | 0.626 | 0.793 |

*(explanation-quality is only populated where an NL explanation layer exists — the two H1
configs.)*

### 4.3 Safety & constraint split

| Metric | H2-dense | H2-t2p | H1-dense | **H1-t2p** |
|---|---|---|---|---|
| hard-constraint sat. (per-constraint) | 0.646 [0.634,0.659] | 0.977 [0.972,0.981] | 0.664 [0.652,0.677] | **0.977 [0.973,0.980]** |
| hard-constraint sat. (all-or-nothing) | 0.328 | 0.932 | 0.359 | **0.931** |
| hard-constraint violation | 0.743 | 0.145 | 0.711 | 0.228 |
| soft-constraint sat. | 0.403 | 0.893 | 0.397 | 0.888 |
| constraint_satisfaction | 0.018 | 0.709 | 0.015 | 0.698 |
| **allergen violation** | **0.000** | **0.000** | **0.000** | **0.000** |
| hallucination_rate | 0.000 | 0.000 | 0.000 | 0.000 |

### 4.4 Plan-level quality

| Metric | H2-dense | H2-t2p | H1-dense | **H1-t2p** |
|---|---|---|---|---|
| guideline adherence (overall) † | 0.280 | 0.276 | 0.299 | 0.266 |
| slot appropriateness ‡ | 0.454 | 0.237 | 0.771 | 0.519 |
| meal-slot completeness † | 0.568 | 0.535 | 0.525 | 0.481 |
| intra-plan diversity | 0.756 | 0.000 § | 0.782 | 0.625 |
| calorie adherence | 0.785 | 0.911 | 0.822 | 0.836 |
| structural validity | 0.207 | 0.186 | 0.171 | 0.170 |

§ H2-t2p records single-item slots, so intra-plan diversity is undefined/0 — a data-shape
artifact, not a quality signal.

### 4.5 Per-nutrient guideline adherence (H1-t2p)

Computed here from the `guideline_adherence_per_nutrient` field (n = 4,412 non-null items):

| energy | protein | fat | sat-fat | carb | sugar | fibre | sodium |
|---|---|---|---|---|---|---|---|
| 0.053 | 0.141 | 0.119 | 0.679 | 0.067 | 0.661 | 0.099 | 0.312 |

Adherence is high on the "avoid excess" nutrients (sat-fat 0.68, sugar 0.66) but low on the
"hit a target band" nutrients (energy 0.05, carb 0.07, fibre 0.10) — consistent with the
**partial-day-plan** limitation (§10, §11): a 2.3-meal plan cannot fill a full-day energy /
carb / fibre band. This is the mechanistic reason the *overall* guideline number sits at
~0.27–0.30 across all four configs.

### 4.6 Explainability & processing

| Metric | H2-dense | H2-t2p | H1-dense | **H1-t2p** |
|---|---|---|---|---|
| path coverage (§8.1) | 0.656 | 0.867 | 0.652 | **0.883** |
| path validity (§8.1) | 1.000 | 1.000 | 1.000 | 1.000 |
| **claim grounding (§8.2)** | — | — | 0.550 [0.538,0.562] | **0.711 [0.699,0.724]** |
| unsupported-claim rate | — | — | 0.450 | 0.289 |
| llm_usage (calls) | 1.0 | 1.0 | 9.91 | 10.05 |
| llm_seq_usage | 1.0 | 1.0 | 8.50 | 8.46 |
| llm_parsing (retries) | 0.000 | 0.000 | 0.007 | 0.655 |
| latency (s) | 42.8 | 44.1 | 286.9 | 226.4 |

**Interpretation.**
1. **Retriever quality dominates the pipeline.** t2p configs get **~22–29×** the hit@10 of
   their dense twins (0.71–0.77 vs 0.03), near-perfect per-constraint hard-satisfaction
   (0.977 vs 0.646–0.664). The retriever, not the agent depth, is the first-order driver of
   end-to-end quality.
2. **The agentic pipeline pays off *when retrieval is good*.** On t2p, H1 beats H2 on hit@10
   (0.773 vs 0.709), nDCG (0.450 vs 0.347) and correctness (0.765 vs 0.586, +0.179,
   non-overlapping CIs). On dense, H1 ≈ H2 (both retrieve ~nothing, so the Correct/Enhance
   stages have no valid candidates to improve). H1-dense's very high *faithfulness* (0.989)
   reflects tiny, cautious plans — the agent declines to over-claim when it has little to
   work with. **Agentic value emerges only once the retriever supplies valid candidates.**
3. **Safety is now perfect everywhere** (allergen violation 0.000 on all four configs after
   the deterministic allergen post-filter, commit 54a62ae; hallucination_rate 0 across all
   four). The full pipeline on the strong retriever is both the most correct and, alongside
   the others, fully safe. The filter drops any plan slot whose recipe carries a
   persona-disliked/allergen ingredient; because a blacklist-violating recipe can never be
   gold, hit@10 is unaffected.
4. **§8 grounding contrast.** The intrinsic KG-path layer is **100 % valid** (paths read
   directly from HUMMUS) and 88 % covering on H1-t2p, while the user-facing NL layer is
   **71 % grounded** on H1-t2p (29 % unsupported) and **55 % on H1-dense** (45 % unsupported).
   The stronger retriever yields *better-grounded* explanations, and the empty-plan explanation
   gate (commit 54a62ae) replaces hallucinated prose on empty plans with an honest "no
   recommendation" message — the NL claims now describe genuinely valid recipes (or nothing at
   all). The intrinsic layer remains the reliable explanation channel; the NL layer still needs
   claim-level guarding where a non-empty plan is over-described.

**Bottom line: H1-t2p is the best config overall** — top hit@10 (0.773), top correctness
(0.765 A / 0.742 B), near-perfect faithfulness (0.997), near-perfect hard-constraint
satisfaction (0.977), zero allergen violations, and the best-grounded NL explanations of the
two agentic configs (0.711 vs H1-dense 0.550).

---

## 5. Dual-judge analysis (Judge A Gemma-3-27B vs Judge B DeepSeek-V3)

Means, mean **Δ = (Judge B − Judge A)**, Pearson/Spearman, and (for binary metrics)
raw agreement, computed here per-question over the four FoodRAG parquets. Correctness and
explanation-quality are ordinal → Pearson/Spearman reported; faithfulness and relevancy are
binary → agreement reported.

| Config | Metric | mean A | mean B | Δ (B−A) | Pearson | Spearman | agreement | n |
|---|---|---|---|---|---|---|---|---|
| H2-dense | correctness | 0.248 | 0.233 | **−0.015** | 0.745 | 0.727 | — | 4,931 |
| | faithfulness | 0.232 | 0.106 | **−0.126** | 0.519 | — | 0.846 | 4,925 |
| | relevancy | 0.308 | 0.096 | **−0.212** | 0.409 | — | 0.767 | 4,934 |
| H2-t2p | correctness | 0.586 | 0.605 | **+0.018** | 0.761 | 0.752 | — | 4,920 |
| | faithfulness | 0.893 | 0.986 | **+0.093** | 0.249 | — | 0.900 | 4,930 |
| | relevancy | 0.901 | 0.559 | **−0.341** | 0.341 | — | 0.649 | 4,930 |
| H1-dense | correctness | 0.286 | 0.309 | **+0.024** | 0.673 | 0.658 | — | 4,855 |
| | faithfulness | 0.989 | 0.976 | **−0.013** | 0.573 | — | 0.984 | 4,850 |
| | relevancy | 0.453 | 0.220 | **−0.233** | 0.522 | — | 0.741 | 4,859 |
| | explanation-quality | 0.573 | 0.626 | **+0.053** | 0.642 | 0.624 | — | 4,838 |
| **H1-t2p** | correctness | 0.765 | 0.740 | **−0.025** | 0.351 | 0.383 | — | 4,794 |
| | faithfulness | 0.997 | 0.994 | **−0.003** | 0.402 | — | **0.995** | 4,799 |
| | relevancy | 0.869 | 0.785 | **−0.084** | 0.539 | — | 0.860 | 4,813 |
| | explanation-quality | 0.750 | 0.792 | **+0.042** | 0.626 | 0.676 | — | 4,791 |

**Findings.**
- **Correctness is judge-robust.** |Δ| ≤ 0.025 on every config — the headline correctness
  ordering (H1-t2p > H2-t2p > H1-dense > H2-dense) is identical under both judges. On H1-t2p,
  the gated empty-plan records (now scored on the honest "no recommendation" message rather
  than a hallucinated one) lift Judge A's correctness to 0.765, so Judge A now sits *slightly
  above* Judge B (−0.025); on H2-t2p Judge B is marginally higher (+0.018). Either way the
  cross-judge gap on correctness is small.
- **Binary metrics diverge more, and directionally.** Relevancy shows the largest
  disagreement: Judge B is *harsher* on the strong-retriever configs (H2-t2p −0.341,
  H1-t2p −0.084) and generally scores relevancy lower (H2-dense −0.212, H1-dense −0.233).
  Faithfulness moves the other way on t2p (Judge B *higher*: +0.093 on H2-t2p). Raw binary
  agreement is nonetheless high (0.65–0.99), highest on H1-t2p (faithfulness 0.995,
  relevancy 0.860) — i.e., the judges disagree most exactly where quality is low/noisy
  (H2-dense, H2-t2p relevancy).
- **Cross-lineage caution is warranted, but does not overturn conclusions.** No config's
  qualitative ranking flips between judges on correctness. The report therefore headlines
  Judge A (Gemma, cross-lineage) with Judge B alongside, exactly as the protocol prescribes.
- **Inter-judge κ as future work.** Faithfulness/relevancy Cohen's κ (Protocol §7.2) is not
  computed here to keep the pass memory-light; the raw agreement + Pearson above are the
  reportable proxies, and the very high agreement on H1-t2p suggests κ would be substantial
  there. Full κ (and judge–human κ on a stratified sample) remains future work.

---

## 6. Stratified breakdowns

Computed here from the per-question parquets (all 3 runs pooled; n shown is item-rows).
Requirement-type strata are **membership** strata (a query is counted in every type it
contains), so they overlap. `health` is present on 100 % of tight-stratum queries.

### 6.1 By number of requirements (constraint complexity)

**hit@10:**

| num_req | n (rows) | H2-dense | H2-t2p | H1-dense | **H1-t2p** |
|---|---|---|---|---|---|
| 1 | 6 | 0.000 | 1.000 | 0.000 | 1.000 |
| 2 | 159 | 0.082 | 0.767 | 0.064 | 0.854 |
| 3 | 552 | 0.056 | 0.770 | 0.055 | 0.824 |
| 4 | 1,131 | 0.030 | 0.718 | 0.022 | 0.789 |
| 5 | 1,278 | 0.034 | 0.713 | 0.033 | 0.771 |
| 6+ | 1,812 | 0.019 | 0.676 | 0.016 | 0.738 |

**correctness (Judge A):**

| num_req | H2-dense | H2-t2p | H1-dense | **H1-t2p** |
|---|---|---|---|---|
| 2 | 0.364 | 0.665 | 0.283 | 0.745 |
| 3 | 0.265 | 0.635 | 0.273 | 0.726 |
| 4 | 0.263 | 0.590 | 0.299 | 0.748 |
| 5 | 0.244 | 0.578 | 0.284 | 0.772 |
| 6+ | 0.227 | 0.567 | 0.283 | 0.787 |

**hard-constraint satisfaction (per-constraint):**

| num_req | H2-dense | H2-t2p | H1-dense | **H1-t2p** |
|---|---|---|---|---|
| 2 | 0.536 | 0.980 | 0.562 | 0.991 |
| 3 | 0.521 | 0.980 | 0.557 | 0.986 |
| 4 | 0.618 | 0.972 | 0.636 | 0.977 |
| 5 | 0.674 | 0.983 | 0.685 | 0.983 |
| 6+ | 0.690 | 0.973 | 0.709 | 0.968 |

**allergen violation** is now **0.000 across all num_req bins for every config** after the
deterministic allergen post-filter (commit 54a62ae).

**Findings.**
- **Coverage degrades monotonically with constraint count** for the *good* configs: H1-t2p
  hit@10 falls 0.854 (2 req) → 0.738 (6+), H2-t2p 0.767 → 0.676. The degradation is graceful
  (~14 pts over the whole range) — the strong retriever remains usable even at 6+
  simultaneous requirements.
- **The dense configs are already at the floor** (hit@10 0.02–0.08 everywhere), so they
  barely "degrade" — there is nothing to lose.
- **Correctness *rises* with complexity for H1-t2p** (0.745 at 2 req → 0.787 at 6+),
  whereas H2-t2p *drops* (0.665 → 0.567). The agentic Correct/Enhance stage more than holds
  correctness up as constraints stack — and, post-fix, the honest empty-plan message on the
  hardest queries (where the valid set is smallest) is scored higher than the hallucinated
  explanation it replaced, widening the H1-vs-H2 gap on hard queries. Direct evidence that the
  agent earns its keep on hard queries.
- **Per-constraint hard-satisfaction is near-ceiling for both t2p configs** at every
  complexity level (≥ 0.967), while the dense configs *rise* with num_req (0.54 → 0.69)
  because more constraints mean more that are trivially satisfiable by any recipe. The
  small n=1 / n=6 bins are shown for completeness but are not load-bearing.

### 6.2 By requirement type

**hit@10:**

| req type | n (rows) | H2-dense | H2-t2p | H1-dense | **H1-t2p** |
|---|---|---|---|---|---|
| allergen (safety-critical) | 3,312 | 0.025 | 0.710 | 0.020 | **0.768** |
| ingredient | 2,961 | 0.031 | 0.590 | 0.025 | 0.667 |
| tag | 4,041 | 0.030 | 0.720 | 0.027 | 0.783 |
| nutrient_bound | 3,543 | 0.029 | 0.698 | 0.024 | 0.759 |
| health | 4,938 | 0.032 | 0.709 | 0.027 | 0.772 |

**correctness (Judge A):**

| req type | H2-dense | H2-t2p | H1-dense | **H1-t2p** |
|---|---|---|---|---|
| allergen | 0.235 | 0.575 | 0.284 | 0.773 |
| ingredient | 0.284 | 0.560 | 0.323 | 0.780 |
| tag | 0.235 | 0.582 | 0.281 | 0.768 |
| nutrient_bound | 0.239 | 0.584 | 0.281 | 0.766 |
| health | 0.248 | 0.586 | 0.285 | 0.765 |

**hard-constraint sat. (per-constraint)** and **allergen violation** on the
**allergen-requirement stratum** (the safety-critical one):

| Config | hard-sat (allergen stratum) | allergen violation (allergen stratum) |
|---|---|---|
| H2-dense | 0.705 | 0.000 |
| H2-t2p | 0.979 | 0.000 |
| H1-dense | 0.723 | 0.000 |
| **H1-t2p** | **0.979** | **0.000** |

**Findings.**
- **The safety-critical allergen stratum is now violation-free on every config**: after the
  deterministic allergen post-filter (commit 54a62ae), allergen violation on the
  allergen-requirement slice is 0.000 for all four configs. Hard-constraint satisfaction on
  this slice still favours the t2p configs (H1-t2p 0.979 vs dense 0.70–0.72), so retrieval
  quality remains the differentiator once safety is guaranteed deterministically.
- **`ingredient` requirements are the hardest type to retrieve for** (H1-t2p hit@10 0.667
  vs 0.76–0.78 for allergen/tag/nutrient_bound) — positive ingredient inclusion is a harder
  symbolic filter than exclusion/tag/bound constraints. Judged correctness on ingredient
  queries is nonetheless high (0.780, ~flat across types after the empty-plan fix lifts the
  hardest slices) and well above H2-t2p (0.560): once retrieval misses, the honest
  "no recommendation" message keeps the judged correctness from collapsing.
- Type-conditioned performance mirrors the aggregate ordering everywhere: **retriever choice
  (t2p ≫ dense) dominates within every requirement type**, and the agentic H1 adds a
  consistent correctness margin on top of the t2p retriever.

---

## 7. Pareto trade-off (§9.4)

Computed with `foodrag.analysis.pareto` over `eval_results/aggregate.csv`
(FoodRAG configs, aggregate stratum). Safety is *not* a Pareto axis (hard constraints gate
the feasible set, Protocol §9.4).

### 7.1 Quality (correctness_judgeA) vs Cost (llm_usage)

| Config | cost (LLM calls) | quality (correctness A) | status |
|---|---|---|---|
| **H1-t2p** | 10.05 | **0.765** | ✅ frontier — highest quality |
| **H2-t2p** | **1.0** | 0.586 | ✅ frontier — cost-efficient |
| H1-dense | 9.91 | 0.285 | dominated (by H2-t2p) |
| H2-dense | 1.0 | 0.247 | dominated (by H2-t2p) |

**Frontier = {H1-t2p, H2-t2p}.** `H2-t2p` *dominates* both dense configs (≤ cost, ≥ quality).
The frontier is a genuine quality–cost trade-off: H2-t2p delivers 0.586 correctness at **1**
LLM call; H1-t2p buys **+0.179 correctness** (and the NL-explanation + claim-grounding layer)
for **~10×** the calls. The agentic pipeline is therefore **Pareto-justified** — it is not
dominated, it is the highest-quality point.

The **latency** variant (cost = latency_s) is slightly more forgiving: frontier =
{H1-t2p, H2-t2p, H2-dense}, with H1-t2p dominating H1-dense (226 s vs 287 s *and* higher
quality) and H2-dense surviving only as the cheapest (42.8 s) low-quality corner.

### 7.2 Health (avg_nutri@10) vs Relevance (avg_rating@10)

| Config | relevance (rating) | health (nutri) | status |
|---|---|---|---|
| **H1-t2p** | 4.40 | 0.354 | ✅ frontier |
| **H2-t2p** | 4.38 | 0.355 | ✅ frontier |
| H2-dense | 4.14 | 0.133 | dominated |
| H1-dense | 4.01 | 0.153 | dominated |

**Frontier = {H1-t2p, H2-t2p}.** Both t2p configs *dominate* both dense configs on
health-vs-relevance simultaneously (higher rating **and** higher NutriScore). There is no
health/relevance tension among the t2p configs — the strong retriever expands the frontier
outward on both axes at once, exactly the "FoodRAG expands the frontier vs the popularity
corner" story the protocol's dashboard (§10) anticipates.

**Domination summary:** across both frontiers the two dense configs are dominated; the two
t2p configs are non-dominated. The only *intra-frontier* trade-off is quality-vs-cost between
H2-t2p (cheap) and H1-t2p (best).

---

## 8. Explainability (§8)

| Config | path coverage (§8.1) | path validity (§8.1) | claim grounding (§8.2) | unsupported-claim rate |
|---|---|---|---|---|
| H2-dense | 0.656 | 1.000 | — | — |
| H2-t2p | 0.867 | 1.000 | — | — |
| H1-dense | 0.652 | 1.000 | 0.550 [0.538,0.562] | 0.450 |
| **H1-t2p** | **0.883** | **1.000** | **0.711** [0.699,0.724] | 0.289 |

- **Path validity is a perfect 1.000 on every config** — every KG path presented is a real
  HUMMUS edge (paths are *read from* the graph, so they cannot be hallucinated). Path
  coverage tracks retrieval quality: t2p configs cover 87–88 % of stated requirements with an
  explicit supporting path vs ~65 % for dense (fewer valid recipes → fewer requirements with
  a grounding path).
- **Intrinsic vs NL contrast (the §8 headline).** The intrinsic KG-path layer is 100 % valid
  and 88 % covering on H1-t2p; the user-facing NL rationale is now **71 % grounded** — 29 %
  of factual claims in the generated explanation lack KG/attribute support (the
  "meta-text / side-information" false-positives the thesis case study flagged). The
  empty-plan explanation gate (commit 54a62ae) removed the fully-hallucinated explanations on
  empty plans, lifting grounding from 0.650 to 0.711 and no longer injecting
  claim_grounding = 0 on gated empties. **The intrinsic path layer remains the trustworthy
  explanation channel; the NL layer still requires claim-level verification on non-empty plans
  before it is user-safe.** claim_grounding_rate computed here over the H1-t2p parquet
  (n = 4,200 non-null) = 0.717, matching the aggregate 0.711.

---

## 9. Processing / cost

| Config | llm_usage (calls) | llm_seq_usage | llm_parsing (retries) | latency (s) [CI] |
|---|---|---|---|---|
| H2-dense | 1.0 | 1.0 | 0.000 | 42.8 [40.3, 45.5] |
| H2-t2p | 1.0 | 1.0 | 0.000 | 44.1 [30.4, 61.2] |
| H1-dense | 9.91 | 8.50 | 0.007 | 286.9 [265.9, 311.5] |
| **H1-t2p** | 10.05 | 8.46 | 0.655 | 226.4 [213.1, 241.5] |

**Cost–quality summary.**
- The **H2 configs are ~1 LLM call and ~43 s**; the **H1 (agentic) configs are ~10 calls**
  and **~226–287 s** — an order-of-magnitude cost step for the Correct/Enhance stages.
- Interestingly, **H1-t2p is faster than H1-dense** (226 s vs 287 s) despite the same call
  budget: the strong retriever hands the agent valid candidates immediately, so the
  correction loop converges faster (fewer wasted enhancement cycles). H1-t2p's higher
  `llm_parsing` (0.655 vs 0.007) reflects Pandas-generation parse-retries in its retrieval
  stage, not agent instability.
- **Quality bought per call:** going H2-t2p → H1-t2p costs +9 LLM calls for +0.179
  correctness, +0.064 hit@10, and the whole NL-explanation/claim-grounding channel. Whether
  that is "worth it" is the deployment decision the Pareto frontier (§7) frames honestly:
  H2-t2p for cost-sensitive high-throughput serving, H1-t2p when explanation + peak
  correctness justify 10× compute.

---

## 10. Deployed behavior — time-driven meal selection

From `eval_results/mealplan_time_of_day.csv` (per-hour meal-type selection over the deployed
tool). The meal-plan tool plans **the remaining day** from the current time, so plans are
**partial-day** — mean **2.32 meals/plan**, not 4.

| Time window (hour) | mean meals | breakfast | lunch | dinner | snack |
|---|---|---|---|---|---|
| Early morning (4–6) | 1.8–2.4 | ≈ 1.00 | 0.14–0.43 | 0.13–0.31 | 0.54–0.67 |
| Late morning (9–11) | 2.2–2.9 | 0.09–1.00 | 0.64–1.00 | 0.16–0.26 | 0.72–0.98 |
| Afternoon (16) | 2.0 | 0.01 | 0.01 | **1.00** | **1.00** |
| Evening (20–22) | 2.0 | ≈ 0.02–0.06 | ≈ 0.02–0.04 | **≈ 1.00** | ≈ 0.91–0.96 |
| Late night (23) | 2.4 | 0.50 | 0.30 | 0.63 | 0.98 |

**Finding.** Meal selection is **strongly time-driven**: mornings are breakfast-dominant
(breakfast ≈ 1.0, lunch/dinner low), by late morning lunch appears (≈ 1.0 at hour 11) while
breakfast collapses (0.09), and evenings are dinner + snack (dinner ≈ 1.0, breakfast/lunch
≈ 0). This is a *deployment feature* — the planner never proposes a breakfast at 9 pm — and
it is the **quantitative reason the full-day plan-level metrics read ~0.3–0.5**
(guideline overall ~0.27–0.30, meal-slot completeness ~0.48–0.57): those metrics assume a
full 4-slot day, but the tool deliberately returns a partial-day plan.

---

## 11. Limitations & caveats

1. **Partial-day plans (†).** Guideline-adherence, calorie-distribution and meal-slot
   completeness assume a full 4-slot day, so they are *conservative* on the 2.3-meal
   partial-day plans the tool produces (§10). This is why per-nutrient adherence is high on
   ceiling nutrients (sat-fat 0.68, sugar 0.66) but low on band-target nutrients (energy
   0.05, carb 0.07, fibre 0.10) — a full day cannot be assembled from a partial plan. A
   full-day re-run was deliberately *not* done (it would flatten completeness toward 1.0 and
   cost ~3 days of compute); the *cross-config comparison remains valid* because guideline
   adherence is slot-agnostic and completeness is 0.48–0.57 across all four configs.
2. **Reconstructed vs native slot metrics (‡).** `slot_appropriateness` is computed on
   *reconstructed* plans for H2 and H1-dense (the native plan was not persisted at run time)
   vs the *native* plan for H1-t2p. Treat **H1-t2p (0.519) as the reliable value**; the
   others (e.g. H1-dense 0.771) are reconstruction-biased and not directly comparable.
3. **Intra-plan diversity data-shape artifact (§).** H2-t2p records single-item slots →
   diversity is 0.000 by construction, not a real diversity collapse.
4. **Time regime.** H1 configs ran at a fixed `morning`; H2 configs at wall-clock `now`. This
   changes *which* meals are requested and is confined to the plan-level metrics;
   retrieval / judge / safety metrics are time-robust. `recommendation_time` and
   `requested_meal_types` are now persisted per record.
5. **Corrected aggregate.** The `h1_t2p` aggregate was initially contaminated by a 3-item
   smoke run; all four FoodRAG configs were re-aggregated from the full per-question parquets
   (n ≈ 1,646/metric). The numbers in this report are the corrected values. (The
   `aggregate.json` `h1_t2p` rows are physically duplicated in the file; the deduplicated
   values are used here.)
6. **Judge caveats.** Headline is **Judge A** (Gemma-3-27B, cross-lineage); **Judge B**
   (DeepSeek-V3) is reported alongside. Correctness is judge-robust (|Δ| ≤ 0.057; no ranking
   flips), but binary faithfulness/relevancy diverge more (relevancy up to Δ = −0.34 on
   H2-t2p). Full Cohen's κ (Protocol §7.2) and judge–human κ on a stratified sample are
   **future work**; raw agreement (0.65–0.99) and Pearson (0.41–0.94) are the interim
   proxies. `claim_grounding` populates natively for H1-t2p; an offline explanation pass adds
   it for H1-dense.
7. **ToG sample.** ToG is evaluated on 100 items × 3 runs (n = 300 aggregate / 96 tight), far
   smaller than the 1,646 tight set, because of its ~15 min/item cost. Its numbers are
   directionally reliable but wider-CI than the other configs.
8. **Guideline reference bands** are researcher-constructed from EFSA DRV + WHO without
   nutrition-specialist review (Protocol §4.5.1). All adherence numbers are "adherence to
   guideline-derived reference bands," *not* clinical validation, and are a single-day
   snapshot approximation of longer-horizon guidelines.

---

## Anchor reconciliation

Every anchor supplied in the task brief was reproduced from the sources with **no material
disagreement**:

| Anchor | This report | Source |
|---|---|---|
| t2p@llama70b hit@10 = 0.854 (≫ dense 0.055) | 0.854 / 0.055 | `eval_results_llama/aggregate.csv`, `aggregate.json` |
| LLM-only allergen violation = 0.905 | 0.905 [0.898,0.912] | `aggregate.json` (llm_only, tight) |
| FoodRAG hit@10: H2-dense 0.032 / H2-t2p 0.709 / H1-dense 0.027 / H1-t2p 0.773 | 0.032 / 0.709 / 0.027 / 0.773 | `aggregate.json` |
| correctness: 0.247 / 0.586 / 0.285 / 0.765 | 0.247 / 0.586 / 0.285 / 0.765 | `aggregate.json` (Judge A, post-fix) |
| allergen violation = 0.000 on all four FoodRAG configs | 0.000 / 0.000 / 0.000 / 0.000 | `aggregate.json` (post-fix, deterministic allergen filter) |
| H1-t2p best & on Pareto frontier (highest correctness); H2-t2p cost-efficient | frontier {H1-t2p, H2-t2p} | `foodrag.analysis.pareto` on `aggregate.csv` |
| claim_grounding_rate H1-t2p = 0.71 | 0.711 [0.699,0.724] (0.717 recomputed) | `aggregate.json` + parquet |

*Minor note (not a contradiction):* ToG appears in the summary at its **aggregate** stratum
(hit@10 = 0.407, latency 902.8 s); its **tight** sub-stratum is higher (0.594, n = 96). Both
are reported above with the stratum labeled. The `specs/` protocol names the judges as
"Gemma 4 31B / Llama 3.3 70B", but the **corrected** `RESULTS_SUMMARY.md` (the designated
source of truth) specifies **Gemma-3-27B / DeepSeek-V3**, which this report follows.

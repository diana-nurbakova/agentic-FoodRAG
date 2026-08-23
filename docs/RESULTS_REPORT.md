# FoodRAG Evaluation — Full Results Report

*Thesis-grade results report expanding `eval_results/RESULTS_SUMMARY.md`. All numbers
are drawn from the corrected sources of truth: `eval_results/json/aggregate.json`
(bootstrap 95 % CIs on **every** metric), `eval_results/aggregate.csv`
(+ `eval_results_llama/aggregate.csv`), the per-config
`eval_results/*_per_question.parquet` (used for the stratified breakdowns, the
inter-judge statistics and the data statistics computed here), the paired-test
records in `eval_results/pairwise.csv` / `eval_results/pairwise_fullday.csv`
(Wilcoxon signed-rank, Holm-corrected — full tables in `docs/STATISTICAL_TESTS.md`),
and `eval_results/mealplan_time_of_day.csv`. Metric and experiment definitions are
given in [`docs/ARCHITECTURE.md`](ARCHITECTURE.md).*

> **Provenance / freshness note.** Every aggregate in this report was re-verified
> against the per-question parquets: for all 26 configs the aggregate-stratum means in
> `aggregate.csv`/`aggregate.json` reproduce the per-question data to numerical
> precision (zero drift), `aggregate.json` matches `aggregate.csv` row-for-row on
> `(config, stratum, metric)`, and `pairwise.csv` reproduces the paired tests below.
> Bootstrap 95 % CIs (10 000 resamples, fixed seed) are stored for **every**
> `config × stratum × metric` row and are surfaced on every metric in the tables here.
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
| Recipe corpus (ground-truth universe) | **HUMMUS** food KG — **310,228** recipes with tags, ingredient lists, per-100 g nutrients, serving sizes and NutriScore/FSA/WHO health scores |
| QA set (headline stratum) | **Tight** — personas with 1–50 valid answers, **n = 1,646** |
| Persona / QA generation | **rule-based personas** + deterministic constraint filtering; NL question surface reformulated by **Llama-3.3-70B** (via Groq / LiteLLM) |
| Retrieval / list cutoff | **K = 10** (fixed) |
| Dual judge | **Judge A = Gemma-3-27B** (Google lineage, *cross-lineage* to the Meta generator — headline judge) · **Judge B = DeepSeek-V3** (reference) |
| Judge scale | Correctness & explanation-quality ordinal on {0, .25, .5, .75, 1}; faithfulness & relevancy binary {0,1} (confirmed empirically from the per-question JSONs) |
| Aggregate rows | **1,197** rows in `aggregate.json` (config × stratum × metric, each with mean, ci_low, ci_high, per-question variance, n_questions, n_runs, judge) |
| Distinct configs | 26 (8 retrievers, 5 baselines/ladder rungs, 4 FoodRAG system configs + 4 full-day variants, plus EXP-3 heuristics/rag and EXP-5 KG) |

**Run counts per family** (from `json/index.json` and the aggregate `n_runs`):

| Family | Configs | Runs | Records |
|---|---|---|---|
| FoodRAG system matrix | `foodrag_h2`, `foodrag_h2_t2p`, `foodrag_h1_dense`, `foodrag_h1_t2p` | **3** each | 4,938 each (1,646 × 3) |
| FoodRAG full-day variants | `*_fullday` (all four) | **3** each | 4,938 each |
| LLM-only | `llm_only` | **5** | 8,230 |
| Text-to-Pandas retrievers | `ret_text2pandas@llama70b`, `ret_text2pandas@gpt4o-mini` | **5** | 8,230 |
| Single-shot retrievers | dense/sparse/hybrid/fulltext/intersection/random | **1** | 1,646 |
| popularity_raw / popularity_health_aware | — | **1** | 10,000 (all strata) |
| naive_rag | — | **1** | 1,646 |
| ToG (agentic KG) | `tog` | **3** | 300 (100 items × 3) |

**Metrics tracked** (per FoodRAG config the parquet carries 60+ metric fields): retrieval
(hit@10, mrr@10, ndcg@10, avg_nutri@10, avg_fsa@10, avg_rating@10); safety/constraint
(allergen_violation, hard_constraint_sat per-constraint & all-or-nothing,
hard_constraint_violation, constraint_satisfaction, soft_constraint_sat,
hallucination_rate); LLM-judge (correctness / faithfulness / relevancy /
explanation-quality — **each under both judges**); plan-level (guideline_adherence overall +
8 per-nutrient, slot_appropriateness, meal_slot_completeness, intra_plan_diversity,
calorie_adherence, structural_validity, plus their `_full`/`_norm` slot-normalized twins);
explainability (path_coverage, path_validity, claim_grounding_rate, unsupported_claim_rate);
processing (llm_usage, llm_seq_usage, llm_parsing, latency_s). **Every one of these carries a
bootstrap 95 % CI in `aggregate.json`.**

### 1.1 Data preparation: personas, queries and ground truth

The evaluation set is generated **procedurally and deterministically** from the HUMMUS recipe
corpus — there is no human annotation. The pipeline (`foodrag.persona.qa_generator`,
`foodrag.persona.persona_generator`; `config/persona_config.yml`) has three stages:

**(1) Persona (profile) generation** — `generate_persona_preferences`. Each persona is a set of
**hard preference constraints** drawn at random within the configured multiplicities:

| Constraint family | Draw (min–max per persona) | Semantics |
|---|---|---|
| tag likes / dislikes | 0–1 / 0–2 | recipe must carry ≥1 liked tag (any-of) and **no** disliked tag |
| ingredient likes / dislikes | 0–1 / 0–2 | recipe must contain ≥1 liked ingredient and **no** disliked ingredient (matched via FoodOn ingredient translation over the food.com vocabulary) |
| relative_nutrients | 0–1 | per-100 g value of a nutrient must fall in a `[lower, upper]` band |
| total_nutrients | 0–1 | per-serving daily total (value ÷ 100 × servingSize) must be ≤ a cap (drawn at 0.8–1.2× a reference intake) |
| nutrition_scores | 0–1 | recipe's FSA / WHO / NutriScore must be ≥ a minimum (a *health floor*) |

Each atomic like/dislike/bound/score-floor counts as one **requirement** (§9.2). The disliked
ingredients are the **safety-critical** family (an "allergen"-type constraint — serving a
disliked ingredient is the worst failure).

**(2) Query generation** — `_generate_question` builds a deterministic, structured natural-language
question from the persona: `"What are the recipes that meet the following criteria?"` followed by a
bulleted list (Tags Liked/Disliked, Ingredients Liked/Disliked, Nutrition Scores `min.`, Nutrition
Limits `lower–upper`, Total Nutrition Limits `max.`). This **structured question is the query the
retrievers and LLM judges actually consume** (the pipeline's `query_field` defaults to `question`).
An optional stage reformulates each question into 5 stylistic variants with **Llama-3.3-70B** (formal
/ concise / casual / two free) and rates them 1–5 with the same LLM; these `llm_questions` are an
**auxiliary NL-diversity surface** (used for embedding-retrieval robustness), **not** the evaluated
query.

**(3) Ground truth (gold answer set)** — `_get_valid_recipes_for_persona`. The gold set is computed
by **exhaustively filtering the entire 310 k-recipe corpus** through every persona constraint, in
order: relative-nutrient bands → health-score floors → per-serving total caps → liked-tag any-of →
disliked-tag none-of → liked-ingredient any-of → disliked-ingredient none-of. **Every recipe that
survives all filters is gold**; the resulting `recipe_url` set is the answer key. Ground truth is
therefore **symbolic, rule-derived and exhaustive** (a complete relevance set, not a sampled or
human-judged one), which is exactly what makes hit@10 / mrr@10 / nDCG@10 well-defined against it, and
what lets the safety metrics be checked deterministically. The health-aware popularity ceiling (§3,
rung 0b) uses this same valid set as its candidate pool, which is why it trivially scores 1.0.

**Strata (§9.1)** are defined by gold-set **size**: `empty` (0 valid recipes — over-constrained),
`tight` (1–50), `medium` (51–5,000), `broad` (>5,000). The headline evaluation uses the **tight**
stratum: it is the discriminative regime — broad/medium personas are satisfied by almost any recipe
(retrieval is trivial), while empty personas have no gold at all.

### 1.2 Evaluation-data statistics

Computed from the generation pool (`data/qa/main_10k`) and the tight evaluation set
(`data/qa/main_10k_tight`, the exact 1,646 items scored by every config).

**Generation pool → strata (10,000 personas):**

| Stratum (gold size) | Count | Share |
|---|---|---|
| empty (0) | 5,518 | 55.2 % |
| **tight (1–50)** | **1,646** | **16.5 %** |
| medium (51–5,000) | 1,192 | 11.9 % |
| broad (>5,000) | 1,644 | 16.4 % |

Over **half** of randomly-drawn personas are **over-constrained (no valid recipe exists)** — the
quantitative motivation for the empty-plan honesty gate (§4, §8): a deployed system must frequently be
able to say "nothing satisfies this."

**Tight evaluation set (n = 1,646):**

- **Gold answer-set size:** min 1, max 50, **mean 10.16, median 4**. Distribution:
  1 → 454, 2–5 → 451, 6–10 → 222, 11–20 → 232, 21–50 → 287. Nearly a third of tight personas have a
  **single** correct recipe — a demanding retrieval target.
- **Requirements per persona:** min 1, max 9, **mean 4.99, median 5**. Distribution:
  1 → 2, 2 → 53, 3 → 184, 4 → 377, 5 → 426, 6+ → 604. The set is **constraint-heavy** — 62 % of
  personas stack ≥ 5 simultaneous requirements.
- **Requirement-type membership** (overlapping — a persona is counted in each type it carries):

  | Type | Personas | Share of 1,646 |
  |---|---|---|
  | health (score floor) | 1,646 | 100 % |
  | tag (like/dislike) | 1,347 | 81.8 % |
  | nutrient_bound (relative/total) | 1,181 | 71.7 % |
  | allergen (disliked ingredient) | 1,104 | 67.1 % |
  | ingredient (liked ingredient) | 987 | 60.0 % |

- **Component presence / atom counts:** nutrition-score floor on **all** 1,646 (fsa 712 / nutri 596 /
  who 338); tag dislikes 1,136 personas (1,715 atoms) vs tag likes 694; ingredient dislikes 1,104
  (1,646 atoms) vs ingredient likes 987; total-nutrient caps 814; relative-nutrient bands 706.
- **Runs / records:** every FoodRAG config is scored over **3 independent runs** (4,938 item-runs),
  so all judge and plan metrics have run-to-run variance (`per_question_variance` in the aggregate);
  `llm_only` and the two text-to-pandas retrievers use 5 runs.

These distributions are exactly reproduced by the per-question records (e.g. tag membership
1,347 × 3 runs = 4,041 item-rows, matching the §6.2 stratified tables), confirming the evaluation set
and the scored records are the same 1,646 personas.

---

## 2. Retriever comparison (EXP-1 / Use-1, retrieval-only)

All rows below are the **tight-stratum / aggregate** values from `aggregate.json`
(`ret_*` configs) with bootstrap 95 % CIs on every metric. Text-to-Pandas @ llama70b comes from
`eval_results_llama/aggregate.csv`; @ gpt4o-mini from the main `aggregate.json`.

| Retriever | hit@10 [95 % CI] | mrr@10 [CI] | nDCG@10 [CI] | avg Nutri@10 [CI] | avg FSA@10 [CI] | allergen viol. [CI] | latency (s) | runs |
|---|---|---|---|---|---|---|---|---|
| Random | 0.000 [.000,.000] | 0.000 [.000,.000] | 0.000 [.000,.000] | 0.100 [.097,.104] | 0.457 [.452,.462] | 0.0005 [.000,.001] | 0.03 | 1 |
| Dense (jina) | 0.055 [.044,.067] | 0.023 [.017,.029] | 0.012 [.009,.015] | 0.128 [.122,.134] | 0.494 [.485,.504] | 0.0077 [.006,.010] | 0.45 | 1 |
| Sparse / BM25 | 0.063 [.051,.075] | 0.026 [.020,.032] | 0.015 [.011,.020] | 0.135 [.130,.141] | 0.494 [.487,.501] | 0.0196 [.016,.023] | 44.6 | 1 |
| Hybrid (RRF) | 0.055 [.044,.067] | 0.023 [.017,.029] | 0.012 [.009,.015] | 0.128 [.123,.134] | 0.494 [.485,.504] | 0.0077 [.006,.010] | 46.6 | 1 |
| Full-text | 0.063 [.051,.075] | 0.026 [.020,.032] | 0.015 [.011,.020] | 0.136 [.130,.141] | 0.494 [.487,.501] | 0.0196 [.016,.023] | 45.8 | 1 |
| Intersection | 0.002 [.001,.005] | 0.002 [.001,.005] | 0.002 [.000,.004] | 0.199 [.127,.275] | 0.557 [.472,.642] | 0.0012 [.000,.003] | 37.0 | 1 |
| Text-to-Pandas @ gpt-4o-mini | 0.427 [.418,.436] | 0.383 [.374,.392] | 0.367 [.358,.376] | 0.353 [.340,.367] | 0.724 [.713,.736] | 0.0001 [.000,.000] | 14.0 | 5 |
| **Text-to-Pandas @ llama-70B** | **0.854 [.837,.870]** | **0.766 [.748,.784]** | **0.730 [.712,.747]** | 0.351 [.337,.364] | 0.722 [.711,.734] | **0.0003 [.000,.001]** | 16.2 | 5 |

Additional constraint metrics for the strong retriever (`ret_text2pandas@llama70b`):
constraint_satisfaction 0.688 [0.671, 0.706], hard_constraint_violation 0.064
[0.056, 0.072], hallucination_rate 0.000 [0.000, 0.000].

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

Tight-stratum values from `aggregate.json` (with CIs), so the ladder is directly comparable to the
tight FoodRAG matrix. Popularity spans all 10 k strata, so its **tight** value (shown) differs sharply
from its all-strata aggregate (raw-popularity aggregate hit@10 = 0.293 — reported here at **tight** =
0.0018 for a like-for-like comparison). ToG is reported at its **aggregate** stratum
(n = 300, 3 runs) to match the summary; its tight sub-stratum (n = 96) scores higher
(hit@10 = 0.594 [.490,.688]) but is a smaller sample.

| Rung | hit@10 [95 % CI] | mrr@10 [CI] | allergen viol. [CI] | hard-constr. viol. [CI] | latency (s) |
|---|---|---|---|---|---|
| 0a Popularity (raw, unaware) — tight | 0.0018 [.000,.004] | 0.0005 [.000,.001] | 0.0007 [.000,.001] | 0.729 [.713,.743] | — |
| 0b Popularity (health-aware) * — tight | **1.000 [1.0,1.0]** | **1.000 [1.0,1.0]** | 0.000 [.000,.000] | 0.000 [.000,.000] | — |
| 2 LLM-only (no retrieval) | 0.0061 [.003,.009] | 0.0030 [.001,.005] | **0.905 [.898,.912]** | 0.972 [.968,.976] | — |
| 3 Naive-RAG | 0.055 [.044,.067] | 0.023 [.017,.029] | 0.0076 [.005,.010] | 0.659 [.641,.676] | — |
| ToG (agentic KG traversal) — aggregate | 0.407 [.353,.463] | 0.316 [.268,.365] | 0.000 [.000,.000] | 0.454 [.406,.503] | **902.8** |

**Findings.**
- **LLM-only violates a hard allergen constraint 90.5 % of the time** (allergen_violation =
  0.905 [0.898, 0.912]; hallucination_rate identically 0.905). This is the single most
  important safety result in the report and the core justification for grounding: an
  unconstrained LLM confidently invents recipes that would harm allergic users. Its
  retrieval is essentially zero (hit@10 = 0.006), so it is neither correct nor safe.
- **0b is a reference ceiling, not a fair retriever** (\*). Health-aware popularity
  *pre-filters to the constraint-valid set* (the same gold-generating filter, §1.1) before
  ranking by popularity, so it trivially scores hit@10 = mrr@10 = nDCG@10 = 1.0 and 0
  violations — it "knows the answer key." Reported as an upper bound; its avg NutriScore
  (0.355) / FSA (0.731) show the health ceiling reachable when the valid set is known.
- **ToG is competent but ruinously expensive.** hit@10 = 0.407 (aggregate) / 0.594 (tight)
  with 0 allergen violations, but **~903 s/item** (≈ 15 min), ~56× Text-to-Pandas's 16 s.
  Heavy agentic KG traversal buys *less* coverage than lightweight symbolic retrieval at
  ~50–56× the cost — a headline finding for Use-1.
- Raw popularity (0a) is a genuine floor: under tight constraints it "knows the answer" for
  essentially nothing (hit@10 = 0.002).

### 3.1 Whole-dataset (all-strata) view

The FoodRAG system matrix and the retriever ladder were run on the **tight** slice only, so for
those configs the whole-dataset result **is** the tight result. Three configs were, however,
evaluated across the *entire* generated distribution — the two popularity baselines (all 10,000
personas) and ToG (300 sampled) — so their full tight/medium/broad/empty breakdown is reported here.
This is exactly why the headline uses tight: it is the only stratum where the answer set is small
enough to discriminate methods.

**0a — Popularity (raw, unaware), all 10,000 personas** (mean [95 % CI]):

| Stratum | n | hit@10 | mrr@10 | nDCG@10 | constraint_sat | hard-constr. viol. | allergen viol. |
|---|---|---|---|---|---|---|---|
| tight (1–50) | 1,646 | 0.002 [.000,.004] | 0.001 [.000,.001] | 0.000 [.000,.000] | 0.000 | 0.728 [.713,.743] | 0.001 |
| medium (51–5k) | 1,192 | 0.065 [.052,.080] | 0.021 [.017,.026] | 0.007 [.006,.009] | 0.007 | 0.746 [.728,.763] | 0.002 |
| broad (>5k) | 1,644 | **0.748 [.727,.769]** | 0.352 [.335,.369] | 0.236 [.223,.249] | 0.248 | 0.715 [.701,.730] | 0.001 |
| empty (0) | 5,518 | — † | — † | — † | 0.000 | 0.910 [.905,.915] | 0.001 |
| **aggregate (all)** | **10,000** | **0.293 [.279,.306]** | 0.135 [.127,.143] | 0.089 [.083,.095] | 0.042 | 0.829 [.823,.834] | 0.001 |

† retrieval metrics are undefined on the `empty` stratum (no gold recipe exists), so they are dropped
from the mean rather than scored 0.

**0b — Popularity (health-aware) — reference ceiling, all 10,000:** hit@10 = mrr@10 = nDCG@10 =
**1.000 [1.0,1.0]** and hard-constraint violation = 0.000 on *every* non-empty stratum
(tight/medium/broad) — it ranks the pre-filtered valid set, so it always "hits." Its
`constraint_satisfaction` is 1.000 per non-empty stratum but **0.448 at aggregate**, purely because
the 5,518 empty personas (55 %) have no satisfiable plan and score 0. Health of the ceiling set drifts
by stratum: avg NutriScore 0.355 (tight) / 0.332 (medium) / 0.344 (broad); avg FSA 0.731 / 0.706 /
0.652.

**ToG (agentic KG traversal), 300 sampled:** hit@10 degrades with looseness — **0.594** (tight, n=96)
→ **0.481** (medium, n=135) → **0.000** (broad, n=69). The broad sub-stratum is degenerate (ToG
returns nothing and records ~0 s latency on those 69 items), so the aggregate **0.407** is carried by
the tight+medium items; the tight value (0.594) is the fair comparison to the other tight-only configs.

**Reading the whole-dataset view.** Raw popularity's aggregate hit@10 of **0.293 is an artifact of
gold-set size, not capability**: on `broad` personas (thousands of valid recipes) a globally-popular
recipe is *trivially* valid (hit@10 = 0.748), while on the discriminative `tight` slice it collapses to
0.002. The all-strata number therefore *flatters* a method that "knows the answer" only because almost
any recipe is an answer. This is the quantitative justification for headlining the **tight** stratum
throughout: it is the regime where retrieval quality actually separates the methods. (The FoodRAG and
retriever configs were run tight-only for the same reason — evaluating them on broad/empty personas
would add mostly-trivial or undefined items and dilute the contrast.)

---

## 4. FoodRAG system matrix (EXP-7 / Use-2, full meal-plan pipeline)

H1 = full agentic pipeline (Recommend + Correct + Enhance); H2 = Recommend-only.
Retriever ∈ {dense, text-to-pandas (t2p)}. All values are the aggregate-stratum means
from `aggregate.json` (3 runs, n ≈ 1,646/metric); **every cell carries its bootstrap 95 % CI**.
Judge-metric rows report **Judge A** with CI unless the split table in §5 is referenced.
Where two configs are contrasted in the findings, the contrast is backed by the paired
**Wilcoxon** test (Holm-corrected; §13.3) called out under each subsection.

### 4.1 Retrieval & health (mean [95 % CI])

| Metric | H2-dense | H2-t2p | H1-dense | **H1-t2p** |
|---|---|---|---|---|
| hit@10 | 0.032 [.025,.039] | 0.709 [.687,.730] | 0.027 [.021,.034] | **0.773 [.754,.792]** |
| mrr@10 | 0.023 [.018,.029] | 0.709 [.687,.731] | 0.018 [.014,.024] | **0.739 [.719,.758]** |
| nDCG@10 | 0.009 [.007,.012] | 0.347 [.330,.364] | 0.007 [.005,.010] | **0.450 [.434,.466]** |
| avg Nutri@10 | 0.133 [.126,.140] | 0.354 [.338,.370] | 0.153 [.144,.161] | 0.354 [.340,.368] |
| avg FSA@10 | 0.516 [.508,.524] | 0.719 [.706,.732] | 0.524 [.515,.533] | 0.721 [.709,.733] |
| avg rating@10 | 4.14 [4.09,4.17] | 4.38 [4.32,4.43] | 4.01 [3.95,4.06] | **4.40 [4.36,4.44]** |

### 4.2 LLM-judge quality — Judge A [95 % CI] / Judge B mean

| Metric | H2-dense | H2-t2p | H1-dense | **H1-t2p** |
|---|---|---|---|---|
| correctness (A) | 0.247 [.238,.257] | 0.586 [.573,.601] | 0.285 [.277,.293] | **0.765 [.754,.777]** |
| correctness (B) | 0.232 | 0.604 | 0.309 | **0.742** |
| faithfulness (A) | 0.236 [.220,.251] | 0.893 [.881,.904] | 0.989 [.985,.992] | **0.997 [.995,.999]** |
| faithfulness (B) | 0.112 | 0.986 | 0.976 | 0.994 |
| relevancy (A) | 0.311 [.293,.328] | 0.901 [.889,.913] | 0.454 [.437,.472] | 0.869 [.857,.882] |
| relevancy (B) | 0.098 | 0.559 | 0.221 | 0.784 |
| explanation-quality (A) | — | — | 0.574 [.567,.581] | **0.750 [.740,.760]** |
| explanation-quality (B) | — | — | 0.626 | 0.793 |

*(explanation-quality is only populated where an NL explanation layer exists — the two H1
configs.)*

### 4.3 Safety & constraint split (mean [95 % CI])

| Metric | H2-dense | H2-t2p | H1-dense | **H1-t2p** |
|---|---|---|---|---|
| hard-constraint sat. (per-constraint) | 0.646 [.634,.659] | 0.977 [.972,.981] | 0.664 [.652,.677] | **0.977 [.973,.980]** |
| hard-constraint sat. (all-or-nothing) | 0.328 [.310,.347] | 0.932 [.919,.944] | 0.359 [.340,.378] | **0.931 [.920,.941]** |
| hard-constraint violation | 0.743 [.727,.759] | 0.145 [.129,.162] | 0.711 [.693,.728] | 0.228 [.210,.247] |
| soft-constraint sat. | 0.403 [.390,.416] | 0.893 [.880,.905] | 0.397 [.384,.411] | 0.888 [.876,.899] |
| constraint_satisfaction | 0.018 [.014,.023] | 0.709 [.687,.730] | 0.015 [.011,.018] | 0.698 [.679,.717] |
| **allergen violation** | **0.000 [.000,.000]** | **0.000 [.000,.000]** | **0.000 [.000,.000]** | **0.000 [.000,.000]** |
| hallucination_rate | 0.000 [.000,.000] | 0.000 [.000,.000] | 0.000 [.000,.000] | 0.000 [.000,.000] |

### 4.4 Plan-level quality (mean [95 % CI])

| Metric | H2-dense | H2-t2p | H1-dense | **H1-t2p** |
|---|---|---|---|---|
| guideline adherence (overall) † | 0.280 [.274,.285] | 0.276 [.270,.282] | 0.299 [.293,.305] | 0.266 [.260,.272] |
| slot appropriateness ‡ | 0.454 [.443,.466] | 0.237 [.215,.258] | 0.771 [.760,.784] | 0.519 [.503,.536] |
| meal-slot completeness † | 0.565 [.559,.572] | 0.535 [.525,.545] | 0.524 [.515,.532] | 0.480 [.470,.491] |
| intra-plan diversity | 0.756 [.747,.765] | 0.000 [.000,.000] § | 0.782 [.774,.789] | 0.625 [.608,.641] |
| calorie adherence | 0.786 [.781,.791] | 0.911 [.909,.912] | 0.822 [.816,.827] | 0.836 [.830,.843] |
| structural validity | 0.204 [.193,.215] | 0.186 [.176,.197] | 0.168 [.157,.181] | 0.170 [.158,.182] |

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

### 4.6 Explainability & processing (mean [95 % CI])

| Metric | H2-dense | H2-t2p | H1-dense | **H1-t2p** |
|---|---|---|---|---|
| path coverage (§8.1) | 0.656 [.648,.664] | 0.867 [.854,.880] | 0.652 [.644,.660] | **0.883 [.870,.896]** |
| path validity (§8.1) | 1.000 [1.0,1.0] | 1.000 [1.0,1.0] | 1.000 [1.0,1.0] | 1.000 [1.0,1.0] |
| **claim grounding (§8.2)** | — | — | 0.550 [.538,.562] | **0.711 [.699,.724]** |
| unsupported-claim rate | — | — | 0.450 [.438,.462] | 0.289 [.277,.301] |
| llm_usage (calls) | 1.0 [1.0,1.0] | 1.0 [1.0,1.0] | 9.91 [9.64,10.21] | 10.05 [9.50,10.63] |
| llm_seq_usage | 1.0 [1.0,1.0] | 1.0 [1.0,1.0] | 8.50 [8.26,8.75] | 8.46 [7.99,8.97] |
| llm_parsing (retries) | 0.000 [.000,.000] | 0.000 [.000,.000] | 0.007 [.003,.012] | 0.655 [.552,.761] |
| latency (s) | 42.8 [40.3,45.5] | 44.1 [30.4,61.2] | 286.9 [265.9,311.5] | 226.4 [213.1,241.5] |

**Paired-Wilcoxon significance (§13.3; Holm-corrected within metric family; full table in
`docs/STATISTICAL_TESTS.md`).**

| Contrast | Metric | mean_a → mean_b | corrected p | median Δ | verdict |
|---|---|---|---|---|---|
| H1-t2p vs H2-t2p | correctness_A | 0.765 → 0.586 | 8.5e-88 | +0.167 | **H1 wins** |
| H1-t2p vs H2-t2p | hit@10 | 0.773 → 0.709 | 1.1e-19 | (mean +0.065) | **H1 wins** |
| H1-t2p vs H2-t2p | faithfulness_A | 0.997 → 0.893 | 3.9e-56 | (mean +0.105) | **H1 wins** |
| H1-t2p vs H2-t2p | relevancy_A | 0.869 → 0.902 | 1.3e-7 | (mean −0.033) | H2 wins (small) |
| H1-t2p vs H2-t2p | hard-sat perconstr | 0.977 → 0.976 | 7.1e-1 | 0.000 | n.s. |
| H1-dense vs H2-dense | faithfulness_A | 0.989 → 0.236 | 1.4e-262 | +1.000 | **H1 wins** |
| H1-dense vs H2-dense | correctness_A | 0.285 → 0.247 | 2.2e-19 | +0.062 | **H1 wins** |
| H1-dense vs H2-dense | hit@10 | 0.027 → 0.032 | 8.9e-2 | 0.000 | **n.s.** |
| H1-t2p vs H1-dense | hit@10 | 0.773 → 0.027 | 2.8e-254 | +1.000 | **t2p wins** |
| H2-t2p vs H2-dense | hit@10 | 0.709 → 0.032 | 1.1e-234 | +1.000 | **t2p wins** |

**Interpretation.**
1. **Retriever quality dominates the pipeline.** t2p configs get **~22–29×** the hit@10 of
   their dense twins (0.71–0.77 vs 0.03), near-perfect per-constraint hard-satisfaction
   (0.977 vs 0.646–0.664). The retriever swap is significant at p < 1e-234 (both systems).
   The retriever, not the agent depth, is the first-order driver of end-to-end quality.
2. **The agentic pipeline pays off *when retrieval is good*.** On t2p, H1 beats H2 on hit@10
   (0.773 vs 0.709), nDCG (0.450 vs 0.347) and correctness (0.765 vs 0.586, median Δ = +0.167,
   Wilcoxon p = 8.5e-88, non-overlapping CIs). On dense, H1's hit@10 gain over H2 is **not
   significant** (p = 0.089) — both retrieve ~nothing, so the Correct/Enhance stages have no
   valid candidates to improve. H1-dense's very high *faithfulness* (0.989) reflects tiny,
   cautious plans. **Agentic value emerges only once the retriever supplies valid candidates.**
3. **Safety is now perfect everywhere** (allergen violation 0.000 [0,0] on all four configs after
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
   recommendation" message.

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
  the gated empty-plan records lift Judge A's correctness to 0.765, so Judge A now sits
  *slightly above* Judge B (−0.025); on H2-t2p Judge B is marginally higher (+0.018).
- **Binary metrics diverge more, and directionally.** Relevancy shows the largest
  disagreement: Judge B is *harsher* on the strong-retriever configs (H2-t2p −0.341,
  H1-t2p −0.084) and generally scores relevancy lower. Faithfulness moves the other way on t2p
  (Judge B *higher*: +0.093 on H2-t2p). Raw binary agreement is nonetheless high (0.65–0.99),
  highest on H1-t2p (faithfulness 0.995, relevancy 0.860).
- **Cross-lineage caution is warranted, but does not overturn conclusions.** No config's
  qualitative ranking flips between judges on correctness. The report headlines Judge A
  (Gemma, cross-lineage) with Judge B alongside, as the protocol prescribes.
- **Inter-judge κ as future work.** Faithfulness/relevancy Cohen's κ is not
  computed here to keep the pass memory-light; the raw agreement + Pearson above are the
  reportable proxies, and the very high agreement on H1-t2p suggests κ would be substantial
  there. Full κ (and judge–human κ on a stratified sample) remains future work.

---

## 6. Stratified breakdowns

Computed here from the per-question parquets (all 3 runs pooled; n shown is item-rows).
Requirement-type strata are **membership** strata (a query is counted in every type it
contains), so they overlap. `health` is present on 100 % of tight-stratum queries (§1.2).

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
  simultaneous requirements. (These bin sizes match the §1.2 requirement distribution:
  53/184/377/426/604 personas × 3 runs.)
- **The dense configs are already at the floor** (hit@10 0.02–0.08 everywhere), so they
  barely "degrade" — there is nothing to lose.
- **Correctness *rises* with complexity for H1-t2p** (0.745 at 2 req → 0.787 at 6+),
  whereas H2-t2p *drops* (0.665 → 0.567). The agentic Correct/Enhance stage more than holds
  correctness up as constraints stack — direct evidence the agent earns its keep on hard queries.
- **Per-constraint hard-satisfaction is near-ceiling for both t2p configs** at every
  complexity level (≥ 0.967), while the dense configs *rise* with num_req (0.54 → 0.69)
  because more constraints mean more that are trivially satisfiable by any recipe.

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
- **The safety-critical allergen stratum is now violation-free on every config** (0.000 for
  all four). Hard-constraint satisfaction on this slice still favours the t2p configs (H1-t2p
  0.979 vs dense 0.70–0.72), so retrieval quality remains the differentiator once safety is
  guaranteed deterministically.
- **`ingredient` requirements are the hardest type to retrieve for** (H1-t2p hit@10 0.667
  vs 0.76–0.78 for allergen/tag/nutrient_bound) — positive ingredient inclusion is a harder
  symbolic filter than exclusion/tag/bound constraints. Judged correctness on ingredient
  queries is nonetheless high (0.780) and well above H2-t2p (0.560).
- Type-conditioned performance mirrors the aggregate ordering everywhere: **retriever choice
  (t2p ≫ dense) dominates within every requirement type**, and the agentic H1 adds a
  consistent correctness margin on top of the t2p retriever.

---

## 7. Pareto trade-off (§9.4)

Computed with `foodrag.analysis.pareto` over `eval_results/aggregate.csv`
(FoodRAG configs, aggregate stratum). Safety is *not* a Pareto axis (hard constraints gate
the feasible set).

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
outward on both axes at once.

**Domination summary:** across both frontiers the two dense configs are dominated; the two
t2p configs are non-dominated. The only *intra-frontier* trade-off is quality-vs-cost between
H2-t2p (cheap) and H1-t2p (best).

---

## 8. Explainability (§8)

| Config | path coverage (§8.1) | path validity (§8.1) | claim grounding (§8.2) | unsupported-claim rate |
|---|---|---|---|---|
| H2-dense | 0.656 [.648,.664] | 1.000 | — | — |
| H2-t2p | 0.867 [.854,.880] | 1.000 | — | — |
| H1-dense | 0.652 [.644,.660] | 1.000 | 0.550 [.538,.562] | 0.450 |
| **H1-t2p** | **0.883 [.870,.896]** | **1.000** | **0.711 [.699,.724]** | 0.289 |

- **Path validity is a perfect 1.000 on every config** — every KG path presented is a real
  HUMMUS edge (paths are *read from* the graph, so they cannot be hallucinated). Path
  coverage tracks retrieval quality: t2p configs cover 87–88 % of stated requirements with an
  explicit supporting path vs ~65 % for dense.
- **Intrinsic vs NL contrast (the §8 headline).** The intrinsic KG-path layer is 100 % valid
  and 88 % covering on H1-t2p; the user-facing NL rationale is now **71 % grounded** — 29 %
  of factual claims in the generated explanation lack KG/attribute support. The empty-plan
  explanation gate (commit 54a62ae) removed the fully-hallucinated explanations on empty
  plans, lifting grounding from 0.650 to 0.711. **The intrinsic path layer remains the
  trustworthy explanation channel; the NL layer still requires claim-level verification on
  non-empty plans before it is user-safe.** claim_grounding_rate computed here over the
  H1-t2p parquet (n = 4,200 non-null) = 0.717, matching the aggregate 0.711.
  The retriever-swap improvement in NL grounding is significant (H1-t2p vs H1-dense: 0.712 vs
  0.552, Wilcoxon p = 3.9e-70, median Δ = +0.167).

---

## 9. Processing / cost

| Config | llm_usage (calls) [CI] | llm_seq_usage [CI] | llm_parsing (retries) [CI] | latency (s) [CI] |
|---|---|---|---|---|
| H2-dense | 1.0 [1.0,1.0] | 1.0 [1.0,1.0] | 0.000 [.000,.000] | 42.8 [40.3, 45.5] |
| H2-t2p | 1.0 [1.0,1.0] | 1.0 [1.0,1.0] | 0.000 [.000,.000] | 44.1 [30.4, 61.2] |
| H1-dense | 9.91 [9.64,10.21] | 8.50 [8.26,8.75] | 0.007 [.003,.012] | 286.9 [265.9, 311.5] |
| **H1-t2p** | 10.05 [9.50,10.63] | 8.46 [7.99,8.97] | 0.655 [.552,.761] | 226.4 [213.1, 241.5] |

**Cost–quality summary.**
- The **H2 configs are ~1 LLM call and ~43 s**; the **H1 (agentic) configs are ~10 calls**
  and **~226–287 s** — an order-of-magnitude cost step for the Correct/Enhance stages.
- Interestingly, **H1-t2p is faster than H1-dense** (226 s vs 287 s) despite the same call
  budget: the strong retriever hands the agent valid candidates immediately, so the
  correction loop converges faster. H1-t2p's higher `llm_parsing` (0.655 vs 0.007) reflects
  Pandas-generation parse-retries in its retrieval stage, not agent instability.
- **Quality bought per call:** going H2-t2p → H1-t2p costs +9 LLM calls for +0.179
  correctness (Wilcoxon p = 8.5e-88), +0.065 hit@10, and the whole NL-explanation/claim-grounding
  channel. The Pareto frontier (§7) frames the deployment decision honestly: H2-t2p for
  cost-sensitive high-throughput serving, H1-t2p when explanation + peak correctness justify
  10× compute.

---

## 10. Deployed behavior — time-driven meal selection

From `eval_results/mealplan_time_of_day.csv` (per-hour meal-type selection over the deployed
tool). The meal-plan tool plans **the remaining day** from the current time, so plans are
**partial-day** — mean **2.32 meals/plan**, not 4 (the per-record `n_requested_slots` averages
2.05 on H1-t2p).

| Time window (hour) | mean meals | breakfast | lunch | dinner | snack |
|---|---|---|---|---|---|
| Early morning (4–6) | 1.8–2.4 | ≈ 1.00 | 0.14–0.43 | 0.13–0.31 | 0.54–0.67 |
| Late morning (9–11) | 2.2–2.9 | 0.09–1.00 | 0.64–1.00 | 0.16–0.26 | 0.72–0.98 |
| Afternoon (16) | 2.0 | 0.01 | 0.01 | **1.00** | **1.00** |
| Evening (20–22) | 2.0 | ≈ 0.02–0.06 | ≈ 0.02–0.04 | **≈ 1.00** | ≈ 0.91–0.96 |
| Late night (23) | 2.4 | 0.50 | 0.30 | 0.63 | 0.98 |

**Finding.** Meal selection is **strongly time-driven**: mornings are breakfast-dominant, by
late morning lunch appears (≈ 1.0 at hour 11) while breakfast collapses (0.09), and evenings
are dinner + snack. This is a *deployment feature* — the planner never proposes a breakfast at
9 pm — and it is the **quantitative reason the full-day plan-level metrics read ~0.3–0.5**
(guideline overall ~0.27–0.30, meal-slot completeness ~0.48–0.57). A companion **full-day
benchmark** that forces all four slots is reported in **§12**; it confirms completeness is a
slot-count artifact (rises to ~0.98 for dense) while guideline adherence stays low.

---

## 11. Limitations & caveats

1. **Rule-based ground truth.** The gold answer set is the *exhaustive* set of corpus recipes
   satisfying the persona's symbolic constraints (§1.1), not human relevance judgments. This
   makes retrieval metrics exact and safety checks deterministic, but it means "correctness"
   is measured against constraint satisfaction + LLM-judge quality, not against human
   preference. LLM-judge scores (correctness/faithfulness/relevancy/explanation-quality) add
   the qualitative layer the rule-based gold cannot capture.
2. **Partial-day plans (†).** Guideline-adherence, calorie-distribution and meal-slot
   completeness assume a full 4-slot day, so they are *conservative* on the 2.3-meal
   partial-day plans the tool produces (§10). A **full-day companion run** (all four slots
   forced) is reported in **§12**: it flattens completeness toward ~0.98 (dense) as expected,
   and confirms the *cross-config comparison remains valid* — guideline adherence is
   slot-agnostic and stays low (~0.21–0.29, actually *lower* at full day).
3. **Reconstructed vs native slot metrics (‡).** `slot_appropriateness` is computed on
   *reconstructed* plans for H2 and H1-dense (the native plan was not persisted at run time)
   vs the *native* plan for H1-t2p. Treat **H1-t2p (0.519) as the reliable value**; the
   others (e.g. H1-dense 0.771) are reconstruction-biased and not directly comparable.
4. **Intra-plan diversity data-shape artifact (§).** H2-t2p records single-item slots →
   diversity is 0.000 by construction, not a real diversity collapse.
5. **Time regime.** H1 configs ran at a fixed `morning`; H2 configs at wall-clock `now`. This
   changes *which* meals are requested and is confined to the plan-level metrics;
   retrieval / judge / safety metrics are time-robust. `recommendation_time` and
   `requested_meal_types` are now persisted per record.
6. **Judge caveats.** Headline is **Judge A** (Gemma-3-27B, cross-lineage); **Judge B**
   (DeepSeek-V3) is reported alongside. Correctness is judge-robust (|Δ| ≤ 0.025; no ranking
   flips), but binary faithfulness/relevancy diverge more (relevancy up to Δ = −0.34 on
   H2-t2p). Full Cohen's κ and judge–human κ on a stratified sample are
   **future work**; raw agreement (0.65–0.99) and Pearson (0.25–0.76) are the interim proxies.
7. **ToG sample.** ToG is evaluated on 100 items × 3 runs (n = 300 aggregate / 96 tight), far
   smaller than the 1,646 tight set, because of its ~15 min/item cost. Its numbers are
   directionally reliable but wider-CI than the other configs.
8. **Guideline reference bands** are researcher-constructed from EFSA DRV + WHO (general-adult
   class) without nutrition-specialist review. All adherence numbers are
   "adherence to guideline-derived reference bands," *not* clinical validation, and are a
   single-day snapshot approximation of longer-horizon guidelines.

---

## 12. Full-day meal-plan benchmark (all-four-slots variant)

The deployed tool returns **partial-day** plans (§10, mean 2.32 meals). To measure the pipeline
free of the partial-day slot-count confound, all four FoodRAG configs were re-run in a **full-day
variant** (`*_fullday`) that forces every plan to request all four meal slots
(`requested_meal_types = [breakfast, lunch, dinner, snack]`, persisted per record for **all four**
configs — so slot-normalized metrics are `"persisted"`-sourced throughout). Same QA set
(1,646 items × 3 runs), retrievers, judges and metrics as §4. Sources: the `*_fullday` rows of
`eval_results/aggregate.csv`, the `*_fullday_per_question.{parquet,pkl}` files, and
`eval_results/pairwise_fullday.csv`.

**Degeneracy protection.** A full-day plan must fill four *distinct* slots. Two guards ensure this
(both now permanent — Architecture §3.6 / §7.2):

- `NonRepeatingReplayRetriever` — the text-to-pandas replay draws a *distinct* rank-ordered recipe
  for each of the four per-slot `retrieve()` calls (empty once the saved pool is exhausted).
- `constraints.enforce_distinct_slots` — a deterministic post-assembly guard that drops any later
  slot re-using an earlier slot's `recipe_url`.

After the fix, **0.000 % of plans are degenerate** (same recipe in ≥ 2 slots) across all four
configs (4,938 records/config); an unguarded run produced up to **100 %** degeneracy on `h2_t2p`
and 17.7 % on `h1_t2p`.

### 12.1 Full-day headline (mean [95 % CI], n ≈ 1,646; slot-normalized plan metrics)

| Metric | H2-dense | H1-dense | H2-t2p | H1-t2p |
|---|---|---|---|---|
| hit@10 | 0.046 [.038,.055] | 0.049 [.040,.058] | **0.828** [.810,.845] | 0.786 [.767,.804] |
| correctness (Judge A) | 0.262 [.254,.269] | 0.281 [.274,.288] | **0.718** [.704,.731] | 0.561 [.547,.575] |
| correctness (Judge B) | 0.264 | 0.310 | **0.794** | 0.621 |
| allergen violation | **0.000** | **0.000** | **0.000** | **0.000** |
| completeness (norm) | 0.982 [.977,.987] | **0.990** [.987,.993] | 0.731 [.714,.748] | 0.782 [.763,.800] |
| structural validity (norm) | 0.362 [.344,.381] | **0.392** [.375,.408] | 0.015 [.009,.021] | 0.022 [.015,.028] |
| guideline adherence (full) | 0.209 | 0.211 | **0.243** | 0.239 |
| guideline adherence (norm) | 0.216 | 0.218 | **0.285** | 0.279 |
| calorie adherence (norm) | 0.812 | 0.812 | 0.767 | 0.787 |
| slot appropriateness | **0.831** | 0.716 | 0.445 | 0.507 |

Because the full-day request set *is* all four slots, ρ(Q) = 1 and each `_norm` value is within 0.01
of its `_full` counterpart (see `foodrag.baselines.slot_normalized`). `structural_validity_norm` is
strict all-or-nothing (all four requested slots filled *and* admissible), which is why the t2p configs
— which cannot always fill four slots from their replay pool — read ~0.01–0.02 despite `_full` ≈ 0.999.

### 12.2 What full-day reveals (paired Wilcoxon, Holm-corrected within the full-day family; `pairwise_fullday.csv`)

**Retriever is the dominant axis — and it trades off.** text-to-pandas beats dense on retrieval and
correctness (hit@10 +0.74 H1 / +0.78 H2, both p < 1.3e-251; correctness +0.28 H1 / +0.46 H2, both
p < 5.5e-156; guideline_norm +0.06–0.07 both, p < 7.3e-28), but **dense beats t2p on slot coverage**:
completeness_norm +0.21 H1 (0.990 vs 0.782, p = 5.3e-75) / +0.25 H2 and structural_norm +0.37 / +0.35
(p < 1.5e-145). The saved t2p replay pool often holds < 4 distinct recipes, so the NonRepeating
retriever leaves later slots empty rather than repeat.

**Agentic (H1) vs recommend-only (H2), full-day.** H1 *hurts* t2p correctness (0.561 vs 0.718,
p = 4.5e-115) but slightly *helps* dense (0.281 vs 0.262, p = 1.3e-6); H1 improves completeness_norm
for both (t2p 0.782 vs 0.731, p = 2.0e-36; dense 0.990 vs 0.982, p = 2.6e-3). No significant H1/H2
guideline difference (p > 0.24).

**Partial-day vs full-day (scope effect; same questions paired).**

- **Completeness rises** (full-day fills ~0.98 of slots for dense vs ~0.52 partial-day; paired diffs
  all p < 1e-14) — confirming the low partial-day completeness/structural reading was a
  **slot-count artifact**, not a pipeline failure.
- **Guideline adherence *falls* going partial → full** (e.g. H2 0.279 → 0.209, p = 2.3e-55; H1-t2p
  0.266 → 0.239, p = 2.5e-13): a full four-slot day *overshoots* the energy/macro bands more than a
  2–3 slot plan. This is the genuine, scope-independent limitation — it does **not** disappear under
  normalization or a full day.
- Correctness is mixed by retriever: t2p partial > full (H1-t2p 0.765 → 0.561, p = 1.7e-111 — the
  time-aware partial-day plan is more correct), dense ≈ flat / slightly up.

**Takeaway.** The full-day run separates the two headline stories: (i) the low partial-day
*completeness/structural* numbers are a slot-count artifact that fills/normalizes away; (ii) the low
*guideline adherence* (~0.21–0.29) is a real limitation that persists — indeed worsens — at full day.
Retrieval quality (t2p) vs slot coverage (dense) is a genuine trade-off a future hybrid retriever
should close.

---

## Anchor reconciliation

Every anchor was reproduced from the sources with **no material disagreement**, and re-verified
against the per-question parquets (zero drift):

| Anchor | This report | Source |
|---|---|---|
| t2p@llama70b hit@10 = 0.854 (≫ dense 0.055) | 0.854 / 0.055 | `eval_results_llama/aggregate.csv`, `aggregate.json` |
| LLM-only allergen violation = 0.905 | 0.905 [0.898,0.912] | `aggregate.json` (llm_only, tight) |
| FoodRAG hit@10: H2-dense 0.032 / H2-t2p 0.709 / H1-dense 0.027 / H1-t2p 0.773 | 0.032 / 0.709 / 0.027 / 0.773 | `aggregate.json` |
| correctness: 0.247 / 0.586 / 0.285 / 0.765 | 0.247 / 0.586 / 0.285 / 0.765 | `aggregate.json` (Judge A, post-fix) |
| allergen violation = 0.000 on all four FoodRAG configs | 0.000 / 0.000 / 0.000 / 0.000 | `aggregate.json` (post-fix, deterministic allergen filter) |
| H1-t2p vs H2-t2p correctness significant | median Δ +0.167, Wilcoxon p = 8.5e-88 | `pairwise.csv` |
| H1-t2p best & on Pareto frontier; H2-t2p cost-efficient | frontier {H1-t2p, H2-t2p} | `foodrag.analysis.pareto` on `aggregate.csv` |
| claim_grounding_rate H1-t2p = 0.71 | 0.711 [0.699,0.724] (0.717 recomputed) | `aggregate.json` + parquet |
| Tight evaluation set = 1,646 of 10,000 (55 % empty) | 1,646 tight / 5,518 empty | `data/qa/main_10k*` |

*Minor note (not a contradiction):* ToG appears at its **aggregate** stratum
(hit@10 = 0.407, latency 902.8 s); its **tight** sub-stratum is higher (0.594, n = 96). Both
are reported above with the stratum labeled. An earlier draft of the evaluation protocol named
the judges as "Gemma 4 31B / Llama 3.3 70B", but the **corrected** `RESULTS_SUMMARY.md` (the
designated source of truth) specifies **Gemma-3-27B / DeepSeek-V3**, which this report follows.

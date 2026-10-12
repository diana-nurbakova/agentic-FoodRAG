# FoodRAG Evaluation — Error Analysis

> Values for `foodrag_h1_t2p`, `foodrag_h1_dense`, `foodrag_h2`, `foodrag_h2_t2p` are reported pre-remediation (records at `74176e8^`). Commit `74176e8` filtered excluded ingredients from 72 stored outputs and replaced 421 empty-plan explanations, then re-scored; post-fix values appear only where labelled.

Companion to `eval_results/RESULTS_SUMMARY.md` and `docs/RESULTS_REPORT.md`. Where those
report *how well* the system performs, this document dissects *where and why it fails*. All
numbers are computed directly from the per-question §13.1 records — for the four FoodRAG
configs the pre-fix records (`git show 74176e8^:eval_results/<config>_per_question.parquet`),
otherwise `eval_results/json/<config>_per_question.json` (1,646 tight-stratum questions × 3 runs =
4,938 item-runs per FoodRAG config), the retriever records
(`ret_text2pandas@llama70b`: 5 runs; `ret_dense`: 1 run), and the gold QA pickle
`data/qa/main_10k_tight/recipe_question_answer_set.pkl`.

Configs (focus): **H2-dense** (`foodrag_h2`, Recommend-only + dense retriever), **H2-t2p**
(`foodrag_h2_t2p`, Recommend-only + Text-to-Pandas), **H1-dense** (`foodrag_h1_dense`,
agentic Recommend+Correct+Enhance + dense), **H1-t2p** (`foodrag_h1_t2p`, agentic +
Text-to-Pandas). Headline judge is **Judge A = Gemma-3-27B** (cross-lineage); **Judge B =
DeepSeek-V3** is the second (reference) judge — like Judge A, it is outside the Llama-3.3-70B
generator's family.

**NaN handling.** Retrieval/safety/constraint metrics are computed on every record. Judge and
grounding metrics are `None` on a minority of records (judge parse failures, or no explanation
emitted); rates below use the **defined-only denominator** for each metric and report that
denominator so the reader can see how much data backs each figure.

---

## 0. Data, personas & ground truth (what a "failure" is measured against)

The evaluation set is generated **procedurally**, so every error below is a failure against a
*known, rule-derived* answer key rather than a human label (see `docs/RESULTS_REPORT.md` §1.1–1.2
for the full pipeline).

- **Personas.** Each of the 1,646 tight questions is a rule-based persona: hard preference
  constraints drawn from `config/persona_config.yml` — tag likes (0–1)/dislikes (0–2), ingredient
  likes (0–1)/dislikes (0–2), per-100 g nutrient bands, per-serving daily caps, and health-score
  floors. The tight set is **constraint-heavy**: mean **4.99** requirements/persona (median 5;
  63 % carry ≥ 5), which is why hard-constraint and retrieval failures concentrate at high
  `num_requirements` (§3a).
- **Query.** The evaluated query is the deterministic structured question
  (`"What are the recipes that meet the following criteria?" + bulleted constraints`); the
  `llm_questions` reformulations are auxiliary and not scored.
- **Ground truth.** The gold answer set is the **exhaustive** set of HUMMUS recipes (270,284-recipe
  corpus, `recipes_df.csv`) satisfying *all* of a persona's constraints, computed by `_get_valid_recipes_for_persona`.
  A "retrieval MISS" therefore means the pipeline's top-10 missed a recipe that provably satisfies
  every stated requirement; an "allergen violation" means a served recipe provably contains a
  disliked ingredient. Both are exact, not judgment calls.
- **Why the empty-plan gate matters.** Of the 10,000 personas generated, **5,518 (55.2 %) are
  over-constrained — no recipe satisfies them** (the `empty` stratum). The deployed tool must often
  legitimately return *nothing*, which is precisely the case the empty-plan explanation gate (§6)
  handles: on the tight set the residual empty-plan rate is 8.6–10.7 % (§1), and honestly refusing
  those is scored as correct/faithful rather than hallucinated.
- **Requirement-type mix (of 1,646, overlapping):** health 1,646 (100 %), tag 1,347, nutrient_bound
  1,181, allergen 1,104, ingredient 987 — so the `allergen` (safety-critical) slice is 3,312 of the
  4,938 item-runs (§3c), and the `ingredient` slice (the hardest to retrieve, §3b) is 2,961.

---

## 1. Failure taxonomy and per-config rates

Rate = fraction of item-runs exhibiting the failure, over the *defined* denominator for that
metric (shown in the note). Failure definitions:

- **Retrieval MISS** — `hit_at_10 == 0` (gold recipe never entered the top-10 candidate set).
- **EMPTY plan** — the final plan has no recipe URLs and no `retrieved_ids`.
- **HARD-constraint violation** — `hard_constraint_violation > 0` or
  `hard_constraint_sat_allornothing == 0` (at least one hard requirement broken).
- **ALLERGEN violation** — `allergen_violation > 0` (**safety-critical**).
- **LOW correctness** — `correctness_judgeA < 0.5`.
- **UNFAITHFUL** — `faithfulness_judgeA == 0` (explanation contradicts the plan/KG).
- **UNSUPPORTED-claim-heavy** — `unsupported_claim_rate > 0.5` (H1 only; H2 emits no NL layer).
- **STRUCTURAL-invalid** — `structural_validity == 0`.

| Failure type | H2-dense | H2-t2p | H1-dense | **H1-t2p** | H1-t2p post-fix (not the evaluated system) |
|---|---|---|---|---|---|
| Retrieval MISS | **96.8%** | 29.1% | **97.3%** | 22.8% | post-fix 22.8% |
| EMPTY plan | 0.9% | 8.6% | 1.6% | 10.7% | post-fix 10.7% |
| HARD-constraint violation | 81.5% | 14.6% | 77.4% | 24.9% | post-fix 24.8% |
| ALLERGEN violation | **0.79%** | 0.06% | **0.55%** | 0.06% | post-fix 0.00% |
| LOW correctness (JudgeA<0.5) | 76.6% | 26.8% | 75.2% | 20.4% | post-fix 12.6% |
| UNFAITHFUL (JudgeA==0) | 76.8% | 10.8% | 1.1% | 8.0% | post-fix 0.3% |
| UNSUPPORTED-claim-heavy (>0.5) | n/a | n/a | 32.0% | 21.2% | post-fix 13.6% |
| STRUCTURAL-invalid | 79.3% | 81.4% | 82.8% | 83.1% | post-fix 83.1% |

**Remediation (commit `74176e8`, applied after evaluation).** The allergen filter removed
excluded-ingredient recipes from 72 stored plans, which sets the post-fix ALLERGEN row to 0.00%
on every config by construction: filter and metric are the same function. The empty-plan
explanation gate's fixed message replaced 421 H1-t2p explanations, and re-judging those records
moves H1-t2p's post-fix LOW-correctness to 12.6%, UNFAITHFUL to 0.3% and UNSUPPORTED-claim-heavy to 13.6% (post-fix column). The
judge credits the "no recommendation" message where it penalised the hallucinated prose (§6),
so these post-fix shifts score the replaced explanation text, not a change in the
recommendations. (The UNSUPPORTED-claim drop is also a denominator effect: replaced
explanations carry no claims, so their `unsupported_claim_rate` is undefined and they leave the
4,605 → 4,200 denominator; see §6.)

Defined denominators (per config, out of 4,938): retrieval/hard/allergen/structural
≈ 4,839–4,938; correctness ≈ 4,788–4,931; faithfulness ≈ 4,798–4,930; unsupported-claim
4,605 (H1-t2p) / 4,644 (H1-dense).

**Read-out.**
- **The dominant failure is retrieval, and it is a clean dense-vs-t2p split.** Both dense
  configs miss the gold ~97% of the time; both t2p configs miss ~23–29%. Every downstream
  failure (hard-constraint, low-correctness) tracks this: on dense, hard-constraint violation
  is ~80% and low-correctness ~76%; on t2p they drop to 15–25% and 20–27%.
- **The agentic H1 pipeline shifts the error profile even at fixed retriever.** H1-t2p vs
  H2-t2p: retrieval MISS drops 29.1%→22.8% and low-correctness 26.8%→20.4% (Correct/Enhance
  recovers gold and improves answers) — but **hard-constraint violation *rises* 14.6%→24.9%**
  and EMPTY plan rises 8.6%→10.7%. The agent trades a small amount of constraint-completeness for
  better recall and correctness (see §2 for why). Allergen violations are rare on both
  (0.06%), and an order of magnitude more frequent on the dense configs (0.55–0.79%; §4).
- **STRUCTURAL-invalid ~80% across *all four* configs is a metric artifact, not a system
  failure.** `structural_validity` demands a full 4-slot day; the deployed meal-planner is
  time-driven and returns partial-day plans (mean 2.32 slots — see RESULTS_SUMMARY §5), so a
  well-formed 2-meal plan scores 0. Confirmed: among H1-t2p `sv==0` records, 3,655/4,020 *have*
  a non-empty plan, and their mean `meal_slot_completeness` is 0.43. Treat this row as
  "not-a-full-day", not "malformed". It is the one row in this table that should **not** drive
  any engineering decision.
- **UNFAITHFUL collapses from 76.8% (H2-dense) to 1.1% on H1-dense and 8.0% on H1-t2p.**
  H2-dense's NL layer (a raw recommend-only description) frequently contradicts its near-empty
  plan; the H1 Enhance stage writes explanations tied to the actual plan, so faithfulness is
  high even when the plan is poor (H1-dense faithfulness 0.989 = cautiously describing a tiny
  plan). H1-t2p's 8.0% is concentrated in its empty plans, whose explanations narrate recipes
  that were never selected (faithfulness 0.922 overall). Replacing those explanations after
  evaluation gives post-fix UNFAITHFUL 0.3% and faithfulness 0.997.

---

## 2. Error-source decomposition — retrieval vs pipeline

The t2p FoodRAG configs **replay** the `ret_text2pandas@llama70b` retrieval, so we can localise
each error. For every question index *i* we compare the retriever's `hit@10` (mean over its
runs) against the FoodRAG config's `hit@10`, and bucket:

- **(a) Retriever MISS** (retriever hit=0) — *retrieval-limited*: the gold was never available.
- **(b) Retriever HIT, pipeline MISS** (retriever=1, foodrag=0) — *pipeline DROPPED the gold*.

(Each run-mean is rounded to a hit/miss at 0.5 before bucketing; the same rule is used on every
row.)
- **(c) Both hit** — retrieval succeeded and the pipeline preserved it.

| Comparison | (a) retriever MISS | (b) pipeline DROPPED | (c) both hit |
|---|---|---|---|
| **H2-t2p** vs ret_t2p | 14.2% (234) | **14.5% (239)** | 71.3% (1173) |
| **H1-t2p** vs ret_t2p | 14.2% (234) | **7.0% (116)** | 78.6% (1293) |
| H2-dense vs ret_dense | 94.5% (1555) | 4.3% (71) | 1.2% (20) |
| H1-dense vs ret_dense | 94.5% (1555) | 4.7% (77) | 0.9% (14) |
| n = 1,646 per row (H1-t2p: 3 undefined) |

**Interpretation.**
- **On t2p, the agentic pipeline halves the "dropped gold" error: bucket (b) falls
  from 14.5% (H2-t2p) to 7.0% (H1-t2p).** This is the clearest quantified win for the agent:
  given the *same* retriever, recommend-only H2 discards the gold recipe from a valid candidate
  set in 1-in-7 cases, while H1's Correct/Enhance loop recovers roughly half of those (both-hit rises
  71.3%→78.6%). The +6.4 pts of hit@10 in RESULTS_SUMMARY (0.709→0.773) is *almost entirely*
  this recovered bucket, not new retrieval.
- **Residual retrieval-limited error on t2p is 14.2%** — the irreducible ceiling for these
  configs. No pipeline change can fix these; they require a better retriever (or larger *k*).
- **On dense, the pipeline is irrelevant.** 94.5% of items are retriever misses; buckets (b)
  and (c) are noise (≤5%). The agent "has nothing to correct". This is the mechanism behind the
  RESULTS_SUMMARY finding that *the agentic pipeline pays off only when retrieval is good*. Note
  bucket (b) is marginally *larger* for H1-dense (4.7%) than H2-dense (4.3%): with no valid
  candidates, the extra Enhance/Correct churn slightly *hurts* the rare dense hit rather than
  helping.

**Localisation.** On the strong retriever, FoodRAG error decomposes as ≈ **14%
retrieval-bound plus ~7% pipeline-drop** (H1-t2p). The biggest remaining pipeline lever is the
~7% gold-dropping in Correct/Enhance; the biggest retrieval lever is the ~14% floor.

---

## 3. Error by query difficulty

### 3a. By number of requirements (`num_requirements`)

Failure rates for **H1-t2p** (and H2-t2p for contrast). Buckets 1 and 2 are tiny (n=6, n=159)
so read them cautiously.

**H1-t2p**

| nreq | n | ret MISS | low-corr | hard-viol |
|---|---|---|---|---|
| 1 | 6 | 0.0% | 83.3% | 0.0% |
| 2 | 159 | 14.6% | 14.0% | 21.0% |
| 3 | 552 | 17.6% | 20.0% | 22.7% |
| 4 | 1131 | 21.1% | 21.1% | 25.9% |
| 5 | 1278 | 22.9% | 21.7% | 23.7% |
| 6+ | 1812 | **26.2%** | 19.7% | 26.3% |

**H2-t2p**

| nreq | n | ret MISS | low-corr | hard-viol |
|---|---|---|---|---|
| 2 | 159 | 23.3% | 19.5% | 11.3% |
| 3 | 552 | 23.0% | 23.7% | 14.1% |
| 4 | 1131 | 28.2% | 25.7% | 15.8% |
| 5 | 1278 | 28.7% | 28.8% | 12.4% |
| 6+ | 1812 | **32.5%** | 27.7% | 15.8% |

**Read-out.** Retrieval difficulty rises monotonically with requirement count (H1-t2p:
14.6%→26.2% from 2 to 6+ requirements; H2-t2p: 23.3%→32.5%) — more constraints shrink the valid
answer set and stress the Text-to-Pandas query generator. **At every complexity level H1-t2p
retrieves better than H2-t2p** (the Correct stage's recovery, §2). Low-correctness for H1-t2p is
roughly flat with nreq (14.0% at 2 req, ~20–22% at 3–5, 19.7% at 6+) and below H2-t2p's
24–29% at every level from 3 requirements up: once retrieval succeeds, adding constraints does
not degrade the judged answer. The nreq=1 low-correctness (83.3%, n=6) is a small-sample fluke.

### 3b. By requirement type (H1-t2p; a record can carry several types)

| Requirement type | n | ret MISS | low-corr | hard-viol | allergen-viol |
|---|---|---|---|---|---|
| allergen | 3312 | 23.2% | 20.3% | 25.2% | 0.09% |
| **ingredient** | 2961 | **33.3%** | **24.8%** | 22.6% | 0.00% |
| tag | 4041 | 21.7% | 19.4% | 25.6% | 0.08% |
| nutrient_bound | 3543 | 24.1% | 21.0% | 25.5% | 0.09% |
| health | 4938 | 22.8% | 20.4% | 24.9% | 0.06% |

**Ingredient requirements are the hardest to retrieve for** — 33.3% retrieval MISS, well above
the ~22–24% of the other types. Free-text ingredient likes/dislikes (e.g. `"low - fat butter"`,
`"cilantro chutney"`) must be matched against noisy recipe ingredient lists, and the
Text-to-Pandas layer mis-translates them more often than it mishandles tags or numeric nutrient
bounds. This is the single most productive query type to target for retrieval improvement.
Ingredient queries are also the worst on judged correctness (24.8% low-correctness vs
19–21% for the other types). Post-fix, with the 421 empty-plan explanations replaced, low-correctness reads a uniform ~12–13%,
which hides that spread. Allergen violation is ≤ 0.09% on every type; the post-fix 0.00% results from the post-evaluation filter (§4).

### 3c. Allergen-requirement subset (safety-critical), all configs

3,312 of 4,938 item-runs carry an `allergen` requirement.

| Config | n (allergen) | ret MISS | low-corr | hard-viol | **allergen-viol** |
|---|---|---|---|---|---|
| H2-dense | 3312 | 97.6% | 78.2% | 82.0% | **1.18%** |
| H2-t2p | 3312 | 29.0% | 27.3% | 15.3% | **0.09%** |
| H1-dense | 3312 | 98.0% | 75.6% | 76.4% | **0.83%** |
| **H1-t2p** | 3312 | 23.2% | 20.3% | 25.2% | **0.09%** |

On this subset the dense configs violate an allergen requirement ~9–13× as often (1.18% /
0.83%) as the t2p configs (0.09%), because their near-random candidates leak disliked
ingredients. The published post-fix 0.00% on every config results from a filter applied to stored outputs after evaluation
(commit `74176e8`, 72 records) and holds by construction, because filter and metric are the same function; it does not
measure the evaluated system.

---

## 4. Allergen errors — deep-dive (remediated after evaluation)

**Evaluated system: allergen violation is rare but not zero.** Pre-fix means are **0.000310 /
0.001969 / 0.003427 / 0.000608** for H1-t2p / H1-dense / H2-dense / H2-t2p (record-pooled;
details below).

**Remediation.** The deterministic allergen post-filter (`constraints.enforce_allergen_safety`,
added to the code in commit 54a62ae) drops any final-plan slot whose recipe's ingredient list
intersects the persona's disliked/allergen set. Commit `74176e8` applied it to the 72 stored
outputs with a violation (`scripts/apply_safety_fixes.py`; no pipeline re-run, so the explanation
text of filtered records was not regenerated) and re-scored them. The published post-fix
`aggregate.json` allergen_violation is 0.000 on all four configs. That value results from a filter applied to stored outputs after evaluation and holds
by construction, because filter and metric are the same function; it is a property of the
remediated records, not a safety measurement of the evaluated system. Because a
blacklist-violating recipe can never be gold, the filter does not change hit@10.

**(Pre-fix) rare but categorically the worst failure.** Pre-fix aggregate rates with 95%
bootstrap CIs were: H2-dense 0.0034 [0.0019, 0.0052]; H1-dense 0.0021 [0.0011, 0.0032]; H2-t2p
0.0006 [0.0000, 0.0018]; H1-t2p 0.0003 [0.0000, 0.0009]. Pre-fix raw item-run counts with
`allergen_violation > 0`:

| Config | # item-runs w/ allergen viol (pre-fix) | distinct questions | post-fix |
|---|---|---|---|
| H2-dense | 39 | 25 (runs 1–3: 15/12/12) | **0** |
| H1-dense | 27 | 20 | **0** |
| H2-t2p | **3** | **1** (q_01495, all 3 runs) | **0** |
| H1-t2p | **3** | **1** (q_01495, all 3 runs) | **0** |

**(Pre-fix) concentration on t2p: a single persona.** For *both* t2p configs, every allergen
violation was **q_01495** across all 3 runs. That question dislikes `green onions`, likes tag
`lobster`, and the valid answer set has size 1; the pipeline served
**`lindas-lobster-rangoon-205992`**, which contains green onions. This was a deterministic,
reproducible single-persona failure, not diffuse noise — and it is exactly the case the new
deterministic filter targets. **In the post-fix records the q_01495 case is filtered**: the offending lobster-rangoon
slot is dropped, so all 3 runs report 0 allergen violations (the filter behaviour is covered by
the commit's test `test_safety_fixes.py`, which includes a synthetic reproduction of the q_01495
case).

**(Pre-fix) dense configs — diffuse and retrieval-driven.** The 39 (H2-dense) / 27 (H1-dense)
violations spread across many distinct personas (e.g. q_00088 shiitake, q_00255 Limburger cheese,
q_00281 Amarula liqueur, q_00522 granola, q_01040 turkey sausage). Because the dense retriever
returns near-random candidates, disliked ingredients leaked through. Most were av=0.25–0.5 (one
of several plan slots offending).

**Allergen posture (evaluated system).** Allergen violations are rare on every config (≤ 0.0034)
but retrieval-driven: an order of magnitude more frequent on dense than on t2p, with complete
failures included (`foodrag_h2` has three records at 1.0, `foodrag_h2_t2p` three). The
post-evaluation filter would veto these slots deterministically; its effect on stored outputs is
reported as a remediation, not as a measured result.

---

## 5. Judge reliability / disagreement (Judge A Gemma vs Judge B DeepSeek)

Per judged metric, across defined records: **MAD** = mean absolute A−B difference;
**disagree≥0.5** = fraction with |A−B| ≥ 0.5 (a full grade apart on the {0, 0.25, 0.5, 0.75, 1}
scale); **B>A** / **A>B** = directional bias.

| Config | metric | n | MAD | disagree≥0.5 | mean A | mean B | B>A | A>B |
|---|---|---|---|---|---|---|---|---|
| H2-dense | correctness | 4931 | 0.097 | 2.6% | 0.248 | 0.233 | 15.2% | 20.4% |
| H2-dense | faithfulness | 4925 | 0.154 | 15.4% | 0.232 | 0.106 | 1.4% | 14.0% |
| H2-dense | **relevancy** | 4934 | **0.233** | 23.3% | 0.307 | 0.096 | 1.1% | 22.3% |
| H2-t2p | correctness | 4920 | 0.140 | 8.5% | 0.586 | 0.605 | 24.7% | 19.9% |
| H2-t2p | faithfulness | 4930 | 0.100 | 10.0% | 0.892 | 0.986 | 9.7% | 0.4% |
| H2-t2p | **relevancy** | 4930 | **0.351** | 35.1% | 0.901 | 0.560 | 0.5% | 34.6% |
| H1-dense | correctness | 4855 | 0.092 | 2.8% | 0.285 | 0.309 | 21.5% | 12.0% |
| H1-dense | faithfulness | 4850 | 0.016 | 1.6% | 0.989 | 0.976 | 0.1% | 1.5% |
| H1-dense | **relevancy** | 4859 | **0.258** | 25.8% | 0.453 | 0.220 | 1.3% | 24.6% |
| H1-dense | expl-quality | 4838 | 0.101 | 2.4% | 0.573 | 0.626 | 28.7% | 9.2% |
| **H1-t2p** | correctness | 4788 | 0.155 | 13.1% | 0.687 | 0.745 | 31.5% | 13.1% |
| **H1-t2p** | faithfulness | 4798 | 0.010 | 1.0% | 0.920 | 0.913 | 0.1% | 0.8% |
| **H1-t2p** | **relevancy** | 4809 | **0.142** | 14.2% | 0.787 | 0.701 | 2.8% | 11.4% |
| **H1-t2p** | expl-quality | 4777 | 0.158 | 12.7% | 0.715 | 0.838 | 41.3% | 5.7% |
| H1-t2p post-fix | correctness | 4794 | 0.214 | 18.9% | 0.765 | 0.740 | 28.2% | 20.0% |
| H1-t2p post-fix | faithfulness | 4799 | 0.005 | 0.5% | 0.997 | 0.994 | 0.1% | 0.4% |
| H1-t2p post-fix | **relevancy** | 4813 | **0.140** | 14.0% | 0.869 | 0.785 | 2.8% | 11.2% |
| H1-t2p post-fix | expl-quality | 4791 | 0.164 | 16.2% | 0.750 | 0.792 | 33.6% | 14.1% |

**Reliability read-out.**
- **`relevancy` is the least reliable metric** — highest MAD and disagreement in three of the four
  configs (up to MAD 0.351 / 35.1% disagreement on H2-t2p; on H1-t2p correctness and
  explanation-quality disagree more), and the direction is consistent: **Judge A
  scores relevancy much higher than Judge B** (A>B ≫ B>A everywhere). Gemma and DeepSeek
  interpret "relevancy" differently; treat relevancy as a soft signal, not a headline number.
- **`faithfulness` is the most reliable** — MAD 0.010–0.016 on both H1 configs (near-perfect
  agreement). Faithfulness is a concrete "does the text match the plan" check both judges agree
  on.
- **The dual-judge design still catches a judge-leniency gap on `explanation-quality`.**
  **Judge B (DeepSeek) scores explanation-quality higher** than cross-lineage Judge A: H1-t2p
  expl-quality meanB 0.838 > meanA 0.715 (**B>A 41.3% vs A>B 5.7%**), and H1-dense meanB 0.626 >
  meanA 0.573 (B>A 28.7% vs A>B 9.2%). Neither judge shares the Llama generator's family, so this
  is a leniency difference between judges rather than self-preference — reporting Judge A (the *lower*, cross-lineage estimate) as
  headline is the conservative and correct choice. **The same leniency shows on correctness:**
  H1-t2p correctness meanB 0.745 > meanA 0.687 (B>A 31.5% vs A>B 13.1%). (Post-fix, the replaced empty-plan explanations raise Judge A to 0.765 while Judge B scores them lower, so Judge A sits slightly above Judge B's 0.740 — the two judges disagree in direction on those records.)
  Faithfulness/relevancy do **not** show B-inflation (relevancy is if anything A-inflated), so
  the residual bias is specific to the open-ended explanation-quality judgment.

---

## 6. Explanation errors (H1 configs)

From `claim_grounding_rate` (cgr) and `unsupported_claim_rate` (ucr = 1 − cgr) on the NL
explanation layer:

| Config | n | mean ucr | mean cgr | ucr p50 | ucr p75 | ucr p90 | frac ucr==1.0 | frac ucr>0.5 |
|---|---|---|---|---|---|---|---|---|
| **H1-t2p** | 4605 | 0.346 | 0.654 | 0.33 | 0.50 | 1.00 | 15.4% | 21.2% |
| H1-dense | 4644 | 0.447 | 0.553 | 0.50 | 0.67 | 1.00 | 15.4% | 32.0% |
| H1-t2p post-fix (not the evaluated system) | 4200 | 0.283 | 0.717 | 0.25 | 0.50 | 0.67 | 7.3% | 13.6% |

**Read-out.** In the evaluated system ~65% of H1-t2p explanation claims are grounded (vs ~55%
on dense), and 15.4% of its explanations are fully ungrounded (ucr=1.0). The dominant failure is
the empty plan: 405 of those records narrate a plan that was never produced. The **empty-plan
explanation gate** (`ExplanationTool.explain()` returns a fixed safe message with no LLM call
when the recommended plan is empty — commit 54a62ae), applied to the stored records after
evaluation, gives the post-fix row: mean claim-grounding **0.654 → 0.717** (post-fix), fully-ungrounded explanations (ucr=1.0) **15.4% → 7.3%** (post-fix), and
majority-ungrounded (ucr>0.5) **21.2% → 13.6%** (post-fix). The p90 of ucr drops from 1.00 to 0.67 —
the fully-hallucinated tail shrinks. These shifts are a denominator effect: the gated message
carries no claims, so the 405 gated records (all ucr=1.0 pre-fix) become undefined and leave the
4,605 → 4,200 denominator; the remaining 4,200 records are essentially unchanged. Better
retrieval yields better-grounded prose (H1-t2p ~65% vs H1-dense ~55%)
(consistent with the §8 grounding contrast in RESULTS_SUMMARY §3).

**Original failure (now gated) — patterns observed in the pre-fix high-ucr explanations:**
1. **Hallucinated recipes on empty plans (gated in code; replaced in the stored records after evaluation)** — e.g. q_00000 (plan all `None`) previously
   narrated *"Scrambled Eggs with Whole Wheat Toast … FSA score of 0.2 … total fat 10g … calorie
   count 200"*, all fabricated (`hit=0`, ucr=1.0); q_00009 similarly invented *"Agar Pudding with
   Fresh Fruit"*. **These empty-plan records now receive the fixed "no recommendation" message
   instead of hallucinated prose** (and drop out of the ucr denominator), which is why post-fix ucr=1.0 halves and post-fix faithfulness on H1-t2p reads 0.997 (§1, §5).
2. **Fabricated precise nutrition figures** — specific FSA scores / gram values / kcal that were
   ungroundable because no recipe backed them; the gate removes them on empty plans.
3. **Admitted constraint violations narrated as acceptable (slot filtered after evaluation)** — the q_01495 explanation
   openly said the plan *"includes recipes that contain ingredients … you dislike, such as green
   onions"*; the deterministic allergen filter (§4) now drops that slot, so the violation no longer
   occurs — but the fix was applied post hoc without regenerating the explanation, so the stored
   post-fix q_01495 explanations still contain this narration.

The residual ~28% ungrounded claims are on *non-empty* plans where the Enhance stage
over-describes a real plan; those are not caught by the emptiness gate and still need
claim-level grounding. The intrinsic KG-path explanation layer (`path_validity` = 1.000
everywhere) remains the reliable channel.

---

## 7. Case Browser — concrete examples

**Case A — ALLERGEN violation (the entire t2p allergen liability; filtered after evaluation).**
`q_01495`, H1-t2p, all 3 runs. Persona dislikes **green onions**, likes tag *lobster*, answer
set size 1. Requirement types: allergen, tag, nutrient_bound, health.
Pre-fix plan → snack/dinner = **`lindas-lobster-rangoon-205992`** (contains green onions),
lunch = `seafood-tea-sandwich-recipe-424001`. Pre-fix metrics: `hit=1.0`,
`allergen_violation=0.5`, `hard_pc=0.875`, correctnessA=0.25, relevancyA=0.0, ucr=0.0.
**Smoking gun** — the pre-fix explanation conceded the violation verbatim:
> *"Although the plan includes recipes that contain ingredients similar to or are the same as
> those you dislike, such as green onions in 'Linda's Lobster Rangoon', the meal plan is
> designed to be both enjoyable and nutritionally balanced …"*

The retriever *found* the gold (hit=1) but the constraint filter did not veto the disliked
ingredient — a missing hard post-filter, not a retrieval failure. **The deterministic allergen
post-filter (commit 54a62ae) now drops the lobster-rangoon slot**, so all 3 fixed records report
`allergen_violation=0.0` (a synthetic reproduction of this case is covered by `test_safety_fixes.py`).
The stored explanation text was not regenerated and still contains the concession above. hit@10 is
unaffected because the offending recipe was not the gold answer.

**Case B — Pipeline DROPPED gold (bucket (b), §2).**
`q_00003`, H1-t2p. Requirements: ingredient (likes *garlic sausage*), tag dislikes
(*meatloaf*, *crusts-pastry-dough-2*), fsa/sugar/fiber bounds. The retriever returned the gold,
but the final plan collapsed to a single slot (breakfast = `baked-mexican-rice-62389`;
snack/lunch/dinner = `None`), so `hit=0`. correctnessA=0.5, `hard_pc=0.67` (hard-constraint
violation). The Correct/Enhance stage discarded the valid gold candidate and under-filled the
plan — the 7% "dropped gold" bucket and the source of H1-t2p's *rise* in hard-violation vs H2.

**Case C — Retrieval MISS with a confident (wrong) plan.**
`q_00015`, H1-t2p, 7 requirements (dislikes *turkey slices*, tags *novelty/oatmeal*, satfat &
cal-from-fat bounds). Retriever missed the gold; the pipeline still produced a full 4-slot
dessert-heavy plan (`caribbean-coco-almond-bars`, `weetbix-deluxe-caramel-slice`,
`chocolate-pecan-cakes`, `greek-cornbread`). `hit=0`, correctnessA=0.0 (Judge A) but faithA=1.0
and relevancyA=1.0 — the explanation faithfully describes a plan that simply isn't the gold
answer. Retrieval-limited (bucket (a)); no pipeline fix applies.

**Case D — Fully hallucinated explanation on an EMPTY plan (ucr=1.0; gated in code, replaced after evaluation).**
`q_00000`, H1-t2p. Plan is entirely `None` (empty). Pre-fix, the explanation narrated a concrete,
invented recipe with fabricated nutrition:
> *"The 'Scrambled Eggs with Whole Wheat Toast' recipe features low-fat butter … FSA score of
> 0.2 … total fat content is 10g … calorie count is 200 …"*

Pre-fix metrics: `hit=0`, correctnessA=0.0, faithfulnessA=0.0, ucr=1.0, cgr=0.0. **The empty-plan
explanation gate (commit 54a62ae) now returns a fixed "no recommendation" message on empty plans
(no LLM call), so the Enhance stage can no longer hallucinate here** — this is the mechanism
behind the post-fix ucr=1.0 share of 7.3% (vs 15.4%; gated records leave the ucr denominator) and the post-fix faithfulness of 0.997 (vs 0.922) (§6, §1). (q_00009 "Agar
Pudding with Fresh Fruit" was an identical pattern, likewise gated.)

**Case E — Judge disagreement on the gated empty-plan record.**
`q_00000` (same empty-plan record as Case D). *Pre-fix*, the explanation was a fluent
hallucination: Judge A (Gemma) scored explanation_quality = 0.0 (correctly penalising it), while
Judge B (DeepSeek) scored 1.0 — rewarding fluency over grounding, the §5 Judge-B leniency
finding. *Post-fix*, the record now carries the honest **"No meal plan could be recommended that
satisfies the given requirements"** message: Judge A now scores it correctnessA=1.0,
faithfulnessA=1.0, explanation_quality 0.5–1.0 (it is a correct, faithful refusal), whereas Judge
B still scores explanation_quality 0.0 — DeepSeek penalises the terse refusal it cannot reward for
fluency. The disagreement direction *flips* on this record, but Judge A is again the more sensible
of the two: it credits the honest refusal that Judge B marks down. This is why the empty-plan gate
lifts Judge A's correctness/faithfulness on H1-t2p (§1, §5).

---

## 8. Takeaways

**Dominant error mode per config**
- **H2-dense / H1-dense** — *retrieval*. ~97% retrieval MISS drags every downstream metric
  (~80% hard-violation, ~76% low-correctness) and most of the allergen violations (0.55–0.79% of item-runs). The agentic layer
  cannot help because there is nothing valid to correct (bucket (c) ≈ 1%). These configs are
  retrieval-bound, full stop.
- **H2-t2p** — *pipeline drops gold*. Retrieval is strong (29% miss) but the recommend-only
  stage discards the gold from a valid set in **14.5%** of items — its single largest avoidable
  error.
- **H1-t2p** (best config) — a mix of an **irreducible ~14% retrieval floor** + **~7%
  pipeline-drop** + a **narrow hard-constraint over-filtering** (hard-violation *rose* to 24.9%
  vs H2's 14.6%, from the agent under-filling plans) + **hallucinated explanations on empty
  plans** (UNFAITHFUL 8.0%, ucr=1.0 on 15.4%). One allergen leak (q_01495, Case A) recurs in all
  three runs.

**Remediations and their effect on the stored records** (code in commit 54a62ae; applied after
evaluation by commit `74176e8` to 493 stored records, which were re-scored — these are post-fix
values, not measurements of the evaluated system)
1. **Deterministic allergen post-filter** (`constraints.enforce_allergen_safety`, applied to the
   final plan in both assemblers) — vetoes any plan slot whose ingredients intersect the
   persona's disliked/allergen set. **Effect on stored records:** post-fix allergen violation 0.000 on all four
   configs by construction (pre-fix 0.003427 / 0.001969 / 0.000608 / 0.000310), including the q_01495 case (Case A) and the 66
   diffuse dense violations (39 H2-dense + 27 H1-dense; §4). hit@10 unaffected (a blacklist-violating recipe is never gold).
2. **Empty-plan explanation gate** (`ExplanationTool.explain()` returns a fixed "no
   recommendation" message, no LLM call, when the plan is empty). **Effect on the 421 replaced H1-t2p records (pre-fix → post-fix):**
   fully-ungrounded explanations (ucr=1.0) 15.4%→7.3% and claim-grounding 0.654→0.717 post-fix (both via
   gated records leaving the ucr denominator);
   post-fix faithfulness 0.922→0.997, post-fix correctness 0.689→0.765, post-fix UNFAITHFUL 8.0%→0.3% (§1, §5,
   §6). The recommendations themselves did not change; Cases D and E are gated in the code from 54a62ae on.
3. **Distinct-slot guard** (`constraints.enforce_distinct_slots`, applied after the allergen
   filter in both assemblers) + `NonRepeatingReplayRetriever` for the full-day t2p replay path.
   Fixes a **degenerate-plan** failure mode — the same recipe placed in multiple slots — surfaced
   by the full-day benchmark: an unguarded full-day run put one dish in all four slots for up to
   **100%** of `h2_t2p` plans (plain replay returns the same pool every slot) and 17.7% of
   `h1_t2p`. **Measured effect:** **0.000%** of full-day plans put one dish in all four slots
   (4,938 records per config). However, the persisted full-day records still repeat a recipe in
   2–3 slots in 23.4% (`h2`), 19.1% (`h1_dense`) and 19.3% (`h1_t2p`) of plans (0% `h2_t2p`), and
   15.6% of the partial-day H1-t2p plans repeat a recipe too, so the stored results do not yet
   reflect the guard's no-recipe-in-two-slots guarantee. See the full-day benchmark
   (RESULTS_REPORT §12).

**Remaining highest-leverage targets** (not yet done)
1. **Retrieval is still the highest-leverage target overall** — moving dense→t2p is a ~70-point
   swing on nearly every failure. Within t2p, **ingredient requirements (33% miss)** are the
   worst-served query type and the best place to improve the Text-to-Pandas query generator.
2. **The Correct stage's ~7% gold-dropping** (bucket (b), §2) is the remaining agentic lever.
3. **Claim-level grounding on *non-empty* plans** — the residual ~28% unsupported claims are on
   real plans the Enhance stage over-describes; the emptiness gate does not touch these.

**Allergen posture (evaluated system).** Allergen violations are rare (≤ 0.0034) but
retrieval-driven: dense's liability (H2-dense 0.003427: 39 item-runs across 25 personas) is an
order of magnitude above t2p's, whose only violation is the self-narrated q_01495 case (Case A).
The post-evaluation filter sets the stored records to 0.000 by construction; that is a
remediation, not a safety measurement.

**Judge caveat for downstream readers.** Report **Judge A (Gemma, cross-lineage)** as headline:
Judge B scores explanation-quality more leniently (+0.12 on H1-t2p, +0.05 on H1-dense) (§5,
Case E) and correctness too (+0.06 on H1-t2p). Treat **relevancy** as low-reliability (up to 35%
cross-judge disagreement); **faithfulness** is the most trustworthy judged metric (MAD ≈
0.010–0.016).

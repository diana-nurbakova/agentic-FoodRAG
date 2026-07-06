# FoodRAG Evaluation — Error Analysis

Companion to `eval_results/RESULTS_SUMMARY.md` and `docs/RESULTS_REPORT.md`. Where those
report *how well* the system performs, this document dissects *where and why it fails*. All
numbers are computed directly from the per-question §13.1 records in
`eval_results/json/<config>_per_question.json` (1,646 tight-stratum questions × 3 runs =
4,938 item-runs per FoodRAG config), the retriever records
(`ret_text2pandas@llama70b`: 5 runs; `ret_dense`: 1 run), and the gold QA pickle
`data/qa/main_10k_tight/recipe_question_answer_set.pkl`.

Configs (focus): **H2-dense** (`foodrag_h2`, Recommend-only + dense retriever), **H2-t2p**
(`foodrag_h2_t2p`, Recommend-only + Text-to-Pandas), **H1-dense** (`foodrag_h1_dense`,
agentic Recommend+Correct+Enhance + dense), **H1-t2p** (`foodrag_h1_t2p`, agentic +
Text-to-Pandas). Headline judge is **Judge A = Gemma-3-27B** (cross-lineage); **Judge B =
DeepSeek-V3** is the same-family control.

**NaN handling.** Retrieval/safety/constraint metrics are computed on every record. Judge and
grounding metrics are `None` on a minority of records (judge parse failures, or no explanation
emitted); rates below use the **defined-only denominator** for each metric and report that
denominator so the reader can see how much data backs each figure.

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

| Failure type | H2-dense | H2-t2p | H1-dense | **H1-t2p** |
|---|---|---|---|---|
| Retrieval MISS | **96.8%** | 29.1% | **97.3%** | 22.8% |
| EMPTY plan | 0.9% | 8.6% | 1.6% | 10.7% |
| HARD-constraint violation | 81.5% | 14.6% | 77.4% | 24.9% |
| ALLERGEN violation (safety) | 0.8% | 0.06% | 0.6% | 0.06% |
| LOW correctness (JudgeA<0.5) | 76.6% | 26.8% | 75.2% | 20.4% |
| UNFAITHFUL (JudgeA==0) | 76.8% | 10.8% | 1.1% | 8.0% |
| UNSUPPORTED-claim-heavy (>0.5) | n/a | n/a | 32.0% | 21.2% |
| STRUCTURAL-invalid | 79.3% | 81.4% | 82.8% | 83.1% |

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
  and EMPTY plan rises 8.6%→10.7%. The agent trades a small amount of constraint-safety and
  plan-completeness for better recall and correctness (see §2 for why).
- **STRUCTURAL-invalid ~80% across *all four* configs is a metric artifact, not a system
  failure.** `structural_validity` demands a full 4-slot day; the deployed meal-planner is
  time-driven and returns partial-day plans (mean 2.32 slots — see RESULTS_SUMMARY §5), so a
  well-formed 2-meal plan scores 0. Confirmed: among H1-t2p `sv==0` records, 3,655/4,020 *have*
  a non-empty plan, and their mean `meal_slot_completeness` is 0.39. Treat this row as
  "not-a-full-day", not "malformed". It is the one row in this table that should **not** drive
  any engineering decision.
- **UNFAITHFUL collapses from 76.8% (H2-dense) to 1.1% (H1-dense).** H2-dense's NL layer (a raw
  recommend-only description) frequently contradicts its near-empty plan; the H1 Enhance stage
  writes explanations tied to the actual plan, so faithfulness is high even when the plan is
  poor (H1-dense faithfulness 0.989 = cautiously describing a tiny plan).

---

## 2. Error-source decomposition — retrieval vs pipeline

The t2p FoodRAG configs **replay** the `ret_text2pandas@llama70b` retrieval, so we can localise
each error. For every question index *i* we compare the retriever's `hit@10` (mean over its
runs) against the FoodRAG config's `hit@10`, and bucket:

- **(a) Retriever MISS** (retriever hit=0) — *retrieval-limited*: the gold was never available.
- **(b) Retriever HIT, pipeline MISS** (retriever=1, foodrag=0) — *pipeline DROPPED the gold*.
- **(c) Both hit** — retrieval succeeded and the pipeline preserved it.

| Comparison | (a) retriever MISS | (b) pipeline DROPPED | (c) both hit |
|---|---|---|---|
| **H2-t2p** vs ret_t2p | 14.2% (234) | **14.5% (239)** | 71.3% (1173) |
| **H1-t2p** vs ret_t2p | 14.2% (234) | **7.0% (116)** | 78.6% (1293) |
| H2-dense vs ret_dense | 94.5% (1555) | 4.3% (71) | 1.2% (20) |
| H1-dense vs ret_dense | 94.5% (1555) | 4.7% (77) | 0.9% (14) |
| n = 1,646 per row (H1-t2p: 3 undefined) |

**Interpretation.**
- **On t2p, the agentic pipeline halves the "dropped gold" error: bucket (b) falls from 14.5%
  (H2-t2p) to 7.0% (H1-t2p).** This is the clearest quantified win for the agent: given the
  *same* retriever, recommend-only H2 discards the gold recipe from a valid candidate set in
  1-in-7 cases, while H1's Correct/Enhance loop recovers roughly half of those (both-hit rises
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

**Localisation.** On the strong retriever, FoodRAG error decomposes as ≈ **14% retrieval-bound +
7% pipeline-drop** (H1-t2p). The biggest remaining pipeline lever is the ~7% gold-dropping in
Correct/Enhance; the biggest retrieval lever is the ~14% floor.

---

## 3. Error by query difficulty

### 3a. By number of requirements (`num_requirements`)

Failure rates for **H1-t2p** (and H2-t2p for contrast). Buckets 1 and 2 are tiny (n=6, n=157)
so read them cautiously.

**H1-t2p**

| nreq | n | ret MISS | low-corr | hard-viol |
|---|---|---|---|---|
| 1 | 6 | 0.0% | 83.3% | 0.0% |
| 2 | 157 | 14.6% | 14.0% | 21.0% |
| 3 | 547 | 17.6% | 20.0% | 22.7% |
| 4 | 1110 | 21.1% | 21.1% | 25.9% |
| 5 | 1246 | 22.9% | 21.7% | 23.7% |
| 6+ | 1773 | **26.2%** | 19.7% | 26.3% |

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
retrieves better than H2-t2p** (the Correct stage's recovery, §2). Low-correctness is roughly
flat in nreq (~20% for H1-t2p) — once retrieval succeeds, adding constraints does not further
degrade the judged answer. The nreq=1 low-correctness (83.3%, n=6) is a small-sample fluke.

### 3b. By requirement type (H1-t2p; a record can carry several types)

| Requirement type | n | ret MISS | low-corr | hard-viol | allergen-viol |
|---|---|---|---|---|---|
| allergen | 3312 | 23.2% | 20.3% | 25.2% | 0.09% |
| **ingredient** | 2961 | **33.3%** | **24.8%** | 22.6% | 0.00% |
| tag | 4041 | 21.7% | 19.4% | 25.6% | 0.08% |
| nutrient_bound | 3543 | 24.1% | 21.0% | 25.5% | 0.09% |
| health | 4938 | 22.8% | 20.4% | 24.9% | 0.06% |

**Ingredient requirements are the hardest** — 33.3% retrieval MISS and the highest
low-correctness (24.8%), well above the ~22–24% of the other types. Free-text ingredient
likes/dislikes (e.g. `"low - fat butter"`, `"cilantro chutney"`) must be matched against noisy
recipe ingredient lists, and the Text-to-Pandas layer mis-translates them more often than it
mishandles tags or numeric nutrient bounds. This is the single most productive query type to
target for retrieval improvement.

### 3c. Allergen-requirement subset (safety-critical), all configs

3,312 of 4,938 item-runs carry an `allergen` requirement.

| Config | n (allergen) | ret MISS | low-corr | hard-viol | **allergen-viol** |
|---|---|---|---|---|---|
| H2-dense | 3312 | 97.6% | 78.2% | 82.0% | **1.18%** |
| H2-t2p | 3312 | 29.0% | 27.3% | 15.3% | 0.09% |
| H1-dense | 3312 | 98.0% | 75.6% | 76.4% | 0.83% |
| **H1-t2p** | 3312 | 23.2% | 20.3% | 25.2% | **0.09%** |

Allergen *violation* rates on this subset mirror §1: the dense configs are ~10–13× worse on
safety (1.18% / 0.83%) than the t2p configs (0.09%). Safety and retrieval quality are coupled —
a retriever that returns valid candidates is also the one that avoids serving allergens.

---

## 4. Safety errors — deep-dive

Allergen violations are **rare but categorically the worst** failure. Aggregate rates with 95%
bootstrap CIs (from `aggregate.json`): H2-dense 0.0034 [0.0019, 0.0052]; H1-dense 0.0021
[0.0011, 0.0032]; H2-t2p 0.0006 [0.0000, 0.0018]; **H1-t2p 0.0003 [0.0000, 0.0009]**. Raw
item-run counts with `allergen_violation > 0`:

| Config | # item-runs w/ allergen viol | distinct questions |
|---|---|---|
| H2-dense | 39 | 39 (all run-1) |
| H1-dense | 27 | ~25 |
| H2-t2p | **3** | **1** (q_01495, all 3 runs) |
| H1-t2p | **3** | **1** (q_01495, all 3 runs) |

**Concentration on t2p: a single persona.** For *both* t2p configs, every allergen violation is
**q_01495** across all 3 runs. That question dislikes `green onions`, likes tag `lobster`, and
the valid answer set has size 1; the pipeline serves **`lindas-lobster-rangoon-205992`**, which
contains green onions. This is a **deterministic, reproducible single-persona failure**, not
diffuse noise — the entire t2p safety liability is one recipe/persona pair and is directly
patchable (see §7 for the smoking-gun explanation text, which openly admits the violation).

**Dense configs — diffuse and retrieval-driven.** The 39 (H2-dense) / 27 (H1-dense) violations
spread across many distinct personas (e.g. q_00088 shiitake, q_00255 Limburger cheese, q_00281
Amarula liqueur, q_00522 granola, q_01040 turkey sausage). Because the dense retriever returns
near-random candidates, the allergen filter has little valid material to work with, and
disliked ingredients leak through. Most are av=0.25–0.5 (one of several plan slots offends). No
single requirement type dominates — all carry the `allergen` type by construction, and they
span ingredient/tag/nutrient_bound mixes.

**Safety posture.** H1-t2p is the safest config (0.0003, one persona, CI includes 0). No config
is perfectly safe, and the *dense* configs carry a real, retrieval-driven allergen liability an
order of magnitude larger than t2p. **The fix for dense is retrieval; the fix for t2p is a hard
post-filter that vetoes any plan recipe whose ingredient list intersects the persona's
disliked/allergen set** — which would have caught q_01495 outright.

---

## 5. Judge reliability / disagreement (Judge A Gemma vs Judge B DeepSeek)

Per judged metric, across defined records: **MAD** = mean absolute A−B difference;
**disagree≥0.5** = fraction with |A−B| ≥ 0.5 (a full grade apart on the {0, 0.25, 0.5, 0.75, 1}
scale); **B>A** / **A>B** = directional bias.

| Config | metric | n | MAD | disagree≥0.5 | mean A | mean B | B>A | A>B |
|---|---|---|---|---|---|---|---|---|
| H2-dense | correctness | 4931 | 0.097 | 2.6% | 0.248 | 0.233 | 15.2% | 20.4% |
| H2-dense | faithfulness | 4925 | 0.154 | 15.4% | 0.232 | 0.106 | 1.4% | 14.0% |
| H2-dense | **relevancy** | 4934 | **0.233** | **23.3%** | 0.307 | 0.096 | 1.1% | 22.3% |
| H2-t2p | correctness | 4920 | 0.140 | 8.5% | 0.586 | 0.605 | 24.7% | 19.9% |
| H2-t2p | faithfulness | 4930 | 0.100 | 10.0% | 0.892 | 0.986 | 9.7% | 0.4% |
| H2-t2p | **relevancy** | 4930 | **0.351** | **35.1%** | 0.901 | 0.560 | 0.5% | 34.6% |
| H1-dense | correctness | 4855 | 0.092 | 2.8% | 0.285 | 0.309 | 21.5% | 12.0% |
| H1-dense | faithfulness | 4850 | 0.016 | 1.6% | 0.989 | 0.976 | 0.1% | 1.5% |
| H1-dense | **relevancy** | 4859 | **0.258** | **25.8%** | 0.453 | 0.220 | 1.3% | 24.6% |
| H1-dense | expl-quality | 4838 | 0.101 | 2.4% | 0.573 | 0.626 | 28.7% | 9.2% |
| **H1-t2p** | correctness | 4788 | 0.155 | 13.1% | 0.687 | 0.745 | 31.5% | 13.1% |
| **H1-t2p** | faithfulness | 4798 | 0.010 | 1.0% | 0.920 | 0.913 | 0.1% | 0.8% |
| **H1-t2p** | **relevancy** | 4809 | **0.142** | 14.2% | 0.787 | 0.701 | 2.8% | 11.4% |
| **H1-t2p** | expl-quality | 4777 | 0.158 | 12.7% | 0.715 | 0.838 | **41.3%** | 5.7% |

**Reliability read-out.**
- **`relevancy` is the least reliable metric** — highest MAD and disagreement in every config
  (up to MAD 0.351 / 35.1% disagreement on H2-t2p), and the direction is consistent: **Judge A
  scores relevancy much higher than Judge B** (A>B ≫ B>A everywhere). Gemma and DeepSeek
  interpret "relevancy" differently; treat relevancy as a soft signal, not a headline number.
- **`faithfulness` is the most reliable** — MAD 0.010–0.016 on both H1 configs (near-perfect
  agreement). Faithfulness is a concrete "does the text match the plan" check both judges agree
  on.
- **The dual-judge design catches a same-family bias, but only on the generative metrics.** On
  **correctness** and **explanation-quality**, same-family **Judge B (DeepSeek) systematically
  scores higher** than cross-lineage Judge A: H1-t2p correctness meanB 0.745 > meanA 0.687
  (B>A 31.5% vs A>B 13.1%), and expl-quality meanB 0.838 > meanA 0.715 (**B>A 41.3% vs A>B
  5.7%**). This is exactly the self-preference inflation the cross-lineage headline judge was
  chosen to avoid — reporting Judge A (the *lower*, cross-lineage estimate) as headline is the
  conservative and correct choice. Faithfulness/relevancy do **not** show B-inflation (relevancy
  is if anything A-inflated), so the bias is specific to the open-ended quality judgments.

---

## 6. Explanation errors (H1 configs)

From `claim_grounding_rate` (cgr) and `unsupported_claim_rate` (ucr = 1 − cgr) on the NL
explanation layer:

| Config | n | mean ucr | mean cgr | ucr p50 | ucr p75 | ucr p90 | frac ucr==1.0 | frac ucr>0.5 |
|---|---|---|---|---|---|---|---|---|
| **H1-t2p** | 4605 | 0.346 | 0.654 | 0.33 | 0.50 | 1.00 | 15.4% | 21.2% |
| H1-dense | 4644 | 0.447 | 0.553 | 0.50 | 0.67 | 1.00 | 15.4% | 32.0% |

**Read-out.** On the stronger retriever ~65% of explanation claims are grounded (H1-t2p),
vs ~55% on dense — better retrieval yields better-grounded prose (consistent with
RESULTS_SUMMARY §8). But the distribution is bimodal: **15.4% of H1-t2p explanations are
*fully* ungrounded (ucr=1.0)**, and 21.2% are majority-ungrounded. These are concentrated on the
EMPTY-plan cases: when the plan is empty (all slots `None`), the Enhance stage still emits fluent
prose **describing recipes that were never selected**, inventing plausible names and nutrition
figures. See §7 for two verbatim examples.

**Failure patterns observed in high-ucr explanations:**
1. **Hallucinated recipes on empty plans** — e.g. q_00000 (plan all `None`) narrates
   *"Scrambled Eggs with Whole Wheat Toast … FSA score of 0.2 … total fat 10g … calorie count
   200"*, all fabricated (`hit=0`, `hard_pc=None`, ucr=1.0). q_00009 similarly invents
   *"Agar Pudding with Fresh Fruit"*.
2. **Fabricated precise nutrition figures** — specific FSA scores / gram values / kcal that are
   ungroundable because no recipe backs them.
3. **Admitted constraint violations narrated as acceptable** — the q_01495 explanation openly
   says the plan *"includes recipes that contain ingredients … you dislike, such as green
   onions"* yet frames the plan as suitable (§7).

The intrinsic KG-path explanation layer (`path_validity` = 1.000 everywhere) remains the
reliable channel; the free-text layer needs an emptiness/grounding guard.

---

## 7. Case Browser — concrete examples

**Case A — ALLERGEN violation (safety, the entire t2p liability).**
`q_01495`, H1-t2p, all 3 runs. Persona dislikes **green onions**, likes tag *lobster*, answer
set size 1. Requirement types: allergen, tag, nutrient_bound, health.
Plan → snack/dinner = **`lindas-lobster-rangoon-205992`** (contains green onions),
lunch = `seafood-tea-sandwich-recipe-424001`. Metrics: `hit=1.0`, `allergen_violation=0.5`,
`hard_pc=0.875`, correctnessA=0.25, relevancyA=0.0, ucr=0.0. **Smoking gun** — the explanation
concedes the violation verbatim:
> *"Although the plan includes recipes that contain ingredients similar to or are the same as
> those you dislike, such as green onions in 'Linda's Lobster Rangoon', the meal plan is
> designed to be both enjoyable and nutritionally balanced …"*

The retriever *found* the gold (hit=1) but the constraint filter did not veto the disliked
ingredient — a missing hard post-filter, not a retrieval failure. Deterministic across runs.

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

**Case D — Fully hallucinated explanation on an EMPTY plan (ucr=1.0).**
`q_00000`, H1-t2p. Plan is entirely `None` (empty), yet the explanation narrates a concrete,
invented recipe with fabricated nutrition:
> *"The 'Scrambled Eggs with Whole Wheat Toast' recipe features low-fat butter … FSA score of
> 0.2 … total fat content is 10g … calorie count is 200 …"*

`hit=0`, correctnessA=0.0, faithfulnessA=0.0, ucr=1.0, cgr=0.0. The Enhance stage should have
emitted a "no valid plan" message; instead it hallucinated. (q_00009 "Agar Pudding with Fresh
Fruit" is an identical pattern.)

**Case E — Judge disagreement (explanation_quality A≪B).**
`q_00000` (same empty-plan record as Case D). Judge A (Gemma) scores explanation_quality = 0.0
(correctly penalising the hallucination); Judge B (DeepSeek) scores 1.0 — rewarding the fluent
prose despite zero grounding. This single record embodies the §5 finding: the same-family judge
rewards fluency over faithfulness, and only the cross-lineage Judge A flags the hallucination.

---

## 8. Takeaways

**Dominant error mode per config**
- **H2-dense / H1-dense** — *retrieval*. ~97% retrieval MISS drags every downstream metric
  (~80% hard-violation, ~76% low-correctness, ~1% allergen). The agentic layer cannot help
  because there is nothing valid to correct (bucket (c) ≈ 1%). These configs are
  retrieval-bound, full stop.
- **H2-t2p** — *pipeline drops gold*. Retrieval is strong (29% miss) but the recommend-only
  stage discards the gold from a valid set in **14.5%** of items — its single largest avoidable
  error.
- **H1-t2p** (best config) — a mix of an **irreducible ~14% retrieval floor** + **~7%
  pipeline-drop** + a **narrow but real hard-constraint/allergen leak** (hard-violation actually
  *rose* to 24.9% vs H2's 14.6%, and the lone allergen persona q_01495 survives all runs).

**Where the biggest wins are**
1. **Retrieval is the highest-leverage target overall** — moving dense→t2p is a ~70-point
   swing on nearly every failure. Within t2p, **ingredient requirements (33% miss)** are the
   worst-served query type and the best place to improve the Text-to-Pandas query generator.
2. **A hard constraint/allergen post-filter is the highest-leverage *pipeline* fix.** H1-t2p's
   hard-violation regression (§1) and the entire t2p allergen liability (§4, one persona) would
   both be caught by vetoing any plan recipe whose ingredients intersect the persona's
   disliked/allergen set. This is cheap, deterministic, and safety-critical.
3. **An empty-plan / grounding guard on the Enhance stage** removes the 15.4% fully-hallucinated
   explanations (Case D) and would raise claim-grounding materially.
4. **The Correct stage's ~7% gold-dropping** (bucket (b)) is the remaining agentic lever once
   the above are in place.

**Safety posture.** Genuinely good on t2p (H1-t2p 0.0003, CI includes 0; the only violations are
one reproducible persona). Genuinely *concerning* on dense (H2-dense 0.0034 across 39 distinct
personas) — a retrieval-driven liability an order of magnitude larger. Do not deploy a dense
retriever for this task; and even on t2p, add the hard allergen post-filter — near-zero is not
zero, and the failing case (Case A) is one the system *knows about and narrates*.

**Judge caveat for downstream readers.** Report **Judge A (Gemma, cross-lineage)** as headline:
same-family Judge B inflates correctness (+0.06) and explanation-quality (+0.12) via
self-preference (§5, Case E). Treat **relevancy** as low-reliability (up to 35% cross-judge
disagreement); **faithfulness** is the most trustworthy judged metric (MAD ≈ 0.01).

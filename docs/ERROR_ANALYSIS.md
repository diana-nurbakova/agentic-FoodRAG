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
| HARD-constraint violation | 81.5% | 14.6% | 77.4% | 24.8% |
| ALLERGEN violation (safety) | **0.0%** | **0.0%** | **0.0%** | **0.0%** |
| LOW correctness (JudgeA<0.5) | 76.6% | 26.8% | 75.2% | 12.6% |
| UNFAITHFUL (JudgeA==0) | 76.8% | 10.8% | 1.1% | 0.3% |
| UNSUPPORTED-claim-heavy (>0.5) | n/a | n/a | 32.0% | 13.6% |
| STRUCTURAL-invalid | 79.3% | 81.4% | 82.8% | 83.1% |

**Post-fix (commit 54a62ae).** The **ALLERGEN row is now 0.0% on every config** — eliminated by
the deterministic allergen filter + the affected-record re-run (§4). The empty-plan explanation
gate cut H1-t2p **LOW-correctness 20.4% → 12.6%**, **UNFAITHFUL 8.0% → 0.3%**, and
**UNSUPPORTED-claim-heavy 21.2% → 13.6%**: the honest "no recommendation" message on gated
empty plans is judged correct/faithful where the previous hallucinated prose was not (§6).

Defined denominators (per config, out of 4,938): retrieval/hard/allergen/structural
≈ 4,839–4,938; correctness ≈ 4,788–4,931; faithfulness ≈ 4,798–4,930; unsupported-claim
4,200 (H1-t2p) / 4,644 (H1-dense).

**Read-out.**
- **The dominant failure is retrieval, and it is a clean dense-vs-t2p split.** Both dense
  configs miss the gold ~97% of the time; both t2p configs miss ~23–29%. Every downstream
  failure (hard-constraint, low-correctness) tracks this: on dense, hard-constraint violation
  is ~80% and low-correctness ~76%; on t2p they drop to 15–25% and 13–27%.
- **The agentic H1 pipeline shifts the error profile even at fixed retriever.** H1-t2p vs
  H2-t2p: retrieval MISS drops 29.1%→22.8% and low-correctness 26.8%→12.6% (Correct/Enhance
  recovers gold and improves answers, and the empty-plan gate keeps the judge from penalising
  gated empties) — but **hard-constraint violation *rises* 14.6%→24.8%** and EMPTY plan rises
  8.6%→10.7%. The agent trades a small amount of constraint-completeness for better recall and
  correctness (see §2 for why). The allergen row, previously the one safety liability, is now
  0.0% everywhere (§4).
- **STRUCTURAL-invalid ~80% across *all four* configs is a metric artifact, not a system
  failure.** `structural_validity` demands a full 4-slot day; the deployed meal-planner is
  time-driven and returns partial-day plans (mean 2.32 slots — see RESULTS_SUMMARY §5), so a
  well-formed 2-meal plan scores 0. Confirmed: among H1-t2p `sv==0` records, 3,655/4,020 *have*
  a non-empty plan, and their mean `meal_slot_completeness` is 0.39. Treat this row as
  "not-a-full-day", not "malformed". It is the one row in this table that should **not** drive
  any engineering decision.
- **UNFAITHFUL collapses from 76.8% (H2-dense) to ≤1.1% on both H1 configs** (H1-dense 1.1%,
  H1-t2p 0.3% post-fix). H2-dense's NL layer (a raw recommend-only description) frequently
  contradicts its near-empty plan; the H1 Enhance stage writes explanations tied to the actual
  plan, so faithfulness is high even when the plan is poor (H1-dense faithfulness 0.989 =
  cautiously describing a tiny plan). On H1-t2p the empty-plan gate removed the remaining
  unfaithful cases (the hallucinated empty-plan narrations), dropping UNFAITHFUL 8.0%→0.3%
  and lifting faithfulness 0.922→0.997.

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
| **H1-t2p** vs ret_t2p | 13.5% (221) | **5.2% (85)** | 81.4% (1337) |
| H2-dense vs ret_dense | 94.5% (1555) | 4.3% (71) | 1.2% (20) |
| H1-dense vs ret_dense | 94.5% (1555) | 4.7% (77) | 0.9% (14) |
| n = 1,646 per row (H1-t2p: 3 undefined) |

**Interpretation.**
- **On t2p, the agentic pipeline more than halves the "dropped gold" error: bucket (b) falls
  from 14.5% (H2-t2p) to 5.2% (H1-t2p).** This is the clearest quantified win for the agent:
  given the *same* retriever, recommend-only H2 discards the gold recipe from a valid candidate
  set in 1-in-7 cases, while H1's Correct/Enhance loop recovers most of those (both-hit rises
  71.3%→81.4%). The +6.4 pts of hit@10 in RESULTS_SUMMARY (0.709→0.773) is *almost entirely*
  this recovered bucket, not new retrieval.
- **Residual retrieval-limited error on t2p is ~13–14%** — the irreducible ceiling for these
  configs. No pipeline change can fix these; they require a better retriever (or larger *k*).
- **On dense, the pipeline is irrelevant.** 94.5% of items are retriever misses; buckets (b)
  and (c) are noise (≤5%). The agent "has nothing to correct". This is the mechanism behind the
  RESULTS_SUMMARY finding that *the agentic pipeline pays off only when retrieval is good*. Note
  bucket (b) is marginally *larger* for H1-dense (4.7%) than H2-dense (4.3%): with no valid
  candidates, the extra Enhance/Correct churn slightly *hurts* the rare dense hit rather than
  helping.

**Localisation.** On the strong retriever, FoodRAG error decomposes as ≈ **13–14%
retrieval-bound plus ~5% pipeline-drop** (H1-t2p). The biggest remaining pipeline lever is the
~5% gold-dropping in Correct/Enhance; the biggest retrieval lever is the ~14% floor.

---

## 3. Error by query difficulty

### 3a. By number of requirements (`num_requirements`)

Failure rates for **H1-t2p** (and H2-t2p for contrast). Buckets 1 and 2 are tiny (n=6, n=157)
so read them cautiously.

**H1-t2p**

| nreq | n | ret MISS | low-corr | hard-viol |
|---|---|---|---|---|
| 1 | 6 | 0.0% | 83.3% | 0.0% |
| 2 | 159 | 14.6% | 8.3% | 21.0% |
| 3 | 552 | 17.6% | 15.2% | 22.7% |
| 4 | 1131 | 21.1% | 14.1% | 25.9% |
| 5 | 1278 | 22.9% | 12.3% | 23.7% |
| 6+ | 1812 | **26.2%** | 11.1% | 26.1% |

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
now low and *falls* with nreq (~15% at 3 req → ~11% at 6+): once retrieval succeeds, adding
constraints does not degrade the judged answer, and on the hardest queries the empty-plan gate's
honest "no recommendation" message is scored correct where the old hallucinated prose was not
(§6). The nreq=1 low-correctness (83.3%, n=6) is a small-sample fluke.

### 3b. By requirement type (H1-t2p; a record can carry several types)

| Requirement type | n | ret MISS | low-corr | hard-viol | allergen-viol |
|---|---|---|---|---|---|
| allergen | 3312 | 23.2% | 12.2% | 25.1% | 0.00% |
| **ingredient** | 2961 | **33.3%** | 12.4% | 22.6% | 0.00% |
| tag | 4041 | 21.7% | 12.1% | 25.5% | 0.00% |
| nutrient_bound | 3543 | 24.1% | 13.1% | 25.4% | 0.00% |
| health | 4938 | 22.8% | 12.6% | 24.8% | 0.00% |

**Ingredient requirements are the hardest to retrieve for** — 33.3% retrieval MISS, well above
the ~22–24% of the other types. Free-text ingredient likes/dislikes (e.g. `"low - fat butter"`,
`"cilantro chutney"`) must be matched against noisy recipe ingredient lists, and the
Text-to-Pandas layer mis-translates them more often than it mishandles tags or numeric nutrient
bounds. This is the single most productive query type to target for retrieval improvement.
Post-fix, low-correctness is uniformly low (~12–13%) across all types — the empty-plan gate
removed the type-dependent penalty that previously made ingredient queries look worst on
judged correctness. Allergen violation is **0.00% on every type** after the deterministic
filter (§4).

### 3c. Allergen-requirement subset (safety-critical), all configs

3,312 of 4,938 item-runs carry an `allergen` requirement.

| Config | n (allergen) | ret MISS | low-corr | hard-viol | **allergen-viol** |
|---|---|---|---|---|---|
| H2-dense | 3312 | 97.6% | 78.2% | 82.0% | **0.00%** |
| H2-t2p | 3312 | 29.0% | 27.3% | 15.3% | **0.00%** |
| H1-dense | 3312 | 98.0% | 75.6% | 76.4% | **0.00%** |
| **H1-t2p** | 3312 | 23.2% | 12.2% | 25.1% | **0.00%** |

Allergen *violation* on this safety-critical subset is now **0.00% on every config** after the
deterministic allergen post-filter (commit 54a62ae + the affected-record re-run). Pre-fix, the
dense configs were ~10–13× worse on safety (1.18% / 0.83%) than the t2p configs (0.09%) because
their near-random candidates leaked disliked ingredients; the deterministic filter now vetoes
those slots regardless of retrieval quality, so safety no longer depends on the retriever.

---

## 4. Safety errors — deep-dive (now FIXED)

**Post-fix state: allergen violation is 0.000 on every config.** The deterministic allergen
post-filter (`constraints.enforce_allergen_safety`, commit 54a62ae) drops any final-plan slot
whose recipe's ingredient list intersects the persona's disliked/allergen set; it is applied to
the final plan in both assemblers, so it guarantees zero violations regardless of retrieval or
LLM validation. The 493 affected records across the four configs were re-run and re-judged, and
the post-fix `aggregate.json` allergen_violation rate is **0.000 [0.000, 0.000] on H2-dense,
H2-t2p, H1-dense and H1-t2p**. Because a blacklist-violating recipe can never be gold, hit@10 is
unaffected — only unsafe non-gold slots are removed.

The remainder of this section documents the **original (pre-fix)** liability the filter
eliminated, for the record.

**(Pre-fix) rare but categorically the worst failure.** Pre-fix aggregate rates with 95%
bootstrap CIs were: H2-dense 0.0034 [0.0019, 0.0052]; H1-dense 0.0021 [0.0011, 0.0032]; H2-t2p
0.0006 [0.0000, 0.0018]; H1-t2p 0.0003 [0.0000, 0.0009]. Pre-fix raw item-run counts with
`allergen_violation > 0`:

| Config | # item-runs w/ allergen viol (pre-fix) | distinct questions | post-fix |
|---|---|---|---|
| H2-dense | 39 | 39 (all run-1) | **0** |
| H1-dense | 27 | ~25 | **0** |
| H2-t2p | **3** | **1** (q_01495, all 3 runs) | **0** |
| H1-t2p | **3** | **1** (q_01495, all 3 runs) | **0** |

**(Pre-fix) concentration on t2p: a single persona.** For *both* t2p configs, every allergen
violation was **q_01495** across all 3 runs. That question dislikes `green onions`, likes tag
`lobster`, and the valid answer set has size 1; the pipeline served
**`lindas-lobster-rangoon-205992`**, which contains green onions. This was a deterministic,
reproducible single-persona failure, not diffuse noise — and it is exactly the case the new
deterministic filter targets. **The q_01495 case is now filtered**: the offending lobster-rangoon
slot is dropped, so all 3 runs report 0 allergen violations (verified by the commit's test
`test_safety_fixes.py`, which includes the q_01495 case).

**(Pre-fix) dense configs — diffuse and retrieval-driven.** The 39 (H2-dense) / 27 (H1-dense)
violations spread across many distinct personas (e.g. q_00088 shiitake, q_00255 Limburger cheese,
q_00281 Amarula liqueur, q_00522 granola, q_01040 turkey sausage). Because the dense retriever
returns near-random candidates, disliked ingredients leaked through. Most were av=0.25–0.5 (one
of several plan slots offending). The deterministic filter removes these slot-level leaks too,
so the dense configs are now at 0.000 as well.

**Safety posture (post-fix).** All four configs are now violation-free (0.000, CIs at 0). The
allergen liability — previously retrieval-driven and an order of magnitude larger on dense than
t2p — is eliminated deterministically: safety no longer depends on retrieval quality. The
residual differentiator between configs is now *retrieval and correctness*, not safety.

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
| **H1-t2p** | correctness | 4794 | 0.214 | 18.9% | 0.765 | 0.740 | 28.2% | 20.0% |
| **H1-t2p** | faithfulness | 4799 | 0.005 | 0.5% | 0.997 | 0.994 | 0.1% | 0.4% |
| **H1-t2p** | **relevancy** | 4813 | **0.140** | 14.0% | 0.869 | 0.785 | 2.8% | 11.2% |
| **H1-t2p** | expl-quality | 4791 | 0.164 | 16.2% | 0.750 | 0.792 | **33.6%** | 14.1% |

**Reliability read-out.**
- **`relevancy` is the least reliable metric** — highest MAD and disagreement in every config
  (up to MAD 0.351 / 35.1% disagreement on H2-t2p), and the direction is consistent: **Judge A
  scores relevancy much higher than Judge B** (A>B ≫ B>A everywhere). Gemma and DeepSeek
  interpret "relevancy" differently; treat relevancy as a soft signal, not a headline number.
- **`faithfulness` is the most reliable** — MAD 0.005–0.016 on both H1 configs (near-perfect
  agreement; H1-t2p MAD 0.005 post-fix, since the gated empty-plan explanations are trivially
  faithful for both judges). Faithfulness is a concrete "does the text match the plan" check
  both judges agree on.
- **The dual-judge design still catches a same-family bias on `explanation-quality`.** Same-family
  **Judge B (DeepSeek) scores explanation-quality higher** than cross-lineage Judge A: H1-t2p
  expl-quality meanB 0.792 > meanA 0.750 (**B>A 33.6% vs A>B 14.1%**), and H1-dense meanB 0.626 >
  meanA 0.573 (B>A 28.7% vs A>B 9.2%). This is the self-preference inflation the cross-lineage
  headline judge was chosen to avoid — reporting Judge A (the *lower*, cross-lineage estimate) as
  headline is the conservative and correct choice. **On correctness the bias is now muted:**
  post-fix, H1-t2p correctness meanA 0.765 sits *slightly above* meanB 0.740 (the empty-plan gate
  raised Judge A's scores on the gated records), though the directional split B>A 28.2% vs A>B
  20.0% still shows Judge B rewarding fluent prose on the records where they disagree.
  Faithfulness/relevancy do **not** show B-inflation (relevancy is if anything A-inflated), so
  the residual bias is specific to the open-ended explanation-quality judgment.

---

## 6. Explanation errors (H1 configs)

From `claim_grounding_rate` (cgr) and `unsupported_claim_rate` (ucr = 1 − cgr) on the NL
explanation layer:

| Config | n | mean ucr | mean cgr | ucr p50 | ucr p75 | ucr p90 | frac ucr==1.0 | frac ucr>0.5 |
|---|---|---|---|---|---|---|---|---|
| **H1-t2p** (post-fix) | 4200 | 0.283 | 0.717 | 0.25 | 0.50 | 0.67 | 7.3% | 13.6% |
| **H1-t2p** (pre-fix) | 4605 | 0.346 | 0.654 | 0.33 | 0.50 | 1.00 | 15.4% | 21.2% |
| H1-dense | 4644 | 0.447 | 0.553 | 0.50 | 0.67 | 1.00 | 15.4% | 32.0% |

**Read-out (post-fix).** The **empty-plan explanation gate** (`ExplanationTool.explain()` returns
a fixed safe message with no LLM call when the recommended plan is empty — commit 54a62ae) fixed
the dominant grounding failure on H1-t2p. Its measured effect: mean claim-grounding rises
**0.654 → 0.717**, fully-ungrounded explanations (ucr=1.0) fall **15.4% → 7.3%**, and
majority-ungrounded (ucr>0.5) fall **21.2% → 13.6%**. The p90 of ucr drops from 1.00 to 0.67 —
the fully-hallucinated tail is gone. On the stronger retriever ~72% of explanation claims are now
grounded (H1-t2p) vs ~55% on dense — better retrieval plus the gate yields better-grounded prose
(consistent with RESULTS_SUMMARY §8).

**Original failure (now gated) — patterns observed in the pre-fix high-ucr explanations:**
1. **Hallucinated recipes on empty plans (FIXED)** — e.g. q_00000 (plan all `None`) previously
   narrated *"Scrambled Eggs with Whole Wheat Toast … FSA score of 0.2 … total fat 10g … calorie
   count 200"*, all fabricated (`hit=0`, ucr=1.0); q_00009 similarly invented *"Agar Pudding with
   Fresh Fruit"*. **These empty-plan records now receive the fixed "no recommendation" message
   instead of hallucinated prose**, which is why ucr=1.0 dropped by half and faithfulness on
   H1-t2p rose to 0.997 (§1, §5).
2. **Fabricated precise nutrition figures** — specific FSA scores / gram values / kcal that were
   ungroundable because no recipe backed them; eliminated on empty plans by the gate.
3. **Admitted constraint violations narrated as acceptable (FIXED)** — the q_01495 explanation
   openly said the plan *"includes recipes that contain ingredients … you dislike, such as green
   onions"*; the deterministic allergen filter (§4) now drops that slot, so the violation and its
   narration no longer occur.

The residual ~28% ungrounded claims are on *non-empty* plans where the Enhance stage
over-describes a real plan; those are not caught by the emptiness gate and still need
claim-level grounding. The intrinsic KG-path explanation layer (`path_validity` = 1.000
everywhere) remains the reliable channel.

---

## 7. Case Browser — concrete examples

**Case A — ALLERGEN violation (safety, the entire t2p liability) — NOW FIXED.**
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
post-filter (commit 54a62ae) now drops the lobster-rangoon slot**, so all 3 re-run records report
`allergen_violation=0.0` (this exact case is covered by `test_safety_fixes.py`). hit@10 is
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

**Case D — Fully hallucinated explanation on an EMPTY plan (ucr=1.0) — NOW FIXED.**
`q_00000`, H1-t2p. Plan is entirely `None` (empty). Pre-fix, the explanation narrated a concrete,
invented recipe with fabricated nutrition:
> *"The 'Scrambled Eggs with Whole Wheat Toast' recipe features low-fat butter … FSA score of
> 0.2 … total fat content is 10g … calorie count is 200 …"*

Pre-fix metrics: `hit=0`, correctnessA=0.0, faithfulnessA=0.0, ucr=1.0, cgr=0.0. **The empty-plan
explanation gate (commit 54a62ae) now returns a fixed "no recommendation" message on empty plans
(no LLM call), so the Enhance stage can no longer hallucinate here** — this is the mechanism
behind ucr=1.0 dropping 15.4%→7.3% and faithfulness rising to 0.997 (§6, §1). (q_00009 "Agar
Pudding with Fresh Fruit" was an identical pattern, likewise gated.)

**Case E — Judge disagreement on the gated empty-plan record.**
`q_00000` (same empty-plan record as Case D). *Pre-fix*, the explanation was a fluent
hallucination: Judge A (Gemma) scored explanation_quality = 0.0 (correctly penalising it), while
Judge B (DeepSeek) scored 1.0 — rewarding fluency over grounding, the §5 same-family-inflation
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
  (~80% hard-violation, ~76% low-correctness). Allergen is now 0.0% (fixed). The agentic layer
  cannot help because there is nothing valid to correct (bucket (c) ≈ 1%). These configs are
  retrieval-bound, full stop.
- **H2-t2p** — *pipeline drops gold*. Retrieval is strong (29% miss) but the recommend-only
  stage discards the gold from a valid set in **14.5%** of items — its single largest avoidable
  error.
- **H1-t2p** (best config) — a mix of an **irreducible ~14% retrieval floor** + **~5%
  pipeline-drop** + a **narrow hard-constraint over-filtering** (hard-violation *rose* to 24.8%
  vs H2's 14.6%, from the agent under-filling plans). Allergen leaks and empty-plan
  hallucinations, previously the two named failures here, are now eliminated (§4, §6).

**Implemented fixes and their measured effect** (commit 54a62ae; 493 affected records re-run
and re-judged)
1. **Deterministic allergen post-filter** (`constraints.enforce_allergen_safety`, applied to the
   final plan in both assemblers) — vetoes any plan slot whose ingredients intersect the
   persona's disliked/allergen set. **Measured effect:** allergen violation **0.000 on all four
   configs** (was 0.0034/0.0021/0.0006/0.0003), including the q_01495 case (Case A) and the ~39
   diffuse dense violations (§4). hit@10 unaffected (a blacklist-violating recipe is never gold).
2. **Empty-plan explanation gate** (`ExplanationTool.explain()` returns a fixed "no
   recommendation" message, no LLM call, when the plan is empty). **Measured effect on H1-t2p:**
   fully-ungrounded explanations (ucr=1.0) **15.4%→7.3%**, claim-grounding **0.654→0.717**,
   faithfulness **0.922→0.997**, correctness **0.689→0.765**, UNFAITHFUL **8.0%→0.3%** (§1, §5,
   §6). Cases D and E are fixed.
3. **Distinct-slot guard** (`constraints.enforce_distinct_slots`, applied after the allergen
   filter in both assemblers) + `NonRepeatingReplayRetriever` for the full-day t2p replay path.
   Fixes a **degenerate-plan** failure mode — the same recipe placed in multiple slots — surfaced
   by the full-day benchmark: an unguarded full-day run put one dish in all four slots for up to
   **100%** of `h2_t2p` plans (plain replay returns the same pool every slot) and 17.7% of
   `h1_t2p`. **Measured effect:** **0.000% degenerate** plans across all four full-day configs
   (4,938 records each); the guard is a no-op on the already-distinct partial-day plans. See the
   full-day benchmark (RESULTS_REPORT §12).

**Remaining highest-leverage targets** (not yet done)
1. **Retrieval is still the highest-leverage target overall** — moving dense→t2p is a ~70-point
   swing on nearly every failure. Within t2p, **ingredient requirements (33% miss)** are the
   worst-served query type and the best place to improve the Text-to-Pandas query generator.
2. **The Correct stage's ~5% gold-dropping** (bucket (b), §2) is the remaining agentic lever.
3. **Claim-level grounding on *non-empty* plans** — the residual ~28% unsupported claims are on
   real plans the Enhance stage over-describes; the emptiness gate does not touch these.

**Safety posture (post-fix).** All four configs are now allergen-violation-free (0.000, CIs at
0), safety guaranteed deterministically rather than emergent from retrieval quality. The prior
concern — dense's retrieval-driven liability (H2-dense 0.0034 across 39 personas) and the
self-narrated t2p case (Case A) — is eliminated by the post-filter. Retrieval quality still
matters for *correctness*, but no longer for safety.

**Judge caveat for downstream readers.** Report **Judge A (Gemma, cross-lineage)** as headline:
same-family Judge B inflates explanation-quality (+0.04 on H1-t2p, +0.05 on H1-dense) via
self-preference (§5, Case E); post-fix the correctness inflation is muted (Judge A now slightly
*above* Judge B on H1-t2p). Treat **relevancy** as low-reliability (up to 35% cross-judge
disagreement); **faithfulness** is the most trustworthy judged metric (MAD ≈ 0.005–0.016).

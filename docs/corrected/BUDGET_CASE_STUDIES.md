# FoodRAG — Budget Case Studies: individual recommendations vs. the meal plan

Companion to `eval_results/RESULTS_SUMMARY.md` and `docs/RESULTS_REPORT.md`. Those report
aggregate metrics; this document walks through **real system outputs** for individual
personas to show two limitations concretely:

1. **the limitation of individual (item-level) recommendation** — a ranked list of single
   recipes has no notion of a *day*, so even a list of individually-valid recipes blows the
   persona's daily budget once you actually eat several of them; and for many personas the
   valid set is too small to compose a day at all;
2. **the limitation of a popularity-based method** — ranking by rating alone (rung 0a) is
   blind to the persona, so most of what it returns is invalid, and the health-aware variant
   (rung 0b) only works because it is handed the constraint-valid set as an oracle.

Everything below is pulled from the shipped evaluation artifacts, not re-run:
`eval_results/popularity_raw_per_question.parquet` (rung 0a),
`popularity_health_aware_per_question.parquet` (rung 0b),
`foodrag_h1_t2p_fullday_per_question.parquet` (rung 4, run 1), and the gold QA pickles.
Reproduce with `scripts/budget_case_study.py --case <qid> --aggregate`.

---

## What "budget" means here (and the individual-recommendation trap)

Personas may carry a `total_nutrients` daily budget. `config/nutrition_config.yml:TOTAL_NUTRIENTS`
defines these as **per-day** reference intakes (2000 kcal, 2300 mg sodium, 20 g saturated fat,
78 g total fat, …); the persona generator perturbs each by ±20%.

The system's own gold-answer logic in `src/foodrag/baselines/constraints.py`, however, applies
each cap **per recipe** — `row[n]/100 · servingSize ≤ cap` — because an item-level retriever
scores one recipe at a time and has nowhere to book the running daily total. **That is the
individual-recommendation limitation in code:** a recipe counts as "within a 2000 kcal/day
budget" as long as that single serving is under 2000 kcal, so a top-k list can hand you four
"valid" recipes that sum to a 3700 kcal day. Both the popularity baselines inherit this blind
spot. The meal-plan pipeline is the only rung that assembles slots into a day and can be
scored at the day level at all.

The case tables below therefore report, for each system, the **day-level total** of the
budgeted nutrient (the same whole-serving quantity, summed across the four recipes) against
the persona's stated daily cap. `100%` = exactly at budget.

---

## Case 1 — `q_00417`: a calorie-budgeted persona, valid recipes still bust the day

**Persona (`q_00417`, tight; gold answer set = 37 recipes):**

```
 - Ingredients Liked:    baking apples
 - Ingredients Disliked: raw tomatoes, light catalina dressing
 - Nutrition Scores:     fsa_score min. 0.31
 - Nutrition Limits:     caloriesFromFat [kcal] 17.93–179.3   (per-100g band)
 - Total Nutrition Limits: calories [kcal] max. 2114.18       (daily budget)
```

**A. Popularity-raw (rung 0a) top-10** — persona-independent, the *same* global list for all
10k personas (ranked by `average_rating`, tie-broken by `number_of_ratings`):

| # | recipe | kcal | satfat | sodium | sugar | NutriScore | FSA | WHO | rating | valid? |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|:--|
| 1 | Company's Coming Mashed Potatoes | 132 | 6.7 | 254 | 0.9 | 0.00 | 0.25 | 0.14 | 5.0 | **FAIL** |
| 2 | Kittencal's Balsamic Vinaigrette | 5043 | 69.5 | 9381 | 117.5 | 0.00 | 0.00 | 0.01 | 5.0 | **FAIL** |
| 3 | The Best Creole/Cajun Seasoning Mix | 1 | 0.0 | 0 | 0.0 | 0.75 | 0.50 | 0.43 | 5.0 | **FAIL** |
| 4 | Shrimp Nicoise Quiche | 748 | 27.9 | 1180 | 2.9 | 0.00 | 0.25 | 0.13 | 5.0 | **FAIL** |
| 5 | Caprese Salad Tomatoes | 96 | 1.3 | 410 | 1.7 | 0.00 | 0.38 | 0.19 | 5.0 | OK |
| 6 | Ragu-Shuka #Ragu | 729 | 13.0 | 1493 | 4.3 | 0.00 | 0.50 | 0.25 | 5.0 | OK |
| 7 | Toasted Almond (2) | 471 | 5.0 | 115 | 47.1 | 0.00 | 0.62 | 0.20 | 5.0 | OK |
| 8 | Roasted Garlic Paste (2) | 9167 | 86.0 | 446 | 25.3 | 0.00 | 0.50 | 0.28 | 5.0 | **FAIL** |
| 9 | A1b Bacon Burger Bites | 2122 | 44.8 | 14833 | 52.5 | 0.25 | 0.75 | 0.21 | 5.0 | **FAIL** |
| 10 | Warm Roasted Root Vegetable & Chicken Salad | 2708 | 16.0 | 1312 | 43.3 | 0.00 | 0.50 | 0.24 | 5.0 | **FAIL** |

Only **3/10 are even valid** — none contain baking apples, and #2/#8/#9 are single-serving
recipes that each blow the *entire* daily calorie budget on their own (5043, 9167, 2122 kcal).
Recorded `hit@10 = 0.0`, `constraint_satisfaction = 0.0`. This is the popularity-method
limitation: ranking by rating is blind to the persona.

**B. Popularity health-aware (rung 0b) top-10** — the same ranker, but restricted to the
constraint-valid set (it is *handed the answer key*, so all ten are individually valid):

| # | recipe | kcal | satfat | sugar | FSA | rating |
|---|---|--:|--:|--:|--:|--:|
| 1 | Boy Scout Apples | 457 | 14.9 | 51.4 | 0.38 | 5.0 |
| 2 | Crock Pot Baked Apples with raisins | 788 | 7.3 | 159.3 | 0.75 | 5.0 |
| 3 | Old-Fashioned Apple Pie (Splenda) | 481 | 6.1 | 18.1 | 0.38 | 5.0 |
| 4 | Apple-Cranberry Wheat Muffins | 1955 | 7.8 | 176.3 | 1.00 | 5.0 |
| … | (10 valid recipes available) | | | | | |

Every recipe is individually valid — yet **the top-4 as a day = 3681 kcal = 174% of the
2114 kcal budget** (recipes 1–4 sum to 457+788+481+1955). The item-level ranker has no way to
know that; each recipe passed its per-recipe check.

**C. FoodRAG H1-t2p meal plan (rung 4)** — four slots assembled as a day:

| slot | recipe | kcal | satfat | sugar | FSA | valid? |
|---|---|--:|--:|--:|--:|:--|
| breakfast | Gluten Free Apple Pecan Cake | 539 | 5.5 | 89 | 0.38 | OK |
| lunch | Altwiener Apfelstrudel | 367 | 8.7 | 21.5 | 0.38 | OK |
| dinner | Old-Fashioned Apple Pie (Splenda) | 481 | 6.1 | 18.1 | 0.38 | OK |
| snack | Fried Apples (6) | 100 | 0.2 | 15.2 | 0.88 | OK |

**Budget status:**

| system | day total (kcal) | % of 2114 kcal cap | within budget? |
|---|--:|--:|:--|
| A. popularity-raw top-4 | 5924 | 280% | ✗ |
| B. popularity health-aware top-4 | 3681 | 174% | ✗ |
| **C. FoodRAG meal plan** | **1487** | **70%** | **✓** |

The plan is 4/4 valid, sits at 70% of the calorie budget, and picks lighter members of the
same valid pool (it uses the pie and a light fried-apple snack rather than the 1955 kcal
muffin), because it is optimising the *day*, not each item in isolation. Recorded
`hit@10 = 1.0`, `hard_sat = 1.00`, `calorie_adherence = 0.93`.

---

## Case 2 — `q_00792`: a total-fat budget, popularity buries the persona in fat

**Persona (`q_00792`, tight; gold answer set = 15):**

```
 - Tags Disliked:        lemon
 - Ingredients Liked:    Campbell's condensed tomato soup
 - Ingredients Disliked: vinegar-pickled sauerkraut
 - Nutrition Scores:     who_score min. 0.09
 - Total Nutrition Limits: totalFat [g] max. 79.61            (daily budget)
```

| system | day total (g fat) | % of 79.6 g cap | within budget? | valid recipes |
|---|--:|--:|:--|:--|
| A. popularity-raw top-4 | 576 | 723% | ✗ | 6/10 valid |
| B. popularity health-aware top-4 | 129 | 162% | ✗ | 10/10 valid |
| **C. FoodRAG meal plan** | **60** | **75%** | **✓** | 4/4 valid |

Same pattern: the raw list is dominated by high-fat crowd-pleasers (the vinaigrette alone is
69.5 g saturated fat), the health-aware list is individually valid but its four top-rated
items still reach 162% of the day's fat budget, and only the day-assembled plan lands under
cap (Best-Ever Meatloaf / Sloppy Joes / spicy tomato soup / a near-zero hamburger snack, 60 g
fat total).

---

## Aggregate — this is not cherry-picked (n = 603)

Across **all 603 tight personas that state a daily budget and received a full 4-slot plan**
(`scripts/budget_case_study.py --aggregate`):

**Day total as % of the persona's stated daily cap** (100% = exactly at budget):

| statistic | A. popularity-raw top-4 | B. pop health-aware top-4 | C. FoodRAG meal plan |
|---|--:|--:|--:|
| median | 264% | 61% | 130% |
| mean | 344% | 80% | 254% |
| 25th–75th pct | 74–556% | 19–121% | 45–276% |

**Share of personas whose *day* stays within the stated daily budget:**

| system | within budget | median % of cap |
|---|--:|--:|
| A. popularity-raw top-4 | 29.5% | 264% |
| B. popularity health-aware top-4 | 66.3% | 61% |
| **C. FoodRAG meal plan** | **42.0%** | 130% |

**Can an individual recommender even fill a day?** For **32.7%** of these budgeted personas
the constraint-valid set has **fewer than 4 recipes** (for 77 personas it is a *single*
recipe) — an item-level recommender physically cannot compose a varied day. And of the 67.3%
whose valid set *is* large enough, **42.9% still exceed the budget** when you eat the top-4,
precisely because per-item validity does not compose.

### Honest caveats

- **The plan is not a budget solver, and the table shows it.** By construction (`constraints.py`)
  no rung — plan included — enforces a *day-level* cap; the plan wins on the budget only as a
  side effect of choosing lighter items and distributing calories across slots
  (`calorie_adherence`). Its median day is still **130% of cap** and it is within budget only
  42% of the time. Framed correctly, the plan's advantage is **diversity and day-level balance**
  (73% of plans use 4 distinct recipes — 442/603, with 81 using 3 and 80 only 2; it never
  repeats the single 1955 kcal muffin four times), not guaranteed budget compliance. A
  genuine day-level budget constraint is a
  **limitation of the current system too**, and a natural next step for the Correct/Enhance
  agents.
- **Health-aware "wins" on the raw budget number** (66% within cap vs 42%) — but only because
  it is an oracle: rung 0b is *given* the constraint-valid set, and its recorded `hit@10`≈1 is
  labelled in `RESULTS_SUMMARY.md` as a reference ceiling, not a fair retriever. It also cannot
  fill a day for a third of personas, and its "day" is just its top-4 with no slot structure,
  calorie distribution, or meal-type appropriateness.
- **HUMMUS serving-size noise inflates the raw-popularity numbers.** 16.8% of the corpus has a
  `servingSize > 500 g` (often `servingsPerRecipe = 1` whole-batch rows, e.g. the 9167 kcal
  garlic paste). This is real data the item-level baseline actually returns, but it means the
  raw-popularity blow-ups are partly a data-hygiene artifact; the health-aware and plan numbers
  (which draw from the constraint-valid pool) are the cleaner comparison.

---

## Reproduce

```bash
# one persona, full per-recipe nutrition + budget + validity breakdown
python scripts/budget_case_study.py --case q_00417 q_00792

# the n=603 aggregate table + per-persona CSV
python scripts/budget_case_study.py --aggregate --out-csv eval_results/budget_day_level.csv
```

# FoodRAG — Architecture & Reproducibility Reference

> A health-aware, personalised **meal-plan recommender** over the **HUMMUS** recipe knowledge
> graph. It takes a natural-language request (e.g. *"a low-sugar dinner without nuts"*) and returns
> a full daily meal plan (breakfast / lunch / dinner / snack) that respects dietary preferences,
> allergies, nutrient constraints and public health guidelines (EFSA / WHO / FSA / NutriScore).
>
> The system has three faces: (1) an **agentic RAG pipeline** built on LlamaIndex, (2) a
> **deterministic + dual-LLM-judge evaluation framework** with a full baseline ladder, and (3) a
> **Streamlit user-study application** (interactive exploration, structured evaluation, results
> report) deployable to a Hugging Face Docker Space.
>
> This document is grounded in the code under `src/foodrag/`, `ui/`, `scripts/`, `deploy/` and
> `config/`. Model ids, versions and hyperparameters are quoted verbatim from `config/models.yml`,
> `pyproject.toml`, `frozen_requirements.txt`, `docker-compose.yml` and the `Dockerfile`. It
> supersedes the earlier architecture notes.

## Table of contents

1. [System overview & use modes](#1-system-overview--use-modes)
2. [Data & provenance](#2-data--provenance)
3. [Persona & QA-set generation](#3-persona--qa-set-generation)
4. [Retrievers](#4-retrievers)
5. [Agents & tools — the H1 chain](#5-agents--tools--the-h1-chain)
6. [Query engine](#6-query-engine)
7. [Baseline & KBQA ladder](#7-baseline--kbqa-ladder)
8. [Models, providers, configuration](#8-models-providers-configuration)
9. [Safeguards](#9-safeguards)
10. [Evaluation framework](#10-evaluation-framework)
11. [Metric catalogue](#11-metric-catalogue)
12. [Guideline reference tables](#12-guideline-reference-tables)
13. [Pipeline options & the config matrix](#13-pipeline-options--the-config-matrix)
14. [User-study application (Streamlit UI)](#14-user-study-application-streamlit-ui)
15. [Deployment & runtime](#15-deployment--runtime)
16. [Reproducibility: run details, versions, seeds](#16-reproducibility-run-details-versions-seeds)
17. [Repository map](#17-repository-map)
18. [Appendix: end-to-end flow diagrams](#18-appendix-end-to-end-flow-diagrams)

---

## 1. System overview & use modes

FoodRAG recommends **meal plans** over the HUMMUS recipe corpus, honouring a *persona*'s dietary
constraints (allergens, nutrient bounds, health scores, liked/disliked tags and ingredients) and
health guidelines (EFSA / WHO nutrient bands). Recipes are **recommended, never generated** — the
system only returns existing HUMMUS recipes (with optional ingredient substitution) so validated
nutrition information is preserved.

There are **two evaluation use-modes** (the "dual-use" design):

- **Use-1 — retrieval-only.** A retriever produces a ranked recipe list scored directly against the
  persona's gold answer set and constraints. Driven by `foodrag.baselines.runner.run_baseline`
  (`src/foodrag/baselines/runner.py:47`) wrapping a `Baseline` such as `NaiveRAG`
  (`src/foodrag/baselines/naive_rag.py:21`).
- **Use-2 — full meal-plan pipeline.** A meal-plan *assembly layer* is attached on top of retrieval
  and the assembled plan is scored end-to-end (deterministic metrics + dual LLM judges). Driven by
  `foodrag.baselines.full_pipeline.run_full_pipeline` (`src/foodrag/baselines/full_pipeline.py:89`).

Within Use-2 there are **two heuristics**:

- **H1 — agentic Recommend + Correct + Enhance + Explain.** A five-agent LlamaIndex `AgentWorkflow`
  (Recommend → Validate/Correct → Substitute/Enhance → Explain → Finalize) built by
  `build_heuristic_workflow_builder` (`src/foodrag/baselines/kbqa/agentic.py:25`) and run through
  `WorkflowMealPlanAssembler` (`full_pipeline.py:311`). Emits a plan **plus an NL explanation**
  (`require_explanation=True`).
- **H2 — Recommend-only.** The meal-plan recommendation tool alone, in `retrieve_only` mode, via
  `RagMealPlanAssembler(retrieve_only=True, require_explanation=False)` (`full_pipeline.py:185`). No
  validation / correction / explanation.

The **H1-vs-H2 delta** isolates what the agentic correction + enhancement + explanation layer adds
over a plain recommend-only plan built from the same retrieval.

**Serving backends.** Recipes live as a Pandas `recipes_df` plus a **pgvector** dense index
(Postgres) and a **Neo4j** graph (text-to-Cypher). All LLM / embedding traffic goes through
**LiteLLM** behind a logging + retry + JSON-recovery wrapper. The Streamlit app (§14) drives the
same `src/foodrag` pipeline live (Flow A) or serves precomputed evaluation plans (Flow B) / results
(Flow C).

### The three system layers

```text
┌──────────────────────────────────────────────────────────────────────────┐
│  ui/  — Streamlit study app         Flow A (live pipeline)                 │
│                                     Flow B (precomputed study plans)       │
│                                     Flow C (read-only results report)      │
├──────────────────────────────────────────────────────────────────────────┤
│  src/foodrag/food_rag/  — agents, tools, retrievers, query engine (H1/H2)  │
│  src/foodrag/baselines/ — Use-1/Use-2 runners, baseline & KBQA ladder,     │
│                            metrics, judges, constraints, guideline adherence│
│  src/foodrag/persona/   — persona + QA-set generation (gold answers)       │
│  src/foodrag/llm/        — LoggedLiteLLM, retry, json_recovery, runstore    │
│  src/foodrag/config/     — models.yml loader, store credentials            │
│  src/foodrag/data/       — data manifest loader / validator                │
├──────────────────────────────────────────────────────────────────────────┤
│  Stores: Postgres+pgvector (dense/sparse/hybrid/fulltext) · Neo4j (Cypher) │
│  Providers via LiteLLM: DeepInfra · OpenRouter · NVIDIA NIM · Pagoda·Ollama │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Data & provenance

### 2.1 Source datasets & the data manifest

`config/data_manifest.yml` (version 1) is the **single source of truth** for every external file.
The `foodrag.data` DataLoader validates presence, auto-unzips local archives and downloads public
sources; auth-gated sources (Kaggle) are documented, not scripted. Validation is by **file presence
and byte size** — there are no content checksums.

| Manifest id | Directory | Source (URL) | Key files (size, bytes) | Public? |
|---|---|---|---|---|
| `hummus_raw` | `data/hummus/hummus_data/raw` | HUMMUS *connected-recipe-data-set* (`doi.org/10.1145/3604915.3609491`) | `recipes.csv` (1,127,130,859) ← `recipes.zip`; `reviews.csv` (721,405,890); `members.csv` (53,958,595) | yes |
| `foodcom_tags` | `data/hummus/hummus_data/additional` | Food.com Recipes & Interactions (Kaggle) | `RAW_recipes.csv` (294,520,189) ← `archive.zip` | **no** (free Kaggle acct) |
| `foodkg` | `data/hummus/food_kg` | FoodKG bundle (HUMMUS food_kg release) | `graph_ingredient_mapping.csv`, `graph_ingredient_labels.csv`, `food_kg_to_food_on.pkl`, `food_kg_to_usda.pkl` (optional), `foodon.owl` (auto-DL from `purl.obolibrary.org/obo/foodon.owl`) | yes |
| `foodon_imports` | `data/hummus/food_kg/imports` | FoodOn import closure (`foodon.org`) | directory, ≥1 file | yes |
| `usda` | `data/hummus/food_kg/usda_data` | USDA FoodData Central (`fdc.nal.usda.gov`) | `sr_legacy_foods.json` (210,486,988); `foundations_foods.json` (5,924,195); `survey_foods.json` (68,413,340); `branded_foods.json` (3,288,618,518, **excluded by default** via `load_util.USDA_EXCLUDED_FILES`) | yes |

Upstream scale (per HUMMUS): **270,284 recipes, 83,881 users, ~1.47M interactions**. After
threshold discarding (`k_recipe=2`, `k_user=2`: keep items with ≥2 interactions), the working
corpus is smaller. The FoodKG ingredient mappings resolve to SR-Legacy foods, so the ~3.3 GB
branded-foods USDA file is never loaded.

### 2.2 The working recipe table (`recipes_df`)

`data/hummus/recipes_df.csv`, loaded by `foodrag.baselines.runner.load_recipes_df`
(`runner.py:32`), which deserialises the list columns `tags` and `ingredient_names` via
`load_util.serialize` so constraint predicates see real lists. The larger `~7GB` `hummus_data`
object (ingredients, reviews, users, FoodOn) is loaded once per process by
`load_util.get_current_hummus("data")` and used only by tools that need it (explanation, ingredient
substitution).

`recipes_df` has **35 columns**:

| Group | Columns |
| --- | --- |
| Identity | `recipe_id`, `title`, `description`, `author_id`, `recipe_url`, `food_kg_locator`, `last_changed_date` |
| Structure | `duration`, `directions`, `ingredients`, `ingredient_names`, `ingredients_sizes`, `direction_size`, `serves`, `servingsPerRecipe`, `servingSize [g]`, `tags`, `meal_types` |
| Ratings | `average_rating`, `number_of_ratings` |
| Nutrients | `calories [kcal]`, `caloriesFromFat [kcal]`, `totalFat [g]`, `saturatedFat [g]`, `cholesterol [mg]`, `sodium [mg]`, `totalCarbohydrate [g]`, `dietaryFiber [g]`, `sugars [g]`, `protein [g]` |
| Health scores (normalised 0–1) | `who_score`, `fsa_score`, `nutri_score` |
| Misc | `Unnamed: 0`, `normalization_comment` |

The **canonical id is `recipe_url`** (`ID_COL` in `runner.py:24` and `full_pipeline.py:38`); gold
sets and retrieved lists are both keyed by it.

### 2.3 FoodOn ontology

The FoodOn RDF ontology (`data/hummus/food_kg/foodon.owl` + FoodKG mappings) is wrapped by
`rdf_util.PreProcessedFoodOn`. It provides an ingredient hierarchy so a preference like
"vegetable" expands to any descendant ingredient. It is used in:

- **QA generation** — answer filtering translates persona ingredients through FoodOn super-classes
  (`persona/qa_generator.py`, via `food_on.translate_ingredients`).
- **Ingredient substitution tool** — `IngredientSubstitutionTool` bridges food.com ↔ USDA ↔ FoodOn.
- **Constraint evaluation** — `constraints.foodon_expander` gives a FoodOn-aware `IngredientExpander`
  so allergen/blacklist checks match descendants, not only exact names. Note: the shipped QA set was
  generated with `include_food_on: false` (verbatim food.com names), so `identity_expand` is the
  runner default and FoodOn expansion is opt-in (`--food_on`).

#### ⚠ Do not use FoodOn expansion for allergen safety

`foodon_expander` is sound for **soft preference** expansion ("vegetable" → descendants), where a
wrong answer costs a mediocre recommendation. It is **not** fit for allergen or blacklist safety.
`identity_expand` (the shipped default, `include_food_on: false`) is the honest and safer choice, and
it is what every enforcement call site uses. Two independent reasons, both verified against
`data/hummus/food_on_ontology.ttl` (603k triples) and `data/hummus/food_on_mappings.pkl`.

**Reason 1 — subsumption ≠ composition (structural).** The expander traverses `rdfs:subClassOf` only
(`rdf_util._get_child_nodes`), so it resolves *is-a* and nothing else. "Wheat flour contains gluten"
is a composition fact and FoodOn does not assert it:

- There is **no "gluten-source" class**. `gluten` (`FOODON_03420177`) is a **childless leaf** whose
  parents are *gluten refined food product* and *protein extract, concentrate or isolate* — FoodOn
  models it as the isolated protein product (the bag of vital wheat gluten), not an allergen
  super-class. No triple anywhere targets it.
- The 15 `subClassOf` ancestors of `wheat flour` (`FOODON_03301116`) are *wheat food product*,
  *flour food product*, *cereal grain food product*, … — **none gluten-bearing**.
- So **`wheat flour ⊑ wheat ⊑ gluten-source` does not exist** and cannot be traversed. Blacklisting
  "gluten" expands to the empty set: string matching misses `wheat flour`, and so does the ontology.
- Consequently `get_ingredient_leaves` returns `[]` for `gluten`, so
  `get_all_nodes(remove_unconnected=True)` drops it from the persona vocabulary entirely
  (`persona_generator.py:58`) — under `include_food_on: true` a persona **cannot state "excludes
  gluten"** in FoodOn terms.

Gluten is, however, **not representative**: 12 of the EU-14 allergens *do* have usable classes with
descendants and leaves (only *sulphites* and *lupin* have no class label at all). The dominant
problem is reason 2.

**Reason 2 — the food.com↔FoodOn bridge is unreliable (dominant).** `food_on_mappings.pkl` (16,647
entries from FoodKG / `idea.rpi.edu/heals`, covering 91.3% of the 14,466 KG ingredients across only
2,249 distinct classes) is a **lexical-similarity join**, and its errors are gross and
safety-relevant in *both* directions:

| food.com ingredient | mapped FoodOn class | matched on |
|---|---|---|
| `dry roasted peanuts` | **nonfat dry milk** | "dry" |
| `baked tofu` | **trout baked** | "baked" |
| `black sesame seeds` | **black bass** | "black" |
| `Spanish peanuts` | **spanish vetch** | "spanish" |
| `anchovies` | **green olive (stuffed with anchovies, canned)** | substring |
| `aluminum foil` | **crab (deviled, shrimp added, in aluminum shell)** | "aluminum" |
| `Reese's pieces` | **shrimp pieces (raw)** | "pieces" |

So a peanut-allergic persona is **not** protected from `dry roasted peanuts` (it lives under
*dairy*), while a dairy-allergic persona **is** wrongly blocked from it. Two further structural
faults compound this: plain `crab` maps to `obsolete: crab` (zero ancestors, disconnected from
everything), and `cooked shrimp` maps to the **organism** class `shrimp` (under *crustacean
species* → *animal*) rather than the *food product* branch — FoodOn carries both hierarchies, and
`get_all_nodes` roots at *food product*, so every organism-branch mapping is invisible to expansion.

Measured recall against a naive substring baseline (indicative — the baseline has its own false
positives, but the individual misses below are verified by hand):

| allergen | recall | verified misses |
|---|---|---|
| egg | 16.8% | `bantam eggs` |
| soy | 31.6% | `baked tofu`, `black soybeans`, `Kikkoman soy sauce` |
| crustacean | 33.1% | `crab`, `cooked shrimp`, `blue crab meat`, `canned lobster` |
| tree nuts | 42.9% | `Diamond walnuts`, `Dole Almonds`, `PLANTERS Walnuts` |
| fish | 56.4% | `anchovies`, `albacore tuna`, `boneless skinless cod` |
| sesame | 63.4% | `black sesame seeds`, `tahini sesame butter` |
| peanut | 70.9% | `dry roasted peanuts`, `chocolate-covered peanuts` |
| milk | 71.6% | `1% fat buttermilk` |
| wheat | 82.2% | `bulgur wheat`, `cream of wheat`, `shredded wheat cereal` |

Precision fails symmetrically: *crustacean* also returns `aluminum foil` and `blue curacao`, *fish*
returns `Butterfinger bar` and `English muffins`, *peanut* returns `valencia oranges`, and — worst
for the allergen framing — `wheat food product` returns **`gluten-free rice flour`**.

**Bottom line for reporting.** FoodOn expansion is worse than substring matching for allergens on
*both* precision and recall. Do not claim ontology-grounded allergen detection for any allergen, and
do not enable `--food_on` for blacklist enforcement. Note the ontology itself is sound and the
subclass traversal is correct — the defect is in the FoodKG mapping layer, which is inherited from
the dataset, not built here. Real allergen support needs a curated allergen→ingredient expansion
table (auditable, testable) or a re-derived mapping; **neither is implemented**.

### 2.4 Vector & graph stores

- **Dense index (pgvector).** Postgres 16 + pgvector, table `hummus_recipes` (dim **1024**, cosine).
  Embeddings are **precomputed offline** (BGE-M3) into `data/embeddings/BAAI_bge-m3/recipe_documents/batch_*.pkl`
  and inserted without recomputation (`backend._ensure_recipe_index`). Idempotent: no-op once the
  table is non-empty.
- **Knowledge graph (Neo4j).** Recipe / Ingredient / Food_on / Tag nodes with `HAS_INGREDIENT`,
  `IS_EQUIVALENT_TO`, `SUB_CLASS_OF`, `HAS_TAG` relations. Built via `LOAD CSV` from
  `data/neo4j_kg/all/*.csv` (`transform_kg`, `Neo4jManager.insert_data`). Backs the text-to-Cypher
  retriever and the KBQA baselines. ≈ 1.9M nodes / 9.5M relations.
- **NebulaGraph** (legacy alternate graph store; `--profile nebula`) — present but not the primary path.

### 2.5 The evaluation QA sets

QA items are pickled lists at `data/qa/<set>/recipe_question_answer_set.pkl`. Each item:

```
question       # natural-language request
llm_questions  # LLM-reformulated variants, each with a judge score (1–5)
answer         # gold answer set: list of recipe_url (the "valid recipes")
persona        # the dietary-preference structure the constraints/metrics read
meal_plan      # per-meal-type valid-recipe sets (for plan-level gold)
```

Shipped sets and their strata (empty / tight / medium / broad):

| Set | n | empty / tight / medium / broad | Use |
|---|---|---|---|
| `calib` | 50 | 29 / 6 / 10 / 5 | calibration/smoke |
| `main_10k` | 10,000 | 5,518 / 1,646 / 1,192 / 1,644 | full generation |
| **`main_10k_tight`** | **1,646** | all tight | **headline stratum** (all runs) |
| `main_10k_tog100` | 100 | 0 / 32 / 45 / 23 | ToG unattended runs |
| `main_10k_s01…s10` | 1,000 each | ~536 / 162 / 142 / 160 | stratified slices |

The **tight** stratum (gold-set size 1–50) is the high-strictness discriminative slice: constraints
narrow the corpus to a small feasible set. Strata are computed post-hoc by
`baselines.base.answer_set_stratum` (`TIGHT_MAX=50`, `MEDIUM_MAX=5000`: 0→`empty`, 1–50→`tight`,
51–5000→`medium`, >5000→`broad`). `empty` is deliberately split out so unsatisfiable personas don't
swamp `tight`.

### 2.6 Data-provenance chain (end-to-end)

```text
Public / Kaggle sources (config/data_manifest.yml)
  HUMMUS GitLab · food.com Kaggle · FoodKG · FoodOn PURL · USDA FDC
       │ offline: DataLoader validates/unzips → data/hummus/, data/hummus/food_kg/
       ▼
Offline pipeline (maintainer machine)
  • hummus_transformation → data/hummus/*.pkl, data/neo4j_kg/all/*.csv
  • notebooks/init_vector_store → data/embeddings/BAAI_bge-m3/recipe_documents/batch_*.pkl
  • persona/qa_generator → data/qa/main_10k*/recipe_question_answer_set.pkl
  • full-day eval runs → eval_results/… → scripts/build_study_plans.py → ui/study_plans/*.json
       │  scripts/push_hf_data.py  (upload_large_folder, minus reviews_df.csv)
       ▼
Private HF Dataset  <FOODRAG_DATA_REPO>   [hummus/**, embeddings/**, neo4j_kg/**]
       │  first boot: scripts/bootstrap_hf_data.py  snapshot_download → /data (guard .hf_data_seeded)
       ▼
HF Docker Space (/data persistent volume)
  ├─ seed_pgvector.py → INSERT precomputed vectors into pgvector table hummus_recipes
  ├─ seed_neo4j.py    → LOAD CSV file:///all/*.csv into Neo4j
  └─ streamlit run app.py  :7860
       │  runtime interaction logs → FOODRAG_DATASET_REPO (HF Dataset) / optional Supabase
```

---

## 3. Persona & QA-set generation

Provenance of the gold answers, produced offline by `src/foodrag/persona/`.

### 3.1 Persona structure

```
persona = {
  'tags':               {'likes': [...], 'dislikes': [...]},
  'ingredients':        {'likes': [...], 'dislikes': [...]},   # dislikes ≈ allergens/blacklist
  'relative_nutrients': {nutrient: {'preference':'low|medium|high','limits':{'lower':X,'upper':Y}}},  # per-100g band
  'total_nutrients':    {nutrient: max_value},                 # per-day ceiling
  'nutrition_scores':   {score_name: min_value},               # who/fsa/nutri floor
}
```

### 3.2 Sampling parameters (`config/persona_config.yml`)

`generate_persona_preferences` (`persona_generator.py:9`) samples by random draw:

| Field | Rule |
|---|---|
| tags | likes ∈ [0,1], dislikes ∈ [0,2]; drawn from `recipe_tags` |
| ingredients | likes ∈ [0,1], dislikes ∈ [0,2]; `include_food_on:false`, `include_food_com:true` (verbatim food.com names) |
| relative_nutrients | count ∈ [0,1]; preference ∈ {low,medium,high} with `limits` from `nutrition_config.RELATIVE_NUTRIENTS` |
| total_nutrients | count ∈ [0,1]; sampled value = `TOTAL_NUTRIENTS[n] × Uniform(0.8,1.2)` |
| nutrition_scores | exactly 1 score; `nutri_score ∈ {1.0,.75,.5,.25,0}` (A–E), others `Uniform(0,1)` |
| meal-plan calories | `calories_per_day ~ 1600–3000`, split B 22% / L 32% / D 32% / snack1 7% / snack2 7% |

### 3.3 Gold answer computation

`_get_valid_recipes_for_persona` (`qa_generator.py:302`) hard-filters `recipes_df` sequentially,
short-circuiting to `[]` as soon as any filter empties the set: relative nutrients (per-100g within
band) → nutrition scores (`≥` floor) → total nutrients (`value/100 × servingSize ≤ max`) → liked
tags (any-of) / disliked tags (none-of) → liked/disliked ingredients (FoodOn-expanded via
`food_on.translate_ingredients`). This is mirrored **clause-for-clause** by
`baselines.constraints.filter_valid` and `kbqa.tfs_kbqa.compile_cypher`, so gold generation,
metric scoring and the TFS-KBQA baseline share one definition of "valid recipe".

### 3.4 The 4-stage QA pipeline

`generate_question_answer_set` (`qa_generator.py:99`) — resumable, each stage persisted to disk:

1. **`simple_qa`** — generate persona → templated NL question → compute gold `answer`. Parallel over
   `ProcessPoolExecutor` (`workers>1`).
2. **`llm_qa`** — LLM produces 5 diverse reformulations per question (formal / concise / casual /
   two free), async batched via litellm `acompletion` (`timeout=90`, `num_retries=3`).
3. **`scored_qa`** — an **LLM judge** rates each reformulation 1–5 for fidelity → `score`
   (prompt: `_rating_prompt`). Scores in `main_10k` / `main_10k_tight` were regenerated on
   2026-10-12 with `scripts/rescore_llm_questions.py`: the original prompt rated recipe questions
   without the original question or persona, so the old scores were invalid. They only fed
   `qa_helper.get_best_questions`; the evaluated query is the templated `question`. The judge is
   lenient (~78% of rewrites score 5; every entry has one), so a 5 is weak evidence of fidelity.
4. **`done`.** Defaults `size=10000`, `batch_size=10`, `storage_frequency=1000`.

Supporting: `persona_validator.py` (boolean AND constraint checks), `restriction_matcher.py`
(reward weighting for the agentic recommender: hard restrictions gate to 0, soft `×1.1`/`×0.9`),
`user_data_extractor.py` (mine real user review histories for occurrence-weighted sampling).

---

## 4. Retrievers

All FoodRAG retrievers implement **`DataFrameRetrieverBase`**
(`src/foodrag/food_rag/retriever/df_retriever_base.py:6`):

```python
class DataFrameRetrieverBase(ABC):
    @abstractmethod
    def retrieve(self, str_or_query_bundle: QueryType) -> pd.DataFrame: ...
    @abstractmethod
    async def aretrieve(self, str_or_query_bundle: QueryType) -> pd.DataFrame: ...
```

i.e. `query → pd.DataFrame of recipe rows` (full `recipes_df` schema). This lets every retriever
plug into `RecipeRAGQueryEngine` (§6) and `NaiveRAG` (Use-1) interchangeably.

| Retriever | Class (file:line) | Mechanism | Store | Key params |
|---|---|---|---|---|
| Dense / Sparse / Hybrid / Full-text | `RecipeRetriever` (`db_retriever.py:9`) | LlamaIndex `VectorIndexRetriever`; mode via `vector_store_query_mode`: `default`=dense cosine, `sparse`=BM25, `hybrid`=dense+sparse (RRF at store), `text_search`=full-text | pgvector `hummus_recipes` (1024-d) | `similarity_top_k=10`, `split_node_ratio=2` |
| Text-to-Pandas | `PandasQueryRetriever` (`pandas_retriever.py:173`) | LLM emits an executable pandas expression (`PANDAS_PROMPT_TEMPLATE` + `INSTRUCTION_STR` + column-desc dict) that **filters and ranks** in one shot; safe-exec/eval, `.head(k)` | Pandas `recipes_df` (no DB) | `similarity_top_k=10` |
| KG text-to-Cypher | `KGQueryEngine` (`kg_retriever.py:57`) | LLM generates Cypher from `DEFAULT_GRAPH_QUERY_SYNTHESIS_PROMPT` templated with the live graph schema; returns `recipe_url`s, normalises scores 0–1, defaults `nutri_score` | Neo4j (bolt) | schema from `data/neo4j_kg/all/*` |
| RRF fusion | `DataFrameFusionRetriever` (`advanced_retriever.py:13`) | Reciprocal-Rank Fusion over children; per-child failures → empty | any children | `ret_fusion`=dense+sparse, `llm=None` |
| Intersection | `IntersectionRetriever` (`advanced_retriever.py:114`) | recipes in **both** children (`df1 ∩ df2`, df1 order) | any | `ret_intersection`=dense∩sparse |
| Majority vote | `MajorityIntersectionRetriever` (`advanced_retriever.py:182`) | weighted majority (keep if summed weight ≥ total/2) | any | `ret_majority`=dense+sparse+hybrid+fulltext |
| Utility | `RandomRetriever`, `Doc2VecRetriever`, `BoWCosineRetriever` (`util_retriever.py`) | random / Doc2Vec cosine / BoW cosine | — | random baseline & ingredient-sub modes |
| Node adapter | `WrapperNodeRecipeRetriever` (`wrapper_retriever.py:10`) | adapts any node-returning `BaseRetriever` to the DataFrame interface | — | — |

### 4.1 Replay retrievers (driver-side, for reproducible re-scoring)

Defined **in the driver** `c:/tmp/run_foodrag_pipeline.py` (not in the package) to replay
saved text-to-pandas retrievals into the FoodRAG configs so no live re-retrieval happens:

- **`ReplayRetriever`** (`:283`) — ignores the query, always returns recipes for a fixed pre-saved
  `recipe_url` list (H2-t2p). One instance per item (thread-isolated).
- **`RotatingReplayRetriever`** (`:314`) — stateful; each `retrieve()` advances a cursor and returns
  the pool **rotated** so a new top recipe leads per slot (`[u0,u1,u2]→[u1,u2,u0]→…`). Needed for
  **live H1-t2p** where the tool calls `retrieve` once per slot and takes `.head(1)`; without it the
  plan collapses to one dish (partial-day).
- **`NonRepeatingReplayRetriever`** — the **full-day** variant: slices forward (`urls[cursor:]`,
  `cursor+=1`) so each slot gets a distinct rank-ordered recipe and goes empty once the pool is
  exhausted — never repeating. Wired for `*_fullday` t2p configs. One of the two full-day
  degeneracy guards (the other is `enforce_distinct_slots`, §9).

The UI backend's retriever factory (`ui/services/backend.py:404`) exposes `dense`/`sparse`/`hybrid`
(`RecipeRetriever`), `text-to-pandas` (`PandasQueryRetriever`, DB-free), and `fusion`
(`DataFrameFusionRetriever` over dense+sparse, best-effort KG).

---

## 5. Agents & tools — the H1 chain

### 5.1 The five-agent workflow

H1 is a **five-agent `AgentWorkflow`** built by `build_heuristic_workflow_builder`
(`src/foodrag/baselines/kbqa/agentic.py:25`). Every agent is a LlamaIndex **`FunctionAgent`** (native
tool calling), composed with **handoffs** (`can_handoff_to`) and a **shared `RecipeState`**:

```python
recommendation_agent = RecommendationAgent(query_engine=query_engine, llm=llm, time=time, date=date)
validation_agent     = ValidationAgent(llm=llm, query_engine=query_engine, recipe_df=recipes_df)
substitution_agent   = SubstitutionAgent(llm=llm)
explanation_agent    = ExplanationAgent(llm=llm, hummus_data=hummus_data)
finalization_agent   = FinalizationAgent(llm=llm)
AgentWorkflow(agents=[…], root_agent=recommendation_agent.name,
              initial_state=RecipeState.get_init_state())
```

| # | Agent (`heuristic_agents.py`) | Tool | `can_handoff_to` | Role |
|---|---|---|---|---|
| 1 | **RecommendationAgent** (`:110`) | `MealPlanRecommendationTool(retrieve_only=True, time, date)` | `[ValidationAgent]` | Produce an initial meal plan; MUST call the tool (never answers from own knowledge) |
| 2 | **ValidationAgent** (`:173`) | `CorrectionTool` | `[ExplanationAgent, SubstitutionAgent]` | Validate & correct; hand off to Substitution if still invalid, else Explanation |
| 3 | **SubstitutionAgent** (`:221`) | `SubstitutionTool` | `[ExplanationAgent]` | Enhance/repair recipes (e.g. "healthier"), then Explanation |
| 4 | **ExplanationAgent** (`:264`) | `ExplanationTool` | `[FinalizationAgent]` | Write the NL "why this plan fits" explanation |
| 5 | **FinalizationAgent** (`:309`) | `FinalizeRecommendationTool` | `[]` (terminal) | Assemble final (plan + explanation) once, then stop |

In the UI backend (`backend.build_workflow`), the three middle stages (Validation, Substitution,
Explanation) are **individually toggleable** — the handoff chain and system prompts are rewired
around dropped stages; Recommend and Finalize are always present. H2 = recommend-only single agent.

### 5.2 Shared state — `RecipeState`

`src/foodrag/food_rag/utils/recipe_context.py` wraps the workflow `Context["state"]` dict:

| Key | Holds |
|---|---|
| `ORIGINAL_USER_QUESTION_KEY` | original question |
| `MEAL_PLAN_QUESTIONS_KEY` | `MealPlanQuestions`: `{meal_type → per-slot reformulated question}` |
| `MEAL_PLAN_RECOMMENDATION_KEY` | `MealPlan`: `{meal_type → recipe dict}` |
| `MEAL_PLAN_VALIDATION_KEY` / `OVERALL_MEAL_PLAN_VALIDATION_KEY` | per-slot / overall verdicts |
| `SUBSTITUTED_MEAL_PLAN_RECOMMENDATION_KEY` | substitution-tool modifications |
| `RECOMMENDATION_TIME_KEY` | resolved `"{date} {time}"` used for slot reformulation (persisted) |
| `MEAL_PLAN_EXPLANATION_KEY` | NL explanation |
| `FINAL_RECOMMENDATION_KEY` | final assembled output |
| `HISTORY_KEY` | tool-execution event log |
| `LLM_USAGE_COUNTER_KEY`, `LLM_SEQUENTIAL_USAGE_COUNTER_KEY`, `LLM_PARSING_USAGE_COUNTER_KEY` | call counters |

`MealPlan` / `MealPlanQuestions` / `MealType` (`{breakfast,lunch,dinner,snack}` enum) live in
`src/foodrag/food_rag/meal_plan/`.

### 5.3 The tools (`src/foodrag/food_rag/tools/`)

- **`MealPlanRecommendationTool`** (`meal_plan_tools.py:40`). Reformulates the query into per-meal-type
  questions (`get_meal_plan_question`), resolving *time* + *date* (if `"now"`/`"today"` → `datetime.now()`,
  else pinned; persisted into `RECOMMENDATION_TIME_KEY`). The LLM decides which of the four meals are
  relevant given that time. `get_meal_plan()` retrieves per slot: `retrieve_only=True` (H1's Recommend
  agent, H2) takes the retriever's top recipe; `retrieve_only=False` lets the query-engine LLM pick.
- **`CorrectionTool`** (`validation_tools.py:346`). Validates each slot; for invalid slots runs a
  **5-level generalization loop** (`generalization_tools.py`): L1 keep everything → L2 generalize
  nutrient bounds to high/med/low → L3 drop nutrients (keep name + ingredients) → L4 drop specific
  ingredients → L5 keep recipe name only. Re-recommends and re-validates after each level; if a slot
  stays invalid after L5 the original is restored. **Allergens/disliked ingredients are never dropped**
  at any level.
- **`SubstitutionTool`** (`substitution_tools.py:12`) & **`IngredientSubstitutionTool`**
  (`ingredient_tools.py:25`). The former asks the LLM to modify a recipe toward objectives while
  preserving identity; the latter proposes alternative ingredients via FoodOn + USDA nutrition
  (retrieval modes EMBEDDING / DOC_2_VEC / BOW_COSINE).
- **`ExplanationTool`** (`explanation_tools.py:14`). Writes a concise user-level rationale for *why*
  the plan fits. **Empty-plan gate (commit 54a62ae):** when the plan is empty (all slots `None`),
  `explain()` short-circuits to a fixed `EMPTY_PLAN_MESSAGE` with **no LLM call**, instead of
  narrating recipes that were never selected (see §9).
- **`FinalizeRecommendationTool`** (`finalization_tool.py:26`). Assembles explanation + plan into
  `FINAL_RECOMMENDATION_KEY`, in `MODE_SIMPLE` (concatenate) or `MODE_LLM_REFINED` (LLM reformulation).

### 5.4 Why the chain works — the commit-`80b7a03` fix

The five-agent workflow originally produced **100% empty plans** with Llama-3.3-70B. Commit
`80b7a03` fixed three root causes:

1. **State-unwrap fix** (`recipe_context.py`). `RecipeState.from_context` assumed `Context["state"]`
   was `[dict]` and indexed `[0]`; `AgentWorkflow` stores a **bare dict**, so every tool hit
   `KeyError: 0`, caught it, and silently dropped all state writes. `_unwrap_state()` now accepts
   both shapes.
2. **ReAct → FunctionAgent** (`heuristic_agents.py`). Text ReAct was unreliable for tool
   emission/handoffs with the 70B model; the five agents became **`FunctionAgent`** (native tool
   calls).
3. **LiteLLM tool-calls** (`logged_llm.py`, §8.1). Force `is_function_calling_model=True` and route
   `astream_chat_with_tools` through the non-streaming path so modern `tool_calls` deltas aren't
   dropped.

`LiveH1Assembler` (`FOODRAG_H1_MODE=live`) is the default; `SeededH1Assembler` seeds the plan from
H2 and runs Correction → Explanation → Finalization directly as a deterministic fallback.

---

## 6. Query engine

**`RecipeRAGQueryEngine`** (`src/foodrag/food_rag/query_engines/rag.py:53`) is both a LlamaIndex
`BaseQueryEngine` and a `DataFrameRetrieverBase`. Two paths:

- **`retrieve(query)`** (`:272`) — delegates to the wrapped retriever, returns the DataFrame.
- **`_query(query)`** (`:107`) — the RAG path: (1) if `return_meal_plan`, rewrite "recipe(s)"→"meal
  plan"; (2) `retriever.retrieve`; (3) optional `ranking_strategy="rating"` sort; (4) select top-`k`
  — for meal plans via `_get_first_k_per_meal_type`, filling slots by each recipe's `meal_types` with
  fallback fillers (`:207`); (5) serialise recipes to text (`hummus_docify.recipes_to_docs`) and
  prompt the LLM with `DEFAULT_TEMPLATE` (single) or `MEAL_PLAN_TEMPLATE` (four slots as JSON); (6)
  if `return_full_recipes`, resolve chosen titles back to full docs.

Knobs: `retriever`, `llm`, `ranking_strategy` (`None`|`"rating"`), `k` (default 5), `prompt_template`,
`llm_calls`, `return_full_recipes`, `return_meal_plan`, `debug`. In Use-2 it is built with
`return_meal_plan=True`, `ranking_strategy="rating"`, `k=5`.

---

## 7. Baseline & KBQA ladder

The comparison ladder lets FoodRAG be scored against progressively stronger references. The
`Baseline` interface (`baselines/base.py:17`) is minimal: each subclass sets `rung`, `config_id`,
and implements `recommend(qa_item, k) -> list[recipe_url]`.

| Rung | Baseline (file:line) | LLM? | What it does | Key params |
|---|---|---|---|---|
| **0a** | `PopularityRaw` (`popularity.py:32`) | no | Global top-k by `average_rating` (tie `number_of_ratings`), **ignores persona** — health-unaware control | precomputed ranking |
| **0b** | `PopularityHealthAware` (`popularity.py:47`) | no | Same ranker, only over `constraints.filter_valid(persona)` recipes | — |
| **1** | item-based CF | — | *not implemented* (follow-up scope) | — |
| **2** | `LLMOnly` (`llm_only.py:37`) | yes | Generation without retrieval: prompt Llama-3.3-70B for k best real recipe *titles* (JSON), ground titles→url; unmatched = hallucinations | temp 0.1 |
| **3** | `NaiveRAG` (`naive_rag.py:21`) | (retriever) | Single-shot retrieve-then-rank, no agentic correction; top-k of any `DataFrameRetrieverBase`; also the EXP-1 retriever wrapper | k=10 |
| **4** | ToG / RoG / TFS-KBQA (`kbqa/`) | ToG:yes; RoG/TFS:fine-tuned | External KBQA over Neo4j (see below) | `KBQAOptions` |
| **4** | pFoodReq / BAMnet (`pfoodreq/`) | trained BAMnet | Prior-work anchor over a food KG (see below) | — |

**Shared fairness ranking (`kbqa/base.py:65`).** Every KBQA system orders candidates by
`(1-α)·evidence + α·health` (both min-max normalised, `health_alpha=0.25`), tie-broken by descending
NutriScore → FSA → rating → #ratings — so every baseline gets the same health information FoodRAG
gets. Tractability knobs (`KBQAOptions`, `base.py:40`): `max_depth=3`, `beam_width=5`, `fanout=50`,
`constraint_first=False`, `health_alpha=0.25`. Graph guards (`kg.py`): `DEFAULT_HUB_DEGREE_MAX=2000`
(hub Food_on classes pruned), `DEFAULT_FANOUT=50`.

- **ToG — Think-on-Graph** (`tog.py`, Sun et al. ICLR 2024). Training-free LLM-driven beam search:
  per depth, prune relations (LLM 0–1, keep top `beam_width`), expand one hop, prune entities (LLM
  0–1, cumulative path score), reasoning-call early stop once ≥k recipes. LLM may be `None` →
  deterministic fallback.
- **RoG — Reasoning-on-Graphs** (`rog.py`, Luo et al. ICLR 2024). Plan-then-retrieve: `RoGPlanner`
  emits relation-path plans over an 8-token `REL_VOCAB` (fine-tuned checkpoint → base-LLM prompt →
  `CANONICAL_PLANS` fallback), ground each plan (bounded one-hop), evidence = plan agreement (0.9
  decay), `blend_order`, top-k. `max_plans=beam_width`.
- **TFS-KBQA — semantic parsing** (`tfs_kbqa.py`). Compile question → structured Cypher → execute.
  `compile_cypher` is a deterministic persona→Cypher compiler mirroring `filter_valid`;
  `TFSParser` (fine-tuned checkpoint → base-LLM prompt → compiler fallback) generates Cypher, execute
  inline with fallback to the compiled query.
- **KBQA training** (`kbqa/training/`). LoRA SFT for RoG planner & TFS parser: `LORA_R=16`,
  `LORA_ALPHA=32`, `LORA_DROPOUT=0.05`, target `[q,k,v,o]_proj`, `MAX_LEN=1024`, `epochs=3`,
  `lr=2e-4`, `batch_size=4`, `grad_accum=4`, bf16 + gradient checkpointing. Suggested base
  `meta-llama/Llama-3.2-3B-Instruct`. TFS uses a three-pass curriculum (parse → exec-filtered → full).
- **pFoodReq / BAMnet** (`pfoodreq/` + vendored `third_party/pfoodreq/`, Apache-2.0). 2018-era
  attention memory-network KBQA over a FoodKG-style graph. Pipeline: HUMMUS→recipe_kg.json →
  vendored QA generator → vectorize → embeddings (**GloVe 840B 300d** faithful, or `BAAI/bge-m3`
  ablation) → train `BAMnetAgent` → recall/precision/F1. BAMnet config: `vocab_embed_size 300`,
  `hidden_size 128`, `mem_size 96`, `num_hops 1`, `lr 0.001`, `batch_size 32`, `num_epochs 100`.
  Driven via `foodrag.baselines.pfoodreq.train_bamnet` / `run_pfoodreq`
  (`--variant {glove,modernemb}`, tight-stratum bounded). Not yet run — no results shipped.

**Dual-use adapters** (`kbqa/adapters.py`): `KBQABaseline` wraps a system as a ladder `Baseline`
(rung-4 / EXP-1 retriever); `KBQARetriever` wraps it as a `DataFrameRetrieverBase` for the meal-plan
assembly layer. Factory `kbqa/factory.py` builds systems, baselines, retrievers, and both
`RagMealPlanAssembler` and the H1 `WorkflowMealPlanAssembler`.

CLI: `foodrag run-baselines` (`pipeline/run_baselines.py`) — defaults `--k 10`, `--rungs 0a,0b`,
`--experiment-id EXP-7`; `--neo4j` gates the KBQA rungs.

---

## 8. Models, providers, configuration

All model ids, endpoints and dims are centralised in **`config/models.yml`** and turned into
LlamaIndex objects by **`src/foodrag/config/settings.py`** (`load_models_config`, `build_llm`,
`build_embedding`, `build_judge`). Keys are read from `.env` by `providers[*].api_key_env` — never
hardcoded. Every OpenAI-compatible provider is routed through **LiteLLM** with the model id prefixed
`openai/` and `api_base` pointed at the provider.

| Role | Model id | Provider | Hyperparameters |
|---|---|---|---|
| Generator / agent LLM | `meta-llama/Llama-3.3-70B-Instruct` | **DeepInfra** | `temperature 0.1`, `max_retries 5`, `request_timeout 120` |
| Dense embedding (active) | `BAAI/bge-m3` | **DeepInfra** | `dim 1024`, `chunk_size 512` (winner of `bench-embeddings`, matches hybrid design) |
| Judge A (headline, cross-lineage) | `google/gemma-3-27b-it` | **OpenRouter** | logical `judgeA`; temperature 0 |
| Judge B (same-family control) | `deepseek-ai/DeepSeek-V3` | **DeepInfra** | `judges.default`; temperature 0 |
| Meal-type annotation (bulk) | `llama3.3:70b` | **Pagoda** (Ollama) | `temp 0.1`, `num_predict -1`, `keep_alive 30m`, `max_attempts 12`, `read_timeout 600` |
| Text-to-pandas backbone A | `meta-llama/Llama-3.3-70B-Instruct` | DeepInfra | `ret_text2pandas@llama70b` |
| Text-to-pandas backbone B | `gpt-4o-mini` | **OpenAI** | `ret_text2pandas@gpt4o-mini` |
| Embedding candidates | `Qwen/Qwen3-Embedding-4B` (2560d), `BAAI/bge-large-en-v1.5` (1024d), `sentence-transformers/all-mpnet-base-v2` (768d, local HF) | mixed | compared by `bench-embeddings` |
| KBQA fine-tune base | `meta-llama/Llama-3.2-3B-Instruct` | (local/HF) | LoRA r=16 / α=32 |

**Providers** (`models.yml`): `deepinfra` (`api.deepinfra.com/v1/openai`, `DEEPINFRA_API_KEY`),
`openrouter` (`openrouter.ai/api/v1`, `OPENROUTER_API_KEY`), `nvidia` NIM
(`integrate.api.nvidia.com/v1`, `NVIDIA_KEY`), `pagoda` Ollama (`OLLAMA_BASE_URL`/`OLLAMA_API_KEY`,
lab VPN), `huggingface` (local sentence-transformers, `HF_TOKEN`).

**Judge ≠ generator is enforced in code.** `build_judge` (`settings.py:199`) raises `ValueError` if a
judge's model equals `llm.model`, so same-model self-grading cannot happen. Judge A (Gemma) is
deliberately cross-lineage to the Meta generator; Judge B (DeepSeek-V3) differs too, making the
same-family-bias delta measurable.

### 8.1 `LoggedLiteLLM`

`src/foodrag/llm/logged_llm.py:34` — a `LiteLLM` subclass used by default (`build_llm(logged=True)`):

- **JSONL call logging.** `complete`/`chat` wrapped in `call_log.record_call(...)`, appending a
  `CallRecord` (kind, model, request_id, truncated prompt/response, latency, prompt/completion/total
  tokens, estimated `cost_usd`, attempt, error, ts) to `data/logs/llm_calls-<date>.jsonl`
  (`FOODRAG_LLM_LOG`).
- **Tenacity retry.** `@llm_retry(max_attempts=6, initial=2.0, max_wait=60.0)` (`retry.py`) —
  exponential backoff + jitter, retrying `RateLimitError` / `Timeout` / `APIConnectionError` / 5xx.
- **Forced function calling.** `metadata` forces `is_function_calling_model=True` (LiteLLM's static
  DB wrongly reports DeepInfra Llama-3.3 as non-tool-calling).
- **Non-streaming tool-call fix.** `astream_chat_with_tools` routes through non-streaming `achat`
  (upstream streaming drops modern `tool_calls` deltas — the empty-plan root cause, §5.4).

Related helpers: `runstore.py` (JSONL resumable item processing), `json_recovery.py` (robust JSON
ladder: `json.loads` → `ast.literal_eval` → repairs → fenced/balanced slice), `call_log.py`,
`ollama_pagoda.py`.

### 8.2 Store credentials (`src/foodrag/config/stores.py`)

Reads `.env` (defaults match `docker-compose.yml`): pgvector Postgres
(`POSTGRES_HOST=localhost:5432`, db `rag_postgres`), Neo4j (`NEO4J_URI=bolt://localhost:7687`, HTTP
7474, user `neo4j`), Nebula (`127.0.0.1:9669`, legacy). Local dev uses port 7687; some experiment
drivers pin Neo4j to bolt `7691`/HTTP `7478` to avoid clashing with other local containers.

### 8.3 Config-file catalog

| File | Purpose |
|---|---|
| `config/models.yml` | LLMs, embeddings, judges, providers, hyperparameters |
| `config/nutrition_config.yml` | `RELATIVE_NUTRIENTS` (per-100g low/med/high bands), `TOTAL_NUTRIENTS` (per-day RDI), `NUTRITION_SCORES` (who/fsa/nutri bands), nutrient/score name lists — used for persona generation & constraint validation |
| `config/guideline_reference_tables.json` | EFSA/WHO persona-class nutrient bands for the Guideline Adherence metric (§12) |
| `config/persona_config.yml` (+ `_2`) | persona sampling parameters (§3.2) |
| `config/data_manifest.yml` | external-data manifest (§2.1) |
| `.env` / `.env.example` | secrets + store credentials + `FOODRAG_*` flags (§15.3) |

---

## 9. Safeguards

FoodRAG layers **deterministic guards** on top of the stochastic LLM pipeline so safety-critical
guarantees never depend on retrieval or LLM behaviour. All are applied to the **final** plan in both
assemblers (`RagMealPlanAssembler` for H2, `WorkflowMealPlanAssembler` for H1).

| Safeguard | Where | Guarantee |
|---|---|---|
| **Deterministic allergen post-filter** (commit 54a62ae) | `constraints.enforce_allergen_safety(plan, persona)` | Drops any slot whose recipe ingredients intersect the persona's disliked/allergen set → **allergen_violation = 0.000 on every config**, regardless of retrieval or LLM validation. A blacklist-violating recipe can never be gold, so hit@10 is unaffected. **Read this as exact-match safety, not ontology-grade safety:** both call sites (`full_pipeline.py:239`, `:360`) use the `identity_expand` default, so the filter and the `allergen_violation` metric are the *same function on the same verbatim names*. The number is a consistency guarantee w.r.t. the QA set's own definition of "disliked" — it is structurally incapable of detecting a hierarchy-level escape (persona dislikes `wheat` → recipe lists `whole wheat flour` passes both filter and metric). See §2.3 for the scope of the FoodOn upgrade. |
| **Empty-plan explanation gate** (commit 54a62ae) | `ExplanationTool.explain()` | On an all-`None` plan, returns fixed `EMPTY_PLAN_MESSAGE` with **no LLM call** — eliminates fully-hallucinated empty-plan explanations (faithfulness 0.922→0.997, ucr=1.0 15.4%→7.3%). |
| **`enforce_distinct_slots` guard** | `constraints.enforce_distinct_slots(plan)` (after allergen filter) | Keeps a recipe in its first slot (B→L→D→S) and drops any later slot re-using the same `recipe_url` — prevents degenerate same-dish-in-many-slots plans. Never mutates input; `None`/empty pass through. Covered by `tests/test_distinct_slots.py`. |
| **`NonRepeatingReplayRetriever`** | driver (full-day t2p path) | Slices the replay pool forward per slot so no slot repeats; goes empty when exhausted. Second full-day degeneracy guard. |
| **Allergens never relaxed** | `CorrectionTool` 5-level generalization (`generalization_tools.py`) | Disliked/allergen ingredients are retained at every relaxation level (L1–L5); only soft preferences and nutrient bounds are dropped. |
| **Hard-ceiling UL breach** | `guideline_adherence.guideline_adherence` | Exceeding a Tolerable Upper Level (e.g. sodium hard-ceiling, §12) forces `hard_constraint_violation → 1.0` — an out-of-band nutrient is not automatically a safety violation, but a UL breach always is. |
| **Judge ≠ generator** | `settings.build_judge` | `ValueError` if a judge model equals the generator, preventing same-model self-grading. |
| **No recipe generation** | design | Only existing HUMMUS recipes are recommended (with optional ingredient substitution), preserving validated nutrition. |
| **Retry / logging / JSON recovery** | `LoggedLiteLLM`, `runstore`, `json_recovery` | Transient provider errors retried; every call logged; messy LLM JSON recovered rather than crashing. |
| **Resumable checkpointing** | drivers + `runstore` | Per-item / per-run JSONL checkpoints, resume by `question_id`; per-item `try/except → null record` so one failure never kills a run. |
| **Single-instance mutex** | drivers | Windows named mutex (`CreateMutexW`, exit on `ERROR_ALREADY_EXISTS`) so a run can't double-launch. |

---

## 10. Evaluation framework

Two entry points share one record schema.

### 10.1 Use-1 — `runner.py`

- **`run_baseline(baseline, qa_set, recipes_df, *, k, experiment_id, run, expand, id_prefix, max_workers)`**
  (`:47`). Phase 1: `baseline.recommend(qa, k)` per item, optionally in a `ThreadPoolExecutor`, each
  wall-clock timed. Phase 2 (sequential, deterministic): resolve ids → rows, compute
  `metrics.deterministic_metrics`, emit one per-question record.
- **`aggregate(records, *, n_boot=10_000, seed=0, judge="")`** (`:142`). Rolls up to `config × stratum
  × metric` rows: per-question means (NaN-aware), **95% percentile bootstrap CI** (10,000 resamples,
  2.5/97.5 pct), mean, and `per_question_variance`. Non-scalar fields coerced to NaN and skipped.
- **`write_results(records, aggregates, out_dir, config_id)`** (`:228`). Writes
  `{config_id}_per_question.parquet` (pickle fallback) and **upserts** rows into
  `eval_results/aggregate.csv` keyed by `(experiment_id, config_id, stratum, metric, judge)`.

### 10.2 Use-2 — `full_pipeline.py`

- **`run_full_pipeline(assembler, qa_set, recipes_df, *, judges, k, experiment_id, rung, run, …)`**
  (`:89`). Per item: `assembler.assemble(qa_item) → AssembledPlan`; then on the assembled plan's
  recipes compute deterministic + plan-quality + guideline-adherence + explainability + dual-judge +
  usage/latency blocks. Emits a record persisting `plan`, `explanation_text`,
  `explanation_structured`, `recommendation_time`, `requested_meal_types`.
- **Assemblers.** `MealPlanAssembler` (ABC) → `RagMealPlanAssembler` (H2) and
  `WorkflowMealPlanAssembler` (H1). `AssembledPlan` (`:42`) carries `question`, `answer_text`, `plan`,
  `recipe_urls`, `contexts`, `has_explanation`, `explanation_text/structured`, `recommendation_time`,
  `requested_meal_types`, usage/latency.
- **`full_day` mode.** Both assemblers take a `full_day` flag → the plan requests **all four** slots
  instead of the time-driven subset. Drives the `*_fullday` configs.

The resilient driver `c:/tmp/run_foodrag_pipeline.py` re-implements the per-item scoring loop
(`score_item`, `:539`) to checkpoint each record immediately; `_fullday` configs strip the suffix in
`base_config()` to resolve seeds/retrievers.

### 10.3 Per-question record (§13.1)

```
experiment_id, config_id, rung, run, question_id (q_{i:05d}),
answer_set_size, stratum, num_requirements, requirement_types, k,
metrics: { …deterministic + plan + guideline + explainability + judge + usage… },
retrieved_ids: [recipe_url, …],
# Use-2 only:
plan: {meal_type: recipe_url|None}, explanation_text, explanation_structured,
recommendation_time, requested_meal_types
```

### 10.4 Aggregate record (§13.2)

```
experiment_id, config_id, stratum, metric, mean, ci_low, ci_high,
per_question_variance, n_questions, n_runs, judge
```
Strata order: `aggregate` + present strata tight → medium → broad → empty.

### 10.5 Dual-judge pass — `judge.py`

`build_judges` (`:91`) builds `judgeA = gemma` (cross-lineage) and `judgeB = deepseek-v3`.
`judge_plan` (`:106`) emits `<metric>_judgeA` / `<metric>_judgeB`:

- **Faithfulness** {0,1} — uses only retrieved recipes?
- **Relevancy** {0,1} — fits question + retrieved set?
- **Correctness** [0,1] — vs question + gold, rated 1–5 → `(s−1)/4`.
- **Explanation Quality** [0,1] — only when `require_explanation` (H1), 1–5 → normalised.

Prompts frozen in the module; judges run at **temperature 0** under the same LiteLLM logging.

### 10.6 Statistical testing

Beyond bootstrap CIs, `foodrag.analysis.stats` computes **paired Wilcoxon signed-rank** over
per-question vectors (aligned on `question_id`, NaN pairs dropped), a paired bootstrap (10,000
resamples, seed 0) of the mean difference, median-Δ effect size, and **Holm–Bonferroni** correction
within each metric family. Output → `eval_results/pairwise.csv` (+ `pairwise_fullday.csv`). See
`docs/STATISTICAL_TESTS.md`. Pareto frontiers (`analysis/pareto.py`) span Health-vs-Relevance,
Quality-vs-Cost, Safety-vs-Coverage.

---

## 11. Metric catalogue

Every metric below is computed in code.

**Retrieval** (`metrics.py`, over top-k `recipe_url` vs gold): `hit_at_{k}`, `mrr_at_{k}`,
`ndcg_at_{k}` (binary-relevance). NaN when gold empty.

**Health** (`metrics.avg_score_at_k`): `avg_nutri_at_{k}`, `avg_fsa_at_{k}`, `avg_rating_at_{k}`
(relevance axis for the Health-vs-Relevance Pareto).

**Safety / constraints** (`metrics.py` + `constraints.py`): `allergen_violation` (FoodOn-aware
fraction of top-k with a disliked ingredient — **0.000 after the post-filter**),
`hard_constraint_violation` (also forced to 1.0 on UL breach), `constraint_satisfaction`,
`hallucination_rate`, and the hard/soft split `hard_constraint_sat_perconstraint`,
`hard_constraint_sat_allornothing`, `soft_constraint_sat`. Unresolved (`None`) recs never get a free pass.

**Plan-level** (`meal_plan_metrics.py`): `structural_validity` (four slots + explanation if
required), `meal_slot_completeness`, `intra_plan_diversity` (1 − mean pairwise ingredient/tag
Jaccard), `calorie_adherence` (1 − mean abs deviation from B 22% / L 32% / D 32% / S 14% targets),
`slot_appropriateness`.

**Guideline adherence** (`guideline_adherence.py` + `config/guideline_reference_tables.json`, §12):
`guideline_adherence_per_nutrient` (per-nutrient 0/1 in-band over 8 nutrients),
`guideline_adherence_overall` (mean), `guideline_class` (resolved class), hard-ceiling breach flag.

**Judge** (`judge.py`): `faithfulness_{A,B}`, `relevancy_{A,B}`, `correctness_{A,B}`,
`explanation_quality_{A,B}` (H1 only).

**Explainability** (`explainability.py`): `path_coverage` (fraction of stated requirements with a
supporting recipe-attribute path), `path_validity`, and (with an explanation) `claim_grounding_rate`
/ `unsupported_claim_rate` from rule-based NL claim extraction (score-min, nutrient-range/max,
ingredient include/exclude, tag-exclude, NutriScore-letter).

**Processing / stability**: `llm_usage`, `llm_seq_usage` (sequential lower bound), `llm_parsing`
(repair calls), `latency_s`, and (aggregate) `per_question_variance`.

---

## 12. Guideline reference tables

`config/guideline_reference_tables.json` (schema 1.0) defines the persona-class nutrient bands for
the Guideline Adherence metric. It is **researcher-constructed from authoritative public guidelines
(EFSA DRV, WHO), NOT clinically validated** — a documented modelling assumption, not clinical advice.

**Why class-based, not individualised.** EFSA energy references depend on body weight, sex, age and
PAL, but GDPR-minimal personas carry only age-range + condition + allergies + tags. Each persona is
mapped to **one** reference class with fixed assumptions: reference body weight 70 kg, reference
energy 2000 kcal (overridden per class), PAL levels {sedentary 1.4, moderate 1.6, active 1.8, very
2.0}, energy-%-to-gram conversion (carb/protein 4 kcal/g, fat 9 kcal/g).

**Reference classes** (per-day bands; `hard_ceiling` = safety UL):

| Class | Energy kcal | Carb g | Sat-fat g | Sodium g (ceiling) | Sugars g |
|---|---|---|---|---|---|
| `general_adult` (default) | 1800–2200 | 225–300 | ≤22 | 0.575–3.5 (**3.5**) | ≤50 |
| `diabetic_adult` | 1800–2200 | **200–250** | ≤22 | 0.575–3.5 (3.5) | **≤25** |
| `hypertension_adult` | 1800–2200 | 225–300 | ≤22 | 0.575–**2.0** (**2.0**) | ≤50 |
| `weight_management_adult` | **1400–1800** | 180–240 | ≤18 | 0.575–3.5 (3.5) | ≤40 |

(Protein 58–116 g; total fat 44–78 g; fibre 25–40/45 g across classes.) **Persona→class mapping**
(first match wins): diabetes → `diabetic_adult`; hypertension/low-sodium/CVD → `hypertension_adult`;
weight-loss/obesity → `weight_management_adult`; else `general_adult`. **Comorbidities** take the
**stricter** band per nutrient (max lower, min upper) across matched classes.

**Metric.** Per tracked nutrient (energy, protein, carb, total-fat, sat-fat, fibre, sodium, sugars):
1 if daily plan total ∈ [low, high] else 0 (record under/over direction); `overall` = mean.
Exceeding a `hard_ceiling` additionally feeds Hard-Constraint Violation (§9, §11). Documented
limitations: single-day snapshot, class-based (not individualised), sat-fat/sugar upper thresholds
are WHO/policy (not EFSA ULs), protein/fibre upper bounds are generous modelling ceilings, not
clinically validated.

`config/nutrition_config.yml` is the separate, older band table used for **persona generation &
constraint validation** (per-100g `RELATIVE_NUTRIENTS`, per-day `TOTAL_NUTRIENTS` RDI, `NUTRITION_SCORES`).

---

## 13. Pipeline options & the config matrix

The system is intentionally modular; a "config" is a `(use-mode, heuristic, retriever, backbone,
day-mode)` tuple. All FoodRAG serving/eval knobs:

| Axis | Options |
|---|---|
| Use-mode | Use-1 (retrieval-only) · Use-2 (full meal-plan pipeline) |
| Heuristic (Use-2) | H1 (five-agent, +explanation) · H2 (recommend-only) |
| Middle agents (H1, UI) | Validation / Substitution / Explanation individually toggleable |
| Retriever | dense · sparse (BM25) · hybrid · fulltext · text-to-pandas (Llama-70B / gpt-4o-mini) · KG text-to-Cypher · RRF fusion · intersection · majority-vote · replay variants |
| Day-mode | partial-day (time-driven meal subset) · full-day (all 4 slots, `*_fullday` configs) |
| Meal-plan tool | `retrieve_only=True` (top-1 per slot) · `retrieve_only=False` (LLM-select) |
| Finalization | `MODE_SIMPLE` · `MODE_LLM_REFINED` |
| Time/date | `FOODRAG_MEALPLAN_TIME` / `FOODRAG_MEALPLAN_DATE` (`now`/`today` → `datetime.now()`, or pinned) |
| Ranking | `None` · `"rating"` (by `average_rating`) |
| k | retrieval k=10, meal-plan k=5 |

The four headline FoodRAG **system configs** (Use-2): `foodrag_h1_dense`, `foodrag_h1_t2p`,
`foodrag_h2` (dense), `foodrag_h2_t2p` — each with a `_fullday` variant. The user study (§14) maps
its three conditions to `foodrag_h1_dense_fullday` (C1), `foodrag_h2_fullday` (C2),
`foodrag_h2_t2p_fullday` (C3). Around these sit the 8 retriever configs, 5+ baseline/ladder rungs,
and the KG/heuristic experiment ids (`EXP-ret-tight`, `EXP-1`, `EXP-3`, `EXP-5`, `EXP-7`, `EXP-10k*`)
— 26 distinct configs / 949 aggregate rows in the shipped `eval_results/`.

---

## 14. User-study application (Streamlit UI)

`app.py` is a single-page Streamlit "wizard": routing is driven by `st.session_state["screen"]` (not
Streamlit `pages/`). It bootstraps Docker stores once (`stores_bootstrap.ensure_stores_up`, cached),
warms the backend engine in a daemon thread (`backend.warm_engine_in_background`), then routes. Two
URL escape hatches: `?admin=true` (password-gated admin) and `?report=true` (Flow C directly).

### 14.1 Session state (`ui/state.py`)

`init_state()` seeds: `session_id` (8-hex uuid), `participant_number` (`int(session_id,16) % 997` —
drives assignment/counterbalancing), `screen` (`welcome`), `flow` (`None`|`exploration`|`evaluation`),
`consent_given`, `admin_config` (from `config_service.effective_config()`), Flow-A keys
(`hexad_scores`/`hexad_type`, `persona`, `config_a`, `config_b`), Flow-B keys (`demographics`,
`assigned_profile_id`, `plan_order`, `plan_index`, `plan_answers`, `attention_results`,
`post_study_answers`, `ranking_result`), and `event_log`. `goto(screen)` sets screen + reruns;
`log_event` delegates to `logging_service`.

### 14.2 The three flows

**Flow A — Interactive Exploration** (`flow="exploration"`): `welcome → consent → persona → main`
(loops). `persona.py` shows cards from `study_data.flow_a_personas()` (the same evaluation profiles,
`flow_a_profiles` default `["P1","P3","P4"]`, + a custom-builder slot); selecting sets `persona` and
prefills `query_text`. `main.py` is the terminal screen: query box, **pipeline strip** (retriever
selectbox + Validation/Substitution/Explanation toggles), run, and an **arena mode** (A/B configs
side by side). It runs the **live** agentic pipeline when `rag_service.real_enabled()` (env
`FOODRAG_REAL_BACKEND` → session `backend_live` → admin `use_real_backend`), else a mock simulation.
**HEXAD is currently bypassed** (`consent.py` routes exploration straight to `persona`; `hexad.py`
exists and is routable but out of the live path).

**Flow B — Structured Evaluation** (`flow="evaluation"`): `welcome → consent → demographics →
profile_assignment → plan_evaluation (×N, default 3) → post_study → ranking → completion`. Flow B
uses **precomputed full-day plans** (`ui/study_plans/*.json`), never the live pipeline.
`profile_assignment` assigns a profile + plan order idempotently (§14.4). `plan_evaluation` iterates
`plan_index` 0..N−1, each with a 7-item ResQue-based Likert block + a rotated attention check;
`post_study` = 7 cross-cutting items; `ranking` = rank the plans; `completion` logs `study_completed`
once and flushes.

**Flow C — Report** (`report.py`, read-only over `eval_results/`): three tiers of tabs — ① Headline
(Overview / Retriever comparison / Safety), ② Attribution (Agentic comparison / Error taxonomy /
Retrieval-vs-full-pipeline), ③ Depth (Pareto / Explainability / Judge reliability / Plan quality /
Guideline adherence / Time-of-day / Case browser / Volatility / Stratified). Data layer
`report_data.py` memoises `aggregate.csv`, `pairwise*.csv`, per-question parquets and derives error
taxonomy, judge agreement (Gemma vs DeepSeek Pearson), time-of-day, volatility, cases; copy from
`ui/report_copy.json`.

### 14.3 Services (`ui/services/`)

| Service | Role |
|---|---|
| `backend.py` | Heavy singletons: `Engine` (llm, embed, cleaned `recipes_df`, `hummus_data`, lazy pgvector/Neo4j), double-checked `get_engine()`, cold-start cache `data/hummus/_ui_engine_cache.pkl`. Retriever factory, `RecipeRAGQueryEngine` (rating, k=5), `build_workflow` (H1 five-agent / H2), `run_pipeline` (drives `AgentWorkflow` on a persistent asyncio loop, `wait_for` timeout 300 s with partial-state salvage). Generator = Llama-3.3-70B (temp 0.1), embeddings = bge-m3 (1024). |
| `rag_service.py` | Single GUI→pipeline entrypoint: `real_enabled()`, `build_question(persona, query)`, `generate_meal_plan(...)` (background thread, maps `RecipeState`→GUI cards + correction log + splits + explanation, deterministic FoodOn `_safety_check`). |
| `mapping.py` | Pure `RecipeState`→GUI adapters (HUMMUS cols → cards; `nutri_letter`, per-serving nutrition, food.com URL). No Streamlit/backend imports. |
| `report_data.py` | Flow-C data layer (memoised, read-only). |
| `stores_bootstrap.py` | `ensure_stores_up()` (`docker compose up -d postgres neo4j`; no-op under pytest / `FOODRAG_IN_CONTAINER` / `FOODRAG_SKIP_STORES_BOOTSTRAP`), `stores_reachable()` (TCP :5432). |
| `study_results.py` | Reads already-run full-day result files ({meal→url} + explanations); the study **selects** plans, never re-runs. Scores diverse, non-degenerate, not-too-easy questions for the admin profile picker. |
| `config_service.py` | JSON persistence of `admin_config` (local `data/study_config.json` + HF Dataset `config/study_config.json`, deep-merged over defaults). |
| `logging_service.py` | §9 event pipeline: enrich → in-memory buffer → 5-min background flush to HF `logs/<session>.jsonl` (`FOODRAG_DATASET_REPO`), optional Supabase sink. |

The judges (Gemma / DeepSeek) appear **only** in Flow C report data — they are offline-eval, not in
the UI serving path.

### 14.4 Study design, plans & counterbalancing

- **`ui/study_data.py`.** `STUDY_CONFIG`: `plans_per_profile=3`, `profile_assignment=random_round_robin`,
  `counterbalancing=latin_square`, `attention_checks=True`, `likert_scale=5`. Conditions C1/C2/C3 →
  real pipeline configs (above); `config` is logged with every rating but hidden in the GUI.
  Instruments: `PLAN_EVAL_ITEMS` (7 ResQue-based: Perceived Quality ×2, Info Adequacy, Usefulness,
  Trust, Transparency, Satisfaction), `ATTENTION_ITEM` (expected "Disagree"=2, inserted at rotated
  positions `[3,6,2]`), `POST_STUDY_ITEMS` (7: Explanation Quality / Constraint Compliance /
  Behavioural Intention / Usability). 5-point `LIKERT_5`.
- **`ui/study_plans/`.** `P1.json`…`P4.json` = per-profile artifacts (`profile_id`, `question_id`,
  `profile_prompt`, `plans[]` each with `cond`, `config`, `explanation`, real recipe cards);
  `profiles.json` = the 4 evaluation profiles with full constraints. Generated offline by
  `scripts/build_study_plans.py` (reads full-day eval results, resolves plans against HUMMUS, applies
  calorie-plausibility filters, no LLM) and committed to the repo.
- **`ui/counterbalancing.py`.** Between-subjects **round-robin** `assign_profile_index(pnum,n)=pnum%n`;
  within-subjects **Latin square** `get_plan_order` (`LATIN_SQUARE_3` balanced 3×3, `LATIN_SQUARE_4`
  Williams 4×4). Participants **role-play an assigned profile** (randomized, to remove self-selection
  bias) rather than choosing.

### 14.5 Persuasion / behaviour-model tailoring

The **HEXAD-12** gamification questionnaire (Tondello et al. 2016; `hexad.py`, 12 items × 7-point
Likert, 6 types) is fully implemented — scoring, logging, and inclusion of `hexad_type`/`hexad_scores`
in every log record — but **not wired into recommendation/explanation tailoring** and currently
out of the active flow. The real tailoring is **constraint-based** (persona → `build_question` →
pipeline). The admin block catalog references further instruments (REBS/Pelletier 2004, cooking-skills
Bölz 2025, ResQue-15, SCS) as catalog entries for future studies. The persuasion add-ons
(explanations, behaviour-model tailoring, gamification, nutrition labels) are designed as modular
extensions motivated by behaviour-change theories (TTM, SDT, HBM, FBM).

### 14.6 Admin panel (`ui/screens/admin/`)

`view.py` is a password-gated shell (`ADMIN_PASSWORD` env, dev fallback `"admin"`) with 6 tabs:
`dashboard` (analytics over `event_log`), `persona_manager` (edit `flow_a_profiles`, personas),
`questionnaire_config` (edit `active_flow`, `plans_per_profile`, `blocks`, + persistence),
`pipeline_defaults` (edit retriever/validation/substitution/explanation/`use_real_backend`),
`study_flow` (edit conditions/profiles, map to `eval_results/` files), `data_export` (JSONL export,
HF flush, config export).

### 14.7 Theme & components

`ui/theme.py`: palette (primary `#2d6a4f`, accent `#e76f51`, official Nutri-Score A–E colors), fonts
Source Sans 3 / Playfair Display, `inject_global_css`, `html()` helper. `ui/components/`:
`pipeline_strip`, `recipe_detail` (modal), `study_common` (step nav, progress, meal-plan card,
explanation panel), plus `attention_check`, `likert_scale`, `ranking_widget`, `recipe_card`,
`nutrition`, `kg_viz`, `query_inspector`, `correction_log`, `pareto_chart`.

---

## 15. Deployment & runtime

### 15.1 Two deployment shapes

| Mode | Where | Stores | Orchestration |
|---|---|---|---|
| **Local dev** | dev machine | Postgres+pgvector & Neo4j as sibling Docker containers | `docker-compose.yml`, auto-started by `stores_bootstrap.ensure_stores_up()` |
| **HF Docker Space** | single HF container | Postgres, Neo4j **and** Streamlit inside one container, `/data` persistent volume | `deploy/start.sh` (no docker-compose in-container) |

The full corpus (KG ≈1.9M nodes/9.5M rels; pgvector several GB) exceeds free managed tiers (Aura
Free 200k/400k, Neon Free 0.5 GB), so the Space self-hosts on the volume. Only a ~5 GB slice of the
54 GB `data/` tree is needed at runtime.

### 15.2 Runtime image (`Dockerfile`) & entrypoint

- **Base** `python:3.11-slim-bookworm`. System deps: `build-essential cmake git`,
  `openjdk-17-jre-headless`, **Postgres 16 + pgvector** (PGDG: `postgresql-16`,
  `postgresql-16-pgvector`, `postgresql-client-16`), **Neo4j 5.26.0 Community**
  (`dist.neo4j.org/neo4j-community-5.26.0-unix.tar.gz` → `/opt/neo4j`, APOC core jar copied to
  `plugins/`).
- **Python** `pip install -e .` + runtime clients `huggingface_hub>=0.24`, `psycopg2-binary>=2.9`,
  `neo4j>=5.14`, `streamlit`.
- **User** `uid 1000` (HF volume owner); baked ENV `APP_DIR=/app`, `DATA_DIR=/data`,
  `NEO4J_HOME=/opt/neo4j`, `HF_HOME=/data/cache/hf`, `PYTHONUNBUFFERED=1`; `EXPOSE 7860`;
  `CMD ["bash","deploy/start.sh"]`.
- **`deploy/start.sh`** (idempotent boot): export `FOODRAG_IN_CONTAINER=1` + store defaults →
  create `/data/{pg,neo4j,hummus}` + symlink `/app/data → /data` → `initdb` Postgres (scram-sha-256,
  first boot) + `CREATE EXTENSION vector` → write `neo4j.local.conf` (import dir `/data/neo4j_kg`,
  APOC) + start Neo4j (wait ≤120 s for Bolt) → run the three seed scripts (non-fatal) →
  `exec streamlit run app.py --server.port=${PORT:-7860} --server.address=0.0.0.0 --server.headless=true`.
- `.dockerignore` keeps the 54 GB `data/` out of the build context (pulled at boot instead); also
  excludes `.venv/`, `eval_results*/`, `.git/`, `notebooks/`, secrets.

### 15.3 Ports, environment variables, secrets

**Ports:** Streamlit **7860** (HF `$PORT`), Postgres/pgvector **5432**, Neo4j HTTP **7474** / Bolt
**7687**, NebulaGraph graphd **9669** (legacy). (Some experiment drivers pin Neo4j to 7478/7691.)

**Provider keys** (resolved via `models.yml`): `DEEPINFRA_API_KEY` (main LLM, embeddings, DeepSeek
judge), `OPENROUTER_API_KEY` (Gemma judge), `HF_TOKEN`/`HUGGING_FACE_HUB_TOKEN` (local embeddings +
model/dataset downloads), `NVIDIA_KEY` (optional NIM), `OLLAMA_BASE_URL`/`OLLAMA_API_KEY` (Pagoda
annotation).

**Store creds:** `POSTGRES_HOST/PORT/USER/PASSWORD/DB` (localhost/5432/postgres/postgres/rag_postgres),
`NEO4J_HTTP_PORT/BOLT_PORT/URI/USER/PASSWORD/DATABASE` (7474/7687/`bolt://localhost:7687`/neo4j/
neo4jfoodrag/neo4j), `NEBULA_HOST/PORT/USER/PASSWORD` (legacy).

**`FOODRAG_*` flags:** `FOODRAG_IN_CONTAINER` (in-container → skip compose), `FOODRAG_SKIP_STORES_BOOTSTRAP`
(disable bootstrap+warmup; auto-off under pytest), `FOODRAG_DATA_REPO` (HF Dataset of the runtime
slice), `FOODRAG_DATA_DIR` (target data dir), `FOODRAG_REAL_BACKEND` (force live Flow-A pipeline),
`FOODRAG_DATASET_REPO` (HF Dataset log sink; empty → in-memory), `FOODRAG_STUDY_CONFIG`,
`FOODRAG_EVAL_RESULTS` (Report-tab results dir), `FOODRAG_LLM_LOG`, `FOODRAG_METRICS`,
`FOODRAG_MEALPLAN_TIME`/`FOODRAG_MEALPLAN_DATE`. Other: `SUPABASE_URL`/`SUPABASE_KEY` (optional log
sink), `ADMIN_PASSWORD`.

**Minimum HF Space secrets** (`docs/DEPLOY_HF.md`): `DEEPINFRA_API_KEY` **or** `OPENROUTER_API_KEY`,
`HF_TOKEN`, `FOODRAG_DATA_REPO`, `POSTGRES_PASSWORD`, `NEO4J_PASSWORD`.

### 15.4 Store provisioning (`docker-compose.yml`)

| Service | Image | Ports | Volumes | Notes |
|---|---|---|---|---|
| `postgres` | `pgvector/pgvector:pg16` | `5432:5432` | `pgdata:/var/lib/postgresql/data` | `pg_isready` healthcheck |
| `neo4j` | `neo4j:5.26-community` | `7474`, `7687` | `neo4jdata:/data`, `./data/neo4j_kg:/var/lib/neo4j/import` | APOC (`NEO4J_PLUGINS=["apoc"]`, `apoc_import_file_enabled=true`) |
| `nebula-*` | `vesoft/nebula-*:v3.8.0` | `9669` | `nebula-meta`, `nebula-storage` | legacy, `--profile nebula` |

### 15.5 Data seeding at boot

- **`scripts/bootstrap_hf_data.py`** — if `FOODRAG_DATA_REPO` set and marker `data/.hf_data_seeded`
  absent, `snapshot_download(repo_type="dataset", allow_patterns=["hummus/**","embeddings/**","neo4j_kg/**"])`
  then writes the marker. Makes files present; does not seed DBs.
- **`scripts/seed_pgvector.py`** — `get_engine()` → `backend._vector_index()` → `_ensure_recipe_index()`:
  no-op if `hummus_recipes` non-empty (persisted via pgdata volume), else load precomputed
  `data/embeddings/*/recipe_documents/batch_*.pkl` and insert without recomputing embeddings.
- **`scripts/seed_neo4j.py`** — no-op if `MATCH (r:Recipe)` > 0, else
  `Neo4jManager.insert_data(mode=MODE_ALL, batch_size=1000)` (`LOAD CSV file:///all/...`).
- **`scripts/push_hf_data.py`** — maintainer-side producer: `upload_large_folder` of the `data/`
  slice (excludes `reviews_df.csv`; `--include-embeddings` ≈3.3 GB, `--include-kg` ≈0.9 GB).

Idempotency guards: `.hf_data_seeded`, `_pg_table_count>0`, `MATCH (r:Recipe)` count, `PG_VERSION`
existence — so every later boot starts in under a minute.

---

## 16. Reproducibility: run details, versions, seeds

**Common to all eval runs:** stratum = **tight** (`data/qa/main_10k_tight`, **n=1,646**); retrieval
`k=10`, meal-plan `k=5`; judges at **temperature 0**; generator temperature **0.1**; bootstrap CIs
10,000 resamples, seed 0; paired bootstrap seed 0. Results under `eval_results/` (LLM-retriever Llama
variants under `eval_results_llama/`). Resilience: single-instance mutex, per-item/per-run JSONL
checkpointing with resume-by-`question_id`, per-item `try/except → null record`, tenacity retry in
`LoggedLiteLLM`.

**Run counts / workers:**

| Family | Driver | Runs | Workers |
|---|---|---|---|
| Deterministic retrievers + composites | `run_composites.py`, `run_retrievers.py` | 1 | 6 |
| Popularity (0a/0b) + Naive-RAG (3) | baseline runners | 1 | — |
| LLM-only (2) | `run_llm_only.py` | 5 | 16 |
| Text-to-pandas (both backbones) | `run_llm_retrievers_llama.py`, `run_t2p_openai.py` | 5 | 16 (12 llama) |
| KG text-to-Cypher | `run_llm_retrievers*.py` | 5 | 8 |
| ToG (rung 1) | `run_tog.py` | 3 | 5 (over `main_10k_tog100`) |
| **FoodRAG H1/H2** (Use-2) | `run_foodrag_pipeline.py` | **3** | 16 (default 12) over full 1,646 |

**FoodRAG time/date:** slot reformulation uses a resolved time+date; per the results context H1 runs
`time=morning`, H2 `time=now`, `date=13.04.2025`; driver defaults `FOODRAG_MEALPLAN_TIME=now`,
`FOODRAG_MEALPLAN_DATE=today`. Both are **persisted per record** (`recommendation_time`,
`requested_meal_types`) so pinning them makes the requested meal set fully deterministic.

**Key pinned versions** (`.python-version`=**3.11**; `pyproject.toml` `requires-python >=3.11,<3.12`;
authoritative pins in `uv.lock`; validated exacts in `frozen_requirements.txt`):

| Package | Constraint | Frozen |
|---|---|---|
| `llama-index` / `-core` | `>=0.12.23,<0.13` / `>=0.12.37,<0.13` | 0.12.23 / 0.12.37 |
| `llama-index-vector-stores-postgres` | `>=0.4,<0.5` | 0.4.1 |
| `llama-index-graph-stores-neo4j` | `>=0.4,<0.5` | 0.4.6 |
| `llama-index-llms-litellm` / `-embeddings-litellm` | `>=0.3,<0.4` | 0.3.0 / 0.3.0 |
| `litellm` | `>=1.49,<2` | 1.49.7 |
| `streamlit` | `>=1.58,<2` | 1.58+ |
| `torch` (transitive) | — | 2.5.0 |
| `transformers` | `>=4.46,<4.49` | 4.48.0 |
| `sentence-transformers` | `>=3.2,<4` | 3.2.0 |
| `huggingface-hub` | `==0.26.0` | 0.26.0 |
| `psycopg2-binary` / `pgvector` / `asyncpg` | `2.9.10` / `0.3.6` / `0.29.0` | same |
| `sqlalchemy[asyncio]` | `==2.0.36` | 2.0.36 |
| `neo4j` driver | `==5.27.0` | 5.27.0 |
| `numpy` / `pandas` / `scipy` / `scikit-learn` | `1.26.4` / `2.2.3` / `1.13.1` / `1.5.2` | same |
| `spacy` | `==3.8.2` | 3.8.2 |
| `ragas` | `==0.2.1` | 0.2.1 |
| `plotly` | `>=5.20,<6` | 5.x |

**Store server versions:** Postgres 16 + pgvector 0.3.6 (image `pgvector/pgvector:pg16`; Space builds
`postgresql-16-pgvector`); Neo4j 5.26 Community (compose `neo4j:5.26-community`, Space tarball
5.26.0); Python neo4j driver 5.27.0 (compatible ≥ server minor). Nebula v3.8.0 (legacy). Extras
(`pyproject`): `nebula`, `local-llm` (`llama-cpp-python>=0.2.90`), `dev` (`pytest 8.3.3`,
`pytest-cov 5.0.0`, `ruff 0.6.9`). `[tool.uv] link-mode="copy"` (OneDrive hardlink workaround).

**Test suite:** ~40 files under `tests/` (`test_full_pipeline`, `test_distinct_slots`,
`test_safety_fixes`, `test_kbqa`, `test_pfoodreq_*`, `test_slot_normalized`, `test_ui_*`, …). The UI
tests set `FOODRAG_SKIP_STORES_BOOTSTRAP` to avoid Docker.

---

## 17. Repository map

```text
agentic-food-rag/
├── app.py                         # Streamlit entry point (routing + bootstrap)
├── config/                        # models.yml, nutrition_config.yml, guideline_reference_tables.json,
│                                  #   persona_config*.yml, data_manifest.yml
├── data/                          # HUMMUS, food_kg, embeddings, neo4j_kg, qa/ (gitignored; HF-seeded)
├── deploy/                        # start.sh (in-container boot)
├── docker-compose.yml · Dockerfile · .dockerignore
├── docs/                          # ARCHITECTURE.md (this) · DEPLOY_HF.md · RESULTS_REPORT.md
│                                  #   STATISTICAL_TESTS.md · ERROR_ANALYSIS.md
├── eval_results/ · eval_results_llama/   # per-question parquets, aggregate.csv, pairwise*.csv
├── scripts/                       # bootstrap_hf_data · push_hf_data · seed_pgvector · seed_neo4j ·
│                                  #   build_study_plans · create_embeddings · …
├── src/foodrag/
│   ├── baselines/                 # runner, full_pipeline, base, popularity, llm_only, naive_rag,
│   │   ├── constraints.py · metrics.py · meal_plan_metrics.py · judge.py · guideline_adherence.py
│   │   ├── explainability.py · slot_normalized.py
│   │   ├── kbqa/                  # tog, rog, tfs_kbqa, agentic, factory, linking, kg, adapters, training/
│   │   └── pfoodreq/              # BAMnet KBQA anchor
│   ├── config/ (settings, stores) · data/ (manifest loader)
│   ├── food_rag/
│   │   ├── agents/ (heuristic_agents, workflow_runner)
│   │   ├── query_engines/ (rag) · retriever/ (df base, db, pandas, kg, advanced, util, wrapper)
│   │   ├── tools/ (meal_plan, validation, generalization, substitution, ingredient,
│   │   │           explanation, finalization, …)
│   │   ├── meal_plan/ (meal_type, structures, json_encoder)
│   │   └── utils/ (recipe_context, hummus_docify, embedding_helper, *_helper)
│   ├── llm/ (logged_llm, retry, json_recovery, call_log, runstore, ollama_pagoda)
│   ├── persona/ (persona_generator, qa_generator, validator, restriction_matcher, user_data_extractor)
│   ├── hummus_transformation/ · analysis/ (stats, pareto) · pipeline/ (CLI drivers) · stores/
│   └── cli.py
├── ui/
│   ├── state.py · study_data.py · counterbalancing.py · admin_config.py · theme.py · report_copy.json
│   ├── screens/ (welcome, consent, hexad, persona, main, report, evaluation/*, admin/*)
│   ├── services/ (backend, rag_service, mapping, report_data, stores_bootstrap, study_results,
│   │              config_service, logging_service)
│   ├── components/ (pipeline_strip, recipe_detail, study_common, …)
│   └── study_plans/ (P1..P4.json, profiles.json)
├── third_party/pfoodreq/          # vendored BAMnet (Apache-2.0)
├── tests/ · notebooks/
└── pyproject.toml · uv.lock · frozen_requirements.txt · requirements.txt · .python-version
```

---

## 18. Appendix: end-to-end flow diagrams

### 18.1 H1 five-agent workflow

```text
                 user question + persona
                          │
                          ▼
        ┌───────────────────────────────────────┐
        │  RecommendationAgent (FunctionAgent)   │
        │  tool: MealPlanRecommendationTool      │
        │        (retrieve_only=True, time/date) │
        │  • reformulate → per-slot questions    │
        │  • resolve+persist recommendation_time │
        │  • retrieve top recipe per meal slot   │
        └───────────────┬───────────────────────┘
                        │ handoff → ValidationAgent
                        ▼
        ┌───────────────────────────────────────┐
        │  ValidationAgent — tool: CorrectionTool│
        │  • validate each slot                  │
        │  • 5-level generalization (allergens   │
        │    never dropped) re-recommend/validate│
        └──────┬───────────────────────┬─────────┘
     invalid? │ handoff                │ valid → handoff
               ▼                        ▼
   ┌───────────────────────┐          (skip)
   │  SubstitutionAgent    │
   │  tool: SubstitutionTool│  • enhance/repair (FoodOn)
   └───────────┬───────────┘
               │ handoff → ExplanationAgent
               ▼
   ┌───────────────────────┐
   │  ExplanationAgent     │  → MEAL_PLAN_EXPLANATION_KEY
   │  tool: ExplanationTool │    (empty-plan gate: fixed msg, no LLM)
   └───────────┬───────────┘
               │ handoff → FinalizationAgent
               ▼
   ┌───────────────────────┐
   │  FinalizationAgent    │  → FINAL_RECOMMENDATION_KEY
   └───────────┬───────────┘
               ▼
   final RecipeState  ──►  AssembledPlan (plan + explanation + usage)
     └─ deterministic post-filters: enforce_allergen_safety → enforce_distinct_slots
```

### 18.2 retrieve → assemble → score (Use-2)

```text
qa_item (question, persona, answer=gold, meal_plan)
    │
    ▼
retriever.retrieve(query) ── DataFrameRetrieverBase ──► candidate recipes (pd.DataFrame)
   (dense pgvector | sparse | hybrid | fulltext | text-to-pandas |
    KG text-to-Cypher | fusion/intersection/majority | replay)
    │
    ▼
assemble:
   H2 → RagMealPlanAssembler(retrieve_only=True)   → AssembledPlan (no explanation)
   H1 → WorkflowMealPlanAssembler(AgentWorkflow)   → AssembledPlan (+ explanation)
   └─ enforce_allergen_safety → enforce_distinct_slots  (deterministic guards)
    │
    ▼
score (per §13.1 record):
   • deterministic  : metrics.deterministic_metrics(plan recipes vs gold, persona)
   • plan quality   : meal_plan_metrics.plan_quality_metrics(plan)
   • guideline      : guideline_adherence(plan, persona)  → UL breach → hard_constraint_violation=1.0
   • explainability : explainability_metrics(persona, recipes, explanation_text)
   • judges         : judge_plan(judgeA=gemma, judgeB=deepseek-v3)   # temperature 0
   • usage/latency  : llm_usage, llm_seq_usage, llm_parsing, latency_s
    │
    ▼
checkpoint (JSONL, resume by question_id) ──► aggregate() ──► write_results()
   bootstrap 95% CI (10k resamples, seed 0)      eval_results/{config}_per_question.parquet
   per-question variance · paired Wilcoxon        eval_results/aggregate.csv (§13.2) · pairwise.csv
```

---

*Grounded in the code as of branch `refactor/uv-loader`. Model ids, versions and hyperparameters are
quoted from `config/models.yml`, `pyproject.toml`, `frozen_requirements.txt`, `docker-compose.yml`
and the `Dockerfile`. Generated as a read-only architecture description; no source was modified.*

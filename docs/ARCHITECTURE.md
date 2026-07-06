# FoodRAG — Architecture

> A health-aware meal-plan recommender over the HUMMUS recipe knowledge graph, with a
> retrieval layer, a five-agent agentic meal-plan pipeline, and a deterministic + dual-LLM-judge
> evaluation framework.
>
> This document describes the system as implemented in `src/foodrag/` and driven by the
> experiment runners in `c:/tmp/`. It is grounded in the code; file paths and line numbers are
> cited throughout. Model ids are taken verbatim from `config/models.yml`.

## Table of contents

1. [System overview](#1-system-overview)
2. [Data](#2-data)
3. [Retrievers](#3-retrievers)
4. [Agents & tools (the H1 chain)](#4-agents--tools-the-h1-chain)
5. [Query engine](#5-query-engine)
6. [Models, providers, configuration](#6-models-providers-configuration)
7. [Evaluation framework](#7-evaluation-framework)
8. [Metric catalogue](#8-metric-catalogue)
9. [Run implementation details](#9-run-implementation-details)
10. [Appendix: end-to-end flow diagrams](#10-appendix-end-to-end-flow-diagrams)

---

## 1. System overview

FoodRAG recommends **meal plans** (breakfast / lunch / dinner / snack) over the HUMMUS recipe
corpus, honouring a user *persona*'s dietary constraints (allergens, nutrient bounds, health
scores, liked/disliked tags and ingredients) and health guidelines (EFSA/WHO nutrient bands).

There are **two use modes** (the "dual-use" design, `specs/FoodRAG_Evaluation_Protocol.md` §5.7):

- **Use-1 — retrieval-only.** A retriever produces a ranked list of recipes; the list is scored
  directly against the persona's gold answer set and constraints. Driven by
  `foodrag.baselines.runner.run_baseline` (`src/foodrag/baselines/runner.py:47`) wrapping a
  `Baseline` such as `NaiveRAG` (`src/foodrag/baselines/naive_rag.py:21`).
- **Use-2 — full meal-plan pipeline.** FoodRAG's meal-plan *assembly layer* is attached on top
  of the retrieval and the resulting **assembled plan** is scored end-to-end (deterministic
  metrics + dual LLM judges). Driven by `foodrag.baselines.full_pipeline.run_full_pipeline`
  (`src/foodrag/baselines/full_pipeline.py:89`).

Within Use-2 there are **two heuristics**:

- **H1 — agentic Recommend + Correct + Enhance.** A five-agent workflow
  (Recommend → Validate/Correct → Substitute/Enhance → Explain → Finalize) built by
  `build_heuristic_workflow_builder` (`src/foodrag/baselines/kbqa/agentic.py:25`) and run through
  `WorkflowMealPlanAssembler` (`src/foodrag/baselines/full_pipeline.py:311`). Emits a plan **plus
  an NL explanation** (`require_explanation = True`).
- **H2 — Recommend-only.** The meal-plan recommendation tool alone, in `retrieve_only` mode, via
  `RagMealPlanAssembler(retrieve_only=True, require_explanation=False)`
  (`src/foodrag/baselines/full_pipeline.py:185`). No validation/correction/explanation.

The H1-vs-H2 delta isolates what the agentic correction + enhancement + explanation layer adds
over a plain recommend-only plan built from the same retrieval.

**Storage & serving backends** (see §6): recipes live as a Pandas `recipes_df` plus a **pgvector**
dense index (Postgres, `rag_postgres`); a **Neo4j** graph (bolt `7691`) backs the text-to-Cypher
retriever; all LLM/embedding calls go through **LiteLLM** behind a logging + retry wrapper.

---

## 2. Data

### 2.1 HUMMUS recipe corpus (`recipes_df`)

The working recipe table is `data/hummus/recipes_df.csv`, loaded by
`foodrag.baselines.runner.load_recipes_df` (`src/foodrag/baselines/runner.py:32`), which reads the
CSV and deserialises the list columns `tags` and `ingredient_names` via `load_util.serialize`
so constraint predicates see real lists. The larger `~7GB` `hummus_data` object (ingredients,
reviews, users, FoodOn) is loaded once per process by `load_util.get_current_hummus("data")`
and used only by tools that need it (e.g. the explanation and ingredient-substitution tools).

`recipes_df` has **35 columns** (verified from the CSV header):

| Group | Columns |
| --- | --- |
| Identity | `recipe_id`, `title`, `description`, `author_id`, `recipe_url`, `food_kg_locator`, `last_changed_date` |
| Structure | `duration`, `directions`, `ingredients`, `ingredient_names`, `ingredients_sizes`, `direction_size`, `serves`, `servingsPerRecipe`, `servingSize [g]`, `tags`, `meal_types` |
| Ratings | `average_rating`, `number_of_ratings` |
| Nutrients | `calories [kcal]`, `caloriesFromFat [kcal]`, `totalFat [g]`, `saturatedFat [g]`, `cholesterol [mg]`, `sodium [mg]`, `totalCarbohydrate [g]`, `dietaryFiber [g]`, `sugars [g]`, `protein [g]` |
| Health scores (normalised 0–1) | `who_score`, `fsa_score`, `nutri_score` |
| Misc | `Unnamed: 0`, `normalization_comment` |

Notable columns for the system: `nutri_score` / `fsa_score` (health metrics, §8), `meal_types`
(slot assignment, §5), `tags` and `ingredient_names` (constraint matching), `average_rating`
(popularity / relevance ranking).

The canonical id is **`recipe_url`** (`ID_COL` in both `runner.py:24` and `full_pipeline.py:38`);
gold answer sets and retrieved lists are both keyed by `recipe_url`.

### 2.2 FoodOn ontology

The FoodOn RDF ontology (`data/hummus/food_on_ontology.ttl`, with FoodKG mappings) is wrapped by
`rdf_util.PreProcessedFoodOn`. It provides ingredient hierarchy so a persona preference such as
"vegetable" expands to any descendant ingredient. It is used in two places:

- **QA generation** — answer filtering translates persona ingredients through FoodOn super-classes
  (`src/foodrag/persona/qa_generator.py`).
- **Ingredient substitution tool** — `IngredientSubstitutionTool`
  (`src/foodrag/food_rag/tools/ingredient_tools.py`) bridges food.com ↔ USDA ↔ FoodOn to propose
  alternative ingredients, using `PreProcessedFoodOn(hummus_data)`.
- **Constraint evaluation** — `constraints.foodon_expander`
  (`src/foodrag/baselines/constraints.py`) provides a FoodOn-aware `IngredientExpander` so
  allergen/blacklist checks match descendants, not just exact names.

### 2.3 QA set (`main_10k_tight`)

The evaluation set is `data/qa/main_10k_tight/recipe_question_answer_set.pkl`, a pickled list of
**1,646** QA items (verified). Each item is a dict with keys:

```
question       # natural-language request
llm_questions  # optional LLM-reformulated variants
answer          # gold answer set: list of recipe_url (the "valid recipes")
persona        # the dietary-preference structure the constraints/metrics read
meal_plan      # per-meal-type valid-recipe sets (for plan-level gold)
```

**Persona structure** (produced by `foodrag.persona.persona_generator`,
`src/foodrag/persona/persona_generator.py`):

```
persona = {
  'tags':              {'likes': [...], 'dislikes': [...]},
  'ingredients':       {'likes': [...], 'dislikes': [...]},   # dislikes ≈ allergens/blacklist
  'relative_nutrients':{nutrient: {'preference': 'low|medium|high',
                                   'limits': {'lower': X, 'upper': Y}}},   # per-100g band
  'total_nutrients':   {nutrient: max_value},                              # per-day ceiling
  'nutrition_scores':  {score_name: min_value},                           # who/fsa/nutri floor
}
```

**Gold answers (valid recipes).** For each persona, `qa_generator` computes the set of recipes
that satisfy **all** persona constraints (relative nutrients per 100g → nutrition-score floors →
total-nutrient ceilings → liked/disliked tags → FoodOn-aware liked/disliked ingredients). That
set is the item's `answer`.

**Tight stratum.** `main_10k_tight` selects only personas whose valid-answer count falls in
**1–50** (verified: min answer size 1, max 50). This is the *high-strictness* stratum — persona
constraints narrow the corpus to a small feasible set, making retrieval and constraint metrics
discriminative. Answer-set strata are defined in `foodrag.baselines.base.answer_set_stratum`
(0 → `empty`, 1–50 → `tight`, larger → `medium`/`broad`).

Other QA directories exist for other slices: `main_10k`, `main_10k_s01`…`s10` (stratified),
`main_10k_tog100` (the 100-question sample used for ToG, §9), and `calib`.

---

## 3. Retrievers

All FoodRAG retrievers implement a single interface,
**`DataFrameRetrieverBase`** (`src/foodrag/food_rag/retriever/df_retriever_base.py:6`):

```python
class DataFrameRetrieverBase(ABC):
    @abstractmethod
    def retrieve(self, str_or_query_bundle: QueryType) -> pd.DataFrame: ...
    @abstractmethod
    async def aretrieve(self, str_or_query_bundle: QueryType) -> pd.DataFrame: ...
```

i.e. `query → pd.DataFrame of recipe rows` (columns are the full `recipes_df` schema). This is
what lets every retriever plug into `RecipeRAGQueryEngine` (§5) and `NaiveRAG` (Use-1)
interchangeably. `NaiveRAG.recommend` (`src/foodrag/baselines/naive_rag.py:35`) simply calls
`retriever.retrieve(question).head(k)["recipe_url"]`.

### 3.1 Dense / Sparse / Hybrid / Full-text — `RecipeRetriever`

`src/foodrag/food_rag/retriever/db_retriever.py:9` —
`RecipeRetriever(VectorIndexRetriever, DataFrameRetrieverBase)`.

- **Mechanism.** Wraps a LlamaIndex `VectorIndexRetriever` over the **pgvector** dense index
  (`hummus_recipes`, dim 1024, cosine). The retrieval *mode* is selected by the
  `vector_store_query_mode` constructor argument:
  - `"default"` → **dense** cosine similarity over BGE-M3 embeddings;
  - `"sparse"` → **sparse/BM25** keyword retrieval;
  - `"hybrid"` → dense + sparse combined (RRF at the store level);
  - `"text_search"` → **full-text** search.
- **I/O.** Query string / `QueryBundle` → `pd.DataFrame` of the matched recipe rows in similarity
  order. Retrieves nodes, extracts unique `recipe_id`s from node metadata, and returns the
  corresponding `recipes_df` rows preserving rank.
- **Key params.** `index`, `recipes_df`, `vector_store_query_mode="default"`,
  `similarity_top_k=10`, `split_node_ratio=2`.
- **Invocation.** Constructed in every runner, e.g.
  `RecipeRetriever(index, recipes_df, vector_store_query_mode="default", similarity_top_k=K)`
  (`c:/tmp/run_foodrag_pipeline.py:190`). The four modes give configs `dense`, `sparse`, `hybrid`,
  `fulltext`.

### 3.2 Text-to-Pandas — `PandasQueryRetriever`

`src/foodrag/food_rag/retriever/pandas_retriever.py:173` —
`PandasQueryRetriever(PandasQueryEngine, BaseRetriever, DataFrameRetrieverBase)`.

- **Mechanism.** An LLM is prompted (`PANDAS_PROMPT_TEMPLATE` + `INSTRUCTION_STR` + a column
  description dict) to emit an **executable pandas expression** whose final line is `eval()`-able
  and returns a DataFrame with *all original columns*. The expression **both filters and ranks**
  (`df[df[...] > x].sort_values(...)`), so text-to-pandas is a single-shot LLM retriever that does
  constraint filtering and ordering in one code generation. Output is parsed via
  `default_output_processor` (safe-exec of setup lines, safe-eval of the final expression, with a
  `custom_eval_pandas` fallback), then `.head(similarity_top_k)`.
- **I/O.** Query → filtered+ranked `pd.DataFrame`.
- **Key params.** `df`, `df_column_description_dict`, `pandas_prompt_template`, `instruction_str`,
  `similarity_top_k=10`.
- **Invocation.** Built in the LLM-retriever drivers, e.g. `c:/tmp/run_llm_retrievers_llama.py:59`
  and `c:/tmp/run_t2p_openai.py:40`, wrapped in `NaiveRAG` for Use-1. Before construction the
  drivers stringify list columns (`tags`, `ingredient_names`, `directions`, `meal_types`) so the
  generated pandas is string-friendly.

### 3.3 KG text-to-Cypher — `KGQueryEngine`

`src/foodrag/food_rag/retriever/kg_retriever.py:57` —
`KGQueryEngine(KnowledgeGraphQueryEngine, BaseRetriever, DataFrameRetrieverBase)`.

- **Mechanism.** An LLM generates **Cypher** from the natural-language query using
  `DEFAULT_GRAPH_QUERY_SYNTHESIS_PROMPT`, which is templated with the live graph schema
  (`node_types`, `node_features`, `relation_types`, `relation_features`) and instructs the model
  to return `recipe_url`s, normalise scores 0–1 (unhealthy <0.3, neutral ≥0.3, healthy ≥0.5),
  express nutrient ranges as two conditions, and default to `nutri_score`. The Cypher is executed
  against **Neo4j** via the injected `StorageContext(graph_store=Neo4jGraphStore(...))`.
- **I/O.** Query → `pd.DataFrame` of recipes matching the graph query, order preserved (for MRR).
- **Neo4j connection.** From `.env`: `NEO4J_URI = bolt://localhost:7691`, `NEO4J_USER`,
  `NEO4J_PASSWORD` (port 7691 chosen to avoid clashing with other local Neo4j containers; default
  in `stores.neo4j_config` is 7687). Schema is assembled in the drivers from the exported
  `data/neo4j_kg/all/node_*.csv` / `relation_*.csv` metadata (`c:/tmp/run_llm_retrievers*.py`).

### 3.4 Composite retrievers — `advanced_retriever.py`

`src/foodrag/food_rag/retriever/advanced_retriever.py`:

- **`DataFrameFusionRetriever(QueryFusionRetriever, ...)`** (line 13) — **Reciprocal Rank Fusion**
  over multiple child retrievers. De-duplicates and re-ranks by RRF score; per-retriever failures
  are caught and treated as empty. Config `ret_fusion` fuses dense + sparse with `llm=None`
  (`c:/tmp/run_composites.py:61`).
- **`IntersectionRetriever(...)`** (line 114) — returns only recipes present in **both** child
  retrievers' results (`set(df1.recipe_id) ∩ set(df2.recipe_id)`, filtered from `df1` to preserve
  order). Config `ret_intersection` = intersection(dense, sparse).
- **`MajorityIntersectionRetriever(...)`** (line 182) — **weighted majority vote** across ≥2
  retrievers; keeps recipes whose summed vote weight ≥ `total_weight / 2`. Config `ret_majority`
  votes over dense + sparse + hybrid + fulltext.

### 3.5 Utility retrievers — `util_retriever.py`

`src/foodrag/food_rag/retriever/util_retriever.py`: `RandomRetriever` (random-sample k nodes,
score 1.0 or random — the random baseline), `Doc2VecRetriever` (Doc2Vec cosine),
`BoWCosineRetriever` (bag-of-words cosine). `WrapperNodeRecipeRetriever`
(`wrapper_retriever.py:10`) adapts any node-returning `BaseRetriever` to the DataFrame interface.

### 3.6 Replay retrievers (driver-side)

Defined **in the driver** `c:/tmp/run_foodrag_pipeline.py`, not in the package, and used only to
replay saved text-to-pandas retrievals into the FoodRAG configs so no live re-retrieval happens:

- **`ReplayRetriever`** (`c:/tmp/run_foodrag_pipeline.py:283`) — a `DataFrameRetrieverBase`-shaped
  retriever that **ignores the query** and always returns the recipes for a fixed pre-saved
  `recipe_url` list. Used by the H2-t2p config (`ReplayH2Assembler`) so `retrieve_only` assembly
  consumes the saved t2p set. One instance per item (thread-isolated).
- **`RotatingReplayRetriever`** (line 314) — the same, but stateful: each `retrieve()` call
  advances a cursor and returns the pool **rotated** so a new top recipe leads while rank order is
  preserved (`[u0,u1,u2] → [u1,u2,u0] → …`). This is needed for the **live H1-t2p** path, where
  the meal-plan tool calls `retrieve` once per slot and takes `.head(1)`; without rotation every
  slot would pick the same recipe and the plan would collapse to one dish.

---

## 4. Agents & tools (the H1 chain)

### 4.1 The five-agent workflow

H1 is a **five-agent `AgentWorkflow`** assembled by `build_heuristic_workflow_builder`
(`src/foodrag/baselines/kbqa/agentic.py:25`). Every agent is a **llama-index `FunctionAgent`**
(native tool calling), composed with **handoffs** (`can_handoff_to`) and a **shared `RecipeState`**
initial state:

```python
# src/foodrag/baselines/kbqa/agentic.py:48
recommendation_agent = RecommendationAgent(query_engine=query_engine, llm=llm, time=time, date=date)
validation_agent     = ValidationAgent(llm=llm, query_engine=query_engine, recipe_df=recipes_df)
substitution_agent   = SubstitutionAgent(llm=llm)
explanation_agent    = ExplanationAgent(llm=llm, hummus_data=hummus_data)
finalization_agent   = FinalizationAgent(llm=llm)
return AgentWorkflow(
    agents=[recommendation_agent, validation_agent, substitution_agent,
            explanation_agent, finalization_agent],
    root_agent=recommendation_agent.name,
    initial_state=RecipeState.get_init_state(),
)
```

Agent classes and handoffs (`src/foodrag/food_rag/agents/heuristic_agents.py`):

| # | Agent | Tool it wraps | `can_handoff_to` | Role |
| --- | --- | --- | --- | --- |
| 1 | **RecommendationAgent** (`:110`) | `MealPlanRecommendationTool(retrieve_only=True, time, date)` | `["ValidationAgent"]` | Produce an initial meal plan; MUST call the tool (no answering from own knowledge). |
| 2 | **ValidationAgent** (`:173`) | `CorrectionTool` | `["ExplanationAgent", "SubstitutionAgent"]` | Validate & correct the plan; hand off to Substitution if still invalid, else to Explanation. |
| 3 | **SubstitutionAgent** (`:221`) | `SubstitutionTool` | `["ExplanationAgent"]` | Enhance/repair recipes (e.g. "healthier"), then hand off to Explanation. |
| 4 | **ExplanationAgent** (`:264`) | `ExplanationTool` | `["FinalizationAgent"]` | Write the NL "why this plan fits" explanation. |
| 5 | **FinalizationAgent** (`:309`) | `FinalizeRecommendationTool` | `[]` (terminal) | Assemble final (plan + explanation) exactly once, then stop. |

### 4.2 Shared state — `RecipeState`

`src/foodrag/food_rag/utils/recipe_context.py`. `RecipeState` wraps the workflow `Context`'s
`"state"` dict and exposes typed getters/setters. Key names include:

| Key | Holds |
| --- | --- |
| `ORIGINAL_USER_QUESTION_KEY` | original question |
| `MEAL_PLAN_QUESTIONS_KEY` | `MealPlanQuestions`: `{meal_type → per-slot reformulated question}` (keys = requested meal types) |
| `MEAL_PLAN_RECOMMENDATION_KEY` | `MealPlan`: `{meal_type → recipe dict}` |
| `MEAL_PLAN_VALIDATION_KEY` / `OVERALL_MEAL_PLAN_VALIDATION_KEY` | per-slot / overall validation verdicts |
| `SUBSTITUTED_MEAL_PLAN_RECOMMENDATION_KEY` | modified recipes from the substitution tool |
| `RECOMMENDATION_TIME_KEY` | the resolved `"{date} {time}"` used for slot reformulation (persisted for reproducibility) |
| `MEAL_PLAN_EXPLANATION_KEY` | the NL explanation |
| `FINAL_RECOMMENDATION_KEY` | final assembled output |
| `HISTORY_KEY` | tool-execution event log |
| `LLM_USAGE_COUNTER_KEY`, `LLM_SEQUENTIAL_USAGE_COUNTER_KEY`, `LLM_PARSING_USAGE_COUNTER_KEY` | LLM-call counters |

`MealPlan` / `MealPlanQuestions` / `MealType` live in
`src/foodrag/food_rag/meal_plan/` (`MealType` is the `{breakfast,lunch,dinner,snack}` enum,
`meal_type.py:4`).

### 4.3 The tools

- **`MealPlanRecommendationTool`** (`src/foodrag/food_rag/tools/meal_plan_tools.py:40`).
  `recommend_meal_plan()` first **reformulates** the user query into per-meal-type questions
  (`get_meal_plan_question`), resolving the current *time* and *date*: if `time`/`date` are
  `"now"`/`"today"` it uses `datetime.now()`, else the pinned value, and **persists** the resolved
  `"{date} {time}"` into `RECOMMENDATION_TIME_KEY`. The LLM then decides which of the four meals
  are relevant given that time. `get_meal_plan()` retrieves per slot: with `retrieve_only=True`
  (H1's RecommendationAgent and H2) it takes the retriever's top recipe per slot; with
  `retrieve_only=False` the query-engine LLM picks from the retrieved set. Two tool names:
  `meal_plan_recommendation_tool` (LLM-select) vs `meal_plan_alternatives_tool` (retrieve_only).
- **`CorrectionTool`** (`src/foodrag/food_rag/tools/validation_tools.py:346`). `correct_recipes()`
  validates each slot, then for invalid slots runs a **5-level generalization loop** (via
  `generalization_tools.py`): L1 keep everything → L2 generalize nutrient bounds to high/med/low →
  L3 drop nutrients (keep name + ingredients) → L4 drop specific ingredients → L5 keep recipe name
  only. After each level it re-recommends and re-validates; if a slot stays invalid after L5 the
  original is restored.
- **`SubstitutionTool`** (`src/foodrag/food_rag/tools/substitution_tools.py:12`) and
  **`IngredientSubstitutionTool`** (`ingredient_tools.py:25`). The former asks the LLM to modify a
  recipe toward objectives (e.g. "healthier") while preserving identity; the latter proposes
  alternative ingredients using the **FoodOn** ontology + USDA nutrition, with retrieval modes
  (EMBEDDING / DOC_2_VEC / BOW_COSINE).
- **`ExplanationTool`** (`explanation_tools.py:14`). `explain()` writes a concise user-level
  rationale for *why* the plan fits the request (not how it was built) into
  `MEAL_PLAN_EXPLANATION_KEY`. **Empty-plan gate (commit 54a62ae):** when the recommended plan is
  empty (all slots `None`), `explain()` short-circuits and returns a fixed `EMPTY_PLAN_MESSAGE`
  ("No meal plan could be recommended that satisfies the given requirements") with **no LLM call**,
  instead of letting the Enhance/Explanation agent narrate recipes that were never selected. This
  removed the fully-hallucinated empty-plan explanations that had dominated the H1-t2p grounding
  failures (see `docs/ERROR_ANALYSIS.md` §6: ucr=1.0 15.4%→7.3%, faithfulness 0.922→0.997).
- **`FinalizeRecommendationTool`** (`finalization_tool.py:26`). `finalize()` assembles
  explanation + plan into `FINAL_RECOMMENDATION_KEY`, in `MODE_SIMPLE` (concatenate) or
  `MODE_LLM_REFINED` (LLM reformulation).

### 4.4 How the workflow is invoked

- **H1 (live).** `WorkflowMealPlanAssembler.assemble` (`full_pipeline.py:338`) binds the retriever
  to the item, runs the async `AgentWorkflow` (`_default_workflow_run`, `:386`), then reads the
  final `RecipeState` into an `AssembledPlan` (plan, `explanation_text`, `explanation_structured`
  = per-slot validation verdicts, `recommendation_time`, `requested_meal_types`, LLM-usage
  counters). In the resilient driver this is wrapped by `LiveH1Assembler`
  (`c:/tmp/run_foodrag_pipeline.py:468`), which builds a *fresh workflow per item* (thread
  isolation) and, for the t2p variant, hands the RecommendationAgent a `RotatingReplayRetriever`.
- **H2 (retrieve-only).** `RagMealPlanAssembler` (`full_pipeline.py:185`) constructs a
  `RecipeRAGQueryEngine(return_meal_plan=True)` and a `MealPlanRecommendationTool(retrieve_only=True)`
  and calls `recommend_meal_plan` once — no agents, no explanation.
- **Deterministic allergen post-filter (commit 54a62ae).** Both assemblers apply
  `constraints.enforce_allergen_safety(plan, persona)` (`constraints.py`) to the **final** plan as
  a hard post-filter (`RagMealPlanAssembler` for H2, `WorkflowMealPlanAssembler` for H1): it drops
  any meal slot whose recipe's ingredient list intersects the persona's disliked/allergen
  (blacklisted) set, guaranteeing **zero allergen violations regardless of retrieval or LLM
  validation**. Because a blacklist-violating recipe can never be the gold answer, only unsafe
  non-gold slots are removed and hit@10 is unaffected. This eliminated the residual allergen
  liability across all four FoodRAG configs (allergen_violation → 0.000; see
  `docs/ERROR_ANALYSIS.md` §4, including the q_01495 green-onion case).

### 4.5 Why the chain works — the commit-`80b7a03` fix

The five-agent workflow originally produced **100% empty plans** with Llama-3.3-70B. Commit
`80b7a03` ("Fix agentic H1 workflow: convert ReAct→FunctionAgent, fix state unwrap + LiteLLM
tool-calls") fixed three root causes:

1. **State-unwrap fix** (`recipe_context.py`). `RecipeState.from_context` assumed the context
   `"state"` was `[dict]` and indexed `[0]`; `AgentWorkflow` stores a **bare dict**, so every tool
   hit `KeyError: 0`, caught it, and silently dropped all state writes. `_unwrap_state()` now
   accepts **both** shapes and `save_to_context` preserves the existing wrapping.
2. **ReAct → FunctionAgent** (`heuristic_agents.py`). Text ReAct (`ReActChatFormatter`,
   `observation_role`) was unreliable for tool emission/handoffs with the 70B model. The five
   agents were converted to **`FunctionAgent`** (native tool calling) and the ReAct plumbing
   dropped.
3. **LiteLLM tool-calls** (`logged_llm.py`, see §6). Force `is_function_calling_model=True` and
   route `astream_chat_with_tools` through the non-streaming path so modern `tool_calls` deltas
   are not dropped.

After the fix the live workflow yields non-empty, fully-explained plans end-to-end (validated at
WORKERS 1 and 8, on dense and t2p). The driver's `SeededH1Assembler`
(`c:/tmp/run_foodrag_pipeline.py:353`) remains as a deterministic *fallback* that seeds the plan
from H2 and runs Correction → Explanation → Finalization directly; `LiveH1Assembler` is the
default (`FOODRAG_H1_MODE=live`).

---

## 5. Query engine

**`RecipeRAGQueryEngine`** (`src/foodrag/food_rag/query_engines/rag.py:53`) is both a LlamaIndex
`BaseQueryEngine` and a `DataFrameRetrieverBase`. It has two paths:

- **`retrieve(query)`** (`:272`) — delegates straight to the wrapped retriever, returning the
  DataFrame (used by tools that only need candidates).
- **`_query(query)`** (`:107`) — the RAG path: (1) if `return_meal_plan`, rewrite "recipe(s)" →
  "meal plan" in the query; (2) `retriever.retrieve(query)`; (3) optionally rank by
  `ranking_strategy="rating"` (`sort_values("average_rating")`); (4) select the top-`k` — for meal
  plans via `_get_first_k_per_meal_type` which fills `breakfast/lunch/dinner/snack` slots by each
  recipe's `meal_types` with fallback fillers (`:207`); (5) serialise the selected recipes to text
  (`hummus_docify.recipes_to_docs`) and prompt the LLM with `DEFAULT_TEMPLATE` (single recipe) or
  `MEAL_PLAN_TEMPLATE` (four slots as JSON); (6) if `return_full_recipes`, resolve the LLM's chosen
  titles back to full recipe docs (`_get_full_recipe_str`).

Constructor knobs: `retriever`, `llm`, `ranking_strategy` (`None` | `"rating"`), `k` (default 5),
`prompt_template`, `llm_calls`, `return_full_recipes`, `return_meal_plan`, `debug`. In Use-2 it is
built with `return_meal_plan=True` and (in `RagMealPlanAssembler`) `ranking_strategy="rating"`,
`k=5`.

---

## 6. Models, providers, configuration

All model ids, endpoints and dims are centralised in **`config/models.yml`** and turned into
LlamaIndex objects by **`src/foodrag/config/settings.py`** (`load_models_config`, `build_llm`,
`build_embedding`, `build_judge`). Keys are read from `.env` by the names in
`providers[*].api_key_env`. Everything OpenAI-compatible is routed through **LiteLLM** with the
model id prefixed `openai/` and `api_base` pointed at the provider.

**Verified model ids (from `config/models.yml`):**

| Role | Model id | Provider | Notes |
| --- | --- | --- | --- |
| Generator / agent LLM | `meta-llama/Llama-3.3-70B-Instruct` | **DeepInfra** | `temperature: 0.1`, `max_retries: 5`, `request_timeout: 120` (`models.yml` `llm:`) |
| Dense embedding | `BAAI/bge-m3` | **DeepInfra** | `dim: 1024`, `chunk_size: 512`; winner of `bench-embeddings`, matches hybrid dense+sparse design (`models.yml` `embedding:`) |
| Judge A (headline, cross-lineage) | `google/gemma-3-27b-it` | **OpenRouter** | logical label `judgeA` (`build_judges` default, `judge.py:91`) |
| Judge B (same-family default) | `deepseek-ai/DeepSeek-V3` | **DeepInfra** | `judges.default: deepseek-v3` |
| Text-to-pandas backbone A | `meta-llama/Llama-3.3-70B-Instruct` | DeepInfra | `ret_text2pandas@llama70b` (`run_llm_retrievers_llama.py`) |
| Text-to-pandas backbone B | `gpt-4o-mini` | **OpenAI** | `ret_text2pandas@gpt4o-mini` (`run_t2p_openai.py`) |
| Meal-type annotation (bulk) | `llama3.3:70b` | **Pagoda** (Ollama) | free/slow bulk pass, `annotation:` block |
| Embedding candidates | `Qwen/Qwen3-Embedding-4B` (2560d), `BAAI/bge-large-en-v1.5` (1024d), `sentence-transformers/all-mpnet-base-v2` (768d, local HF) | mixed | compared by `bench-embeddings` |

**Providers** (`models.yml` `providers:`): `deepinfra`, `openrouter`, `nvidia` (NIM),
`pagoda` (Ollama on lab server), `huggingface` (local embeddings). All OpenAI-compatible ones go
through LiteLLM.

**Judge ≠ generator is enforced in code.** `build_judge` (`settings.py:199`) raises `ValueError`
if a judge's model equals `llm.model`, so same-model self-grading cannot happen; Judge A (Gemma)
is deliberately cross-lineage and Judge B (DeepSeek-V3) differs from the Llama generator, making
the same-family-bias delta measurable.

### 6.1 `LoggedLiteLLM`

`src/foodrag/llm/logged_llm.py:34` — a `LiteLLM` subclass used by default (`build_llm(logged=True)`):

- **JSONL call logging.** `complete`/`chat` are wrapped in `call_log.record_call(...)`
  (`src/foodrag/llm/call_log.py`), which times the call and appends a `CallRecord`
  (kind, model, request_id, prompt/response truncated, latency, prompt/completion/total tokens,
  estimated `cost_usd`, attempt, error, ts) to `data/logs/llm_calls-<date>.jsonl`.
- **Tenacity retry.** Each call runs under `@llm_retry(max_attempts=6, initial=2.0, max_wait=60.0)`
  (`src/foodrag/llm/retry.py`), exponential backoff + jitter, retrying transient provider errors
  (`RateLimitError`, `Timeout`, `APIConnectionError`, 5xx, …).
- **Forced function calling.** The `metadata` property forces
  `is_function_calling_model = True` (LiteLLM's static DB wrongly reports DeepInfra Llama-3.3 as
  non-tool-calling, which stripped tools).
- **Non-streaming tool-call fix.** `astream_chat_with_tools` is routed through the non-streaming
  `achat` path, because upstream streaming only aggregates the legacy `function_call` delta and
  drops modern `tool_calls` deltas — the direct cause of the empty-plan failure (§4.5).

Two related helpers: `runstore.py` (JSONL resumable item processing) and `json_recovery.py`
(robust JSON extraction ladder: `json.loads` → `ast.literal_eval` → repairs → fenced/balanced
slice) used to parse messy LLM output.

### 6.2 Stores

`src/foodrag/config/stores.py` reads credentials from `.env` (defaults match `docker-compose.yml`):

- **pgvector / Postgres** — `POSTGRES_HOST=localhost`, `POSTGRES_PORT=5432`, db `rag_postgres`;
  the dense index `hummus_recipes` (embed dim 1024) is opened via
  `get_vector_store_provider("pgvector").index("hummus_recipes", embed_dim=...)`.
- **Neo4j** — `NEO4J_URI=bolt://localhost:7691` (HTTP 7478), user `neo4j`; backs the text-to-Cypher
  retriever. Ports 7478/7691 were chosen to avoid other local Neo4j containers.
- **Nebula** (alternate graph store) — host `127.0.0.1:9669`, present but not the primary path.

---

## 7. Evaluation framework

Two entry points share one record schema (`specs/FoodRAG_Evaluation_Protocol.md` §13):

### 7.1 Use-1 — `runner.py`

`foodrag.baselines.runner`:

- **`run_baseline(baseline, qa_set, recipes_df, *, k, experiment_id, run, expand, id_prefix,
  max_workers)`** (`:47`). Phase 1: `baseline.recommend(qa, k)` per item, optionally in a
  `ThreadPoolExecutor` (`max_workers>1`), each call wall-clock timed for per-question latency
  (§4.6). Phase 2 (sequential, deterministic): resolve retrieved ids → rows, compute
  `metrics.deterministic_metrics`, and emit one **§13.1 per-question record** per item.
- **`aggregate(records, *, n_boot=10_000, seed=0, judge="")`** (`:142`). Rolls records up to
  **§13.2** `config × stratum × metric` rows. For each metric it computes per-question means
  (NaN-aware), a **95% percentile bootstrap CI** (`_bootstrap_ci`, `n_boot = 10,000` resamples,
  2.5/97.5 percentiles, `:131`), the mean, and `per_question_variance` (mean over questions of
  across-run variance). Non-scalar fields (e.g. `guideline_adherence_per_nutrient` dict,
  `guideline_class` str) are coerced to NaN and skipped.
- **`write_results(records, aggregates, out_dir, config_id)`** (`:228`). Writes
  `{config_id}_per_question.parquet` (pickle fallback) and **upserts** rows into
  `eval_results/aggregate.csv` keyed by `(experiment_id, config_id, stratum, metric, judge)`, so a
  re-run replaces only its own rows.

### 7.2 Use-2 — `full_pipeline.py`

`foodrag.baselines.full_pipeline`:

- **`run_full_pipeline(assembler, qa_set, recipes_df, *, judges, k, experiment_id, rung, run,
  ...)`** (`:89`). Per item: `assembler.assemble(qa_item) → AssembledPlan`; then compute, on the
  **assembled plan's recipes**, the deterministic block (`metrics.deterministic_metrics`) + plan
  quality (`meal_plan_metrics.plan_quality_metrics`) + guideline adherence
  (`guideline_adherence.guideline_adherence`, with a hard-ceiling UL breach forcing
  `hard_constraint_violation → 1.0`) + explainability (`explainability.explainability_metrics`) +
  the dual-judge block (`judge.judge_plan`) + LLM-usage/latency. Emits a §13.1 record that also
  persists `plan` (per-slot `recipe_url`), `explanation_text`, `explanation_structured`,
  `recommendation_time`, `requested_meal_types`.
- **Assemblers.** `MealPlanAssembler` (ABC) → `RagMealPlanAssembler` (H2 / plain-RAG) and
  `WorkflowMealPlanAssembler` (H1). `AssembledPlan` (`:42`) is the interface dataclass carrying
  `question`, `answer_text`, `plan`, `recipe_urls`, `contexts`, `has_explanation`,
  `explanation_text/structured`, `recommendation_time`, `requested_meal_types`, and usage/latency.

The resilient driver `c:/tmp/run_foodrag_pipeline.py` re-implements this per-item scoring loop
(`score_item`, `:539`) so it can checkpoint each record immediately (see §9).

### 7.3 §13.1 per-question record schema

```
experiment_id, config_id, rung, run, question_id (q_{i:05d}),
answer_set_size, stratum, num_requirements, requirement_types, k,
metrics: { …all deterministic + plan + guideline + explainability + judge + usage… },
retrieved_ids: [recipe_url, …],
# Use-2 only:
plan: {meal_type: recipe_url | None}, explanation_text, explanation_structured,
recommendation_time, requested_meal_types
```

### 7.4 §13.2 aggregate schema

```
experiment_id, config_id, stratum, metric, mean, ci_low, ci_high,
per_question_variance, n_questions, n_runs, judge
```
(`AGG_COLUMNS`, `runner.py:28`; strata are `aggregate` + present strata in order
tight → medium → broad → empty.)

### 7.5 Dual-judge pass — `judge.py`

`foodrag.baselines.judge`. `build_judges` (`:91`) builds `judgeA = gemma` (cross-lineage) and
`judgeB = <default>` (`deepseek-v3`). `judge_plan` (`:106`) scores each judge and emits
`<metric>_judgeA` / `<metric>_judgeB`:

- **Faithfulness** {0,1} — uses only retrieved recipes? (`_binary`, `{"passing": …}`)
- **Relevancy** {0,1} — fits question + retrieved set?
- **Correctness** [0,1] — vs question + gold, rated 1–5, normalised `(s−1)/4` (`_scaled`).
- **Explanation Quality** [0,1] — only when `require_explanation` (H1), rated 1–5 → normalised.

Prompts are frozen in the module (`FAITHFULNESS_PROMPT`, …). Judges run at **temperature 0** (set
by the driver, e.g. `build_judges` in `run_foodrag_pipeline.build_shared`, judges are the
DeepInfra/OpenRouter models under the same LiteLLM logging).

---

## 8. Metric catalogue

Definition source: `specs/FoodRAG_Evaluation_Protocol.md` §4/§8/§9.4. Every metric below is
actually computed in code.

**Retrieval (§4.1, `metrics.py`)** — over the top-k `recipe_url` list vs the gold set:
`hit_at_{k}`, `mrr_at_{k}`, `ndcg_at_{k}` (binary-relevance nDCG). NaN when the gold set is empty.

**Health (§4.2, `metrics.py:avg_score_at_k`)** — mean over resolved top-k of `nutri_score`
(`avg_nutri_at_{k}`) and `fsa_score` (`avg_fsa_at_{k}`); plus `avg_rating_at_{k}` (the relevance
axis for the Health-vs-Relevance Pareto).

**Safety / constraints (§4.3, `metrics.py` + `constraints.py`)** —
`allergen_violation` (fraction of top-k containing a disliked/allergen ingredient, FoodOn-aware),
`hard_constraint_violation` (fraction breaching a hard nutrient bound; **also forced to 1.0 when a
plan breaches a guideline UL hard-ceiling**, §7.2), `constraint_satisfaction` (fraction satisfying
**all** requirements), `hallucination_rate` (fraction not resolving to a real recipe), and the
**hard/soft split**: `hard_constraint_sat_perconstraint` (mean satisfied/total hard),
`hard_constraint_sat_allornothing` (fraction satisfying all hard), `soft_constraint_sat`
(mean satisfied/total soft). Unresolved (`None`) recs never get a free pass.

**Plan-level (§4.4–4.5, `meal_plan_metrics.py`)** — `structural_validity` (all four slots present,
+ explanation if required), `meal_slot_completeness` (fraction of the four slots filled),
`intra_plan_diversity` (1 − mean pairwise ingredient/tag Jaccard), `calorie_adherence`
(1 − mean abs. deviation from targets B 22% / L 32% / D 32% / S 14%, `CALORIE_TARGETS`),
`slot_appropriateness` (fraction of filled slots whose `meal_types` overlap the slot synonym set).

**Guideline adherence (§4.5.1, `guideline_adherence.py` + `config/guideline_reference_tables.json`)** —
`guideline_adherence_per_nutrient` (per-nutrient 0/1 in-band over 8 nutrients: energy, protein,
carb, fat, satfat, fibre, sodium, sugar), `guideline_adherence_overall` (mean of those),
`guideline_class` (resolved reference class), and the hard-ceiling breach flag. Persona → class
mapping (`general_adult`, `diabetic_adult`, `hypertension_adult`, `weight_management_adult`) takes
the **stricter** band per nutrient across matched classes; bands are EFSA DRV (macro/fibre/sodium)
+ WHO (sugars, sodium), per-day.

**Judge (§4.4/§7, `judge.py`)** — `faithfulness_{judgeA,judgeB}`, `relevancy_{…}`,
`correctness_{…}`, `explanation_quality_{…}` (H1 only).

**Explainability (§8, `explainability.py`)** — `path_coverage` (fraction of stated requirements
with a supporting recipe-attribute path), `path_validity` (1.0 iff ≥1 covered, since every path is
a real attribute), and, when an explanation is present, `claim_grounding_rate` /
`unsupported_claim_rate` from rule-based NL claim extraction (score-min, nutrient-range/max,
ingredient include/exclude, tag-exclude, NutriScore-letter) checked against the plan's recipes.

**Processing / stability (§4.6)** — `llm_usage`, `llm_seq_usage` (sequential lower bound),
`llm_parsing` (repair calls), `latency_s` (per-question wall time), and (aggregate)
`per_question_variance`.

**Pareto (§9.4, `analysis/pareto.py`)** — non-dominated frontiers over configs on three planes:
**Health vs Relevance** (`avg_nutri/fsa` vs `avg_rating`/relevancy, max×max), **Quality vs Cost**
(`correctness` vs `llm_usage`/`latency`, max×min), **Safety vs Coverage** (stratum strictness vs
`meal_slot_completeness`). Safety is never a Pareto axis — hard constraints gate feasibility.

---

## 9. Run implementation details

Common to all runs: **stratum = tight** (`data/qa/main_10k_tight`, **n = 1,646**); retrieval
`k = 10`, meal-plan `k = 5`; judges at **temperature 0**; results written under `eval_results/`
(LLM-retriever Llama variants under `eval_results_llama/`). Resilience primitives appear across
drivers: **single-instance named mutex** (Windows `CreateMutexW`, exit if `ERROR_ALREADY_EXISTS`),
**per-item / per-run checkpointing** with **resume-by-`question_id`**, per-item `try/except → null
record`, and **tenacity retry** inside `LoggedLiteLLM`.

| Family | Driver | Runs | Workers | Notes |
| --- | --- | --- | --- | --- |
| Deterministic retrievers (dense/sparse/hybrid/fulltext) + composites (intersection/majority/fusion) | `run_composites.py`, `run_retrievers.py` | **1** | 6 | Deterministic → single run; skip-if-already-in-`aggregate.csv`; mutex. |
| Popularity (rungs 0a/0b) + Naive-RAG (rung 3) | `run_naive_rag.py`, baseline runners | **1** | — | Deterministic. Naive-RAG = retriever top-k as the recommendation. |
| LLM-only (rung 2) | `run_llm_only.py` | **5** | 16 | 5 stochastic runs over the full 1,646. |
| Text-to-pandas (both backbones) | `run_llm_retrievers_llama.py` (Llama-70B), `run_t2p_openai.py` (gpt-4o-mini) | **5** | 16 (12 for llama) | Per-query errors → empty result; per-run checkpoint + retry. Saved retrievals are later **replayed** into the FoodRAG t2p configs (no re-retrieval). |
| KG text-to-Cypher | `run_llm_retrievers*.py` (`ret_kg` / `ret_kg@llama70b`) | 5 | 8 | Neo4j bolt 7691; schema built from `data/neo4j_kg/all/*`. |
| ToG (external KBQA, rung 1) | `run_tog.py` | **3** | 5 | Over the **100-question** sample `main_10k_tog100`; chunked (10) checkpointing for a ~15–20 h unattended run. |
| **FoodRAG H1 / H2** (Use-2) | `run_foodrag_pipeline.py` | **3** | 16 (`WORKERS`, default 12) full 1,646 | One `hummus_data` (~7GB), one `recipes_df`, one dense retriever, one judge pair, shared read-only across threads. Per-(config,run) JSONL checkpoint (`data/qa/main_10k_tight/foodrag_ckpt/…`), resume by `question_id`. |

**FoodRAG time/date.** The meal-plan tool's slot reformulation uses a resolved time+date.
Per this context H1 runs at **time = morning** and H2 at **time = now**, **date = 13.04.2025**;
the driver defaults are `FOODRAG_MEALPLAN_TIME=now`, `FOODRAG_MEALPLAN_DATE=today` (→ `datetime.now()`),
and both are **persisted per record** (`recommendation_time`, `requested_meal_types`) so the run is
reproducible; pinning `FOODRAG_MEALPLAN_TIME`/`DATE` makes the requested meal set fully
deterministic. The driver's config matrix is `foodrag_h2_t2p`, `foodrag_h1_dense`, `foodrag_h1_t2p`
(`foodrag_h2_dense` already done). H1-t2p uses the `RotatingReplayRetriever`; H2-t2p uses the plain
`ReplayRetriever`; both seed from the saved text-to-pandas@llama70b parquet
(`eval_results_llama/ret_text2pandas@llama70b_per_question.parquet`, top-5).

---

## 10. Appendix: end-to-end flow diagrams

### 10.1 H1 five-agent workflow (ASCII)

```
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
        │  ValidationAgent                       │
        │  tool: CorrectionTool                  │
        │  • validate each slot                  │
        │  • 5-level generalization for invalid  │
        │    slots (re-recommend + re-validate)  │
        └──────┬───────────────────────┬─────────┘
     invalid? │ handoff                │ valid → handoff
               ▼                        ▼
   ┌───────────────────────┐          (skip)
   │  SubstitutionAgent    │
   │  tool: SubstitutionTool│
   │  • enhance/repair      │
   │    recipes (FoodOn)    │
   └───────────┬───────────┘
               │ handoff → ExplanationAgent
               ▼
   ┌───────────────────────┐
   │  ExplanationAgent     │
   │  tool: ExplanationTool │
   │  • write NL "why fits" │  → MEAL_PLAN_EXPLANATION_KEY
   └───────────┬───────────┘
               │ handoff → FinalizationAgent
               ▼
   ┌───────────────────────┐
   │  FinalizationAgent    │
   │  tool: Finalize…Tool   │  → FINAL_RECOMMENDATION_KEY
   └───────────┬───────────┘
               ▼
        final RecipeState  ──►  AssembledPlan (plan + explanation + usage)

   Shared throughout: RecipeState in AgentWorkflow Context["state"]
   (MEAL_PLAN_QUESTIONS_KEY, MEAL_PLAN_RECOMMENDATION_KEY,
    MEAL_PLAN_VALIDATION_KEY, …, LLM_*_USAGE_COUNTER_KEY)
```

### 10.2 retrieve → assemble → judge pipeline (Use-2)

```
qa_item (question, persona, answer=gold, meal_plan)
    │
    ▼
retriever.retrieve(query)  ── DataFrameRetrieverBase ──►  candidate recipes (pd.DataFrame)
   (dense pgvector | sparse | hybrid | fulltext | text-to-pandas |
    KG text-to-Cypher | fusion/intersection/majority | ReplayRetriever)
    │
    ▼
assemble:
   H2 → RagMealPlanAssembler(retrieve_only=True)        → AssembledPlan (no explanation)
   H1 → WorkflowMealPlanAssembler(AgentWorkflow)        → AssembledPlan (+ explanation)
    │
    ▼
score (per §13.1 record):
   • deterministic  : metrics.deterministic_metrics(plan recipes vs gold, persona)
   • plan quality   : meal_plan_metrics.plan_quality_metrics(plan)
   • guideline      : guideline_adherence.guideline_adherence(plan, persona)
                       └─ UL hard-ceiling breach → hard_constraint_violation = 1.0
   • explainability : explainability.explainability_metrics(persona, recipes, explanation_text)
   • judges         : judge.judge_plan(judgeA=gemma, judgeB=deepseek-v3)   # temperature 0
   • usage/latency  : llm_usage, llm_seq_usage, llm_parsing, latency_s
    │
    ▼
checkpoint (JSONL, resume by question_id)  ──►  aggregate()  ──►  write_results()
   bootstrap 95% CI (10k resamples)             eval_results/{config}_per_question.parquet
   per-question variance                         eval_results/aggregate.csv  (§13.2)
```

---

*Generated as a read-only architecture description; no code was modified.*

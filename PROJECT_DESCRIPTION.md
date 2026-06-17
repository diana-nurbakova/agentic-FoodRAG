# Agentic Food RAG - Project Description

## 1. Overview

This project implements a **health-aware, personalised meal plan recommender** that accepts natural language user queries (e.g. *"I'd like a low-sugar dinner without nuts"*) and returns a full daily meal plan (breakfast, lunch, dinner, snack) that respects:

- **Dietary preferences** (liked/disliked tags, ingredients)
- **Allergies and blacklisted ingredients**
- **Nutritional constraints** (calorie budgets, macro ranges, NutriScore thresholds)
- **General health guidelines** (WHO, FSA recommendations)

The system is built on top of the **HUMMUS** knowledge graph (270 284 recipes from food.com, linked to FoodOn ontology and USDA nutrition data) and uses **Retrieval-Augmented Generation (RAG)** with an agentic multi-step pipeline to retrieve, validate, correct, and explain recommendations.

Two recommender approaches are implemented:

| Approach | Description |
|---|---|
| **KBQA Recommender** | Constrained question-answering over the HUMMUS KG using a trained KBQA model with SPARQL candidate retrieval and constraint-label encoding |
| **Heuristic FoodRAG** | LLM-based RAG with a three-agent pipeline (Recommend, Correct, Enhance) that retrieves via vector/graph stores and generates meal plans with validation and explanation |

The **Heuristic FoodRAG** is the primary system implemented in this codebase.

---

## 2. Technology Stack

| Layer | Technology |
|---|---|
| RAG Framework | LlamaIndex |
| LLM | Llama 3.1 70B (quantised, via LiteLLM/OpenAI-compatible API) |
| Embedding Models | Jina Embeddings v2 base (EN/DE), Snowflake Arctic Embed L |
| Vector Database | PostgreSQL + pgvector (HNSW indexing) |
| Knowledge Graphs | Neo4j, NebulaGraph |
| Data Processing | pandas, NumPy, gensim, spaCy |
| Evaluation | Custom metrics + RAGAS framework |
| Configuration | YAML (nutrition/persona configs) |

---

## 3. Project Structure

```
agentic-food-rag/
│
├── config/                              # Configuration files
│   ├── nutrition_config.yml             # Nutrient thresholds (WHO, FSA, NutriScore)
│   ├── persona_config.yml               # Persona generation parameters
│   └── persona_config_2.yml             # Alternative persona config
│
├── data/                                # Data directory (generated at runtime)
│   ├── hummus/                          # HUMMUS dataset (recipes, users, reviews, FoodKG)
│   │   ├── hummus_data/preprocessed/    # Preprocessed recipe/user/review CSVs
│   │   ├── hummus_data/raw/             # Raw additional data (tags, etc.)
│   │   └── food_kg/                     # FoodKG and FoodOn ontology files
│   ├── embeddings/                      # Pre-computed recipe embeddings
│   └── qa/                              # Generated QA evaluation sets
│
├── src/                                 # Source code
│   ├── food_rag/                        # Core recommender system
│   │   ├── agents/                      # Agent orchestration
│   │   ├── evaluation/                  # Evaluation metrics and evaluators
│   │   ├── meal_plan/                   # Meal plan data structures (Pydantic)
│   │   ├── query_engines/               # RAG query engine
│   │   ├── retriever/                   # 9 retriever implementations
│   │   ├── tools/                       # 13 agent tool modules
│   │   └── utils/                       # RAG-specific helpers
│   ├── hummus_transformation/           # KG data transformation (Neo4j/Nebula)
│   ├── persona/                         # Persona & QA generation
│   └── utils/                           # General utilities (I/O, parsing, scoring)
│
├── thesis/                              # Thesis manuscript (PDF)
│
│  --- Notebooks (execution order below) ---
├── init_hummus.ipynb                    # Step 1: Initialise HUMMUS dataset
├── create_embeddings.ipynb (.py)        # Step 2: Compute recipe embeddings
├── init_vector_store.ipynb              # Step 3: Load embeddings into pgvector
├── init_graph_store_1.ipynb             # Step 4a: Build Neo4j graph store
├── init_graph_store_2.ipynb             # Step 4b: Build NebulaGraph store
├── qa_generation_1.ipynb                # Step 5a: Generate QA set (recipe-based)
├── qa_generation_2.ipynb                # Step 5b: Generate QA set (ingredient-class)
├── food_RAG_agents.ipynb                # Step 6: Run the FoodRAG agent pipeline
├── eval_1.ipynb (.py)                   # Step 7a: Evaluate 10 retriever configs
├── eval_2.ipynb                         # Step 7b: Extended evaluation
├── recipes-dataset.ipynb                # Utility: Dataset exploration
└── test.ipynb                           # Utility: Testing
```

---

## 4. Execution Order

The project follows a sequential setup pipeline, then runtime inference and evaluation.

### Phase 1: Data Initialisation

#### Step 1 - `init_hummus.ipynb`

Loads and preprocesses the raw HUMMUS dataset.

- Loads configs (`nutrition_config.yml`, `persona_config.yml`)
- Initialises LLM (Llama 3.1 via LiteLLM) and embedding model (Jina v2)
- Calls `load_util.get_hummus()` which:
  - Reads raw CSVs (recipes, users, reviews) from `data/hummus/`
  - Applies threshold discarding (`k_recipe=2`, `k_user=2`): removes items with < 2 interactions
  - Links recipes to FoodKG nutrition data, FoodOn ontology, and USDA ingredients
  - Annotates meal types (breakfast/lunch/dinner/snack) via LLM classification
  - Cleans food categories and computes NutriScore/FSA scores
- **Output**: Processed `hummus_data` object cached to `data/`

**Key function**: `src/foodrag/utils/load_util.py` -> `get_hummus()`

#### Step 2 - `create_embeddings.ipynb` / `create_embeddings.py`

Computes vector embeddings for all recipes.

- Converts each recipe into a text document containing: title, author, description, directions, nutritional info, ingredients, tags
- Embeds each document using the configured embedding model (Jina v2 base EN, 768-dim)
- Handles batch processing with error recovery and intermediate saves
- **Output**: Embeddings saved to `data/embeddings/jina_v2_base_de/`

**Key function**: `src/foodrag/food_rag/utils/embedding_helper.py` -> `compute_embeddings()`
**Key function**: `src/foodrag/food_rag/utils/hummus_docify.py` -> `recipes_to_docs()`

#### Step 3 - `init_vector_store.ipynb`

Loads embeddings into PostgreSQL with pgvector.

- Connects to PostgreSQL and initialises pgvector extension
- Creates HNSW index for fast approximate nearest-neighbour search
- Configures hybrid search (dense + sparse/BM25)
- **Output**: Populated vector store ready for retrieval queries

**Key class**: `src/foodrag/food_rag/utils/postgres_helper.py` -> `PostgresHelper`

#### Step 4a/4b - `init_graph_store_1.ipynb` / `init_graph_store_2.ipynb`

Builds knowledge graph stores in Neo4j and/or NebulaGraph.

- Transforms HUMMUS data to graph format (nodes: recipes, ingredients, tags, users; edges: contains, has_tag, reviewed_by, etc.)
- Processes FoodOn ontology RDF triples for ingredient hierarchies
- Loads data via Cypher bulk import
- Creates indices for efficient traversal
- **Output**: Populated graph databases

**Key class**: `src/foodrag/hummus_transformation/transform_kg.py` -> `Neo4JTransformer`
**Key helpers**: `src/foodrag/food_rag/utils/neo4J_helper.py`, `src/foodrag/food_rag/utils/nebula_helper.py`

### Phase 2: QA Set Generation

#### Step 5 - `qa_generation_1.ipynb` / `qa_generation_2.ipynb`

Generates the question-answer evaluation dataset (10 000 personas, ~4 555 non-empty QA pairs).

The 5-step pipeline:

1. **Persona generation** - Random personas with tag/ingredient preferences and nutrition constraints
2. **Answer querying** - Query HUMMUS to find all valid recipes matching each persona
3. **Question generation** - LLM converts structured persona into natural language questions
4. **Question rating** - LLM-as-a-judge rates each generated question (1-5)
5. **QA set finalisation** - Best question + valid recipe set form the final QA pair

**Key functions**:
- `src/foodrag/persona/persona_generator.py` -> `generate_persona_preferences()`
- `src/foodrag/persona/qa_generator.py` -> `generate_question_answer_set()`

### Phase 3: Running the Recommender

#### Step 6 - `food_RAG_agents.ipynb`

Main entry point for running the FoodRAG meal plan recommender.

Sets up and runs the **Heuristic Agentic FoodRAG** pipeline:

```
User Query (natural language)
       │
       ▼
┌─────────────────────┐
│ RECOMMENDATION AGENT │  ← Agent 1
│                       │
│  ┌─ Question Gen ────┐│  Tool: Splits query into per-meal-type questions
│  │  (LLM + time/date)││  using current time and date
│  └───────┬───────────┘│
│          ▼             │
│  ┌─ FoodRAG ─────────┐│  Tool: For each meal type:
│  │  Retriever → LLM  ││    1. Retrieve top-k recipes (embedding similarity)
│  │  → Meal Plan       ││    2. LLM selects best recipe per meal type
│  └───────┬───────────┘│
└──────────┼────────────┘
           ▼
┌─────────────────────┐
│  CORRECTION AGENT    │  ← Agent 2 (loop up to max_gen rounds)
│                       │
│  ┌─ Validation ──────┐│  Tool: LLM checks each recipe against:
│  │  Recipe vs Query   ││    - ingredient blacklist (allergies)
│  │  + KG facts        ││    - nutrition ranges
│  └───────┬───────────┘│    - tag requirements
│          ▼             │
│  if INVALID:           │
│  ┌─ Question General.─┐│  Tool: Relaxes constraints progressively
│  │  Relax constraints ││  (keeps allergens, drops soft prefs)
│  └───────┬───────────┘│
│          ▼             │
│  ┌─ FoodRAG (single) ─┐│  Tool: Re-retrieves for just the invalid meal
│  │  Re-recommend      ││
│  └───────┬───────────┘│
│          ▼             │
│  Status: OK → next     │
│  Status: REDO → restart│
│  Status: REJECT → stop │
└──────────┼────────────┘
           ▼
┌─────────────────────┐
│  ENHANCEMENT AGENT   │  ← Agent 3
│                       │
│  ┌─ Substitution ────┐│  Tool: LLM modifies ingredients/directions
│  │  Swap ingredients ││  using FoodOn hierarchy for healthier options
│  └───────┬───────────┘│
│          ▼             │
│  ┌─ Explanation ─────┐│  Tool: LLM generates natural language rationale
│  │  History → Text   ││  from the full generation history
│  └───────┬───────────┘│
└──────────┼────────────┘
           ▼
   Final Meal Plan + Explanation
```

**Key classes**:
- `src/foodrag/food_rag/agents/heuristic_agents.py` -> `HeuristicMealPlanReActAgent`
- `src/foodrag/food_rag/query_engines/rag.py` -> `RecipeRAGQueryEngine`
- `src/foodrag/food_rag/utils/recipe_context.py` -> `RecipeState`

### Phase 4: Evaluation

#### Step 7 - `eval_1.ipynb` / `eval_1.py` / `eval_2.ipynb`

Evaluates the system across 10 retriever configurations and multiple RAG setups.

**Retriever configurations evaluated**:

| # | Retriever | Type |
|---|---|---|
| 1 | Dense vector store | Embedding similarity (default) |
| 2 | Sparse vector store | BM25 keyword matching |
| 3 | Hybrid vector store | Dense + Sparse combined |
| 4 | Text search | Full-text search on PostgreSQL |
| 5 | Text-to-Pandas | LLM generates pandas query |
| 6 | Text-to-Cypher (Nebula) | LLM generates Cypher for NebulaGraph |
| 7 | Dense + Nebula | Embedding retrieval over graph store |
| 8 | Fusion (3-way) | Reciprocal rank fusion of multiple retrievers |
| 9 | Intersection | AND logic across retrievers |
| 10 | Majority voting | Voting-based intersection |

**Metrics computed**:
- *Retrieval*: Hit Rate, MRR, NDCG, Precision, Recall, F1 (at k=100)
- *Nutrition*: WHO score, FSA score, NutriScore (absolute and delta vs ground truth)

**Key class**: `src/foodrag/food_rag/evaluation/retriever_evaluator.py` -> `DataFrameRetrieverEvaluator`

---

## 5. Core Source Code Modules

### 5.1 Agents (`src/foodrag/food_rag/agents/`)

| File | Key Class | Purpose |
|---|---|---|
| `heuristic_agents.py` | `HeuristicMealPlanReActAgent` | Three-agent pipeline: Recommend → Correct → Enhance. Each agent has a fixed set of tools and follows a heuristic workflow |
| `workflow_runner.py` | `run_all_questions()` | Batch execution of the agent pipeline over QA sets for evaluation |

### 5.2 Query Engine (`src/foodrag/food_rag/query_engines/`)

| File | Key Class | Purpose |
|---|---|---|
| `rag.py` | `RecipeRAGQueryEngine` | Core RAG pipeline: retrieves top-k recipes via a configured retriever, ranks them, and passes to LLM with a prompt template to produce a meal plan or single recipe recommendation |

### 5.3 Retrievers (`src/foodrag/food_rag/retriever/`)

| File | Key Class | Purpose |
|---|---|---|
| `df_retriever_base.py` | `DataFrameRetrieverBase` | Abstract base class; all retrievers return a pandas DataFrame of matching recipes |
| `db_retriever.py` | `RecipeRetriever` | Vector store retriever (dense, sparse, hybrid, text_search modes) using pgvector |
| `kg_retriever.py` | `KGQueryEngine` | Text-to-Cypher retriever; LLM translates natural language to Cypher queries over Neo4j/Nebula |
| `pandas_retriever.py` | `PandasQueryRetriever` | Text-to-Pandas retriever; LLM generates pandas filter expressions executed on the recipe DataFrame |
| `advanced_retriever.py` | `DataFrameFusionRetriever`, `IntersectionRetriever`, `MajorityIntersectionRetriever` | Multi-retriever fusion strategies combining results from multiple retrievers |
| `wrapper_retriever.py` | - | Adapter wrapper for graph store retrievers |
| `util_retriever.py` | - | Shared retriever utility functions |

### 5.4 Tools (`src/foodrag/food_rag/tools/`)

Each tool wraps a specific capability and is assigned to one of the three agents:

| File | Tool Class | Agent | Purpose |
|---|---|---|---|
| `meal_plan_tools.py` | `MealPlanRecommendationTool` | Recommendation | Generates initial meal plan by calling FoodRAG per meal type. Augments query with time/date context |
| `meal_plan_tools.py` | `MealPlanUpdateTool` | Recommendation | Updates an existing meal plan with new recipes |
| `recipe_tools.py` | `RecipeRecommendationTool` | Recommendation | Single recipe recommendation |
| `validation_tools.py` | `MealPlanValidationTool` | Correction | Validates each recipe against user constraints using KG data and an LLM judge. Checks ingredient blacklists, nutrition ranges, tag requirements |
| `generalization_tools.py` | `RecipeQuestionGeneralizationTool` | Correction | Progressively relaxes user constraints (levels 1-5) while keeping allergens intact |
| `substitution_tools.py` | `MealPlanSubstitutionTool` | Enhancement | LLM-powered ingredient/direction substitution using FoodOn hierarchy for healthier alternatives |
| `explanation_tools.py` | `ExplanationTool` | Enhancement | Generates natural language explanation from the full generation history |
| `finalization_tool.py` | `FinalizeRecommendationTool` | Enhancement | Formats the final output (simple, detailed, or with_nutrition mode) |
| `ingredient_tools.py` | `IngredientSubstitutionTool` | Enhancement | Embedding-based ingredient similarity search (doc2vec, food_product embeddings) |
| `simple_recommender_tools.py` | - | - | Simpler recommendation alternatives for baseline comparisons |
| `workflow_tools.py` | - | - | Workflow management utilities |
| `human_loop_chat.py` | - | - | Human-in-the-loop interface |
| `tool_base.py` | `RecipeToolBase` | - | Abstract base class for all tools |
| `util_tools.py` | `JsonExtractorTool` | - | JSON extraction utilities |

### 5.5 Meal Plan Structures (`src/foodrag/food_rag/meal_plan/`)

| File | Key Class | Purpose |
|---|---|---|
| `meal_type.py` | `MealType` | Enum: `breakfast`, `lunch`, `dinner`, `snack` |
| `meal_plan_structures.py` | `MealPlanQuestions`, `MealPlan`, `MealPlanValidation`, `MealPlanSubstitution` | Pydantic data models for the pipeline's intermediate and final outputs |
| `json_encoder.py` | - | Custom JSON serialisation for meal plan objects |

### 5.6 Evaluation (`src/foodrag/food_rag/evaluation/`)

| File | Key Class | Purpose |
|---|---|---|
| `retriever_evaluator.py` | `DataFrameRetrieverEvaluator` | Orchestrates batch evaluation of retrievers against QA sets |
| `custom_metrics.py` | - | Custom metric implementations |
| `retrieval_metrics.py` | - | Hit rate, MRR, NDCG, precision, recall, F1 |
| `recipe_scores.py` | - | Nutrition-based recipe scoring (WHO, FSA, NutriScore) |
| `llm_evaluator.py` | - | LLM-as-a-judge evaluation (RAGAS-style) |
| `eval_parser.py` | - | Result parsing and aggregation |
| `qa_helper.py` | - | QA set loading, filtering, embedding attachment |

### 5.7 Utilities

#### RAG Utilities (`src/foodrag/food_rag/utils/`)

| File | Purpose |
|---|---|
| `hummus_docify.py` | Converts raw recipe/user/review DataFrames into LlamaIndex Document objects for embedding |
| `embedding_helper.py` | Batch embedding computation with error recovery and intermediate saves |
| `postgres_helper.py` | pgvector store initialisation, HNSW index creation, hybrid search setup |
| `neo4J_helper.py` | Neo4j connection management, Cypher execution, data loading |
| `nebula_helper.py` | NebulaGraph connection management, entity cleaning |
| `llm_helper.py` | RAG context formatting, response cleaning, markdown helpers |
| `recipe_context.py` | `RecipeState` - workflow state management, history tracking, LLM usage counters |
| `qa_helper.py` | QA set operations (load, filter, score) |
| `ragas_helper.py` | RAGAS evaluation integration |

#### General Utilities (`src/foodrag/utils/`)

| File | Purpose |
|---|---|
| `load_util.py` | Primary I/O: `get_hummus()` (dataset loading), `get_config()` (YAML), `store_data()`, `load_file()` |
| `hummus_util.py` | Data parsing (`try_parse()`), dict operations, data cleaning |
| `score_util.py` | Nutrition scoring utilities |
| `tag_util.py` | Tag processing |
| `recipe_similarity_util.py` | Recipe similarity computation |
| `rdf_util.py` | RDF/FoodOn ontology processing |
| `graphDB_util.py` | Graph database utilities |
| `cluster_util.py` | User clustering (MiniBatchKMeans) |
| `network_util.py` | Network configuration (Neo4j, Nebula connection params) |
| `analysis_util.py` | Data analysis helpers |

### 5.8 Data Transformation (`src/foodrag/hummus_transformation/`)

| File | Purpose |
|---|---|
| `transform_kg.py` | `Neo4JTransformer` - transforms HUMMUS data to Neo4j/Nebula format. Modes: all, nousda, unlinkedhummus, onlyfoodon. Handles RDF processing and CSV export for bulk loading |
| `usda_fields.py` | USDA nutrient field definitions |

### 5.9 Persona Generation (`src/foodrag/persona/`)

| File | Purpose |
|---|---|
| `persona_generator.py` | `generate_persona_preferences()` - creates random user personas with tag/ingredient preferences and nutrition constraints |
| `qa_generator.py` | `generate_question_answer_set()` - 5-step QA generation pipeline (persona → query → question → rating → QA pair) |
| `persona_validator.py` | Persona validation logic |
| `user_data_extractor.py` | Extracts persona profiles from real user data |
| `restriction_matcher.py` | Matches recipes against persona restrictions |

---

## 6. Configuration

### `config/nutrition_config.yml`

Defines all nutrition thresholds used for recipe scoring and validation:

- **Relative nutrients** (per 100g per serving): calories, fats, saturated fat, cholesterol, sodium, carbohydrates, dietary fibre, sugars, protein - each with low/medium/high ranges
- **Total nutrients** (daily intake): same nutrients with daily reference values (e.g. 2000 kcal/day)
- **Nutrition scores**: WHO, FSA, and NutriScore threshold definitions

### `config/persona_config.yml`

Controls QA set persona generation:

- **Tags**: 0-1 likes, 0-2 dislikes (randomly selected from available tags)
- **Ingredients**: 0-1 likes, 0-2 dislikes (from FoodOn or food.com sources)
- **Relative nutrients**: 0-1 nutrient constraints with low/medium/high ranges
- **Total nutrients**: 0-1 daily intake constraints with +/-20% deviation
- **Calorie distribution**: 22% breakfast, 32% lunch, 32% dinner, 14% snacks

---

## 7. Data Sources

The system builds on the **HUMMUS** knowledge graph which integrates:

| Source | Content | Scale |
|---|---|---|
| **food.com** (Kaggle) | Recipes, reviews, user profiles | 270 284 recipes, 83 881 users, 1.47M interactions |
| **FoodOn** | Food ontology (ingredient hierarchies, classifications) | OWL ontology |
| **USDA FoodData** | Detailed nutritional information per ingredient | Linked via FoodKG |
| **FoodKG** | Knowledge graph linking recipes to nutrition data | RDF triples |

After threshold discarding (k_recipes=2, k_users=2), the working dataset retains recipes and users with at least 2 interactions each.

---

## 8. Key Design Decisions

1. **Open-access LLMs only**: Uses quantised Llama 3.1 70B to ensure reproducibility and avoid API deprecation
2. **Modular architecture**: All components (retrievers, LLMs, vector stores, evaluators) are exchangeable via LlamaIndex interfaces
3. **Three-agent heuristic**: A single autonomous LLM agent was unstable; the fixed Recommend → Correct → Enhance pipeline provides reliability while maintaining flexibility within each stage
4. **Allergen safety**: The question generalisation tool never removes disliked ingredients (potential allergens) even at the highest relaxation level
5. **No recipe generation**: The system only recommends existing HUMMUS recipes (with optional ingredient substitution) to preserve validated nutritional information
6. **k=10 retriever results**: Balances context quality vs. noise for the LLM; tested k in {1, 5, 10, 15, 20}
7. **Persuasion integration**: Explanations, behaviour model tailoring, gamification, and nutrition labels are designed as modular add-ons motivated by behaviour change theories (TTM, SDT, HBM, FBM)

---
title: FoodRAG Demo
emoji: 🍽️
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
short_description: Health-aware agentic meal-plan recommender over the HUMMUS KG
---

<!-- HF reads the YAML block above and renders the rest as the Space card.
     Deployment runbook: docs/DEPLOY_HF.md -->

# Agentic Food RAG

A health-aware, personalised **meal-plan recommender**: it takes a natural-language request
(e.g. *"a low-sugar dinner without nuts"*) and returns a full daily meal plan that respects
dietary preferences, allergies, nutritional constraints, and health guidelines (EFSA/WHO/FSA/NutriScore).
It is built on the **HUMMUS** knowledge graph (food.com recipes linked to the FoodOn ontology and
USDA nutrition data) with an **agentic Retrieval-Augmented Generation** pipeline (LlamaIndex).

The system has three faces:

- a **Streamlit study app** (`app.py`) — interactive exploration, a structured user study, and a
  read-only results report — deployable to a Hugging Face Docker Space;
- an **agentic RAG pipeline** — a five-agent Recommend → Correct → Enhance → Explain → Finalize
  workflow (H1), plus a recommend-only variant (H2), over dense/sparse/hybrid/graph retrievers;
- an **evaluation framework** — deterministic + dual-LLM-judge metrics with a full baseline ladder
  (popularity, LLM-only, naive-RAG, ToG/RoG/TFS-KBQA, pFoodReq).

This is a refactored, reproducible version of the original research prototype
[gitlab.com/felix134/agentic-food-rag](https://gitlab.com/felix134/agentic-food-rag) — packaged with
`uv`, explicit data loading, configurable model providers, Dockerized stores, logging/retry/JSON
recovery, resumable runs, deterministic safety guards, and a test suite.

**Full architecture & reproducibility reference:** [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)

> **Status:** functional and deployable on branch `refactor/uv-loader`. Data-prep, embedding,
> stores, QA generation, the agentic pipeline, the evaluation framework and the study app are all
> wired up; headline results ship under `eval_results/`. Running the full pipeline at scale incurs
> real API cost (see *Costs*).

---

## Quickstart

```bash
# 1. Install Python 3.11 + dependencies (uv manages both)
uv sync --extra dev

# 2. Configure secrets + store settings
cp .env.example .env        # then fill in DEEPINFRA_API_KEY, OPENROUTER_API_KEY, HF_TOKEN

# 3. Start the backing stores (Postgres+pgvector, Neo4j)
docker compose up -d        # add --profile nebula for the legacy NebulaGraph path

# 4. Check models + data are reachable
uv run foodrag models-check         # 1-token LLM + embedding smoke test
uv run foodrag init-data            # validate the dataset manifest

# 5. Launch the demo app
uv run streamlit run app.py         # opens on http://localhost:8501
```

## The demo app

`app.py` is a single-page Streamlit app (routing driven by session state) with three flows:

- **Flow A — Interactive exploration.** Pick a persona, type a request, and run the **live** agentic
  pipeline (Llama-3.3-70B / BGE-M3). Toggle retriever and the Validate/Substitute/Explain agents in
  the pipeline strip; an arena mode compares two configs side by side. Falls back to a mock
  simulation when the real backend is disabled.
- **Flow B — Structured evaluation.** A between-subjects user study: a participant is assigned a
  dietary profile (round-robin) and rates **precomputed** full-day meal plans on ResQue-based Likert
  items with attention checks, Latin-square counterbalanced. Plans come from `ui/study_plans/` — the
  study selects, it never re-runs the pipeline.
- **Flow C — Results report** (`?report=true`) — a read-only dashboard over `eval_results/`
  (retriever comparison, safety, agentic attribution, error taxonomy, Pareto, judge reliability, …).

A password-gated admin panel (`?admin=true`) configures flows, personas, pipeline defaults and study
conditions, and exports interaction logs.

By default Flow A runs the mock backend; set `FOODRAG_REAL_BACKEND=1` (or enable it in the admin
panel) to run the live pipeline, which needs the stores up and API keys in `.env`.

## Deploy

The app ships as a **Docker image** (`Dockerfile` + `deploy/start.sh`) that runs Streamlit,
Postgres+pgvector and Neo4j inside one container on a persistent `/data` volume — the shape used for
**Hugging Face Docker Spaces** (`app_port: 7860`). Runtime data (recipes, embeddings, KG) is pulled
once from a private HF Dataset (`FOODRAG_DATA_REPO`) at first boot and seeded into the stores. See
the runbook: **[docs/DEPLOY_HF.md](docs/DEPLOY_HF.md)**.

## Models & providers

Locked in [config/models.yml](config/models.yml); keys come from `.env` (never hardcoded):

| Role | Default | Provider |
|---|---|---|
| Main LLM (agent) | `meta-llama/Llama-3.3-70B-Instruct` (temp 0.1) | DeepInfra |
| Embedding | `BAAI/bge-m3` (1024-d) | DeepInfra |
| Judge A (headline, cross-lineage) | `google/gemma-3-27b-it` | OpenRouter |
| Judge B (same-family control) | `deepseek-ai/DeepSeek-V3` | DeepInfra |

A judge model may never equal the generator (enforced in `config/settings.py`), so same-model
self-grading is impossible. Judges run at temperature 0.

The embedding model is **benchmark-then-lock**: `uv run foodrag bench-embeddings --smoke` compares
candidates (BGE-M3, Qwen3-Embedding-4B, bge-large-en, all-mpnet-base-v2) on a retrieval task.
Run it on the real QA set to pick the production embedding.

## Data

All external inputs are described once in [config/data_manifest.yml](config/data_manifest.yml)
(path, size, source URL, how to unzip/download). The loader validates them and recovers what it can:

```bash
uv run python -m foodrag.data status     # show presence of every required file
uv run python -m foodrag.data prepare    # unzip local archives + download public files (FoodOn)
uv run python -m foodrag.data validate   # non-zero exit if anything required is missing
```

Sources: HUMMUS recipes/reviews/members ([GitLab](https://gitlab.com/felix134/connected-recipe-data-set)),
food.com tags ([Kaggle](https://www.kaggle.com/shuyangli94/food-com-recipes-and-user-interactions),
free account), FoodKG mappings, the [FoodOn](https://foodon.org/) ontology, and
[USDA FoodData](https://fdc.nal.usda.gov/download-datasets). Data lives under `data/` (gitignored).

## Pipeline

Each stage is available as a **CLI subcommand**, a thin **script**, and an importable
**`foodrag.pipeline`** module (with an injectable `run()` for testing). Run in order:

| Stage | CLI | Module |
|---|---|---|
| 1. Init HUMMUS | `foodrag init-hummus` | `foodrag.pipeline.init_hummus` |
| 2. Embed | `foodrag embed [--sample N]` | `foodrag.pipeline.embed` |
| 3. Vector store | `foodrag init-vector-store` | `foodrag.pipeline.vector_store` |
| 4. Graph store | `foodrag init-graph-stores` | `foodrag.pipeline.graph_stores` |
| 5. Generate QA | `foodrag gen-qa` | `foodrag.pipeline.gen_qa` |
| 6. Run baselines | `foodrag run-baselines` | `foodrag.pipeline.run_baselines` |
| 7. Run agents (H1/H2) | — (driver) | `foodrag.pipeline.run_agents` |
| 8. Evaluate | — (driver) | `foodrag.pipeline.evaluate` |

The original Jupyter **notebooks** are preserved under [notebooks/](notebooks/) and made runnable
(they import `foodrag.*` and call the config factory). Each starts with a bootstrap cell
(`from foodrag.notebook import setup; setup()`) that runs from the repo root and loads `.env`.

## Evaluation

Two use-modes share one record schema and are scored with deterministic metrics **and** the dual
LLM judges:

- **Use-1 (retrieval-only)** — a retriever's ranked list is scored against each persona's gold
  answer set (`foodrag.baselines.runner`).
- **Use-2 (full pipeline)** — the assembled meal plan is scored end-to-end
  (`foodrag.baselines.full_pipeline`): **H1** (five-agent, with explanation) vs **H2** (recommend-only).

Metrics span retrieval (hit/MRR/nDCG), health (NutriScore/FSA), safety & constraints (allergen /
hard-constraint / hallucination), plan quality (structure, diversity, calorie & slot adequacy),
guideline adherence (EFSA/WHO persona-class bands), explainability (claim grounding), and dual-judge
faithfulness/relevancy/correctness/explanation-quality. Aggregation reports bootstrap 95% CIs (10k
resamples) plus paired Wilcoxon tests with Holm–Bonferroni correction.

Deterministic **safety guards** are applied to every final plan: an allergen post-filter (zero
allergen violations by construction), an empty-plan explanation gate, and a distinct-slot guard.

Headline results (tight stratum, n=1,646) and analysis ship under `eval_results/` and in
[docs/RESULTS_REPORT.md](docs/RESULTS_REPORT.md), [docs/STATISTICAL_TESTS.md](docs/STATISTICAL_TESTS.md)
and [docs/ERROR_ANALYSIS.md](docs/ERROR_ANALYSIS.md).

## Reliability

- **Logging** — every LLM/embedding call is appended to `data/logs/llm_calls-<date>.jsonl`
  (model, prompt, response, latency, tokens, cost). See `foodrag.llm.call_log`.
- **Retry** — one standard exponential-backoff policy (`foodrag.llm.llm_retry`).
- **JSON recovery** — robust parsing of messy LLM output (`foodrag.llm.recover_json`).
- **Resumable runs** — `foodrag.llm.RunStore` checkpoints per item so a crashed
  embedding/QA/eval run re-runs only the missing/errored items.

## Tests

```bash
uv run pytest -m unit                 # fast, no data/services (CI default)
uv run pytest -m integration          # needs `docker compose up` (pgvector/neo4j)
uv run pytest -m "data or llm"        # needs local data / API keys (opt-in)
```

## Costs

Embedding all ~270k recipes and LLM meal-type annotation/QA generation go through paid APIs and can
take hours. Use `--sample N` and small QA sizes for smoke tests; run full-scale deliberately.

## Repository layout

```
app.py         Streamlit entry point (Flows A/B/C)
ui/            study app — screens/ services/ components/ study_plans/ + state, theme, admin
config/        models.yml, data_manifest.yml, guideline_reference_tables.json, nutrition/persona configs
src/foodrag/   config/ data/ llm/ stores/ pipeline/ analysis/ + food_rag/ baselines/ persona/ hummus_transformation/
scripts/       per-stage CLI wrappers + deploy/seed scripts (bootstrap_hf_data, seed_pgvector, seed_neo4j, build_study_plans)
deploy/        start.sh (in-container boot); Dockerfile + .dockerignore at root
docs/          ARCHITECTURE.md, DEPLOY_HF.md, RESULTS_REPORT.md, STATISTICAL_TESTS.md, ERROR_ANALYSIS.md
eval_results/  per-question parquets, aggregate.csv, pairwise tests
notebooks/     original notebooks (runnable, legacy)
tests/         pytest suite (incl. UI/study tests)
docker-compose.yml, pyproject.toml, uv.lock
```

## Credits

Refactor of the Food RAG system by Felix Bölz et al.; built on **HUMMUS**
(Bölz, Nurbakova, Calabretto, Gerl, Brunie, Kosch — RecSys '23,
[doi:10.1145/3604915.3609491](https://doi.org/10.1145/3604915.3609491)), LlamaIndex, FoodKG, FoodOn,
and USDA FoodData Central.

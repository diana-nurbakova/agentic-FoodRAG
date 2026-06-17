# Agentic Food RAG

A health-aware, personalised **meal-plan recommender**: it takes a natural-language request
(e.g. *"a low-sugar dinner without nuts"*) and returns a full daily meal plan that respects
dietary preferences, allergies, nutritional constraints, and health guidelines (WHO/FSA/NutriScore).
It is built on the **HUMMUS** knowledge graph (food.com recipes linked to the FoodOn ontology and
USDA nutrition data) with an **agentic Retrieval-Augmented Generation** pipeline (LlamaIndex).

This is a refactored, reproducible version of the original research prototype
[gitlab.com/felix134/agentic-food-rag](https://gitlab.com/felix134/agentic-food-rag) — packaged with
`uv`, explicit data loading, configurable model providers, Dockerized stores, logging/retry/JSON
recovery, resumable runs, and a test suite. See [PROJECT_DESCRIPTION.md](PROJECT_DESCRIPTION.md) for
the methodology and architecture.

> **Status:** refactor in progress on branch `refactor/uv-loader`. The data-prep, embedding,
> store, QA-generation, agent and evaluation stages are wired to the new infrastructure; running
> the full pipeline at scale incurs real API cost (see *Costs*).

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
```

## Models & providers

Locked in [config/models.yml](config/models.yml); keys come from `.env` (never hardcoded):

| Role | Default | Provider |
|---|---|---|
| Main LLM (agent) | `meta-llama/Llama-3.3-70B-Instruct` | DeepInfra |
| Embedding | `BAAI/bge-m3` (1024-d) | DeepInfra |
| Judge(s) | `deepseek-ai/DeepSeek-V3`, Gemma | DeepInfra / OpenRouter |

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
| 6. Run agents | — (see notebook) | `foodrag.pipeline.run_agents` |
| 7. Evaluate | — (see notebook) | `foodrag.pipeline.evaluate` |

The original Jupyter **notebooks** are preserved under [notebooks/](notebooks/) and made runnable
(they import `foodrag.*` and call the config factory). Each starts with a bootstrap cell
(`from foodrag.notebook import setup; setup()`) that runs from the repo root and loads `.env`.

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
config/        models.yml, data_manifest.yml, nutrition/persona configs
src/foodrag/   config/ data/ llm/ stores/ experiments/ pipeline/ + food_rag/ persona/ utils/ hummus_transformation/
scripts/       thin per-stage CLI wrappers
notebooks/     original notebooks (runnable, legacy)
tests/         pytest suite
docker-compose.yml, pyproject.toml, uv.lock
```

## Credits

Refactor of the Food RAG system by Felix Bölz et al.; built on **HUMMUS**
(Bölz, Nurbakova, Calabretto, Gerl, Brunie, Kosch — RecSys '23,
[doi:10.1145/3604915.3609491](https://doi.org/10.1145/3604915.3609491)), LlamaIndex, FoodKG, FoodOn,
and USDA FoodData Central.

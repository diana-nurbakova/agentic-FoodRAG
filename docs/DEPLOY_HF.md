# Deploying FoodRAG to Hugging Face Spaces

This runbook deploys the **full system** (Text→Pandas, Dense/Sparse/Hybrid,
Fusion, and KG retrievers) to a single HF **Docker Space** running Postgres +
pgvector, Neo4j, and Streamlit together, backed by a persistent volume.

## Why this shape

- HF Spaces is **one container** — there is no `docker compose`, so the local
  `ensure_stores_up()` path can't spawn sibling DB containers. We run both DBs
  *inside* the container and point the app at `localhost` (it already reads all
  connection info from env — [`stores.py`](../src/foodrag/config/stores.py)).
- The full corpus does **not** fit free managed DB tiers (KG ≈ 1.9M nodes / 9.5M
  rels ≫ Aura Free's 200k/400k; pgvector ≈ several GB ≫ Neon Free 0.5 GB), so
  self-hosting on a **persistent volume** is both cheaper and simpler than paid
  managed DBs for this size.
- Only a **~5 GB slice** of the 54 GB `data/` is ever needed at runtime; it lives
  in a private **HF Dataset** and is pulled on first boot.

```
HF Docker Space (Pro) ── /data = persistent volume (>=20 GB)
  deploy/start.sh
    ├─ postgres (pgvector)   PGDATA=/data/pg
    ├─ neo4j                 data=/data/neo4j, import=/data/neo4j_kg
    └─ streamlit             :7860
  first boot: snapshot_download(<dataset>) → /data ; seed pg + neo4j (idempotent)
```

## Prerequisites

- **HF Pro** (for Dev Mode SSH + higher limits) and a **persistent-storage
  add-on** on the Space (Small **20 GB** is enough: pg ~6 GB + neo4j ~3 GB +
  hummus ~1 GB + embeddings/KG staging ~4 GB). Storage is billed on top of Pro.
- Free Space hardware is 2 vCPU / **16 GB RAM** — enough to *start*, but tight
  with both DBs + the bge-m3 model. If you hit OOM, upgrade to an upgraded-CPU
  (32 GB) tier. Neo4j heap/pagecache are left at conservative defaults; cap them
  in `deploy/start.sh`'s `neo4j.local.conf` block if needed.
- An LLM key: `OPENROUTER_API_KEY` **or** `DEEPINFRA_API_KEY`.

## What's in this repo for the deploy

| File | Role |
|---|---|
| [`Dockerfile`](../Dockerfile) | Space image: Python 3.11 + Postgres 16/pgvector + Neo4j 5.26 + app |
| [`.dockerignore`](../.dockerignore) | Keeps `data/`, `.venv/`, `.git/` out of the build context |
| [`deploy/start.sh`](../deploy/start.sh) | Entrypoint: init/start both DBs on `/data`, seed once, launch Streamlit |
| [`scripts/push_hf_data.py`](../scripts/push_hf_data.py) | **(your machine)** upload the runtime slice to a private HF Dataset |
| [`scripts/bootstrap_hf_data.py`](../scripts/bootstrap_hf_data.py) | **(first boot)** `snapshot_download` the slice onto `/data` |
| [`scripts/seed_neo4j.py`](../scripts/seed_neo4j.py) | **(first boot)** LOAD CSV the KG into Neo4j (idempotent) |
| [`scripts/seed_pgvector.py`](../scripts/seed_pgvector.py) | **(first boot)** warm the pgvector ingest (idempotent) |
| `FOODRAG_IN_CONTAINER` guard | `ensure_stores_up()` no-ops in-container (no docker compose) |

---

## Step 1 — Push the data slice to a private HF Dataset (from your machine)

```bash
huggingface-cli login                     # or: export HF_TOKEN=hf_...
python scripts/push_hf_data.py \
    --repo <user>/foodrag-data --private \
    --data-dir ./data \
    --include-embeddings --include-kg
```

This uploads (mirroring the `data/` layout so paths resolve unchanged on the Space):
- `hummus/*.pkl|csv|json|ttl` **excluding `reviews_df.csv`** (loaded but unused at
  inference → saves ~381 MB),
- `embeddings/BAAI_bge-m3/...` (needed once to seed pgvector),
- `neo4j_kg/all` + `nousda` (needed once to seed Neo4j).

> Leaner option: omit `--include-embeddings`/`--include-kg` and seed the two DBs
> yourself over Dev Mode SSH instead (see Step 5). Then the dataset is ~1 GB and
> first boot is fast.

## Step 2 — Create the Space

Create a new **Docker** Space, then push this repo to it (or connect the GitHub
repo). Add a Space `README.md` with HF metadata:

```yaml
---
title: FoodRAG Demo
emoji: 🍽️
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---
```

In **Settings → Persistent storage**, attach **Small (20 GB)** (mounts at `/data`).

## Step 3 — Set Space secrets

**Settings → Variables and secrets** (Secrets, not public Variables):

| Secret | Value |
|---|---|
| `OPENROUTER_API_KEY` *(or `DEEPINFRA_API_KEY`)* | your LLM key |
| `HF_TOKEN` | token with **read** access to the private dataset |
| `FOODRAG_DATA_REPO` | `<user>/foodrag-data` |
| `POSTGRES_PASSWORD` | any strong value |
| `NEO4J_PASSWORD` | any strong value |

`FOODRAG_IN_CONTAINER=1`, `POSTGRES_*` and `NEO4J_*` hosts default correctly in
`start.sh`; only the passwords above must be set. (`start.sh` also exports
`FOODRAG_IN_CONTAINER=1` so the docker-compose bootstrap is skipped.)

## Step 4 — First boot

The Space builds the image, then `start.sh`:
1. links `/app/data → /data` and creates the DB dirs on the volume,
2. `initdb` + starts Postgres, creates the DB and `CREATE EXTENSION vector`,
3. configures + starts Neo4j (data on `/data/neo4j`, import root `/data/neo4j_kg`),
4. `bootstrap_hf_data.py` downloads the slice (guarded by `/data/.hf_data_seeded`),
5. `seed_neo4j.py` imports the KG (skips if Recipe nodes already exist),
6. `seed_pgvector.py` warms the vector ingest (skips if the table is non-empty),
7. launches Streamlit on `:7860`.

**First boot is slow** (data download + KG import + embedding ingest = many
minutes) and may look idle — watch the build/run logs. Every later boot
short-circuits on the markers/non-empty stores and starts in well under a minute.

## Step 5 — (Optional) seed via Dev Mode instead

If you skipped `--include-embeddings/--include-kg`, enable **Dev Mode** (Pro),
SSH in, and seed against the already-running in-container DBs:

```bash
cd /app
# copy embeddings + neo4j_kg onto /data by your preferred means, then:
python scripts/seed_neo4j.py
python scripts/seed_pgvector.py
```

## Verifying

- Logs show `Started`/`Recipe nodes.`/`vector index ready.` on first boot.
- The Streamlit app loads; the **Text→Pandas** retriever works with no DB.
- **Dense/Sparse/Hybrid** return results (pgvector seeded).
- **Fusion/KG** contribute graph paths (Neo4j seeded) — the KG retriever is
  best-effort and silently degrades to dense+sparse if Neo4j is unreachable.

## Troubleshooting

- **OOM / Space restarts under load** → upgrade to a 32 GB CPU tier, and/or cap
  Neo4j memory in the `neo4j.local.conf` heredoc in `start.sh`
  (`server.memory.heap.max_size`, `server.memory.pagecache.size`).
- **Neo4j import finds no files** → the dataset must contain `neo4j_kg/all/*`;
  confirm `server.directories.import=/data/neo4j_kg` and that
  `file:///all/node_Recipe.csv` resolves there.
- **pgvector stays empty** → embeddings weren't downloaded; check
  `data/embeddings/BAAI_bge-m3/recipe_documents/batch_*.pkl` exists on `/data`.
- **First request very slow, then fine** → expected if `seed_pgvector.py` was
  skipped; the lazy ingest runs on the first query instead.
- **spaCy model missing** at runtime → add `python -m spacy download
  en_core_web_sm` to the Dockerfile if a retriever needs it.

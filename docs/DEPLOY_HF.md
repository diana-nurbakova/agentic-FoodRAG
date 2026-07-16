# Deploying FoodRAG to Hugging Face Spaces

This runbook deploys the **full system** (Text→Pandas, Dense/Sparse/Hybrid,
Fusion, and KG retrievers) to a single HF **Docker Space** running Postgres +
pgvector, Neo4j, and Streamlit together on the Space's local disk, kept alive by
**always-on paid hardware**.

## Why this shape

- HF Spaces is **one container** — there is no `docker compose`, so the local
  `ensure_stores_up()` path can't spawn sibling DB containers. We run both DBs
  *inside* the container and point the app at `localhost` (it already reads all
  connection info from env — [`stores.py`](../src/foodrag/config/stores.py)).
- The full corpus does **not** fit free managed DB tiers (KG ≈ 1.9M nodes / 9.5M
  rels ≫ Aura Free's 200k/400k; pgvector ≈ several GB ≫ Neon Free 0.5 GB), so
  self-hosting inside the Space is far cheaper than paid managed DBs at this size.
- Only a **~5.6 GB slice** of the 54 GB `data/` is ever needed at runtime; it
  lives in a private **HF Dataset** and is pulled on first boot.

### Storage: read this before changing anything

HF's old **"Persistent storage" (Small/Medium/Large) disk add-on no longer
exists** — `HfApi.request_space_storage()` now 404s. Its replacement,
[Storage Buckets](https://huggingface.co/docs/hub/storage-buckets), is *Xet-backed
object storage* mounted lazily over NFS/FUSE. **Do not put `PGDATA` or the Neo4j
store on a bucket**: databases need fsync durability, file locking and atomic
renames, which lazily-fetched object storage does not provide. Buckets are for
checkpoints/artifacts, not database data directories.

What we use instead: a Space's **local disk (50 GB, ephemeral)**, which is ample
for our ~14 GB. Ephemeral only matters across restarts, and we avoid restarts by
running on **always-on paid hardware** (free Spaces sleep after ~48 h idle, and a
wake would re-run the whole 30–90 min seed).

```
HF Docker Space (cpu-upgrade, always-on) ── /data = local disk (ephemeral, 50 GB)
  deploy/start.sh
    ├─ postgres (pgvector)   PGDATA=/data/pg
    ├─ neo4j                 data=/data/neo4j, import=/data/neo4j_kg
    └─ streamlit             :7860
  boot: snapshot_download(<dataset>) → /data ; seed pg + neo4j (idempotent)
```

## Prerequisites

- **HF Pro** — for Dev Mode SSH (invaluable for debugging the boot) and
  "Protected" visibility (private source, publicly reachable app: ideal for a
  paper artifact link).
- **`cpu-upgrade` hardware** (8 vCPU / 32 GB, ~$0.03/hr ≈ $21/mo). This is
  *required for practical use*: it never sleeps, so the one-time seed survives.
  On free `cpu-basic` the Space sleeps and every wake re-seeds from scratch.
- **No persistent storage / bucket is needed** (see above).
- A **DeepInfra key** — `config/models.yml` uses DeepInfra for BOTH the generator
  (Llama-3.3-70B) *and* the bge-m3 query embeddings. `OPENROUTER_API_KEY` alone is
  NOT sufficient unless you also repoint `models.yml`. Cap your DeepInfra spend
  before making the Space publicly reachable.
- RAM note: embeddings are an **API call**, not a local model — nothing loads
  bge-m3 into memory, so RAM pressure is much lower than a local-embedding setup.

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

Do **not** attach storage — the add-on is gone and a bucket must not host the DBs
(see "Storage" above). Instead upgrade the hardware so the Space never sleeps:

```python
api.request_space_hardware("<user>/foodrag-demo", hardware="cpu-upgrade")
```

## Step 3 — Set Space secrets

**Settings → Variables and secrets** (Secrets, not public Variables). These can
also be set programmatically with `api.add_space_secret(...)`:

| Secret | Value |
|---|---|
| `DEEPINFRA_API_KEY` | **required** — `models.yml` uses DeepInfra for the LLM *and* embeddings |
| `HF_TOKEN` | token with **read** access to the private dataset |
| `FOODRAG_DATA_REPO` | `<user>/foodrag-data` |
| `POSTGRES_PASSWORD` | any strong value |
| `NEO4J_PASSWORD` | any strong value |
| `ADMIN_PASSWORD` | for the `?admin=true` view |

Public **Variables**: `FOODRAG_IN_CONTAINER=1`, `POSTGRES_HOST=localhost`,
`POSTGRES_PORT=5432`, `POSTGRES_USER=postgres`, `POSTGRES_DB=rag_postgres`,
`NEO4J_URI=bolt://localhost:7687`, `NEO4J_USER=neo4j`.

`FOODRAG_IN_CONTAINER=1` makes `ensure_stores_up()` skip the docker-compose path
(there is no Docker daemon in a Space); `start.sh` exports it too as a fallback.

## Step 4 — First boot

The Space builds the image, then `start.sh`:
1. links `/app/data → /data` and creates the DB dirs on the local disk,
2. `initdb` + starts Postgres, creates the DB and `CREATE EXTENSION vector`,
3. configures + starts Neo4j (data on `/data/neo4j`, import root `/data/neo4j_kg`),
4. `bootstrap_hf_data.py` downloads the slice (guarded by `/data/.hf_data_seeded`),
5. `seed_neo4j.py` imports the KG (skips if Recipe nodes already exist),
6. `seed_pgvector.py` warms the vector ingest (skips if the table is non-empty),
7. launches Streamlit on `:7860`.

**Boot is slow**: 5.6 GB download, then a `LOAD CSV` import of ~1.9M nodes / 9.5M
relationships (realistically **30–90 min**), then the pgvector ingest of 137
embedding batches. It may look idle — watch the run logs. The seeding steps are
idempotent, so a re-run within the same container returns immediately.

**Because the disk is ephemeral, a container restart re-seeds from scratch.** That
is exactly why the Space runs on always-on `cpu-upgrade` hardware. Avoid
unnecessary rebuilds/restarts; each one costs a full re-seed. If you need to iterate
on code frequently, use Dev Mode SSH rather than pushing commits.

Watch the logs programmatically (needs a newer `huggingface_hub` than 0.26):
`api.fetch_space_logs(repo_id, follow=True)` / `hf spaces logs <id> -f`.

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

- **`request_space_storage()` returns 404** → expected. The persistent-storage
  add-on was removed from HF; there is nothing to attach. See "Storage" above.
- **OOM / Space restarts under load** → `cpu-upgrade` already gives 32 GB. Cap
  Neo4j memory in the `neo4j.local.conf` heredoc in `start.sh`
  (`server.memory.heap.max_size`, `server.memory.pagecache.size`). Note the
  embedding model is remote (DeepInfra), so it costs no RAM here.
- **Neo4j import finds no files** → the dataset must contain `neo4j_kg/all/*`;
  confirm `server.directories.import=/data/neo4j_kg` and that
  `file:///all/node_Recipe.csv` resolves there.
- **pgvector stays empty** → embeddings weren't downloaded; check
  `data/embeddings/BAAI_bge-m3/recipe_documents/batch_*.pkl` exists on `/data`.
- **First request very slow, then fine** → expected if `seed_pgvector.py` was
  skipped; the lazy ingest runs on the first query instead.
- **spaCy model missing** at runtime → add `python -m spacy download
  en_core_web_sm` to the Dockerfile if a retriever needs it.

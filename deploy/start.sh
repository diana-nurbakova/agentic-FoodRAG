#!/usr/bin/env bash
# Entrypoint for the Hugging Face Docker Space.
#
# Boots Postgres (pgvector) + Neo4j inside this single container, both backed by
# the persistent volume at /data, seeds them once (idempotent), then launches
# Streamlit. On restarts the marker files / non-empty stores short-circuit the
# seeding, so only the very first boot of a fresh volume is slow.
set -euo pipefail

APP_DIR="${APP_DIR:-/app}"
DATA="${DATA_DIR:-/data}"
export FOODRAG_IN_CONTAINER=1
export POSTGRES_HOST="${POSTGRES_HOST:-localhost}"
export POSTGRES_PORT="${POSTGRES_PORT:-5432}"
export POSTGRES_USER="${POSTGRES_USER:-postgres}"
export POSTGRES_PASSWORD="${POSTGRES_PASSWORD:-postgres}"
export POSTGRES_DB="${POSTGRES_DB:-rag_postgres}"
export NEO4J_USER="${NEO4J_USER:-neo4j}"
export NEO4J_PASSWORD="${NEO4J_PASSWORD:-neo4jfoodrag}"

echo "== FoodRAG Space boot =="

# ── persistent layout + symlink so repo_root()/data lands on the volume ──────
mkdir -p "$DATA/pg" "$DATA/neo4j/data" "$DATA/neo4j/logs" "$DATA/hummus"
# The app reads <repo_root>/data/... ; point that at the persistent volume.
if [ ! -e "$APP_DIR/data" ]; then
  ln -s "$DATA" "$APP_DIR/data"
fi

# ── Postgres (pgvector) ──────────────────────────────────────────────────────
export PGDATA="$DATA/pg"
export PATH="/usr/lib/postgresql/16/bin:$PATH"
if [ ! -s "$PGDATA/PG_VERSION" ]; then
  echo "[pg] initializing cluster at $PGDATA"
  pwfile="$(mktemp)"; printf '%s' "$POSTGRES_PASSWORD" > "$pwfile"
  initdb -U "$POSTGRES_USER" --auth=scram-sha-256 --pwfile="$pwfile" >/dev/null
  rm -f "$pwfile"
  echo "listen_addresses='localhost'" >> "$PGDATA/postgresql.conf"
  echo "port=$POSTGRES_PORT"          >> "$PGDATA/postgresql.conf"
fi
echo "[pg] starting"
pg_ctl -D "$PGDATA" -l "$DATA/pg/server.log" -w -t 60 start
# Create the app DB + pgvector extension once.
psql -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U "$POSTGRES_USER" -d postgres -tc \
  "SELECT 1 FROM pg_database WHERE datname='$POSTGRES_DB'" | grep -q 1 \
  || psql -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U "$POSTGRES_USER" -d postgres \
       -c "CREATE DATABASE \"$POSTGRES_DB\""
psql -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U "$POSTGRES_USER" -d "$POSTGRES_DB" \
  -c "CREATE EXTENSION IF NOT EXISTS vector"

# ── Neo4j ────────────────────────────────────────────────────────────────────
export NEO4J_HOME="${NEO4J_HOME:-/opt/neo4j}"
export PATH="$NEO4J_HOME/bin:$PATH"
# Data/import/logs dirs live on the volume; import root matches insert_data's
# expectation (data/neo4j_kg → file:///all/...). Written idempotently each boot.
{
  echo "server.directories.data=$DATA/neo4j/data"
  echo "server.directories.logs=$DATA/neo4j/logs"
  echo "server.directories.import=$DATA/neo4j_kg"
  echo "server.default_listen_address=0.0.0.0"
  echo "dbms.security.allow_csv_import_from_file_urls=true"
  echo "dbms.security.procedures.unrestricted=apoc.*"
} > "$NEO4J_HOME/conf/neo4j.local.conf"
grep -qxF 'include neo4j.local.conf' "$NEO4J_HOME/conf/neo4j.conf" 2>/dev/null \
  || echo 'include neo4j.local.conf' >> "$NEO4J_HOME/conf/neo4j.conf"

if [ ! -d "$DATA/neo4j/data/dbms" ]; then
  echo "[neo4j] setting initial password"
  neo4j-admin dbms set-initial-password "$NEO4J_PASSWORD" || true
fi
echo "[neo4j] starting"
neo4j start
# Wait for Bolt (7687) to accept connections.
for i in $(seq 1 60); do
  (echo > /dev/tcp/localhost/7687) >/dev/null 2>&1 && break
  sleep 2
done

# ── data + seeds (all idempotent) ────────────────────────────────────────────
cd "$APP_DIR"
echo "[seed] downloading runtime data slice (first boot only)"
python scripts/bootstrap_hf_data.py
echo "[seed] importing KG into Neo4j (first boot only)"
python scripts/seed_neo4j.py || echo "[seed] neo4j seed non-fatal error; continuing"
echo "[seed] warming pgvector (first boot only)"
python scripts/seed_pgvector.py || true

# ── app ──────────────────────────────────────────────────────────────────────
echo "[app] launching Streamlit on :${PORT:-7860}"
exec streamlit run app.py \
  --server.port="${PORT:-7860}" \
  --server.address=0.0.0.0 \
  --server.headless=true \
  --browser.gatherUsageStats=false

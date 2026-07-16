#!/bin/sh
# Entrypoint for the Hugging Face Docker Space.
#
# Boots Postgres (pgvector) + Neo4j inside this single container on the Space's
# local disk (/data), seeds them once (idempotent), then launches Streamlit.
# The seed steps short-circuit on markers / non-empty stores, so a re-run inside
# the same container is fast. NOTE: the disk is ephemeral -- a container restart
# re-seeds, which is why the Space runs on always-on hardware.
#
# POSIX sh ONLY (no bashisms): HF executes this with dash, where `set -o pipefail`
# and /dev/tcp do not exist. Keep it portable.
set -eu

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
# psql below connects over TCP; with scram-sha-256 auth it would otherwise block
# on an interactive password prompt.
export PGPASSWORD="$POSTGRES_PASSWORD"

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
PGLOG="$DATA/pg-server.log"   # NOTE: outside PGDATA on purpose
if [ ! -s "$PGDATA/PG_VERSION" ]; then
  echo "[pg] initializing cluster at $PGDATA"
  pwfile="$(mktemp)"; printf '%s' "$POSTGRES_PASSWORD" > "$pwfile"
  initdb -U "$POSTGRES_USER" --auth=scram-sha-256 --pwfile="$pwfile" >/dev/null
  rm -f "$pwfile"
  echo "listen_addresses='localhost'" >> "$PGDATA/postgresql.conf"
  echo "port=$POSTGRES_PORT"          >> "$PGDATA/postgresql.conf"
  # The compiled-in socket dir (/var/run/postgresql) is owned by the 'postgres'
  # system user and is not writable by the uid-1000 user this container runs as,
  # so the postmaster fails to create its lock file and start. We talk to PG over
  # TCP on localhost anyway, so just put the socket somewhere writable.
  echo "unix_socket_directories='/tmp'" >> "$PGDATA/postgresql.conf"
fi
echo "[pg] starting"
# Surface the server log on failure -- pg_ctl only says "could not start server",
# and the container dies before anyone can read the log.
pg_ctl -D "$PGDATA" -l "$PGLOG" -w -t 60 start || {
  echo "[pg] FAILED to start. ---- server log ----"
  cat "$PGLOG" 2>/dev/null || echo "(no log at $PGLOG)"
  echo "[pg] ---- end server log ----"
  exit 1
}
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
# Data/import/logs dirs; import root matches insert_data's expectation
# (data/neo4j_kg -> file:///all/...).
#
# NOTE: neo4j.conf has NO `include` directive, and Neo4j 5 runs strict config
# validation that hard-fails on any unknown setting -- so these must be appended
# directly. conf/ lives in the image, so it is pristine on every container start;
# the marker guard just keeps this idempotent if the script is ever re-run.
NEO4J_CONF="$NEO4J_HOME/conf/neo4j.conf"
if ! grep -q '^# FOODRAG-SETTINGS' "$NEO4J_CONF" 2>/dev/null; then
  # The stock conf already declares some of these (e.g. server.directories.import),
  # and Neo4j 5 hard-fails on a setting "declared multiple times" -- so strip any
  # existing declaration of each key before appending ours.
  sed -i -E '/^[[:space:]]*(server\.directories\.data|server\.directories\.logs|server\.directories\.import|server\.default_listen_address|dbms\.security\.allow_csv_import_from_file_urls|dbms\.security\.procedures\.unrestricted)[[:space:]]*=/d' "$NEO4J_CONF"
  {
    echo ""
    echo "# FOODRAG-SETTINGS"
    echo "server.directories.data=$DATA/neo4j/data"
    echo "server.directories.logs=$DATA/neo4j/logs"
    echo "server.directories.import=$DATA/neo4j_kg"
    echo "server.default_listen_address=0.0.0.0"
    echo "dbms.security.allow_csv_import_from_file_urls=true"
    echo "dbms.security.procedures.unrestricted=apoc.*"
  } >> "$NEO4J_CONF"
fi
# Fail loudly here rather than deep in the import: validate before starting.
neo4j-admin server validate-config 2>&1 | tail -5 || true

if [ ! -d "$DATA/neo4j/data/dbms" ]; then
  echo "[neo4j] setting initial password"
  neo4j-admin dbms set-initial-password "$NEO4J_PASSWORD" || true
fi
echo "[neo4j] starting"
# Non-fatal: backend._build_kg_retriever() is best-effort and fusion degrades to
# dense+sparse if Neo4j is unreachable, so a Neo4j problem must not take the whole
# Space down -- the other retrievers are still worth serving.
neo4j start || {
  echo "[neo4j] FAILED to start; KG/fusion will be degraded. ---- neo4j log ----"
  tail -40 "$DATA/neo4j/logs/neo4j.log" 2>/dev/null || echo "(no neo4j.log)"
  tail -40 "$DATA/neo4j/logs/debug.log" 2>/dev/null || true
  echo "[neo4j] ---- end log ----"
}
# Wait for Bolt (7687). Uses python (always present) rather than bash's /dev/tcp,
# which does not exist in dash.
python - <<'PY'
import socket, sys, time
deadline = time.time() + 180
while time.time() < deadline:
    try:
        with socket.create_connection(("localhost", 7687), timeout=2):
            print("[neo4j] bolt is up")
            sys.exit(0)
    except OSError:
        time.sleep(2)
print("[neo4j] WARNING: bolt not reachable after 180s; continuing anyway")
PY

# ── data + seeds (all idempotent) ────────────────────────────────────────────
cd "$APP_DIR"
# `python scripts/foo.py` puts scripts/ on sys.path, NOT the cwd -- so
# seed_pgvector's `from ui.services import backend` fails with "No module named
# 'ui'" without this. (seed_neo4j only needs the installed foodrag package, and
# `streamlit run app.py` adds /app itself, which is why only this one broke.)
export PYTHONPATH="$APP_DIR${PYTHONPATH:+:$PYTHONPATH}"
echo "[seed] downloading runtime data slice (first boot only)"
python scripts/bootstrap_hf_data.py
echo "[seed] importing KG into Neo4j (first boot only)"
python scripts/seed_neo4j.py || echo "[seed] neo4j seed non-fatal error; continuing"
# pgvector ingest (137 embedding batches + the 1 GB hummus load) takes tens of
# minutes. Run it in the BACKGROUND so Streamlit binds its port promptly rather
# than leaving the Space in APP_STARTING long enough to look hung: the app is
# usable immediately (Text->Pandas needs no vectors) and the vector retrievers
# light up when this finishes. Safe to overlap: _ensure_recipe_index() no-ops
# once the table is non-empty.
echo "[seed] warming pgvector in background (first boot only)"
{
  python scripts/seed_pgvector.py 2>&1 | sed 's/^/[seed-pgvector] /'
  echo "[seed-pgvector] background warm-up finished"
} &

# ── app ──────────────────────────────────────────────────────────────────────
echo "[app] launching Streamlit on :${PORT:-7860}"
exec streamlit run app.py \
  --server.port="${PORT:-7860}" \
  --server.address=0.0.0.0 \
  --server.headless=true \
  --browser.gatherUsageStats=false

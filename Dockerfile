# Hugging Face Docker Space for FoodRAG.
#
# Single container running Postgres+pgvector, Neo4j and Streamlit, all backed by
# the Space's persistent volume at /data. See docs/DEPLOY_HF.md for the full
# runbook. Requires: HF Pro + a persistent-storage add-on (>=20 GB) on the Space.
FROM python:3.11-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    APP_DIR=/app \
    DATA_DIR=/data \
    NEO4J_HOME=/opt/neo4j \
    HF_HOME=/data/cache/hf \
    PYTHONUNBUFFERED=1

# ── system deps: build toolchain, Java (Neo4j), Postgres 16 + pgvector ───────
# Postgres 16 + pgvector come from the PGDG apt repo (bookworm ships pg15).
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl wget gnupg lsb-release \
        build-essential cmake git \
        openjdk-17-jre-headless \
    && install -d /usr/share/postgresql-common/pgdg \
    && curl -fsSL https://www.postgresql.org/media/keys/ACCC4CF8.asc \
        -o /usr/share/postgresql-common/pgdg/apt.postgresql.org.asc \
    && echo "deb [signed-by=/usr/share/postgresql-common/pgdg/apt.postgresql.org.asc] \
        http://apt.postgresql.org/pub/repos/apt bookworm-pgdg main" \
        > /etc/apt/sources.list.d/pgdg.list \
    && apt-get update && apt-get install -y --no-install-recommends \
        postgresql-16 postgresql-16-pgvector postgresql-client-16 \
    && rm -rf /var/lib/apt/lists/*

# ── Neo4j 5.26 community (tarball → /opt/neo4j, owned by the runtime user) ────
RUN wget -qO /tmp/neo4j.tar.gz \
        "https://dist.neo4j.org/neo4j-community-5.26.0-unix.tar.gz" \
    && tar -xzf /tmp/neo4j.tar.gz -C /opt \
    && mv /opt/neo4j-community-5.26.0 "$NEO4J_HOME" \
    && rm /tmp/neo4j.tar.gz

# APOC (used by the KG import) — drop the bundled plugin into Neo4j's plugin dir.
RUN cp "$NEO4J_HOME"/labs/apoc-*-core.jar "$NEO4J_HOME/plugins/" 2>/dev/null || true

# ── python deps ──────────────────────────────────────────────────────────────
WORKDIR ${APP_DIR}
COPY pyproject.toml requirements.txt README.md ./
COPY src ./src
# Install the project + the extra runtime clients the Space needs.
RUN pip install --upgrade pip \
    && pip install -e . \
    && pip install "huggingface_hub>=0.24" "psycopg2-binary>=2.9" "neo4j>=5.14" streamlit

# ── app code + entrypoint ────────────────────────────────────────────────────
COPY . ${APP_DIR}
RUN chmod +x deploy/start.sh

# HF mounts the persistent volume at /data and runs the container as uid 1000,
# which must own /data and the Neo4j install. Create that user and hand over.
RUN useradd -m -u 1000 user \
    && mkdir -p ${DATA_DIR} \
    && chown -R user:user ${APP_DIR} ${DATA_DIR} ${NEO4J_HOME}
USER user

ENV PATH="/usr/lib/postgresql/16/bin:${NEO4J_HOME}/bin:${PATH}"

# HF routes traffic to $PORT (7860 by default). Declare app_port: 7860 in README.
EXPOSE 7860
CMD ["bash", "deploy/start.sh"]

"""Store connection settings, read from the environment / .env.

Replaces the hardcoded credential dicts in the notebooks and helpers. Defaults
match docker-compose.yml so a fresh `docker compose up` works out of the box.
"""
from __future__ import annotations

import os
from typing import Any

from foodrag.config.settings import load_env


def _env(key: str, default: str) -> str:
    return os.environ.get(key, default)


def postgres_credentials() -> dict[str, str]:
    """Credentials dict for PostgresHelper / PGVectorStore."""
    load_env()
    return {
        "host": _env("POSTGRES_HOST", "localhost"),
        "port": _env("POSTGRES_PORT", "5432"),
        "user": _env("POSTGRES_USER", "postgres"),
        "password": _env("POSTGRES_PASSWORD", "postgres"),
        "db_name": _env("POSTGRES_DB", "rag_postgres"),
    }


def neo4j_config(database: str | None = None) -> dict[str, str]:
    load_env()
    return {
        "uri": _env("NEO4J_URI", "bolt://localhost:7687"),
        "user": _env("NEO4J_USER", "neo4j"),
        "password": _env("NEO4J_PASSWORD", "neo4jfoodrag"),
        # Neo4j Community supports a single database named 'neo4j'.
        "database": database or _env("NEO4J_DATABASE", "neo4j"),
    }


def nebula_config() -> dict[str, Any]:
    load_env()
    host = _env("NEBULA_HOST", "127.0.0.1")
    port = int(_env("NEBULA_PORT", "9669"))
    user = _env("NEBULA_USER", "root")
    password = _env("NEBULA_PASSWORD", "nebula")
    # The llama-index Nebula store reads these env vars.
    os.environ.setdefault("NEBULA_USER", user)
    os.environ.setdefault("NEBULA_PASSWORD", password)
    os.environ.setdefault("NEBULA_ADDRESS", f"{host}:{port}")
    return {"host": host, "port": port, "user": user, "password": password}

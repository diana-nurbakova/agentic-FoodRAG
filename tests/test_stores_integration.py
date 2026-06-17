"""Integration tests for the backing stores (require `docker compose up`).

Run with:  uv run pytest -m integration
Skipped automatically when the stores are not reachable.
"""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.integration


def _skip_unless(cond, msg):
    if not cond:
        pytest.skip(msg)


def test_pgvector_roundtrip():
    from foodrag.stores import get_vector_store_provider
    from llama_index.core.schema import TextNode
    from llama_index.core.vector_stores.types import VectorStoreQuery

    try:
        provider = get_vector_store_provider("pgvector", reset=True)
        vs = provider.connect(table_name="pytest_smoke", embed_dim=4)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Postgres/pgvector not reachable: {exc}")

    vs.add([
        TextNode(text="apple pie", embedding=[1.0, 0.0, 0.0, 0.0], id_="n1"),
        TextNode(text="beef stew", embedding=[0.0, 1.0, 0.0, 0.0], id_="n2"),
    ])
    res = vs.query(VectorStoreQuery(query_embedding=[0.9, 0.1, 0.0, 0.0], similarity_top_k=1))
    assert res.ids and res.ids[0] == "n1"


def test_neo4j_connectivity():
    from neo4j import GraphDatabase

    from foodrag.config.stores import neo4j_config

    cfg = neo4j_config()
    try:
        drv = GraphDatabase.driver(cfg["uri"], auth=(cfg["user"], cfg["password"]))
        with drv.session() as s:
            value = s.run("RETURN 1 AS x").single()["x"]
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Neo4j not reachable: {exc}")
    finally:
        try:
            drv.close()
        except Exception:
            pass
    assert value == 1

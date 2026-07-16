"""One-shot Neo4j KG seeding for the in-container deploy.

Loads the transform_kg CSVs (``data/neo4j_kg/all/*``) into the local Neo4j via
``Neo4jManager.insert_data`` — the same LOAD CSV path used locally. Idempotent:
if the graph already has Recipe nodes it no-ops, so it's safe to run on every
boot. The heavy import runs only on the first boot of a fresh persistent volume.

The CSVs are read by Neo4j through ``file:///all/...`` relative to its configured
import directory, which the Dockerfile points at ``data/neo4j_kg`` (mirroring the
docker-compose mount ``./data/neo4j_kg:/var/lib/neo4j/import``).

Env: NEO4J_URI / NEO4J_USER / NEO4J_PASSWORD / NEO4J_DATABASE, FOODRAG_DATA_DIR.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


def _data_path() -> str:
    override = os.environ.get("FOODRAG_DATA_DIR")
    if override:
        return override
    from foodrag.config.settings import repo_root

    return str(Path(repo_root()) / "data")


def _recipe_count(mgr) -> int:
    try:
        rows = mgr.query("MATCH (r:Recipe) RETURN count(r) AS n")
        return int((rows or [{}])[0].get("n", 0))
    except Exception:  # noqa: BLE001 — treat any error as "not seeded yet"
        return 0


def main() -> int:
    from foodrag.config.stores import neo4j_config
    from foodrag.food_rag.utils.neo4J_helper import Neo4jManager
    from foodrag.hummus_transformation import transform_kg

    cfg = neo4j_config()
    mgr = Neo4jManager(cfg["uri"], cfg["user"], cfg["password"], cfg["database"])
    try:
        existing = _recipe_count(mgr)
        if existing > 0:
            print(f"[seed-neo4j] graph already populated ({existing} Recipe nodes); skipping.")
            return 0

        data_path = _data_path()
        kg_dir = Path(data_path) / transform_kg.SUB_DIRECTORY_NAME / transform_kg.MODE_ALL
        if not (kg_dir / transform_kg.CONFIG_FILE_NAME).exists():
            print(f"[seed-neo4j] KG import files not found under {kg_dir}; nothing to seed.")
            return 0

        print(f"[seed-neo4j] importing KG from {kg_dir} (this runs once) ...")
        # use_absolute_paths=False: paths are emitted relative to the import root
        # (data/neo4j_kg), which is exactly what the Dockerfile sets as Neo4j's
        # server.directories.import.
        mgr.insert_data(data_path=data_path, mode=transform_kg.MODE_ALL,
                        batch_size=1000, verbose=True, use_absolute_paths=False)
        print(f"[seed-neo4j] done - {_recipe_count(mgr)} Recipe nodes.")
        return 0
    finally:
        mgr.close()


if __name__ == "__main__":
    sys.exit(main())

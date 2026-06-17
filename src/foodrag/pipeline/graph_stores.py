"""Stage 4: build the Neo4j knowledge graph from HUMMUS data.

Transforms the HUMMUS data into Neo4j-importable CSVs, persists them, and bulk
loads them into Neo4j (creds from .env). The Nebula path is legacy/optional and
not built here.

    foodrag-init-graph-stores [--data-path data] [--mode all] [--limit N] [--no-insert]
"""
from __future__ import annotations

import argparse
from typing import Any, Optional


def run(
    data_path: str = "data",
    *,
    mode: Optional[str] = None,
    limit_size: Optional[int] = None,
    insert: bool = True,
    clear_first: bool = True,
    hummus_data: Any = None,
    manager: Any = None,
    config: Any = None,
) -> dict:
    """Transform + (optionally) load the KG into Neo4j.

    ``hummus_data`` and ``manager`` (a Neo4jManager) are injectable for testing.
    NOTE: ``mode`` selects the transform content/file naming; on Neo4j Community
    the data still lands in the single 'neo4j' database (from .env).
    """
    from foodrag.config.stores import neo4j_config
    from foodrag.hummus_transformation import transform_kg
    from foodrag.utils import load_util

    mode = mode or transform_kg.MODE_ALL
    if hummus_data is None:
        hummus_data = load_util.get_current_hummus(data_path)

    transformer = transform_kg.Neo4JTransformer(mode=mode)
    transformer.transform(hummus_data, limit_size=limit_size)
    transformer.save(data_path=data_path)

    inserted = False
    if insert:
        if manager is None:
            from foodrag.food_rag.utils.neo4J_helper import Neo4jManager

            cfg = neo4j_config()
            manager = Neo4jManager(cfg["uri"], cfg["user"], cfg["password"], cfg["database"])
        if clear_first:
            manager.delete_all()
        manager.insert_data(data_path=data_path, mode=mode, batch_size=1000, verbose=True)
        inserted = True

    return {"mode": mode, "inserted": inserted}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build the Neo4j KG from HUMMUS data.")
    p.add_argument("--data-path", default="data")
    p.add_argument("--mode", default="all", choices=["all", "nousda", "unlinkedhummus", "onlyfoodon"])
    p.add_argument("--limit", type=int, help="limit transform size (for testing)")
    p.add_argument("--no-insert", action="store_true", help="transform + save only, skip Neo4j load")
    p.add_argument("--no-clear", action="store_true", help="don't delete existing nodes first")
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    out = run(
        data_path=args.data_path, mode=args.mode, limit_size=args.limit,
        insert=not args.no_insert, clear_first=not args.no_clear,
    )
    print(f"KG mode={out['mode']} transformed{' + inserted into Neo4j' if out['inserted'] else ' (not inserted)'}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

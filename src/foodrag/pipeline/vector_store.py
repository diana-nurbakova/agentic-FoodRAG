"""Stage 3: load the recipe embeddings into the pgvector store.

Loads the precomputed recipe embedding batches and builds the vector index in
PostgreSQL+pgvector (creds from .env). The embedding dim comes from the active
embedding model in config/models.yml.

    foodrag-init-vector-store [--data-path data] [--table hummus_recipes] [--no-reset]
"""
from __future__ import annotations

import argparse
from typing import Any, Optional


def run(
    data_path: str = "data",
    *,
    table_name: str = "hummus_recipes",
    reset: bool = True,
    embed_dim: Optional[int] = None,
    hummus_data: Any = None,
    provider: Any = None,
    config: Any = None,
) -> dict:
    """Build the pgvector index from the recipe embedding batches.

    ``hummus_data`` and ``provider`` are injectable for testing.
    """
    from foodrag.config.settings import embedding_dim, load_models_config
    from foodrag.food_rag.utils import embedding_helper, hummus_docify
    from foodrag.pipeline.embed import embedding_path_for
    from foodrag.stores import get_vector_store_provider
    from foodrag.utils import load_util

    cfg = config or load_models_config()
    embed_dim = embed_dim or embedding_dim(config=cfg)
    if hummus_data is None:
        hummus_data = load_util.get_current_hummus(data_path)
    provider = provider or get_vector_store_provider("pgvector", reset=reset)

    recipe_documents = hummus_docify.recipes_to_docs(hummus_data["recipes_df"])
    epath = embedding_path_for(data_path, cfg)
    doc_nodes_dict, _ = embedding_helper.load_existing_batches(epath + "/recipe_documents")
    embedding_helper.check_batches(recipe_documents, doc_nodes_dict)
    nodes = [node for node_list in doc_nodes_dict.values() for node in node_list]

    from llama_index.core import VectorStoreIndex

    vector_store = provider.connect(table_name=table_name, embed_dim=embed_dim)
    index = VectorStoreIndex.from_vector_store(vector_store)
    index.build_index_from_nodes(nodes=nodes)
    return {"table": table_name, "embed_dim": embed_dim, "n_nodes": len(nodes)}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Load recipe embeddings into pgvector.")
    p.add_argument("--data-path", default="data")
    p.add_argument("--table", default="hummus_recipes")
    p.add_argument("--no-reset", action="store_true", help="don't drop/recreate the database")
    p.add_argument("--embed-dim", type=int, help="override embedding dim (defaults to active model)")
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    out = run(
        data_path=args.data_path, table_name=args.table,
        reset=not args.no_reset, embed_dim=args.embed_dim,
    )
    print(f"pgvector table {out['table']} (dim {out['embed_dim']}): {out['n_nodes']} nodes loaded")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

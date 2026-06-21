"""Stage 2: compute embeddings for the HUMMUS documents.

Embeds each document type (recipe/user/review/ingredient/food_product/food_on)
with the active embedding model into batches under
``<data_path>/embeddings/<model_id>/<type>_documents``. The output dir is keyed
by the model id so switching embedding models doesn't clobber prior runs.

    foodrag-embed [--data-path data] [--doc-types recipe,user] [--sample N]
"""
from __future__ import annotations

import argparse
from typing import Any, Optional

# name -> (hummus_docify function, hummus_data key, minimize_memory)
DOC_TYPES: dict[str, tuple[str, str, bool]] = {
    "recipe": ("recipes_to_docs", "recipes_df", False),
    "user": ("users_to_docs", "users_df", False),
    "review": ("reviews_to_docs", "reviews_df", True),
    "ingredient": ("ingredients_to_docs", "ingredients_df", False),
    "food_product": ("food_products_to_docs", "usda_df", True),
    "food_on": ("food_on_to_docs", "food_on_ontology", False),
}


def embedding_path_for(data_path: str, config: Any = None) -> str:
    from foodrag.config.settings import load_models_config

    cfg = config or load_models_config()
    return f"{data_path}/embeddings/" + cfg.embedding["model"].replace("/", "_")


def run(
    data_path: str = "data",
    *,
    doc_types: Optional[list[str]] = None,
    sample: Optional[int] = None,
    batch_size: int = 1000,
    chunk_size: int = 300,
    embed_model: Any = None,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    hummus_data: Any = None,
    config: Any = None,
) -> dict:
    """Embed the selected document types; return path + per-type counts.

    ``embed_model`` and ``hummus_data`` are injectable for testing. If
    ``embed_model`` is None, ``provider``/``model`` override the embedding
    backend (e.g. provider='huggingface' for a local GPU bge-m3 on Colab). Keep
    the SAME model id (BAAI/bge-m3) so the output dir and vectors stay
    consistent with the API embeddings.
    """
    from foodrag.config.settings import build_embedding, load_models_config
    from foodrag.food_rag.utils import embedding_helper, hummus_docify
    from foodrag.utils import load_util

    cfg = config or load_models_config()
    if embed_model is None:
        if provider:
            spec = {"provider": provider, "model": model or cfg.embedding["model"],
                    "dim": cfg.embedding.get("dim")}
            embed_model = build_embedding(spec=spec, config=cfg)
        else:
            embed_model = build_embedding(config=cfg)
    if hummus_data is None:
        hummus_data = load_util.get_current_hummus(data_path)

    epath = embedding_path_for(data_path, cfg)
    names = doc_types or list(DOC_TYPES)
    counts: dict[str, int] = {}
    for name in names:
        if name not in DOC_TYPES:
            raise KeyError(f"Unknown doc type {name!r}. Known: {list(DOC_TYPES)}")
        fn_name, key, minimize_memory = DOC_TYPES[name]
        df = hummus_data.get(key)
        if df is None:
            continue
        docs = getattr(hummus_docify, fn_name)(df)
        if sample:
            docs = docs[:sample]
        embedding_helper.compute_embeddings(
            docs, embed_model, batch_size=batch_size, chunk_size=chunk_size,
            path=f"{epath}/{name}_documents", minimize_memory=minimize_memory,
        )
        counts[name] = len(docs)
    return {"embedding_path": epath, "counts": counts}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Compute HUMMUS embeddings.")
    p.add_argument("--data-path", default="data")
    p.add_argument("--doc-types", help=f"comma-separated subset of {list(DOC_TYPES)}")
    p.add_argument("--sample", type=int, help="embed only the first N docs per type (smoke)")
    p.add_argument("--batch-size", type=int, default=1000)
    p.add_argument("--chunk-size", type=int, default=300)
    p.add_argument("--embed-provider", help="override embedding provider (e.g. huggingface for local GPU)")
    p.add_argument("--embed-model", help="override embedding model id (default keeps the active model)")
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    out = run(
        data_path=args.data_path,
        doc_types=args.doc_types.split(",") if args.doc_types else None,
        sample=args.sample,
        batch_size=args.batch_size,
        chunk_size=args.chunk_size,
        provider=args.embed_provider,
        model=args.embed_model,
    )
    print(f"Embeddings -> {out['embedding_path']}")
    for name, n in out["counts"].items():
        print(f"  {name}: {n} docs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

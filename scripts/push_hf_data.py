"""Push the FoodRAG runtime data slice to a (private) HF Dataset repo.

Run this ONCE from your machine to upload the ~5 GB the deployed Space needs:
the HUMMUS pickles (minus the unused reviews_df), the precomputed recipe
embeddings, and the Neo4j KG import CSVs. The Space then pulls this on first boot
(scripts/bootstrap_hf_data.py) and seeds Postgres + Neo4j from it.

The dataset mirrors the repo's data/ layout so paths resolve unchanged on the
Space:
    hummus/        (top-level *.pkl / *.csv, excluding reviews_df.csv)
    embeddings/    (BAAI_bge-m3/recipe_documents, food_product_documents)
    neo4j_kg/      (all/, nousda/)

Usage:
    huggingface-cli login          # or set HF_TOKEN
    python scripts/push_hf_data.py --repo <user>/foodrag-data [--private] \\
        [--data-dir ./data] [--include-embeddings] [--include-kg]

Notes:
  * reviews_df.csv (~381 MB) is excluded — it's loaded into hummus_data but never
    read on the inference path (get_recipe_interactions has no callers, personas
    are preset). Dropping it trims the hummus slice to ~1 GB.
  * embeddings/ (3.3 GB) and neo4j_kg/ (0.9 GB) are only needed until the Space's
    Postgres/Neo4j are seeded. If you seed those DBs another way (e.g. Dev Mode),
    omit --include-embeddings / --include-kg to keep the dataset ~1 GB.
"""
from __future__ import annotations

import argparse
from pathlib import Path


def build_patterns(include_embeddings: bool, include_kg: bool) -> tuple[list[str], list[str]]:
    # NOTE: huggingface_hub filters with fnmatch, where '*' also matches '/'.
    # So "hummus/*.pkl" alone would ALSO pull hummus/food_kg/**, hummus_graph/**
    # etc. (~34 GB of pipeline-only data). The ignore list below is what actually
    # restricts us to the top-level hummus files that _load_hummus reads.
    allow = [
        "hummus/*.pkl",
        "hummus/*.csv",
        "hummus/*.json",
        "hummus/*.ttl",
    ]
    ignore = [
        "hummus/reviews_df.csv",   # loaded into hummus_data but unused at inference
        # Derived cold-start cache (backend._engine_cache_path). Keyed on the
        # (size, mtime) of recipes_df.csv, and snapshot_download rewrites mtime —
        # so a shipped cache is always invalidated on arrival. The Space rebuilds
        # it once onto the persistent volume instead. Don't waste 1.3 GB on it.
        "hummus/_ui_engine_cache.pkl",
        # Pipeline/build inputs — never read at serving time (see docs/DEPLOY_HF.md).
        "hummus/food_kg/*",
        "hummus/hummus_data/*",
        "hummus/hummus_graph/*",
        "hummus/crawler_init/*",
        "hummus/annotations/*",
    ]
    if include_embeddings:
        # Only recipe_documents is used — backend._recipe_embeddings_dir() globs
        # data/embeddings/*/recipe_documents for the pgvector ingest.
        allow += ["embeddings/BAAI_bge-m3/recipe_documents/*"]
    if include_kg:
        allow += ["neo4j_kg/all/*", "neo4j_kg/nousda/*"]
    return allow, ignore


def _resolve_token() -> str | None:
    """Token from .env (HF_WRITE_TOKEN / HF_TOKEN), else the huggingface-cli login cache."""
    try:
        from dotenv import load_dotenv

        load_dotenv(override=False)
    except Exception:  # noqa: BLE001 — dotenv is optional
        pass
    import os

    return os.environ.get("HF_WRITE_TOKEN") or os.environ.get("HF_TOKEN") or None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", required=True, help="target dataset id, e.g. user/foodrag-data")
    ap.add_argument("--data-dir", default="data", help="local data/ dir (default: ./data)")
    ap.add_argument("--private", action="store_true", help="create the dataset as private")
    ap.add_argument("--include-embeddings", action="store_true",
                    help="also upload data/embeddings (3.3 GB) for first-boot pgvector seed")
    ap.add_argument("--include-kg", action="store_true",
                    help="also upload data/neo4j_kg (0.9 GB) for first-boot Neo4j seed")
    args = ap.parse_args()

    data_dir = Path(args.data_dir).resolve()
    if not (data_dir / "hummus").exists():
        raise SystemExit(f"{data_dir}/hummus not found — point --data-dir at the repo data/ dir.")

    from huggingface_hub import HfApi, create_repo

    token = _resolve_token()
    create_repo(args.repo, repo_type="dataset", private=args.private,
                exist_ok=True, token=token)
    allow, ignore = build_patterns(args.include_embeddings, args.include_kg)

    print(f"Uploading {data_dir} -> dataset {args.repo}")
    print(f"  allow : {allow}")
    print(f"  ignore: {ignore}")
    HfApi(token=token).upload_large_folder(
        repo_id=args.repo,
        repo_type="dataset",
        folder_path=str(data_dir),
        allow_patterns=allow,
        ignore_patterns=ignore,
    )
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

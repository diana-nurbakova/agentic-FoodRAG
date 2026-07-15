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
    allow = [
        # hummus top-level files only (subdirs like food_kg/, hummus_data/ are
        # pipeline inputs and are never read at serving time).
        "hummus/*.pkl",
        "hummus/*.csv",
        "hummus/*.json",
        "hummus/*.ttl",
    ]
    ignore = ["hummus/reviews_df.csv"]  # loaded but unused at inference → skip
    if include_embeddings:
        allow += ["embeddings/BAAI_bge-m3/recipe_documents/**",
                  "embeddings/BAAI_bge-m3/food_product_documents/**"]
    if include_kg:
        allow += ["neo4j_kg/all/**", "neo4j_kg/nousda/**"]
    return allow, ignore


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

    create_repo(args.repo, repo_type="dataset", private=args.private, exist_ok=True)
    allow, ignore = build_patterns(args.include_embeddings, args.include_kg)

    print(f"Uploading {data_dir} → dataset {args.repo}")
    print(f"  allow : {allow}")
    print(f"  ignore: {ignore}")
    HfApi().upload_large_folder(
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

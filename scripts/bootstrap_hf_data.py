"""First-boot data bootstrap for the Hugging Face Space deploy.

On a HF Docker Space the code repo is small and the ~1 GB HUMMUS runtime slice
lives in a **private HF Dataset repo**. This script pulls that slice onto the
persistent volume (``/data`` via the ``data`` symlink) exactly once, guarded by a
marker file, so restarts/rebuilds don't re-download it.

It intentionally does NOT seed the databases — pgvector is seeded lazily on the
first ``get_engine()`` call (``backend._ensure_recipe_index``), and Neo4j is
seeded by ``scripts/seed_neo4j.py``. This script only makes the on-disk files
present. Idempotent and safe to run on every boot.

Env:
  FOODRAG_DATA_REPO   HF Dataset id holding the runtime slice (e.g. "me/foodrag-data")
  FOODRAG_DATA_DIR    target data dir (default: <repo_root>/data)
  HF_TOKEN            token with read access to the (private) dataset
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


def _data_dir() -> Path:
    override = os.environ.get("FOODRAG_DATA_DIR")
    if override:
        return Path(override)
    try:
        from foodrag.config.settings import repo_root

        return Path(repo_root()) / "data"
    except Exception:  # noqa: BLE001 — fall back to CWD/data if foodrag isn't importable yet
        return Path.cwd() / "data"


def main() -> int:
    repo_id = os.environ.get("FOODRAG_DATA_REPO")
    if not repo_id:
        print("[bootstrap] FOODRAG_DATA_REPO not set — skipping data download "
              "(assuming data is already on the persistent volume).")
        return 0

    data_dir = _data_dir()
    marker = data_dir / ".hf_data_seeded"
    if marker.exists():
        print(f"[bootstrap] data already present ({marker}); skipping download.")
        return 0

    data_dir.mkdir(parents=True, exist_ok=True)
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    from huggingface_hub import snapshot_download

    # Pull the dataset snapshot straight into <data_dir>. The dataset's internal
    # layout MUST mirror the repo's data/ layout (hummus/, embeddings/, neo4j_kg/)
    # so paths resolve unchanged. allow_patterns keeps us to the runtime slice.
    print(f"[bootstrap] downloading {repo_id} → {data_dir} …")
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(data_dir),
        token=token,
        allow_patterns=[
            "hummus/**",
            "embeddings/**",   # only needed until pgvector is seeded; drop if you pre-seed the DB
            "neo4j_kg/**",     # only needed until Neo4j is seeded
        ],
    )
    marker.write_text("ok\n", encoding="utf-8")
    print("[bootstrap] data download complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

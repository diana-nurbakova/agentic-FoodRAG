"""One-shot pgvector warm-up / seed for the in-container deploy.

The pgvector table is seeded lazily the first time ``backend.get_engine()`` builds
the vector index (``_ensure_recipe_index`` ingests the precomputed embedding
batches from ``data/embeddings/*/recipe_documents`` when the table is empty).
Running it here at boot moves that one-time cost off the first user request.

Idempotent and best-effort: if the table is already populated it no-ops, and any
failure (e.g. no LLM key yet, embeddings not downloaded) is non-fatal — the lazy
path will retry on first use. Requires an embedding model + the embedding batches
on disk; does NOT require an LLM key (only the vector index is built here).
"""
from __future__ import annotations

import sys


def main() -> int:
    try:
        from ui.services import backend

        engine = backend.get_engine()
        backend._vector_index(engine)  # triggers _ensure_recipe_index if empty
        print("[seed-pgvector] vector index ready.")
        return 0
    except Exception as exc:  # noqa: BLE001 — non-fatal; lazy path retries on first request
        print(f"[seed-pgvector] skipped/failed (non-fatal): {exc}")
        return 0


if __name__ == "__main__":
    sys.exit(main())

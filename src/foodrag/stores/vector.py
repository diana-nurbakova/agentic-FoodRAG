"""Pluggable vector-store interface.

The experiments run on PostgreSQL + pgvector (native dense + sparse/BM25 +
full-text, which the 10-retriever evaluation depends on). This thin abstraction
lets a future GUI swap in an embedded store (Qdrant / LanceDB) without Postgres,
without touching retriever/agent code.

    from foodrag.stores import get_vector_store_provider
    provider = get_vector_store_provider("pgvector")   # creds from .env
    index = provider.index("hummus_recipes", embed_dim=1024)
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

from foodrag.config.stores import postgres_credentials


class VectorStoreProvider(ABC):
    """Backend-agnostic vector store factory."""

    name: str = "base"

    @abstractmethod
    def connect(self, table_name: str, embed_dim: int) -> Any:
        """Return a llama-index BasePydanticVectorStore for ``table_name``."""

    def index(self, table_name: str, embed_dim: int) -> Any:
        """Return a VectorStoreIndex built on top of :meth:`connect`."""
        from llama_index.core import VectorStoreIndex

        return VectorStoreIndex.from_vector_store(self.connect(table_name, embed_dim))


class PgVectorProvider(VectorStoreProvider):
    """PostgreSQL + pgvector (hybrid dense+sparse+full-text). The default."""

    name = "pgvector"

    def __init__(self, credentials: Optional[dict] = None, reset: bool = False):
        self.credentials = credentials or postgres_credentials()
        self.reset = reset

    def connect(self, table_name: str, embed_dim: int) -> Any:
        from foodrag.food_rag.utils.postgres_helper import PostgresHelper

        return PostgresHelper(self.credentials, reset=self.reset).connect(
            table_name=table_name, embed_dim=embed_dim
        )


# Registry of available providers. Embedded backends (qdrant/lancedb) can be
# registered here later for the GUI without changing call sites.
_PROVIDERS: dict[str, type[VectorStoreProvider]] = {
    "pgvector": PgVectorProvider,
}


def register_provider(name: str, cls: type[VectorStoreProvider]) -> None:
    _PROVIDERS[name] = cls


def available_providers() -> list[str]:
    return sorted(_PROVIDERS)


def get_vector_store_provider(name: str = "pgvector", **kwargs) -> VectorStoreProvider:
    if name not in _PROVIDERS:
        raise KeyError(f"Unknown vector store {name!r}. Available: {available_providers()}")
    return _PROVIDERS[name](**kwargs)
